"""The study pages for a newcomer: the key message, the folded tables, the
glossary, and questions answered from one page under the number check. No Qt.
"""

from __future__ import annotations

import pytest

from onset_agent import knowledge, pageask
from onset_agent.backends import AssistantMessage
from onset_review import studies


class _Model:
    """A language model that says what it is told to say."""

    is_language_model = True
    name = "fake:model"

    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def chat(self, messages, tools):
        self.messages = messages
        return AssistantMessage(content=self.reply)

    def abort(self):
        pass


class _NoModel:
    is_language_model = False
    name = "scripted"


@pytest.mark.parametrize("key", [k for k, _, _ in studies.STUDIES])
def test_every_study_page_has_a_key_message_from_its_tables(key):
    message = studies.key_message(key, subject="sub-01")
    assert len(message["lead"].split(". ")) >= 2, "two or three sentences"
    assert "could not" not in message["lead"]
    assert 1 <= len(message["tiles"]) <= 4
    for value, caption in message["tiles"]:
        assert value and caption
    # Every number in the lead is on the page or in its own tiles, allowing
    # for a share written as a percentage ("43%" of 0.429).
    page = studies.build(key, subject="sub-01")
    stated = knowledge.numbers_in(page) | knowledge.numbers_in(
        " ".join(str(value) for value, _ in message["tiles"]))

    def traced(number: float) -> bool:
        return any(abs(number - s) <= max(0.05, 0.01 * abs(s)) for s in stated)

    for number in knowledge.numbers_in(message["lead"]):
        assert traced(number) or traced(number / 100), f"{key}: {number} is not on the page"


def test_the_key_numbers_are_the_tables_numbers():
    from app import panels

    best = panels.operating_points("rank_rho")
    rho = float(best[best["band"] == "ripple"]["rank_rho"].max())
    tiles = dict((c, v) for v, c in studies.key_message("detectors")["tiles"])
    assert tiles["channel-rank ρ, best detector"] == f"{rho:.2f}"
    cohort = panels.cohort_overview()
    free = cohort["seizure_free"].astype(bool)
    hit = cohort["top_resected_expert"] == 1
    tiles = dict((c, v) for v, c in studies.key_message("patients")["tiles"])
    assert tiles["seizure-free: busiest channel resected"] == \
        f"{int((hit & free).sum())}/{int(free.sum())}"


def test_long_tables_fold_and_short_ones_stay():
    page = studies.build("detectors")
    folded, count = studies.fold_tables(page)
    assert count >= 1 and "is folded" in folded
    assert "|---" not in folded.split("is folded")[0].split("## The sweep")[1], \
        "the sweep table is gone from the folded text"
    whole, none = studies.fold_tables(page, max_rows=10_000)
    assert none == 0 and whole == page
    assert studies.fold_tables("# T\n\n| a |\n|---|\n| 1 |\n")[1] == 0, "a short table stays"


def test_numeric_columns_are_right_aligned():
    import pandas as pd

    text = studies.table(pd.DataFrame({"name": ["a", "b"], "value": [1.5, 2.25],
                                       "note": ["x", "—"]}))
    assert "|---|---:|---|" in text


def test_the_glossary_maps_terms_to_definitions():
    glossary = knowledge.glossary()
    assert "HFO" in glossary and "ripple" in glossary["HFO"].lower()
    assert "iEEG" in glossary and "SEEG" in glossary
    assert "Channel" not in glossary and "Contact" not in glossary, "furniture, not jargon"
    for term, definition in glossary.items():
        assert len(term) >= 3 and definition and len(definition) <= 320


def test_a_summary_without_a_model_is_the_key_message():
    page = studies.build("outcome")
    lead = studies.key_message("outcome")["lead"]
    answer = pageask.answer(_NoModel(), "Summarise this page", "Outcome", page, lead)
    assert not answer.refused and answer.verified and answer.text == lead
    assert answer.sources[0][0] == "Outcome, key message"
    other = pageask.answer(_NoModel(), "What does the window stability figure show?",
                           "Outcome", page, lead)
    assert not other.refused and other.text.startswith("From Outcome")
    assert "window" in other.text.lower()


def test_a_model_answer_is_checked_against_the_page():
    page = studies.build("outcome")
    lead = studies.key_message("outcome")["lead"]
    honest = _Model("The expert AUC is 0.71 with p = 0.12 on 13 against 7 patients.")
    answer = pageask.answer(honest, "What is the main result?", "Outcome", page, lead)
    assert not answer.refused and answer.verified
    assert "Outcome, key message" in honest.messages[1]["content"]
    assert "passages" in honest.messages[1]["content"].lower()
    liar = _Model("The AUC was 0.99 and 40 patients were studied.")
    answer = pageask.answer(liar, "What is the main result?", "Outcome", page, lead)
    assert answer.refused and "0.99" in answer.reason and "40" in answer.reason
    quiet = _Model('{"refusal": "The page does not say."}')
    answer = pageask.answer(quiet, "How tall was the surgeon?", "Outcome", page, lead)
    assert not answer.refused and answer.text == "The page does not say."


def test_medical_and_empty_questions_are_refused_before_any_model():
    page = studies.build("outcome")
    model = _Model("anything")
    assert pageask.answer(model, "", "Outcome", page).refused
    assert model.messages is None
    medical = pageask.answer(model, "Should my patient have surgery?", "Outcome", page)
    assert medical.refused and model.messages is None
    about = pageask.answer(model, "How many patients were studied?", "Outcome", page)
    assert not about.refused and model.messages is not None, \
        "a question about the patients on the page is not a request for advice"


def test_summary_questions_are_recognised():
    assert pageask.is_summary_question("Summarise this page in three sentences.")
    assert pageask.is_summary_question("What is the main result?")
    assert pageask.is_summary_question("tl;dr")
    assert not pageask.is_summary_question("What does channel-rank ρ mean?")
