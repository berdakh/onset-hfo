"""Background questions: the documents, the routing, and the two new paths."""

from __future__ import annotations

import json

import pytest

from onset_agent import knowledge
from onset_agent.agent import OnsetAgent
from onset_agent.backends import AssistantMessage, Backend, ScriptedBackend


@pytest.fixture(scope="module")
def sections():
    out = knowledge.load_sections()
    assert out, "the checkout's docs are the knowledge base"
    return out


def test_the_documents_split_into_titled_sections_and_the_glossary_into_entries(sections):
    docs = {s.doc for s in sections}
    assert docs == set(knowledge.DOCUMENTS)
    titles = [s.title for s in sections if s.doc == "GLOSSARY"]
    assert "Ripple" in titles or any(t.startswith("Ripple") for t in titles)
    assert any(t.startswith("HFO") for t in titles)
    assert sum(len(s.text) > knowledge.CHUNK_CHARS * 2 for s in sections) <= 5
    assert all(s.terms for s in sections)


def test_retrieval_finds_the_entry_the_question_is_about(sections):
    hits = knowledge.retrieve("What is a ripple?", sections)
    # The glossary defines the ripple inside its HFO entry, in bold; the
    # emphasis is indexed as an alias of the entry, so that entry comes first.
    assert hits and hits[0].doc == "GLOSSARY" and hits[0].title.startswith("HFO")
    assert "ripple" in hits[0].aliases
    hits = knowledge.retrieve("What is an HFO?", sections)
    assert hits and any(s.doc == "GLOSSARY" and s.title.startswith("HFO") for s in hits)
    hits = knowledge.retrieve("Why is the notch 2 Hz wide?", sections)
    assert hits and hits[0].doc == "METHODS"
    assert sum(len(s.text) for s in hits) <= knowledge.MAX_CHARS + knowledge.CHUNK_CHARS
    # A question the documents do not cover returns nothing, not the least-bad section.
    assert knowledge.retrieve("Who was Hans Berger?", sections) == []
    assert knowledge.retrieve("", sections) == []


def test_questions_are_sorted_with_the_data_path_as_the_default():
    data = ["Which channel had the highest ripple rate?", "What was analysed?",
            "Does AR1-AR2 survive a stricter threshold?", "What are the limitations?",
            "Did the leader change between the two minutes?", "Is AHR6-AHR7 noisy?",
            "How many events were rejected?"]
    for q in data:
        assert knowledge.kind_of(q) == "data", q
    for q in ("What is an HFO?", "Why is the notch 2 Hz wide?", "Is a ripple always pathological?",
              "Explain the bipolar montage"):
        assert knowledge.kind_of(q) == "background", q
    # An imperative with no question in it takes the guarded path: its
    # failure is a refusal, which is the safe one.
    assert knowledge.kind_of("Write a haiku about brains") == "data"
    assert knowledge.kind_of("Delete the results.") == "data"
    assert knowledge.kind_of("Who was Hans Berger?") == "background"
    assert knowledge.looks_medical("Should my patient have surgery?")
    assert not knowledge.looks_medical("What is a ripple?")


def test_plain_strips_emphasis():
    assert knowledge.plain("**Ripple**: 80–250 Hz, *physiological* too") == \
        "Ripple: 80–250 Hz, physiological too"


def test_numbers_in_text():
    assert knowledge.numbers_in("ripples run 80 to 250 Hz, lasting 30.5 ms") == {80.0, 250.0, 30.5}
    assert knowledge.numbers_in("sub-01 at 2000 Hz") == {2000.0}


class _Model(Backend):
    """A model that says what it is told to, recording what it was given."""

    name = "fake-model"
    is_language_model = True

    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def describe(self):
        return "fake"

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        return AssistantMessage(content=self.reply)


def test_a_background_question_is_answered_from_the_documents_and_cited(store):
    agent = OnsetAgent(store, backend=ScriptedBackend())
    answer = agent.ask("What is an HFO?")
    assert not answer.refused and answer.mode == "background" and answer.verified
    assert answer.sources and answer.sources[0][0].startswith("GLOSSARY")
    assert answer.text.startswith("From GLOSSARY")
    assert any(e.get("type") == "routed" and e.get("kind") == "background" for e in answer.trace)

    model = _Model("An HFO is a brief oscillation above 80 Hz [GLOSSARY, HFO (high-frequency "
                   "oscillation)].")
    answer = OnsetAgent(store, backend=model).ask("What is an HFO?")
    assert not answer.refused and answer.mode == "background" and answer.verified
    messages, tools = model.calls[0]
    assert tools == [] and "Passages:" in messages[1]["content"]
    assert "GLOSSARY" in messages[1]["content"] and "from the passages" in messages[0]["content"]


def test_a_background_answer_with_a_number_the_documents_do_not_state_is_refused(store):
    model = _Model("Ripples are oscillations between 80 and 999 Hz.")
    answer = OnsetAgent(store, backend=model).ask("What is a ripple?")
    assert answer.refused and answer.mode == "background"
    assert "999" in answer.reason and "do not" in answer.reason
    model = _Model("Ripples run from 80 to 250 Hz.")
    answer = OnsetAgent(store, backend=model).ask("What is a ripple?")
    assert not answer.refused and answer.verified


def test_a_question_nothing_covers_is_the_models_own_and_says_so(store):
    model = _Model("Hans Berger, in 1924.")
    answer = OnsetAgent(store, backend=model).ask("Who was Hans Berger?")
    assert not answer.refused and answer.mode == "general" and not answer.verified
    assert "not checked" in answer.reason
    messages, _ = model.calls[0]
    assert "own knowledge" in messages[0]["content"]
    # Without a model there is nothing to answer it with, and that is said.
    answer = OnsetAgent(store, backend=ScriptedBackend()).ask("Who was Hans Berger?")
    assert answer.refused and answer.mode == "general" and "no model" in answer.text.lower()


def test_medical_and_out_of_scope_questions_are_still_refused_before_any_model(store):
    model = _Model("Yes, operate.")
    for question in ("Should my patient have surgery?", "Which channel should be resected?",
                     "What is the prognosis for a patient with ripples?"):
        answer = OnsetAgent(store, backend=model).ask(question)
        assert answer.refused, question
    assert model.calls == []


def test_a_data_question_still_takes_the_guarded_path(store):
    answer = OnsetAgent(store, backend=ScriptedBackend()).ask(
        "Which channel had the highest ripple rate?")
    assert answer.mode == "data" and answer.looked_at and not answer.refused
    assert json.dumps(answer.sources) == "[]"


def test_what_i_can_do_mentions_background_questions():
    from onset_agent.prompts import what_i_can_do

    text = what_i_can_do("sub-01")
    assert "what is an HFO" in text and "labelled as unchecked" in text
