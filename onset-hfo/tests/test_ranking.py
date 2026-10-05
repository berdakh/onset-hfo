"""The ranking rule: what a re-test is allowed to do to a channel's score.

`rank_channels` decides every orchestration number this project publishes, and
until now it had no tests. These cover the rule itself rather than the loop
around it, and the headline is the distinction the rule exists to make:

* a channel that went quiet while its neighbours kept firing has been
  **refuted**, and should be demoted;
* a threshold at which the *whole band* went quiet has measured **nothing**,
  and must not be allowed to demote anybody.

Conflating the two is not a rounding error. A re-test that found nothing
anywhere sets every channel's robustness to 0, so every score becomes 0, and a
score-ordered ranking with no variance left in it degenerates to alphabetical
order -- which is how the quietest contact in the recording ends up presented
as the top candidate, with no indication that anything went wrong.
"""

from __future__ import annotations

import pytest

from onset_agent.contract import ToolRun
from onset_agent.evidence import EvidenceStore
from onset_agent.planner import detected_nothing, rank_channels

#: Deliberately anti-correlated with rate. Real electrode names carry no
#: relationship to how active a contact is, and a bug whose symptom is
#: "falls back to alphabetical" is invisible in a fixture that is already
#: alphabetical by rate.
SURVEY = {"Z1-Z2": 12.0, "M1-M2": 6.0, "A1-A2": 3.0}

BY_RATE = ["Z1-Z2", "M1-M2", "A1-A2"]


def _run(run_id: str, threshold: float, rates: dict[str, float],
         n_accepted: int | None = None, ok: bool = True) -> ToolRun:
    output = {
        "threshold_sd": threshold,
        "channels": {channel: {"rate_per_min": rate, "n_events": int(rate)}
                     for channel, rate in rates.items()},
    }
    if n_accepted is not None:
        output["n_accepted"] = n_accepted
    return ToolRun(run_id=run_id, tool="detect_hfo", input={}, output=output, ok=ok)


def _store(*runs: ToolRun) -> EvidenceStore:
    store = EvidenceStore(subject="sub-01")
    for run in runs:
        store.append(run)
    return store


def _survey(n_accepted: int | None = 21) -> ToolRun:
    return _run("survey", 3.0, SURVEY, n_accepted=n_accepted)


def _ranked(store: EvidenceStore) -> list[str]:
    return [channel.channel for channel in rank_channels(store)]


# -- the rule working as intended -------------------------------------------


def test_an_unchallenged_channel_keeps_robustness_one():
    """*Unchallenged*, not *verified* -- the distinction the planner's whole
    fixed-versus-replanning comparison rests on."""
    ranked = rank_channels(_store(_survey()))
    assert [r.channel for r in ranked] == BY_RATE
    assert {r.robustness for r in ranked} == {1.0}
    assert all(r.retested_at == [] for r in ranked)
    assert all(r.score == pytest.approx(r.survey_rate_per_min) for r in ranked)


def test_a_re_test_can_lower_a_score_but_never_raise_it():
    """A channel that *gained* events at a stricter threshold is capped at 1.0:
    a re-test is allowed to reduce confidence, not manufacture it."""
    store = _store(_survey(), _run("strict", 5.0, {"A1-A2": 99.0}, n_accepted=99))
    found = {r.channel: r for r in rank_channels(store)}
    assert found["A1-A2"].robustness == 1.0
    assert found["A1-A2"].score == pytest.approx(3.0)


def test_a_partially_silent_re_test_still_demotes_what_went_quiet():
    """The rule earning its place. Some channels keep events and others do not,
    so the zeros discriminate and must be allowed to bite."""
    store = _store(_survey(),
                   _run("strict", 5.0,
                        {"Z1-Z2": 9.0, "M1-M2": 0.0, "A1-A2": 0.0}, n_accepted=9))
    found = {r.channel: r for r in rank_channels(store)}
    assert found["Z1-Z2"].robustness == pytest.approx(0.75)
    assert found["Z1-Z2"].score == pytest.approx(9.0)
    assert found["M1-M2"].robustness == 0.0
    assert found["A1-A2"].robustness == 0.0
    # Nothing was waved away: this was a real re-test for all three.
    assert all(r.retested_at == [5.0] for r in found.values())
    assert all(r.unmeasured_at == [] for r in found.values())


def test_the_worst_of_several_re_tests_is_the_one_that_counts():
    store = _store(_survey(),
                   _run("mild", 4.0, {"Z1-Z2": 9.0}, n_accepted=9),
                   _run("harsh", 5.0, {"Z1-Z2": 6.0}, n_accepted=6))
    found = {r.channel: r for r in rank_channels(store)}
    assert found["Z1-Z2"].robustness == pytest.approx(0.5)
    assert found["Z1-Z2"].retested_at == [4.0, 5.0]


# -- the silent re-test ------------------------------------------------------


def test_a_silent_re_test_does_not_touch_the_ranking():
    """The headline. Against the previous rule every score here was 0.0 and the
    ranking came back exactly reversed -- the quietest contact first."""
    store = _store(_survey(), _run("silent", 5.0,
                                   dict.fromkeys(SURVEY, 0.0), n_accepted=0))
    ranked = rank_channels(store)
    assert [r.channel for r in ranked] == BY_RATE
    assert {r.robustness for r in ranked} == {1.0}
    assert all(r.score == pytest.approx(r.survey_rate_per_min) for r in ranked)


def test_a_silent_re_test_is_recorded_rather_than_dropped():
    """Skipping it quietly would be its own bug: the reader cannot tell a
    threshold that was never tried from one that found nothing."""
    store = _store(_survey(), _run("silent", 5.0,
                                   dict.fromkeys(SURVEY, 0.0), n_accepted=0))
    ranked = rank_channels(store)
    assert all(r.unmeasured_at == [5.0] for r in ranked)
    assert all(r.retested_at == [] for r in ranked)
    # And the run still appears in the provenance, because it did happen.
    assert all("silent" in r.run_ids for r in ranked)
    payload = ranked[0].as_dict()
    assert payload["unmeasured_at"] == [5.0]
    assert "no events on any channel" in payload["unmeasured_note"]


def test_a_ranking_with_nothing_left_in_it_does_not_look_confident():
    """What the bug actually cost: a score-ordered list with zero variance is
    alphabetical, and nothing in the old output said so."""
    store = _store(_survey(), _run("silent", 5.0,
                                   dict.fromkeys(SURVEY, 0.0), n_accepted=0))
    assert _ranked(store) != sorted(SURVEY)        # not alphabetical
    assert _ranked(store) == BY_RATE               # the measured order


def test_a_silent_re_test_among_real_ones_is_the_only_one_skipped():
    store = _store(_survey(),
                   _run("real", 4.0, {"Z1-Z2": 6.0}, n_accepted=6),
                   _run("silent", 5.0, {"Z1-Z2": 0.0}, n_accepted=0))
    found = {r.channel: r for r in rank_channels(store)}
    assert found["Z1-Z2"].retested_at == [4.0]
    assert found["Z1-Z2"].unmeasured_at == [5.0]
    assert found["Z1-Z2"].robustness == pytest.approx(0.5)


def test_a_silent_survey_is_not_rescued_by_this():
    """If the *broad* look found nothing there is no rate to defend, and the
    channel scores zero on its own merits rather than through a re-test."""
    store = _store(_run("survey", 3.0, dict.fromkeys(SURVEY, 0.0), n_accepted=0))
    ranked = rank_channels(store)
    assert {r.score for r in ranked} == {0.0}
    assert {r.robustness for r in ranked} == {1.0}


# -- telling silence from truncation ----------------------------------------


def test_silence_is_judged_on_the_run_total_not_the_reported_channels():
    """`channels` holds only the top k. A run whose top k happen to be quiet
    while channel 11 was firing is not silent, and `n_accepted` is the only
    field that knows."""
    assert detected_nothing({"n_accepted": 0, "channels": {}}) is True
    assert detected_nothing(
        {"n_accepted": 40,
         "channels": {"A1-A2": {"rate_per_min": 0.0}}}) is False


def test_an_old_run_without_the_total_falls_back_to_the_rates():
    """Rows are ordered by rate, so a leading zero means every rate is zero."""
    assert detected_nothing(
        {"channels": {c: {"rate_per_min": 0.0} for c in SURVEY}}) is True
    assert detected_nothing(
        {"channels": {"Z1-Z2": {"rate_per_min": 4.0},
                      "A1-A2": {"rate_per_min": 0.0}}}) is False


def test_an_output_that_says_nothing_either_way_is_not_called_silent():
    """Absence of evidence would suppress a real re-test, which is the bug
    this fix exists to remove, pointing the other way."""
    assert detected_nothing({}) is False
    assert detected_nothing({"channels": {}}) is False
    assert detected_nothing({"channels": {"A1-A2": {}}}) is False
    assert detected_nothing({"n_accepted": None, "channels": {}}) is False


def test_a_silent_re_test_that_failed_is_ignored_like_any_failed_run():
    """`ok_only` already excludes it; this pins that the new branch did not
    accidentally start reading failed runs."""
    store = _store(_survey(),
                   _run("broken", 5.0, dict.fromkeys(SURVEY, 0.0),
                        n_accepted=0, ok=False))
    ranked = rank_channels(store)
    assert all(r.unmeasured_at == [] for r in ranked)
    assert [r.channel for r in ranked] == BY_RATE
