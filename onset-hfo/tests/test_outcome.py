"""Resected-zone mapping and the outcome statistics.

All offline: the clinical sheet is replaced by hand-written fixtures so the
tests fail on a parsing regression rather than on a network hiccup. The
fixtures use the real grammar from ``ds003498``'s sheet, typo included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.clinical import Resection, classify_channels
from onset_hfo.datasets import expand_contact_ranges
from onset_hfo.outcome import (
    _auc,
    _share_in_resection,
    _top_channel_resected,
    _top_k_resected,
    compare_groups,
    min_detectable_auc,
    rank_comparison,
)

# -- contact-range expansion ----------------------------------------------

def test_range_expands_inclusively():
    """``ahr1-4`` is four contacts, not three and not a subtraction."""
    contacts, unparsed = expand_contact_ranges("ahr1-4, ar1-4, phr1-4")
    assert contacts[:4] == ["ahr1", "ahr2", "ahr3", "ahr4"]
    assert len(contacts) == 12 and unparsed == []


def test_unparsable_chunk_is_reported_not_dropped():
    """The sheet's real typo (``1ll22-24`` for ``tll22-24``) must surface.

    A parser that returned the other three ranges and said nothing would
    silently shrink the resected zone for that subject.
    """
    contacts, unparsed = expand_contact_ranges("tll6-8, tll14-16, 1ll22-24, tll31-32")
    assert unparsed == ["1ll22-24"]
    assert "tll22" not in contacts and len(contacts) == 8


def test_absurd_range_is_refused():
    """A 200-contact range is a typo, not an electrode, so it is not expanded."""
    contacts, unparsed = expand_contact_ranges("g1-200")
    assert contacts == [] and unparsed == ["g1-200"]


# -- channel classification ------------------------------------------------

@pytest.fixture
def resection():
    return Resection(subject="sub-01", resected=("AHR1", "AHR2", "AHR3"),
                     eloquent=("AHR5",), rz_text="ahr1-3", excluded_text="ahr5")


def test_both_contacts_inside_is_resected(resection):
    frame = classify_channels(["AHR1-AHR2"], resection).set_index("channel")
    assert frame.loc["AHR1-AHR2", "zone"] == "resected"


def test_margin_channel_is_partial_not_resected(resection):
    """AHR3-AHR4 straddles the resection edge and is claimed by neither side."""
    frame = classify_channels(["AHR3-AHR4"], resection).set_index("channel")
    assert frame.loc["AHR3-AHR4", "zone"] == "partial"
    assert frame.loc["AHR3-AHR4", "n_resected_contacts"] == 1


def test_outside_channel_is_spared(resection):
    frame = classify_channels(["AHR5-AHR6"], resection).set_index("channel")
    assert frame.loc["AHR5-AHR6", "zone"] == "spared"


def test_eloquent_is_a_flag_not_a_zone(resection):
    """Stimulation-positive contacts are marked, not relabelled."""
    frame = classify_channels(["AHR5-AHR6"], resection).set_index("channel")
    assert bool(frame.loc["AHR5-AHR6", "eloquent"]) is True
    assert frame.loc["AHR5-AHR6", "zone"] == "spared"


def test_monopolar_channel_names_still_classify(resection):
    frame = classify_channels(["AHR1"], resection).set_index("channel")
    assert frame.loc["AHR1", "zone"] == "resected" and frame.loc["AHR1", "n_contacts"] == 1


def test_coverage_reports_unrecorded_resected_contacts(resection):
    """Only one of three resected contacts was recorded, and it says so."""
    cov = resection.coverage(["AHR1-AHR2"])
    assert cov["rz_contacts_listed"] == 3 and cov["rz_contacts_recorded"] == 2
    assert cov["missing"] == "AHR3"
    assert cov["rz_coverage"] == pytest.approx(2 / 3)


# -- the per-subject share -------------------------------------------------

def test_margin_activity_counts_against_the_share():
    """Events on a margin channel are in the denominator, not the numerator."""
    rates = pd.Series({"A": 6.0, "B": 3.0, "C": 1.0})
    zones = pd.Series({"A": "resected", "B": "partial", "C": "spared"})
    share = _share_in_resection(rates, zones)
    assert share["share_in_rz"] == pytest.approx(0.6)
    assert share["share_in_rz_incl_partial"] == pytest.approx(0.9)
    assert share["n_events_spared"] == pytest.approx(1.0)


def test_silent_recording_gives_nan_not_zero():
    """No events means "we cannot say", which is not the same as "none inside"."""
    rates = pd.Series({"A": 0.0, "B": 0.0})
    zones = pd.Series({"A": "resected", "B": "spared"})
    assert np.isnan(_share_in_resection(rates, zones)["share_in_rz"])
    assert np.isnan(_top_channel_resected(rates, zones))


# -- refusing to pick a winner the data did not pick -----------------------

def test_a_clear_leader_gives_a_set_of_one():
    """When the leader's interval clears the field, nothing changes."""
    from onset_hfo.outcome import candidate_channels

    counts = pd.Series({"A": 300.0, "B": 20.0, "C": 5.0})
    assert candidate_channels(counts, duration_min=5.0) == ["A"]


def test_overlapping_intervals_all_join_the_candidate_set():
    """Three channels within noise of each other are three candidates."""
    from onset_hfo.outcome import candidate_channels

    counts = pd.Series({"A": 30.0, "B": 28.0, "C": 25.0, "D": 3.0})
    assert candidate_channels(counts, duration_min=5.0) == ["A", "B", "C"]


def test_the_set_tightens_as_the_window_grows():
    """The rule is an interval, not a constant, so more data narrows it.

    Same rates, five times the recording: the Poisson intervals shrink and
    channels that were tied separate. This is why no threshold needed tuning.
    """
    from onset_hfo.outcome import candidate_channels

    short = pd.Series({"A": 30.0, "B": 20.0, "C": 3.0})
    long = pd.Series({"A": 150.0, "B": 100.0, "C": 15.0})
    assert candidate_channels(short, 5.0) == ["A", "B"]     # cannot be separated
    assert candidate_channels(long, 25.0) == ["A"]          # five times the data can


def test_candidates_reduce_to_the_argmax_when_the_set_is_one():
    """A strict generalisation: identical answer whenever the leader is alone."""
    from onset_hfo.outcome import _candidate_metrics

    rates = pd.Series({"A": 300.0, "B": 20.0})
    zones = pd.Series({"A": "resected", "B": "spared"})
    out = _candidate_metrics(rates, zones, duration_min=5.0)
    assert out["n_candidates"] == 1
    assert out["candidates_resected"] == _top_channel_resected(rates, zones)
    assert out["leader_alone"] == 1.0


def test_a_score_that_is_not_a_count_is_refused_rather_than_truncated():
    """The tie-aware metric is undefined for the planner's multiplied score, and
    `run_robustness_ablation.py` leaves it NaN for that reason. That discipline
    used to live in a comment in that one script: a float arriving here was
    floored, so `rate x robustness` of 8.37 became a Poisson interval around 8
    events -- a confident interval for a quantity that counts nothing.
    """
    from onset_hfo.outcome import NotACount, candidate_channels

    multiplied = pd.Series({"A": 8.37, "B": 6.12, "C": 1.5})
    with pytest.raises(NotACount) as raised:
        candidate_channels(multiplied, duration_min=5.0)
    # The message has to say which values and why, or the next caller guesses.
    assert "A=8.37" in str(raised.value)
    assert "not a count" in str(raised.value)
    # Catchable as a ValueError, so an existing handler is not bypassed.
    assert isinstance(raised.value, ValueError)


def test_integral_floats_are_counts_and_still_work():
    """Counts live in float64 columns all over this codebase -- the reviewer
    passes `findings["n_events"]`. A dtype check instead of a value check would
    have broken every real caller."""
    from onset_hfo.outcome import candidate_channels

    assert candidate_channels(pd.Series({"A": 300.0, "B": 20.0}), 5.0) == ["A"]
    assert candidate_channels(pd.Series({"A": 300, "B": 20}), 5.0) == ["A"]


def test_a_fractional_count_does_not_sneak_past_on_a_quiet_channel():
    """The check covers every value, not just the leader's."""
    from onset_hfo.outcome import NotACount, candidate_channels

    with pytest.raises(NotACount):
        candidate_channels(pd.Series({"A": 300.0, "B": 20.0, "C": 0.4}), 5.0)


def test_a_tied_set_reports_a_fraction_not_a_winner():
    from onset_hfo.outcome import _candidate_metrics

    rates = pd.Series({"A": 30.0, "B": 28.0, "C": 25.0})
    zones = pd.Series({"A": "resected", "B": "spared", "C": "resected"})
    out = _candidate_metrics(rates, zones, duration_min=5.0)
    assert out["n_candidates"] == 3
    assert out["candidates_resected"] == pytest.approx(2 / 3)
    assert out["candidates_all_resected"] == 0.0
    assert out["leader_alone"] == 0.0


def test_a_silent_recording_has_no_candidates():
    from onset_hfo.outcome import _candidate_metrics, candidate_channels

    rates = pd.Series({"A": 0.0, "B": 0.0})
    zones = pd.Series({"A": "resected", "B": "spared"})
    assert candidate_channels(rates, 5.0) == []
    assert np.isnan(_candidate_metrics(rates, zones, 5.0)["candidates_resected"])


def test_the_candidate_set_does_not_depend_on_channel_order():
    """Otherwise the metric would move when the montage is built differently."""
    from onset_hfo.outcome import candidate_channels

    counts = pd.Series({"A": 30.0, "B": 30.0, "C": 3.0})
    assert candidate_channels(counts, 5.0) == candidate_channels(counts[::-1], 5.0)


def test_top_k_area_is_a_fraction_of_the_busiest_channels():
    """The "HFO area" version: how much of the top-3 did the surgeon take?"""
    rates = pd.Series({"A": 9.0, "B": 8.0, "C": 7.0, "D": 1.0})
    zones = pd.Series({"A": "resected", "B": "spared", "C": "resected", "D": "resected"})
    assert _top_k_resected(rates, zones, k=3) == pytest.approx(2 / 3)


def test_top_channel_is_binary():
    rates = pd.Series({"A": 1.0, "B": 9.0})
    zones = pd.Series({"A": "resected", "B": "spared"})
    assert _top_channel_resected(rates, zones) == 0.0
    assert _top_channel_resected(rates, pd.Series({"A": "spared", "B": "resected"})) == 1.0


# -- statistics ------------------------------------------------------------

def test_auc_is_a_probability_with_ties_at_half():
    assert _auc(np.array([1.0, 2.0]), np.array([0.0])) == 1.0
    assert _auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert _auc(np.array([0.0]), np.array([1.0])) == 0.0


def test_perfect_separation_is_significant_and_reported_as_such():
    free = np.array([0.9, 0.8, 0.7, 0.75, 0.95, 0.85])
    recur = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.05])
    out = rank_comparison(free, recur, n_boot=200)
    assert out["auc"] == 1.0 and out["rank_biserial"] == 1.0
    assert out["p_permutation"] < 0.01


def test_no_difference_is_not_significant():
    values = np.array([0.4, 0.5, 0.6, 0.45, 0.55, 0.5])
    out = rank_comparison(values, values, n_boot=200)
    assert out["auc"] == pytest.approx(0.5, abs=0.05)
    assert out["p_permutation"] > 0.5


def test_too_few_subjects_returns_nan_rather_than_a_number():
    out = rank_comparison(np.array([0.5]), np.array([0.1]), n_boot=50)
    assert np.isnan(out["auc"]) and np.isnan(out["p_permutation"])


def test_nan_shares_are_excluded_from_the_comparison():
    """A subject with no events must not be counted as a zero."""
    out = rank_comparison(np.array([0.9, 0.8, np.nan]), np.array([0.1, 0.2]), n_boot=50)
    assert out["n_seizure_free"] == 2


def test_exact_permutation_matches_a_naive_enumeration():
    """The vectorised rank version must agree with the obvious slow one.

    The speed-up is what makes the exact path affordable, so it is worth a
    test that it did not also change the answer.
    """
    from itertools import combinations

    from onset_hfo.outcome import _permutation_p

    free = np.array([0.9, 0.4, 0.8, 0.55, 0.7])
    recur = np.array([0.3, 0.6, 0.2])
    auc = _auc(free, recur)
    fast = _permutation_p(free, recur, auc)

    pooled = np.concatenate([free, recur])
    observed = abs(auc - 0.5)
    hits = total = 0
    for idx in combinations(range(len(pooled)), len(free)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idx)] = True
        hits += abs(_auc(pooled[mask], pooled[~mask]) - 0.5) >= observed - 1e-12
        total += 1
    assert fast == pytest.approx(hits / total)


def test_monte_carlo_path_agrees_with_the_exact_one():
    """Forcing the sampled path must land close to the enumerated answer."""
    from onset_hfo.outcome import _permutation_p

    free = np.array([0.9, 0.4, 0.8, 0.55, 0.7, 0.65])
    recur = np.array([0.3, 0.6, 0.2, 0.35])
    auc = _auc(free, recur)
    exact = _permutation_p(free, recur, auc)
    sampled = _permutation_p(free, recur, auc, exact_limit=1, n_draws=40000, seed=3)
    assert sampled == pytest.approx(exact, abs=0.02)


def test_ties_are_handled_as_half_credit_in_the_permutation_test():
    """A binary metric is nothing but ties; the p must still be a probability."""
    from onset_hfo.outcome import _permutation_p

    free = np.array([1.0] * 5 + [0.0])
    recur = np.array([0.0] * 4)
    p = _permutation_p(free, recur, _auc(free, recur))
    assert 0.0 < p <= 1.0


def test_power_floor_is_high_for_this_cohort():
    """13 vs 7 can only detect a large effect, and the number must say so."""
    floor = min_detectable_auc(13, 7, n_sims=300, seed=1)
    assert 0.75 <= floor <= 0.95


def test_compare_groups_splits_on_published_outcome():
    subjects = pd.DataFrame({
        "subject": ["sub-01", "sub-02", "sub-03", "sub-04"],
        "source": ["expert"] * 4, "scope": ["reviewed"] * 4, "band": ["ripple"] * 4,
        "share_in_rz": [0.9, 0.8, 0.1, 0.2],
        "share_in_rz_incl_partial": [0.9, 0.8, 0.1, 0.2],
        "top_channel_resected": [1.0, 1.0, 0.0, 0.0],
        "top3_resected": [1.0, 0.667, 0.0, 0.333],
        "candidates_resected": [1.0, 1.0, 0.0, 0.5],
        "candidates_all_resected": [1.0, 1.0, 0.0, 0.0],
    })
    participants = pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03", "sub-04"],
                                 "outcome": ["S", "S", "F", "F"]})
    groups = compare_groups(subjects, participants)
    row = groups.query("metric == 'share_in_rz'").iloc[0]
    assert row["n_seizure_free"] == 2 and row["n_recurrence"] == 2
    assert row["auc"] == 1.0
