"""Splitting ripples into sub-populations, and refusing to label them.

Roadmap item 6. A rate that merges physiological and epileptic ripples is the
largest silent assumption in this project, and no public archive carries the
label that would resolve it. These tests pin the honest half: the split is
exhaustive, the sub-populations are reported side by side, and the machinery
that asks *whether the split separates anything* can find a difference when
one exists and does not invent one when it does not.

The measured answer on real data lives in ``docs/EVALUATION.md`` §4 rather
than here, because the permutation takes minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.detectors.base import Event
from onset_hfo.populations import (
    INDEPENDENT,
    MORPHOLOGY,
    SPIKE_COUPLED,
    compare_populations,
    population_notes,
    population_rates,
    split_by_discharge,
)


def _event(channel="A1-A2", start=0.0, coupled=False, accepted=True, **fields):
    return Event(channel=channel, start=start, stop=start + 0.03, detector="rms",
                 band=(80.0, 250.0), co_occurs_with_spike=coupled, accepted=accepted,
                 **fields)


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def test_the_split_is_exhaustive_and_disjoint():
    events = [_event(start=i, coupled=i % 3 == 0) for i in range(30)]
    parts = split_by_discharge(events)
    assert len(parts[SPIKE_COUPLED]) + len(parts[INDEPENDENT]) == len(events)
    assert not (set(map(id, parts[SPIKE_COUPLED])) & set(map(id, parts[INDEPENDENT])))


def test_both_populations_exist_even_when_one_is_empty():
    """"No ripple here rode a discharge" is a finding; a missing key is not."""
    parts = split_by_discharge([_event(start=i) for i in range(5)])
    assert set(parts) == {SPIKE_COUPLED, INDEPENDENT}
    assert parts[SPIKE_COUPLED] == []


def test_rejected_events_are_excluded_by_default_like_every_other_rate():
    events = [_event(start=0, coupled=True), _event(start=1, coupled=True, accepted=False)]
    assert len(split_by_discharge(events)[SPIKE_COUPLED]) == 1
    assert len(split_by_discharge(events, accepted_only=False)[SPIKE_COUPLED]) == 2


def test_an_empty_recording_splits_into_two_empty_populations():
    parts = split_by_discharge([])
    assert parts[SPIKE_COUPLED] == [] and parts[INDEPENDENT] == []


# --------------------------------------------------------------------------
# The rates
# --------------------------------------------------------------------------


def test_the_two_rates_add_up_to_the_total_this_project_has_always_reported():
    """A reader must be able to reconcile the split with every earlier number."""
    from onset_hfo.metrics import channel_rates

    events = [_event(channel="A1-A2", start=i, coupled=i % 4 == 0) for i in range(20)]
    events += [_event(channel="B1-B2", start=i) for i in range(6)]
    channels = ["A1-A2", "B1-B2", "C1-C2"]

    split = population_rates(events, 60.0, channels).set_index("channel")
    total = channel_rates(events, 60.0, channels).set_index("channel")
    for ch in channels:
        assert split.loc[ch, "n_total"] == total.loc[ch, "n_events"]
        assert split.loc[ch, "rate_total"] == pytest.approx(total.loc[ch, "rate_per_min"])


def test_a_channel_nobody_detected_on_is_kept_at_zero_not_dropped():
    frame = population_rates([_event(channel="A1-A2")], 60.0, ["A1-A2", "Q9-Q10"])
    assert set(frame["channel"]) == {"A1-A2", "Q9-Q10"}
    quiet = frame[frame.channel == "Q9-Q10"].iloc[0]
    assert quiet["n_total"] == 0 and quiet["rate_total"] == 0.0
    assert np.isnan(quiet["share_spike_coupled"]), \
        "a channel with no events has no share, which is not the same as 0%"


def test_the_share_column_separates_channels_that_share_a_total_rate():
    """Two channels at the same rate are not the same object."""
    events = [_event(channel="A1-A2", start=i, coupled=True) for i in range(10)]
    events += [_event(channel="B1-B2", start=i, coupled=False) for i in range(10)]
    frame = population_rates(events, 60.0, ["A1-A2", "B1-B2"]).set_index("channel")
    assert frame.loc["A1-A2", "rate_total"] == frame.loc["B1-B2", "rate_total"]
    assert frame.loc["A1-A2", "share_spike_coupled"] == 1.0
    assert frame.loc["B1-B2", "share_spike_coupled"] == 0.0


# --------------------------------------------------------------------------
# Does the split separate anything? The controls that make a null readable.
# --------------------------------------------------------------------------


def _population(n, coupled, loc, rng):
    return [_event(start=float(i), coupled=coupled,
                   peak_frequency_hz=float(rng.normal(loc, 8.0)))
            for i in range(n)]


def test_a_difference_that_exists_by_construction_is_found():
    """The positive control. Without it a null result means nothing at all."""
    rng = np.random.default_rng(0)
    events = _population(60, True, 220.0, rng) + _population(240, False, 150.0, rng)
    frame = compare_populations(events, features=("peak_frequency_hz",),
                                n_boot=200, with_power=False)
    row = frame.iloc[0]
    assert row["auc"] > 0.95, "a separation of nine SDs should be near-perfect"
    assert bool(row["separates"]), "the test cannot detect a difference it was built on"


def test_a_split_with_no_difference_behind_it_reports_none():
    """The negative control, on the same machinery."""
    rng = np.random.default_rng(1)
    events = _population(60, True, 150.0, rng) + _population(240, False, 150.0, rng)
    frame = compare_populations(events, features=("peak_frequency_hz",),
                                n_boot=200, with_power=False)
    row = frame.iloc[0]
    assert 0.35 < row["auc"] < 0.65
    assert not bool(row["separates"])


def test_every_feature_tested_is_corrected_for_every_other():
    """Testing five features and quoting the smallest p is how a null becomes news."""
    rng = np.random.default_rng(2)
    events = _population(20, True, 150.0, rng) + _population(40, False, 150.0, rng)
    frame = compare_populations(events, features=MORPHOLOGY[:2], n_boot=100,
                                with_power=False)
    assert len(frame) == 2
    finite = frame[frame["p_permutation"].notna()]
    assert len(finite), "no feature produced a p-value to correct"
    assert (finite["p_bonferroni"] >= finite["p_permutation"]).all()
    assert (finite["p_bonferroni"] <= 1.0).all()


def test_the_power_floor_is_carried_so_a_null_can_be_read():
    """An AUC the sample could never have resolved is not evidence of no effect."""
    rng = np.random.default_rng(3)
    events = _population(15, True, 150.0, rng) + _population(45, False, 150.0, rng)
    frame = compare_populations(events, features=("peak_frequency_hz",),
                                n_boot=100, with_power=True)
    floor = frame.iloc[0]["min_detectable_auc"]
    assert 0.5 < floor <= 1.0
    assert frame["min_detectable_auc"].nunique() == 1, \
        "the floor depends on the sample sizes alone, so it is one number"


def test_a_population_too_small_to_compare_yields_no_verdict():
    events = [_event(start=0, coupled=True, peak_frequency_hz=150.0)]
    events += [_event(start=float(i), peak_frequency_hz=150.0) for i in range(1, 30)]
    frame = compare_populations(events, features=("peak_frequency_hz",),
                                n_boot=50, with_power=False)
    assert np.isnan(frame.iloc[0]["auc"]), "one event cannot separate anything"
    assert not bool(frame.iloc[0]["separates"])


# --------------------------------------------------------------------------
# What the report says
# --------------------------------------------------------------------------


def test_the_notes_state_the_split_and_refuse_to_label_it():
    events = {"rms": [_event(start=i, coupled=i < 5) for i in range(20)]}
    notes = population_notes(events, 60.0)
    assert any("5 of 20" in n for n in notes)
    assert any("proxy" in n and "not a label" in n for n in notes), \
        "the report must not let a proxy read as a label"
    assert any("nothing here can tell a physiological ripple" in n for n in notes)


def test_no_note_is_produced_for_a_detector_that_found_nothing():
    assert population_notes({"rms": []}, 60.0) == []


def test_the_pipeline_reports_both_populations_and_the_caveat():
    """End to end: the split reaches the saved result and the report."""
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording

    rec = make_synthetic_recording(duration_s=30.0, seed=4, verbose=False)
    result = run_pipeline(rec, verbose=False)
    assert set(result.populations) == set(result.events)
    for det, frame in result.populations.items():
        assert {"rate_spike_coupled", "rate_independent", "rate_total",
                "share_spike_coupled"} <= set(frame.columns)
        assert len(frame) == result.prepared.n_channels, \
            f"{det} dropped channels from the split view"
    assert any("proxy" in note for note in result.report.data_quality)


def test_the_saved_analysis_carries_the_split_as_its_own_table(tmp_path):
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording

    rec = make_synthetic_recording(duration_s=20.0, seed=5, verbose=False)
    result = run_pipeline(rec, verbose=False)
    out = result.save(tmp_path)
    written = {p.name for p in out.iterdir()}
    assert "populations_rms.csv" in written and "populations_line_length.csv" in written
