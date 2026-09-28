"""Splitting ripples into sub-populations, and refusing to label them.

Roadmap item 6. A rate that merges physiological and epileptic ripples is the
largest silent assumption in this project, and no public archive carries the
label that would resolve it. These tests pin the honest half: the split is
exhaustive, the sub-populations are reported side by side, and the machinery
that asks *whether the split separates anything* can find a difference when
one exists and does not invent one when it does not.

The measured answer on real data lives in ``docs/EVALUATION.md`` §3b rather
than here, because the permutation takes minutes.
"""

from __future__ import annotations

import functools

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


# The localisation screen, pinned
# --------------------------------------------------------------------------
# The measured answer to "does the split localise the resection better?"
# The screen itself takes ~40 minutes, so these read the committed extract
# and the reduction the script applies to it.


def _screen():
    import pandas as pd

    from onset_hfo.config import PROJECT_ROOT

    return pd.read_csv(PROJECT_ROOT / "data" / "outcome" / "subpopulation_screen.csv")


@functools.lru_cache(maxsize=1)
def _analysed_full():
    import sys

    from onset_hfo.config import PROJECT_ROOT

    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_subpopulation_outcome import analyse

    return analyse(_screen())


def _analysed():
    return _analysed_full().set_index(["band", "metric", "population"])["auc"]


def test_the_screen_covers_every_band_population_and_patient():
    frame = _screen()
    assert set(frame["band"]) == {"ripple", "fast_ripple"}
    assert set(frame["population"]) == {"merged", SPIKE_COUPLED, INDEPENDENT}
    assert frame["subject"].nunique() == 20
    assert frame["window"].nunique() == 5
    assert len(frame) == 2 * 20 * 5 * 3
    for column in ("share_in_rz", "top_channel_resected", "candidates_resected"):
        # NaN exactly where the population found nothing -- a window with no
        # events has no share and no busiest channel, and inventing a 0 there
        # would read as "none of them were resected".
        present = frame[column].notna()
        assert frame.loc[present, column].between(0.0, 1.0).all()
        assert (frame.loc[~present, "n_events"] == 0).all(), \
            f"{column} is NaN on a window that did have events"
    assert ((frame["n_events"] == 0) == frame["share_in_rz"].isna()).all(), \
        "every empty window should be NaN and every NaN an empty window"


def test_the_split_does_not_localise_better_than_the_merged_rate():
    """The answer, pinned because a null is what quietly drifts into a claim.

    Measured in the ripple band, where the populations are large enough to
    rank channels with. Every arm is at or below merged.
    """
    auc = _analysed()
    for metric in ("top_channel_resected", "candidates_resected", "share_in_rz"):
        merged = auc[("ripple", metric, "merged")]
        for population in (SPIKE_COUPLED, INDEPENDENT):
            gain = auc[("ripple", metric, population)] - merged
            assert gain <= 0.0, (
                f"{population} now beats merged on {metric} by {gain:+.3f} in "
                f"the ripple band; EVALUATION.md §3b says the split buys "
                f"nothing and would need rewriting")


def test_the_fast_ripple_coupled_population_is_too_small_to_rank_channels():
    """Why the answer comes from the ripple band at all.

    The pre-specified arm is fast ripples. Its spike-coupled population has a
    median of one event per 60 s window, so an argmax over it is a coin flip
    -- and it is that arm, not the ripple one, that appears to beat merged.
    """
    frame = _screen()
    coupled = frame[frame["population"] == SPIKE_COUPLED]
    fast = coupled[coupled["band"] == "fast_ripple"]["n_events"]
    ripple = coupled[coupled["band"] == "ripple"]["n_events"]

    assert fast.median() <= 2, f"fast-ripple coupled median is now {fast.median()}"
    assert (fast < 5).sum() > len(fast) // 2, \
        "most fast-ripple windows should have almost no coupled events"
    assert ripple.median() > 10 * max(fast.median(), 1), \
        "the ripple band is used because it carries the events; it no longer does"


def test_the_tie_aware_metric_contradicts_the_flattering_fast_ripple_argmax():
    """The check that retired the apparent gain, pinned as a regression test.

    `candidates_resected` was added (item 10 step 3) to stop an argmax
    overclaiming when channels are statistically tied. Here it earns that: on
    fast ripples the coupled arm looks best by `top_channel_resected` and
    worst -- below chance -- by the tie-aware metric.
    """
    auc = _analysed()
    argmax = auc[("fast_ripple", "top_channel_resected", SPIKE_COUPLED)]
    tie_aware = auc[("fast_ripple", "candidates_resected", SPIKE_COUPLED)]

    assert argmax > auc[("fast_ripple", "top_channel_resected", "merged")]
    assert tie_aware < 0.5 < argmax, (
        f"the contradiction that retired the fast-ripple gain is gone: "
        f"argmax {argmax:.3f}, tie-aware {tie_aware:.3f}")


def test_the_merged_arm_replicates_across_bands_and_the_coupled_arm_does_not():
    """The evidence that the fast-ripple coupled AUC was noise in one arm.

    A band effect would move both arms. Only the coupled one collapses.
    """
    auc = _analysed()
    merged = [auc[(b, "top_channel_resected", "merged")]
              for b in ("fast_ripple", "ripple")]
    coupled = [auc[(b, "top_channel_resected", SPIKE_COUPLED)]
               for b in ("fast_ripple", "ripple")]

    assert abs(merged[0] - merged[1]) < 0.05, \
        f"merged no longer replicates across bands: {merged[0]:.3f} vs {merged[1]:.3f}"
    assert coupled[0] - coupled[1] > 0.25, \
        f"the coupled arm no longer collapses: {coupled[0]:.3f} vs {coupled[1]:.3f}"


#: ``min_detectable_auc`` per group size, for the four sizes the screen's arms
#: have. A property of the group sizes, not of the data, and it simulates for
#: about forty seconds a call -- so it is recorded here rather than recomputed,
#: and `test_the_power_floor_is_carried_so_a_null_can_be_read` above covers the
#: function itself. §3b quotes these.
POWER_FLOORS = {(13, 7): 0.85, (12, 7): 0.86, (13, 6): 0.87, (12, 6): 0.88}


def test_nothing_in_the_screen_clears_its_own_power_floor():
    """13 seizure-free against 7 recurrences resolves an AUC of 0.85 or better.

    So the negative result is 'no gain to report', not 'no effect' -- and no
    positive one here would have been reportable either.

    Asserted against the *lowest* of the four floors rather than each arm's
    own: if no arm reaches 0.85 then none reaches its own floor either, since
    every floor is at least that. Losing patients only raises the floor.
    """
    result = _analysed_full()
    sizes = {(int(r.n_SF), int(r.n_rec)) for r in result.itertuples()}
    assert sizes <= set(POWER_FLOORS), f"unrecorded group size in {sizes}"

    lowest = min(POWER_FLOORS[s] for s in sizes)
    worst = result.loc[result.auc.idxmax()]
    assert worst.auc < lowest, (
        f"{worst.band}/{worst.metric}/{worst.population} now reaches "
        f"{worst.auc:.3f} against a floor of at least {lowest} at n = "
        f"{worst.n_SF}/{worst.n_rec}; §3b's hedging needs revisiting")


def test_the_recorded_power_floors_are_what_the_simulation_gives():
    """Guard the hardcoded floors above against drifting from the function.

    At a fraction of the simulations the answer lands within a few steps of
    the grid, so this checks agreement to 0.03 rather than exactly -- enough to
    catch a floor that is simply wrong, cheap enough to run every time.
    """
    from onset_hfo.outcome import min_detectable_auc

    for (n_free, n_recurrence), recorded in POWER_FLOORS.items():
        estimate = min_detectable_auc(n_free, n_recurrence, n_sims=300, seed=1)
        assert abs(estimate - recorded) <= 0.03, (
            f"min_detectable_auc({n_free}, {n_recurrence}) is about "
            f"{estimate}, not the recorded {recorded}")


def test_the_arms_are_scored_on_nearly_the_same_patients():
    """The arms must stay comparable to each other, and be seen to.

    A sub-population with no events in any window drops that patient from that
    arm alone, so `merged` and `spike_coupled` are not scored on identical
    people. Two facts keep the comparison honest and are pinned here: `merged`
    never loses anyone, and the ripple band -- the one the conclusion comes
    from -- loses at most one patient. The fast ripple band loses two, which is
    part of why its numbers are not the ones reported. §3b discloses both.
    """
    result = _analysed_full()
    merged = result[result.population == "merged"]
    assert set(zip(merged.n_SF, merged.n_rec, strict=True)) == {(13, 7)}, \
        "the merged arm should always carry every patient"

    for row in result.itertuples():
        lost = (13 - int(row.n_SF)) + (7 - int(row.n_rec))
        limit = 1 if row.band == "ripple" else 2
        assert lost <= limit, (
            f"{row.band}/{row.metric}/{row.population} is scored on "
            f"{row.n_SF}/{row.n_rec} patients, {lost} fewer than merged "
            f"(limit {limit}); the arms are drifting apart")
