"""The signal-processing half: data, preprocessing, detectors, metrics, report."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from onset_hfo.config import DetectorConfig, PreprocessConfig, ValidationConfig
from onset_hfo.detectors import detect_line_length, detect_rms, detect_spikes
from onset_hfo.detectors.base import (
    Event,
    events_to_frame,
    frame_to_events,
    robust_scale,
    sliding_line_length,
    sliding_rms,
    threshold_segments,
)
from onset_hfo.evaluate import evaluate_detections, validation_benefit
from onset_hfo.metrics import compare_rankings, match_events, poisson_ci
from onset_hfo.preprocess import bipolar_pairs, prepare
from onset_hfo.validate import validate_events

# -- primitives ------------------------------------------------------------

def test_robust_scale_ignores_outliers():
    """The whole point of a MAD-based SD: a few huge samples must not move it."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    contaminated = x.copy()
    contaminated[:50] = 100.0
    clean = float(np.squeeze(robust_scale(x)[1]))
    dirty = float(np.squeeze(robust_scale(contaminated)[1]))
    assert abs(clean - dirty) < 0.05
    assert np.std(contaminated) > 2 * dirty  # a plain SD would be wrecked


def test_sliding_features_have_the_same_length_as_the_signal():
    x = np.sin(np.linspace(0, 40, 500))
    assert sliding_rms(x, 7).shape == x.shape
    assert sliding_line_length(x, 7).shape == x.shape


def test_threshold_segments_merges_then_filters():
    trace = np.array([0, 5, 5, 0, 5, 5, 0, 0, 0, 5, 0], dtype=float)
    # gap of one sample is merged; the isolated late crossing is too short
    assert threshold_segments(trace, 1.0, min_len=3, merge_gap=2) == [(1, 6)]


def test_bipolar_pairs_never_bridge_a_missing_contact():
    pairs = bipolar_pairs(["AD1", "AD2", "AD4", "B1", "B2"])
    assert ("AD1", "AD2") in pairs
    assert ("AD2", "AD4") not in pairs       # AD3 is missing: no pair
    assert ("AD4", "B1") not in pairs        # different electrodes never pair
    assert ("B1", "B2") in pairs


def test_poisson_interval_brackets_the_rate():
    lo, hi = poisson_ci(12, duration_min=1.0)
    assert lo < 12 < hi
    wide_lo, wide_hi = poisson_ci(3, duration_min=1.0)
    assert (wide_hi - wide_lo) / 3 > (hi - lo) / 12   # fewer events, relatively wider


# -- preprocessing ---------------------------------------------------------

def test_preprocessing_produces_bipolar_microvolts(prepared, recording):
    assert prepared.montage == "bipolar"
    assert prepared.n_channels == len(prepared.pairs)
    assert prepared.sfreq == recording.sfreq
    # iEEG in microvolts: tens to hundreds, never volts or nanovolts
    assert 1.0 < float(np.std(prepared.data)) < 5000.0
    assert any("notch" in step for step in prepared.steps)
    assert any("bipolar" in step for step in prepared.steps)


def test_reported_times_are_original_recording_times(recording):
    """A window in a report must be findable in the archive file."""
    recording.t_offset = 50.0
    prep = prepare(recording, verbose=False)
    events = detect_rms(prep, describe_spectrum=False)
    assert events, "the synthetic recording should contain detectable ripples"
    assert all(e.start >= 50.0 for e in events)
    recording.t_offset = 0.0


# -- detectors -------------------------------------------------------------

def test_detectors_find_the_hot_channels(result, recording):
    """The contacts with 20x the implanted event rate must come out on top."""
    hot = set(recording.marked_contacts)
    for name in ("rms", "line_length"):
        top = result.rates[name].head(3)["channel"]
        assert any(set(ch.split("-")) & hot for ch in top), f"{name} missed the hot contacts"


def test_detected_events_are_oscillations(result):
    for name in ("rms", "line_length"):
        accepted = [e for e in result.events[name] if e.accepted]
        assert accepted
        assert all(e.n_peaks >= 6 for e in accepted)          # oscillation criterion
        assert all(5.0 <= e.duration_ms <= 250.0 for e in accepted)
        assert all(80.0 <= e.peak_frequency_hz <= 250.0
                   for e in accepted if np.isfinite(e.peak_frequency_hz))


def test_a_flat_channel_produces_no_events(prepared):
    """Zero signal must produce zero detections, not a divide-by-zero storm."""
    import copy

    flat = copy.copy(prepared)
    flat.data = np.zeros_like(prepared.data[:2])
    flat.ch_names = prepared.ch_names[:2]
    flat.pairs = prepared.pairs[:2]
    assert detect_rms(flat) == []
    assert detect_spikes(flat) == []


def test_rejected_events_are_kept_with_a_reason(result):
    rejected = [e for evs in result.events.values() for e in evs if not e.accepted]
    assert rejected, "the synthetic recording contains artifacts that must be rejected"
    assert all(e.reject_reason for e in rejected)


def test_validation_raises_precision_without_gutting_recall(result, recording):
    table = validation_benefit(result.events["rms"], recording.ground_truth)
    raw = table[table["stage"] == "raw detector output"].iloc[0]
    validated = table[table["stage"] == "after artifact rejection"].iloc[0]
    assert validated["precision"] > raw["precision"] + 0.2
    assert validated["recall"] > raw["recall"] - 0.15


def test_spike_detector_rejects_discontinuities(prepared):
    """An electrode pop is a step; after a 5-60 Hz filter it looks like a spike."""
    import copy

    doctored = copy.copy(prepared)
    data = np.zeros((1, int(prepared.sfreq * 4)))
    rng = np.random.default_rng(1)
    data[0] = rng.normal(0, 20, data.shape[1])
    jump = int(prepared.sfreq * 2)
    data[0, jump:jump + 60] += 1500.0 * np.exp(-np.arange(60) / 20.0)   # a pop
    doctored.data = data
    doctored.ch_names = ["POP1-POP2"]
    doctored.pairs = [("POP1", "POP2")]
    events = detect_spikes(doctored)
    assert not any(abs(e.mid - 2.0) < 0.1 for e in events)


def test_scores_are_stable_across_reruns(prepared):
    first = detect_rms(prepared, describe_spectrum=False)
    second = detect_rms(prepared, describe_spectrum=False)
    assert [(e.channel, round(e.start, 4)) for e in first] == \
           [(e.channel, round(e.start, 4)) for e in second]


def test_thresholds_behave_monotonically(prepared):
    low = detect_rms(prepared, DetectorConfig(threshold_sd=3.0), describe_spectrum=False)
    high = detect_rms(prepared, DetectorConfig(threshold_sd=7.0), describe_spectrum=False)
    assert len(low) > len(high)


# -- metrics and evaluation -------------------------------------------------

def test_channel_rates_keep_silent_channels(result):
    rates = result.rates["rms"]
    assert len(rates) == result.prepared.n_channels
    assert (rates["rate_per_min"] >= 0).all()
    assert (rates["rate_ci_low"] <= rates["rate_per_min"]).all()
    assert (rates["rate_ci_high"] >= rates["rate_per_min"]).all()


def test_matching_is_one_to_one():
    a = [Event("C1", 1.0, 1.05, "rms", (80, 250)), Event("C1", 2.0, 2.05, "rms", (80, 250))]
    b = [Event("C1", 1.01, 1.06, "line_length", (80, 250))]
    pairs, only_a, only_b = match_events(a, b)
    assert pairs == [(0, 0)] and only_a == [1] and only_b == []


def test_disagreement_needs_a_leading_channel():
    import pandas as pd

    a = pd.DataFrame({"channel": ["X", "Y"], "rate_per_min": [10.0, 1.0], "n_events": [10, 1]})
    b = pd.DataFrame({"channel": ["X", "Y"], "rate_per_min": [1.0, 10.0], "n_events": [1, 10]})
    comparison = compare_rankings(a, b, "rms", "line_length", disagreement_ranks=1, top_k=1)
    assert comparison["disagrees"].any()


def test_evaluation_matches_through_bipolar_contacts(recording, result):
    scores = evaluate_detections(result.events["rms"], recording.ground_truth, detector="rms")
    assert 0.2 < scores.recall <= 1.0
    assert scores.precision > 0.8
    assert scores.n_truth == int((recording.ground_truth["kind"] == "ripple").sum())


def test_events_survive_a_table_round_trip(result):
    events = result.events["rms"][:20]
    restored = frame_to_events(events_to_frame(events))
    assert [e.channel for e in restored] == [e.channel for e in sorted(
        events, key=lambda x: (x.channel, x.start))]
    assert all(np.isclose(a.start, b.start) for a, b in
               zip(restored, sorted(events, key=lambda x: (x.channel, x.start)), strict=False))


# -- report ----------------------------------------------------------------

def test_report_has_no_recommendation_anywhere(result):
    """No recommendation field, and no directive language in the text."""
    assert "recommendation" not in result.report.as_dict()
    payload = result.report.to_json().lower()
    for banned in ("we recommend", "should be resected", "should be removed", "resect the",
                   "treatment plan", "ablate"):
        assert banned not in payload, f"the report must never contain {banned!r}"
    # The only mention of a recommendation is the statement that there is none.
    for hit in re.finditer(r"recommend", payload):
        window = payload[max(0, hit.start() - 30):hit.start()]
        assert "no " in window or "contains no" in window


def test_every_finding_carries_evidence(result):
    for finding in result.report.findings:
        assert finding.evidence, f"{finding.channel} has no evidence window"
        for window in finding.evidence:
            assert window.stop > window.start
            assert window.evidence_id.count("|") == 3


def test_report_states_limitations_and_data_quality(result):
    assert len(result.report.limitations) >= 3
    assert any("artifact validation" in q for q in result.report.data_quality)
    assert "no diagnosis" in result.report.to_markdown().lower()


@pytest.mark.parametrize("kind", ["ripple", "spike"])
def test_synthetic_ground_truth_is_actually_detectable(recording, result, kind):
    events = result.spikes if kind == "spike" else result.events["rms"]
    scores = evaluate_detections(events, recording.ground_truth, kind=kind)
    assert scores.recall > 0.3, f"{kind} recall collapsed: {scores.summary()}"


def test_validation_config_is_honoured(prepared):
    events = detect_line_length(prepared)
    validate_events(events, prepared, ValidationConfig(min_peak_prominence_db=99.0))
    assert not any(e.accepted for e in events), "an impossible threshold must reject everything"


def test_the_offline_guard_actually_refuses(tmp_path):
    """The suite claims to be offline; this is what makes that a guarantee.

    If this test ever fails, the guard has stopped working and every other
    test in the suite may be silently reaching the archive again.
    """
    from onset_hfo.datasets import OfflineError, _http_get, offline

    assert offline(), "the session fixture should have enabled offline mode"
    with pytest.raises(OfflineError, match="ONSET_HFO_OFFLINE"):
        _http_get("https://s3.amazonaws.com/openneuro.org/ds003029/participants.tsv")


def test_labels_fall_back_gracefully_when_the_archive_is_unreachable():
    """soz_labels must degrade to what it has locally, not raise."""
    from onset_hfo.cohort import soz_labels

    labels = soz_labels("sub-pt01")
    assert labels.source in ("none", "clinical_summary")


# -- preprocessing options the GUI exposes ---------------------------------
#
# These used to be unreachable without editing a config in Python, which in
# practice meant nobody changed them and nobody checked whether the defaults
# suited their recording. Exposing them in the interface makes two things
# load-bearing: that the defaults still produce exactly what they produced
# before, and that a setting which would destroy the band is refused rather
# than applied quietly.

def test_the_defaults_are_unchanged_by_the_new_options(recording):
    """The measured defaults are the shipped ones; adding controls moved none."""
    cfg = PreprocessConfig()
    assert (cfg.highpass, cfg.lowpass) == (1.0, None)
    assert (cfg.notch, cfg.notch_width, cfg.notch_harmonics) == (True, 2.0, True)
    assert (cfg.bipolar, cfg.average_reference) == (True, False)
    assert (cfg.resample, cfg.drop_bads, cfg.exclude) == (None, True, ())

    prepared = prepare(recording, cfg, verbose=False)
    plain = prepare(recording, None, verbose=False)
    assert np.array_equal(prepared.data, plain.data)
    assert prepared.steps == plain.steps
    assert prepared.montage == "bipolar"


@pytest.mark.parametrize("changes,expected", [
    ({"lowpass": 400.0}, "low-pass 400 Hz"),
    ({"notch_width": 4.0}, "4 Hz wide"),
    ({"notch_harmonics": False}, "fundamental only"),
    ({"resample": 1000.0}, "resampled 2000 Hz -> 1000 Hz"),
    ({"bipolar": False, "average_reference": True}, "common average reference"),
])
def test_each_option_reaches_the_signal_and_is_logged(recording, changes, expected):
    """A step a report does not mention is a step nobody can check."""
    prepared = prepare(recording, replace(PreprocessConfig(), **changes),
                       verbose=False)
    assert any(expected in step for step in prepared.steps), prepared.steps


def test_resampling_changes_the_rate(recording):
    prepared = prepare(recording, replace(PreprocessConfig(), resample=1000.0),
                       verbose=False)
    assert prepared.sfreq == 1000.0


def test_the_average_reference_says_what_it_costs(recording):
    """It re-introduces the shared noise the bipolar montage exists to remove,
    and the step has to say so, because the number it changes looks the same."""
    prepared = prepare(recording,
                       replace(PreprocessConfig(), bipolar=False,
                               average_reference=True), verbose=False)
    assert prepared.montage == "average"
    note = next(s for s in prepared.steps if "common average" in s)
    assert "bipolar montage exists to avoid" in note


def test_reviewer_marked_channels_are_logged_apart_from_the_datasets(recording):
    """One is the archive's claim, the other is this reviewer's judgement.

    Merging them in the report would attribute a reviewer's call to the
    dataset, which is the sort of thing that survives into a paper.
    """
    victim = recording.raw.ch_names[0]
    prepared = prepare(recording, replace(PreprocessConfig(), exclude=(victim,)),
                       verbose=False)
    note = next(s for s in prepared.steps if "reviewer marked bad" in s)
    assert victim in note
    assert all(victim not in name.split("-") for name in prepared.ch_names)


@pytest.mark.parametrize("changes,message", [
    ({"lowpass": 0.5}, "passes nothing"),
    ({"resample": 500.0, "lowpass": 400.0}, "Nyquist"),
    ({"notch_width": 0.0}, "notch width must be positive"),
    ({"resample": 0.0}, "resample rate must be positive"),
    ({"resample": -250.0}, "resample rate must be positive"),
])
def test_settings_that_would_produce_numbers_not_measurements_are_refused(
        recording, changes, message):
    """Every one of these runs fine and returns a plausible-looking answer.

    That is exactly why they are refused here rather than left to whoever reads
    the result.
    """
    with pytest.raises(ValueError, match=message):
        prepare(recording, replace(PreprocessConfig(), **changes), verbose=False)


def test_excluding_every_channel_is_refused(recording):
    with pytest.raises(ValueError, match="nothing to analyse"):
        prepare(recording, replace(PreprocessConfig(),
                                   exclude=tuple(recording.raw.ch_names)),
                verbose=False)
