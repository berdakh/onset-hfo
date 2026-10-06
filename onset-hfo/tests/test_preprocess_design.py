"""Filter design, references and artifact annotation in `prepare` (Qt-free)."""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.config import PreprocessConfig, QualityConfig
from onset_hfo.preprocess import (
    REFERENCES,
    describe,
    effective_reference,
    filter_description,
    learn_ptp_threshold,
    prepare,
)
from onset_hfo.quality import REASONS, segment_quality


def test_the_filter_description_is_read_off_mnes_design(recording):
    sfreq = float(recording.raw.info["sfreq"])
    fir = filter_description(PreprocessConfig(), sfreq)
    assert "FIR" in fir and "taps" in fir and "zero phase" in fir
    narrow = filter_description(PreprocessConfig(transition_bandwidth=0.25), sfreq)
    wide = filter_description(PreprocessConfig(transition_bandwidth=0.5), sfreq)
    taps = lambda text: int(text.split(" of ")[1].split(" taps")[0])  # noqa: E731
    assert taps(narrow) > taps(wide), "a narrower transition is a longer filter"
    # A transition band as wide as the cut-off has no stop band: refused, and
    # said in the warnings before it can be applied.
    with pytest.raises(ValueError):
        prepare(recording, PreprocessConfig(transition_bandwidth=2.0), verbose=False)
    _, warnings = describe(PreprocessConfig(transition_bandwidth=2.0), (80.0, 250.0), sfreq)
    assert any("below 0 Hz" in w for w in warnings)
    iir = filter_description(PreprocessConfig(filter_method="iir", iir_order=3), sfreq)
    assert "Butterworth" in iir and "order 3" in iir and "effective order 6" in iir
    assert filter_description(PreprocessConfig(highpass=0.0), sfreq) == "no high- or low-pass"


def test_an_iir_design_runs_and_is_written_into_the_steps(recording):
    prep = prepare(recording, PreprocessConfig(filter_method="iir", iir_order=4), verbose=False)
    assert any("Butterworth" in step for step in prep.steps)
    default = prepare(recording, verbose=False)
    assert prep.data.shape == default.data.shape
    assert not np.allclose(prep.data, default.data), "a different filter, a different signal"
    with pytest.raises(ValueError):
        prepare(recording, PreprocessConfig(filter_method="wavelet"), verbose=False)
    with pytest.raises(ValueError):
        prepare(recording, PreprocessConfig(filter_phase="backwards"), verbose=False)


def test_every_reference_scheme_has_a_sentence_and_runs(recording):
    assert effective_reference(PreprocessConfig()) == "bipolar"
    assert effective_reference(PreprocessConfig(bipolar=False, average_reference=True)) == "average"
    assert effective_reference(PreprocessConfig(bipolar=False)) == "none"
    said = {"bipolar": "bipolar montage", "average": "common average",
            "median": "common median", "shaft": "per-shaft average", "none": None}
    for scheme in REFERENCES:
        prep = prepare(recording, PreprocessConfig(reference=scheme), verbose=False)
        if said[scheme]:
            assert any(said[scheme] in step for step in prep.steps), scheme
        assert prep.montage == ("bipolar" if scheme == "bipolar" else
                                "monopolar" if scheme == "none" else scheme)
    with pytest.raises(ValueError):
        effective_reference(PreprocessConfig(reference="laplace"))


def test_the_median_and_shaft_references_remove_what_they_say(recording):
    plain = prepare(recording, PreprocessConfig(reference="none"), verbose=False)
    median = prepare(recording, PreprocessConfig(reference="median"), verbose=False)
    assert np.allclose(np.median(median.data, axis=0), 0.0, atol=1e-6)
    shaft = prepare(recording, PreprocessConfig(reference="shaft"), verbose=False)
    from onset_hfo.preprocess import _shaft_of

    for name in set(_shaft_of(n) for n in shaft.ch_names):
        members = [i for i, n in enumerate(shaft.ch_names) if _shaft_of(n) == name]
        if len(members) > 1:
            assert np.allclose(shaft.data[members].mean(axis=0), 0.0, atol=1e-6), name
    assert plain.ch_names == median.ch_names == shaft.ch_names


def test_the_learned_ceiling_keeps_the_typical_epoch_and_rejects_the_loud_one():
    rng = np.random.default_rng(3)
    quiet = rng.normal(0.0, 10.0, size=(40, 500))
    loud = quiet.copy()
    loud[::8] *= 25.0                        # every eighth epoch is a pop
    ceiling = learn_ptp_threshold(loud)
    ptp = loud.max(axis=1) - loud.min(axis=1)
    assert ptp[1] < ceiling < ptp[0], "between the typical epoch and the loud one"
    assert learn_ptp_threshold(quiet[:4]) == float("inf"), "too few epochs: no ceiling"


def test_amplitude_annotation_marks_seconds_that_the_quality_stage_sets_aside(recording):
    cfg = PreprocessConfig(annotate_amplitude=True, amplitude_ptp_uv=1.0)
    prep = prepare(recording, cfg, verbose=False)
    assert prep.annotations and all(a["reason"] == "annotated_amplitude" for a in prep.annotations)
    assert all(a["channel"] in prep.ch_names for a in prep.annotations), "pairs, not contacts"
    assert any("set aside by the quality stage" in step for step in prep.steps)
    segments = segment_quality(prep, QualityConfig())
    marked = segments[segments["reason"] == "annotated_amplitude"]
    assert len(marked) > 0 and not marked["good"].any()
    # Without an annotation the same seconds are fine.
    plain = segment_quality(prepare(recording, verbose=False), QualityConfig())
    assert "annotated_amplitude" not in set(plain["reason"])
    assert REASONS["annotated_amplitude"] and REASONS["annotated_muscle"]


def test_muscle_annotation_runs_through_mne_and_reports_in_the_steps(recording):
    prep = prepare(recording, PreprocessConfig(annotate_muscle=True, muscle_z=2.0), verbose=False)
    assert any("muscle or movement" in step for step in prep.steps)
    for note in prep.annotations:
        assert note["channel"] is None and note["reason"] == "annotated_muscle"
        assert note["t_stop"] > note["t_start"]
    if prep.annotations:
        segments = segment_quality(prep, QualityConfig())
        assert (segments["reason"] == "annotated_muscle").any()


def test_describe_names_the_design_the_reference_and_the_annotators(recording):
    sfreq = float(recording.raw.info["sfreq"])
    band = (80.0, 250.0)
    text, warnings = describe(PreprocessConfig(filter_method="iir", iir_order=10,
                                               reference="median", annotate_muscle=True,
                                               annotate_amplitude=True), band, sfreq)
    assert "Butterworth" in text and "common median" in text
    assert "mark muscle" in text and "learned per contact" in text
    joined = " ".join(warnings)
    assert "rings hard" in joined and "common reference" in joined
    assert "ripple band" in joined and "discharges" in joined
    assert "Check data quality" in joined
    _, quiet = describe(PreprocessConfig(), band, sfreq)
    assert not any("rings" in w or "muscle" in w for w in quiet)
