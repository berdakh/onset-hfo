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


# --------------------------------------------------------------------------
# Phase 3: reference regression and ICA, experimental
# --------------------------------------------------------------------------

def _with_ecg(recording):
    """The synthetic recording with an ECG lead mixed into every contact at a
    known weight, and the lead itself as a channel of type ecg."""
    import dataclasses

    import mne

    raw = recording.raw.copy()
    sfreq = raw.info["sfreq"]
    t = raw.times
    ecg = 50e-6 * np.sin(2 * np.pi * 1.2 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
    data = raw.get_data()
    weights = np.linspace(0.2, 1.0, data.shape[0])
    data = data + weights[:, None] * ecg[None, :]
    info = mne.create_info(raw.ch_names + ["ECG"], sfreq,
                           ch_types=["seeg"] * len(raw.ch_names) + ["ecg"], verbose="ERROR")
    mixed = mne.io.RawArray(np.vstack([data, ecg[None, :]]), info, verbose="ERROR")
    return dataclasses.replace(recording, raw=mixed), ecg


def test_regressing_an_ecg_lead_removes_what_it_put_in(recording):
    import mne

    from onset_hfo.preprocess import prepare

    mixed, ecg = _with_ecg(recording)
    plain = prepare(mixed, PreprocessConfig(bipolar=False, reference="none"), verbose=False)
    regressed = prepare(mixed, PreprocessConfig(bipolar=False, reference="none",
                                                regress_channels=("ECG",)), verbose=False)
    assert "ECG" not in regressed.ch_names and regressed.ch_names == plain.ch_names
    lead = mne.filter.filter_data(ecg[None, :], mixed.raw.info["sfreq"], 1.0, None,
                                  verbose="ERROR")[0]
    before = np.mean([abs(np.corrcoef(plain.data[i], lead)[0, 1])
                      for i in range(plain.data.shape[0])])
    after = np.mean([abs(np.corrcoef(regressed.data[i], lead)[0, 1])
                     for i in range(regressed.data.shape[0])])
    assert before > 0.3 and after < before / 5
    assert any("regressed ECG out of" in step and "not analysed" in step
               for step in regressed.steps)
    assert any("kept ECG to regress out" in step for step in regressed.steps)
    with pytest.raises(ValueError, match="no channel NOPE"):
        prepare(mixed, PreprocessConfig(regress_channels=("NOPE",)), verbose=False)


def test_ica_is_fitted_scored_and_applied_only_as_chosen(recording):
    import dataclasses

    from onset_hfo.preprocess import prepare

    mixed, _ecg = _with_ecg(recording)
    cfg = PreprocessConfig(ica=True, ica_n_components=6)
    fitted = prepare(mixed, cfg, verbose=False)
    record = fitted.ica
    assert record is not None and record["n_components"] == 6
    assert record["sources"].shape[0] == 6 and record["loadings"].shape[1] == 6
    # The synthetic recording carries no electrode positions, so MNE's muscle
    # score is not run rather than reported as a row of zeros.
    assert record["muscle_scored"] is False and record["muscle_scores"] == []
    assert any("Not scored for muscle" in note for note in record["notes"])
    assert any("not scored for muscle (no electrode positions)" in s for s in fitted.steps)
    assert len(record["hf_share"]) == 6
    assert record["ecg_channel"] == "ECG" and record["excluded"] == []
    assert abs(sum(record["variance_share"]) - 1.0) < 0.5
    assert any("ICA (fastica, 6 components" in s and "nothing removed" in s
               and "experimental" in s for s in fitted.steps)
    # The same seed finds the same components.
    again = prepare(mixed, cfg, verbose=False)
    assert np.allclose(again.ica["sources"][:2], record["sources"][:2])
    assert np.allclose(again.data, fitted.data), "nothing removed means nothing changed"
    # Removing a component changes the data; an index that does not exist is said, not applied.
    removed = prepare(mixed, dataclasses.replace(cfg, ica_exclude=(0, 99)), verbose=False)
    assert removed.ica["excluded"] == [0]
    assert not np.allclose(removed.data, fitted.data)
    assert any("99" in note and "ignored" in note for note in removed.ica["notes"])
    assert any("removed 0 (the reviewer's choice)" in s for s in removed.steps)
    # Off by default, and absent from the record when off.
    assert prepare(mixed, PreprocessConfig(), verbose=False).ica is None


def test_ica_settings_are_checked_and_described():
    from onset_hfo.preprocess import ICA_METHODS, _check, describe, ica_methods_available

    assert "fastica" in ica_methods_available() and set(ica_methods_available()) <= set(ICA_METHODS)
    with pytest.raises(ValueError, match="ica_method"):
        _check(PreprocessConfig(ica=True, ica_method="magic"), 2000.0)
    with pytest.raises(ValueError, match="at least 1"):
        _check(PreprocessConfig(ica=True, ica_n_components=0), 2000.0)
    with pytest.raises(ValueError, match="non-negative"):
        _check(PreprocessConfig(ica=True, ica_exclude=(-1,)), 2000.0)
    _check(PreprocessConfig(ica=False, ica_method="magic"), 2000.0)   # off: not checked
    text, warnings = describe(PreprocessConfig(ica=True, ica_exclude=(2, 5),
                                               regress_channels=("ECG",)), (80, 250), 2000.0)
    assert "regress ECG out of every brain channel" in text
    assert "fit ICA (fastica, up to 20 components) and remove component(s) 2, 5" in text
    assert any("experimental" in w and "Nothing is removed until you choose" in w
               for w in warnings)
    assert any("least squares" in w for w in warnings)
    text, _ = describe(PreprocessConfig(ica=True, ica_n_components=8), (80, 250), 2000.0)
    assert "8 components), removing nothing until you choose" in text


def test_a_solver_whose_package_is_missing_is_not_offered_and_is_named_when_asked_for(
        monkeypatch):
    """Seen on a fresh install: the panel offered fastica, scikit-learn was not
    there, and Apply failed in the window with scikit-learn's ImportError.
    The list now says what runs here, the default follows it, and asking for
    an absent solver names the package."""
    from onset_hfo import preprocess
    from onset_hfo.preprocess import _check, default_ica_method, ica_methods_available

    monkeypatch.setattr(preprocess, "_importable", lambda module: module == "mne")
    assert ica_methods_available() == ("infomax",)
    assert default_ica_method() == "infomax"
    with pytest.raises(ValueError, match="scikit-learn"):
        _check(PreprocessConfig(ica=True, ica_method="fastica"), 2000.0)
    _check(PreprocessConfig(ica=True, ica_method="infomax"), 2000.0)
    monkeypatch.setattr(preprocess, "_importable", lambda module: True)
    assert ica_methods_available() == ("fastica", "infomax", "picard")
    assert default_ica_method() == "fastica"


def test_ica_scores_muscle_only_with_electrode_positions(recording):
    import dataclasses

    import mne

    from onset_hfo.preprocess import _has_positions, prepare

    assert not _has_positions(recording.raw)
    raw = recording.raw.copy()
    rng = np.random.default_rng(3)
    positions = {name: rng.normal(scale=0.04, size=3) for name in raw.ch_names}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head"),
                    on_missing="ignore", verbose="ERROR")
    assert _has_positions(raw)
    placed = dataclasses.replace(recording, raw=raw)
    record = prepare(placed, PreprocessConfig(ica=True, ica_n_components=4), verbose=False).ica
    if record["muscle_scored"]:
        assert len(record["muscle_scores"]) == 4
    else:
        assert any("did not run" in note for note in record["notes"])
