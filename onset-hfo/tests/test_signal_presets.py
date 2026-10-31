"""The Signal page asks how to read the recording before it asks for filters.

The redesign's third step. What has to hold:

* a preset sets the signal chain and nothing else: the channels left out, the
  sampling rate, the mains and the kind of recording are kept; a config that
  is exactly a preset is recognised as it, and anything else is Custom;
* on the page, Standard and Noisy fold the filter, reference, artefact and
  experimental settings away and Custom shows them; Reset goes back to
  Standard;
* the preview filters the stretch it shows with the page's own filters, from
  the recording as read, and without a start-up transient at its edges;
* the import dialog keeps its channel table folded when the file states its
  channel types, and open when it does not or when the reader picks another
  kind.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from onset_hfo.config import PreprocessConfig
from onset_review.filterpreview import SECONDS, Preview, apply_filters, preview_signal
from onset_review.session import ReviewRequest, session_from_recording

# The presets live beside the Signal page's widgets, so they need Qt; the
# preview's signal and filters do not, and run without it.
try:
    from onset_review.preprocessing import PRESETS, matching_preset, preset_config
except ImportError:            # the review extra is not installed
    PRESETS = None
needs_qt = pytest.mark.skipif(PRESETS is None, reason="the review extra is not installed")


# -- the presets as configs --------------------------------------------------------------
@needs_qt
def test_a_preset_sets_the_chain_and_keeps_what_it_does_not_decide():
    base = PreprocessConfig(highpass=80.0, reference="average", exclude=("SA1",),
                            resample=1000.0, line_freq=60.0, ica=True)
    standard = preset_config("standard", base)
    assert standard.highpass == PreprocessConfig().highpass
    assert standard.reference == PreprocessConfig().reference and not standard.ica
    assert (standard.exclude, standard.resample, standard.line_freq) == (
        ("SA1",), 1000.0, 60.0)
    noisy = preset_config("noisy", base)
    assert noisy.annotate_muscle and noisy.annotate_amplitude
    assert dataclasses.replace(noisy, annotate_muscle=False,
                               annotate_amplitude=False) == standard
    assert preset_config("custom", base) is base


@needs_qt
def test_a_config_is_named_by_the_preset_it_is_exactly():
    assert [key for key, _label, _what in PRESETS] == ["standard", "noisy", "custom"]
    assert matching_preset(PreprocessConfig()) == "standard"
    assert matching_preset(PreprocessConfig(exclude=("SA1",), line_freq=50.0)) == "standard"
    assert matching_preset(PreprocessConfig(annotate_muscle=True,
                                            annotate_amplitude=True)) == "noisy"
    assert matching_preset(PreprocessConfig(annotate_muscle=True)) == "custom"
    assert matching_preset(PreprocessConfig(highpass=80.0)) == "custom"


# -- the preview, without Qt -------------------------------------------------------------
def _request(recording, preprocess=None):
    return ReviewRequest(dataset="", subject="synthetic", t_start=0.0,
                         t_stop=float(recording.duration), detectors=("rms",),
                         preprocess=preprocess)


@pytest.fixture(scope="module")
def window(recording):
    return session_from_recording(recording, _request(recording))


def test_the_preview_is_the_busiest_channel_from_the_recording_as_read(window, recording):
    preview = preview_signal(window)
    assert preview is not None
    assert preview.label == window.leader.get("leader", window.findings["channel"].iloc[0])
    assert len(preview.times) == int(round(SECONDS * preview.sfreq))
    assert preview.line_freq == recording.line_freq
    first, second = preview.label.split("-", 1)
    raw = recording.raw
    start = int(round((preview.t0 - getattr(window, "t_offset", 0.0)) * preview.sfreq))
    stop = start + len(preview.times)
    expected = (raw.get_data(picks=[first])[0] - raw.get_data(picks=[second])[0])[start:stop]
    np.testing.assert_allclose(preview.signal[preview.shown], expected)


def test_the_preview_filters_with_the_page_s_settings(window):
    preview = preview_signal(window)
    before = preview.signal[preview.shown]
    after, done = apply_filters(preview, PreprocessConfig())
    assert done == ["high-pass 1 Hz", f"notch at {preview.line_freq:g} Hz and its harmonics"]
    assert after.shape == before.shape
    assert not np.allclose(after, before), "the filters changed the trace"
    nothing, said = apply_filters(preview, PreprocessConfig(highpass=0.0, notch=False))
    np.testing.assert_allclose(nothing, before)
    assert said == []
    ripples, said = apply_filters(preview, PreprocessConfig(highpass=80.0, lowpass=250.0))
    assert said[0] == "high-pass 80 Hz and low-pass 250 Hz"
    assert np.std(ripples) < np.std(before)


def _notch_specifications(monkeypatch, sfreq, mains, cfg):
    """Capture what each real filtering path asks MNE to notch."""
    import mne

    from onset_hfo.preprocess import prepare
    from onset_hfo.synthetic import make_synthetic_recording

    recording = make_synthetic_recording(
        duration_s=8, sfreq=sfreq, line_freq=mains,
        n_leads=1, contacts_per_lead=4, verbose=False)
    analysis, preview = [], []
    raw_notch = mne.io.BaseRaw.notch_filter
    array_notch = mne.filter.notch_filter

    def capture_analysis(raw, *args, **kwargs):
        analysis.append((tuple(kwargs["freqs"]), kwargs["notch_widths"]))
        return raw_notch(raw, *args, **kwargs)

    def capture_preview(data, sfreq, freqs, **kwargs):
        preview.append((tuple(freqs), kwargs["notch_widths"]))
        return array_notch(data, sfreq, freqs, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mne.io.BaseRaw, "notch_filter", capture_analysis)
        prepare(recording, cfg, verbose=False)
    source = Preview(label=recording.raw.ch_names[0], t0=0.0, sfreq=sfreq,
                     signal=recording.raw.get_data()[0], line_freq=mains)
    with monkeypatch.context() as patch:
        patch.setattr(mne.filter, "notch_filter", capture_preview)
        apply_filters(source, cfg)
    return analysis, preview


@pytest.mark.parametrize("sfreq,mains,expected", [
    (1000.0, 50.0, tuple(range(50, 401, 50))),
    (1000.0, 60.0, tuple(range(60, 421, 60))),
    (2000.0, 50.0, tuple(range(50, 851, 50))),
    (2000.0, 60.0, tuple(range(60, 841, 60))),
])
@pytest.mark.parametrize("harmonics", [True, False])
def test_the_preview_and_analysis_notch_the_same_frequencies(
        monkeypatch, sfreq, mains, expected, harmonics):
    cfg = PreprocessConfig(notch_harmonics=harmonics)
    analysis, preview = _notch_specifications(monkeypatch, sfreq, mains, cfg)
    wanted = expected if harmonics else (mains,)
    assert analysis == preview == [(wanted, 2.0)]


@pytest.mark.parametrize("recorded,override,expected", [
    (50.0, 60.0, tuple(range(60, 421, 60))),
    (60.0, 50.0, tuple(range(50, 401, 50))),
])
def test_the_preview_and_analysis_use_the_mains_override_and_notch_width(
        monkeypatch, recorded, override, expected):
    cfg = PreprocessConfig(line_freq=override, notch_width=6.0)
    analysis, preview = _notch_specifications(monkeypatch, 1000.0, recorded, cfg)
    assert analysis == preview == [(expected, 6.0)]


def test_the_preview_and_analysis_do_not_notch_when_disabled(monkeypatch):
    cfg = PreprocessConfig(notch=False)
    analysis, preview = _notch_specifications(monkeypatch, 1000.0, 60.0, cfg)
    assert analysis == preview == []


@pytest.mark.parametrize("mains", [450.0, 480.0])
def test_the_preview_and_analysis_leave_fundamentals_at_the_cutoff_or_above(
        monkeypatch, mains):
    cfg = PreprocessConfig(line_freq=mains, notch_harmonics=False)
    analysis, preview = _notch_specifications(monkeypatch, 1000.0, 50.0, cfg)
    assert analysis == preview == []


def test_the_preview_has_no_filter_start_up_at_its_edges():
    sfreq = 1000.0
    t = np.arange(int((SECONDS + 6) * sfreq)) / sfreq
    signal = 1e-4 + 20e-6 * np.sin(2 * np.pi * 10 * t)        # an offset under a 10 Hz wave
    preview = Preview(label="X", t0=0.0, sfreq=sfreq, signal=signal, line_freq=None)
    after, _ = apply_filters(preview, PreprocessConfig(notch=False))
    wave = signal[preview.shown] - 1e-4
    assert np.max(np.abs(after - wave)) < 2e-6, "the offset is gone and the wave kept"


def test_without_the_recording_there_is_no_preview():
    class Bare:
        recording = None

    assert preview_signal(Bare()) is None


# -- the page --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_standard_and_noisy_fold_the_settings_and_custom_shows_them(qapp, window):
    from onset_review.preprocessing import PreprocessPanel

    panel = PreprocessPanel(window)
    try:
        assert panel.preset == "standard"
        assert panel.preset_buttons["standard"].isChecked()
        assert all(group.isHidden() for group in panel.custom_groups)
        assert not panel.apply.isEnabled(), "nothing to apply on the window's own settings"
        assert panel.preview.available

        panel.choose_preset("noisy")
        assert panel.config().annotate_muscle and panel.config().annotate_amplitude
        assert panel.apply.isEnabled()
        assert "muscle" in panel.preset_about.text()
        assert all(group.isHidden() for group in panel.custom_groups)

        panel.choose_preset("custom")
        assert not any(group.isHidden() for group in panel.custom_groups)
        assert panel.config().annotate_muscle, "Custom starts from what is set"
        panel.highpass.setValue(80.0)
        panel.preview.show_config(panel.config(), now=True)
        assert "high-pass 80 Hz" in panel.preview.caption.text()
        assert "reference, the artefact marking and ICA" in panel.preview.caption.text()

        panel.reset_to_defaults()
        assert panel.preset == "standard" and panel.config() == preset_config(
            "standard", panel.config())
        assert all(group.isHidden() for group in panel.custom_groups)
    finally:
        panel.deleteLater()


def test_a_window_read_another_way_opens_on_custom(qapp, recording):
    from onset_review.preprocessing import PreprocessPanel

    session = session_from_recording(
        recording, _request(recording, PreprocessConfig(highpass=2.0)))
    panel = PreprocessPanel(session)
    try:
        assert panel.preset == "custom"
        assert not any(group.isHidden() for group in panel.custom_groups)
        assert panel.highpass.value() == 2.0
    finally:
        panel.deleteLater()


# -- the import dialog -----------------------------------------------------------------
def test_the_channel_table_is_folded_when_the_file_states_its_types(qapp, recording,
                                                                   tmp_path):
    from onset_review.importer import ImportDialog

    path = tmp_path / "ieeg_raw.fif"
    recording.raw.save(path, overwrite=True, verbose="ERROR")
    dialog = ImportDialog(path)
    assert dialog.stated
    assert dialog.table.isHidden() and not dialog.channels_toggle.isHidden()
    assert "states its channel types" in dialog.warning.text()
    assert dialog.counts.text().endswith("will be analysed (intracranial)")
    dialog.channels_toggle.setChecked(True)
    assert not dialog.table.isHidden()
    dialog.channels_toggle.setChecked(False)
    dialog.set_kind("eeg")
    assert not dialog.table.isHidden(), "another kind: the channels are shown to check"


def test_the_channel_table_is_open_when_the_file_does_not_say(qapp, recording, tmp_path):
    from onset_hfo.synthetic import as_modality
    from onset_review.importer import ImportDialog

    path = tmp_path / "eeg_raw.fif"
    as_modality(recording, "eeg").raw.save(path, overwrite=True, verbose="ERROR")
    dialog = ImportDialog(path)
    assert not dialog.stated
    assert not dialog.table.isHidden() and dialog.channels_toggle.isHidden()
