"""Opening a file as intracranial EEG, scalp EEG or MEG.

What has to hold: the import dialog asks what the recording is, guessing MEG
when the file declares MEG sensors and intracranial otherwise (a clinical
export declares its contacts `eeg`, so scalp EEG is never the guess); saying
what it is suggests every channel's type again and counts what will actually
be analysed; a MEG sensor can only be re-typed as another MEG sensor or as
not analysed; the request carries the kind, and the window opened from it is
analysed as that kind -- scalp EEG in µV on the double banana, MEG in fT/cm
and never re-referenced; and in an intracranial file, a channel the reviewer
types scalp EEG is left out rather than analysed with the contacts.
"""

from __future__ import annotations

import os

import mne
import numpy as np
import pytest

widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QDialogButtonBox  # noqa: E402

from onset_hfo.synthetic import as_modality, make_synthetic_recording  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield widgets.QApplication.instance() or widgets.QApplication([])


@pytest.fixture(scope="module")
def base():
    return make_synthetic_recording(verbose=False, duration_s=30)


def _save(raw, path):
    raw.save(path, overwrite=True, verbose="ERROR")
    return path


@pytest.fixture(scope="module")
def meg_file(base, tmp_path_factory):
    """18 gradiometers, 3 magnetometers, an EOG and a trigger, as FIF."""
    raw = as_modality(base, "meg_grad").raw.copy()
    sfreq, n = raw.info["sfreq"], raw.n_times
    rng = np.random.default_rng(0)
    extra = mne.io.RawArray(
        np.vstack([rng.normal(size=(3, n)) * 1e-13, rng.normal(size=(1, n)) * 1e-5,
                   np.zeros((1, n))]),
        mne.create_info(["MEG0111", "MEG0121", "MEG0131", "EOG061", "STI101"], sfreq,
                        ["mag", "mag", "mag", "eog", "stim"]), verbose="ERROR")
    raw.add_channels([extra], force_update_info=True)
    return _save(raw, tmp_path_factory.mktemp("meg") / "meg_raw.fif")


@pytest.fixture(scope="module")
def scalp_file(base, tmp_path_factory):
    return _save(as_modality(base, "eeg").raw, tmp_path_factory.mktemp("eeg") / "eeg_raw.fif")


@pytest.fixture(scope="module")
def ieeg_file(base, tmp_path_factory):
    return _save(base.raw, tmp_path_factory.mktemp("ieeg") / "ieeg_raw.fif")


def _dialog(path):
    from onset_review.importer import ImportDialog

    return ImportDialog(path)


def _open(dialog):
    from onset_review.session import load_session

    return load_session(dialog.request())


def test_a_meg_file_is_guessed_meg_and_only_its_gradiometers_are_analysed(qapp, meg_file):
    dialog = _dialog(meg_file)
    assert dialog.kind_name() == "meg"
    assert dialog.counts.text() == "18 of 23 will be analysed (gradiometers)"
    assert "one sensor type at a time" in dialog.warning.text()
    assert dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    sensor = dialog.table.cellWidget(0, 2)
    offered = [sensor.itemData(i) for i in range(sensor.count())]
    assert offered == ["grad", "mag", "ref_meg", "misc"], "a sensor stays a sensor"
    assert not any(b.isVisibleTo(dialog) for b in dialog.all_buttons.values())
    request = dialog.request()
    assert request.modality == "meg"

    session = _open(dialog)
    assert session.montage == "monopolar"
    assert session.steps[0].startswith("analysed as MEG (planar gradiometers), in fT/cm")
    assert any("not re-referenced" in step for step in session.steps)
    assert len(session.findings) == 18


def test_without_gradiometers_the_magnetometers_are_analysed(qapp, meg_file):
    dialog = _dialog(meg_file)
    for row in range(dialog.table.rowCount()):
        if dialog.table.cellWidget(row, 2).currentData() == "grad":
            dialog.table.cellWidget(row, 2).setCurrentIndex(3)         # not analysed
    assert dialog.counts.text() == "3 of 23 will be analysed (magnetometers)"


def test_scalp_eeg_is_said_not_guessed_and_then_read_on_the_double_banana(qapp, scalp_file):
    dialog = _dialog(scalp_file)
    assert dialog.kind_name() == "ieeg", "a clinical export declares its contacts eeg too"
    assert dialog.counts.text() == "18 of 18 will be analysed (intracranial)"
    dialog.set_kind("eeg")
    assert set(dialog.channel_types().values()) == {"eeg"}, "every type suggested again"
    assert dialog.counts.text() == "18 of 18 will be analysed (scalp EEG)"
    assert "validated only there" in dialog.warning.text()
    assert dialog.all_buttons["eeg"].isVisibleTo(dialog)
    assert not dialog.all_buttons["seeg"].isVisibleTo(dialog)
    assert dialog.request().modality == "eeg"

    session = _open(dialog)
    assert session.steps[0].startswith("analysed as scalp EEG, in µV")
    assert "Fp1-F7" in list(session.findings["channel"])


def test_in_an_intracranial_file_a_channel_typed_scalp_eeg_is_left_out(qapp, ieeg_file):
    dialog = _dialog(ieeg_file)
    assert dialog.kind_name() == "ieeg"
    first = dialog.table.item(0, 0).text()
    assert dialog.set_channel(first, "eeg")
    assert dialog.counts.text() == "17 of 18 will be analysed (intracranial)"

    session = _open(dialog)
    assert any(step.startswith(f"left out 1 channels typed scalp EEG ({first})")
               for step in session.steps)
    assert not any(first in str(channel).split("-") for channel in session.findings["channel"])


def test_nothing_of_the_named_kind_offers_nothing_to_open(qapp, meg_file):
    dialog = _dialog(meg_file)
    dialog.set_kind("eeg")
    button = dialog.buttons.button(QDialogButtonBox.Open)
    assert dialog.counts.text() == "0 of 23 will be analysed (scalp EEG)"
    assert not button.isEnabled() and "Nothing is marked scalp EEG" in button.toolTip()


@pytest.fixture(scope="module")
def meg_window(qapp, base):
    """A MEG session laid out as pages, as the app would open it."""
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    from onset_review import window
    from onset_review.session import ReviewRequest, session_from_recording

    session = session_from_recording(as_modality(base, "meg_grad"), ReviewRequest(
        t_start=0.0, t_stop=30.0, modality="meg"))
    figure = window.open_trace(session, show=False)
    parts = window.decorate(figure, session, mode="pages")
    parts.session = session
    yield parts
    try:
        figure.close()
    except Exception:
        pass


def test_a_meg_window_keeps_its_unit_everywhere(meg_window):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel

    from onset_hfo.modality import trace_scale
    from onset_review import detail

    session = meg_window.session
    assert (session.modality, session.unit) == ("meg_grad", "fT/cm")
    assert set(session.raw.get_channel_types()) == {"grad"}, "MNE's browser says fT/cm"
    in_unit = session.raw.get_data() * trace_scale(session.raw)
    assert 100 < float(in_unit.std()) < 2000, "fT/cm, as analysed"
    hfo = next(e for e in session.events if e.detector != "spike")
    snap = detail.snapshot(session, hfo)
    assert 1 < snap.measurements["amplitude_uv"] < 5000
    import pandas as pd

    from onset_review.panels import DataFrameModel

    assert meg_window.panels["events"].model.unit == "fT/cm"
    assert meg_window.panels["findings"].model.unit == "fT/cm"
    shown = DataFrameModel(pd.DataFrame({"amplitude_uv": [1.0]}), unit="fT/cm")
    assert shown.headerData(0, Qt.Horizontal) == "Ampl fT/cm"
    host = meg_window.host
    for page in ("contacts", "map"):
        note = host.findChild(QLabel, f"onset_not_implanted_{page}")
        assert note is not None and "MEG (planar gradiometers)" in note.text(), page
