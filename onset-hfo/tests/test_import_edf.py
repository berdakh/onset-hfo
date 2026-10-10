"""An EDF the way a clinical system exports it, opened the way a reviewer would.

The archive declares which channels are intracranial. A clinical export
declares every channel `eeg`, the EKG and the DC channel included, and the
first thing a newcomer does with this software is open one. So this is the
whole route on that file: MNE writes a short EDF, the import dialog reads it
through MNE's EDF reader, refuses until something is marked intracranial,
builds the request the reviewer confirmed, and the session, the window and
the report follow from that request with the EKG nowhere in them.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("edfio")      # what mne.export needs to write an EDF; in [dev]
pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")



@pytest.fixture(scope="module")
def qapp():
    import os

    from qtpy.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield QApplication.instance() or QApplication([])


SFREQ = 2000.0
SECONDS = 20.0
NAMES = ["AR1", "AR2", "AR3", "AR4", "HL1", "HL2", "EKG", "DC1"]


@pytest.fixture(scope="module")
def clinical_edf(tmp_path_factory):
    """Eight channels, all declared `eeg`: six depth contacts, an EKG lead
    with a beat every second, a flat DC channel. AR1 carries a 140 Hz burst
    every 1.5 s so the detector has something to find on AR1-AR2."""
    import mne

    rng = np.random.default_rng(3)
    n = int(SFREQ * SECONDS)
    t = np.arange(n) / SFREQ
    data = rng.normal(0.0, 8e-6, size=(len(NAMES), n))
    burst = np.arange(int(0.06 * SFREQ)) / SFREQ
    ripple = 60e-6 * np.sin(2 * np.pi * 140.0 * burst) * np.hanning(len(burst))
    for start in np.arange(0.5, SECONDS - 0.5, 1.5):
        at = int(start * SFREQ)
        data[0, at:at + len(burst)] += ripple
    data[6] = 400e-6 * np.exp(-((t % 1.0 - 0.3) / 0.02) ** 2)     # the EKG
    data[7] = 0.0
    info = mne.create_info(NAMES, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path_factory.mktemp("edf") / "study-007.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


def test_the_edf_is_read_as_the_export_declared_it(qapp, clinical_edf):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(clinical_edf)
    assert dialog.error == ""
    assert dialog.table.rowCount() == len(NAMES)
    assert [dialog.table.item(row, 0).text() for row in range(len(NAMES))] == NAMES
    assert [dialog.table.item(row, 1).text() for row in range(len(NAMES))] == ["eeg"] * 8
    text = " ".join(label.text() for label in dialog.findChildren(qt_label()))
    assert "2000 Hz" in text and "20 s" in text
    assert "Ripples or fast ripples: the band is chosen after opening" in text
    dialog.close()


def test_the_dialog_refuses_until_a_channel_is_intracranial_then_builds_the_request(
        qapp, clinical_edf):
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    dialog = ImportDialog(clinical_edf)
    button = dialog.buttons.button(QDialogButtonBox.Open)
    dialog.set_all("misc")
    assert not button.isEnabled() and "nothing to analyse" in button.toolTip()

    dialog.set_all("seeg")
    for name, kind in (("EKG", "ecg"), ("DC1", "misc")):
        row = NAMES.index(name)
        chooser = dialog.table.cellWidget(row, 2)
        chooser.setCurrentIndex(
            [i for i in range(chooser.count()) if chooser.itemData(i) == kind][0])
    assert button.isEnabled()
    assert dialog.counts.text() == "6 of 8 will be analysed (intracranial)"
    dialog.subject.setText("study-007")
    request = dialog.request()
    dialog.close()

    assert request.imported is True
    assert request.path == clinical_edf
    assert request.subject == "study-007"
    assert request.t_start == 0.0 and request.t_stop == SECONDS
    types = dict(request.channel_types)
    assert types["EKG"] == "ecg" and types["DC1"] == "misc"
    assert all(types[name] == "seeg" for name in NAMES[:6])


def _request(path):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(path)
    dialog.set_all("seeg")
    for name, kind in (("EKG", "ecg"), ("DC1", "misc")):
        chooser = dialog.table.cellWidget(NAMES.index(name), 2)
        chooser.setCurrentIndex(
            [i for i in range(chooser.count()) if chooser.itemData(i) == kind][0])
    dialog.subject.setText("study-007")
    request = dialog.request()
    dialog.close()
    return request


@pytest.fixture(scope="module")
def imported(qapp, clinical_edf):
    from onset_review.session import load_session

    return load_session(_request(clinical_edf))


def test_the_session_analyses_the_confirmed_channels_and_nothing_else(imported):
    session = imported
    assert session.request.imported and session.has_expert is False
    channels = set(session.findings["channel"])
    assert channels == {"AR1-AR2", "AR2-AR3", "AR3-AR4", "HL1-HL2"}, \
        "bipolar pairs of the six contacts; the EKG and the DC channel are gone"
    assert not any("EKG" in c or "DC1" in c for c in channels)
    leader = session.findings.iloc[0]
    assert leader["channel"] == "AR1-AR2", "the planted bursts rank first"
    assert int(leader["n_events"]) >= 5
    assert session.resection is None
    # Every time reported is a time in the file, not in the window.
    assert all(0.0 <= e.start <= e.stop <= SECONDS for e in session.events)


def test_the_window_and_the_report_follow_from_the_imported_session(
        qapp, imported, tmp_path):
    import pandas as pd

    from onset_review import window
    from onset_review.report import write_review

    figure = window.open_trace(imported, show=False)
    try:
        parts = window.decorate(
            figure, imported, mode="pages", cached=lambda: pd.DataFrame(),
            on_open_cached=lambda row: None, on_relayout=lambda name: None,
            on_import=lambda: None, on_window=lambda a, b: None,
            on_step_window=lambda d: None)
        pages = parts.pages
        pages.show_page("recording")
        findings = parts.panels["findings"]
        assert findings.model.rowCount() == 4
        assert findings.model.row_value(0, "channel") == "AR1-AR2"
        assert findings.model.row_value(0, "annotators") == "—", "no archive, no markings"
        assert "study-007" in pages.where.text()
        pages.show_page("patients")
        patient = parts.panels["patient"]
        text = " ".join(label.text() for label in patient.findChildren(qt_label()))
        assert "imported from a file" in text and "study-007.edf" in text
    finally:
        try:
            figure.close()
        except Exception:
            pass

    out = tmp_path / "study-007.md"
    write_review(imported, out)
    report = out.read_text()
    assert "study-007" in report and "AR1-AR2" in report
    assert "EKG" not in report
    assert "expert" not in report.lower().split("limitations")[0] or "no expert" in report.lower()


def qt_label():
    from qtpy.QtWidgets import QLabel

    return QLabel
