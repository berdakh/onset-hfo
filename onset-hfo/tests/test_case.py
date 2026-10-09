"""A case: from a clinical file to a patient's recordings, step by step.

Asked for: the clinic's recordings in one structure, whatever system made
them, then annotated, the start of an end-to-end clinical workflow. What has
to hold:

* the format is BIDS-iEEG, written in chunks, and reads back sample for
  sample with types, bad contacts, mains and marks;
* a clinical file is converted through its bridge: checksummed, refused if
  converted already, channels typed as the reader says, marks mapped with
  their text kept, and nothing that names the patient written anywhere;
* every change to a case is in its log;
* the overview samples, says so, and is cached;
* in the window: a file converts, channels retype, marks are added, drawn in
  MNE's browser and brought back, and a minute opens in the review window.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess

import mne
import numpy as np
import pandas as pd
import pytest

from onset_hfo.case import annotations as marks
from onset_hfo.case import bids, bridges, overview
from onset_hfo.case.model import Case


@pytest.fixture
def clinical_edf(recording, tmp_path):
    """The synthetic recording as a clinical system exports it: every channel
    'eeg', a patient's name, a date, and free-text marks."""
    raw = recording.raw.copy()
    raw.set_channel_types({n: "eeg" for n in raw.ch_names})
    raw.set_meas_date(datetime.datetime(2025, 3, 4, 22, 15, 0, tzinfo=datetime.timezone.utc))
    raw.info["subject_info"] = {"his_id": "MRN123456", "first_name": "Jane",
                                "last_name": "Doe", "birthday": datetime.date(1980, 1, 2)}
    raw.set_annotations(mne.Annotations(
        [2.0, 10.0, 12.5, 20.0], [0.0, 5.0, 0.0, 0.0],
        ["Sz onset", "seizure", "pt pushed button", "called family"],
        orig_time=raw.info["meas_date"]))
    path = tmp_path / "clinic.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


# -- the format ---------------------------------------------------------------------------
def test_a_run_reads_back_sample_for_sample(recording, tmp_path, monkeypatch):
    monkeypatch.setattr(bids, "_CHUNK_SAMPLES", 7_001)       # many chunks, a ragged last one
    raw = recording.raw.copy()
    where = bids.BidsRun(tmp_path, "P01")
    channels = pd.DataFrame([{"name": "SA1", "type": "ecog", "status": "bad",
                              "status_description": "noisy"}])
    events = pd.DataFrame([marks.new_mark(5.0, "seizure", 2.5, ("SA2", "SA3"), "Sz 1")])
    write = bids.write_run(where, raw, channels=channels, events=events, line_freq=60)
    back = bids.read_run(write, preload=True)
    assert back.n_times == raw.n_times and back.info["sfreq"] == raw.info["sfreq"]
    assert np.abs(back.get_data() - raw.get_data()).max() < 1e-9      # float32 in µV
    assert back.get_channel_types()[0] == "ecog" and back.info["bads"] == ["SA1"]
    assert back.info["line_freq"] == 60.0
    assert list(back.annotations.description) == ["seizure"]
    assert back.annotations.ch_names[0] == ("SA2", "SA3")
    sidecar = json.loads(where.path("ieeg.json").read_text())
    assert sidecar["SamplingFrequency"] == raw.info["sfreq"]
    assert sidecar["RecordingDuration"] == pytest.approx(raw.n_times / raw.info["sfreq"])
    table = bids.read_channels(where)
    assert list(table.columns) == list(bids.CHANNEL_COLUMNS)
    assert table.loc[0, "status_description"] == "noisy"


def test_marks_map_to_the_vocabulary_and_keep_their_text():
    assert marks.kind_of("Sz onset") == "seizure-onset"
    assert marks.kind_of("clinical onset - head turn") == "seizure-clinical-onset"
    assert marks.kind_of("SZ END") == "seizure-offset"
    assert marks.kind_of("seizure 3") == "seizure"
    assert marks.kind_of("artifact chewing") == "artefact"
    assert marks.kind_of("N2") == "sleep-N2" and marks.kind_of("REM") == "sleep-REM"
    assert marks.kind_of("lights on") == "sleep-W"
    assert marks.kind_of("called family") == "note"
    assert marks.kind_of("sleep-N3") == "sleep-N3"
    with pytest.raises(ValueError):
        marks.new_mark(1.0, "sneeze")
    with pytest.raises(ValueError):
        marks.new_mark(-1.0, "note")


# -- converting --------------------------------------------------------------------------
def test_a_clinical_file_converts_without_the_patient_s_name(clinical_edf, tmp_path):
    info = bridges.inspect(clinical_edf)
    assert info["format"] == "European Data Format" and info["start_time_of_day"] == "22:15:00"
    assert set(info["identifying"].values()) >= {"first name", "last name",
                                                  "patient identifier", "recording date"}
    case = Case.create(tmp_path / "case", "P017")
    types = {n: "seeg" for n in info["channels"]["name"]}
    report = bridges.convert(clinical_edf, case, channel_types=types, bad=["SB3"],
                             line_freq=60)
    assert report.run == "01" and report.types == {"SEEG": 18} and len(report.retyped) == 18
    assert report.marks == {"seizure-onset": 1, "seizure": 1, "button": 1, "note": 1}
    assert report.free_text == 1 and report.bad == ["SB3"]
    assert "first name" in report.left_out and "recording date" in report.left_out
    events = case.marks("01")
    assert list(events["trial_type"]) == ["seizure-onset", "seizure", "button", "note"]
    assert list(events["value"]) == ["Sz onset", "seizure", "pt pushed button", "called family"]
    assert events.loc[1, "duration"] == pytest.approx(5.0)
    # Nothing in the case names the patient or dates the recording.
    found = subprocess.run(["grep", "-rIl", "-e", "Jane", "-e", "Doe", "-e", "MRN123456",
                            "-e", "2025-03-04", "-e", "1980", str(case.root)],
                           capture_output=True, text=True).stdout
    assert found == ""
    again = Case.open(case.root)
    assert again.recordings[0].source_sha256 == report.sha256
    assert again.step_done("import")
    assert [e["action"] for e in again.log][:3] == ["created the case", "added a recording",
                                                    "completed step import"]
    with pytest.raises(ValueError, match="in the case already"):
        bridges.convert(clinical_edf, again)
    assert (case.derivatives / "conversion").is_dir()
    md = next((case.derivatives / "conversion").glob("*_conversion.md")).read_text()
    assert "Left out (identifying):" in md and "22:15:00" in md


def test_a_file_no_bridge_reads_says_what_to_do(tmp_path):
    for name, hint in (("study.erd", "EDF+"), ("x.TRC", "neo"), ("notes.docx", "EDF+")):
        path = tmp_path / name
        path.write_bytes(b"0")
        with pytest.raises(ValueError, match=hint):
            bridges.bridge_for(path)


def test_channel_and_mark_changes_are_logged(clinical_edf, tmp_path):
    case = Case.create(tmp_path / "case", "P018")
    bridges.convert(clinical_edf, case, channel_types={})
    table = case.channels("01")
    table.loc[table["name"] == "SA1", "type"] = "SEEG"
    table.loc[table["name"] == "SA2", "status"] = "bad"
    table.loc[table["name"] == "SA2", "status_description"] = "broken lead"
    case.save_channels("01", table, by="dr test")
    log = Case.open(case.root).log[-1]
    assert log["by"] == "dr test" and "SA1 type EEG → SEEG" in log["detail"]
    assert "SA2 status good → bad" in log["detail"]
    events = case.marks("01")
    events = pd.concat([events, pd.DataFrame([marks.new_mark(25.0, "artefact", 2.0)])],
                       ignore_index=True)
    case.save_marks("01", events.iloc[1:], by="dr test")
    assert Case.open(case.root).log[-1]["detail"] == "run 01: 1 added, 1 removed"
    with pytest.raises(FileExistsError):
        Case.create(case.root, "P018")
    with pytest.raises(ValueError):
        case.mark_step("dance")


def test_the_overview_samples_says_so_and_is_cached(clinical_edf, tmp_path, monkeypatch):
    case = Case.create(tmp_path / "case", "P019")
    bridges.convert(clinical_edf, case, channel_types={})
    calls = []
    real = overview.envelope
    monkeypatch.setattr(overview, "envelope",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    rec = case.recordings[0]
    frame = overview.cached_envelope(case, rec, block_s=5.0, sample_s=1.0)
    assert list(frame.columns) == list(overview.COLUMNS) and len(frame) == 6
    assert frame.attrs["coverage"] == pytest.approx(0.2) and (frame["rms_uv"] > 0).all()
    again = overview.cached_envelope(case, rec, block_s=5.0, sample_s=1.0)
    assert len(calls) == 1 and again.attrs["coverage"] == pytest.approx(0.2)
    # A contact marked bad changes what is drawn: drawn again.
    table = case.channels(rec)
    table.loc[0, "status"] = "bad"
    case.save_channels(rec, table)
    overview.cached_envelope(case, rec, block_s=5.0, sample_s=1.0)
    assert len(calls) == 2
    assert overview.cached_envelope(case, rec, block_s=10.0, compute=False) is None


# -- the window ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def _settle(app, n=10):
    from qtpy.QtCore import QThread

    for _ in range(n):
        app.processEvents()
        QThread.msleep(20)


def test_the_case_window_converts_types_marks_and_analyses(qapp, clinical_edf, tmp_path):
    from onset_review.casewindow import CaseWindow, ConvertDialog

    case = Case.create(tmp_path / "case", "P020")
    window = CaseWindow(case, reader="dr test")
    try:
        assert window.current_step() == "import"
        assert not window.show_step("report"), "a later phase's step is not open yet"
        dialog = ConvertDialog(clinical_edf, window)
        assert dialog.table.rowCount() == 18 and dialog.marks_table.rowCount() == 4
        dialog.set_brain_type("ecog")
        dialog.bad["SC6"].setChecked(True)
        window.import_page.convert(clinical_edf, dialog.channel_types(),
                                   dialog.bad_channels(), 60.0, wait=True)
        _settle(qapp)
        assert window.import_page.table.rowCount() == 1
        assert "Left out (identifying)" in window.import_page.report.toPlainText()
        assert window.steps.item(0).text().startswith("✓")

        assert window.show_step("channels")
        page = window.channels_page
        assert page.table.rowCount() == 18
        assert page.table.cellWidget(0, 1).currentText() == "ecog"
        assert page.table.cellWidget(17, 2).isChecked(), "SC6 marked bad at conversion"
        page.table.cellWidget(0, 2).setChecked(True)
        page.table.setItem(0, 3, page.table.item(0, 0).clone())
        page.table.item(0, 3).setText("flat")
        page.save()
        assert case.channels("01").loc[0, "status_description"] == "flat"
        page.done_button.click()
        assert case.step_done("channels")

        assert window.show_step("annotate")
        annotate = window.annotate_page
        assert annotate.marks_table.rowCount() == 4
        annotate.draw_overview(wait=True)
        _settle(qapp)
        assert annotate._overview is not None and "Sampled" in annotate.overview_note.text()
        annotate.time.setValue(15.0)
        assert "22:15" in annotate.time_label.text()
        annotate.add_mark("artefact", duration=3.0, note="cable")
        assert annotate.marks_table.rowCount() == 5
        added = case.marks("01").set_index("trial_type").loc["artefact"]
        assert (added["by"], added["value"], added["onset"]) == ("dr test", "cable", 15.0)

        # Drawn in MNE's browser, brought back on close.
        browser = annotate.open_trace(show=False)
        assert "seizure-onset" in browser.mne.new_annotation_labels
        raw = annotate._trace_raw
        raw.annotations.append(18.0, 1.0, "sleep-N2")
        raw.annotations.delete([list(raw.annotations.description).index("button")])
        assert annotate.take_marks_from_trace() == 5
        kinds = list(case.marks("01")["trial_type"])
        assert "sleep-N2" in kinds and "button" not in kinds and "artefact" in kinds
        kept = case.marks("01").set_index("trial_type")
        assert kept.loc["seizure", "value"] == "seizure", "an unmoved mark keeps its text"
        browser.close()

        emitted = []
        window.openRequested.connect(emitted.append)
        annotate.time.setValue(5.0)
        request = annotate.open_review()
        assert emitted == [request]
        assert request.path == case.where("01").header and request.line_freq == 60.0
        assert dict(request.channel_types)["SA1"] == "ecog"
        assert set(request.preprocess.exclude) == {"SA1", "SC6"}
        assert (request.t_start, request.t_stop) == (0.0, pytest.approx(30.0))
        from onset_review.session import load_session

        session = load_session(request)
        assert len(session.findings) and session.request.path == request.path
        annotate.done_button.click()
        assert case.step_done("annotate")
    finally:
        window.close()


def test_the_terminal_makes_converts_and_lists(clinical_edf, tmp_path, capsys):
    from onset_hfo.case.__main__ import main

    folder = tmp_path / "case"
    assert main(["new", str(folder), "P021"]) == 0
    missing = tmp_path / "missing.edf"
    code = main(["convert", str(folder), str(clinical_edf), str(missing), "--all-as", "ecog",
                 "--mains", "60", "--bad", "SA1"])
    out = capsys.readouterr()
    assert code == 1 and "missing.edf: not converted" in out.err
    assert "run 01, 18 ch at 2000 Hz" in out.out and "first name" in out.out
    case = Case.open(folder)
    channels = case.channels("01")
    assert set(channels["type"]) == {"ECOG"}
    assert channels.set_index("name").loc["SA1", "status"] == "bad"
    assert main(["list", str(folder)]) == 0
    assert "run 01: clinic.edf" in capsys.readouterr().out
