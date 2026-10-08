"""Many recordings at once, and a whole analysis saved as one file.

Asked for: analyse many recordings in one go and turn them into a cohort;
save an analysis and reopen it later or elsewhere. What has to hold:

* every recording in a batch is analysed with the same settings, a file's
  channels are typed by the one rule the reader stated, and a recording
  that fails is a row with its reason while the batch goes on;
* a batch row joins the cohort measured exactly as the same recording open
  in the window would be;
* a project brings back the request, the verdicts, the scripts, the console's
  variables, the cohort and the page; a recording it does not carry is
  found or named, and a different file is said to be different;
* the window saves a project and reopens it, end to end.
"""

from __future__ import annotations

import dataclasses
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from onset_review import batchreview, project, yourstudy
from onset_review.session import ReviewRequest, session_from_recording


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ONSET_REVIEW_READS", str(tmp_path / "reads"))
    return tmp_path


@pytest.fixture
def analyse(recording):
    """A loader that analyses the synthetic recording under any request."""
    seen = []

    def load(request):
        seen.append(request)
        if request.subject == "broken":
            raise OSError("not on this machine")
        return session_from_recording(recording, dataclasses.replace(
            request, t_start=0.0, t_stop=float(recording.duration)))
    load.seen = seen
    return load


# -- the batch ---------------------------------------------------------------------------
def test_one_rule_types_every_file_s_channels():
    rule = batchreview.FileRule(all_as="ecog")
    typed = dict(rule.channel_types(["G1", "G2", "EKG1", "ecg-r", "EMG", "DC03", "Photic"]))
    assert typed == {"G1": "ecog", "G2": "ecog", "EKG1": "ecg", "ecg-r": "ecg", "EMG": "emg",
                     "DC03": "misc", "Photic": "misc"}
    assert "every channel ECOG" in rule.describe()


def test_a_batch_analyses_alike_and_keeps_going_past_a_failure(analyse, config, tmp_path):
    template = ReviewRequest(detectors=("rms", "line_length"), band="fast_ripple",
                             threshold_sd=4.0, check_quality=False)
    items = [batchreview.BatchItem(dataset="ds003498", subject="sub-01", t_start=0, t_stop=30),
             batchreview.BatchItem(dataset="ds003498", subject="broken"),
             batchreview.BatchItem(dataset="ds003498", subject="sub-02", t_start=60, t_stop=90)]
    seen = []
    result = batchreview.run_batch(items, template, folder=tmp_path / "b", load=analyse,
                                   progress=lambda i, n, label: seen.append((i, n)))
    assert seen == [(0, 3), (1, 3), (2, 3)]
    for request in analyse.seen:
        assert (request.detectors, request.band, request.threshold_sd, request.check_quality) \
            == (("rms", "line_length"), "fast_ripple", 4.0, False), "analysed alike"
    rows = result.rows
    assert list(rows["error"]) == ["", "OSError: not on this machine", ""]
    first = rows.iloc[0]
    assert first["leader"] and first["n_tied"] >= 1 and "rms" in first["events"]
    assert {"summary.csv", "summary.md", "settings.json", "findings"} <= \
        {p.name for p in (tmp_path / "b").iterdir()}
    settings = json.loads((tmp_path / "b" / "settings.json").read_text())
    assert settings["band"] == "fast_ripple" and settings["threshold_sd"] == 4.0
    assert "2 analysed, 1 not" in (tmp_path / "b" / "summary.md").read_text()
    assert batchreview.findings_for(result, first.to_dict()) is not None


def test_a_batch_stops_between_recordings(analyse, config, tmp_path):
    items = [batchreview.BatchItem(dataset="d", subject=f"s{i}") for i in range(3)]
    calls = []
    result = batchreview.run_batch(items, ReviewRequest(), folder=tmp_path / "s", load=analyse,
                                   progress=lambda *a: calls.append(a),
                                   should_stop=lambda: len(calls) >= 1)
    assert list(result.rows["error"])[1:] == ["stopped before it ran"] * 2


def test_a_file_is_requested_with_the_rule_s_types(tmp_path, monkeypatch):
    from onset_hfo import io

    monkeypatch.setattr(io, "channel_overview",
                        lambda path: pd.DataFrame({"name": ["A1", "A2", "EKG"]}))
    rule = batchreview.FileRule(all_as="seeg", t_start=10, t_stop=40, line_freq=60.0)
    request = batchreview.request_for(batchreview.BatchItem(path=tmp_path / "p01.edf"),
                                      ReviewRequest(detectors=("rms",)), rule)
    assert request.path == tmp_path / "p01.edf" and request.subject == "p01"
    assert dict(request.channel_types) == {"A1": "seeg", "A2": "seeg", "EKG": "ecg"}
    assert (request.t_start, request.t_stop, request.line_freq) == (10, 40, 60.0)


def test_a_batch_row_joins_the_cohort_as_the_open_window_would(analyse, config, tmp_path):
    result = batchreview.run_batch([batchreview.BatchItem(dataset="d", subject="sub-09")],
                                   ReviewRequest(detectors=("rms",)), folder=tmp_path / "c",
                                   load=analyse)
    row = result.rows.iloc[0].to_dict()
    session = analyse(ReviewRequest(dataset="d", subject="sub-09", detectors=("rms",)))
    resected = yourstudy.contacts_of(session.candidates[:1])
    from_batch = yourstudy.add_batch_row(row, batchreview.findings_for(result, row),
                                         result.settings, "recurrence", resected, label="b")
    from_window = yourstudy.measure(session, resected)
    for key in ("candidates", "candidates_resected", "top_channel_resected", "n_candidates"):
        assert from_batch[key] == from_window[key], key
    assert from_batch["from_batch"] and from_batch["outcome"] == "recurrence"
    with pytest.raises(ValueError):
        yourstudy.add_batch_row({**row, "analysed_seconds": 0}, batchreview.findings_for(
            result, row), result.settings, "recurrence", resected)


# -- projects --------------------------------------------------------------------------------
def test_a_request_survives_json_with_its_preprocessing():
    from onset_hfo.config import PreprocessConfig

    request = ReviewRequest(subject="sub-04", t_start=5, t_stop=25, detectors=("rms", "hilbert"),
                            threshold_sd=3.5, keep_channels=("A1-A2",),
                            channel_types=(("A1", "seeg"), ("EKG", "ecg")),
                            preprocess=PreprocessConfig(highpass=2.0, exclude=("B3",),
                                                        ica_exclude=(0, 2)))
    data = json.loads(json.dumps(project.request_to_dict(request)))
    assert project.request_from_dict(data) == request
    assert project.request_from_dict({**data, "from_a_newer_version": 1}) == request


def test_a_project_brings_back_what_was_saved(config, tmp_path):
    from onset_review import adjudication

    request = ReviewRequest(subject="sub-05", t_start=0, t_stop=30)
    read = adjudication.Adjudication(window=adjudication.window_id(request), reader="Dr R",
                                     note="looked at it")
    yourstudy.set_cohort([{"id": "x", "label": "theirs"}])
    saved = project.save_project(tmp_path / "work", request=request, read=read, page="map",
                                 editor_tabs=[{"title": "power.py", "path": None,
                                               "text": "p = 1\n"}],
                                 console_log=["mine = 7"],
                                 variables={"mine": np.arange(4), "fn": lambda: 1})
    assert saved["path"].endswith("work.onsetproj")
    assert saved["variables"] == ["mine"] and "fn" in saved["variables_left_out"]
    assert {"project.json", "read.json", "variables.pkl", "study/cohort.json"} <= \
        set(zipfile.ZipFile(saved["path"]).namelist())

    yourstudy.set_cohort([{"id": "y", "label": "mine"}])
    adjudication.save(request, adjudication.Adjudication(window=read.window, note="local"))
    state = project.open_project(saved["path"])
    assert state.request == request and state.page == "map"
    assert state.read.note == "looked at it" and state.console_log == ["mine = 7"]
    assert state.editor_tabs[0]["text"] == "p = 1\n"
    assert "fn" in " ".join(state.notes)
    backup = project.restore_read(request, state.read)
    assert backup is not None and json.loads(backup.read_text())["note"] == "local"
    assert adjudication.load(request).note == "looked at it"
    moved = project.restore_study(state)
    assert [e["label"] for e in yourstudy.cohort_entries()] == ["theirs"]
    assert json.loads((moved / "cohort.json").read_text())[0]["label"] == "mine"


def test_a_file_recording_travels_or_is_found_and_checked(config, tmp_path):
    original = tmp_path / "patient.edf"
    original.write_bytes(b"0123456789" * 100)
    request = ReviewRequest(dataset="", subject="patient", path=original)

    carried = project.save_project(tmp_path / "with", request=request, include_recording=True)
    state = project.open_project(carried["path"], folder=tmp_path / "unpacked")
    assert state.request.path == tmp_path / "unpacked" / "recording" / "patient.edf"
    assert state.request.path.read_bytes() == original.read_bytes()

    named = project.save_project(tmp_path / "without", request=request)
    assert named["recording"]["included"] is False and named["recording"]["sha256"]
    original.rename(tmp_path / "moved.edf")
    lost = project.open_project(named["path"], folder=tmp_path / "u2")
    assert lost.request is None and "patient.edf" in lost.missing[0]
    other = tmp_path / "other.edf"
    other.write_bytes(b"different")
    found = project.open_project(named["path"], folder=tmp_path / "u3",
                                 locate=lambda entry: str(other))
    assert found.request.path == other
    assert "is not the file the project was saved with" in " ".join(found.notes)
    same = project.open_project(named["path"], folder=tmp_path / "u4",
                                locate=lambda entry: str(tmp_path / "moved.edf"))
    assert same.request.path == tmp_path / "moved.edf" and not same.notes


# -- in the window ------------------------------------------------------------------------------
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
        QThread.msleep(15)


def test_the_batch_window_runs_shows_and_adds_to_the_cohort(qapp, analyse, config,
                                                             monkeypatch):
    from onset_review import session as session_module
    from onset_review.batchwindow import BatchWindow

    monkeypatch.setattr(session_module, "load_session", analyse)
    window = BatchWindow(ReviewRequest(detectors=("rms",)), folder=config / "out")
    try:
        added = window.add_items([batchreview.BatchItem(dataset="d", subject="sub-01"),
                                  batchreview.BatchItem(dataset="d", subject="broken")])
        assert added == 2 and window.add_items([batchreview.BatchItem(
            dataset="d", subject="sub-01")]) == 0, "the same recording once"
        window.run(wait=True)
        _settle(qapp)
        assert window.table.rowCount() == 2
        from onset_review.batchwindow import SHOWN

        error_column = [key for key, _ in SHOWN].index("error")
        assert window.table.item(1, error_column).text().startswith("OSError")
        assert "1 of 2 analysed" in window.status.text()
        assert window.result.folder.parent == config / "out"
        emitted = []
        window.openRequested.connect(emitted.append)
        window.table.selectRow(0)
        request = window.open_selected()
        assert emitted == [request] and request.subject == "sub-01"
        channels = window.result.rows.iloc[0]["channel_names"].split("|")
        entry = window.add_to_cohort(0, "seizure-free", yourstudy.contacts_of(channels[:1]))
        assert entry is not None and yourstudy.cohort_entries()[-1]["from_batch"]
    finally:
        window.close()


def test_every_window_offers_the_batch_and_projects(qapp, config):
    from onset_review import window as window_module

    start = window_module.decorate_start(cached=lambda: pd.DataFrame())
    try:
        for action in start.menuBar().actions():
            if action.text().replace("&", "") == "File":
                entries = {a.text().replace("&", ""): a for a in action.menu().actions()}
        assert entries["Analyse many recordings…"].isEnabled()
        assert entries["Open project…"].isEnabled()
        assert not entries["Save project…"].isEnabled(), "nothing to save before a recording"
    finally:
        start.close()


def test_the_window_saves_a_project_and_reopens_it(qapp, recording, config, tmp_path,
                                                    monkeypatch):
    from onset_review import launcher
    from onset_review.app import _Review, build_parser

    def load(request, cache_dir=None, parent=None):
        return session_from_recording(recording, request)

    monkeypatch.setattr(launcher, "load_with_progress", load)
    request = ReviewRequest(dataset="", subject="synthetic", t_start=0.0,
                            t_stop=float(recording.duration), detectors=("rms",),
                            threshold_sd=4.5)
    first = _Review(qapp, request, False, build_parser().parse_args([]))
    first._open_request(request)
    _settle(qapp)
    try:
        first.parts.session.read.note = "the leader looks real"
        first.editor.new_file("rate = findings['rate_per_min'].max()\n", title="mine.py")
        first.console.run("kept = 41 + 1")
        _settle(qapp)
        first.parts.pages.show_page("map")
        manifest = first.save_project(tmp_path / "study.onsetproj", include_recording=False)
        assert manifest["variables"] == ["kept"] and manifest["page"] == "map"
    finally:
        first.console.clear_user_variables()
        first.parts.host.close()
        _settle(qapp)

    second = _Review(qapp, None, False, build_parser().parse_args([]))
    second.console = first.console
    state = second.open_project(tmp_path / "study.onsetproj", ask=False)
    _settle(qapp)
    try:
        assert state.request == request
        assert second.parts.session.request == request
        assert second.parts.session.read.note == "the leader looks real"
        texts = {e.name(): e.toPlainText() for e in second.parts.panels["editor"].editors()}
        assert texts["mine.py"].startswith("rate = findings")
        assert "kept = 41 + 1" in texts["Console history"]
        assert second.parts.panels["console"].namespace["kept"] == 42
        assert second.parts.pages.current_page() == "map"
    finally:
        second.parts.panels["console"].clear_user_variables()
        second.parts.host.close()
        second.parts.panels["console"].shutdown()
