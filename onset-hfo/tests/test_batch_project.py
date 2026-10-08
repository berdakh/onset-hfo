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


# -- across the batch ---------------------------------------------------------------------
def _rows(n: int, loud: int | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"recording": f"rec-{i}", "subject": "sub-01" if i < 3 else f"sub-{i:02d}",
                     "leader": "AD3" if i in (0, 2) else f"B{i}",
                     "leader_rate_per_min": 10.0 + i, "leader_ci_low": 5.0 + i,
                     "leader_ci_high": 16.0 + i, "median_rate_per_min": 2.0,
                     "stands_out": i % 2 == 0, "n_tied": 1 if i % 2 == 0 else 4,
                     "events_total": 300 if i == loud else 20 + i, "analysed_seconds": 60.0,
                     "channels": 40, "quality_set_aside": 2, "expert_f1": np.nan,
                     "error": ""})
    rows.append({"recording": "missing.edf", "subject": "missing", "error": "OSError: gone"})
    return pd.DataFrame(rows)


def test_robust_z_is_the_modified_z_with_a_fallback_for_a_zero_mad():
    from onset_review.batchresults import robust_z

    z = robust_z([1.0, 2.0, 3.0, 4.0, 100.0])
    assert z[2] == 0 and z[4] == pytest.approx(0.6745 * 97 / 1.0)
    z = robust_z([5.0, 5.0, 5.0, 5.0, 9.0])          # MAD 0: the mean deviation stands in
    assert z[4] == pytest.approx(4.0 / (1.2533 * 0.8)) and z[0] == 0
    assert list(robust_z([3.0, 3.0, np.nan])[:2]) == [0.0, 0.0]


def test_across_the_batch_reads_leaders_repeats_agreement_and_outliers():
    from onset_review import batchresults

    scores = pd.DataFrame([{"recording": f"rec-{i}", "detector": d, "f1": f1,
                            "rank_rho": 0.5}
                           for i in range(3) for d, f1 in (("rms", 0.2 + i / 10),
                                                           ("line_length", 0.1))])
    a = batchresults.across(_rows(6, loud=4), scores)
    assert len(a.leaders) == 6 and a.n_failed == 1 and a.can_flag
    assert a.leaders.loc[4, "events_per_min"] == 300.0
    assert list(a.outliers["recording"]) == ["rec-4"]
    assert a.outliers.iloc[0]["measure"] == "accepted events per minute"
    assert a.outliers.iloc[0]["direction"] == "higher"
    assert a.recurring.to_dict("records") == [{"subject": "sub-01", "windows": 3,
                                               "leader": "AD3", "times": 2}]
    rms = a.agreement.set_index("detector").loc["rms"]
    assert rms["n"] == 3 and rms["median_f1"] == pytest.approx(0.3)
    text = " ".join(a.sentences)
    assert "6 of 7 recordings analysed; 1 not" in text
    assert "In 3 of 6 the busiest channel stands out" in text
    assert "sub-01: the busiest channel was AD3 in 2 of 3 windows" in text
    assert "RMS energy agreed best with them (median F1 0.30" in text
    assert "rec-4" in text
    few = batchresults.across(_rows(4, loud=3))
    assert few.outliers.empty and not few.can_flag
    assert "fewer than 5 recordings" in " ".join(few.sentences)
    assert "None of these recordings carries expert markings" in " ".join(few.sentences)


def test_the_figures_draw_every_recording_and_detector():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from onset_review import batchresults

    a = batchresults.across(_rows(6), pd.DataFrame([
        {"recording": "rec-0", "detector": "rms", "f1": 0.4, "rank_rho": 0.1},
        {"recording": "rec-1", "detector": "hilbert", "f1": 0.2, "rank_rho": 0.2}]))
    figure = Figure()
    points = batchresults.draw_leaders(figure, figure.add_subplot(111), a.leaders)
    assert set(points) == {f"rec-{i}" for i in range(6)}
    filled = points["rec-0"].get_markerfacecolor()
    assert filled != "white" and points["rec-1"].get_markerfacecolor() == "white"
    figure = Figure()
    dots = batchresults.draw_agreement(figure, figure.add_subplot(111), a.scores)
    assert set(dots) == {"rms", "hilbert"}


def test_a_batch_writes_and_reads_back_its_expert_scores(analyse, config, tmp_path):
    from onset_review import batchresults

    items = [batchreview.BatchItem(dataset="d", subject=f"sub-0{i}") for i in range(1, 3)]
    result = batchreview.run_batch(items, ReviewRequest(detectors=("rms", "line_length")),
                                   folder=tmp_path / "e", load=analyse)
    row = result.rows.iloc[0]
    assert row["events_total"] >= 0 and np.isfinite(row["leader_ci_low"])
    assert row["leader_ci_low"] <= row["leader_rate_per_min"] <= row["leader_ci_high"]
    back = batchreview.load_batch(tmp_path / "e")
    if result.scores is not None:
        assert (tmp_path / "e" / "scores.csv").exists()
        assert set(back.scores["detector"]) == set(result.scores["detector"])
        assert row["expert_f1"] == pytest.approx(result.scores.iloc[0]["f1"], abs=1e-3)
    a = batchresults.across(back.rows, back.scores)
    assert len(a.leaders) == 2 and a.sentences[0].startswith("2 of 2 recordings analysed")


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
        across = window.across_view
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "Table", "Across the batch"]
        assert across.summary.text().startswith("1 of 2 recordings analysed; 1 not")
        window.tabs.setCurrentIndex(1)
        _settle(qapp)
        assert not across.leaders_canvas.isHidden() and across.leaders_canvas.parent() is not None
        assert "Needs 5 analysed recordings" in across.outliers_note.text()
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
