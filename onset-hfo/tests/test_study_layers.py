"""The study pages, answering to the analysis that is open.

Asked for: pages that are interactive and "based on each data analysis".
What has to hold, above all, is that no published number moves:

* *This recording* is scored exactly as the study scored a patient (a window
  whose markings are its own detections scores 1.0), is headed and caveated
  as one window, and is absent when nothing is open;
* the published page text is the same with or without a recording open;
* a re-run is the study's own functions with the reader's settings, written
  beside the reader's settings, never over the committed tables, with every
  skipped patient named;
* a reader's cohort is measured by the study's own tie-aware rule;
* the notebook a page opens as rebuilds the page's numbers exactly;
* the live chart reads out the committed sweep and opens a clicked patient.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

from onset_review import studies, thisrecording, yourstudy
from onset_review.session import ReviewRequest, session_from_recording

needs_tables = pytest.mark.skipif(not studies.available(),
                                  reason="the study tables are not in this checkout")


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path


@pytest.fixture
def window(recording):
    """A window whose expert markings are its own RMS detections."""
    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration),
                                 detectors=("rms", "line_length")))
    return session


def _marked(session):
    accepted = [e for e in session.events if e.detector == "rms" and e.accepted]
    truth = pd.DataFrame({"channel": [e.channel for e in accepted],
                          "start": [e.start for e in accepted],
                          "stop": [e.stop for e in accepted], "kind": "ripple"})
    session.recording = types.SimpleNamespace(**{**vars(session.recording),
                                                 "ground_truth": truth})
    session.reviewed_channels = list(session.raw.ch_names)
    return session


# -- this recording --------------------------------------------------------------


def test_a_window_is_scored_the_way_the_study_scored_a_patient(window):
    session = _marked(window)
    scores = thisrecording.score_window(session).set_index("detector")
    rms = scores.loc["rms"]
    assert (rms.precision, rms.recall, rms.f1, rms.rank_rho) == (1.0, 1.0, 1.0, 1.0), \
        "markings equal to the detections score perfectly"
    assert rms.threshold_sd == thisrecording.threshold_for(session, "rms")
    assert 0 < scores.loc["line_length"].f1 < 1
    assert thisrecording.marks(session, "f1", "fast_ripple") == [], "only the window's band"
    marks = thisrecording.marks(session, "f1", "ripple")
    assert {m["detector"] for m in marks} == {"rms", "line_length"}


def test_without_markings_the_page_says_so_and_compares_the_detectors(window):
    assert thisrecording.score_window(window) is None
    pairs = thisrecording.detector_agreement(window)
    assert list(pairs["pair"]) == ["rms · line_length"]
    text = thisrecording.section("detectors", window)
    assert text.startswith("## This recording — ")
    assert "carries no expert markings" in text and "agreement, not accuracy" in text


def test_every_section_is_headed_as_this_recording_and_nothing_without_one(window):
    for key in ("detectors", "outcome", "patients", "data", "architecture"):
        text = thisrecording.section(key, window)
        assert text.startswith("## This recording — "), key
        assert thisrecording.section(key, None) == ""
    assert thisrecording.section("research", window) == ""
    assert not thisrecording.in_cohort(window), "a synthetic window is no study patient"
    assert "cannot be applied to one patient" in thisrecording.section("outcome", window)
    steps = thisrecording.section("architecture", window)
    assert all(str(step) in steps for step in window.steps)


@needs_tables
def test_a_study_patient_is_found_in_the_cohort(window, monkeypatch):
    cohort = studies._panels().cohort_overview()
    subject = str(cohort["subject"].iloc[0])
    window.recording = types.SimpleNamespace(**{**vars(window.recording),
                                                "dataset_id": "ds003498"})
    window.request = ReviewRequest(subject=subject, t_start=0.0, t_stop=30.0)
    assert thisrecording.in_cohort(window)
    text = thisrecording.section("outcome", window, cohort)
    assert "is one of the study's patients" in text and subject in text


@needs_tables
@pytest.mark.parametrize("key", ["outcome", "data", "architecture", "research", "patients"])
def test_the_published_text_is_the_same_with_a_recording_open(window, config, key):
    alone = studies.build(key)
    beside = studies.build(key, session=window)
    section = thisrecording.section(key, window,
                                    studies._panels().cohort_overview()
                                    if key == "outcome" else None)
    if section:
        assert beside.split("\n", 1)[0] == alone.split("\n", 1)[0]
        assert section.strip() in beside
        beside = beside.replace(section, "").replace("\n\n\n\n", "\n\n")
    def lines(text: str) -> list[str]:
        return [line for line in text.split("\n")
                if line.strip() and "outcome-patients" not in line]

    assert lines(beside) == lines(alone)


@needs_tables
def test_the_sweep_marks_this_window_and_names_the_mark(window, config):
    text = studies.build("detectors", session=_marked(window))
    assert text.index("## This recording") < text.index("## The reference")
    assert "Stars: this window" in text


# -- your re-run ----------------------------------------------------------------------


def _fake_score(subject, **settings):
    if subject == "sub-03":
        raise OSError("not on this machine")
    value = {"sub-01": 0.2, "sub-02": 0.4}[subject]
    rows = [{"subject": subject, "band": "ripple", "detector": d, "threshold_sd": t,
             "precision": value, "recall": value, "f1": value, "n_detections": 10 * t}
            for d in settings["detectors"] for t in settings["thresholds"]]
    ranks = [{"subject": subject, "band": "ripple", "detector": d, "threshold_sd": t,
              "spearman_rho": value, "top5_overlap": 3}
             for d in settings["detectors"] for t in settings["thresholds"]]
    return types.SimpleNamespace(rows=rows, ranks=ranks)


def test_a_re_run_is_kept_beside_your_settings_and_names_what_it_skipped(config):
    seen = []
    result = yourstudy.rerun_detectors(["sub-01", "sub-02", "sub-03"], [2, 3], ["rms"],
                                       progress=lambda i, n, s: seen.append((i, n, s)),
                                       score=_fake_score)
    assert seen == [(0, 3, "sub-01"), (1, 3, "sub-02"), (2, 3, "sub-03")]
    summary = result["summary"]
    assert list(summary["threshold_sd"]) == [2.0, 3.0]
    assert summary["f1"].tolist() == pytest.approx([0.3, 0.3]), "the mean over patients"
    assert summary["n_subjects"].tolist() == [2, 2]
    assert result["skipped"] == {"sub-03": "OSError: not on this machine"}
    frame, settings = yourstudy.load_your_sweep()
    assert settings["n_subjects"] == 2 and settings["thresholds"] == [2.0, 3.0]
    assert frame["f1"].tolist() == pytest.approx([0.3, 0.3])
    assert str(yourstudy.state_dir()).startswith(str(config / "config"))


def test_a_re_run_stops_between_patients(config):
    calls = []
    result = yourstudy.rerun_detectors(["sub-01", "sub-02"], [2], ["rms"],
                                       should_stop=lambda: bool(calls),
                                       progress=lambda *a: calls.append(a), score=_fake_score)
    assert result["skipped"] == {"sub-02": "stopped before it ran"}


@needs_tables
def test_your_re_run_shows_under_its_own_heading_and_dashed(config):
    yourstudy.rerun_detectors(["sub-01", "sub-02", "sub-03"], [2, 3], ["rms"],
                              score=_fake_score)
    text = studies.build("detectors")
    assert "## Your re-run — not the published study" in text
    assert "sub-03 (OSError: not on this machine)" in text
    assert "Dashed: your re-run" in text
    data = studies.chart_data("detectors")
    assert data["yours"] is not None and len(data["yours"]) == 2


def test_the_outcome_study_re_runs_with_your_detector(config):
    def measure(subject, resection, **settings):
        share = {"sub-01": 1.0, "sub-02": 0.5, "sub-03": 0.0, "sub-04": 0.25}[subject]
        rows = [{"subject": subject, "band": settings["bands"][0], "scope": "reviewed",
                 "source": source, "candidates_resected": share, "share_in_rz": share,
                 "share_in_rz_incl_partial": share, "top_channel_resected": share,
                 "top3_resected": share, "candidates_all_resected": share}
                for source in ("expert", settings["detector"])]
        return rows, [], {"subject": subject}

    participants = pd.DataFrame({"subject": ["sub-01", "sub-02", "sub-03", "sub-04"],
                                 "outcome": ["S", "S", "F", "F"]})
    resections = {s: ("A1",) for s in participants["subject"]}
    result = yourstudy.rerun_outcome(list(participants["subject"]) + ["sub-05"],
                                     detector="line_length", band="ripple",
                                     measure_subject=measure, resections=resections,
                                     participants=participants)
    cohort = result["cohort"].set_index("subject")
    assert cohort.loc["sub-01", "candidates_resected_rms"] == 1.0
    assert bool(cohort.loc["sub-01", "seizure_free"]) and not cohort.loc["sub-03",
                                                                         "seizure_free"]
    assert result["skipped"] == {"sub-05": "no resected zone in the clinical sheet"}
    groups = result["groups"]
    row = groups[(groups["source"] == "line_length")
                 & (groups["metric"] == "candidates_resected")].iloc[0]
    assert row["auc"] == 1.0
    frame, settings = yourstudy.load_your_outcome()
    assert settings["detector"] == "line_length" and len(frame) == 4


# -- your cohort ---------------------------------------------------------------------


def test_your_patient_is_measured_by_the_study_s_tie_aware_rule(window, config):
    from onset_hfo.outcome import candidate_channels

    counts = window.findings.set_index("channel")["n_events"].astype(float) \
        .reindex(window.raw.ch_names, fill_value=0.0)
    tied = candidate_channels(counts, float(window.span[1] - window.span[0]) / 60.0)
    contacts = yourstudy.contacts_of(tied[:1])
    entry = yourstudy.add_to_cohort(window, "seizure-free", contacts, label="mine")
    assert entry["candidates"] == tied
    assert entry["candidates_resected"] == pytest.approx(1 / len(tied))
    assert entry["top_channel_resected"] == 1.0
    with pytest.raises(ValueError):
        yourstudy.add_to_cohort(window, "cured", contacts)
    with pytest.raises(ValueError):
        yourstudy.add_to_cohort(window, "recurrence", [])
    again = yourstudy.add_to_cohort(window, "recurrence", contacts, label="mine")
    assert len(yourstudy.cohort_entries()) == 1, "the same window replaces itself"
    assert yourstudy.remove_from_cohort(again["id"]) and not yourstudy.cohort_entries()


def test_your_cohort_is_compared_only_with_two_in_each_group(config):
    frame = pd.DataFrame({"subject": list("abcd"), "seizure_free": [True, True, False, False],
                          "candidates_resected_rms": [1.0, 0.8, 0.2, 0.0],
                          "band": "ripple", "detector": "rms"})
    result = yourstudy.compare_cohort(frame)
    assert result["auc"] == 1.0 and result["n_seizure_free"] == 2
    assert np.isnan(result["floor"]), "two against two cannot reach power for any effect"
    six = pd.concat([frame, frame.assign(subject=list("efgh"))], ignore_index=True)
    assert 0.9 <= yourstudy.compare_cohort(six)["floor"] <= 1.0
    small = yourstudy.compare_cohort(frame.iloc[:3])
    assert np.isnan(small["auc"]) and np.isnan(small["floor"])


# -- notebooks --------------------------------------------------------------------------


@needs_tables
@pytest.mark.parametrize("key", ["detectors", "outcome", "patients", "research"])
def test_a_page_s_notebook_runs_and_rebuilds_its_numbers(key, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from onset_review import cells, studynotebooks

    path = studynotebooks.write_study_notebook(key, tmp_path, "# Page\n\ntext")
    assert path.name == f"onset-{key}.ipynb"
    assert studynotebooks.write_study_notebook(key, tmp_path).name == f"onset-{key}-2.ipynb"
    script = cells.notebook_to_script(cells.read_notebook(path))
    namespace: dict = {}
    for cell in cells.split_cells(script):
        source = cell.source(script.split("\n"))
        if not cell.markdown and source.strip():
            exec(compile(source, str(path), "exec"), namespace)      # noqa: S102
    if key == "outcome":
        groups = pd.read_csv(os.path.join(studies._panels().STUDIES,
                                          "outcome_groups_300s.csv"))
        published = groups.query("scope == 'reviewed' and band == 'fast_ripple' and "
                                 "metric == 'candidates_resected' and source == 'rms'").iloc[0]
        assert namespace["result"]["auc"] == pytest.approx(published["auc"])
        assert namespace["result"]["p_permutation"] == pytest.approx(published["p_permutation"])


# -- in the window ---------------------------------------------------------------------
# Skipped here, not at the top of the file, so the Qt-free tests above still
# run on the job that installs no `review` extra.


@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def _settle(app, n=8):
    from qtpy.QtCore import QThread

    for _ in range(n):
        app.processEvents()
        QThread.msleep(15)


@needs_tables
def test_the_live_sweep_reads_out_the_committed_numbers(qapp, window, config):
    from onset_review.chartview import ChartWindow

    data = studies.chart_data("detectors", session=_marked(window))
    chart = ChartWindow("sweep", data)
    try:
        assert chart.threshold == 5.0, "opens at the window's own threshold"
        frame = data["frame"]
        row = frame[(frame.band == "ripple") & (frame.detector == "rms")
                    & (frame.threshold_sd == 3.0)].iloc[0]
        chart.slider.setValue(chart._thresholds.index(3.0))
        assert f"F1 {row.f1:.2f}" in chart.readout.text()
        assert f"ρ {row.rank_rho:.2f}" in chart.readout.text()
    finally:
        chart.close()


@needs_tables
def test_a_patient_dot_answers_the_pointer_and_opens_their_window(qapp, config):
    from matplotlib.backend_bases import MouseEvent

    from onset_review.chartview import ChartWindow

    data = studies.chart_data("outcome")
    chart = ChartWindow("outcome", data)
    clicked = []
    chart.subjectClicked.connect(clicked.append)
    chart.resize(1000, 600)
    chart.show()
    _settle(qapp)
    chart.canvas.draw()
    dots, subjects, values, label = chart.dots[0]
    x, y = dots.get_offsets()[0]
    px, py = dots.axes.transData.transform((x, y))
    event = MouseEvent("button_press_event", chart.canvas, px, py, button=1)
    assert chart.describe_at(event).startswith(f"{subjects[0]} — {label}")
    chart._click(event)
    assert clicked == [subjects[0]]
    chart.close()


@pytest.fixture
def pages(qapp, window, config, monkeypatch):
    from onset_review import window as window_module

    figure = window_module.open_trace(window, show=False)
    parts = window_module.decorate(figure, window, mode="pages",
                                   cached=lambda: pd.DataFrame())
    parts.pages.resize(1680, 980)
    parts.pages.show()
    _settle(qapp)
    yield parts
    parts.pages.close()
    parts.panels["console"].shutdown()
    try:
        figure.close()
    except Exception:
        pass


@needs_tables
def test_the_study_pages_carry_this_recording_and_their_actions(qapp, pages):
    host = pages.pages
    host.show_page("detectors")
    _settle(qapp)
    page = host._pages["detectors"].body
    assert page.session is pages.session
    assert "## This recording" in page.text
    assert set(page.actions) == {"explore", "rerun", "notebook"}
    assert set(host._pages["outcome"].body.actions) == {"explore", "rerun", "add", "remove",
                                                        "notebook"}
    assert set(host._pages["data"].body.actions) == {"notebook"}


@needs_tables
def test_a_re_run_from_the_page_runs_off_the_window_and_shows_as_yours(qapp, pages):
    host = pages.pages
    host.show_page("detectors")
    _settle(qapp)
    page = host._pages["detectors"].body
    seen = []

    def runner(settings, progress=None, should_stop=None):
        return yourstudy.rerun_detectors(settings["subjects"], settings["thresholds"],
                                         settings["detectors"],
                                         progress=lambda *a: (seen.append(a), progress(*a)),
                                         should_stop=should_stop, score=_fake_score)

    page.study_runner = runner
    from onset_review.studypages import RerunDialog

    dialog = RerunDialog("detectors", pages.session, page)
    dialog.subjects.setText("sub-01, sub-02, sub-03")
    dialog.thresholds.setText("2, 3")
    settings = dialog.settings()
    assert settings["detectors"] == ["rms", "line_length"] and settings["offline"]
    result = page.run_study(settings)
    assert len(seen) == 3 and result["skipped"] == {"sub-03": "OSError: not on this machine"}
    assert "## Your re-run" in page.text
    assert "Re-run on 2 of 3 patients" in page.note.text()


def test_the_re_run_dialog_refuses_what_cannot_run(qapp, window):
    from onset_review.studypages import RerunDialog

    dialog = RerunDialog("detectors", window)
    dialog.thresholds.setText("two")
    dialog._accept()
    assert "numbers" in dialog.problem.text()
    dialog.thresholds.setText("2")
    for box in dialog.detector_boxes.values():
        box.setChecked(False)
    dialog._accept()
    assert "detector" in dialog.problem.text()
    outcome = RerunDialog("outcome", window)
    assert outcome.settings()["threshold_sd"] is None, "blank means the measured default"
    assert outcome.t_stop.value() == 300.0


@needs_tables
def test_this_recording_joins_your_cohort_from_the_outcome_page(qapp, pages):
    from onset_review.studypages import CohortDialog

    host = pages.pages
    host.show_page("outcome")
    _settle(qapp)
    page = host._pages["outcome"].body
    dialog = CohortDialog(pages.session, page)
    dialog._accept()
    assert "outcome" in dialog.problem.text()
    dialog.recurrence.setChecked(True)
    from qtpy.QtCore import Qt

    dialog.contacts.item(0).setCheckState(Qt.Checked)
    outcome, ticked, label = dialog.values()
    entry = page.add_to_cohort(outcome, ticked, label)
    assert entry is not None and entry["outcome"] == "recurrence"
    assert "## Your cohort" in page.text and label in page.text
    assert page.actions["remove"].isEnabled()


@needs_tables
def test_open_as_notebook_puts_the_page_on_the_analysis_page(qapp, pages, tmp_path):
    host = pages.pages
    host.show_page("outcome")
    _settle(qapp)
    path = host._pages["outcome"].body.open_notebook(folder=tmp_path)
    _settle(qapp)
    assert host.current_page() == "analysis"
    editor = pages.panels["editor"]
    assert editor.current().path == path.resolve() and editor.current().is_notebook


def test_the_study_s_pages_are_the_library_one_place_at_the_foot(qapp, pages, window):
    from onset_review.studies import STUDIES

    host = pages.pages
    assert host.show_section("library") and host.current_page() == "quickstart"
    assert list(host.segment_buttons) == ["quickstart"] + [k for k, _, _ in STUDIES]
    host.segment_buttons["outcome"].click()
    assert host.current_page() == "outcome" and host.title.text() == "Library"
    host.show_page("recording")
    assert host.show_section("library") and host.current_page() == "outcome", \
        "a place comes back on the page last shown in it"
