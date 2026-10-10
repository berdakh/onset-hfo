"""A case's interictal analysis: segments by rule, analysed alike, pooled.

What has to hold: segments come from the rule and nowhere else (stage,
distance from seizures, artefacts, flat stretches), spread evenly in time,
with what each rule took out; one segment pooled alone is exactly the review
window's analysis of that segment -- the same rates, intervals, busiest
channel and tied set; more segments pool by summing events and clean
minutes; analysed segments are reused; and the window runs it all.
"""

from __future__ import annotations

import os

import mne
import numpy as np
import pandas as pd
import pytest

from onset_hfo.case import annotations as marks
from onset_hfo.case import bridges, segments
from onset_hfo.case.model import Case
from onset_hfo.case.pooling import pool
from onset_hfo.metrics import poisson_ci


def _marks(rows):
    return pd.DataFrame([marks.new_mark(*row) for row in rows])


# -- choosing segments ----------------------------------------------------------------------
NIGHT = _marks([(0, "sleep-W", 1500), (1500, "sleep-N2", 900), (2400, "sleep-N3", 1800),
                (3898, "seizure-onset", 0), (3900, "seizure", 75), (4200, "sleep-W", 600),
                (4800, "sleep-N2", 1500), (6300, "sleep-REM", 900),
                (5900, "artefact", 30)])


def test_segments_follow_the_rule_and_say_what_each_rule_took():
    rule = segments.SegmentRule(min_from_seizure_s=1800)
    frame, info = segments.choose([{"run": "01", "duration": 7200, "marks": NIGHT,
                                    "flat": [(6000, 6060)]}], rule)
    spans = list(zip(frame["start"], frame["stop"], strict=True))
    # N2 1500-2400 up to 1800 s before the seizure (onset 3898, no offset: 120 s);
    # N3 is all within 30 min of it; the later N2 starts 30 min after, minus the
    # artefact (5 s margin) and the flat minute.
    assert spans == [(1500.0, 1800.0), (1800.0, 2098.0), (5818.0, 5895.0), (5935.0, 6000.0),
                     (6060.0, 6300.0)]
    assert set(frame["stage"]) == {"sleep-N2"}
    assert frame.loc[0, "nearest_seizure_s"] == pytest.approx(3898 - 1800)
    assert info["excluded"]["not in the chosen stages"] == pytest.approx(3000)
    assert info["excluded"]["near a seizure"] == pytest.approx(3120)
    assert info["excluded"]["artefact"] == pytest.approx(40.0)      # 5895-5935: 30 s and margins
    assert info["excluded"]["flat"] == pytest.approx(60)
    text = segments.describe_choice(frame, info, rule)
    assert text.startswith("5 segment(s)") and "near a seizure" in text


def test_seizures_pair_onsets_with_offsets_and_merge():
    spans = segments.seizure_spans(_marks([(100, "seizure-onset", 0), (160, "seizure-offset", 0),
                                           (150, "seizure", 40), (1000, "seizure-onset", 0),
                                           (5000, "seizure-offset", 0)]))
    assert spans == [(100.0, 190.0), (1000.0, 1000.0 + segments.SEIZURE_FALLBACK_S)]


def test_unscored_sleep_says_so_and_no_stages_means_any_time():
    awake = _marks([(10, "note", 0)])
    frame, info = segments.choose([{"run": "01", "duration": 3600, "marks": awake}],
                                  segments.SegmentRule())
    assert frame.empty and info["no_sleep_scored"]
    assert "No sleep is scored" in segments.describe_choice(frame, info, segments.SegmentRule())
    rule = segments.SegmentRule(stages=(), total_s=900)
    frame, info = segments.choose([{"run": "01", "duration": 3600, "marks": awake},
                                   {"run": "02", "duration": 3600, "marks": awake}], rule)
    assert len(frame) == 3 and info["n_available"] == 24
    assert list(frame["run"]) == ["01", "01", "02"] or frame["run"].nunique() == 2
    assert frame["start"].iloc[0] == 0.0 and frame["stop"].iloc[-1] == 3600.0, "spread out"


def test_the_choice_is_saved_and_logged(tmp_path):
    case = Case.create(tmp_path / "case", "P30")
    rule = segments.SegmentRule(stages=(), total_s=600)
    frame, info = segments.choose([{"run": "01", "duration": 3600, "marks": _marks([])}], rule)
    segments.save_segments(case, frame, rule, info)
    back, rule_back, info_back = segments.load_segments(case)
    assert back["segment"].tolist() == frame["segment"].tolist() and rule_back == rule
    assert info_back["chosen_s"] == 600 and case.log[-1]["action"] == "chose segments"


# -- pooling ------------------------------------------------------------------------------
def test_pooling_sums_events_and_clean_minutes():
    table, leader = pool([
        {"counts": {"A": 10, "B": 2}, "seconds": {"A": 60.0, "B": 60.0}, "tied": ["A"]},
        {"counts": {"A": 5, "B": 4}, "seconds": {"A": 30.0, "B": 60.0}, "tied": ["A", "B"]}])
    a = table.set_index("channel").loc["A"]
    assert a["n_events"] == 15 and a["minutes"] == pytest.approx(1.5)
    assert a["rate_per_min"] == pytest.approx(10.0)
    assert (a["rate_ci_low"], a["rate_ci_high"]) == pytest.approx(poisson_ci(15, 1.5))
    assert a["segments_tied"] == 2 and a["segments_analysed"] == 2 and a["rank"] == 1
    assert leader["leader"] == "A"


@pytest.fixture(scope="module")
def three_minutes(tmp_path_factory):
    from onset_hfo.synthetic import make_synthetic_recording

    folder = tmp_path_factory.mktemp("edf")
    raw = make_synthetic_recording(duration_s=180, seed=11, verbose=False).raw
    path = folder / "night.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path, list(raw.ch_names)


@pytest.fixture
def case(three_minutes, tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_READS", str(tmp_path / "reads"))
    path, names = three_minutes
    case = Case.create(tmp_path / "case", "P31")
    bridges.convert(path, case, channel_types={n: "seeg" for n in names}, bad=["SC6"])
    case.save_marks("01", _marks([(0, "sleep-N2", 180)]))
    return case


def _segments(*spans):
    return pd.DataFrame([{"segment": f"seg-{i + 1:02d}", "run": "01", "start": a, "stop": b,
                          "duration": b - a, "stage": "sleep-N2",
                          "nearest_seizure_s": float("inf")}
                         for i, (a, b) in enumerate(spans)])


def test_one_segment_pooled_is_the_review_window(case):
    from onset_review import caseinterictal as ci
    from onset_review.session import load_session

    result = ci.run_interictal(case, _segments((60.0, 120.0)))
    request = ci.segment_request(case, "01", 60.0, 120.0, ci.template_for(case))
    assert "SC6" in request.preprocess.exclude, "the case's bad contact is left out"
    window = load_session(request)
    mine = result.table.set_index("channel")
    theirs = window.findings.set_index("channel")
    assert set(mine.index) == set(theirs.index)
    for column in ("rate_per_min", "rate_ci_low", "rate_ci_high"):
        assert np.allclose(mine.loc[theirs.index, column], theirs[column]), column
    assert (mine.loc[theirs.index, "n_events"] == theirs["n_events"]).all()
    assert result.leader["leader"] == window.leader["leader"]
    assert result.leader["distinguishable"] == window.leader["distinguishable"]
    assert set(mine.index[mine["tied"]]) == set(window.candidates)


def test_segments_pool_and_are_reused(case, monkeypatch):
    from onset_review import caseinterictal as ci
    from onset_review import session as session_module

    calls = []
    real = session_module.load_session
    monkeypatch.setattr(session_module, "load_session",
                        lambda request, *a, **k: calls.append(request) or real(request))
    first = ci.run_interictal(case, _segments((0.0, 60.0), (60.0, 120.0)))
    assert len(calls) == 2
    second = ci.run_interictal(case, _segments((0.0, 60.0), (60.0, 120.0), (120.0, 180.0)))
    assert len(calls) == 3, "the two analysed before are reused"
    by_segment = second.per_segment.groupby("channel")["n_events"].sum()
    pooled = second.table.set_index("channel")["n_events"]
    assert (pooled.loc[by_segment.index] == by_segment).all()
    assert second.n_segments == 3 and second.minutes == pytest.approx(3.0)
    for name in ("pooled.tsv", "per_segment.tsv", "detections.tsv", "settings.json",
                 "summary.md", "result.json"):
        assert (second.folder / name).exists(), name
    detections = pd.read_csv(second.folder / "detections.tsv", sep="\t")
    assert detections["onset"].between(0, 180).all()
    back = ci.latest_result(case)
    assert back.folder == second.folder and back.leader == second.leader
    assert list(back.table["channel"]) == list(second.table["channel"])
    assert first.folder != second.folder
    # A different threshold is a different analysis: analysed again.
    import dataclasses

    ci.save_template(case, dataclasses.replace(ci.template_for(case), threshold_sd=4.0))
    ci.run_interictal(case, _segments((0.0, 60.0)))
    assert len(calls) == 4
    assert case.log[-1]["action"] == "ran the interictal analysis"


# -- the window ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_the_window_chooses_sets_runs_and_reviews(qapp, case):
    from onset_review import adjudication
    from onset_review.casewindow import CaseWindow

    window = CaseWindow(case, reader="dr test")
    try:
        assert window.show_step("segments")
        page = window.segments_page
        page.segment_min.setValue(1.0)
        page.total_min.setValue(2.0)
        page.from_seizure.setValue(0.0)
        frame = page.choose()
        assert list(zip(frame["start"], frame["stop"], strict=True)) == [(0.0, 60.0),
                                                                          (120.0, 180.0)]
        assert page.table.rowCount() == 2 and page.summary.text().startswith("2 segment(s)")

        assert window.show_step("preprocess")
        settings = window.settings_page
        settings.reference.setCurrentIndex(settings.reference.findData("laplacian"))
        settings.detectors["hilbert"].setChecked(True)
        settings.threshold.setValue(3.5)
        assert settings.save() is not None
        assert case.analysis["preprocess"]["reference"] == "laplacian"
        assert case.analysis["threshold_sd"] == 3.5
        settings.done_button.click()
        assert case.step_done("preprocess")

        assert window.show_step("interictal")
        interictal = window.interictal_page
        assert "2 segment(s)" in interictal.what.text()
        interictal.run(wait=True)
        assert interictal.result is not None and interictal.result.n_segments == 2
        assert interictal.statement.text().startswith("Over 2 segment(s), 2.0 minutes")
        assert interictal.table.rowCount() == len(interictal.result.table)

        assert window.show_step("review")
        review = window.review_page
        assert review.table.rowCount() == 2
        emitted = []
        window.openRequested.connect(emitted.append)
        review.table.selectRow(1)
        request = review.open_selected()
        assert emitted == [request] and (request.t_start, request.t_stop) == (120.0, 180.0)
        assert request.preprocess.reference == "laplacian" and request.threshold_sd == 3.5
        read = adjudication.load(request)
        read.judge_channel(request.channel_types[0][0], "accept", reader="dr test")
        adjudication.save(request, read)
        review.refresh()
        assert review.table.item(1, 5).text() == "1"
    finally:
        window.close()


def test_two_runs_in_the_same_second_get_their_own_folders(tmp_path, monkeypatch):
    import time

    from onset_hfo.case.model import new_run_folder

    monkeypatch.setattr(time, "strftime", lambda fmt, *a: "run-20261010-140242")
    first, second, third = (new_run_folder(tmp_path / "interictal") for _ in range(3))
    assert [f.name for f in (first, second, third)] == [
        "run-20261010-140242", "run-20261010-140242-02", "run-20261010-140242-03"]
    assert sorted(p.name for p in (tmp_path / "interictal").glob("run-*"))[-1] == third.name
