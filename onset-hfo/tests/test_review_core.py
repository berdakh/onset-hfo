"""The desktop reviewer's numbers, checked without a display.

Every figure a clinician reads off the screen is computed in
`onset_review.session`, `trends` and `report`, none of which import Qt. So they
are tested here, in a file that runs on any machine with the base install --
including the CI job that does not have the `review` extra.

That separation is the reason this file exists at all. These tests used to live
beside the Qt ones, under a module-level `importorskip`; a skip there takes the
whole module with it, so on a machine without Qt the numbers were not checked
at all while the suite reported green. `tests/test_review_app.py` now holds only
what genuinely needs a window.

Everything runs on the synthetic recording, so none of it needs a cached slice
of `ds003498` or a network. That is why `session_from_recording` exists as a
function separate from `load_session`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_review import report, trends
from onset_review.session import (
    DETECTOR_LABELS,
    ReviewRequest,
    ReviewSession,
    annotations_for,
    cached_windows,
    session_from_recording,
)


@pytest.fixture(scope="module")
def review(recording) -> ReviewSession:
    """One analysed window of the synthetic recording, reused by every test."""
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


# -- the request -----------------------------------------------------------

def test_request_refuses_an_unknown_detector():
    with pytest.raises(ValueError, match="unknown detector"):
        ReviewRequest(detectors=("wavelet",))


def test_request_refuses_an_empty_window():
    with pytest.raises(ValueError, match="before it starts"):
        ReviewRequest(t_start=60.0, t_stop=60.0)


def test_request_names_every_detector_it_offers():
    """The launcher shows `DETECTOR_LABELS`; a missing one would show a key."""
    from onset_hfo.detectors import HFO_DETECTORS

    assert set(DETECTOR_LABELS) == set(HFO_DETECTORS)


# -- the session's own numbers ---------------------------------------------

def test_every_channel_appears_in_the_findings(review, prepared):
    """Zero is a result. A channel with no events must still have a row."""
    assert set(review.findings["channel"]) == set(prepared.ch_names)


def test_rates_agree_with_the_event_list(review):
    """The table and the list are two views of one set of events, not two counts."""
    listed = trends.event_table(review)
    primary = listed[listed["detector"] == review.request.primary]
    counted = primary.groupby("channel").size()
    for _, row in review.findings.iterrows():
        assert int(row["n_events"]) == int(counted.get(row["channel"], 0))


def test_rate_is_the_count_over_the_window(review):
    minutes = review.request.duration / 60.0
    expected = review.findings["n_events"] / minutes
    assert np.allclose(review.findings["rate_per_min"], expected)


def test_the_interval_contains_the_rate(review):
    low, rate, high = (review.findings[c] for c in
                       ("rate_ci_low", "rate_per_min", "rate_ci_high"))
    assert ((low <= rate + 1e-9) & (rate <= high + 1e-9)).all()


def test_the_caveat_says_which_of_the_two_things_it_is(review):
    """The status bar's sentence has to follow `leader_separation`, not drift."""
    caveat = review.caveat()
    if review.leader.get("distinguishable"):
        assert "stands out" in caveat
    else:
        assert "NO CHANNEL STANDS OUT" in caveat


def test_the_leader_is_in_the_candidate_set(review):
    if review.candidates:
        assert review.leader["leader"] in review.candidates


def test_rejected_events_are_kept_but_not_counted(review):
    """Rejections are the filter's working; losing them loses the explanation."""
    assert len(review.events) >= len(review.accepted)
    rejected = [e for e in review.events if not e.accepted]
    if rejected:
        shown = trends.event_table(review)
        assert len(shown) == len(review.accepted)
        assert len(trends.event_table(review, include_rejected=True)) == len(review.events)


# -- annotations -----------------------------------------------------------

def test_annotations_are_in_the_trace_time_base(review):
    """Events carry archive time; the Raw starts at zero. Mix them and marks slide."""
    annotations = annotations_for(review.accepted, review.t_offset)
    assert len(annotations) == len(review.accepted)
    assert (np.asarray(annotations.onset) >= 0).all()
    # Compared as sorted sequences: MNE orders annotations by onset, while the
    # event list comes out grouped by detector, so pairing them positionally
    # would compare two different events and pass or fail by accident.
    expected = sorted(e.start - review.t_offset for e in review.accepted)
    assert np.allclose(sorted(annotations.onset), expected, atol=1e-9)


def test_annotations_are_named_for_the_band_not_the_detector(review):
    """A clinician asks "is that a fast ripple", never "did line length fire"."""
    descriptions = set(annotations_for(review.accepted, review.t_offset).description)
    assert descriptions <= {"ripple", "fast ripple", "spike"}


def test_the_raw_is_in_volts(review, prepared):
    """`prepare` returns microvolts and MNE scales from SI; the factor is 1e6."""
    assert review.raw.get_data().shape == prepared.data.shape
    assert np.allclose(review.raw.get_data() * 1e6, prepared.data, atol=1e-6)


# -- the trend -------------------------------------------------------------

def test_the_trend_holds_every_event_of_its_detector(review):
    matrix = trends.rate_matrix(review, bin_s=5.0)
    assert int(matrix.to_numpy().sum()) == len(review.events_of(review.request.primary))


def test_the_trend_covers_the_whole_window(review):
    """A trend that stops short of the trace hides whatever is in the tail."""
    for bin_s in (1.0, 5.0, 7.0):          # 7 does not divide 30 or 60
        matrix = trends.rate_matrix(review, bin_s=bin_s)
        assert matrix.shape[1] * bin_s >= review.request.duration
        assert int(matrix.to_numpy().sum()) == len(
            review.events_of(review.request.primary))


def test_trend_rows_follow_the_findings_order(review):
    """The two panels are read across, so row n has to be the same channel."""
    matrix = trends.rate_matrix(review)
    assert list(matrix.index) == list(review.findings["channel"])


def test_the_curve_is_a_rate_not_a_count(review):
    curve = trends.rate_curve(review, bin_s=5.0)
    widths = curve["t_stop"] - curve["t_start"]
    assert np.allclose(curve["rate_per_min"], curve["n_events"] / (widths / 60.0))


# -- agreement -------------------------------------------------------------

def test_agreement_is_absent_rather_than_zero_without_expert_marks(review):
    """A recording nobody annotated has no agreement; 0% would be a lie."""
    if not review.has_expert:
        summary = trends.agreement_summary(review)
        assert summary["available"] is False
        assert trends.agreement(review).empty


def test_agreement_restricts_itself_to_reviewed_channels():
    """A detection on an unreviewed channel is unjudged, not a false positive."""
    import inspect

    source = inspect.getsource(trends.agreement)
    assert "reviewed_channels" in source


# -- the exported review ---------------------------------------------------

def test_the_report_leads_with_the_disclaimer(review):
    text = report.review_markdown(review)
    assert text.index(report.DISCLAIMER) < text.index("## What was reviewed")


def test_the_report_carries_the_caveat_above_the_table(review):
    text = report.review_markdown(review)
    assert review.caveat() in text
    assert text.index(review.caveat()) < text.index("## Per-channel findings")


def test_the_report_states_the_provenance(review):
    text = report.review_markdown(review)
    for step in review.steps:
        assert step in text
    assert review.request.subject in text
    assert f"{review.sfreq:g} Hz" in text


def test_the_report_needs_no_optional_dependency(review, monkeypatch):
    """An export is the last step of a review; it must not need `tabulate`."""
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("tabulate is not installed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert "| rank |" in report.review_markdown(review)


def test_the_report_formats_what_pandas_actually_hands_it():
    """NumPy scalars, not Python ones, are what come out of a DataFrame row.

    `np.bool_` is not a subclass of `bool`, so a `reviewed` column printed
    without checking for it reads "True" and "False" in a clinical document
    that says "yes" and "no" everywhere else.
    """
    from onset_review.report import _cell

    assert _cell(np.bool_(True)) == "yes"
    assert _cell(np.bool_(False)) == "no"
    assert _cell(np.int64(3)) == "3"
    assert _cell(np.float64(2.5)) == "2.50"
    for missing in (None, np.nan, pd.NA, pd.NaT):
        assert _cell(missing) == "—"
    assert _cell("AR1-AR2") == "AR1-AR2"


def test_the_exported_tables_hold_no_raw_python_repr(review):
    """Nothing in the document should read `True`, `nan` or `None`."""
    text = report.review_markdown(review)
    for row in [line for line in text.splitlines() if line.startswith("| ")]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert not ({"True", "False", "nan", "None", "NaN", "<NA>"} & set(cells))


def test_write_review_round_trips(review, tmp_path):
    written = report.write_review(review, tmp_path / "r.md", reviewer="Dr Test")
    assert "Dr Test" in written.read_text()
    page = report.write_review(review, tmp_path / "r.html")
    assert page.read_text().lstrip().startswith("<!doctype html>")


# -- the cache catalogue ---------------------------------------------------

def test_cached_windows_is_empty_rather_than_failing_on_an_empty_cache(tmp_path):
    frame = cached_windows(tmp_path)
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty


def test_cached_windows_reads_the_slice_metadata(tmp_path):
    """The launcher offers only what is here, so a bad read means a bad list."""
    import json

    slice_dir = tmp_path / "ds003498" / "sub-09_run-01_0-60s"
    slice_dir.mkdir(parents=True)
    (slice_dir / "slice.json").write_text(json.dumps({
        "dataset": "ds003498", "subject": "sub-09", "task": None, "run": "01",
        "t_start": 0, "t_stop": 60, "sfreq": 2000.0, "n_channels": 50}))
    low = tmp_path / "ds003029" / "sub-pt01_task-ictal_run-01_0-10s"
    low.mkdir(parents=True)
    (low / "slice.json").write_text(json.dumps({
        "dataset": "ds003029", "subject": "sub-pt01", "task": "ictal",
        "run": "01", "t_start": 0, "t_stop": 10, "sfreq": 1000.0,
        "n_channels": 98}))

    frame = cached_windows(tmp_path).set_index("subject")
    assert frame.loc["sub-09", "bands"] == "ripple, fast_ripple"
    # 1000 Hz cannot carry a 500 Hz band, and the dialog must not offer it.
    assert frame.loc["sub-pt01", "bands"] == "ripple"


def test_cached_windows_survives_a_damaged_entry(tmp_path):
    slice_dir = tmp_path / "ds003498" / "sub-01_run-01_0-60s"
    slice_dir.mkdir(parents=True)
    (slice_dir / "slice.json").write_text("{ not json")
    assert cached_windows(tmp_path).empty
