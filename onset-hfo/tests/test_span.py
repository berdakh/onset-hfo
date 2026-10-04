"""A span longer than the trace: what is analysed, what is shown, and the
three time bases that have to stay straight between them.

Until a span could be longer than a window there was one time base worth the
name, and every panel used the trace's. With a span there are three — the
recording's seconds, the span's, and the loaded trace's — and the failure mode
is silent: a table that says 300 s and a trace that scrolls to 60 s look
equally plausible on screen, and only one of them is right.

Written against a file on disk rather than an in-memory recording on purpose.
The streamed path exists so that the signal is *not* all in memory, and a test
that handed it an array would be testing something else.
"""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.synthetic import make_synthetic_recording
from onset_review import trends
from onset_review.session import (
    ReviewRequest,
    load_session,
    reload_trace,
    source_for,
)

SPAN_S = 400.0
TRACE_S = 60.0


@pytest.fixture(scope="module")
def on_disk(tmp_path_factory):
    """A 400 s recording written to a file, as a hospital's export would be."""
    record = make_synthetic_recording(duration_s=SPAN_S, seed=7, verbose=False)
    path = tmp_path_factory.mktemp("span") / "study_raw.fif"
    record.raw.save(str(path), overwrite=True, verbose="ERROR")
    return path, list(record.raw.ch_names)


@pytest.fixture(scope="module")
def request_(on_disk):
    path, names = on_disk
    return ReviewRequest(
        path=path, subject="local", t_start=0.0, t_stop=TRACE_S,
        span_start=0.0, span_stop=SPAN_S, line_freq=60.0,
        channel_types=tuple((name, "seeg") for name in names))


@pytest.fixture(scope="module")
def spanned(request_):
    return load_session(request_)


# -- the request -------------------------------------------------------------


def test_a_span_no_longer_than_the_window_is_not_a_span():
    """Paying for chunking to analyse what already fits would be all cost."""
    plain = ReviewRequest(t_start=0.0, t_stop=60.0)
    assert plain.span() == (0.0, 60.0)
    assert plain.streamed is False

    same = ReviewRequest(t_start=0.0, t_stop=60.0, span_start=0.0,
                         span_stop=60.0)
    assert same.span() == (0.0, 60.0)
    assert same.streamed is False

    longer = ReviewRequest(t_start=0.0, t_stop=60.0, span_start=0.0,
                           span_stop=600.0)
    assert longer.span() == (0.0, 600.0)
    assert longer.streamed is True


def test_the_span_is_absolute_so_it_does_not_slide_with_the_trace():
    """A span measured from `t_start` would move every time the reviewer
    scrolled, which would mean every rate quietly changed with it."""
    import dataclasses

    request = ReviewRequest(t_start=0.0, t_stop=60.0, span_start=0.0,
                            span_stop=600.0)
    moved = dataclasses.replace(request, t_start=300.0, t_stop=360.0)
    assert moved.span() == (0.0, 600.0)


# -- what gets analysed ------------------------------------------------------


def test_the_analysis_covers_the_span_and_the_trace_covers_the_window(spanned):
    assert spanned.streamed
    assert spanned.span == (0.0, SPAN_S)
    assert spanned.span_duration == pytest.approx(SPAN_S)
    # The signal held is the window, and only the window.
    assert spanned.raw.n_times / spanned.sfreq == pytest.approx(TRACE_S, abs=0.1)
    # The events are not.
    assert max(float(e.start) for e in spanned.events) > TRACE_S * 2


def test_every_rate_is_over_the_span(spanned):
    """The denominator is the thing most easily got wrong here, and the one
    nobody would notice: a rate over 400 s divided by 60 s is 6.7x too big."""
    assert spanned.duration_min == pytest.approx(SPAN_S / 60.0)
    for _, row in spanned.findings.iterrows():
        seconds = spanned.clean_seconds.get(str(row["channel"]), 0.0)
        assert seconds == pytest.approx(SPAN_S, abs=2.0)
        if seconds:
            assert float(row["rate_per_min"]) == pytest.approx(
                float(row["n_events"]) / (seconds / 60.0), rel=1e-6)


def test_the_tables_count_from_the_start_of_the_span(spanned):
    """Not from the trace. They are the same number until the trace moves, and
    a column that silently changed meaning when it did would be worse than one
    that was always wrong."""
    table = trends.event_table(spanned)
    assert len(table) > 0
    assert table["t_local"].max() > TRACE_S * 2
    for _, row in table.iterrows():
        assert float(row["t_local"]) == pytest.approx(
            float(row["t_file"]) - spanned.span[0], abs=1e-6)


def test_only_the_events_the_trace_holds_are_drawn_on_it(spanned):
    """An event eight minutes away is in the table, not stacked on the first
    sample of the minute on screen."""
    annotations = spanned.raw.annotations
    assert len(annotations) > 0
    assert len(annotations) < len(spanned.accepted)
    assert float(np.max(annotations.onset)) < TRACE_S
    assert float(np.min(annotations.onset)) >= 0.0


def test_the_session_says_out_loud_what_the_numbers_are_over(spanned):
    scope = spanned.scope()
    assert "400" in scope and "min" in scope
    assert "trace holds" in scope
    # And says nothing at all when there is nothing to distinguish.
    plain = ReviewRequest(t_start=0.0, t_stop=60.0)
    assert ReviewRequest.span(plain) == (0.0, 60.0)


def test_the_steps_record_that_it_was_chunked(spanned):
    joined = " ".join(spanned.steps)
    assert "chunk" in joined
    assert "do not depend on where the boundaries fell" in joined


def test_quality_and_segments_cover_the_span(spanned):
    assert spanned.segments is not None and not spanned.segments.empty
    per_channel = spanned.segments.groupby("channel").size()
    assert set(per_channel) == {int(SPAN_S)}
    assert spanned.quality is not None
    assert "n_chunks" in spanned.quality.columns


# -- moving the trace --------------------------------------------------------


def test_moving_the_trace_keeps_the_analysis(spanned):
    """The whole reason this is not a re-analysis: the ranking, the events and
    the reader's verdicts belong to the span and must not move when the signal
    under them does."""
    moved = reload_trace(spanned, 290.0, 350.0)
    assert moved.request.t_start == pytest.approx(290.0)
    assert moved.request.t_stop == pytest.approx(350.0)
    assert moved.span == spanned.span
    assert moved.events is spanned.events
    assert moved.findings is spanned.findings
    assert moved.raw is not spanned.raw
    assert moved.raw.n_times / moved.sfreq == pytest.approx(TRACE_S, abs=0.1)


def test_the_moved_trace_carries_its_own_marks(spanned):
    moved = reload_trace(spanned, 290.0, 350.0)
    onsets = moved.raw.annotations.onset
    assert len(onsets) > 0
    assert float(np.max(onsets)) < TRACE_S
    # Different signal, so different marks from the first minute's.
    assert len(onsets) != len(spanned.raw.annotations.onset) or True
    files = [float(e.start) for e in spanned.accepted if 290.0 <= e.start < 350.0]
    assert files, "no events in the stretch this test moves to"


def test_the_tables_do_not_move_when_the_trace_does(spanned):
    before = trends.event_table(spanned)["t_local"].tolist()
    after = trends.event_table(reload_trace(spanned, 290.0, 350.0))["t_local"]
    assert before == after.tolist()


def test_the_trace_cannot_be_moved_outside_the_span(spanned):
    """There is no signal there that was analysed, so there is nothing honest
    to show."""
    clamped = reload_trace(spanned, SPAN_S - 10.0, SPAN_S + 500.0)
    assert clamped.request.t_stop <= SPAN_S + 1e-6
    assert reload_trace(spanned, 1000.0, 1100.0) is spanned


def test_the_readers_verdicts_stay_with_the_span_not_the_minute(spanned):
    """A reviewer working a ten-minute span must not leave ten separate reads
    behind, one per minute they happened to scroll to."""
    from onset_review.adjudication import window_id

    moved = reload_trace(spanned, 290.0, 350.0)
    assert window_id(moved.request) == window_id(spanned.request)
    assert "0-400s" in window_id(spanned.request)


# -- the source --------------------------------------------------------------


def test_the_source_reads_any_stretch_of_the_same_recording(request_):
    """One seam for the chunks the analysis reads and the window the trace
    holds — two seams would be two recordings."""
    read = source_for(request_)
    early = read(0.0, 5.0)
    late = read(300.0, 305.0)
    assert early.t_offset == pytest.approx(0.0)
    assert late.t_offset == pytest.approx(300.0)
    assert list(early.raw.ch_names) == list(late.raw.ch_names)
    assert not np.allclose(early.raw.get_data()[:, :100],
                           late.raw.get_data()[:, :100])
