"""Analysing a span in chunks, and the claim that it is the same analysis.

`onset_hfo.streaming` already proves that detection over pre-prepared chunks
returns what the whole array returns. These tests cover the part that was
never covered, and the part a real source forces: **each chunk is preprocessed
separately**, so the 1 Hz high-pass's impulse response — not the detector's
band-pass — is the longest filter at every boundary.

The headline assertion is chunk-size invariance. If three chunk lengths give
byte-identical HFO events, no boundary is corrupting anything, and whatever
residue remains against a whole-array run is the baseline sketch's bounded
memory rather than the chunking. That separation is the whole argument, so it
is tested as two facts rather than one.
"""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.config import PipelineConfig
from onset_hfo.longrun import (
    DEFAULT_CHUNK_S,
    DEFAULT_OVERLAP_S,
    STREAM_ABOVE_S,
    SpanAnalysis,
    analyse_span,
    recording_source,
)
from onset_hfo.pipeline import HFO_DETECTORS
from onset_hfo.preprocess import prepare
from onset_hfo.streaming import OverlapTooShort
from onset_hfo.synthetic import make_synthetic_recording

#: Long enough to be past the baseline sketch's 200 000-sample cap at 2 kHz,
#: so that the subsampling the module documents is actually exercised rather
#: than assumed away by a short fixture.
LONG_S = 200.0

#: Short enough that the sketch holds every feature sample, which is the case
#: where the streamed answer must agree with the whole array to the last
#: decimal.
SHORT_S = 60.0


@pytest.fixture(scope="module")
def long_recording():
    return make_synthetic_recording(duration_s=LONG_S, seed=7, verbose=False)


@pytest.fixture(scope="module")
def short_recording():
    return make_synthetic_recording(duration_s=SHORT_S, seed=7, verbose=False)


def _whole_array(record, config):
    prep = prepare(record, config.preprocess, verbose=False)
    return HFO_DETECTORS["rms"](prep, config.rms, describe_spectrum=True)


def _stamps(events, detector="rms", places=9):
    return sorted((e.channel, round(float(e.start), places))
                  for e in events if e.detector == detector)


def _pair_up(left, right, tolerance=0.02):
    """Match two event lists one for one, returning the onset gaps.

    Raises if anything is left over on either side: "roughly the same events"
    is not a claim worth making, and a bijection is what the code promises.
    """
    pool = list(right)
    gaps = []
    for channel, when in left:
        hit = next((item for item in pool
                    if item[0] == channel and abs(item[1] - when) < tolerance),
                   None)
        assert hit is not None, f"no match for {channel} at {when:g} s"
        pool.remove(hit)
        gaps.append(abs(hit[1] - when))
    assert not pool, f"{len(pool)} streamed events with no counterpart"
    return gaps


# -- the headline ------------------------------------------------------------


@pytest.mark.parametrize("chunk_s", [30.0, 60.0, 100.0])
def test_the_hfo_events_do_not_depend_on_the_chunk_size(long_recording, chunk_s):
    """The whole point of `streaming`, now holding through per-chunk
    preprocessing as well as per-chunk detection.

    Byte-identical, not "close": a different chunk length must not move an
    onset by a sample, or the overlap is too short somewhere and the filter
    transient is inside somebody's core.
    """
    config = PipelineConfig()
    source = recording_source(long_recording)
    reference = analyse_span(source, 0.0, LONG_S, config, detectors=("rms",),
                             with_spikes=False, chunk_s=30.0,
                             overlap_s=DEFAULT_OVERLAP_S, check_quality=False)
    got = analyse_span(source, 0.0, LONG_S, config, detectors=("rms",),
                       with_spikes=False, chunk_s=chunk_s,
                       overlap_s=DEFAULT_OVERLAP_S, check_quality=False)
    assert _stamps(got.events) == _stamps(reference.events)
    assert got.chunks == pytest.approx(np.ceil(LONG_S / chunk_s))


def test_a_short_span_matches_the_whole_array_to_the_last_decimal(short_recording):
    """Where the baseline sketch holds every sample there is nothing left to
    differ: same events, same onsets, bar floating-point representation."""
    config = PipelineConfig()
    direct = _whole_array(short_recording, config)
    got = analyse_span(recording_source(short_recording), 0.0, SHORT_S, config,
                       detectors=("rms",), with_spikes=False, chunk_s=20.0,
                       overlap_s=DEFAULT_OVERLAP_S, check_quality=False)

    gaps = _pair_up(_stamps(direct), _stamps(got.events))
    assert len(gaps) == len(direct) > 0
    assert max(gaps) < 1e-6


def test_a_long_span_matches_the_whole_array_event_for_event(long_recording):
    """Past the sketch's cap the thresholds are drawn from a subsample, so an
    onset can land a sample or two out. No event is gained or lost, and the
    residue is bounded — and it is the sketch's, not the chunking's, which the
    chunk-size test above is what proves."""
    config = PipelineConfig()
    direct = _whole_array(long_recording, config)
    got = analyse_span(recording_source(long_recording), 0.0, LONG_S, config,
                       detectors=("rms",), with_spikes=False,
                       chunk_s=DEFAULT_CHUNK_S, overlap_s=DEFAULT_OVERLAP_S,
                       check_quality=False)

    assert len(_stamps(got.events)) == len(direct)
    gaps = _pair_up(_stamps(direct), _stamps(got.events))
    assert max(gaps) < 0.010         # measured 6.5 ms


def test_discharge_counts_move_a_little_with_the_chunk_size(long_recording):
    """The one thing that does move, measured rather than waved at.

    The spike detector takes its threshold from a robust scale of whatever it
    is given and has no baseline to inject, so its answer is chunk-dependent
    in a way the HFO detectors' is not. The module says so; this says by how
    much, and fails if that ever stops being a rounding error.
    """
    config = PipelineConfig()
    source = recording_source(long_recording)
    counts = []
    for chunk_s in (30.0, 60.0, 100.0):
        got = analyse_span(source, 0.0, LONG_S, config, detectors=("rms",),
                           with_spikes=True, chunk_s=chunk_s,
                           overlap_s=DEFAULT_OVERLAP_S, check_quality=False)
        counts.append(sum(1 for e in got.events if e.detector == "spike"))
    assert min(counts) > 0
    spread = (max(counts) - min(counts)) / max(counts)
    assert spread < 0.05, f"discharge counts moved {spread:.1%} with chunk size"


# -- what comes back ---------------------------------------------------------


def test_the_result_carries_no_signal(long_recording):
    """The point of the whole exercise: analyse an hour, hold none of it."""
    got = analyse_span(recording_source(long_recording), 0.0, LONG_S,
                       detectors=("rms",), with_spikes=False,
                       check_quality=False)
    assert isinstance(got, SpanAnalysis)
    assert not hasattr(got, "data")
    assert not hasattr(got, "raw")
    assert got.ch_names and got.sfreq > 0
    assert got.montage == "bipolar"
    assert got.duration == pytest.approx(LONG_S)


def test_the_steps_say_it_was_chunked_and_what_that_cost(long_recording):
    """A provenance list that did not mention the chunking would be a
    provenance list for a different analysis."""
    got = analyse_span(recording_source(long_recording), 0.0, LONG_S,
                       detectors=("rms",), with_spikes=True,
                       chunk_s=60.0, check_quality=False)
    joined = " ".join(got.steps)
    assert "chunk" in joined
    assert "do not depend on where the boundaries fell" in joined
    assert "per chunk" in joined          # the discharge caveat


def test_segments_cover_the_span_exactly_once(long_recording):
    """Per-second and concatenated from the cores, so the overlap is not
    counted twice and no second is missing."""
    got = analyse_span(recording_source(long_recording), 0.0, LONG_S,
                       detectors=("rms",), with_spikes=False, chunk_s=60.0,
                       check_quality=True)
    assert got.segments is not None and not got.segments.empty
    per_channel = got.segments.groupby("channel").size()
    assert set(per_channel) == {int(LONG_S)}
    for channel, group in got.segments.groupby("channel"):
        starts = sorted(float(value) for value in group["t_start"])
        assert len(set(starts)) == len(starts)
        assert starts[0] == pytest.approx(0.0)
        assert starts[-1] == pytest.approx(LONG_S - 1.0)


def test_the_quality_verdict_is_per_chunk_and_says_how_many(long_recording):
    """It is not a whole-span statistic and does not claim to be: a contact
    condemned by one chunk in seven is a different claim from one condemned by
    all seven, and `chunks_bad` is what lets a reviewer tell them apart."""
    got = analyse_span(recording_source(long_recording), 0.0, LONG_S,
                       detectors=("rms",), with_spikes=False, chunk_s=30.0,
                       check_quality=True)
    quality = got.quality
    assert quality is not None and not quality.empty
    assert {"channel", "reason", "good", "flagged", "chunks_bad",
            "n_chunks"} <= set(quality.columns)
    assert set(quality["n_chunks"]) == {got.chunks}
    assert (quality["chunks_bad"] <= quality["n_chunks"]).all()
    # A clean channel is clean, and the measurements survive as numbers.
    assert quality["good"].any()
    assert np.isfinite(quality["amplitude_uv"].to_numpy()).all()


def test_a_contact_bad_in_one_chunk_is_set_aside_for_the_span():
    """Conservative on purpose: a contact that was flat for a minute in ten is
    not one whose rate over the ten means anything."""
    import pandas as pd

    from onset_hfo.longrun import _merge_quality

    good = pd.DataFrame([{"channel": "A1-A2", "amplitude_uv": 40.0,
                          "good": True, "flagged": False, "reason": ""}])
    flat = pd.DataFrame([{"channel": "A1-A2", "amplitude_uv": 0.1,
                          "good": False, "flagged": False, "reason": "flat"}])
    merged = _merge_quality([good, good, flat, good])
    row = merged.iloc[0]
    assert row["reason"] == "flat"
    assert bool(row["good"]) is False
    assert int(row["chunks_bad"]) == 1
    assert int(row["n_chunks"]) == 4


def test_the_worst_reason_wins_when_chunks_disagree():
    import pandas as pd

    from onset_hfo.longrun import _merge_quality

    def frame(reason, good=True):
        return pd.DataFrame([{"channel": "A1-A2", "amplitude_uv": 40.0,
                              "good": good, "flagged": bool(reason) and good,
                              "reason": reason}])

    merged = _merge_quality([frame("amplitude"), frame("clipped", good=False),
                             frame("hf_active")])
    assert merged.iloc[0]["reason"] == "clipped"
    assert bool(merged.iloc[0]["good"]) is False


# -- refusals ----------------------------------------------------------------


def test_an_overlap_too_short_for_the_filter_is_refused(short_recording):
    """The refusal is the useful part: a boundary that corrupts the filter
    output is wrong in a way nothing downstream reports."""
    with pytest.raises(OverlapTooShort):
        analyse_span(recording_source(short_recording), 0.0, SHORT_S,
                     detectors=("rms",), with_spikes=False, chunk_s=20.0,
                     overlap_s=0.2, check_quality=False)


def test_an_empty_span_is_refused(short_recording):
    with pytest.raises(ValueError):
        analyse_span(recording_source(short_recording), 30.0, 30.0)


def test_progress_is_reported_in_order(long_recording):
    seen = []
    analyse_span(recording_source(long_recording), 0.0, LONG_S,
                 detectors=("rms",), with_spikes=False, chunk_s=60.0,
                 check_quality=False,
                 progress=lambda fraction, message: seen.append(fraction))
    assert seen
    assert seen == sorted(seen)
    assert 0.0 <= seen[0] and seen[-1] == pytest.approx(1.0)


def test_the_streaming_threshold_is_above_what_fits_comfortably():
    """A span that fits in memory should not pay for chunking, or for any of
    the caveats that come with it."""
    assert STREAM_ABOVE_S >= 120.0
    assert DEFAULT_CHUNK_S <= STREAM_ABOVE_S
