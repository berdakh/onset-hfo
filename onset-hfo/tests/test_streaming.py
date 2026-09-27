"""Analysing a long recording in chunks must not change the answer.

Roadmap item 7. The reason this module exists is not memory, it is the
threshold: every detector fires at ``median + k x robustSD`` of the channel's
own feature trace, so a chunk that measures its own median makes the
detector's output depend on where the boundaries fell. These tests hold the
streaming path to three claims, in order of how much they matter:

1. With a baseline that sees every sample, streaming returns **exactly** the
   events the whole-array pipeline returns, at every chunk size.
2. Above that, the result is still **exactly invariant to chunk size** -- the
   subsample is defined by absolute position in the recording, not by the
   chunk.
3. The naive alternative -- let each chunk measure its own baseline -- is
   measurably wrong, which is what buys the second pass.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from onset_hfo.config import PipelineConfig
from onset_hfo.detectors.base import ChannelBaseline
from onset_hfo.pipeline import HFO_DETECTORS, run_pipeline
from onset_hfo.preprocess import prepare
from onset_hfo.streaming import (
    SUBSAMPLE_PER_CHANNEL,
    OverlapTooShort,
    measure_baselines,
    plan_chunks,
    stream_detect,
)
from onset_hfo.synthetic import make_synthetic_recording

DETECTORS = ("rms", "line_length")


def _chunk_factory(prep, chunk_s: float, overlap_s: float = 2.0):
    """Serve `prep` in chunks, standing in for a loader that reads byte ranges.

    The real source is ``datasets.iter_slices``; this one needs no network and
    exercises exactly the same contract.
    """
    def chunks():
        for plan in plan_chunks(0.0, prep.duration, chunk_s, overlap_s):
            lo = int(round(plan.read_start * prep.sfreq))
            hi = int(round(plan.read_stop * prep.sfreq))
            sub = copy.copy(prep)
            sub.data = np.ascontiguousarray(prep.data[:, lo:hi])
            sub.t_offset = prep.t_offset + plan.read_start
            yield plan, sub
    return chunks


def _fingerprint(events) -> list[tuple[str, float]]:
    return sorted((e.channel, round(e.start, 4)) for e in events)


@pytest.fixture(scope="module")
def short():
    """60 s at 2 kHz: 120,000 samples per channel, under the subsample cap.

    Chosen so the baseline keeps *every* sample, which is what makes exact
    equality with the whole-array path a meaningful assertion rather than a
    coincidence.
    """
    rec = make_synthetic_recording(duration_s=60.0, seed=3, verbose=False)
    prep = prepare(rec, verbose=False)
    assert prep.data.shape[1] < SUBSAMPLE_PER_CHANNEL, \
        "this fixture is only meaningful below the subsample cap"
    whole = run_pipeline(rec, detectors=DETECTORS, with_spikes=False, verbose=False)
    return prep, {d: _fingerprint(e) for d, e in whole.events.items()}


# --------------------------------------------------------------------------
# 1. The claim that matters most
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_s", [8.0, 10.0, 15.0, 20.0, 30.0])
def test_streaming_returns_exactly_what_the_whole_array_returns(short, chunk_s):
    prep, reference = short
    got = stream_detect(_chunk_factory(prep, chunk_s), detectors=DETECTORS,
                        describe_spectrum=False, verbose=False)
    for det in DETECTORS:
        assert _fingerprint(got[det]) == reference[det], \
            f"{det} disagrees with the in-memory pipeline at {chunk_s:g} s chunks"


def test_no_event_is_reported_twice_from_the_overlap(short):
    """Every chunk reads into its neighbours; only one of them may keep an event."""
    prep, _ = short
    got = stream_detect(_chunk_factory(prep, 10.0), detectors=("rms",),
                        describe_spectrum=False, verbose=False)
    keys = [(e.channel, round(e.start, 6)) for e in got["rms"]]
    assert len(keys) == len(set(keys)), "an event was kept by two chunks"


def test_events_that_straddle_a_boundary_survive_whole(short):
    """An event crossing a chunk edge must appear once, with its full duration."""
    prep, reference = short
    chunk_s = 10.0
    got = stream_detect(_chunk_factory(prep, chunk_s), detectors=("rms",),
                        describe_spectrum=False, verbose=False)
    whole = run_pipeline(prep.recording, detectors=("rms",), with_spikes=False,
                         verbose=False).events["rms"]
    by_key = {(e.channel, round(e.start, 4)): e for e in whole}
    straddlers = [e for e in got["rms"]
                  if any(e.start < b < e.stop for b in
                         np.arange(chunk_s, prep.duration, chunk_s))]
    assert straddlers, "the fixture has no boundary-crossing event to check"
    for e in straddlers:
        twin = by_key[(e.channel, round(e.start, 4))]
        assert e.stop == pytest.approx(twin.stop), "a boundary truncated an event"


# --------------------------------------------------------------------------
# 2. Invariance above the cap
# --------------------------------------------------------------------------


def test_the_chunk_size_is_not_a_parameter_of_the_result():
    """Long enough to force subsampling; every chunking must still agree.

    240,000 samples per channel against a 200,000 cap, so the baseline thins
    to a stride of 2. The retained samples are chosen by absolute position, so
    thinning cannot depend on where the chunks were cut -- and this is the
    test that failed before that was true, by three events out of 460.
    """
    rec = make_synthetic_recording(duration_s=120.0, seed=3, verbose=False)
    prep = prepare(rec, verbose=False)
    assert prep.data.shape[1] > SUBSAMPLE_PER_CHANNEL

    results = {}
    for chunk_s in (10.0, 15.0, 20.0, 30.0, 60.0):
        got = stream_detect(_chunk_factory(prep, chunk_s), detectors=DETECTORS,
                            describe_spectrum=False, verbose=False)
        results[chunk_s] = {d: _fingerprint(e) for d, e in got.items()}

    first = results[10.0]
    for chunk_s, result in results.items():
        for det in DETECTORS:
            assert result[det] == first[det], \
                f"{det} at {chunk_s:g} s chunks differs from 10 s chunks"


def test_subsampling_costs_under_a_percent_against_the_exhaustive_baseline():
    """What the bounded memory actually costs, stated rather than assumed."""
    rec = make_synthetic_recording(duration_s=120.0, seed=3, verbose=False)
    prep = prepare(rec, verbose=False)
    whole = run_pipeline(rec, detectors=("rms",), with_spikes=False, verbose=False)
    got = stream_detect(_chunk_factory(prep, 20.0), detectors=("rms",),
                        describe_spectrum=False, verbose=False)
    a, b = set(_fingerprint(whole.events["rms"])), set(_fingerprint(got["rms"]))
    disagreement = len(a ^ b) / max(len(a), 1)
    assert disagreement < 0.01, f"subsampled baseline moved {disagreement:.1%} of events"


# --------------------------------------------------------------------------
# 3. Why the second pass is worth its cost
# --------------------------------------------------------------------------


def test_a_per_chunk_baseline_would_be_materially_wrong(short):
    """The measurement that justifies measuring baselines over the whole run.

    Letting each chunk threshold against its own median is the obvious cheap
    implementation. It is also wrong by tens of percent: a busy chunk raises
    its own threshold and hides its own events, a quiet one lowers it and
    invents them.
    """
    prep, reference = short
    cfg = PipelineConfig()
    naive = []
    for plan in plan_chunks(0.0, prep.duration, 10.0, 2.0):
        lo = int(round(plan.read_start * prep.sfreq))
        hi = int(round(plan.read_stop * prep.sfreq))
        sub = copy.copy(prep)
        sub.data = np.ascontiguousarray(prep.data[:, lo:hi])
        sub.t_offset = prep.t_offset + plan.read_start
        naive.extend(e for e in HFO_DETECTORS["rms"](sub, cfg.rms, describe_spectrum=False)
                     if plan.owns(e.start))

    truth = set(reference["rms"])
    got = set(_fingerprint(naive))
    wrong = len(truth ^ got) / len(truth)
    assert wrong > 0.10, (
        "a per-chunk baseline agreed with the whole-recording answer, which "
        "would mean this module's second pass is unnecessary -- check before "
        "deleting it")


# --------------------------------------------------------------------------
# The plan, and its refusals
# --------------------------------------------------------------------------


def test_chunks_tile_the_window_with_no_gap_and_no_overlap_in_their_cores():
    plans = plan_chunks(0.0, 100.0, chunk_s=30.0, overlap_s=2.0)
    assert plans[0].core_start == 0.0 and plans[-1].core_stop == 100.0
    for a, b in zip(plans, plans[1:], strict=False):
        assert a.core_stop == b.core_start, "cores must tile exactly"
    assert all(p.read_start <= p.core_start and p.read_stop >= p.core_stop for p in plans)


def test_every_time_in_the_window_is_owned_by_exactly_one_chunk():
    plans = plan_chunks(0.0, 100.0, chunk_s=30.0, overlap_s=2.0)
    for t in np.linspace(0.0, 99.999, 500):
        assert sum(p.owns(float(t)) for p in plans) == 1


def test_an_overlap_shorter_than_the_longest_event_is_refused():
    with pytest.raises(OverlapTooShort, match="longest acceptable event"):
        plan_chunks(0.0, 100.0, chunk_s=30.0, overlap_s=0.1, max_event_ms=500.0)


def test_a_ripple_band_overlap_of_two_seconds_is_accepted():
    """The shipped default is not merely plausible; it clears both bounds."""
    plans = plan_chunks(0.0, 600.0, chunk_s=60.0, overlap_s=2.0,
                        sfreq=2000.0, band=(80.0, 250.0))
    assert len(plans) == 10


def test_an_overlap_shorter_than_the_filter_ring_in_is_refused():
    """A low band edge means a long filter, and the plan works that out itself.

    At a 1 Hz lower edge and 2 kHz the FIR runs to about 3.3 s, far longer
    than any event, so an overlap that would be generous for ripples is not
    nearly enough here. The refusal names which of the two bounds it was.
    """
    with pytest.raises(OverlapTooShort, match="ring-in"):
        plan_chunks(0.0, 600.0, chunk_s=60.0, overlap_s=1.0,
                    sfreq=2000.0, band=(1.0, 40.0))


def test_a_chunk_with_no_core_left_is_refused():
    with pytest.raises(OverlapTooShort, match="no core"):
        plan_chunks(0.0, 100.0, chunk_s=3.0, overlap_s=2.0)


def test_the_window_must_run_forwards():
    with pytest.raises(ValueError):
        plan_chunks(10.0, 5.0, chunk_s=1.0)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_baselines_cover_every_channel_and_detector(short):
    prep, _ = short
    baselines = measure_baselines(_chunk_factory(prep, 15.0)(), PipelineConfig(),
                                  DETECTORS, verbose=False)
    assert set(baselines) == set(DETECTORS)
    for det in DETECTORS:
        assert set(baselines[det]) == set(prep.ch_names)
        for base in baselines[det].values():
            assert isinstance(base, ChannelBaseline)
            assert base.feature_scale > 0 and base.amplitude_scale > 0


def test_a_streamed_baseline_matches_the_one_measured_on_the_whole_array(short):
    """Below the cap this is an identity, not an approximation."""
    from onset_hfo.detectors.base import bandpass, robust_scale, sliding_rms

    prep, _ = short
    cfg = PipelineConfig()
    baselines = measure_baselines(_chunk_factory(prep, 15.0)(), cfg, ("rms",),
                                  verbose=False)["rms"]
    filtered = bandpass(prep.data, prep.sfreq, cfg.rms.band)
    win = max(1, int(round(cfg.rms.rms_window_ms * prep.sfreq / 1000.0)))
    for row, ch in enumerate(prep.ch_names):
        centre, scale = robust_scale(sliding_rms(filtered[row], win))
        assert baselines[ch].feature_center == pytest.approx(float(np.squeeze(centre)))
        assert baselines[ch].feature_scale == pytest.approx(float(np.squeeze(scale)))


def test_a_missing_baseline_is_an_error_rather_than_a_silent_fallback(short):
    """The real hazard: a chunk source that is not the same on both passes.

    ``stream_detect`` reads the recording twice. If the second pass turns up
    channels the first never measured, the honest options are to fail or to
    let those channels threshold against their own chunk -- and the second is
    precisely the bug this module exists to prevent, so it fails.
    """
    prep, _ = short
    factory = _chunk_factory(prep, 15.0)
    calls = {"n": 0}

    def inconsistent():
        calls["n"] += 1
        first_pass = calls["n"] == 1
        for plan, chunk in factory():
            if first_pass:
                chunk = copy.copy(chunk)
                chunk.ch_names = chunk.ch_names[:2]
                chunk.data = chunk.data[:2]
            yield plan, chunk

    with pytest.raises(RuntimeError, match="no baseline"):
        stream_detect(inconsistent, detectors=("rms",), describe_spectrum=False,
                      verbose=False)


def test_memory_does_not_grow_with_the_recording():
    """The sketch is capped, so a longer recording keeps the same buffer."""
    from onset_hfo.streaming import _Sketch

    sketch = _Sketch()
    rng = np.random.default_rng(0)
    for block in range(40):
        sketch.add(rng.normal(size=50_000), first_index=block * 50_000)
        assert sketch.count <= SUBSAMPLE_PER_CHANNEL
    assert sketch.seen == 2_000_000
    assert sketch.stride > 1, "a 2M-sample channel should have been thinned"


def test_thinning_keeps_samples_by_absolute_position_not_by_chunk():
    """The property that makes the chunk size irrelevant, asserted directly."""
    from onset_hfo.streaming import _Sketch

    values = np.arange(1_000_000, dtype=np.float64)
    coarse = _Sketch()
    coarse.add(values, first_index=0)

    fine = _Sketch()
    for start in range(0, values.size, 50_000):
        fine.add(values[start:start + 50_000], first_index=start)

    assert coarse.stride == fine.stride
    assert np.array_equal(np.concatenate(coarse.kept), np.concatenate(fine.kept))


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_the_stream_command_is_wired_and_names_its_window():
    """`--stop` is required: a chunked run with no end is a download loop."""
    from onset_hfo.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["stream", "--stop", "600", "--chunk", "60"])
    assert args.func.__name__ == "_cmd_stream"
    assert (args.stop, args.chunk, args.overlap) == (600.0, 60.0, 2.0)

    with pytest.raises(SystemExit):
        parser.parse_args(["stream", "--chunk", "60"])


def test_the_stream_command_refuses_an_impossible_plan_before_downloading(capsys):
    """The refusal must come before the first byte, not after an hour of them."""
    from onset_hfo.cli import build_parser

    args = build_parser().parse_args(
        ["stream", "--stop", "600", "--chunk", "60", "--overlap", "0.05"])
    assert args.func(args) == 2
    assert "overlap" in capsys.readouterr().out
