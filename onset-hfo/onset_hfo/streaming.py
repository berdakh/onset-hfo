"""Analyse a recording longer than memory, without letting the chunks show.

Roadmap item 7. ``run`` and ``benchmark`` took a slice because the whole
pipeline assumes the recording is an array. A minute is enough to demonstrate
a method and not enough to measure a patient: clinical HFO rates come from
ten-minute or hour-long interictal windows, and an hour of 2 kHz float64 on
60 channels is about 3.5 GB before a single filter runs.

**The hard part is not the memory, it is the threshold.** Every detector here
fires at ``median + k x robustSD`` *of the channel's own feature trace*. Cut
the recording into chunks and let each chunk measure its own median, and the
detector's answer starts depending on where the boundaries fell -- a busy
five minutes raises its own threshold and hides its own events, while a quiet
five minutes lowers it and invents them. This project has already published a
result that changed when the analysis window changed (``docs/OUTCOME.md``);
reintroducing the same defect as a scaling optimisation would be worse than
not scaling at all.

So this module makes two passes:

1. **Measure.** Stream every chunk, build a bounded subsample of each
   channel's feature trace, and compute one median and one MAD per channel
   *over the whole recording*.
2. **Detect.** Stream the chunks again with those fixed numbers injected, and
   keep each event exactly once.

Two passes cost roughly twice the filtering. That is the price of an answer
that does not depend on the chunk size, and the tests hold it to exactly
that: :func:`stream_pipeline` on a chunked recording returns the **same
events** as :func:`onset_hfo.pipeline.run_pipeline` on the whole array.

**What still is not here.** Per-channel parallelism (the detectors are
embarrassingly parallel and this runs them in a loop), and a chunk source that
does not cache each chunk to disk. Both are speed; neither changes an answer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from onset_hfo.config import DetectorConfig, PipelineConfig
from onset_hfo.detectors.base import MAD_TO_SD, ChannelBaseline, Event, bandpass
from onset_hfo.pipeline import HFO_DETECTORS
from onset_hfo.preprocess import Prepared

__all__ = [
    "ChunkPlan",
    "plan_chunks",
    "measure_baselines",
    "stream_detect",
    "OverlapTooShort",
]

#: Feature samples kept per channel when measuring a baseline. 200k samples
#: pin a median to about a tenth of a percentile at this project's sampling
#: rates, and cost 1.6 MB per channel -- flat in the recording's length, which
#: is the whole point. Raising it improves the estimate and the memory
#: proportionally; it is a constant rather than a config knob because no
#: result should ever turn on it.
SUBSAMPLE_PER_CHANNEL = 200_000

#: Seconds of chunk kept on each side purely to be thrown away. It has to
#: cover the FIR filter's ring-in *and* the longest event the detector will
#: accept, or an event straddling a boundary is seen by neither chunk whole.
DEFAULT_OVERLAP_S = 2.0


class OverlapTooShort(ValueError):
    """The requested overlap cannot hide the chunk boundary.

    Raised rather than warned about: a too-short overlap does not fail
    loudly, it quietly returns slightly wrong events near every boundary,
    which is the kind of error that reaches a paper.
    """


@dataclass(frozen=True)
class ChunkPlan:
    """Where one chunk starts, where it ends, and which part of it counts.

    ``read_start``/``read_stop`` are what gets loaded and filtered.
    ``core_start``/``core_stop`` are the part this chunk is responsible for:
    an event is kept by the chunk whose core contains its **start**, so the
    overlap never yields the same event twice.
    """

    index: int
    read_start: float
    read_stop: float
    core_start: float
    core_stop: float

    @property
    def read_duration(self) -> float:
        return self.read_stop - self.read_start

    def owns(self, event_start: float) -> bool:
        return self.core_start <= event_start < self.core_stop


def plan_chunks(t_start: float, t_stop: float, chunk_s: float,
                overlap_s: float = DEFAULT_OVERLAP_S,
                max_event_ms: float = 500.0,
                sfreq: float | None = None,
                band: tuple[float, float] | None = None) -> list[ChunkPlan]:
    """Cut ``[t_start, t_stop)`` into overlapping chunks, or refuse to.

    The refusal is the useful part. ``overlap_s`` must exceed both the longest
    event the detector will accept and the band-pass filter's impulse
    response, because a chunk edge corrupts the filter output for about that
    long and an event that straddles the boundary must be wholly inside
    somebody's read window.
    """
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")

    longest_event = max_event_ms / 1000.0
    ring_in = 0.0
    if sfreq is not None and band is not None:
        # MNE's firwin design: the transition band sets the filter length, and
        # the lower edge is the narrow one. This mirrors its own rule of thumb
        # rather than guessing.
        low = max(band[0], 1e-6)
        transition = max(min(low, 2.0), 0.1 * low)
        ring_in = int(round(3.3 * sfreq / transition)) / sfreq
    needed = max(longest_event, ring_in)
    if overlap_s < needed:
        why = ("the band-pass filter's ring-in" if ring_in > longest_event
               else "the longest acceptable event")
        raise OverlapTooShort(
            f"overlap {overlap_s:g} s is shorter than the {needed:.2f} s this "
            f"recording needs for {why}. Raise overlap_s, or the events near "
            f"every chunk boundary will be wrong in a way nothing reports.")
    if chunk_s <= 2 * overlap_s:
        raise OverlapTooShort(
            f"chunk_s {chunk_s:g} s leaves no core after {overlap_s:g} s of "
            f"overlap on each side; use a longer chunk.")

    plans: list[ChunkPlan] = []
    n = max(1, math.ceil((t_stop - t_start) / chunk_s))
    for i in range(n):
        core_start = t_start + i * chunk_s
        core_stop = min(t_stop, core_start + chunk_s)
        if core_start >= core_stop:
            break
        plans.append(ChunkPlan(
            index=i,
            read_start=max(t_start, core_start - overlap_s),
            read_stop=min(t_stop, core_stop + overlap_s),
            core_start=core_start,
            core_stop=core_stop,
        ))
    return plans


@dataclass
class _Sketch:
    """A bounded subsample of one channel's feature values, over the whole run.

    The retained set is exactly *the samples whose index from the start of the
    recording is a multiple of ``stride``*, and ``stride`` only ever doubles.
    That definition is what makes the baseline independent of the chunking: it
    is a property of absolute position in the recording, so cutting the same
    recording into 15-second or 60-second chunks retains the same samples and
    yields the same median to the last bit.

    The first version of this class chose a stride per chunk from the chunk's
    own size. It was bounded and deterministic and still wrong: a 60-second
    chunking thinned four times as hard as a 15-second one, moved the median
    slightly, and moved three events out of 460 across the threshold with it.
    Small, but it made the chunk size a parameter of the result, which is the
    exact defect this module exists to avoid.

    Deterministic striding rather than reservoir sampling, because the same
    recording must give the same threshold on every run and a seeded RNG is
    one more thing to get wrong.
    """

    cap: int = SUBSAMPLE_PER_CHANNEL
    stride: int = 1
    kept: list[np.ndarray] = field(default_factory=list)
    count: int = 0
    seen: int = 0

    def add(self, values: np.ndarray, first_index: int) -> None:
        """Take ``values``, which start at ``first_index`` samples into the run."""
        self.seen += values.size
        offset = (-first_index) % self.stride
        if offset < values.size:
            take = np.ascontiguousarray(values[offset::self.stride])
            self.kept.append(take)
            self.count += take.size
        while self.count > self.cap:
            self._halve()

    def _halve(self) -> None:
        # Everything kept sits at indices 0, s, 2s, ... in order, so dropping
        # every other element leaves 0, 2s, 4s, ... -- the same rule at twice
        # the stride, with no dependence on where the chunk edges were.
        merged = np.concatenate(self.kept) if self.kept else np.empty(0)
        merged = np.ascontiguousarray(merged[::2])
        self.kept = [merged]
        self.count = merged.size
        self.stride *= 2

    def robust(self) -> tuple[float, float]:
        if not self.kept:
            return 0.0, float(np.finfo(float).eps)
        values = np.concatenate(self.kept)
        centre = float(np.median(values))
        scale = MAD_TO_SD * float(np.median(np.abs(values - centre)))
        return centre, max(scale, float(np.finfo(float).eps))


def _feature_of(name: str) -> Callable[[np.ndarray, int], np.ndarray]:
    from onset_hfo.detectors.base import (
        sliding_energy,
        sliding_hilbert_envelope,
        sliding_line_length,
        sliding_rms,
    )

    return {"rms": sliding_rms, "line_length": sliding_line_length,
            "hilbert": sliding_hilbert_envelope,
            "short_time_energy": sliding_energy}[name]


def measure_baselines(chunks: Iterable[tuple[ChunkPlan, Prepared]],
                      config: PipelineConfig,
                      detectors: tuple[str, ...] = ("rms", "line_length"),
                      verbose: bool = True,
                      ) -> dict[str, dict[str, ChannelBaseline]]:
    """Pass one: one median and one MAD per channel, over the whole recording.

    Only each chunk's **core** contributes, so the overlap is not counted
    twice and a chunk boundary cannot tilt the estimate.
    """
    feature_sketch: dict[tuple[str, str], _Sketch] = {}
    amplitude_sketch: dict[tuple[str, str], _Sketch] = {}
    n_chunks = 0
    origin: float | None = None

    for plan, prep in chunks:
        n_chunks += 1
        if origin is None:
            origin = plan.core_start
        # Where this chunk's core begins, counted from the start of the whole
        # recording. The sketches index off this, not off the chunk, which is
        # what makes them chunk-size independent.
        first_index = int(round((plan.core_start - origin) * prep.sfreq))
        lo = int(round((plan.core_start - plan.read_start) * prep.sfreq))
        hi = lo + int(round((plan.core_stop - plan.core_start) * prep.sfreq))
        hi = min(hi, prep.data.shape[1])
        if hi <= lo:
            continue
        for det in detectors:
            cfg: DetectorConfig = getattr(config, det)
            filtered = bandpass(prep.data, prep.sfreq, cfg.band)
            win = max(1, int(round(cfg.rms_window_ms * prep.sfreq / 1000.0)))
            feature = _feature_of(det)
            for row, ch in enumerate(prep.ch_names):
                trace = feature(filtered[row], win)[lo:hi]
                feature_sketch.setdefault((det, ch), _Sketch()).add(trace, first_index)
                # The band-passed signal itself, not its magnitude: the
                # amplitude scale a detector uses is robust_scale(x)[1], which
                # is the MAD *about x's own median*, and |x| cannot recover
                # that median.
                amplitude_sketch.setdefault((det, ch), _Sketch()).add(
                    filtered[row][lo:hi], first_index)

    out: dict[str, dict[str, ChannelBaseline]] = {det: {} for det in detectors}
    for (det, ch), sketch in feature_sketch.items():
        centre, scale = sketch.robust()
        _, amplitude = amplitude_sketch[(det, ch)].robust()
        out[det][ch] = ChannelBaseline(feature_center=centre, feature_scale=scale,
                                       amplitude_scale=amplitude)
    if verbose:
        channels = len(next(iter(out.values()))) if out else 0
        print(f"[onset-hfo] baselines measured over {n_chunks} chunks, "
              f"{channels} channels, {len(detectors)} detectors")
    return out


def stream_detect(chunks: Callable[[], Iterator[tuple[ChunkPlan, Prepared]]],
                  config: PipelineConfig | None = None,
                  detectors: tuple[str, ...] = ("rms", "line_length"),
                  describe_spectrum: bool = True,
                  verbose: bool = True) -> dict[str, list[Event]]:
    """Detect over a chunked recording as if it had been one array.

    ``chunks`` is a **factory**, not an iterator: this makes two passes over
    the recording and the second one needs the chunks again.
    """
    cfg = config or PipelineConfig()
    baselines = measure_baselines(chunks(), cfg, detectors, verbose=verbose)

    found: dict[str, list[Event]] = {det: [] for det in detectors}
    for plan, prep in chunks():
        for det in detectors:
            missing = [c for c in prep.ch_names if c not in baselines[det]]
            if missing:
                raise RuntimeError(
                    f"no baseline for {len(missing)} channel(s) in chunk "
                    f"{plan.index} ({', '.join(missing[:3])}...). The measure "
                    f"pass and the detect pass saw different channels, which "
                    f"means the thresholds do not describe this data.")
            events = HFO_DETECTORS[det](prep, getattr(cfg, det),
                                        describe_spectrum=describe_spectrum,
                                        baselines=baselines[det])
            found[det].extend(e for e in events if plan.owns(e.start))
    for det in found:
        found[det].sort(key=lambda e: (e.channel, e.start))
    if verbose:
        for det, evs in found.items():
            print(f"[onset-hfo] {det}: {len(evs)} events across the whole recording")
    return found
