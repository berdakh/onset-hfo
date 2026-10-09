"""Analyse ten minutes, or an hour, without holding it in memory.

:mod:`onset_hfo.streaming` has had the hard part since roadmap item 7 -- a
two-pass scheme that measures every detector threshold over the *whole*
recording, so the answer does not depend on where the chunk boundaries fell --
and until now nothing in this project called it. The library could analyse an
hour; the products could not. This module is the missing half: the thing that
turns a source of signal and a span into chunks, runs them through
`stream_detect`, and puts the quality stage back on top.

Why it matters is arithmetic. A minute of 2 kHz float64 on 18 bipolar channels
is 17 MB and fits anywhere. An hour on 64 is 3.7 GB before a single filter
runs, and the review window keeps a second copy for the trace. Clinical HFO
rates are quoted from ten-minute or hour-long interictal windows, so a tool
that can only analyse a minute is a tool that can only demonstrate a method.

**Three things here are not obvious and all three are correctness, not speed.**

*The overlap has to cover the preprocessing, not just the detector.*
`plan_chunks` sizes its overlap against the band-pass filter's ring-in and the
longest acceptable event, which is right for the chunks its own tests feed it
-- those are slices of an array that was prepared once, as a whole. Prepare
each chunk separately, as anything reading from a real source must, and the
longest filter in the system is no longer the 80 Hz band-pass but the 1 Hz
high-pass, whose impulse response is about 3.3 s at 2 kHz. An overlap sized
for the detector would leave several seconds of filter transient at every
boundary, inside the core, silently wrong. Hence `DEFAULT_OVERLAP_S` here,
which is not `streaming.DEFAULT_OVERLAP_S`, and a test that holds the streamed
answer to the whole-array answer *exactly* rather than approximately.

*The quality stage is per chunk, and says so.* `channel_quality` measures a
contact against the montage around it over whatever it is given. There is no
honest way to merge ten chunks' medians into one whole-span median without
re-deriving the whole stage as a streaming computation, so this does not
pretend to: each chunk is judged, and a contact set aside in any chunk is set
aside for the span. That is deliberately conservative -- a contact that was
flat or clipped for one minute in ten is not one whose rate over the ten means
anything -- and the fraction of chunks that condemned it is reported so the
reviewer can disagree. Segments are exact: they are per-second and simply
concatenated.

*Discharge thresholds are per chunk.* The spike detector takes its threshold
from a robust scale of the data it is given and has no baseline parameter, so
unlike the HFO detectors its answer does move with the chunk size. Rather than
hide that, the chunk length is fixed by default so a given version is
reproducible, the step list says it out loud, and the number is measured
rather than waved at: over the same 200 s, chunks of 30 s, 60 s and 100 s give
597, 600 and 601 discharges — a spread of 0.7%. Small, real, and now quotable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from onset_hfo.config import BANDS, PipelineConfig
from onset_hfo.detectors import DETECTORS
from onset_hfo.detectors.base import Event
from onset_hfo.preprocess import prepare
from onset_hfo.quality import channel_quality, segment_quality
from onset_hfo.streaming import plan_chunks, stream_detect
from onset_hfo.validate import flag_spike_cooccurrence, validate_events

__all__ = ["SpanAnalysis", "analyse_span", "DEFAULT_CHUNK_S",
           "DEFAULT_OVERLAP_S", "STREAM_ABOVE_S", "Source"]

#: A thing that can produce a `Recording` for an arbitrary stretch of time.
#: Everything in this project that holds signal can be one in three lines:
#: `fetch_slice` for the archive, `open_recording` for a local file, a closure
#: over an in-memory `Raw` for the synthetic generator.
Source = Callable[[float, float], object]

#: Seconds of signal per chunk. Two minutes on 64 channels at 2 kHz is about
#: 120 MB prepared -- comfortable on any machine this runs on, and long enough
#: that the per-chunk discharge threshold is measured over a sensible stretch
#: of recording rather than a sample of one. Fixed rather than tuned, because
#: it is the one parameter that moves the discharge count (see the module
#: docstring) and a result that depends on a knob nobody set is not a result.
DEFAULT_CHUNK_S = 120.0

#: Seconds read on each side of a chunk and thrown away. It has to cover the
#: longest filter in the chain, which is the 1 Hz high-pass in `prepare`, not
#: the band-pass the detector uses: at 2 kHz that FIR runs about 3.3 s each
#: side. Eight seconds is that with room for a lower high-pass or a lower
#: sampling rate, and costs 13% of a two-minute chunk read twice.
DEFAULT_OVERLAP_S = 8.0

#: Above this many seconds, analyse by streaming. Below it, the whole span
#: fits in memory comfortably and the direct path is both simpler and free of
#: every caveat above, so there is no reason to pay for chunking.
STREAM_ABOVE_S = 180.0

#: Worst first. `channel_quality` picks one reason per channel in a fixed
#: order of seriousness; this is that order, used to merge the per-chunk
#: verdicts into one. The first four set a contact aside; the rest flag it.
REASON_SEVERITY = ("flat", "clipped", "line_noise", "mostly_bad_segments",
                   "hf_noise", "hf_active", "amplitude")


@dataclass
class SpanAnalysis:
    """Everything a review of a long span needs, with the signal let go."""

    events: list = field(default_factory=list)
    ch_names: list = field(default_factory=list)
    sfreq: float = 0.0
    montage: str = ""
    line_freq: float = 0.0
    t_start: float = 0.0
    t_stop: float = 0.0
    segments: pd.DataFrame | None = None
    quality: pd.DataFrame | None = None
    steps: list = field(default_factory=list)
    chunks: int = 0

    @property
    def duration(self) -> float:
        return self.t_stop - self.t_start


def analyse_span(source: Source, t_start: float, t_stop: float,
                 config: PipelineConfig | None = None,
                 detectors: tuple = ("rms",),
                 with_spikes: bool = True,
                 chunk_s: float = DEFAULT_CHUNK_S,
                 overlap_s: float = DEFAULT_OVERLAP_S,
                 check_quality: bool = True,
                 progress: Callable | None = None,
                 collect: Callable | None = None,
                 verbose: bool = False,
                 positions: dict | None = None,
                 positions_from: str = "") -> SpanAnalysis:
    """Detect over `[t_start, t_stop)` in chunks, holding one chunk at a time.

    `source(read_start, read_stop)` returns a `Recording` for that stretch.
    It is called twice per chunk, because `stream_detect` makes two passes and
    the second one needs the signal again -- the alternative is caching every
    prepared chunk, which is the memory this exists to avoid.

    `positions` and `positions_from` go to `prepare`, for a Laplacian whose
    neighbours come from a reader's electrode file.

    `collect(record, plan)`, if given, is called once per chunk on the third
    pass with the unprepared `Recording`. It is the seam for anything that
    lives on the recording rather than in the signal -- expert markings above
    all, which a caller comparing a span against them has to gather chunk by
    chunk or else compare ten minutes of detections against one minute of
    annotations and call the difference a false positive rate.

    The result carries no signal. That is the point: a caller can analyse an
    hour and then load one minute of it to look at.
    """
    cfg = config or PipelineConfig()
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")

    plans = plan_chunks(t_start, t_stop, chunk_s, overlap_s)
    say = progress or (lambda fraction, message: None)
    total = max(1, len(plans))

    # Everything the chunk factory learns on the way past, so that the span's
    # metadata does not need a separate load to find out the sampling rate.
    seen: dict = {}
    per_chunk_quality: list = []
    segments: list = []

    def prepared(plan, keep_record: bool = False):
        record = source(plan.read_start, plan.read_stop)
        if keep_record and collect is not None:
            collect(record, plan)
        prep = prepare(record, cfg.preprocess, verbose=False, positions=positions,
                       positions_from=positions_from)
        if not seen:
            seen["sfreq"] = float(prep.sfreq)
            seen["ch_names"] = list(prep.ch_names)
            seen["montage"] = prep.montage
            seen["line_freq"] = float(prep.line_freq)
            seen["steps"] = list(prep.steps)
            # Now that the sampling rate is known, let `plan_chunks` do the
            # arithmetic it could not do before the first chunk existed. It
            # raises rather than returning, which is what should happen: a
            # boundary that corrupts the filter output is wrong in a way
            # nothing downstream reports.
            plan_chunks(t_start, t_stop, chunk_s, overlap_s,
                        sfreq=prep.sfreq, band=_widest_band(cfg, detectors))
        return prep

    # `stream_detect` calls this factory twice -- once to measure the
    # thresholds and once to detect with them -- so a progress bar driven off
    # the chunk index alone runs to 80% and then starts again. Counting the
    # passes keeps it monotonic, which is the only thing a progress bar owes
    # anybody.
    passes = {"n": 0}

    def chunks():
        passes["n"] += 1
        first, last = ((0.05, 0.40) if passes["n"] == 1 else (0.40, 0.80))
        label = ("Measuring thresholds over the whole span" if passes["n"] == 1
                 else "Detecting")
        for index, plan in enumerate(plans):
            say(first + (last - first) * index / total,
                f"{label} — chunk {index + 1} of {len(plans)}, "
                f"{plan.core_start:g}–{plan.core_stop:g} s")
            yield plan, prepared(plan)

    found = stream_detect(chunks, config=cfg, detectors=tuple(detectors),
                          verbose=verbose)
    events: list[Event] = [e for name in detectors for e in found.get(name, [])]

    # A third pass for everything `stream_detect` does not do: validation,
    # discharges, and the quality stage. All three need the signal, so they
    # share one read of it rather than taking one each.
    #
    # Validation is per chunk because it reads the signal around each event,
    # and each event is validated by the chunk that owns it -- the same rule
    # `stream_detect` uses to keep an event exactly once, so an event near a
    # boundary is judged by the chunk that has the whole of it.
    say(0.82, "Discharges, validation and data quality")
    by_chunk: dict = {}
    for event in events:
        for plan in plans:
            if plan.owns(event.start):
                by_chunk.setdefault(plan.index, []).append(event)
                break

    for index, plan in enumerate(plans):
        say(0.82 + 0.1 * index / total,
            f"Checking chunk {index + 1} of {len(plans)}")
        prep = prepared(plan, keep_record=True)
        mine = by_chunk.get(plan.index, [])
        validate_events(mine, prep, cfg.validation)
        if with_spikes:
            spikes = [e for e in DETECTORS["spike"](prep, cfg.spikes)
                      if plan.owns(e.start)]
            validate_events(spikes, prep, cfg.validation)
            flag_spike_cooccurrence(mine, spikes)
            events.extend(spikes)
        if check_quality:
            chunk_segments = segment_quality(prep, cfg.quality)
            if not chunk_segments.empty:
                core = chunk_segments[
                    (chunk_segments["t_start"] >= plan.core_start)
                    & (chunk_segments["t_start"] < plan.core_stop)]
                segments.append(core)
            per_chunk_quality.append(
                channel_quality(prep, cfg.quality, segments=chunk_segments))

    events.sort(key=lambda e: (e.start, e.channel))

    steps = list(seen.get("steps", []))
    steps.append(
        f"analysed in {len(plans)} chunk(s) of {chunk_s:g} s with "
        f"{overlap_s:g} s discarded on each side; detector thresholds were "
        f"measured once over the whole {t_stop - t_start:g} s, so they do not "
        f"depend on where the boundaries fell")
    if with_spikes:
        steps.append(
            "discharge thresholds, unlike the HFO detectors', were measured "
            "per chunk — the spike detector takes its scale from the data it "
            "is given and has no baseline to inject, so that count moves a "
            "little with the chunk length")

    say(1.0, "Ready")
    return SpanAnalysis(
        events=events,
        ch_names=list(seen.get("ch_names", [])),
        sfreq=float(seen.get("sfreq", 0.0)),
        montage=str(seen.get("montage", "")),
        line_freq=float(seen.get("line_freq", 0.0)),
        t_start=float(t_start), t_stop=float(t_stop),
        segments=(pd.concat(segments, ignore_index=True) if segments else None),
        quality=_merge_quality(per_chunk_quality) if per_chunk_quality else None,
        steps=steps, chunks=len(plans))


def _widest_band(cfg: PipelineConfig, detectors) -> tuple:
    """The lowest band edge any detector will use, which is the one that sets
    the filter's ring-in and therefore the overlap it needs."""
    lows = [tuple(getattr(cfg, name).band) for name in detectors
            if hasattr(cfg, name)]
    if not lows:
        return tuple(BANDS["ripple"])
    return min(lows, key=lambda band: band[0])


def _merge_quality(frames: list) -> pd.DataFrame:
    """One row per channel out of one row per channel per chunk.

    The measurements are summarised by their **median across chunks**, which
    is a real number about the span rather than a reconstruction of a
    statistic that was never computed over it. The verdict is the worst seen,
    and `chunks_bad` says how often: a contact set aside on the strength of
    one chunk in ten is a different claim from one set aside on all ten, and a
    reviewer who wants to keep it needs to be able to tell them apart.
    """
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    joined = pd.concat(frames, ignore_index=True)
    numeric = [column for column in joined.columns
               if column not in ("channel", "good", "flagged", "reason")]

    rows = []
    for channel, group in joined.groupby("channel", sort=False):
        reasons = [str(value) for value in group["reason"] if str(value)]
        worst = min(reasons, key=_severity) if reasons else ""
        row = {"channel": str(channel)}
        for column in numeric:
            row[column] = float(pd.to_numeric(group[column],
                                              errors="coerce").median())
        row["reason"] = worst
        row["good"] = bool(group["good"].all())
        row["flagged"] = bool(worst) and bool(group["good"].all())
        row["chunks_bad"] = int(sum(1 for value in group["reason"] if str(value)))
        row["n_chunks"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _severity(reason: str) -> int:
    try:
        return REASON_SEVERITY.index(reason)
    except ValueError:
        return len(REASON_SEVERITY)


def recording_source(record) -> Source:
    """Treat one already-loaded `Recording` as a source of sub-spans.

    For the synthetic generator and for anything already in memory. Cropping
    rather than re-reading means this does not demonstrate the memory saving
    -- it is here so the streamed path can be tested against the direct one on
    the same signal, which is the only way to know the two agree.
    """
    import dataclasses

    def read(t_start: float, t_stop: float):
        raw = record.raw.copy().crop(tmin=max(0.0, t_start - record.t_offset),
                                     tmax=t_stop - record.t_offset,
                                     include_tmax=False)
        return dataclasses.replace(record, raw=raw, t_offset=float(t_start))

    return read
