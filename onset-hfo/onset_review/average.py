"""The average event: every detection on a channel, aligned and averaged.

One event is a judgement; fifty are a morphology. A channel whose ripples
average to a spindle-shaped burst with energy confined to the band is a
different thing from a channel whose "ripples" average to a sharp transient
with a column of energy under it, and the two can carry the same rate. The
field reads this picture routinely -- MNE's epochs-and-evoked, applied to
detections instead of stimuli -- and the software lacked it.

Qt-free. Every accepted event of the ranking detector on the channel is cut
from the preprocessed signal, aligned at the positive peak of its band-passed
trace,
and averaged: the wideband and band-passed means with their spread, and the
mean of each event's own time-frequency picture (each relative to its own
surroundings, so the mean is in dB above background like the single-event
view). The island-or-column reading is the same rule the detail view applies
to one event, applied to the mean. :mod:`onset_review.averageview` draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onset_hfo.modality import trace_scale
from onset_review import detail

__all__ = ["AverageEvent", "compute", "channels_with_events", "describe", "HALF_S",
           "MAX_EVENTS"]

#: Half-width of the averaged window round the aligned peak, in seconds. A
#: ripple is tens of milliseconds; 150 ms either side shows it with the
#: discharge it may ride on and nothing of its neighbours.
HALF_S = 0.15
#: The time-frequency picture is taken over twice that, so the background it
#: is relative to is signal outside the event and not the event itself.
TFR_HALF_S = 0.30
#: Enough events for a stable mean, few enough to compute in a moment.
MAX_EVENTS = 60


@dataclass
class AverageEvent:
    """What the view draws, with nothing left to compute."""

    channel: str
    detector: str
    band: tuple
    n: int                                  #: events averaged
    n_available: int                        #: accepted events on the channel
    times: np.ndarray = field(default_factory=lambda: np.empty(0))   #: s from the peak
    mean_wideband: np.ndarray = field(default_factory=lambda: np.empty(0))
    sd_wideband: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_band: np.ndarray = field(default_factory=lambda: np.empty(0))
    sd_band: np.ndarray = field(default_factory=lambda: np.empty(0))
    tfr_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    freqs: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_power_db: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    n_tfr: int = 0                          #: events in the time-frequency mean
    mean_duration_ms: float = float("nan")
    mean_frequency_hz: float = float("nan")
    band_contrast_db: float | None = None
    reading: str = "unclear"
    why: str = ""
    notes: tuple = ()

    @property
    def available(self) -> bool:
        return self.n > 0 and self.times.size > 0

    @property
    def cycles(self) -> float:
        if not np.isfinite(self.mean_duration_ms) or not np.isfinite(self.mean_frequency_hz):
            return float("nan")
        return self.mean_duration_ms / 1000.0 * self.mean_frequency_hz


def channels_with_events(session, detector: str | None = None) -> list[tuple[str, int]]:
    """(channel, accepted events) in rank order, channels with none left out."""
    detector = detector or getattr(getattr(session, "request", None), "primary", None)
    counts: dict[str, int] = {}
    for event in getattr(session, "events", []) or []:
        if event.accepted and (detector is None or event.detector == detector):
            counts[str(event.channel)] = counts.get(str(event.channel), 0) + 1
    findings = getattr(session, "findings", None)
    order: list[str] = []
    if findings is not None and not findings.empty and "channel" in findings.columns:
        order = [str(c) for c in findings.sort_values("rank")["channel"]]
    ranked = [c for c in order if c in counts] + sorted(c for c in counts if c not in order)
    return [(c, counts[c]) for c in ranked]


def _subsample(events: list, limit: int) -> list:
    """Up to `limit` events, evenly spaced through the window rather than the
    first `limit`, so a burst at the start does not stand for the hour."""
    if len(events) <= limit:
        return list(events)
    index = np.unique(np.round(np.linspace(0, len(events) - 1, limit)).astype(int))
    return [events[i] for i in index]


def compute(session, channel: str, detector: str | None = None,
            half_s: float = HALF_S, max_events: int = MAX_EVENTS) -> AverageEvent:
    """Align and average the accepted events on `channel`.

    Returns an empty `AverageEvent` (``available`` false) rather than raising
    when there is nothing to average: no signal, no events, or every event
    too near the edge of the loaded window to cut.
    """
    detector = detector or getattr(getattr(session, "request", None), "primary", None) or ""
    raw = getattr(session, "raw", None)
    chosen = [e for e in (getattr(session, "events", []) or [])
              if e.accepted and str(e.channel) == str(channel) and e.detector == detector]
    chosen.sort(key=lambda e: float(e.start))
    band = tuple(float(v) for v in (chosen[0].band if chosen and chosen[0].band
                                    else (0.0, 0.0)))
    empty = AverageEvent(str(channel), detector, band, 0, len(chosen))
    names = list(getattr(raw, "ch_names", [])) if raw is not None else []
    if raw is None or not chosen or str(channel) not in names:
        return empty

    sfreq = float(raw.info["sfreq"])
    offset = float(getattr(session, "t_offset", 0.0))
    total = int(raw.n_times)
    pick = names.index(str(channel))
    events = _subsample(chosen, max_events)
    notes: list[str] = []
    if len(events) < len(chosen):
        notes.append(f"{len(events)} of the {len(chosen)} events, evenly spaced through "
                     "the window, are averaged.")

    half = int(round(half_s * sfreq))
    tfr_half = int(round(TFR_HALF_S * sfreq))
    margin = int(round(0.1 * sfreq))                  # for the band-pass's edges
    wides, bands, powers, durations, frequencies = [], [], [], [], []
    freqs = np.empty(0)
    skipped = 0
    tfr_notes: set[str] = set()
    for event in events:
        start_s = float(event.start) - offset
        stop_s = float(event.stop) - offset
        first = int(round(start_s * sfreq)) - tfr_half - margin
        last = int(round(stop_s * sfreq)) + tfr_half + margin
        if first < 0 or last > total:
            skipped += 1
            continue
        data = np.asarray(raw.get_data(picks=[pick], start=first, stop=last),
                          dtype=float)[0] * trace_scale(raw)
        passed = detail._band_pass(data, sfreq, band)
        on = int(round(start_s * sfreq)) - first
        off = max(on + 1, int(round(stop_s * sfreq)) - first)
        # The positive peak rather than the largest excursion: traces aligned
        # at a peak of either sign cancel in the mean where the polarity varies.
        peak = on + int(np.argmax(passed[on:off]))
        if peak - tfr_half < 0 or peak + tfr_half + 1 > data.size:
            skipped += 1
            continue
        wides.append(data[peak - half: peak + half + 1])
        bands.append(passed[peak - half: peak + half + 1])
        durations.append(float(getattr(event, "duration_ms", (stop_s - start_s) * 1000.0)))
        frequency = float(getattr(event, "peak_frequency_hz", float("nan")))
        if np.isfinite(frequency) and frequency > 0:
            frequencies.append(frequency)
        # Each event's time-frequency picture relative to its own surroundings,
        # over the wider cut, so the mean is "dB above background" like the
        # single-event view and not dB above a background that is the event.
        wide_cut = data[peak - tfr_half: peak + tfr_half + 1]
        tfr_times = (np.arange(wide_cut.size) - tfr_half) / sfreq
        f, power_db, more = detail._time_frequency(
            wide_cut, sfreq, band, tfr_times, start_s - peak / sfreq - first / sfreq,
            stop_s - peak / sfreq - first / sfreq)
        tfr_notes.update(more)
        if power_db.size and (freqs.size == 0 or f.shape == freqs.shape):
            freqs = f
            powers.append(power_db)

    if not wides:
        empty.notes = tuple(notes + ["Every event sits too near the edge of the loaded "
                                     "window to cut a stretch of signal round it."])
        return empty
    if skipped:
        notes.append(f"{skipped} event(s) near the edge of the loaded window were left out.")
    notes.extend(sorted(tfr_notes))

    wide_stack, band_stack = np.vstack(wides), np.vstack(bands)
    times = (np.arange(wide_stack.shape[1]) - half) / sfreq
    mean_duration = float(np.mean(durations)) if durations else float("nan")
    mean_frequency = float(np.mean(frequencies)) if frequencies else float("nan")
    out = AverageEvent(
        str(channel), detector, band, int(wide_stack.shape[0]), len(chosen),
        times=times,
        mean_wideband=wide_stack.mean(axis=0), sd_wideband=wide_stack.std(axis=0),
        mean_band=band_stack.mean(axis=0), sd_band=band_stack.std(axis=0),
        mean_duration_ms=mean_duration, mean_frequency_hz=mean_frequency,
        notes=tuple(notes))
    if powers:
        stack = np.stack(powers)
        out.tfr_times = (np.arange(stack.shape[2]) - tfr_half) / sfreq
        out.freqs = freqs
        out.mean_power_db = stack.mean(axis=0)
        out.n_tfr = int(stack.shape[0])
        out.band_contrast_db = _contrast(out)
    cycles = out.cycles
    out.reading, out.why = detail.read_contrast(
        out.band_contrast_db, cycles if np.isfinite(cycles) else None, band)
    return out


def _contrast(avg: AverageEvent) -> float | None:
    """In-band minus out-of-band energy over the mean event's own span, in
    dB, the same measure the single-event reading uses."""
    if not avg.mean_power_db.size or not avg.freqs.size:
        return None
    half_dur = (avg.mean_duration_ms / 1000.0) / 2.0 if np.isfinite(avg.mean_duration_ms) else 0.0
    during = np.abs(avg.tfr_times) <= max(half_dur, 1.0 / max(avg.mean_frequency_hz, 1.0))
    return detail.band_contrast(avg.mean_power_db, avg.freqs, during, avg.band)


def describe(avg: AverageEvent) -> str:
    """The headline: what was averaged and what the mean reads as."""
    if not avg.available:
        return ""
    kind = "fast ripples" if avg.band and float(avg.band[0]) >= 200 else "ripples"
    bits = [f"<b>{avg.channel}</b> — {avg.n} {kind} averaged at their band-passed peak"]
    if np.isfinite(avg.mean_duration_ms):
        bits.append(f"mean {avg.mean_duration_ms:.0f} ms")
    if np.isfinite(avg.mean_frequency_hz):
        bits.append(f"{avg.mean_frequency_hz:.0f} Hz")
    if avg.band_contrast_db is not None:
        bits.append(f"mean time-frequency reads as <b>{avg.reading}</b> "
                    f"({avg.band_contrast_db:+.1f} dB in band)")
    else:
        bits.append("no time-frequency mean")
    return " · ".join(bits)
