"""Artifact rejection: deciding which detected events are worth reporting.

Every HFO detector ever published produces false positives, and they are not
random: a sharp transient (an epileptiform spike, an electrode pop, a movement
artifact) run through an 80-250 Hz filter *rings*, and the ringing looks like a
beautiful ripple. Publishing the raw detector output would make the prototype
look productive and be wrong.

Three checks, applied to every event, all of which look at the *unfiltered*
signal or at the event's own shape:

1. **Cycle count.** A ripple lasting ``d`` seconds at ``f`` Hz has ``d*f``
   cycles. Fewer than ``min_cycles`` means a transient, not an oscillation.
2. **Spectral peak.** Filter ringing has no peak of its own: its spectrum is
   the recording's 1/f background pushed up by the transient. A real
   oscillation leaves a bump. We require the in-band peak to rise
   ``min_peak_prominence_db`` dB above the fitted background
   (see :mod:`onset_hfo.spectral`).
3. **Unmeasurable spectrum plus a large transient.** When the window is too
   short to measure a spectrum, an event that coincides with a very large
   low-frequency deflection is rejected as probable ringing.

An event that co-occurs with a detected interictal discharge is *not*
rejected -- HFOs riding on spikes are real and clinically interesting -- it is
flagged, so the two populations can be counted separately.

Rejected events are kept in the table with ``accepted = False`` and a reason.
Nothing is deleted: a reviewer can always see what the detector proposed and
why the pipeline disagreed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from onset_hfo.config import BANDS, ValidationConfig
from onset_hfo.detectors.base import Event, bandpass, robust_scale
from onset_hfo.preprocess import Prepared

__all__ = ["validate_events", "flag_spike_cooccurrence", "rejection_summary"]


def validate_events(events: list[Event], prep: Prepared,
                    cfg: ValidationConfig | None = None) -> list[Event]:
    """Mark each event as accepted or rejected. Modifies and returns the list."""
    cfg = cfg or ValidationConfig()
    if not events:
        return events

    channels = sorted({e.channel for e in events})
    idx = [prep.ch_names.index(c) for c in channels]
    low = bandpass(prep.data[idx], prep.sfreq, BANDS.low)
    _, low_scale = robust_scale(low)
    low_scale = {c: float(s) for c, s in zip(channels, low_scale.ravel(), strict=False)}
    low_rows = {c: low[i] for i, c in enumerate(channels)}

    for e in events:
        if e.detector == "spike":
            continue  # the spike detector has its own morphology criteria
        cycles = e.n_cycles
        if np.isfinite(cycles) and cycles < cfg.min_cycles:
            _reject(e, f"only {cycles:.1f} cycles (< {cfg.min_cycles:g})")
            continue
        prominence = e.spectral_prominence_db
        if np.isfinite(prominence):
            if prominence < cfg.min_peak_prominence_db:
                _reject(e, f"in-band spectral peak only {prominence:.1f} dB above the 1/f "
                           f"background (< {cfg.min_peak_prominence_db:g} dB): probable filter ringing")
                continue
        else:
            i0 = max(0, int(round((e.start - prep.t_offset) * prep.sfreq)))
            i1 = min(prep.data.shape[1], int(round((e.stop - prep.t_offset) * prep.sfreq)))
            window = low_rows[e.channel][i0:i1]
            if window.size and np.max(np.abs(window)) > cfg.max_low_band_sd * low_scale[e.channel]:
                _reject(e, "spectrum not measurable and a large low-frequency transient is "
                           "present: probable filter ringing")
                continue
        e.accepted = True
        e.reject_reason = None
    if getattr(prep, "modality", "ieeg") == "eeg":
        reject_concurrent(events, len(prep.ch_names), cfg.scalp_max_concurrent_fraction)
    return events


def reject_concurrent(events: list[Event], n_channels: int, fraction: float) -> list[Event]:
    """Reject HFOs seen on too many channels at once: muscle or movement.

    For each accepted HFO, count the other channels holding a candidate of the
    same detector that overlaps it in time (accepted or not, since a burst of
    muscle produces candidates that fail the other checks too). More than
    `fraction` of the other channels, and it is rejected. Scalp EEG only: an
    implanted contact sees a field a few millimetres across, and a seizure
    or a discharge spreading across a montage is not an artefact there.
    """
    if not fraction or fraction <= 0 or n_channels < 3:
        return events
    limit = fraction * (n_channels - 1)
    by_detector: dict[str, list[Event]] = {}
    for e in events:
        if e.detector != "spike":
            by_detector.setdefault(e.detector, []).append(e)
    for group in by_detector.values():
        starts = np.array([e.start for e in group])
        stops = np.array([e.stop for e in group])
        names = np.array([e.channel for e in group])
        for e in group:
            if not e.accepted:
                continue
            overlap = (starts < e.stop) & (stops > e.start) & (names != e.channel)
            others = len(set(names[overlap]))
            if others > limit:
                _reject(e, f"seen on {others} other channels at once (more than "
                           f"{fraction:.0%} of them): probable muscle or movement artefact")
    return events


def _reject(event: Event, reason: str) -> None:
    event.accepted = False
    event.reject_reason = reason


def flag_spike_cooccurrence(hfo_events: list[Event], spike_events: list[Event],
                            tolerance: float = 0.05) -> list[Event]:
    """Flag HFOs that overlap an interictal discharge on the same channel."""
    by_channel: dict[str, list[Event]] = {}
    for s in spike_events:
        by_channel.setdefault(s.channel, []).append(s)
    for ch in by_channel:
        by_channel[ch].sort(key=lambda e: e.start)
    for e in hfo_events:
        e.co_occurs_with_spike = any(e.overlaps(s, tolerance) for s in by_channel.get(e.channel, []))
    return hfo_events


def rejection_summary(events: list[Event]) -> dict[str, int]:
    """Count of rejection reasons -- printed in the report's data-quality block."""
    counts = Counter(e.reject_reason.split(":")[0] if e.reject_reason else "accepted"
                     for e in events)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
