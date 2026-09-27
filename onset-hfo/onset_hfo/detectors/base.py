"""Shared machinery for every detector: the event record and the small set of
signal-processing primitives the detectors are built from.

Keeping these in one place means the HFO detectors differ *only* in the
feature they threshold -- RMS energy, line length, the Hilbert envelope, or
short-time energy -- which is exactly the comparison the project wants to make
visible.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

__all__ = [
    "ChannelBaseline",
    "Event",
    "events_to_frame",
    "frame_to_events",
    "robust_scale",
    "bandpass",
    "sliding_rms",
    "sliding_line_length",
    "sliding_hilbert_envelope",
    "sliding_energy",
    "threshold_segments",
    "extend_segments",
    "merge_events",
]

MAD_TO_SD = 1.4826  # scale factor that makes the MAD a consistent estimator of SD


@dataclass
class Event:
    """One detected candidate event.

    Times are in **original-recording seconds** (see ``Recording.t_offset``),
    so that a window quoted in a report can be found again in the archive file.

    Attributes
    ----------
    channel:
        Channel the event was detected on (a bipolar pair name such as
        ``"AD1-AD2"`` when a bipolar montage is used).
    start, stop:
        Event boundaries in seconds.
    detector:
        Which algorithm produced it (``"rms"``, ``"line_length"``, ``"spike"``).
    band:
        Frequency band the detector looked in, in Hz.
    score:
        How far the detector's feature rose above its own threshold, as a
        ratio (1.0 = exactly at threshold). Comparable within a detector and a
        channel; not a probability and not comparable across detectors.
    peak_amplitude_uv:
        Largest absolute amplitude of the band-passed signal inside the event.
    peak_frequency_hz:
        Frequency of the strongest in-band spectral peak of the *unfiltered*
        signal in the event window.
    spectral_prominence_db:
        How far that peak rises above the recording's own 1/f background. This
        is the number that separates a genuine oscillation from filter ringing.
    n_peaks:
        Rectified peaks above the secondary threshold -- the "is it really an
        oscillation" count from Staba's criteria.
    accepted:
        Whether the event survived artifact rejection (:mod:`onset_hfo.validate`).
    reject_reason:
        Why it did not, when it did not.
    co_occurs_with_spike:
        True when an interictal discharge was detected on the same channel at
        the same time. Not a rejection: HFOs riding on spikes are a real and
        clinically interesting phenomenon. It is reported so that a reader can
        tell the two populations apart.
    """

    channel: str
    start: float
    stop: float
    detector: str
    band: tuple[float, float]
    score: float = 0.0
    peak_amplitude_uv: float = 0.0
    peak_frequency_hz: float = float("nan")
    spectral_prominence_db: float = float("nan")
    n_peaks: int = 0
    n_cycles: float = float("nan")
    accepted: bool = True
    reject_reason: str | None = None
    co_occurs_with_spike: bool = False
    contacts: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.stop - self.start) * 1000.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.stop)

    def overlaps(self, other: Event, tolerance: float = 0.0) -> bool:
        """True if the two events share time (optionally within a tolerance)."""
        return (self.start - tolerance) < other.stop and (other.start - tolerance) < self.stop

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = list(self.band)
        d["duration_ms"] = self.duration_ms
        return d


def events_to_frame(events: list[Event]) -> pd.DataFrame:
    """Events as a tidy table -- the format every downstream step consumes."""
    if not events:
        return pd.DataFrame(columns=[
            "channel", "start", "stop", "duration_ms", "detector", "band_low", "band_high",
            "score", "peak_amplitude_uv", "peak_frequency_hz", "spectral_prominence_db",
            "n_peaks", "n_cycles", "accepted", "reject_reason", "co_occurs_with_spike",
            "contacts"])
    rows = []
    for e in events:
        d = e.to_dict()
        d["band_low"], d["band_high"] = d.pop("band")
        d["contacts"] = "|".join(e.contacts)
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["channel", "start"]).reset_index(drop=True)


def frame_to_events(df: pd.DataFrame) -> list[Event]:
    """Inverse of :func:`events_to_frame` (used when reloading saved results)."""
    out = []
    for _, r in df.iterrows():
        out.append(Event(
            channel=str(r["channel"]), start=float(r["start"]), stop=float(r["stop"]),
            detector=str(r["detector"]), band=(float(r["band_low"]), float(r["band_high"])),
            score=float(r.get("score", 0.0)),
            peak_amplitude_uv=float(r.get("peak_amplitude_uv", 0.0)),
            peak_frequency_hz=float(r.get("peak_frequency_hz", float("nan"))),
            spectral_prominence_db=float(r.get("spectral_prominence_db", float("nan"))),
            n_peaks=int(r.get("n_peaks", 0) or 0),
            n_cycles=float(r.get("n_cycles", float("nan"))),
            accepted=bool(r.get("accepted", True)),
            reject_reason=(None if pd.isna(r.get("reject_reason")) else str(r.get("reject_reason"))),
            co_occurs_with_spike=bool(r.get("co_occurs_with_spike", False)),
            contacts=[c for c in str(r.get("contacts", "")).split("|") if c],
        ))
    return out


# --------------------------------------------------------------------------
# Signal primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelBaseline:
    """The three scale numbers one channel's detection is thresholded against.

    Normally computed from the channel's own trace inside
    :func:`~onset_hfo.detectors.engine.detect_with_feature`. They are named and
    passed around separately for one reason: **when a recording is analysed in
    chunks, these must be properties of the whole recording rather than of the
    chunk.** A per-chunk baseline would make the detector's answer depend on
    where the chunk boundaries fell, which is the failure this project already
    documented once for analysis windows (``docs/OUTCOME.md``) and does not
    intend to reintroduce in the name of scaling.
    """

    feature_center: float     #: median of the feature trace
    feature_scale: float      #: 1.4826 x MAD of the feature trace
    amplitude_scale: float    #: 1.4826 x MAD of the band-passed signal


def robust_scale(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD-based standard deviation.

    Why not mean/SD: a channel with many HFOs inflates its own SD, which
    raises its own threshold and hides its events -- the detector would punish
    exactly the channels we care about. The median absolute deviation is
    almost unaffected by a few percent of outliers.
    """
    center = np.median(x, axis=axis, keepdims=True)
    scale = MAD_TO_SD * np.median(np.abs(x - center), axis=axis, keepdims=True)
    scale = np.where(scale <= 0, np.finfo(float).eps, scale)
    return center, scale


def bandpass(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase FIR band-pass, applied to every channel of ``data``.

    Zero-phase matters: a causal filter shifts an event in time, and HFO work
    depends on saying *when* something happened.
    """
    import mne

    low, high = band
    high = min(high, 0.45 * sfreq * 2)  # keep below Nyquist with margin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.filter.filter_data(
            np.asarray(data, dtype=np.float64), sfreq=sfreq, l_freq=low, h_freq=high,
            method="fir", fir_design="firwin", phase="zero", pad="reflect_limited",
            verbose="ERROR")


def _sliding_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sum over a centred sliding window, same length as the input."""
    if window <= 1:
        return values.astype(np.float64)
    cs = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0), axis=-1)
    out = cs[..., window:] - cs[..., :-window]
    pad_left = window // 2
    pad_right = values.shape[-1] - out.shape[-1] - pad_left
    return np.pad(out, [(0, 0)] * (values.ndim - 1) + [(pad_left, pad_right)], mode="edge")


def sliding_rms(x: np.ndarray, window: int) -> np.ndarray:
    """Root-mean-square energy in a sliding window (Staba et al., 2002)."""
    return np.sqrt(np.maximum(_sliding_sum(x ** 2, window) / max(window, 1), 0.0))


def sliding_line_length(x: np.ndarray, window: int) -> np.ndarray:
    """Line length: mean absolute sample-to-sample change (Gardner et al., 2007).

    Line length rises with both amplitude and frequency, which is why it
    responds to oscillations that RMS alone can miss -- and why it also reacts
    to sharp transients, which is why events are validated afterwards.
    """
    diffs = np.abs(np.diff(x, axis=-1, prepend=x[..., :1]))
    return _sliding_sum(diffs, window) / max(window, 1)


def sliding_hilbert_envelope(x: np.ndarray, window: int) -> np.ndarray:
    """Amplitude envelope from the analytic signal, smoothed over a window.

    The envelope is :math:`|x + i\,\mathcal{H}\{x\}|`, where
    :math:`\mathcal{H}` is the Hilbert transform. Unlike RMS it is defined
    sample by sample rather than accumulated over a window, so it follows the
    rise and fall of a burst more sharply; the short moving average afterwards
    is only to stop single-sample excursions from setting the threshold.

    This is the feature used by envelope-based HFO detectors. It is **not** a
    reimplementation of the MNI detector of Zelmann et al. (2012), whose
    distinguishing contribution is an automatic baseline-selection stage on a
    wavelet-entropy criterion. Only the feature is shared; see
    :mod:`onset_hfo.detectors.hilbert` for why that distinction is kept
    explicit.
    """
    envelope = np.abs(hilbert(np.asarray(x, dtype=np.float64), axis=-1))
    if window <= 1:
        return envelope
    return _sliding_sum(envelope, window) / window


def sliding_energy(x: np.ndarray, window: int) -> np.ndarray:
    """Short-time energy: the *sum* of squares in a sliding window.

    Related to :func:`sliding_rms` by a monotone transform --
    ``rms = sqrt(energy / window)`` -- so under a *fixed* threshold the two
    would select identical samples. They differ substantially here because the
    threshold is ``median + k * robustSD`` of the feature's own distribution,
    and squaring is not affine: the same ``k`` lands near the 98th percentile
    of an RMS trace and the 96th of an energy trace. See
    :mod:`onset_hfo.detectors.short_time_energy`, where that was predicted
    to be a small effect and measured to be a large one.
    """
    return _sliding_sum(np.asarray(x, dtype=np.float64) ** 2, window)


def threshold_segments(metric: np.ndarray, threshold: float, min_len: int,
                       merge_gap: int) -> list[tuple[int, int]]:
    """Contiguous stretches where ``metric`` exceeds ``threshold``.

    Stretches closer than ``merge_gap`` samples are merged first, then those
    shorter than ``min_len`` samples are dropped -- in that order, so that an
    oscillation whose envelope dips briefly below threshold stays one event.
    """
    above = metric > threshold
    if not above.any():
        return []
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        stops.append(len(above))
    segs = list(zip(starts, stops, strict=True))
    merged: list[tuple[int, int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


def extend_segments(segments: list[tuple[int, int]], metric: np.ndarray,
                    extend_threshold: float, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Grow each segment outwards while ``metric`` stays above a lower threshold.

    Detection thresholds are set high so that noise does not trigger them, but
    an oscillation's envelope rises and falls: the part above a 5-SD threshold
    is only its loud middle. Measuring duration there would systematically
    underestimate every event -- and therefore its cycle count. So the
    threshold decides *whether* an event exists and this hysteresis decides
    *how long* it is, which is the standard two-threshold arrangement.
    """
    if not segments:
        return []
    n = metric.shape[-1]
    grown: list[tuple[int, int]] = []
    above = metric > extend_threshold
    for s, e in segments:
        i = s
        while i > 0 and above[i - 1]:
            i -= 1
        j = e
        while j < n and above[j]:
            j += 1
        if grown and i - grown[-1][1] <= merge_gap:
            grown[-1] = (grown[-1][0], max(grown[-1][1], j))
        else:
            grown.append((i, j))
    return grown


def count_oscillation_peaks(segment: np.ndarray, threshold: float) -> int:
    """Rectified peaks above ``threshold`` -- the oscillation-count criterion.

    A single sharp transient has one or two; a ripple of n cycles has ~n.
    """
    if segment.size < 3:
        return 0
    peaks, _ = find_peaks(np.abs(segment), height=threshold)
    return int(peaks.size)


def merge_events(events: list[Event], gap: float = 0.0) -> list[Event]:
    """Merge overlapping events of the same detector on the same channel.

    Keeps the strongest score and the widest extent, so that one physical
    oscillation is counted once no matter how the threshold wobbled.
    """
    out: list[Event] = []
    for ev in sorted(events, key=lambda e: (e.channel, e.detector, e.start)):
        if out and out[-1].channel == ev.channel and out[-1].detector == ev.detector \
                and ev.start - out[-1].stop <= gap:
            prev = out[-1]
            prev.stop = max(prev.stop, ev.stop)
            if ev.score > prev.score:
                prev.score = ev.score
                prev.peak_amplitude_uv = max(prev.peak_amplitude_uv, ev.peak_amplitude_uv)
                prev.peak_frequency_hz = ev.peak_frequency_hz
            prev.n_peaks = max(prev.n_peaks, ev.n_peaks)
        else:
            out.append(ev)
    return out
