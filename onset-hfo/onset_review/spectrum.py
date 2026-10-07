"""The power spectrum of each channel, and what it says about the contact.

Before any detector runs, a spectrum answers the question the Quality page's
flag only raises: is this contact noisy, or busy? A noisy amplifier lays a
flat carpet of high-frequency power; a contact full of real activity keeps
the brain's steep fall-off with bursts riding on it; a mains-contaminated
one has a comb of peaks at the harmonics. All three look alike in a rate
table and nothing alike on a log-log plot.

Qt-free. Welch's estimate from MNE (``psd_array_welch``) over the
preprocessed signal the window shows, the aperiodic slope fitted on the
log-log spectrum away from the mains lines, and the share of power in the
band being analysed. The panel in :mod:`onset_review.spectrumview` draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["Spectrum", "compute", "slope_fit", "mains_lines", "band_share", "summarise"]

#: Welch segment length in seconds: two gives 0.5 Hz bins, enough to see a
#: 2 Hz notch and the mains lines, over a minute of signal.
SEGMENT_S = 2.0
#: The frequency range the aperiodic slope is fitted on. Above the slow
#: rhythms, below the band being analysed, so the fit says what the
#: background is doing where the detector will look.
SLOPE_RANGE = (4.0, 80.0)


class Spectrum:
    """Frequencies, one spectrum per channel (µV²/Hz), and the window's facts."""

    def __init__(self, freqs: np.ndarray, power: np.ndarray, channels: list[str],
                 sfreq: float, line_freq: float, band: tuple[float, float]):
        self.freqs = np.asarray(freqs, dtype=float)
        self.power = np.asarray(power, dtype=float)
        self.channels = list(channels)
        self.sfreq = float(sfreq)
        self.line_freq = float(line_freq)
        self.band = (float(band[0]), float(band[1]))

    @property
    def available(self) -> bool:
        return self.freqs.size > 0 and self.power.size > 0

    def index(self, channel: str) -> int:
        return self.channels.index(channel)


def compute(session, segment_s: float = SEGMENT_S) -> Spectrum:
    """Welch's spectrum of every channel in the window on screen.

    From the session's own preprocessed signal, so the spectrum is of what
    the detectors saw: high-passed, notched, re-referenced. Microvolts
    squared per hertz, because every amplitude in this window is in µV.
    """
    raw = getattr(session, "raw", None)
    band = tuple(getattr(getattr(session, "request", None), "band_hz", (80.0, 250.0)))
    line_freq = float(getattr(session, "line_freq", 50.0) or 50.0)
    if raw is None or not getattr(raw, "n_times", 0):
        return Spectrum(np.empty(0), np.empty((0, 0)), [], 0.0, line_freq, band)
    from mne.time_frequency import psd_array_welch

    sfreq = float(raw.info["sfreq"])
    data = np.asarray(raw.get_data(), dtype=float) * 1e6
    n_per_seg = int(min(data.shape[1], max(16, round(segment_s * sfreq))))
    power, freqs = psd_array_welch(data, sfreq, fmin=0.5, fmax=sfreq / 2.0,
                                   n_fft=n_per_seg, n_per_seg=n_per_seg,
                                   n_overlap=n_per_seg // 2, average="median",
                                   verbose="ERROR")
    return Spectrum(freqs, power, list(raw.ch_names), sfreq, line_freq, band)


def mains_lines(spectrum: Spectrum) -> list[float]:
    """The mains frequency and its harmonics up to Nyquist."""
    if not spectrum.available or spectrum.line_freq <= 0:
        return []
    nyquist = spectrum.sfreq / 2.0
    return [float(f) for f in np.arange(spectrum.line_freq, nyquist, spectrum.line_freq)]


def _away_from_mains(freqs: np.ndarray, lines: list[float], width: float = 2.5) -> np.ndarray:
    keep = np.ones(freqs.shape, dtype=bool)
    for line in lines:
        keep &= np.abs(freqs - line) > width
    return keep


def slope_fit(spectrum: Spectrum, channel: str,
              fmin: float = SLOPE_RANGE[0], fmax: float = SLOPE_RANGE[1]) -> tuple[float, float]:
    """The aperiodic slope of one channel: a straight line through log power
    against log frequency between `fmin` and `fmax`, the mains lines left
    out. Returns (slope, intercept) in log10 units; the slope of a healthy
    intracranial contact is around -2, and a flat carpet of amplifier noise
    is nearer 0."""
    if not spectrum.available:
        return float("nan"), float("nan")
    i = spectrum.index(channel)
    freqs, power = spectrum.freqs, spectrum.power[i]
    keep = (freqs >= fmin) & (freqs <= fmax) & (power > 0)
    keep &= _away_from_mains(freqs, mains_lines(spectrum))
    if keep.sum() < 4:
        return float("nan"), float("nan")
    x, y = np.log10(freqs[keep]), np.log10(power[keep])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def band_share(spectrum: Spectrum, channel: str) -> float:
    """The share of this channel's total power that lies in the analysed
    band, the mains lines left out of both. A noisy contact's share is high
    and so is a busy one's; the slope tells them apart."""
    if not spectrum.available:
        return float("nan")
    i = spectrum.index(channel)
    freqs, power = spectrum.freqs, spectrum.power[i]
    keep = _away_from_mains(freqs, mains_lines(spectrum)) & (freqs >= 0.5)
    total = float(power[keep].sum())
    if total <= 0:
        return float("nan")
    inside = keep & (freqs >= spectrum.band[0]) & (freqs <= spectrum.band[1])
    return float(power[inside].sum() / total)


def mains_share(spectrum: Spectrum, channel: str, width: float = 1.0) -> float:
    """The share of power within `width` Hz of the mains lines."""
    if not spectrum.available:
        return float("nan")
    i = spectrum.index(channel)
    freqs, power = spectrum.freqs, spectrum.power[i]
    total = float(power[freqs >= 0.5].sum())
    if total <= 0:
        return float("nan")
    near = np.zeros(freqs.shape, dtype=bool)
    for line in mains_lines(spectrum):
        near |= np.abs(freqs - line) <= width
    return float(power[near].sum() / total)


def summarise(spectrum: Spectrum) -> pd.DataFrame:
    """One row per channel: slope, band share, mains share, and a word.

    The word is a reading of the two numbers, not a verdict: ``steep`` is
    the brain's usual fall-off, ``flat`` is what a noisy or poorly coupled
    contact looks like, ``mains`` says the harmonics dominate. The quality
    stage decides what is analysed; this says what the spectrum looks like.
    """
    rows = []
    for channel in spectrum.channels:
        slope, _ = slope_fit(spectrum, channel)
        share = band_share(spectrum, channel)
        mains = mains_share(spectrum, channel)
        if np.isnan(slope):
            word = "unknown"
        elif mains > 0.3:
            word = "mains"
        elif slope > -1.0:
            word = "flat"
        else:
            word = "steep"
        rows.append({"channel": channel, "slope": slope, "band_share": share,
                     "mains_share": mains, "reading": word})
    return pd.DataFrame(rows, columns=["channel", "slope", "band_share", "mains_share",
                                       "reading"])


def describe(spectrum: Spectrum, channel: str) -> str:
    """One sentence about one channel, for the panel's headline."""
    if not spectrum.available or channel not in spectrum.channels:
        return ""
    slope, _ = slope_fit(spectrum, channel)
    share = band_share(spectrum, channel)
    mains = mains_share(spectrum, channel)
    lo, hi = spectrum.band
    parts = [f"<b>{channel}</b>"]
    if not np.isnan(slope):
        parts.append(f"aperiodic slope {slope:.2f} over {SLOPE_RANGE[0]:.0f}–"
                     f"{SLOPE_RANGE[1]:.0f} Hz"
                     + (" (steep: the brain's usual fall-off)" if slope <= -1.0
                        else " (flat: a carpet of high-frequency power, as a noisy or "
                             "poorly coupled contact lays down)"))
    if not np.isnan(share):
        parts.append(f"{100 * share:.1f} % of its power in {lo:.0f}–{hi:.0f} Hz")
    if not np.isnan(mains):
        parts.append(f"{100 * mains:.1f} % within 1 Hz of the mains lines")
    return "; ".join(parts) + "."
