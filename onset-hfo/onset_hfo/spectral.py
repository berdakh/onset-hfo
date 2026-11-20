"""Spectral description of a single event window.

One measurement matters more than any other in HFO work: *is there a peak in
the high-frequency band that stands above this recording's own 1/f
background?* A sharp transient (an epileptiform spike, an electrode pop)
contains energy at all frequencies, so a band-pass filter turns it into
something that looks like a ripple -- the classic false positive. A genuine
oscillation leaves a bump in the spectrum; ringing does not.

That is what :func:`event_spectral_peak` measures, and what
:mod:`onset_hfo.validate` thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import detrend

__all__ = ["event_psd", "event_spectral_peak", "background_level_db",
           "event_peak_contrast"]


def event_psd(x: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density of one short, unfiltered event window.

    A single Hann-tapered, zero-padded periodogram -- *not* Welch averaging.
    Why it matters: an event is 30-100 ms long, so splitting it into
    overlapping Welch segments leaves 32-sample segments whose frequency grid
    is 31 Hz wide. On such a grid every ripple's "peak frequency" lands in the
    same bin (93.75 Hz on a 1000 Hz recording), which is a property of the bin
    edges, not of the brain.

    Zero-padding to at least 8x the window length interpolates the spectrum
    onto a fine grid. It does not create resolution that the window length
    cannot support -- a 40 ms window still cannot separate 150 from 160 Hz --
    but it does stop the estimate from being quantised to the band edge, and
    the peak it reports is the peak of the actual spectrum.
    """
    n = int(x.size)
    if n < 16:
        return np.array([]), np.array([])
    segment = detrend(np.asarray(x, dtype=np.float64), type="linear")
    window = np.hanning(n)
    nfft = int(max(512, 2 ** int(np.ceil(np.log2(n * 8)))))
    spectrum = np.fft.rfft(segment * window, n=nfft)
    scale = sfreq * float(np.sum(window ** 2))
    psd = (np.abs(spectrum) ** 2) / max(scale, 1e-30)
    psd[1:-1] *= 2.0
    return np.fft.rfftfreq(nfft, 1.0 / sfreq), psd


def background_level_db(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float],
                        fit_from: float = 10.0) -> np.ndarray | None:
    """Fit the 1/f background in log-log space, excluding the band of interest.

    Returns the fitted background in dB, evaluated at every frequency in
    ``freqs``, or ``None`` when there are too few usable points.
    """
    nyq = freqs[-1] if freqs.size else 0.0
    usable = (freqs >= fit_from) & (freqs <= 0.9 * nyq) & (psd > 0)
    fit_mask = usable & ~((freqs >= band[0] * 0.8) & (freqs <= band[1] * 1.2))
    if fit_mask.sum() < 5:
        return None
    coeffs = np.polyfit(np.log10(freqs[fit_mask]), 10.0 * np.log10(psd[fit_mask]), 1)
    with np.errstate(divide="ignore"):
        logf = np.log10(np.where(freqs > 0, freqs, np.nan))
    return np.polyval(coeffs, logf)


def event_spectral_peak(x: np.ndarray, sfreq: float, band: tuple[float, float]
                        ) -> tuple[float, float]:
    """Peak frequency inside ``band`` and its prominence over the background.

    Returns
    -------
    (peak_frequency_hz, prominence_db)
        ``(nan, nan)`` when the window is too short to measure -- callers must
        treat that as "unknown", never as "rejected".
    """
    freqs, psd = event_psd(x, sfreq)
    if freqs.size == 0:
        return float("nan"), float("nan")
    in_band = (freqs >= band[0]) & (freqs <= min(band[1], 0.95 * sfreq / 2))
    if in_band.sum() < 2 or not np.any(psd[in_band] > 0):
        return float("nan"), float("nan")
    background = background_level_db(freqs, psd, band)
    if background is None:
        # No usable background fit: fall back to the raw spectral maximum and
        # report the prominence as unknown rather than guessing it.
        idx = np.flatnonzero(in_band)[int(np.argmax(psd[in_band]))]
        return float(freqs[idx]), float("nan")
    # The peak is where the spectrum most EXCEEDS its own 1/f background, not
    # where it is largest. Those are different frequencies: power falls as 1/f,
    # so the raw maximum inside 80-250 Hz sits at the band's lower edge almost
    # every time, which says more about the band than about the oscillation.
    with np.errstate(divide="ignore"):
        excess = 10.0 * np.log10(np.maximum(psd, 1e-30)) - background
    excess = np.where(in_band & np.isfinite(excess), excess, -np.inf)
    idx = int(np.argmax(excess))
    if not np.isfinite(excess[idx]):
        return float("nan"), float("nan")
    return float(freqs[idx]), float(excess[idx])


def event_peak_contrast(x: np.ndarray, sfreq: float, band: tuple[float, float]
                        ) -> float:
    """How far the in-band peak stands above the troughs either side of it, dB.

    :func:`event_spectral_peak` asks how far the spectrum rises above a
    straight line fitted through the frequencies outside the band. A sharp
    discharge fails to be a straight line: its spectrum is curved, sits above
    the fit all the way through the band, and scores a "prominence" of many
    dB with no peak in it anywhere -- the largest excess is simply at the
    band's lower edge. This asks the other question, the one a reader asks of
    a time-frequency picture: is there an *island*? The peak is the one
    `event_spectral_peak` reports; the troughs are the lowest points of the
    same excess curve from half an octave below the band up to the peak, and
    from the peak to an octave above the band; the contrast is the peak over
    the higher of the two. An oscillation has a trough on both sides. Energy
    running through the band has none on the side it came from.

    The same rule the event view reads its picture by
    (`onset_review.detail.band_contrast`), measured here on the event's own
    spectrum so that the detector and the viewer cannot disagree about what
    counts. ``nan`` when it cannot be measured, which callers must treat as
    unknown and never as a rejection.
    """
    freqs, psd = event_psd(x, sfreq)
    if freqs.size == 0:
        return float("nan")
    top = min(band[1], 0.95 * sfreq / 2)
    in_band = (freqs >= band[0]) & (freqs <= top)
    if in_band.sum() < 2 or not np.any(psd[in_band] > 0):
        return float("nan")
    background = background_level_db(freqs, psd, band)
    if background is None:
        return float("nan")
    with np.errstate(divide="ignore"):
        excess = 10.0 * np.log10(np.maximum(psd, 1e-30)) - background
    finite = np.isfinite(excess)
    peak = int(np.argmax(np.where(in_band & finite, excess, -np.inf)))
    if not np.isfinite(excess[peak]):
        return float("nan")
    index = np.arange(freqs.size)
    left = finite & (freqs >= 0.5 * band[0]) & (index < peak)
    right = finite & (freqs <= min(2.0 * band[1], 0.95 * sfreq / 2)) & (index > peak)
    if not left.any() or not right.any():
        return float("nan")
    return float(excess[peak] - max(excess[left].min(), excess[right].min()))
