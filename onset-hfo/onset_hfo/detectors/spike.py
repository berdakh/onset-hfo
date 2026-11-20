"""Interictal epileptiform discharge (IED) detector.

"Epileptiform discharge" here means the classic interictal spike or sharp
wave: a brief (20-200 ms), high-amplitude, *sharp* deflection that stands out
from the background of its own channel.

The algorithm is deliberately the simplest thing that encodes what a reviewer
actually looks for:

1. band-pass 5-60 Hz (removes drift and most muscle/HF noise, keeps the
   spike's shape);
2. find peaks whose amplitude exceeds ``threshold_sd`` robust SDs of that
   channel;
3. require *sharpness*: the steepest slope inside the candidate must exceed
   ``slope_sd`` robust SDs of the channel's first derivative -- this is what
   separates a spike from a large slow wave;
4. take the event boundaries at half of the peak amplitude, and keep the event
   only if its duration falls in the accepted range;
5. reject candidates that sit on a *discontinuity* in the unfiltered signal:
   after a 5-60 Hz band-pass an electrode pop looks like a perfectly good
   sharp wave, so this check is made on the raw samples, where a pop jumps in
   one sample and a discharge does not;
6. enforce a refractory period so one discharge is counted once.

It is not a trained classifier and it does not try to be: it is a transparent
baseline that a clinician can argue with, and the number it produces
(discharges per minute per channel) is the quantity every HFO study reports
alongside HFO rate.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from onset_hfo.config import SpikeConfig
from onset_hfo.detectors.base import Event, bandpass, robust_scale
from onset_hfo.preprocess import Prepared

__all__ = ["detect_spikes"]


def detect_spikes(prep: Prepared, cfg: SpikeConfig | None = None,
                  channels: list[str] | None = None, **_: object) -> list[Event]:
    """Detect interictal epileptiform discharges on every requested channel."""
    cfg = cfg or SpikeConfig()
    sf = prep.sfreq
    names = channels or prep.ch_names
    idx = [prep.ch_names.index(c) for c in names]
    filt = bandpass(prep.data[idx], sf, cfg.band)
    # Unfiltered sample-to-sample differences: the artifact test below reads
    # these, because filtering is exactly what hides a discontinuity.
    raw_diff = np.diff(prep.data[idx], axis=1, prepend=prep.data[idx][:, :1])
    _, raw_diff_scale = robust_scale(raw_diff)
    raw_diff_scale = np.asarray(raw_diff_scale).ravel()

    refractory = max(1, int(round(cfg.refractory_ms * sf / 1000.0)))
    min_len = cfg.min_duration_ms * sf / 1000.0
    max_len = cfg.max_duration_ms * sf / 1000.0

    events: list[Event] = []
    for row, ch in enumerate(names):
        x = filt[row]
        _, scale = robust_scale(x)
        scale = float(np.squeeze(scale))
        amp_threshold = cfg.threshold_sd * scale
        deriv = np.diff(x, prepend=x[:1]) * sf
        _, dscale = robust_scale(deriv)
        slope_threshold = cfg.slope_sd * float(np.squeeze(dscale))

        peaks, props = find_peaks(np.abs(x), height=amp_threshold, distance=refractory)
        for p, height in zip(peaks, props["peak_heights"], strict=False):
            half = 0.5 * height
            left = p
            while left > 0 and abs(x[left]) > half and p - left < max_len:
                left -= 1
            right = p
            n = x.size
            while right < n - 1 and abs(x[right]) > half and right - p < max_len:
                right += 1
            width = right - left
            if not (min_len <= width <= max_len):
                continue
            if np.max(np.abs(deriv[left:right + 1])) < slope_threshold:
                continue  # large but not sharp: a slow wave, not a spike
            jump = float(np.max(np.abs(raw_diff[row, max(0, left - 5):right + 5])))
            if (jump > cfg.max_raw_jump_sd * float(raw_diff_scale[row])
                    and _is_step(prep.data[idx[row], max(0, left - 5):right + 5],
                                 sf, cfg)):
                continue  # a discontinuity: instrumentation, not brain
            start = prep.t_offset + left / sf
            stop = prep.t_offset + right / sf
            events.append(Event(
                channel=ch, start=start, stop=stop, detector="spike", band=tuple(cfg.band),
                score=float(height / amp_threshold) if amp_threshold > 0 else 0.0,
                peak_amplitude_uv=float(height),
                n_peaks=1,
                contacts=prep.contacts_of(ch),
            ))
    return events


def _is_step(window: np.ndarray, sfreq: float, cfg: SpikeConfig) -> bool:
    """Whether the unfiltered signal crosses most of its excursion at once.

    The fraction of the window's peak-to-peak range covered within
    `cfg.step_ms`. It is a property of the waveform's shape: the same
    discharge gives the same answer at 500 Hz and at 2 kHz, on a quiet
    contact and a noisy one, which the jump in robust SDs does not.
    """
    limit = float(getattr(cfg, "max_step_fraction", 0.0) or 0.0)
    if limit <= 0:
        return True                     # the SD test alone, as it was
    span = float(np.ptp(window)) if window.size else 0.0
    if span <= 0:
        return False
    lag = max(1, int(round(float(cfg.step_ms) * sfreq / 1000.0)))
    if window.size <= lag:
        return False
    moved = float(np.max(np.abs(window[lag:] - window[:-lag])))
    return moved / span > limit
