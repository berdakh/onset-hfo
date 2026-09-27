"""The shared detection engine.

Both HFO detectors follow the same five steps; they differ only in the feature
computed in step 2. Writing it once keeps the comparison between them honest:
any difference in their output comes from the feature, not from an
implementation accident.

1. Band-pass the channel (zero-phase FIR).
2. Compute a feature in a short sliding window (RMS energy, or line length).
3. Threshold it at ``median + k * MAD-SD`` of that channel's own feature trace.
4. Extend each crossing outwards while the feature stays above a lower
   hysteresis threshold, merge near-adjacent crossings, then require a
   minimum duration.
5. Require ``min_peaks`` rectified peaks above a lower secondary threshold, so
   that a single transient cannot pass as an oscillation.

Everything downstream (artifact validation, rates, reports) reads the
:class:`~onset_hfo.detectors.base.Event` objects produced here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import (
    ChannelBaseline,
    Event,
    bandpass,
    count_oscillation_peaks,
    extend_segments,
    robust_scale,
    threshold_segments,
)
from onset_hfo.preprocess import Prepared
from onset_hfo.spectral import event_spectral_peak

FeatureFn = Callable[[np.ndarray, int], np.ndarray]


def detect_with_feature(
    prep: Prepared,
    cfg: DetectorConfig,
    feature: FeatureFn,
    name: str,
    filtered: np.ndarray | None = None,
    channels: list[str] | None = None,
    describe_spectrum: bool = True,
    baselines: dict[str, ChannelBaseline] | None = None,
) -> list[Event]:
    """Run the five-step detection above on every requested channel.

    Parameters
    ----------
    prep:
        Preprocessed signals (microvolts).
    cfg:
        Thresholds and durations; see :class:`onset_hfo.config.DetectorConfig`.
    feature:
        ``f(signal, window_samples) -> feature trace`` of the same length.
    name:
        Detector name stamped onto every event.
    filtered:
        Pre-computed band-passed data, to avoid filtering twice when several
        detectors share a band. Must match ``prep.data``'s shape.
    channels:
        Restrict detection to these channels (default: all).
    describe_spectrum:
        Measure peak frequency and spectral prominence per event. Costs time;
        switch off for quick parameter sweeps.
    baselines:
        Per-channel median/MAD to threshold against, instead of measuring them
        from ``prep``. Used when a long recording is analysed in chunks: the
        baseline has to be a property of the whole recording, or the answer
        depends on where the chunk boundaries fell. See
        :mod:`onset_hfo.streaming`. A channel absent from the mapping falls
        back to measuring its own chunk, which is the right behaviour for the
        in-memory path and the wrong one for a chunk -- so the streaming code
        supplies every channel and checks that it did.
    """
    sf = prep.sfreq
    names = channels or prep.ch_names
    idx = [prep.ch_names.index(c) for c in names]
    if filtered is None:
        filtered = bandpass(prep.data[idx], sf, cfg.band)
        filt_lookup = {c: i for i, c in enumerate(names)}
    else:
        filt_lookup = {c: prep.ch_names.index(c) for c in names}

    win = max(1, int(round(cfg.rms_window_ms * sf / 1000.0)))
    min_len = max(1, int(round(cfg.min_duration_ms * sf / 1000.0)))
    max_len = int(round(cfg.max_duration_ms * sf / 1000.0))
    gap = max(1, int(round(cfg.merge_gap_ms * sf / 1000.0)))

    events: list[Event] = []
    for ch in names:
        x = filtered[filt_lookup[ch]]
        trace = feature(x, win)
        given = (baselines or {}).get(ch)
        if given is not None:
            center, scale = given.feature_center, given.feature_scale
            amp_scale = given.amplitude_scale
        elif cfg.baseline == "sd":
            center, scale = float(np.mean(trace)), float(np.std(trace) or np.finfo(float).eps)
            amp_scale = float(np.squeeze(robust_scale(x)[1]))
        else:
            c, s = robust_scale(trace)
            center, scale = float(np.squeeze(c)), float(np.squeeze(s))
            amp_scale = float(np.squeeze(robust_scale(x)[1]))
        threshold = center + cfg.threshold_sd * scale
        extend_threshold = center + cfg.extend_sd * scale
        peak_threshold = cfg.peak_threshold_sd * amp_scale

        segments = extend_segments(threshold_segments(trace, threshold, min_len, gap),
                                   trace, extend_threshold, gap)
        for i0, i1 in segments:
            if i1 - i0 > max_len:
                continue
            seg = x[i0:i1]
            n_peaks = count_oscillation_peaks(seg, peak_threshold)
            if n_peaks < cfg.min_peaks:
                continue
            start = prep.t_offset + i0 / sf
            stop = prep.t_offset + i1 / sf
            peak_freq = prominence = float("nan")
            if describe_spectrum:
                pad = int(round(0.05 * sf))
                raw_seg = prep.data[prep.ch_names.index(ch),
                                    max(0, i0 - pad): min(prep.data.shape[1], i1 + pad)]
                peak_freq, prominence = event_spectral_peak(raw_seg, sf, cfg.band)
            duration = stop - start
            events.append(Event(
                channel=ch, start=start, stop=stop, detector=name, band=tuple(cfg.band),
                score=float(np.max(trace[i0:i1]) / threshold) if threshold > 0 else 0.0,
                peak_amplitude_uv=float(np.max(np.abs(seg))),
                peak_frequency_hz=peak_freq,
                spectral_prominence_db=prominence,
                n_peaks=n_peaks,
                n_cycles=float(duration * peak_freq) if np.isfinite(peak_freq) else float("nan"),
                contacts=prep.contacts_of(ch),
            ))
    return events
