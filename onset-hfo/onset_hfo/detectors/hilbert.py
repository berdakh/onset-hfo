"""Hilbert-envelope HFO detector.

The feature is the amplitude envelope of the band-passed signal, taken from
the analytic signal and smoothed over the same short window the other
detectors use. Envelope-based detection is the family the MNI detector belongs
to, and the envelope responds to a burst's shape rather than to its
accumulated energy, so it rises and falls more sharply than RMS on the same
data.

**What this is not.** It is not a reimplementation of the MNI detector of
Zelmann et al., Clin Neurophysiol 123:106 (2012). That detector's
contribution is not the envelope -- which was already standard -- but an
*automatic baseline-selection* stage: it finds stretches of the recording that
are plausibly free of oscillatory activity, using a wavelet-entropy criterion,
and sets the threshold from those rather than from the whole channel. That
stage is the part that makes it work on long clinical recordings with varying
vigilance state, and it is absent here.

Calling this "the MNI detector" would therefore be wrong in the way that
matters: someone comparing our numbers to that paper's would be comparing a
threshold rule we did not implement. The honest description is *an
envelope-based detector sharing this project's baseline convention*, which is
median + k robust SD of the whole channel, the same convention the other three
use. Implementing the baseline-selection stage properly is a worthwhile piece
of work and is not this file.

**The threshold is inherited, not measured.** ``threshold_sd`` defaults to
5.0, copied from the energy detector on the assumption that the envelope and
the RMS trace have broadly similar distributions. Unlike the same assumption
made for short-time energy, that one survived contact with the data: these two
are the closest pair of the four, agreeing at Jaccard 0.79 on the synthetic
cohort, and they land at comparable event counts.

**And the sweep has now been run.** Across the 20 patients of ds003498 the
envelope's best threshold is **1.5 robust SD in the ripple band and 5.0 in
fast ripples** -- essentially what RMS wants (2.0 and 5.0), which is the
inherited default vindicated in the fast-ripple band and a little high in the
ripple band. The two really are the closest pair of the four, on real data as
on synthetic.

**What it does not buy is a better ranking.** At its own best threshold it
reaches rank agreement 0.613 in fast ripples against RMS's 0.610, and 0.578 in
ripples against RMS's 0.655 -- a tie in one band and clearly behind in the
other. Adding it was worth doing to learn that the disagreement between
detectors is structural; it was not worth doing to improve the answer. See
``docs/EVALUATION.md`` §1b.
"""

from __future__ import annotations

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import Event, sliding_hilbert_envelope
from onset_hfo.detectors.engine import detect_with_feature
from onset_hfo.preprocess import Prepared

__all__ = ["detect_hilbert"]


def detect_hilbert(prep: Prepared, cfg: DetectorConfig | None = None, **kwargs) -> list[Event]:
    """Detect band-limited events by the smoothed Hilbert envelope."""
    cfg = cfg or DetectorConfig()
    return detect_with_feature(prep, cfg, lambda x, w: sliding_hilbert_envelope(x, w),
                               "hilbert", **kwargs)
