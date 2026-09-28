"""Short-time-energy HFO detector.

The feature is the sum of squares in a sliding window -- the oldest and
simplest energy measure, and the one most implementations of "the Staba
detector" actually use, since the square root is often dropped as a monotone
transform that cannot change which samples cross a fixed threshold.

**A prediction written here was wrong, and the measurement is the point.**
This docstring previously argued that short-time energy would be nearly
redundant with RMS, on the grounds that ``energy = window * rms^2`` is a
monotone transform and could not change which samples cross a *fixed*
threshold. That reasoning is correct and the conclusion is not, because the
threshold here is not fixed: it is ``median + k * robustSD`` of the feature's
own distribution, and squaring is not affine.

Measured on the synthetic cohort, this is the pair that agrees **least**
(Jaccard 0.47 against RMS -- the lowest cell in the matrix), and the pair that
agrees most is RMS with the Hilbert envelope (0.79). The cause is visible
directly: ``median + 5 robustSD`` sits at roughly the 98th percentile of the
RMS trace and the 96th of the energy trace, so the same ``k`` is a materially
more permissive operating point on the squared feature. On one recording it
finds 294 events where RMS at 5.0 SD finds 132, and RMS has to come down to
about 2.5 SD (250 events) before the counts are comparable.

**Its higher F1 on the simulator is not a better detector** -- 0.82 against
0.68 for RMS -- because ``docs/EVALUATION.md`` §3 showed that lower thresholds
score higher F1 *on this simulator*, whose signal-to-noise distribution is a
guess. The generalisable conclusion stands and is now measured on real data:
**a threshold in robust SDs is not a portable operating point between
features.**

**Swept on real data, and the verdict here was too harsh.** This docstring
used to call it "mostly the energy detector at an untuned, lower operating
point". Half right. Across the 20 patients of ds003498 its own best threshold
is **4.0 robust SD in the ripple band and 12.0 in fast ripples**, against 2.0
and 5.0 for RMS -- so at a *shared* threshold it is badly mis-set, exactly as
claimed. But at its own, it matches RMS rather than trading recall for
precision: rank agreement 0.658 against 0.655 in ripples, 0.601 against 0.610
in fast ripples. A competitive detector that wants a different number, not a
mis-set copy of one. ``docs/EVALUATION.md`` §1b has the tables.

**Run it at RMS's threshold and it is genuinely worse**: 5.0 SD in the
fast-ripple band gives rank agreement 0.485 where its own 12.0 gives 0.601.
That gap is the whole reason this file says what it says.
"""

from __future__ import annotations

from onset_hfo.config import DetectorConfig
from onset_hfo.detectors.base import Event, sliding_energy
from onset_hfo.detectors.engine import detect_with_feature
from onset_hfo.preprocess import Prepared

__all__ = ["detect_short_time_energy"]


def detect_short_time_energy(prep: Prepared, cfg: DetectorConfig | None = None,
                             **kwargs) -> list[Event]:
    """Detect band-limited events by short-time energy (sum of squares)."""
    cfg = cfg or DetectorConfig()
    return detect_with_feature(prep, cfg, lambda x, w: sliding_energy(x, w),
                               "short_time_energy", **kwargs)
