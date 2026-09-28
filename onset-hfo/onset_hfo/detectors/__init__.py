"""The detectors.

Four HFO detectors (deliberately simple, deliberately different) and one
interictal-spike detector:

* :func:`detect_rms` -- energy-based, after Staba et al. (2002).
* :func:`detect_line_length` -- waveform-length based, after Gardner et al. (2007).
* :func:`detect_hilbert` -- the smoothed analytic-signal envelope. The feature
  family the MNI detector belongs to; **not** that detector, whose automatic
  baseline selection is not implemented here.
* :func:`detect_short_time_energy` -- the sum of squares, related to
  :func:`detect_rms` by a monotone transform and expected to be nearly
  redundant with it. Included so that redundancy is *measured* rather than
  assumed; see :func:`onset_hfo.metrics.agreement_matrix`.
* :func:`detect_spikes` -- amplitude + sharpness, an interictal epileptiform
  discharge (IED) detector.

They share the primitives in :mod:`onset_hfo.detectors.base`, so that the only
real difference between the HFO detectors is the feature they threshold. That
is what makes "the detectors disagree here" a meaningful statement.

**Only ``rms`` and ``line_length`` run by default.** The other two are opt-in
via ``run_pipeline(detectors=...)``, because their thresholds are inherited
rather than measured and because adding them silently would change every
number this project has already published.
"""

from onset_hfo.detectors.base import Event, events_to_frame, frame_to_events  # noqa: F401
from onset_hfo.detectors.hilbert import detect_hilbert  # noqa: F401
from onset_hfo.detectors.line_length import detect_line_length  # noqa: F401
from onset_hfo.detectors.rms import detect_rms  # noqa: F401
from onset_hfo.detectors.short_time_energy import detect_short_time_energy  # noqa: F401
from onset_hfo.detectors.spike import detect_spikes  # noqa: F401

#: Every HFO detector, by name. Excludes the spike detector, which answers a
#: different question and is not interchangeable with these.
#:
#: One registry, because there were four. `benchmark.py` kept its own copy and
#: was still listing two detectors months after four existed, so the sweep
#: silently could not see the new ones -- which is exactly the measurement
#: roadmap item 4 was waiting for.
HFO_DETECTORS = {"rms": detect_rms, "line_length": detect_line_length,
                 "hilbert": detect_hilbert,
                 "short_time_energy": detect_short_time_energy}

#: The HFO detectors plus the interictal-discharge detector.
DETECTORS = {**HFO_DETECTORS, "spike": detect_spikes}

__all__ = ["Event", "events_to_frame", "frame_to_events", "detect_rms",
           "detect_line_length", "detect_hilbert", "detect_short_time_energy",
           "detect_spikes", "DETECTORS", "HFO_DETECTORS"]
