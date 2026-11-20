"""Filter ringing on a sharp discharge: not a ripple, and still a discharge.

The largest single source of false HFO detections, and the one the
simulator's own background hides.
"""

from __future__ import annotations

import pytest

# -- a sharp discharge is not a ripple, and is a discharge -------------------
#
# The simulator's background is 1/f with a good share of its power in the
# ripple band, which hides filter ringing. These signals are steeper (1/f^2
# with a small white floor, as an amplifier has) and carry discharges with no
# oscillation in them at all.

def _coloured(n, sfreq, rng, alpha=2.0):
    import numpy as np

    spectrum = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    shape = np.ones_like(freqs)
    shape[1:] = freqs[1:] ** (-alpha / 2)
    shape[0] = 0
    x = np.fft.irfft(spectrum * shape, n=n)
    return x / np.std(x)


def _cusp(sfreq, amp=600.0, dur=0.07):
    import numpy as np

    t = np.linspace(-3, 3, int(dur * sfreq))
    w = -np.exp(-np.abs(t) / 0.35) + 0.45 * np.exp(-0.5 * ((t - 1.8) / 1.2) ** 2)
    return amp * w / np.max(np.abs(w))


def _pop(sfreq, amp=1000.0):
    import numpy as np

    return amp * np.exp(-np.arange(int(0.06 * sfreq)) / sfreq / 0.01)


def _discharges_only(shape, sfreq=2000.0, seconds=40.0, seed=1):
    import numpy as np

    from onset_hfo.preprocess import Prepared

    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    data = np.stack([50 * _coloured(n, sfreq, rng) + 2.0 * rng.normal(size=n)
                     for _ in range(2)])
    count = 0
    for row in range(2):
        for at in np.arange(1.0, seconds - 1.0, 1.0):
            wave = shape(sfreq)
            i = int((at + rng.uniform(0, 0.3)) * sfreq)
            data[row, i:i + wave.size] += wave * rng.uniform(0.7, 1.3)
            count += 1
    return Prepared(data=data, ch_names=["A1", "A2"], sfreq=sfreq, t_offset=0.0,
                    montage="monopolar"), count


@pytest.mark.parametrize("detector", ["rms", "line_length"])
def test_a_discharge_with_no_oscillation_in_it_is_not_counted_as_a_ripple(detector):
    """Every one of these passed 4 dB of prominence over the fitted 1/f line,
    "peaking" at the band's lower edge, and was accepted: a ripple rate that
    was a spike rate. There is energy in the band and no peak in it."""
    from onset_hfo.config import DetectorConfig, ValidationConfig
    from onset_hfo.detectors import DETECTORS
    from onset_hfo.validate import reason_kind, validate_events

    prep, count = _discharges_only(_cusp)
    threshold = 5.0 if detector == "rms" else 3.0
    events = DETECTORS[detector](prep, DetectorConfig(threshold_sd=threshold))
    assert len(events) >= 0.9 * count, "the ringing is there to be rejected"

    old = validate_events(list(events), prep, ValidationConfig(min_peak_contrast_db=0.0))
    assert sum(e.accepted for e in old) >= 0.9 * count, "and used to be accepted"

    validate_events(events, prep, ValidationConfig())
    assert sum(e.accepted for e in events) <= 0.02 * count
    kinds = {reason_kind(e.reject_reason) for e in events if not e.accepted}
    assert "no peak inside the band, probably ringing on a sharp transient" in kinds


def test_the_peak_contrast_of_a_real_ripple_is_far_above_the_floor(prepared):
    """What the new check costs: implanted ripples keep a wide margin."""
    import numpy as np

    from onset_hfo.config import DetectorConfig, ValidationConfig
    from onset_hfo.detectors import DETECTORS
    from onset_hfo.validate import validate_events

    events = DETECTORS["rms"](prepared, DetectorConfig())
    before = validate_events(list(events), prepared,
                             ValidationConfig(min_peak_contrast_db=0.0))
    kept_before = [e for e in before if e.accepted]
    measured = np.array([e.peak_contrast_db for e in kept_before])
    assert np.isfinite(measured).mean() > 0.95
    assert np.nanmedian(measured) > 3 * ValidationConfig().min_peak_contrast_db
    # On the simulator, whose events are real oscillations, it removes a few
    # percent of what the older checks accepted and no more.
    validate_events(events, prepared, ValidationConfig())
    assert sum(e.accepted for e in events) >= 0.95 * len(kept_before)


@pytest.mark.parametrize("sfreq", [500.0, 1000.0, 2000.0])
def test_a_sharp_discharge_is_a_discharge_at_any_sampling_rate(sfreq):
    """The pop test compared the largest sample-to-sample jump with that
    channel's own robust SD of jumps -- a statement about the sampling rate
    and the noise floor. At 2 kHz on a quiet contact it discarded every one
    of these as instrumentation."""
    from onset_hfo.detectors import DETECTORS

    prep, count = _discharges_only(_cusp, sfreq=sfreq)
    found = DETECTORS["spike"](prep)
    assert len(found) >= 0.9 * count


@pytest.mark.parametrize("sfreq", [500.0, 1000.0, 2000.0])
def test_an_electrode_pop_is_still_not_a_discharge(sfreq):
    from onset_hfo.detectors import DETECTORS

    prep, count = _discharges_only(_pop, sfreq=sfreq)
    assert len(DETECTORS["spike"](prep)) <= 0.05 * count


# -- broadband bursts and everything-at-once: flagged, never removed ---------


def _muscle(sfreq=2000.0, seconds=40.0, seed=3):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from onset_hfo.preprocess import Prepared

    rng = np.random.default_rng(seed)
    n = int(seconds * sfreq)
    data = np.stack([50 * _coloured(n, sfreq, rng) + 2.0 * rng.normal(size=n)
                     for _ in range(2)])
    sos = butter(4, [20, 400], btype="band", fs=sfreq, output="sos")
    for row in range(2):
        for at in np.arange(1.0, seconds - 1.0, 1.0):
            length = int(rng.uniform(0.1, 0.4) * sfreq)
            burst = sosfiltfilt(sos, rng.normal(size=length + 400))[200:200 + length]
            i = int((at + rng.uniform(0, 0.4)) * sfreq)
            data[row, i:i + length] += burst * 15 * np.hanning(length)
    return Prepared(data=data, ch_names=["A1", "B1"], sfreq=sfreq, t_offset=0.0,
                    montage="monopolar")


def test_a_broadband_burst_is_flagged_and_still_counted():
    """Muscle passes every test an oscillation passes inside the band. It is
    just as strong above it, which a ripple is not."""
    from onset_hfo.config import DetectorConfig
    from onset_hfo.detectors import DETECTORS
    from onset_hfo.validate import flag_note, flag_summary, validate_events

    prep = _muscle()
    events = validate_events(DETECTORS["rms"](prep, DetectorConfig()), prep)
    accepted = [e for e in events if e.accepted]
    assert len(accepted) > 30, "the bursts are accepted: nothing is rejected here"
    flagged = [e for e in accepted if "broadband" in e.flags]
    assert len(flagged) >= 0.75 * len(accepted)
    assert flag_summary(events)["broadband"] == len(flagged)
    assert "FLAGGED, still counted" in flag_note(events)


def test_most_real_ripples_are_not_flagged_broadband(prepared):
    from onset_hfo.config import DetectorConfig
    from onset_hfo.detectors import DETECTORS
    from onset_hfo.validate import validate_events

    events = validate_events(DETECTORS["rms"](prepared, DetectorConfig()), prepared)
    accepted = [e for e in events if e.accepted]
    assert sum("broadband" in e.flags for e in accepted) <= 0.2 * len(accepted)


def test_an_event_on_most_electrodes_at_once_is_flagged():
    from onset_hfo.config import ValidationConfig
    from onset_hfo.detectors.base import Event
    from onset_hfo.validate import flag_events

    def at(channel, start):
        return Event(channel=channel, start=start, stop=start + 0.03, detector="rms",
                     band=(80.0, 250.0), accepted=True)

    together = [at("A1-A2", 1.0), at("B1-B2", 1.005), at("C1-C2", 1.01), at("D1-D2", 1.0)]
    focal = [at("A1-A2", 5.0), at("A2-A3", 5.0)]       # one electrode, two pairs
    lone = [at("D3-D4", 9.0)]
    flag_events(together + focal + lone, ValidationConfig())
    assert all("multichannel" in e.flags for e in together)
    assert not any(e.flags for e in focal + lone)
    assert all(e.accepted for e in together)


def test_flags_survive_a_trip_through_the_saved_table(prepared):
    from onset_hfo.config import DetectorConfig
    from onset_hfo.detectors import DETECTORS
    from onset_hfo.detectors.base import events_to_frame, frame_to_events
    from onset_hfo.validate import validate_events

    events = validate_events(DETECTORS["rms"](prepared, DetectorConfig()), prepared)
    back = frame_to_events(events_to_frame(events))
    assert sorted(tuple(e.flags) for e in back) == sorted(tuple(e.flags) for e in events)
    assert any(e.flags for e in back)
