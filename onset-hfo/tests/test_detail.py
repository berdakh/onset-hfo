"""The event detail view, and the claim it is built on.

The panel's whole argument is that a real oscillation and filter ringing look
different in time-frequency: an island against a column. That is a falsifiable
claim about the computation, so it is tested here against signals whose nature
is known by construction — a windowed sine is an oscillation, a one-sample
step is not — rather than asserted in a caption and hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from onset_review import detail

SFREQ = 2000.0
PAD = 1.0          # seconds of signal either side, so the snapshot has room


@dataclass
class _Event:
    channel: str
    start: float
    stop: float
    detector: str = "rms"
    band: tuple = (80.0, 250.0)
    peak_amplitude_uv: float = 100.0
    peak_frequency_hz: float = 150.0
    spectral_prominence_db: float = 12.0
    n_peaks: int = 6
    n_cycles: float = 4.0
    accepted: bool = True
    reject_reason: str | None = None
    co_occurs_with_spike: bool = False
    duration_ms: float = 30.0


@dataclass
class _Session:
    raw: object
    events: list = field(default_factory=list)
    t_offset: float = 0.0


def _session_from(signal_uv: np.ndarray, channel: str = "A1-A2") -> _Session:
    import mne

    info = mne.create_info([channel], SFREQ, ch_types="seeg")
    raw = mne.io.RawArray(signal_uv[None, :] / 1e6, info, verbose="ERROR")
    return _Session(raw=raw)


def _background(seconds: float, seed: int = 0) -> np.ndarray:
    """Pink-ish noise: white noise shaped by 1/f, which is what a real
    recording's background looks like and what makes a flat-spectrum test
    misleadingly easy."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SFREQ)
    white = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SFREQ)
    spectrum[1:] /= np.sqrt(freqs[1:])
    # Scaled so that the in-band (80-250 Hz) noise is about 11 uV, which is
    # what a bipolar iEEG background actually looks like; a detectable ripple
    # is then a few times that rather than buried under it.
    return np.fft.irfft(spectrum, n) * 300.0


def _with_oscillation(centre_s: float, freq: float = 150.0,
                      cycles: float = 5.0, amplitude: float = 60.0,
                      seconds: float = 3.0) -> np.ndarray:
    """A Gaussian-windowed sine: an oscillation, by construction."""
    signal = _background(seconds)
    t = np.arange(signal.size) / SFREQ
    width = cycles / freq / 2.0
    envelope = np.exp(-0.5 * ((t - centre_s) / width) ** 2)
    return signal + amplitude * envelope * np.sin(2 * np.pi * freq * (t - centre_s))


def _with_transient(centre_s: float, amplitude: float = 600.0,
                    rise_s: float = 0.004, fall_s: float = 0.016,
                    seconds: float = 3.0) -> np.ndarray:
    """An interictal discharge: a sharp deflection, and not an oscillation.

    Shaped like the real thing -- a fast rise and a slower fall over about
    20 ms -- rather than as a one-sample impulse. An impulse is a tempting
    model and a wrong one: a two-sample doublet's spectrum is
    `sin(pi f dt)`, which *nulls* the low frequencies, so it is not broadband
    at all and tests nothing. A 20 ms spike is both what a band-pass filter
    actually turns into a convincing ripple and what is genuinely broadband.
    """
    signal = _background(seconds)
    t = np.arange(signal.size) / SFREQ - centre_s
    shape = np.where((t >= -rise_s) & (t < 0), (t + rise_s) / rise_s,
                     np.where((t >= 0) & (t < fall_s), 1.0 - t / fall_s, 0.0))
    return signal - amplitude * shape


def _hot_fraction(snap, decibels: float = 6.0) -> float:
    """Of the frequencies drawn, what fraction is lit up at the event's onset.

    This is the island-versus-column measurement, made numeric: a confined
    oscillation lights a few neighbouring rows, a broadband transient lights
    nearly all of them at once.
    """
    column = int(np.argmin(np.abs(snap.times)))
    return float((snap.power_db[:, column] > decibels).mean())


# -- the claim --------------------------------------------------------------


def test_an_oscillation_is_an_island():
    centre = 1.5
    session = _session_from(_with_oscillation(centre))
    event = _Event("A1-A2", centre - 0.015, centre + 0.015)
    snap = detail.snapshot(session, event)

    assert snap.available
    assert snap.power_db.size
    # Lit up where the oscillation is...
    peak_row = int(np.argmax(snap.power_db[:, int(np.argmin(np.abs(snap.times)))]))
    assert 120.0 < snap.freqs[peak_row] < 190.0
    # ...and not everywhere else at the same instant.
    assert _hot_fraction(snap) < 0.5


def test_a_sharp_transient_is_a_column():
    """The failure mode the panel exists to expose.

    Band-passed, this signal is a convincing-looking burst at the band's
    centre frequency. In time-frequency it is a line from the bottom of the
    plot to the top, which is what tells a reviewer it is the filter's work
    and not the brain's.
    """
    centre = 1.5
    session = _session_from(_with_transient(centre))
    event = _Event("A1-A2", centre - 0.015, centre + 0.015)
    snap = detail.snapshot(session, event)

    assert snap.available
    assert _hot_fraction(snap) > 0.7
    # And the trap itself: band-passed, this does look like a ripple, which is
    # why the filtered trace cannot be the evidence.
    assert np.ptp(snap.band_passed) > 4 * np.std(snap.band_passed[:200])


def test_the_two_are_told_apart_by_the_measurement_the_panel_shows():
    """Not merely different — separated, by a wide margin, in the direction
    the caption claims."""
    centre = 1.5
    island = detail.snapshot(_session_from(_with_oscillation(centre)),
                             _Event("A1-A2", centre - 0.015, centre + 0.015))
    column = detail.snapshot(_session_from(_with_transient(centre)),
                             _Event("A1-A2", centre - 0.015, centre + 0.015))
    assert _hot_fraction(column) - _hot_fraction(island) > 0.3


# -- the snapshot ------------------------------------------------------------


def test_the_snapshot_is_centred_on_the_event():
    centre = 1.5
    session = _session_from(_with_oscillation(centre))
    snap = detail.snapshot(session, _Event("A1-A2", centre, centre + 0.03))
    assert snap.onset == pytest.approx(0.0)
    assert snap.offset == pytest.approx(0.03, abs=1e-3)
    assert snap.duration_ms == pytest.approx(30.0, abs=1.0)
    assert snap.times[0] == pytest.approx(-detail.PAD_S, abs=0.01)
    assert snap.times[-1] == pytest.approx(0.03 + detail.PAD_S, abs=0.01)


def test_the_file_time_base_is_respected():
    """Events carry seconds in the original recording; the loaded window does
    not start there. Getting this wrong cuts a different event out."""
    centre = 1.5
    session = _session_from(_with_oscillation(centre))
    session.t_offset = 600.0                  # the window starts 10 min in
    event = _Event("A1-A2", 600.0 + centre, 600.0 + centre + 0.03)
    snap = detail.snapshot(session, event)
    peak_row = int(np.argmax(snap.power_db[:, int(np.argmin(np.abs(snap.times)))]))
    assert 120.0 < snap.freqs[peak_row] < 190.0


def test_an_event_at_the_edge_says_so_rather_than_failing():
    session = _session_from(_with_oscillation(0.05, seconds=1.0))
    snap = detail.snapshot(session, _Event("A1-A2", 0.04, 0.07))
    assert snap.available
    assert any("edge" in note for note in snap.notes)


def test_a_channel_that_is_not_in_the_window_draws_nothing():
    session = _session_from(_with_oscillation(1.5))
    snap = detail.snapshot(session, _Event("ZZ9-ZZ10", 1.4, 1.43))
    assert not snap.available


def test_the_frequency_axis_reaches_past_the_detectors_band_both_ways():
    """Downwards so the discharge under a ripple is visible, upwards so a
    ripple riding on a discharge is. A view restricted to the detector's own
    band would agree with the detector by construction."""
    session = _session_from(_with_oscillation(1.5))
    snap = detail.snapshot(session, _Event("A1-A2", 1.485, 1.515,
                                           detector="spike", band=(5.0, 60.0)))
    assert snap.freqs[0] < 20.0
    assert snap.freqs[-1] > 250.0


def test_the_colours_are_relative_to_the_signal_around_the_event():
    """Not to the event itself, and not to an absolute scale: against the 1/f
    background of a real recording an absolute scale shows the bottom of the
    plot lit and nothing else."""
    centre = 1.5
    session = _session_from(_with_oscillation(centre))
    snap = detail.snapshot(session, _Event("A1-A2", centre - 0.015, centre + 0.015))
    quiet = np.abs(snap.times) > 0.3
    # The background sits near 0 dB by construction of the baseline...
    assert abs(float(np.median(snap.power_db[:, quiet]))) < 3.0
    # ...and the event rises well clear of it.
    assert float(snap.power_db[:, np.argmin(np.abs(snap.times))].max()) > 8.0


def test_the_wideband_trace_is_the_preprocessed_signal_not_a_second_filtering():
    """The panel calls it wideband, not raw, and must not quietly filter it."""
    centre = 1.5
    signal = _with_oscillation(centre)
    session = _session_from(signal)
    snap = detail.snapshot(session, _Event("A1-A2", centre - 0.015, centre + 0.015))
    first = int(round((centre - 0.015 - detail.PAD_S) * SFREQ))
    expected = signal[first:first + snap.wideband.size]
    assert np.allclose(snap.wideband, expected, atol=1e-6)


def test_find_event_matches_the_key_a_verdict_is_filed_under():
    from onset_review.adjudication import event_key

    event = _Event("A1-A2", 1.485, 1.515)
    session = _session_from(_with_oscillation(1.5))
    session.events = [event]
    key = event_key(event.channel, event.start, event.detector)
    assert detail.find_event(session, key) is event
    assert detail.find_event(session, "A1-A2|99.000|rms") is None
