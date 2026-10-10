"""One event, close up: the three pictures it is actually judged on.

Everything else in this window is a count. This is the view that decides
whether a count means anything, because the question a reviewer has to answer
about an HFO cannot be answered from a number: **is this an oscillation, or is
it the filter ringing on a sharp transient?**

A band-pass filter applied to a spike produces something that looks exactly
like a ripple in the filtered trace -- a brief burst of oscillation at the
band's centre frequency, with the right duration and a plausible amplitude.
That artifact is the single largest source of false HFO detections in the
literature, and no amount of counting distinguishes it. Three pictures side by
side do:

1. **The wideband signal.** A real ripple rides on a background that does not
   itself jump; a filtered spike has a step, a sharp edge or a discharge
   underneath it that gives the game away immediately.
2. **The band-passed signal.** What the detector saw. On its own it is not
   evidence, which is the point of showing it next to the other two rather
   than alone.
3. **The time-frequency plot.** This is the discriminator. A genuine
   oscillation is an *island*: energy confined to a band of frequencies and a
   span of milliseconds. Filter ringing is a *column*: a sharp transient is
   broadband by definition, so its energy runs from the bottom of the plot to
   the top at one instant. The two are not subtle once seen together.

`onset_hfo.validate` already rejects candidates on a spectral-peak criterion
for exactly this reason, and `spectral_prominence_db` on every event is the
number that criterion is built from. This module does not second-guess it. It
shows the reviewer what that number was computed from, so that agreeing or
disagreeing with the software is something they can do from evidence rather
than from trust.

**"Wideband" is not "raw".** The signal here is the one the detector saw: high
-passed, notched and re-referenced by `onset_hfo.preprocess`. Saying "raw"
would be a lie in the one view whose whole job is to let someone check the
analysis, and a notch that removed a real harmonic would be invisible in a
plot that claimed to be unprocessed. The panel labels it accordingly, and the
preprocessing chain is one tab away.

Qt-free, like `trends` and `anatomy`: everything here is computable and
testable without a display, and `onset_review.eventview` only draws it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onset_hfo.modality import trace_scale

__all__ = ["Snapshot", "snapshot", "PAD_S", "N_FREQS", "find_event", "read_event",
           "read_contrast", "band_contrast"]

#: Seconds of signal shown on each side of the event. Half a second is enough
#: to see what led into it -- a discharge, a step, a movement artifact -- and
#: short enough that a 30 ms ripple is still a visible oscillation rather than
#: a thickening of the line. It is the same argument as the trace's default
#: window length, one order of magnitude down.
PAD_S = 0.5

#: Frequencies in the time-frequency plot. Log-spaced, because an octave is
#: the unit the eye reads a spectrum in and a linear axis spends half its
#: height on the top octave where nothing interesting happens.
N_FREQS = 60

#: Wavelet length, in cycles, as a function of frequency: `freq / 25`, held
#: between these bounds. Short wavelets on purpose. A long wavelet has fine
#: frequency resolution and smears in time, which is precisely the wrong trade
#: here: the question is whether energy is confined to an instant (ringing) or
#: lasts for several cycles (an oscillation), and a wavelet that smears every
#: transient across 200 ms cannot answer it. Three cycles is the floor below
#: which "oscillation" stops meaning anything.
CYCLES_PER_HZ = 1.0 / 25.0
MIN_CYCLES, MAX_CYCLES = 3.0, 8.0

#: Guard around the event, in seconds, excluded from the baseline along with
#: the event itself. A wavelet has support either side of its centre, so
#: samples just outside the event still carry its energy; including them in
#: the baseline would normalise the event partly against itself.
BASELINE_GUARD_S = 0.05


@dataclass
class Snapshot:
    """Everything the detail view draws, with nothing left to compute."""

    channel: str
    kind: str
    detector: str
    #: Seconds relative to the event's onset: negative before, positive after.
    times: np.ndarray
    #: Microvolts. `wideband` is the preprocessed signal -- see the module
    #: docstring on why it is not called raw.
    wideband: np.ndarray
    band_passed: np.ndarray
    band: tuple
    #: Event bounds in the same relative seconds as `times`.
    onset: float = 0.0
    offset: float = 0.0
    #: Time-frequency power in dB above the surrounding background, shaped
    #: (frequency, time). Empty when it could not be computed.
    freqs: np.ndarray = field(default_factory=lambda: np.empty(0))
    power_db: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    sfreq: float = 0.0
    #: The event's own measurements, for the caption.
    measurements: dict = field(default_factory=dict)
    #: Anything the reader has to know to read the pictures correctly.
    notes: tuple = ()

    @property
    def duration_ms(self) -> float:
        return (self.offset - self.onset) * 1000.0

    @property
    def available(self) -> bool:
        return self.times.size > 0


def find_event(session, key: str):
    """The event a verdict key names, or None.

    The key is `onset_review.adjudication`'s, so the detail view and the
    reader's verdict are talking about the same event by construction rather
    than by both re-deriving it from a table row.
    """
    from onset_review.adjudication import event_key

    for event in session.events:
        if event_key(event.channel, event.start, event.detector) == key:
            return event
    return None


def snapshot(session, event, pad_s: float = PAD_S) -> Snapshot:
    """Cut `event` out of the window and compute the three pictures.

    Returns an empty `Snapshot` rather than raising when the signal is not
    there: a panel that cannot draw an event must still be a panel, and the
    reviewer has eleven other ways to look at the same window.
    """
    raw = getattr(session, "raw", None)
    if raw is None or event is None:
        return Snapshot("", "", "", np.empty(0), np.empty(0), np.empty(0),
                        (0.0, 0.0))
    names = list(getattr(raw, "ch_names", []))
    if event.channel not in names:
        return Snapshot(event.channel, "", event.detector, np.empty(0),
                        np.empty(0), np.empty(0), tuple(event.band or (0, 0)))

    sfreq = float(raw.info["sfreq"])
    offset = float(getattr(session, "t_offset", 0.0))
    # File seconds to sample index in the loaded window.
    start_s = float(event.start) - offset
    stop_s = float(event.stop) - offset
    first = int(round((start_s - pad_s) * sfreq))
    last = int(round((stop_s + pad_s) * sfreq))
    total = int(raw.n_times)
    notes = []
    if first < 0 or last > total:
        notes.append("The event is near the edge of the loaded window, so "
                     "there is less signal on one side of it than the other.")
    first, last = max(0, first), min(total, last)
    if last - first < 8:
        return Snapshot(event.channel, _kind(event), event.detector,
                        np.empty(0), np.empty(0), np.empty(0),
                        tuple(event.band or (0, 0)), notes=tuple(notes))

    pick = names.index(event.channel)
    data = np.asarray(raw.get_data(picks=[pick], start=first, stop=last),
                      dtype=float)[0] * trace_scale(raw)   # SI back to µV, fT/cm or fT
    times = np.arange(data.size) / sfreq + (first / sfreq) - start_s

    band = tuple(float(v) for v in (event.band or (0.0, 0.0)))
    band_passed = _band_pass(data, sfreq, band)
    freqs, power_db, tfr_notes = _time_frequency(
        data, sfreq, band, times, 0.0, stop_s - start_s)
    notes.extend(tfr_notes)

    return Snapshot(
        channel=event.channel, kind=_kind(event), detector=event.detector,
        times=times, wideband=data, band_passed=band_passed, band=band,
        onset=0.0, offset=stop_s - start_s, freqs=freqs, power_db=power_db,
        sfreq=sfreq, measurements=_measurements(event), notes=tuple(notes))


def read_event(snap: Snapshot) -> dict:
    """The numbers the three pictures are made of, and what they read as.

    An oscillation is an island: energy confined to a band of frequencies and
    lasting several cycles. Filter ringing is a column: a sharp transient is
    broadband, so at the moment of the event its spectrum runs through the
    band without a peak. The prominence of the in-band peak, in dB, at the
    event's own moment and against the surrounding signal, is what the
    bottom picture shows; this reads it off and says which.
    """
    out = {"channel": snap.channel, "detector": snap.detector, "kind": snap.kind,
           "band_hz": list(snap.band), "duration_ms": round(float(snap.duration_ms), 1)}
    out.update({k: (round(v, 2) if isinstance(v, float) else v)
                for k, v in snap.measurements.items()})
    frequency = float(snap.measurements.get("frequency_hz", 0.0) or 0.0)
    cycles = (snap.duration_ms / 1000.0) * frequency if frequency > 0 else float("nan")
    out["cycles"] = round(cycles, 1) if np.isfinite(cycles) else None
    contrast = None
    if snap.available and snap.power_db.size and snap.freqs.size:
        during = (snap.times >= snap.onset) & (snap.times <= snap.offset)
        contrast = band_contrast(snap.power_db, snap.freqs, during, snap.band)
    out["band_contrast_db"] = round(contrast, 1) if contrast is not None else None
    reading, why = read_contrast(contrast, None if out["cycles"] is None else cycles, snap.band)
    out["reading"] = reading
    out["why"] = why
    out["notes"] = list(snap.notes)
    return out


def band_contrast(power_db: np.ndarray, freqs: np.ndarray, during: np.ndarray,
                  band: tuple) -> float | None:
    """How far the in-band peak of the event's own spectrum stands above the
    troughs either side of it, in dB: the peak's prominence.

    The spectrum at the event's moment is averaged over the event, its
    highest point inside the band found, and the lowest points on either
    side of that peak -- down to half an octave below the band, up to an
    octave above it or as far as the picture goes -- taken; the contrast is
    the peak over the higher of the two troughs. An oscillation makes a peak
    in the band; a sharp transient's energy runs through the band without
    one, however much of it there is. Prominence rather than a mean over the
    band, because a ripple at 90 Hz is not made less of a ripple by the
    band running to 250 Hz, and rather than a comparison with everything
    below the band, because the discharge a ripple rides on puts its energy
    at 5-40 Hz and is part of the finding, not evidence against it.
    """
    if not power_db.size or not freqs.size or not during.any():
        return None
    lo, hi = (float(band[0]), float(band[1])) if band else (0.0, 0.0)
    if not (0 < lo < hi):
        return None
    moment = np.nanmean(power_db[:, during], axis=1)
    inside = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if inside.size == 0:
        return None
    peak = inside[int(np.nanargmax(moment[inside]))]
    left = np.flatnonzero((freqs >= 0.5 * lo) & (np.arange(freqs.size) < peak))
    right = np.flatnonzero((freqs <= 2.0 * hi) & (np.arange(freqs.size) > peak))
    troughs = [float(np.nanmin(moment[side])) for side in (left, right) if side.size]
    if not troughs:
        return None
    return float(moment[peak] - max(troughs))


def read_contrast(contrast: float | None, cycles: float | None, band: tuple) -> tuple[str, str]:
    """The reading -- island, column or unclear -- from the band contrast in
    dB and the length in cycles, in the words the panel uses. Shared with the
    average-event view so one event and the mean of fifty are read by the
    same rule."""
    lo, hi = (float(band[0]), float(band[1])) if band else (0.0, 0.0)
    if contrast is None or cycles is None:
        return "unclear", "the time-frequency picture could not be computed here"
    if contrast >= 3.0 and cycles >= 3.0:
        return "island", (
            f"at the event's moment the spectrum peaks inside {lo:.0f}–{hi:.0f} Hz, "
            f"{contrast:.1f} dB above the troughs either side, over {cycles:.1f} cycles: "
            "confined in frequency and sustained, which is what an oscillation looks like")
    if contrast < 1.5:
        return "column", (
            f"at the event's moment the spectrum has no real peak inside the band, only "
            f"{contrast:.1f} dB above the troughs either side: broadband, which is what a "
            "sharp transient and the filter's ringing look like, whatever the band-passed "
            "trace shows")
    if cycles < 3.0:
        return "unclear", (
            f"confined in frequency ({contrast:.1f} dB) but only {cycles:.1f} cycles long; "
            "too short to call an oscillation on this picture")
    return "unclear", (
        f"an in-band peak only {contrast:.1f} dB above the troughs either side over "
        f"{cycles:.1f} cycles: between an island and a column; open it on the trace")


def _kind(event) -> str:
    if event.detector == "spike":
        return "discharge"
    band = event.band or (0.0, 0.0)
    return "fast ripple" if float(band[0]) >= 200 else "ripple"


def _measurements(event) -> dict:
    return {
        "duration_ms": float(getattr(event, "duration_ms", 0.0)),
        "amplitude_uv": float(event.peak_amplitude_uv),
        "frequency_hz": float(event.peak_frequency_hz),
        "prominence_db": float(event.spectral_prominence_db),
        "n_peaks": int(event.n_peaks),
        "n_cycles": float(getattr(event, "n_cycles", float("nan"))),
        "accepted": bool(event.accepted),
        "reject_reason": event.reject_reason or "",
        "with_spike": bool(event.co_occurs_with_spike),
    }


def _band_pass(data: np.ndarray, sfreq: float, band: tuple) -> np.ndarray:
    """The detector's own filter, on this snippet.

    `onset_hfo.detectors.base.bandpass` rather than a second implementation,
    so that what the reviewer is shown is what the detector saw and not
    something that merely resembles it.
    """
    low, high = (float(band[0]), float(band[1])) if band else (0.0, 0.0)
    if not (0 < low < high) or high >= sfreq / 2:
        high = min(high, 0.45 * sfreq)
    if not (0 < low < high):
        return np.zeros_like(data)
    from onset_hfo.detectors.base import bandpass

    return bandpass(data[None, :], sfreq, (low, high))[0]


def _time_frequency(data: np.ndarray, sfreq: float, band: tuple,
                    times: np.ndarray, onset: float, offset: float):
    """Morlet power, in dB above the signal either side of the event.

    The baseline is the surrounding signal with the event (and a guard band
    for the wavelets' own support) left out, per frequency. Normalising this
    way is what turns the plot into the test it is meant to be: against a flat
    background the eye cannot separate "a lot of energy here" from "a lot of
    energy at this frequency everywhere", and the 1/f slope of every brain
    recording guarantees the second. Against the neighbouring 400 ms, an
    island is an island and a column is a column.

    When the event fills the snippet there is nothing left to be a baseline;
    the whole snippet is used instead and the caller is told so, because a
    normalisation that includes the event understates it.
    """
    notes = []
    # Wider than the detector's own band, in both directions, and deliberately.
    # Downwards, because the discharge a ripple is riding on is the thing that
    # explains it, and it lives at 5-30 Hz where the band-pass cannot show it.
    # Upwards to at least the ripple range, because "does an HFO ride on this
    # discharge" is a question about a spike, and a plot that stopped at the
    # spike band could not answer it. Restricting the view to the band the
    # detector used would make this picture agree with the detector by
    # construction, which is the opposite of what it is for.
    low = 8.0
    high = min(0.45 * sfreq,
               max(float(band[1]) * 1.25 if band and band[1] else 0.0, 300.0))
    if high <= low * 1.1 or data.size < 32:
        return np.empty(0), np.empty((0, 0)), notes

    freqs = np.logspace(np.log10(low), np.log10(high), N_FREQS)
    cycles = np.clip(freqs * CYCLES_PER_HZ, MIN_CYCLES, MAX_CYCLES)
    # A wavelet needs to fit inside the snippet, and MNE raises rather than
    # truncating when it does not. Its Morlet runs to five standard deviations
    # each side, and sigma is `n_cycles / (2 pi f)`, so the kernel is
    # `10 * n_cycles / (2 pi f)` seconds long -- about 1.6 times the naive
    # `n_cycles / f`, which is the factor that made the first version of this
    # crash on an event near the edge of a window. The 0.9 is margin.
    longest = 10.0 * cycles / (2.0 * np.pi * freqs)
    fits = longest <= (data.size / sfreq) * 0.9
    if not fits.any():
        return np.empty(0), np.empty((0, 0)), notes
    if not fits.all():
        notes.append("The lowest frequencies are not shown: the wavelet they "
                     "need is longer than the signal around this event.")
    freqs, cycles = freqs[fits], cycles[fits]

    import warnings

    from mne.time_frequency import tfr_array_morlet

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            power = tfr_array_morlet(
                data[None, None, :], sfreq=sfreq, freqs=freqs, n_cycles=cycles,
                output="power", verbose="ERROR")[0, 0]
    except (ValueError, MemoryError) as error:
        # The two traces are still worth drawing without the third, and a
        # panel that takes the window down because one event sits awkwardly in
        # the recording is worse than a panel with one picture missing.
        notes.append(f"No time-frequency plot for this event: {error}")
        return np.empty(0), np.empty((0, 0)), notes

    outside = (times < onset - BASELINE_GUARD_S) | (times > offset + BASELINE_GUARD_S)
    if outside.sum() < max(8, int(0.05 * times.size)):
        outside = np.ones_like(times, dtype=bool)
        notes.append("The event fills most of the window around it, so the "
                     "colours are relative to a background that includes the "
                     "event itself and understate it.")
    baseline = np.median(power[:, outside], axis=1, keepdims=True)
    baseline = np.maximum(baseline, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny)
                                   / baseline)
    return freqs, np.nan_to_num(power_db, nan=0.0, posinf=0.0, neginf=0.0), notes
