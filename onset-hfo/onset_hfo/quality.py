"""Which contacts, and which seconds, are fit to be analysed.

An HFO rate ranking is unusually easy to poison, and both ways it fails look
like findings.

A contact with a noisy amplifier produces band-limited energy continuously.
The detector finds it. :mod:`onset_hfo.validate` cannot throw it out, because
its checks ask whether a genuine narrow-band oscillation is present and the
answer is yes -- it just is not the brain's. That contact then tops the
ranking, with a tight Poisson interval, and every panel in the reviewer agrees
with every other panel about it. A dead contact fails the other way: no
events, bottom of the table, reassuring.

So this module answers two questions before any rate is computed.

**Which channels are fit?** Five checks, all on the preprocessed signal, none
needing a coordinate: flat, clipped at the amplifier's rail, dominated by
mains, an amplitude outlier against the rest of the montage, and -- the one
that matters most -- an outlier in the ratio of in-band to broadband power.

**Which seconds are fit?** The window is cut into fixed segments and each
channel's segments are tested against *that channel's own* distribution.
Contacts differ by an order of magnitude in background amplitude, so one
absolute threshold would scrub the quiet ones and ignore the loud ones.

Two decisions are worth stating plainly, because both are the opposite of
what a conventional M/EEG cleaner does.

**Nothing is interpolated.** ``autoreject`` and its relatives repair a bad
channel from its neighbours, which is right when the quantity of interest is
an evoked response averaged across sensors, and wrong here: the output of this
software is a *per-channel* rate, so an interpolated channel's rate is
borrowed from the contacts beside it and would be read as a finding about that
contact. A bad channel is dropped and named. (On these two archives it is also
moot: interpolation needs contact coordinates, and neither ships any. MNE will
refuse, loudly, and it is right to.)

**Nothing is deleted.** A rejected channel keeps its row and gains a reason; a
rejected segment keeps its place in the timeline. The reviewer sees what was
excluded and why, and can overrule it. This is the same rule
:mod:`onset_hfo.validate` follows for events, for the same reason: a cleaner
that silently removes data is indistinguishable from a bug.

The cost of being wrong here is asymmetric, and it decided the design.
Keeping a little noise costs some precision. Rejecting the seconds around an
interictal discharge removes the time that carries the pathology, biases every
rate downward where it matters most, and says nothing at all. The first
version of the segment test used peak-to-peak amplitude and did exactly that:
on sub-01 of ds003498 it threw away six seconds of `AR2-AR3`, the second
busiest HFO channel in the window, because an interictal discharge is a large
deflection. It now tests for a *discontinuity* instead, which is the statistic
this project had already measured as separating discharges (3-5 SD) from
artifact steps (25-64 SD). `tests/test_quality.py` pins both halves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from onset_hfo.config import BANDS, RAYLEIGH_BURSTINESS, QualityConfig
from onset_hfo.detectors.base import robust_scale
from onset_hfo.preprocess import Prepared

__all__ = ["channel_quality", "segment_quality", "clean_seconds",
           "analysable_seconds", "usable_channels", "quality_summary",
           "reject_unusable_events", "REASONS", "SET_ASIDE"]

#: Reasons that **set a contact aside**: faults where no physiology produces
#: the signal, so there is nothing to adjudicate. Everything else in `REASONS`
#: that applies to a channel is a flag -- the contact is analysed normally and
#: a human is asked to look at it, because the measurement is as consistent
#: with the finding as with a fault. See `QualityConfig.max_hf_ratio_sd` for
#: the recording that settled which is which.
SET_ASIDE = ("flat", "clipped", "line_noise", "mostly_bad_segments")

#: What each verdict means, in a sentence a reviewer can act on. The interface
#: and the report both read these, so there is one wording per reason.
REASONS: dict[str, str] = {
    "flat": "almost no signal — a disconnected, shorted or dead contact",
    "clipped": "pinned at the amplifier's limit for part of the window; a "
               "clipped edge rings through an 80–250 Hz filter like a ripple",
    "line_noise": "mains interference dominates; its harmonics land inside "
                  "the band being counted",
    "hf_noise": "far more high-frequency energy than the rest of this "
                "montage, and it arrives as a steady carpet rather than as "
                "events — which is what a noisy or poorly-coupled contact "
                "looks like. Analysed and flagged, not removed: look at it",
    "hf_active": "far more high-frequency energy than the rest of this "
                 "montage, and it arrives in discrete bursts over a quiet "
                 "floor rather than as a steady carpet. That is what a "
                 "contact full of real ripples looks like — and also what a "
                 "repeating artifact looks like. Open it on the trace",
    "amplitude": "overall amplitude far outside what the rest of this "
                 "montage is doing. A gain fault looks like this, and so "
                 "does a contact sitting where the pathology is; analysed "
                 "and flagged rather than removed",
    "mostly_bad_segments": "too little of the window survived segment "
                           "rejection to call a rate",
    "annotated_muscle": "broadband high-frequency power rose across the whole "
                        "montage at once here, as muscle and movement do; marked "
                        "by the preprocessing stage's muscle annotator",
    "annotated_amplitude": "peak-to-peak amplitude above the ceiling the "
                           "preprocessing stage was given, or learned, for this "
                           "contact; marked, never interpolated",
    "segment_jump": "the unfiltered signal steps discontinuously here — a "
                    "pop or a disconnection, not a discharge, which is sharp "
                    "but continuous",
    "segment_ceiling": "peak-to-peak beyond any amplitude the brain "
                       "produces, whatever the rest of this channel does",
    "segment_flat": "no signal at all in this second — the recording "
                    "dropped out, or the contact did",
}


def channel_quality(prep: Prepared, cfg: QualityConfig | None = None,
                    segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per channel: the five measurements, the verdict and the reason.

    Every channel is listed, including the ones that pass, because "we looked
    and it was fine" is what makes the ones that failed meaningful.

    ``good`` means the contact was analysed; ``flagged`` means it was
    analysed *and* something about it is worth a look. The two are different
    questions and the distinction is the point -- see ``SET_ASIDE``.

    Columns: ``channel``, ``amplitude_uv``, ``clipped_fraction``,
    ``line_fraction``, ``hf_ratio``, ``burstiness``, ``amplitude_sd``,
    ``hf_ratio_sd``, ``bad_segment_fraction``, ``good``, ``flagged`` and
    ``reason``.
    """
    cfg = cfg or QualityConfig()
    data, names = prep.data, list(prep.ch_names)
    if not names:
        return pd.DataFrame(columns=["channel", "amplitude_uv",
                                     "clipped_fraction", "line_fraction",
                                     "hf_ratio", "burstiness", "amplitude_sd",
                                     "hf_ratio_sd", "bad_segment_fraction",
                                     "good", "flagged", "reason"])

    _, scale = robust_scale(data)
    amplitude = scale.ravel().astype(float)
    clipped = np.array([_clipped_fraction(row) for row in data])
    line = _line_fraction(data, prep.sfreq, prep.line_freq)
    hf_ratio = _band_fraction(data, prep.sfreq, BANDS.ripple)
    # Within-channel dynamic range of the band envelope. Unlike the ratio
    # above it needs no comparison with the montage, which is what lets it
    # say *why* a contact's band power is high.
    burstiness = _burstiness(data, prep.sfreq, BANDS.ripple)

    # Both outlier tests are against *this montage's* own distribution rather
    # than an absolute number, because a depth contact in white matter and one
    # in hippocampus do not have the same background, and neither does an
    # electrode from a different manufacturer.
    amplitude_sd = _robust_z(np.log(np.maximum(amplitude, 1e-12)))
    hf_ratio_sd = _robust_z(hf_ratio)
    # ...and, for amplitude only, a plain ratio to the montage median beside
    # it: a robust SD across fifteen channels is not a reliable scale, and the
    # SD test alone throws ordinary contacts out of a small montage. The band
    # ratio gets no such guard, because its absolute value is not comparable
    # between recordings -- see `QualityConfig.amplitude_outlier_ratio`.
    amplitude_ratio = _ratio_to_median(amplitude)

    bad_fraction = _bad_segment_fraction(segments, names)

    rows = []
    for i, name in enumerate(names):
        reason = ""
        if amplitude[i] < cfg.flat_uv:
            reason = "flat"
        elif clipped[i] > cfg.max_clipped_fraction:
            reason = "clipped"
        elif line[i] > cfg.max_line_fraction:
            reason = "line_noise"
        elif hf_ratio_sd[i] > cfg.max_hf_ratio_sd:
            # Both are flags and neither removes anything; the split only
            # changes which sentence the reviewer reads, and the measured
            # ratio is shown beside it.
            reason = ("hf_active"
                      if burstiness[i] >= cfg.bursty_ratio * RAYLEIGH_BURSTINESS
                      else "hf_noise")
        elif (abs(amplitude_sd[i]) > cfg.max_amplitude_sd
                and (amplitude_ratio[i] > cfg.amplitude_outlier_ratio
                     or amplitude_ratio[i] < 1.0 / cfg.amplitude_outlier_ratio)):
            reason = "amplitude"
        elif bad_fraction.get(name, 0.0) > cfg.max_bad_segment_fraction:
            reason = "mostly_bad_segments"
        rows.append({
            "channel": name,
            "amplitude_uv": float(amplitude[i]),
            "clipped_fraction": float(clipped[i]),
            "line_fraction": float(line[i]),
            "hf_ratio": float(hf_ratio[i]),
            "burstiness": float(burstiness[i]),
            "amplitude_sd": float(amplitude_sd[i]),
            "hf_ratio_sd": float(hf_ratio_sd[i]),
            "bad_segment_fraction": float(bad_fraction.get(name, 0.0)),
            # `good` is "was it analysed", which a flagged contact was.
            "good": reason not in SET_ASIDE,
            "flagged": bool(reason) and reason not in SET_ASIDE,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def segment_quality(prep: Prepared, cfg: QualityConfig | None = None) -> pd.DataFrame:
    """One row per channel per segment: whether that second is usable, and why.

    What is tested is **discontinuity**, not amplitude: the largest
    sample-to-sample jump in the segment, in robust SDs of that channel's own
    difference distribution. Per channel because contacts differ by an order
    of magnitude in background, and by jump because amplitude cannot tell an
    interictal discharge from an electrode pop -- see
    :class:`onset_hfo.config.QualityConfig.segment_jump_sd` for the real
    recording that settled it.

    Columns: ``channel``, ``segment``, ``t_start``, ``t_stop``, ``ptp_uv``,
    ``jump_sd``, ``good``, ``reason``. Times are in original-recording seconds.
    """
    cfg = cfg or QualityConfig()
    data, names = prep.data, list(prep.ch_names)
    width = max(1, int(round(cfg.segment_s * prep.sfreq)))
    n_segments = int(data.shape[1] // width)
    if n_segments == 0 or not names:
        return pd.DataFrame(columns=["channel", "segment", "t_start", "t_stop",
                                     "ptp_uv", "jump_sd", "good", "reason"])

    # Trailing samples that do not fill a segment are left out of both the
    # numerator and the denominator rather than padded into a short segment
    # whose statistics would not be comparable with the others'.
    kept = data[:, :n_segments * width]
    block = kept.reshape(len(names), n_segments, width)
    ptp = block.max(axis=2) - block.min(axis=2)

    # The difference is taken across the whole window and then reshaped, so a
    # step that happens to land on a segment boundary is still seen. Taking it
    # per segment would miss exactly the discontinuities at the edges.
    diff = np.diff(kept, axis=1, prepend=kept[:, :1])
    _, diff_scale = robust_scale(diff)
    jump = np.abs(diff).reshape(len(names), n_segments, width).max(axis=2)
    jump_sd = jump / np.maximum(diff_scale, np.finfo(float).eps)

    starts = prep.t_offset + np.arange(n_segments) * (width / prep.sfreq)
    stops = starts + width / prep.sfreq
    # Seconds the preprocessing stage's annotators marked, per channel or for
    # every channel: a segment that overlaps one is set aside with that reason.
    marked: dict[str | None, list[tuple[float, float, str]]] = {}
    for note in getattr(prep, "annotations", None) or []:
        marked.setdefault(note.get("channel"), []).append(
            (float(note["t_start"]), float(note["t_stop"]), str(note.get("reason") or "annotated")))

    def annotated(name: str, j: int) -> str:
        for key in (None, name):
            for t0, t1, why in marked.get(key, ()):
                if t0 < stops[j] and t1 > starts[j]:
                    return why
        return ""

    rows = []
    for i, name in enumerate(names):
        for j in range(n_segments):
            spread, step = float(ptp[i, j]), float(jump_sd[i, j])
            if spread <= 0.0:
                reason = "segment_flat"
            elif spread > cfg.segment_ceiling_uv:
                reason = "segment_ceiling"
            elif step > cfg.segment_jump_sd:
                reason = "segment_jump"
            else:
                reason = annotated(name, j) if marked else ""
            rows.append({
                "channel": name,
                "segment": j,
                "t_start": float(starts[j]),
                "t_stop": float(starts[j] + width / prep.sfreq),
                "ptp_uv": spread,
                "jump_sd": step,
                "good": not reason,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def clean_seconds(segments: pd.DataFrame, channels: list[str] | None = None,
                  total_s: float | None = None) -> dict[str, float]:
    """How many seconds of each channel survived. The rate's denominator.

    This is the number that makes segment rejection honest. Dividing every
    channel by the window's nominal length after throwing away four seconds of
    one of them reports a rate per minute of a minute that was not analysed,
    and the Poisson interval inherits the error.
    """
    if segments is None or segments.empty:
        if channels is None or total_s is None:
            return {}
        return {name: float(total_s) for name in channels}
    span = (segments["t_stop"] - segments["t_start"]).astype(float)
    kept = segments.assign(seconds=span.where(segments["good"], 0.0))
    out = kept.groupby("channel")["seconds"].sum().to_dict()
    for name in channels or []:
        out.setdefault(name, 0.0)
    return {str(k): float(v) for k, v in out.items()}


def analysable_seconds(segments: pd.DataFrame | None,
                      quality: pd.DataFrame | None,
                      channels: list[str] | None = None,
                      total_s: float | None = None,
                      keep: tuple[str, ...] = ()) -> dict[str, float]:
    """Clean seconds, but **zero** for a channel the checks set aside entirely.

    The distinction matters in the table a reviewer reads. A channel with some
    clean time left gets a rate over that time. A channel that was set aside
    gets no rate at all -- not zero, which would read as "we looked for eight
    seconds and found nothing" when in fact nothing on it was looked at. Its
    row stays, its verdict is in the quality table, and its rate is blank.
    """
    seconds = clean_seconds(segments, channels, total_s)
    if quality is None or quality.empty:
        return seconds
    overruled = set(keep)
    for row in quality.itertuples():
        name = str(row.channel)
        if not row.good and name not in overruled:
            seconds[name] = 0.0
    return seconds


def usable_channels(quality: pd.DataFrame,
                    keep: tuple[str, ...] = ()) -> list[str]:
    """The channels to analyse. `keep` is the reviewer overruling the machine.

    An override is deliberately one-directional: a reviewer can reinstate a
    channel this module rejected, having looked at it. Excluding a channel the
    module passed is already possible, and more honestly recorded, through
    `PreprocessConfig.exclude`.
    """
    if quality is None or quality.empty:
        return []
    overruled = set(keep)
    return [str(row.channel) for row in quality.itertuples()
            if row.good or str(row.channel) in overruled]


def quality_summary(quality: pd.DataFrame, segments: pd.DataFrame | None = None,
                    kept: tuple[str, ...] = ()) -> str:
    """One sentence for the status bar and the report.

    Set-aside and flagged are reported separately and always in that order,
    because they ask different things of the reviewer: one is a channel that
    was not analysed, the other is a channel that was and wants a look.
    """
    if quality is None or quality.empty:
        return "No channels were checked."
    aside = quality[~quality["good"]]
    flagged = quality[quality.get("flagged", False)]
    names = [str(c) for c in aside["channel"]]
    reinstated = [n for n in names if n in set(kept)]

    parts = []
    if names:
        detail = ", ".join(f"{int(n)} {reason.replace('_', ' ')}"
                           for reason, n in aside["reason"].value_counts().items())
        parts.append(f"{len(names)} of {len(quality)} contacts set aside "
                     f"({detail})")
        if reinstated:
            parts.append(f"{len(reinstated)} reinstated by the reviewer: "
                         + ", ".join(sorted(reinstated)))
    if len(flagged):
        detail = ", ".join(f"{int(n)} {reason.replace('_', ' ')}"
                           for reason, n in flagged["reason"].value_counts().items())
        parts.append(
            f"{len(flagged)} analysed but flagged ({detail}): "
            + ", ".join(sorted(str(c) for c in flagged["channel"]))
            + " — look at these on the trace before reading their rank, "
              "because this cannot tell a noisy amplifier from a contact "
              "full of real ripples")
    if not names and not len(flagged):
        parts.append(f"All {len(quality)} contacts passed the quality checks")
    if segments is not None and not segments.empty:
        dropped = int((~segments["good"]).sum())
        if dropped:
            share = dropped / len(segments)
            parts.append(f"{share:.1%} of contact-seconds rejected; each "
                         f"contact's rate is over the time that survived")
    return ". ".join(parts) + "."


# -- the measurements ------------------------------------------------------

def _clipped_fraction(row: np.ndarray) -> float:
    """Share of samples sitting at this channel's own extreme value.

    An amplifier at its rail repeats one number exactly, so the test is for
    samples within a hair of the extreme rather than above a fixed voltage --
    the rail is wherever this recording's hardware put it.
    """
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return 1.0
    extreme = float(np.max(np.abs(finite)))
    if extreme <= 0.0:
        return 1.0
    close = np.abs(np.abs(finite) - extreme) <= 1e-6 * extreme
    return float(np.count_nonzero(close) / finite.size)


def _line_fraction(data: np.ndarray, sfreq: float, line_freq: float) -> np.ndarray:
    """Share of each channel's power at the mains frequency and its harmonics.

    Harmonics are the point rather than a refinement: 50 Hz puts energy at 200
    and 250 Hz and 60 Hz puts it at 180 and 240, all of which are inside the
    ripple band this software counts events in.
    """
    if not np.isfinite(line_freq) or line_freq <= 0:
        return np.zeros(data.shape[0])
    freqs, power = _spectrum(data, sfreq)
    total = power.sum(axis=1)
    total = np.where(total <= 0, np.finfo(float).eps, total)
    mask = np.zeros_like(freqs, dtype=bool)
    harmonic = line_freq
    while harmonic < sfreq / 2:
        mask |= np.abs(freqs - harmonic) <= 2.0
        harmonic += line_freq
    return (power[:, mask].sum(axis=1) / total).astype(float)


def _band_fraction(data: np.ndarray, sfreq: float,
                   band: tuple[float, float]) -> np.ndarray:
    """Share of each channel's power inside `band`. Zero above Nyquist."""
    low, high = band
    if sfreq / 2 <= low:
        return np.zeros(data.shape[0])
    freqs, power = _spectrum(data, sfreq)
    total = power.sum(axis=1)
    total = np.where(total <= 0, np.finfo(float).eps, total)
    mask = (freqs >= low) & (freqs <= min(high, sfreq / 2))
    return (power[:, mask].sum(axis=1) / total).astype(float)


def _burstiness(data: np.ndarray, sfreq: float,
                band: tuple[float, float]) -> np.ndarray:
    """99th percentile of each channel's band envelope over its 10th.

    The question this answers is *how* a contact's band power is delivered.
    Rare large excursions over a quiet floor are events; a steady carpet is
    not. It is a ratio within the channel, so a 2 uV contact and a 200 uV one
    are directly comparable and no montage-wide calibration is needed --
    which matters, because the band-power ratio it accompanies is unreliable
    for exactly the opposite reason.

    Its floor is algebra rather than a measurement: band-passed Gaussian noise
    has a Rayleigh envelope, for which this ratio is
    :data:`onset_hfo.config.RAYLEIGH_BURSTINESS` whatever the amplitude.

    What it does **not** do is tell real activity from a repeating artifact.
    An electrode popping once a second is bursty too, and scores like a
    hippocampus full of ripples. It separates *events* from *carpet*, which is
    one of the two questions a reviewer has; the other one still needs the
    trace.
    """
    from scipy.signal import hilbert

    from onset_hfo.detectors.base import bandpass

    # The band's top is held below the Nyquist frequency: a recording sampled
    # too slowly for the whole band (500 Hz against 80-250 Hz ripples) still
    # gets a burstiness, of the part of the band it carries. It is compared
    # within the montage, so a narrower band shifts every channel alike.
    top = min(band[1], 0.9 * sfreq / 2)
    if top <= band[0]:
        return np.full(data.shape[0], float(RAYLEIGH_BURSTINESS))
    envelope = np.abs(hilbert(bandpass(data, sfreq, (band[0], top)), axis=-1))
    low, high = np.percentile(envelope, [10, 99], axis=-1)
    return high / np.maximum(low, np.finfo(float).eps)


def _spectrum(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectrum, one row per channel.

    Welch rather than a single periodogram: these windows are tens of seconds
    and a one-shot FFT of that is almost all variance, which would make the
    across-channel comparison below noise.
    """
    from scipy.signal import welch

    nperseg = int(min(data.shape[1], max(256, round(sfreq))))
    freqs, power = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    return np.asarray(freqs), np.atleast_2d(np.asarray(power))


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Distance from the median in robust SDs, across channels.

    Robust rather than mean/SD for the usual reason, with an extra one here: a
    montage with three broken contacts would have its mean and SD set by the
    broken ones, and they would then look normal.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:               # nothing to be an outlier against
        return np.zeros_like(values)
    center, scale = robust_scale(values, axis=0)
    return (values - center) / scale


def _ratio_to_median(values: np.ndarray) -> np.ndarray:
    """Each value over the montage's median. The scale-free half of the test."""
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 0:
        return np.ones_like(values)
    return values / median


def _bad_segment_fraction(segments: pd.DataFrame | None,
                          names: list[str]) -> dict[str, float]:
    if segments is None or segments.empty:
        return {}
    share = segments.groupby("channel")["good"].apply(lambda s: 1.0 - s.mean())
    return {str(k): float(v) for k, v in share.items()}


def reject_unusable_events(events, segments: pd.DataFrame | None,
                           quality: pd.DataFrame | None = None,
                           keep: tuple[str, ...] = ()) -> int:
    """Mark events that fall on rejected time or a rejected channel. In place.

    Returns how many were newly rejected. Nothing is deleted, and an event
    already rejected for another reason keeps its original one: the first
    explanation is the informative one, and overwriting it would hide that the
    validator had already caught it.

    This has to happen before rates are computed and after detection, because
    an event inside a rejected second is counted in a numerator whose
    denominator no longer includes that second. Left in, it would raise the
    rate of exactly the channel the rejection was meant to make honest.
    """
    if not events:
        return 0
    bad_channels = set()
    if quality is not None and not quality.empty:
        overruled = set(keep)
        bad_channels = {str(row.channel) for row in quality.itertuples()
                        if not row.good and str(row.channel) not in overruled}

    bad_spans: dict[str, list[tuple[float, float]]] = {}
    if segments is not None and not segments.empty:
        for row in segments[~segments["good"]].itertuples():
            bad_spans.setdefault(str(row.channel), []).append(
                (float(row.t_start), float(row.t_stop)))

    rejected = 0
    for event in events:
        if not event.accepted:
            continue
        channel = str(event.channel)
        if channel in bad_channels:
            event.accepted = False
            event.reject_reason = "on a channel set aside by the quality checks"
            rejected += 1
            continue
        # The event's own times are in original-recording seconds, which is
        # also what the segment table carries, so no offset arithmetic here.
        for start, stop in bad_spans.get(channel, ()):
            if event.start < stop and event.stop > start:
                event.accepted = False
                event.reject_reason = (
                    f"inside a rejected segment ({start:g}-{stop:g} s)")
                rejected += 1
                break
    return rejected
