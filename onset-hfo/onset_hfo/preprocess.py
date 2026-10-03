"""Preprocessing: from a raw recording to the channels a detector sees.

Four steps, in this order, each of which is logged into ``Prepared.steps`` so
that a report can state exactly what was done to the signal:

1. **Channel selection** -- keep intracranial data channels (ECoG/SEEG/EEG),
   drop DC/trigger/ECG/misc channels and everything the dataset flagged ``bad``.
2. **High-pass** at 1 Hz to remove drift.
3. **Notch** at the mains frequency and its harmonics. Narrow notches
   (2 Hz wide) are used on purpose: harmonics at 180/240 Hz sit inside the
   ripple band, and a wide notch would carve a hole in the signal we are
   trying to measure.
4. **Bipolar montage** -- subtract neighbouring contacts on the same
   electrode. This is standard for HFO work: a common reference shares its
   noise with every channel and produces HFOs that appear everywhere at once.

Caveat worth knowing (and stated in the report): on a rectangular grid,
"neighbouring by number" (G8 - G9) is not always neighbouring in space,
because numbering wraps at the end of a row. For depth electrodes and strips
the numbering is spatially ordered, which is the case that matters most here.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from onset_hfo.config import PreprocessConfig
from onset_hfo.datasets import Recording

__all__ = ["Prepared", "prepare", "bipolar_pairs"]

_CONTACT_RE = re.compile(r"^([A-Za-z]+[A-Za-z']*?)(\d{1,3})$")
#: Channel name prefixes that are never intracranial recordings in this dataset.
_NON_BRAIN = ("DC", "EKG", "ECG", "EMG", "EOG", "TRIG", "STIM", "REF", "EVENT", "MARK")


@dataclass
class Prepared:
    """Preprocessed signals plus the story of how they were produced."""

    data: np.ndarray              #: (n_channels, n_times), microvolts
    ch_names: list[str]           #: bipolar pair names ("G1-G2") or contact names
    sfreq: float
    t_offset: float               #: seconds; add to a local time to get file time
    montage: str                  #: "bipolar" or "monopolar"
    #: Mains frequency actually notched, resolved from the config or the
    #: recording. Carried here so no later step has to guess it again.
    line_freq: float = 60.0
    pairs: list[tuple[str, str]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    recording: Recording | None = None

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.sfreq

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    def contacts_of(self, ch: str) -> list[str]:
        """The physical contact(s) a (possibly bipolar) channel is built from."""
        if self.montage == "bipolar" and ch in self.ch_names:
            a, b = self.pairs[self.ch_names.index(ch)]
            return [a, b]
        return [ch]

    def index(self, ch: str) -> int:
        return self.ch_names.index(ch)

    def segment(self, ch: str, t0: float, t1: float, absolute: bool = True) -> np.ndarray:
        """Signal for one channel between two times (default: file times)."""
        if absolute:
            t0, t1 = t0 - self.t_offset, t1 - self.t_offset
        i0 = max(0, int(round(t0 * self.sfreq)))
        i1 = min(self.data.shape[1], int(round(t1 * self.sfreq)))
        return self.data[self.index(ch), i0:i1]


def _is_brain_channel(name: str, ch_type: str) -> bool:
    if ch_type not in ("ecog", "seeg", "eeg"):
        return False
    upper = name.upper()
    return not any(upper.startswith(p) for p in _NON_BRAIN)


def bipolar_pairs(ch_names: list[str], exclude: set[str] | None = None) -> list[tuple[str, str]]:
    """Consecutive contacts on the same electrode, e.g. ``("AD1", "AD2")``.

    Both contacts must be present and neither may be excluded, so a bad
    contact removes the two pairs it would have formed rather than silently
    bridging a gap (AD1-AD3 would mix two distances and is never produced).
    """
    exclude = exclude or set()
    by_lead: dict[str, dict[int, str]] = {}
    for name in ch_names:
        m = _CONTACT_RE.match(name)
        if not m:
            continue
        by_lead.setdefault(m.group(1), {})[int(m.group(2))] = name
    pairs: list[tuple[str, str]] = []
    for lead in sorted(by_lead):
        numbers = sorted(by_lead[lead])
        for n in numbers:
            if n + 1 in by_lead[lead]:
                a, b = by_lead[lead][n], by_lead[lead][n + 1]
                if a not in exclude and b not in exclude:
                    pairs.append((a, b))
    return pairs


def _check(cfg: PreprocessConfig, sfreq: float) -> None:
    """Refuse settings that would produce numbers rather than a measurement.

    Every one of these runs perfectly well and returns a plausible-looking
    answer, which is exactly why they are refused here instead of being left to
    whoever reads the result. A low-pass under the high-pass passes nothing; a
    low-pass above the new Nyquist after downsampling is asking for a band the
    signal can no longer carry.
    """
    if cfg.highpass and cfg.lowpass and cfg.lowpass <= cfg.highpass:
        raise ValueError(
            f"low-pass {cfg.lowpass:g} Hz is at or below the high-pass "
            f"{cfg.highpass:g} Hz, which passes nothing")
    # `is not None` rather than truthiness: a resample of 0 Hz is nonsense, and
    # treating it as "no resampling" would accept it silently.
    if cfg.resample is not None and float(cfg.resample) <= 0:
        raise ValueError(
            f"resample rate must be positive, not {float(cfg.resample):g} Hz")
    rate = float(cfg.resample) if cfg.resample else float(sfreq)
    if cfg.lowpass and cfg.lowpass >= rate / 2.0:
        raise ValueError(
            f"low-pass {cfg.lowpass:g} Hz is at or above the Nyquist frequency "
            f"of {rate / 2.0:g} Hz"
            + (f" that resampling to {rate:g} Hz would leave" if cfg.resample
               else ""))
    if cfg.notch_width <= 0:
        raise ValueError(f"notch width must be positive, not {cfg.notch_width:g} Hz")


def prepare(rec: Recording, cfg: PreprocessConfig | None = None, verbose: bool = True) -> Prepared:
    """Run the four preprocessing steps and return the array a detector reads.

    The returned data is in **microvolts**, because every threshold and plot in
    this project is expressed in µV and silent unit changes are how analyses go
    wrong.
    """
    cfg = cfg or PreprocessConfig()
    _check(cfg, float(rec.raw.info["sfreq"]))
    raw = rec.raw.copy()
    steps: list[str] = []

    # 1. channel selection ------------------------------------------------
    types = raw.get_channel_types()
    keep = [n for n, t in zip(raw.ch_names, types, strict=False) if _is_brain_channel(n, t)]
    dropped_type = [n for n in raw.ch_names if n not in keep]
    if not keep:
        raise ValueError("No intracranial data channels found in this recording")
    raw.pick(keep)
    steps.append(f"kept {len(keep)} intracranial channels; dropped {len(dropped_type)} "
                 f"non-brain channels (DC/trigger/ECG/misc)")

    bads = [b for b in rec.bads if b in raw.ch_names]
    if cfg.drop_bads and bads:
        raw.drop_channels(bads)
        steps.append(f"dropped {len(bads)} channels flagged bad by the dataset: {', '.join(sorted(bads))}")

    # Channels the reviewer excluded themselves. Logged separately from the
    # dataset's own flags because they are a different kind of claim: one is
    # the archive's, the other is this reviewer's judgement on this window, and
    # a report that merged them would attribute the second to the first.
    marked = [c for c in dict.fromkeys(cfg.exclude) if c in raw.ch_names]
    if marked:
        if len(marked) >= len(raw.ch_names):
            raise ValueError(
                "excluding those channels would leave nothing to analyse "
                f"({len(marked)} of {len(raw.ch_names)} channels)")
        raw.drop_channels(marked)
        steps.append(f"dropped {len(marked)} channels the reviewer marked bad: "
                     f"{', '.join(sorted(marked))}")

    # 2/3. filtering ------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.highpass or cfg.lowpass:
            raw.filter(l_freq=cfg.highpass or None, h_freq=cfg.lowpass,
                       fir_design="firwin", phase="zero", verbose="ERROR")
            if cfg.highpass:
                steps.append(f"high-pass {cfg.highpass:g} Hz (zero-phase FIR)")
            if cfg.lowpass:
                steps.append(f"low-pass {cfg.lowpass:g} Hz (zero-phase FIR)")
        line_freq = cfg.line_freq if cfg.line_freq is not None else rec.line_freq
        if cfg.notch:
            source = "configured" if cfg.line_freq is not None else "from the dataset"
            nyq = raw.info["sfreq"] / 2.0
            freqs = ([f for f in np.arange(line_freq, nyq, line_freq) if f < 0.9 * nyq]
                     if cfg.notch_harmonics
                     else ([line_freq] if line_freq < 0.9 * nyq else []))
            if freqs:
                raw.notch_filter(freqs=freqs, notch_widths=cfg.notch_width,
                                 fir_design="firwin", phase="zero", verbose="ERROR")
                harmonics = " + harmonics" if cfg.notch_harmonics else " (fundamental only)"
                steps.append(f"notch {line_freq:g} Hz ({source}){harmonics} "
                             f"({', '.join(f'{f:g}' for f in freqs)} Hz, "
                             f"{cfg.notch_width:g} Hz wide)")

    if cfg.resample and abs(float(cfg.resample) - raw.info["sfreq"]) > 1e-9:
        target = float(cfg.resample)
        before = float(raw.info["sfreq"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.resample(target, npad="auto", verbose="ERROR")
        steps.append(f"resampled {before:g} Hz -> {target:g} Hz"
                     + (" (upsampled; adds no information)" if target > before else ""))

    data = raw.get_data(picks="all") * 1e6  # volts -> microvolts
    names = list(raw.ch_names)

    # 4. montage ----------------------------------------------------------
    pairs: list[tuple[str, str]] = []
    if cfg.bipolar:
        pairs = bipolar_pairs(names, exclude=set())
        if not pairs:
            steps.append("bipolar montage requested but no consecutive contact pairs were found; "
                         "kept the original (monopolar) channels")
        else:
            idx = {n: i for i, n in enumerate(names)}
            data = np.stack([data[idx[a]] - data[idx[b]] for a, b in pairs])
            names = [f"{a}-{b}" for a, b in pairs]
            steps.append(f"bipolar montage: {len(pairs)} pairs of neighbouring contacts")
    elif cfg.average_reference and len(names) > 1:
        data = data - data.mean(axis=0, keepdims=True)
        steps.append(
            f"common average reference across {len(names)} channels — note "
            f"that this shares every channel's noise with every other, which "
            f"is the effect the bipolar montage exists to avoid for HFO work")
    montage = ("bipolar" if pairs
               else "average" if cfg.average_reference and len(names) > 1
               else "monopolar")

    prepared = Prepared(data=np.ascontiguousarray(data, dtype=np.float64), ch_names=names,
                        sfreq=float(raw.info["sfreq"]), t_offset=rec.t_offset, montage=montage,
                        line_freq=float(line_freq),
                        pairs=pairs, steps=steps, recording=rec)
    if verbose:
        print(f"[onset-hfo] preprocessed: {prepared.n_channels} {montage} channels, "
              f"{prepared.duration:.1f} s @ {prepared.sfreq:g} Hz")
        for s in steps:
            print(f"[onset-hfo]   - {s}")
    return prepared
