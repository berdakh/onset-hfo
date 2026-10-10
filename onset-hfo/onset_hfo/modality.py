"""What kind of recording this is: intracranial EEG, scalp EEG or MEG.

The detectors threshold each channel against its own robust spread, so the
same code runs on any of the three; what differs is which channels are the
signal, what unit they are read in, and which montage makes sense:

=============  ======================  ===========  =====================================
modality       channels analysed       unit         "bipolar" means
=============  ======================  ===========  =====================================
``ieeg``       SEEG, ECoG (and EEG)    µV           neighbouring contacts on one lead
``eeg``        scalp EEG               µV           the longitudinal "double banana"
``meg_grad``   planar gradiometers     fT/cm        nothing: MEG is not re-referenced
``meg_mag``    magnetometers           fT           nothing
=============  ======================  ===========  =====================================

The intracranial selection is the one this project has always used (EEG-typed
channels ride along, as before), so nothing measured on iEEG changes. MEG is
analysed one sensor type at a time, because magnetometers and gradiometers
are in different units and differ a hundredfold in scale; gradiometers come
first when both are there, being the more local of the two.

Everything this project has validated, it validated on intracranial
recordings. The caveat each modality carries says what that means for it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Modality", "MODALITIES", "detect", "resolve", "DOUBLE_BANANA",
           "double_banana_pairs", "TRACE_TYPE", "UNIT_BY_TYPE", "SCALE_BY_TYPE",
           "trace_scale", "trace_unit"]


@dataclass(frozen=True)
class Modality:
    key: str
    label: str                      #: for people: "scalp EEG", "MEG (gradiometers)"
    types: tuple[str, ...]          #: MNE channel types analysed
    unit: str                       #: the unit the analysis reads them in
    scale: float                    #: SI value x scale = value in `unit`
    caveat: str


MODALITIES: dict[str, Modality] = {
    "ieeg": Modality(
        "ieeg", "intracranial EEG", ("seeg", "ecog", "eeg"), "µV", 1e6,
        "intracranial EEG: the recording this project's detectors and studies were "
        "validated on"),
    "eeg": Modality(
        "eeg", "scalp EEG", ("eeg",), "µV", 1e6,
        "scalp EEG: the detectors run as they do on intracranial recordings, but they "
        "were validated only there. Scalp HFOs are far smaller, muscle sits inside the "
        "ripple band, and no rate here has been checked against a scalp reference"),
    "meg_grad": Modality(
        "meg_grad", "MEG (planar gradiometers)", ("grad",), "fT/cm", 1e13,
        "MEG: the detectors run as they do on intracranial recordings, but they were "
        "validated only there. Amplitudes are in fT/cm; thresholds are in robust SDs "
        "of each sensor, so they carry over, but no rate here has been checked against "
        "a MEG reference"),
    "meg_mag": Modality(
        "meg_mag", "MEG (magnetometers)", ("mag",), "fT", 1e15,
        "MEG: the detectors run as they do on intracranial recordings, but they were "
        "validated only there. Amplitudes are in fT; magnetometers see distant sources "
        "and environmental noise more than gradiometers do"),
}

#: The MNE channel type a review window's trace carries for each modality, so
#: MNE's own browser labels it in the right unit (µV, fT/cm or fT).
TRACE_TYPE = {"ieeg": "seeg", "eeg": "eeg", "meg_grad": "grad", "meg_mag": "mag"}
#: The unit, and SI-to-unit factor, each analysed channel type is read in.
UNIT_BY_TYPE = {"seeg": "µV", "ecog": "µV", "eeg": "µV", "grad": "fT/cm", "mag": "fT"}
SCALE_BY_TYPE = {"seeg": 1e6, "ecog": 1e6, "eeg": 1e6, "grad": 1e13, "mag": 1e15}


def trace_scale(raw) -> float:
    """SI x this = the unit `raw`'s channels are read in (1e6 for µV)."""
    types = raw.get_channel_types() if raw is not None and len(raw.ch_names) else []
    return SCALE_BY_TYPE.get(types[0], 1e6) if types else 1e6


def trace_unit(raw) -> str:
    types = raw.get_channel_types() if raw is not None and len(raw.ch_names) else []
    return UNIT_BY_TYPE.get(types[0], "µV") if types else "µV"


#: The longitudinal bipolar montage of the 10-20 system ("double banana"),
#: left temporal, left parasagittal, right parasagittal, right temporal, then
#: the midline. Each chain is read front to back.
DOUBLE_BANANA: tuple[tuple[str, ...], ...] = (
    ("Fp1", "F7", "T7", "P7", "O1"),
    ("Fp1", "F3", "C3", "P3", "O1"),
    ("Fp2", "F4", "C4", "P4", "O2"),
    ("Fp2", "F8", "T8", "P8", "O2"),
    ("Fz", "Cz", "Pz"),
)
#: Older 10-20 names for the same places.
_OLD_NAMES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def detect(types) -> str:
    """The modality a recording's channel types say it is: intracranial when
    any SEEG or ECoG channel is there, else MEG when it has MEG sensors
    (gradiometers before magnetometers), else scalp EEG when it has EEG."""
    present = set(types)
    if present & {"seeg", "ecog", "dbs"}:
        return "ieeg"
    if "grad" in present:
        return "meg_grad"
    if "mag" in present:
        return "meg_mag"
    if "eeg" in present:
        return "eeg"
    return "ieeg"           # nothing analysable: the intracranial error says so


def resolve(requested: str | None, types) -> Modality:
    """The modality to analyse as: the one asked for, or `detect`'s."""
    key = (requested or "auto").lower()
    if key in ("auto", ""):
        key = detect(types)
    if key == "meg":
        key = "meg_grad" if "grad" in set(types) else "meg_mag"
    if key not in MODALITIES:
        raise ValueError(f"modality must be auto, {', '.join(MODALITIES)} or meg, "
                         f"not {requested!r}")
    return MODALITIES[key]


def double_banana_pairs(names) -> list[tuple[str, str]]:
    """The double banana's pairs among the channels present, by name in any
    case and with the older names (T3, T4, T5, T6) accepted. A pair needs
    both electrodes, so a missing one removes its two links rather than
    joining its neighbours across a gap."""
    lookup = {str(n).upper(): n for n in names}

    def find(label: str):
        for candidate in (label, _OLD_NAMES.get(label, "")):
            if candidate and candidate.upper() in lookup:
                return lookup[candidate.upper()]
        return None

    pairs: list[tuple[str, str]] = []
    for chain in DOUBLE_BANANA:
        for a, b in zip(chain, chain[1:], strict=False):
            first, second = find(a), find(b)
            if first is not None and second is not None:
                pairs.append((first, second))
    return pairs
