"""Opening a recording that did not come from the archive.

The question this module answers is whether to define a file format. It does
not, and should not. MNE-Python already reads thirty, including the ones an
epilepsy monitoring unit actually produces -- Persyst, Nihon Kohden, Nicolet,
BrainVision, EDF, Curry, Blackrock, MEF3 -- and every one of them has had its
edge cases found by more people than will ever read this repository. Writing a
thirty-first format would mean maintaining thirty parsers badly and losing
every format MNE adds later.

So there is no proprietary structure here, and there was never one to write:
:class:`onset_hfo.datasets.Recording` has always held an `mne.io.BaseRaw` plus
the provenance a report needs. This module is the adapter, not a format.

**Two things make importing a foreign file different from loading the
archive, and both are the reviewer's problem rather than the parser's.**

The first is channel type. `ds003498` declares which channels are ECoG; a
clinical EDF export usually declares everything `eeg`, or `misc`, or nothing at
all. The pipeline keeps channels typed `ecog`, `seeg` or `eeg` and drops the
rest by name, so an imported scalp recording would be analysed as intracranial
and produce rates -- confidently, and wrongly. `channel_overview` exists so the
interface can put the file's own declaration in front of a reviewer and have
them confirm it before anything is detected.

The second is that an imported file carries no expert markings, no resected
zone and no participant record. Everything downstream already handles their
absence by saying so; nothing here invents them.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from onset_hfo.config import BANDS
from onset_hfo.datasets import Recording

__all__ = ["Format", "FORMATS", "detect_format", "supported_suffixes",
           "file_filter", "channel_overview", "recording_info", "recording_kind",
           "suggest_type",
           "open_recording", "recording_root", "INTRACRANIAL_TYPES"]

#: Channel types the pipeline will analyse. Everything else is dropped in
#: preprocessing, which is why getting these right on import matters more than
#: anything else this module does.
INTRACRANIAL_TYPES = ("seeg", "ecog")


@dataclass(frozen=True)
class Format:
    """One file format MNE can read, and what it takes to read it."""

    name: str
    reader: str                      #: the `mne.io` function, by name
    suffixes: tuple[str, ...]
    note: str = ""
    #: Arguments the reader requires beyond the filename. `nicolet` needs a
    #: channel type up front, for instance, because its header does not carry
    #: one -- which is the same problem this module exists to surface.
    needs: tuple[str, ...] = ()
    #: False for readers that cannot defer loading, so a window has to be cut
    #: after the whole file is in memory.
    lazy: bool = True
    #: True when the reader wants the *directory* a recording is spread across
    #: rather than one file in it. Neuralynx writes one `.ncs` per channel, so
    #: a reviewer picking a channel file in a file dialog is picking the
    #: recording, and this module hands the reader the folder.
    folder: bool = False

    @property
    def function(self):
        return getattr(mne.io, self.reader)


#: The formats offered, most likely first. Not every reader MNE has: the ones
#: an intracranial, scalp EEG or MEG recording plausibly arrives in, plus the
#: general-purpose interchange formats. MEGIN (Elekta) MEG is FIF, read by the
#: MNE entry; CTF and KIT/Yokogawa have their own. fNIRS readers are left out
#: because a file this software can do nothing useful with should not be in
#: the dialog.
FORMATS: tuple[Format, ...] = (
    Format("BrainVision", "read_raw_brainvision", (".vhdr",),
           "header file; the .eeg and .vmrk sit beside it"),
    Format("European Data Format", "read_raw_edf", (".edf", ".rec"),
           "the usual clinical export; rarely declares intracranial channels"),
    Format("BioSemi", "read_raw_bdf", (".bdf",)),
    Format("General Data Format", "read_raw_gdf", (".gdf",)),
    Format("Persyst", "read_raw_persyst", (".lay",),
           "layout file; the .dat sits beside it"),
    Format("Nihon Kohden", "read_raw_nihon", (".eeg", ".EEG"),
           "common in epilepsy monitoring units"),
    Format("Nicolet", "read_raw_nicolet", (".data",),
           "its header carries no channel type, so one must be chosen",
           needs=("ch_type",)),
    Format("Curry", "read_raw_curry", (".cdt", ".dap", ".dat"),),
    Format("Blackrock", "read_raw_nsx", (".ns2", ".ns3", ".ns4", ".ns5", ".ns6")),
    Format("Neuralynx", "read_raw_neuralynx", (".ncs",),
           "a directory of .ncs files; pick any one of them", folder=True),
    Format("MEF3", "read_raw_mef", (".mefd",),
           "long-term intracranial archive; a directory"),
    Format("EEGLAB", "read_raw_eeglab", (".set",)),
    Format("MNE / Elekta", "read_raw_fif", (".fif", ".fif.gz"),
           "what MNE itself writes"),
    Format("EGI", "read_raw_egi", (".mff", ".raw")),
    Format("Neuroscan", "read_raw_cnt", (".cnt",)),
    Format("Eximia", "read_raw_eximia", (".nxe",)),
    Format("CTF MEG", "read_raw_ctf", (".ds",),
           "a directory; pick any file inside it, such as the .meg4"),
    Format("KIT / Yokogawa MEG", "read_raw_kit", (".sqd", ".con")),
)


def supported_suffixes() -> list[str]:
    """Every extension the importer will attempt, lower-cased and unique."""
    seen: dict[str, None] = {}
    for fmt in FORMATS:
        for suffix in fmt.suffixes:
            seen.setdefault(suffix.lower(), None)
    return list(seen)


def file_filter() -> str:
    """A Qt file-dialog filter string, built from the table above.

    Generated rather than written out so the dialog cannot offer a format the
    importer does not handle, or omit one it does.
    """
    every = " ".join(f"*{s}" for s in supported_suffixes())
    parts = [f"Recordings ({every})"]
    parts += [f"{fmt.name} ({' '.join('*' + s for s in fmt.suffixes)})"
              for fmt in FORMATS]
    parts.append("All files (*)")
    return ";;".join(parts)


def recording_root(path: str | Path) -> Path:
    """What the reader should be given, for the path a reviewer picked.

    Three of these formats are directories, and a file dialog opens files. So
    a `.ncs` picked out of a Neuralynx folder, or anything picked out of a MEF3
    `.mefd` bundle or a CTF `.ds` folder, resolves to the folder that *is* the
    recording. Everything else is returned unchanged.
    """
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.suffix.lower() in _FOLDER_SUFFIXES:
            return parent
    return path


#: Recordings that are a directory named with their format's suffix.
_FOLDER_SUFFIXES = (".mefd", ".ds")


def detect_format(path: str | Path) -> Format | None:
    """Which reader to use, from the filename. `None` when nothing matches.

    `.eeg` is the one genuine ambiguity: BrainVision uses it for the *data*
    file beside a `.vhdr` header, and Nihon Kohden uses it for the recording
    itself. A sibling header settles it, which is also what tells a reviewer
    who picked the wrong file of a BrainVision triplet.
    """
    path = recording_root(path)
    suffix = path.suffix.lower()

    if path.is_dir() and suffix not in _FOLDER_SUFFIXES:
        # A recording that is a folder: MEF3 names the folder `.mefd` and is
        # matched below, Neuralynx names it nothing in particular and is
        # recognised by what is inside it.
        if any(path.glob("*.ncs")):
            return _by_reader("read_raw_neuralynx")
        return None

    if suffix == ".eeg":
        header = path.with_suffix(".vhdr")
        if header.exists():
            return _by_reader("read_raw_brainvision")
        return _by_reader("read_raw_nihon")

    if path.name.lower().endswith(".fif.gz"):
        return _by_reader("read_raw_fif")

    for fmt in FORMATS:
        if suffix in {s.lower() for s in fmt.suffixes}:
            return fmt
    return None


def _by_reader(reader: str) -> Format | None:
    return next((f for f in FORMATS if f.reader == reader), None)


def _read(path: Path, fmt: Format, preload: bool, **extra):
    """Call the reader, asking it not to load the data when it can.

    Deferring matters: a night of intracranial EEG is tens of gigabytes, and
    this software analyses a minute of it. Readers that cannot defer get
    `preload=True` and the window is cut afterwards, which is slower but not
    wrong.
    """
    kwargs = dict(extra)
    if fmt.lazy:
        kwargs["preload"] = preload
    if fmt.folder and path.is_file():
        path = path.parent          # the reviewer picked one channel of many
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fmt.function(path, verbose="ERROR", **kwargs)


def recording_kind(declared_types) -> str:
    """The kind of recording a file's own channel types suggest: ``"meg"``
    when it has MEG sensors, else ``"ieeg"``. Scalp EEG is never the guess,
    because clinical intracranial exports declare their contacts `eeg` too:
    the reviewer says so when it is scalp."""
    return "meg" if set(declared_types) & {"grad", "mag"} else "ieeg"


def suggest_type(declared: str, kind: str = "ieeg") -> str:
    """The type to offer for a channel the file declares `declared`, in a
    recording of `kind` (``"ieeg"``, ``"eeg"`` or ``"meg"``)."""
    if declared in ("eeg", "ecog", "seeg"):
        if kind == "eeg":
            return "eeg"
        if kind == "meg":
            return "eeg" if declared == "eeg" else declared
        return "seeg" if declared == "eeg" else declared
    return declared


def channel_overview(path: str | Path, **reader_kwargs) -> pd.DataFrame:
    """What the file says its channels are, without reading any signal.

    The interface puts this in front of a reviewer before anything is detected,
    because the file's own answer is often wrong in a way that cannot be
    detected later: a clinical EDF declares every channel `eeg`, and the
    pipeline would analyse a scalp montage as intracranial and report rates for
    it.

    Columns: `name`, `declared` (the file's type), `suggested` (what this
    software would use if nobody intervened) and `analysed` (whether the
    pipeline would keep it).
    """
    path = recording_root(path)
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(_unsupported(path))
    raw = _read(path, fmt, preload=False, **reader_kwargs)
    from onset_hfo.preprocess import _is_brain_channel

    rows = []
    kind = recording_kind(raw.get_channel_types())
    for name, declared in zip(raw.ch_names, raw.get_channel_types(), strict=True):
        suggested = suggest_type(declared, kind)
        rows.append({
            "name": name,
            "declared": declared,
            "suggested": suggested,
            "analysed": _is_brain_channel(name, declared),
        })
    return pd.DataFrame(rows, columns=["name", "declared", "suggested",
                                       "analysed"])


def recording_info(path: str | Path, **reader_kwargs) -> dict:
    """Sampling rate, duration and channel count, read from the header alone.

    The import dialog needs these before anything is analysed: a 60 s window
    offered on a 12 s file, or a fast-ripple band offered at 500 Hz, are both
    mistakes that are cheap to refuse here and expensive to discover after a
    load. No signal is read.
    """
    path = recording_root(path)
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(_unsupported(path))
    raw = _read(path, fmt, preload=False, **reader_kwargs)
    sfreq = float(raw.info["sfreq"])
    return {
        "sfreq": sfreq,
        "n_times": int(raw.n_times),
        "duration": float(raw.n_times) / sfreq,
        "n_channels": int(len(raw.ch_names)),
        "format": fmt.name,
        "reader": fmt.reader,
    }


#: SI per unit, for the units channels.tsv writes.
_SI_PER_UNIT = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "nv": 1e-9,
                "t": 1.0, "ft": 1e-15, "pt": 1e-12, "t/m": 1.0, "ft/cm": 1e-13,
                "ft/mm": 1e-12}


def _units_from_channels_tsv(raw, channels: pd.DataFrame | None) -> dict[str, str]:
    """Read a channel in the unit channels.tsv gives when the file states none.

    An EDF whose physical-dimension field is blank is read by MNE as volts,
    which turns a 50 µV scalp signal into 50 V; every amplitude after that is
    wrong by a million and the quality checks set every channel aside. The
    dataset's own channels.tsv says what the numbers are. Returns the channels
    rescaled, with the unit they were read in.
    """
    original = getattr(raw, "_orig_units", None) or {}
    if channels is None or not {"name", "units"} <= set(channels.columns) or not original:
        return {}
    done = {}
    for name, unit in zip(channels["name"], channels["units"], strict=False):
        name, unit = str(name), str(unit).strip()
        factor = _SI_PER_UNIT.get(unit.lower())
        if (name not in raw.ch_names or factor is None or factor == 1.0
                or str(original.get(name, "")).strip().lower() not in ("", "n/a", "na")):
            continue
        raw._data[raw.ch_names.index(name)] *= factor
        done[name] = unit
    return done


def _channels_tsv_beside(path: Path) -> pd.DataFrame | None:
    """The BIDS channels table that describes `path`, if one sits beside it:
    ``sub-01_task-x_channels.tsv`` for ``sub-01_task-x_eeg.edf``, or a plain
    ``channels.tsv`` (how a window fetched from OpenNeuro keeps it)."""
    stem = path.name.split(".")[0]
    head, _, _ = stem.rpartition("_")
    candidates = ([path.parent / f"{head}_channels.tsv"] if head else []) \
        + [path.parent / "channels.tsv"]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return pd.read_csv(candidate, sep="\t", dtype=str, keep_default_na=False)
            except Exception:       # noqa: BLE001 - an unreadable table is no table
                return None
    return None


def open_recording(path: str | Path, *, t_start: float = 0.0,
                   t_stop: float | None = None, subject: str | None = None,
                   line_freq: float = 50.0,
                   channel_types: dict[str, str] | None = None,
                   run: str = "01", task: str = "imported",
                   reader_kwargs: dict | None = None) -> Recording:
    """Read a window of a local file into the same `Recording` the archive gives.

    Everything downstream -- preprocessing, the detectors, the review window,
    the report -- is unchanged, because there is only one recording type in
    this project and it is MNE's with provenance attached.

    `channel_types` is the reviewer's correction, applied before anything else
    sees the data. Without it the file's own declaration stands, and on most
    clinical exports that means every channel is `eeg` and the pipeline
    analyses whatever is in the file.

    `line_freq` has no safe default. The archive carries it; a bare EDF does
    not, and notching the wrong mains frequency leaves the interference in
    place *and* carves a hole where there was none. 50 Hz is the commoner
    answer worldwide and is what is assumed when nobody says.
    """
    path = recording_root(path)
    if not path.exists():
        raise FileNotFoundError(f"no such recording: {path}")
    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(_unsupported(path))

    extra = dict(reader_kwargs or {})
    missing = [name for name in fmt.needs if name not in extra]
    if missing:
        raise ValueError(
            f"{fmt.name} files need {', '.join(missing)} supplied as well: its "
            f"header does not carry it. {fmt.note or ''}".strip())

    raw = _read(path, fmt, preload=False, **extra)
    duration = float(raw.n_times) / float(raw.info["sfreq"])
    stop = duration if t_stop is None else min(float(t_stop), duration)
    if t_start >= stop:
        raise ValueError(
            f"the requested window {t_start:g}–{stop:g} s is empty; this "
            f"recording is {duration:g} s long")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.crop(tmin=float(t_start), tmax=float(stop), include_tmax=False)
        raw.load_data(verbose="ERROR")
        if channel_types:
            # Only what the reviewer changed: re-typing a MEG sensor as itself
            # would be a no-op at best and a refused unit change at worst.
            current = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
            wanted = {n: t for n, t in channel_types.items()
                      if n in current and current[n] != t}
            if wanted:
                raw.set_channel_types(wanted, verbose="ERROR")
        with raw.info._unlock():
            raw.info["line_freq"] = float(line_freq)
        renamed = clean_contact_names(raw.ch_names)
        if renamed:
            raw.rename_channels(renamed, verbose="ERROR")

    rescaled = _units_from_channels_tsv(raw, _channels_tsv_beside(path))
    types = set(raw.get_channel_types())
    notes = [
        f"imported from {path.name} ({fmt.name}, via mne.io.{fmt.reader})",
        f"window {t_start:g}–{stop:g} s of a {duration:g} s recording; all "
        f"times reported by this pipeline are in original-recording seconds",
        f"mains frequency {line_freq:g} Hz, as given to this import and not "
        f"read from the file -- notching the wrong one leaves the interference "
        f"in place and carves a hole where there was none",
        "channel types "
        + ("confirmed by the reviewer" if channel_types
           else "taken from the file as-is, which most clinical exports get "
                "wrong; nothing here has checked them"),
        "no expert markings, no resected zone and no participant record come "
        "with an imported file, so anything in this software that compares "
        "against them is unavailable rather than empty",
        *([f"{len(renamed)} channel name(s) carried the exporter's prefix or "
           f"reference suffix and are read without it "
           f"({', '.join(f'{a} as {b}' for a, b in list(renamed.items())[:4])}"
           f"{', …' if len(renamed) > 4 else ''}): left as written, no contact on "
           f"them is recognised and no bipolar pair is formed"] if renamed else []),
        *([f"the file does not state the unit of {len(rescaled)} channels; the BIDS "
           f"channels.tsv beside it gives it ({', '.join(sorted(set(rescaled.values())))}), "
           "so they were read in that unit rather than as volts"] if rescaled else []),
    ]
    # Two things the file can say about itself that the person importing it
    # may not know, and that decide whether an HFO count means anything.
    seen = mains_in_signal(raw)
    if seen is not None and abs(seen - float(line_freq)) > 1.0:
        notes.append(
            f"WARNING: the signal carries mains pickup at {seen:g} Hz, and this "
            f"import was told {line_freq:g} Hz. The notch is on the wrong "
            f"frequency: the interference and its harmonics "
            f"({2 * seen:g}, {3 * seen:g}, {4 * seen:g} Hz) are still in the "
            f"signal, inside or beside the ripple band. Import again with mains "
            f"set to {seen:g} Hz")
    cut = acquisition_lowpass(raw)
    if cut is not None:
        lost = [f"{label} ({band[0]:g}–{band[1]:g} Hz)"
                for label, band in (("ripples", BANDS.ripple),
                                    ("fast ripples", BANDS.fast_ripple))
                if cut < band[1]]
        if lost:
            notes.append(
                f"WARNING: the file's header says it was low-pass filtered at "
                f"{cut:g} Hz when it was recorded or exported. Activity above "
                f"that is attenuated however fast the file is sampled, so few "
                f"or no events in {' or '.join(lost)} is what this file would "
                f"show whether or not the brain produced them")

    if not (types & set(INTRACRANIAL_TYPES)):
        from onset_hfo.modality import MODALITIES, detect

        kind = MODALITIES[detect(types)]
        notes.append(
            f"no channel in this window is typed seeg or ecog, so it is analysed as "
            f"{kind.label} ({kind.unit}): {kind.caveat}. If it is in fact intracranial, "
            "say so when importing, or every channel is analysed as the wrong kind"
            if kind.key != "ieeg" else
            "WARNING: no channel in this window is typed seeg, ecog, eeg or MEG, so "
            "there is nothing the pipeline can analyse")

    return Recording(
        raw=raw, source=f"local:{path.name}",
        subject=subject or path.stem, task=task, run=run,
        t_offset=float(t_start), line_freq=float(line_freq),
        dataset_id="", citation="", notes=notes,
    )


#: What a clinical exporter wraps a contact name in: "POL LA1", "EEG LA1-Ref",
#: "SEEG LA 1". None of it is part of the name the implantation plan uses.
_NAME_PREFIX = re.compile(r"^(?:POL|EEG|SEEG|ECOG|IEEG)[\s_:]+", re.IGNORECASE)
_NAME_SUFFIX = re.compile(r"[\s_-]+(?:REF|REFERENCE|AVG|AV|LE|MONO)\s*$", re.IGNORECASE)
_CONTACT_NAME = re.compile(r"^[A-Za-z]+'?\d{1,3}$")


def clean_contact_names(names) -> dict[str, str]:
    """`{name in the file: contact name}` for the names an exporter decorated.

    Only a name that *becomes* a contact name (letters, then a number) by
    losing a known prefix, a reference suffix and its spaces is changed, and
    only if the result collides with nothing: a rename that merged two
    channels would be worse than the decoration. Everything else is left
    exactly as the file has it.
    """
    wanted: dict[str, str] = {}
    for name in names:
        if _CONTACT_NAME.match(name):
            continue
        bare = _NAME_SUFFIX.sub("", _NAME_PREFIX.sub("", str(name).strip()))
        bare = re.sub(r"\s+", "", bare)
        if bare != name and _CONTACT_NAME.match(bare):
            wanted[name] = bare
    taken = [n for n in names if n not in wanted] + list(wanted.values())
    return {old: new for old, new in wanted.items() if taken.count(new) == 1}


def mains_in_signal(raw, picks=None) -> float | None:
    """50 or 60 if the signal plainly carries that mains frequency, else None.

    A bare EDF does not say where it was recorded, and the import assumes
    50 Hz when nobody does. This reads the answer off the data instead: the
    power in a narrow line at each candidate against the spectrum either side
    of it, on the median channel. None when neither stands out, or when both
    do -- a clean recording is not evidence for either.
    """
    from scipy.signal import welch

    sfreq = float(raw.info["sfreq"])
    if sfreq < 150.0:
        return None
    try:
        data = raw.get_data(picks=picks if picks is not None else "data")
    except Exception:                       # noqa: BLE001 -- no data channels
        return None
    if data.size == 0 or data.shape[1] < int(2 * sfreq):
        return None
    data = data[np.isfinite(data).all(axis=1)]
    if not len(data):
        return None
    freqs, power = welch(data, fs=sfreq, nperseg=int(min(data.shape[1], 4 * sfreq)),
                         axis=-1)
    power = np.median(power, axis=0)

    def prominence(line: float) -> float:
        peak = power[(freqs >= line - 0.5) & (freqs <= line + 0.5)]
        around = power[((freqs >= line - 5) & (freqs <= line - 2))
                       | ((freqs >= line + 2) & (freqs <= line + 5))]
        if not len(peak) or not len(around) or float(np.median(around)) <= 0:
            return 0.0
        return float(peak.max() / np.median(around))

    at = {50.0: prominence(50.0), 60.0: prominence(60.0)}
    best = max(at, key=at.get)
    other = min(at, key=at.get)
    if at[best] < MAINS_PROMINENCE or at[best] < 3.0 * at[other]:
        return None
    return best


#: How far a mains line must rise above the spectrum either side of it, as a
#: power ratio, before the signal is taken to say which mains it was recorded
#: on. Ten is a line nobody would miss on a spectrum; a notch at acquisition
#: or a well-shielded room leaves nothing, and then nothing is concluded.
MAINS_PROMINENCE = 10.0


def acquisition_lowpass(raw) -> float | None:
    """The low-pass the file's header declares, when it is below what the
    sampling rate alone would allow. None when the header does not say, which
    MNE reports as a low-pass at the Nyquist frequency."""
    try:
        cut = float(raw.info.get("lowpass") or 0.0)
    except (TypeError, ValueError):
        return None
    nyquist = float(raw.info["sfreq"]) / 2.0
    if not np.isfinite(cut) or cut <= 0 or cut >= 0.98 * nyquist:
        return None
    return cut


def _unsupported(path: Path) -> str:
    names = ", ".join(sorted({f.name for f in FORMATS}))
    return (f"{path.name} is not a format this software reads. MNE's readers "
            f"cover: {names}. If MNE can read it and it is missing here, add "
            f"it to onset_hfo.io.FORMATS -- there is no conversion step.")
