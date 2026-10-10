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

import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd

from onset_hfo.datasets import Recording

__all__ = ["Format", "FORMATS", "detect_format", "supported_suffixes",
           "file_filter", "channel_overview", "recording_info",
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
#: an intracranial recording plausibly arrives in, plus the general-purpose
#: interchange formats. MEG-only and fNIRS readers are left out because a file
#: this software can do nothing useful with should not be in the dialog.
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

    Two of these formats are directories, and a file dialog opens files. So a
    `.ncs` picked out of a Neuralynx folder, or anything picked out of a MEF3
    `.mefd` bundle, resolves to the folder that *is* the recording. Everything
    else is returned unchanged.
    """
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.suffix.lower() == ".mefd":
            return parent
    return path


def detect_format(path: str | Path) -> Format | None:
    """Which reader to use, from the filename. `None` when nothing matches.

    `.eeg` is the one genuine ambiguity: BrainVision uses it for the *data*
    file beside a `.vhdr` header, and Nihon Kohden uses it for the recording
    itself. A sibling header settles it, which is also what tells a reviewer
    who picked the wrong file of a BrainVision triplet.
    """
    path = recording_root(path)
    suffix = path.suffix.lower()

    if path.is_dir() and suffix != ".mefd":
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
    for name, declared in zip(raw.ch_names, raw.get_channel_types(), strict=True):
        suggested = "seeg" if declared in ("eeg", "ecog", "seeg") else declared
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
            wanted = {n: t for n, t in channel_types.items() if n in raw.ch_names}
            if wanted:
                raw.set_channel_types(wanted, verbose="ERROR")
        with raw.info._unlock():
            raw.info["line_freq"] = float(line_freq)

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
    ]
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


def _unsupported(path: Path) -> str:
    names = ", ".join(sorted({f.name for f in FORMATS}))
    return (f"{path.name} is not a format this software reads. MNE's readers "
            f"cover: {names}. If MNE can read it and it is missing here, add "
            f"it to onset_hfo.io.FORMATS -- there is no conversion step.")
