"""From a clinical system's file to a case: the bridges. Qt-free.

Every recording system writes its own format; a case holds one
(`onset_hfo.case.bids`). A bridge reads one format and hands over the same
three things -- the signal, what each channel is, the system's own marks --
and `convert` writes them into the case, the same way for every format:

1. the source file is checksummed (SHA-256) and never changed, and a file
   already in the case is refused rather than converted twice;
2. channel types come from the reader's confirmation (a clinical export
   usually calls every channel scalp EEG), bad channels likewise;
3. the system's marks are mapped to the case's vocabulary, their text kept
   (`onset_hfo.case.annotations`);
4. nothing that names the patient is written: the file's patient name,
   identifier, birth date and recording date are left behind, and the report
   says which of them the file had; the time of day it started is kept,
   because night and day matter to the analysis;
5. a conversion report says what was done, in JSON and in words, under
   ``derivatives/onset/conversion/``.

The formats are MNE's readers (`onset_hfo.io.FORMATS`): EDF and EDF+, BDF,
BrainVision, Nihon Kohden, Nicolet, Persyst, Blackrock, Neuralynx, MEF3,
EEGLAB and more. Micromed and others come through the `neo` package when it
is installed. Natus/XLTEK is read from its EDF+ export, which every Natus
system writes.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo import io as onset_io
from onset_hfo.case import annotations as marks_module
from onset_hfo.case import bids
from onset_hfo.case.model import Case, Recording

__all__ = ["Bridge", "BRIDGES", "bridge_for", "inspect", "convert", "ConversionReport",
           "source_files", "sha256_of", "MIN_FAST_RIPPLE_HZ", "MIN_RIPPLE_HZ"]

#: Below these sampling rates a band cannot be analysed (its top must be below
#: half the rate, with room for the filter).
MIN_RIPPLE_HZ = 600.0
MIN_FAST_RIPPLE_HZ = 1200.0


@dataclass(frozen=True)
class Bridge:
    """One format: how to recognise it and how to read it (lazily)."""

    name: str
    suffixes: tuple
    read: object                      #: path, **kwargs -> mne Raw
    note: str = ""
    needs: tuple = ()


def _mne_bridge(fmt) -> Bridge:
    def read(path, **kwargs):
        return onset_io._read(onset_io.recording_root(path), fmt, preload=False, **kwargs)
    return Bridge(fmt.name, tuple(fmt.suffixes), read, fmt.note, tuple(fmt.needs))


def _neo_bridges() -> list[Bridge]:
    """Formats MNE has no reader for, through `neo`, when it is installed."""
    try:
        import neo  # noqa: F401
    except ImportError:
        return []

    def micromed(path, **_kwargs):
        import mne
        import neo

        reader = neo.io.MicromedIO(filename=str(path))
        block = reader.read_block(lazy=False)
        signal = block.segments[0].analogsignals[0]
        data = np.asarray(signal.rescale("V").magnitude, dtype=float).T
        names = [str(n) for n in signal.array_annotations.get(
            "channel_names", [f"Ch{i + 1}" for i in range(data.shape[0])])]
        info = mne.create_info(names, float(signal.sampling_rate.rescale("Hz").magnitude),
                               "eeg")
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        events = block.segments[0].events
        if events:
            onsets, texts = [], []
            for group in events:
                onsets += [float(t) for t in group.times.rescale("s").magnitude]
                texts += [str(x) for x in group.labels]
            raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), texts))
        return raw

    return [Bridge("Micromed (via neo)", (".trc",), micromed,
                   "read whole into memory by neo; long files need the memory")]


def _bridges() -> tuple[Bridge, ...]:
    return tuple([_mne_bridge(f) for f in onset_io.FORMATS] + _neo_bridges())


BRIDGES: tuple[Bridge, ...] = _bridges()

#: Formats asked about that no bridge reads, and what to do instead.
NOT_READ = {
    ".erd": "Natus/XLTEK: export the study as EDF+ from NeuroWorks and convert that",
    ".trc": "Micromed: install the neo package (pip install neo), or export as EDF+",
    ".e": "Nicolet/Natus .e: export as EDF+",
}


def bridge_for(path: str | Path) -> Bridge:
    """The bridge for a file; ValueError saying what to do when there is none."""
    path = Path(path)
    fmt = onset_io.detect_format(path)
    if fmt is not None:
        return next(b for b in BRIDGES if b.name == fmt.name)
    suffix = path.suffix.lower()
    for bridge in BRIDGES:
        if suffix in {s.lower() for s in bridge.suffixes}:
            return bridge
    hint = NOT_READ.get(suffix)
    raise ValueError(f"No bridge reads {path.name}." + (f" {hint}." if hint else
                     " Export it from the recording system as EDF+ and convert that."))


def source_files(path: str | Path) -> list[Path]:
    """Every file the recording is made of, for the checksum: a BrainVision
    header with its data and markers, a Persyst layout with its data, a
    folder's contents; else the file itself."""
    path = onset_io.recording_root(Path(path))
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    siblings = {".vhdr": (".vmrk", ".eeg"), ".lay": (".dat",), ".eeg": (".21e", ".pnt",
                                                                     ".log", ".EEG")}
    files = [path]
    for suffix in siblings.get(path.suffix.lower(), ()):
        for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
            if candidate.exists() and candidate not in files:
                files.append(candidate)
    return files


def sha256_of(files) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(Path(path).name.encode())
        with open(path, "rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    return digest.hexdigest()


def _first_time(raw) -> float:
    return float(raw.first_time) if raw.annotations.orig_time is not None else 0.0


def _identifying(raw) -> dict:
    """What in the file names the patient or dates the recording (never written)."""
    info = raw.info
    found = {}
    subject = info.get("subject_info") or {}
    for key, label in (("his_id", "patient identifier"), ("first_name", "first name"),
                       ("last_name", "last name"), ("middle_name", "middle name"),
                       ("birthday", "birth date")):
        if subject.get(key):
            found[key] = label
    if info.get("meas_date") is not None:
        found["meas_date"] = "recording date"
    for key, label in (("experimenter", "operator"), ("description", "description")):
        if info.get(key):
            found[key] = label
    return found


def inspect(path: str | Path, **reader_kwargs) -> dict:
    """What a file is, before converting it: format, rate, length, channels as
    the file declares them and as this software would type them, and the
    marks it carries with the kind each maps to. Reads headers only."""
    bridge = bridge_for(path)
    raw = bridge.read(path, **reader_kwargs)
    from onset_hfo.preprocess import _is_brain_channel

    channels = pd.DataFrame([
        {"name": n, "declared": t,
         "suggested": "seeg" if t in ("eeg", "ecog", "seeg") and _is_brain_channel(n, "seeg")
         else ("misc" if t == "eeg" else t)}
        for n, t in zip(raw.ch_names, raw.get_channel_types(), strict=True)])
    mapped = marks_module.map_marks(raw.annotations, _first_time(raw))
    sfreq = float(raw.info["sfreq"])
    start = raw.info.get("meas_date")
    return {"format": bridge.name, "sfreq": sfreq, "n_channels": len(raw.ch_names),
            "duration_s": float(raw.n_times) / sfreq, "channels": channels, "marks": mapped,
            "identifying": _identifying(raw),
            "start_time_of_day": start.strftime("%H:%M:%S") if start is not None else ""}


@dataclass
class ConversionReport:
    case_id: str
    run: str
    source: str
    source_format: str
    sha256: str
    sfreq: float
    n_channels: int
    duration_s: float
    types: dict = field(default_factory=dict)          #: BIDS type -> count
    retyped: list = field(default_factory=list)        #: (name, declared, given)
    bad: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)          #: kind -> count
    free_text: int = 0
    left_out: list = field(default_factory=list)       #: identifying fields not written
    start_time_of_day: str = ""
    line_freq: float = 50.0
    warnings: list = field(default_factory=list)
    seconds: float = 0.0
    files: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(self.__dict__)

    def markdown(self) -> str:
        hours = self.duration_s / 3600
        lines = [f"# Conversion — case {self.case_id}, run {self.run}", "",
                 f"- **Source:** {self.source} ({self.source_format}), SHA-256 `{self.sha256}`",
                 f"- **Signal:** {self.n_channels} channels at {self.sfreq:g} Hz, "
                 f"{self.duration_s:.1f} s ({hours:.2f} h)"
                 + (f", starting at {self.start_time_of_day}" if self.start_time_of_day
                    else ""),
                 f"- **Mains:** {self.line_freq:g} Hz (as given at conversion)",
                 "- **Channel types:** " + ", ".join(f"{n} {t}" for t, n in
                                                    sorted(self.types.items())),
                 f"- **Retyped from the file's declaration:** {len(self.retyped)}",
                 f"- **Marked bad:** {', '.join(self.bad) or 'none'}",
                 "- **Marks from the clinical system:** "
                 + (", ".join(f"{n} {k}" for k, n in sorted(self.marks.items())) or "none"),
                 f"- **Free-text marks** (text kept; read them for anything identifying): "
                 f"{self.free_text}",
                 "- **Left out (identifying):** " + (", ".join(self.left_out) or
                                                     "the file carried none"),
                 f"- **Took:** {self.seconds:.1f} s", ""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + [""]
        if self.retyped:
            lines += ["## Retyped channels", "", "| channel | in the file | converted as |",
                      "|---|---|---|"]
            lines += [f"| {n} | {a} | {b} |" for n, a, b in self.retyped]
        return "\n".join(lines) + "\n"


def convert(path: str | Path, case: Case, *, channel_types: dict | None = None,
            bad=(), line_freq: float = 50.0, session: str = "implant01",
            task: str = "monitoring", reference: str = "n/a", note: str = "",
            reader_kwargs: dict | None = None, progress=None,
            should_stop=None) -> ConversionReport:
    """Convert one clinical file into `case` as a new run. `channel_types`
    (name -> MNE type) is the reader's confirmation; `bad` the contacts known
    bad. Raises ValueError for a file the case holds already."""
    started = time.monotonic()
    path = Path(path)
    bridge = bridge_for(path)
    files = source_files(path)
    checksum = sha256_of(files)
    already = case.has_source(checksum)
    if already is not None:
        raise ValueError(f"{path.name} is in the case already, as run {already.run} "
                         f"(the same SHA-256).")
    raw = bridge.read(path, **(reader_kwargs or {}))
    declared = dict(zip(raw.ch_names, raw.get_channel_types(), strict=True))
    wanted = {n: t for n, t in (channel_types or {}).items() if n in raw.ch_names}
    if wanted:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_channel_types(wanted, verbose="ERROR")
    bad = [b for b in bad if b in raw.ch_names]
    raw.info["bads"] = list(bad)
    identifying = _identifying(raw)
    start = raw.info.get("meas_date")
    start_of_day = start.strftime("%H:%M:%S") if start is not None else ""
    mapped = marks_module.map_marks(raw.annotations, _first_time(raw))

    run = case.next_run(session)
    where = bids.BidsRun(case.root, case.subject, session, bids.label(task), run)
    extra = {"RecordingStartTimeOfDay": start_of_day} if start_of_day else {}
    bids.write_run(where, raw, events=mapped, line_freq=line_freq, reference=reference,
                   extra_json=extra, progress=progress, should_stop=should_stop)

    sfreq = float(raw.info["sfreq"])
    types: dict[str, int] = {}
    for kind in raw.get_channel_types():
        name = bids.BIDS_TYPES.get(kind, "MISC")
        types[name] = types.get(name, 0) + 1
    retyped = [(n, declared[n], t) for n, t in zip(raw.ch_names, raw.get_channel_types(),
                                                   strict=True) if declared[n] != t]
    warns = []
    if not (types.get("SEEG") or types.get("ECOG")):
        warns.append("No channel is typed SEEG or ECOG: nothing would be analysed as "
                     "intracranial. Set the types on the Channels step.")
    if sfreq < MIN_RIPPLE_HZ:
        warns.append(f"At {sfreq:g} Hz neither ripples nor fast ripples can be analysed.")
    elif sfreq < MIN_FAST_RIPPLE_HZ:
        warns.append(f"At {sfreq:g} Hz fast ripples (250–500 Hz) cannot be analysed; "
                     "ripples can.")
    free_text = int(sum(1 for v in mapped["trial_type"] if v == "note"))
    counts = mapped["trial_type"].value_counts().to_dict() if len(mapped) else {}
    report = ConversionReport(
        case_id=case.case_id, run=run, source=path.name, source_format=bridge.name,
        sha256=checksum, sfreq=sfreq, n_channels=len(raw.ch_names),
        duration_s=float(raw.n_times) / sfreq, types=types, retyped=retyped, bad=list(bad),
        marks={str(k): int(v) for k, v in counts.items()}, free_text=free_text,
        left_out=sorted(identifying.values()), start_time_of_day=start_of_day,
        line_freq=float(line_freq), warnings=warns,
        seconds=round(time.monotonic() - started, 1),
        files={"header": str(where.header.relative_to(case.root))})
    folder = case.derivatives / "conversion"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{where.stem}_conversion.json").write_text(
        json.dumps(report.to_json(), indent=1, default=str) + "\n")
    (folder / f"{where.stem}_conversion.md").write_text(report.markdown(), encoding="utf-8")
    case.add_recording(Recording(
        run=run, session=session, task=where.task, source_name=path.name,
        source_format=bridge.name, source_sha256=checksum, sfreq=sfreq,
        n_channels=len(raw.ch_names), duration_s=report.duration_s,
        converted_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        converted_by=case.log[-1]["by"] if case.log else "", note=note))
    if not case.step_done("import"):
        case.mark_step("import", True, f"first recording: run {run}")
    return report
