"""A case: one patient's recordings and everything done with them. Qt-free.

A case is a BIDS-iEEG folder (`onset_hfo.case.bids`) with one file of its own
at the top, ``case.json``:

* the pseudonym the case goes by -- never the patient's name;
* each recording converted into it: where it came from (file name, format,
  SHA-256), when and by whom, its sampling rate, channels and duration;
* the steps of the work, each done or not, by whom and when;
* an audit log: every conversion, every change to channels or marks, in
  order, append-only.

Results go under ``derivatives/onset/`` beside the recordings, as BIDS asks.
Nothing in a case names the patient: the pseudonym is the only link, and
the table from pseudonym to patient stays with the clinic.
"""

from __future__ import annotations

import getpass
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from onset_hfo.case import bids

__all__ = ["Case", "STEPS", "CASE_FILE", "Recording"]

CASE_FILE = "case.json"
SCHEMA = 1

#: The steps of a case, in order: (key, title, phase it arrives in). Later
#: phases' steps are listed so the bar shows where the work is going.
STEPS = (
    ("import", "Import", 1),
    ("channels", "Channels & electrodes", 1),
    ("annotate", "Annotate", 1),
    ("segments", "Segments", 2),
    ("preprocess", "Preprocess", 2),
    ("interictal", "Interictal", 2),
    ("ictal", "Ictal onset", 3),
    ("review", "Review", 2),
    ("map", "Map", 4),
    ("report", "Report", 5),
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:       # noqa: BLE001 - no user name is not a reason to fail
        return "unknown"


@dataclass
class Recording:
    """One run of the case, as `case.json` lists it."""

    run: str
    session: str = "implant01"
    task: str = "monitoring"
    source_name: str = ""
    source_format: str = ""
    source_sha256: str = ""
    sfreq: float = 0.0
    n_channels: int = 0
    duration_s: float = 0.0
    converted_at: str = ""
    converted_by: str = ""
    note: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, data: dict) -> Recording:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Case:
    root: Path
    case_id: str
    created: str = ""
    created_by: str = ""
    note: str = ""
    recordings: list = field(default_factory=list)
    steps: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    # -- making and opening ----------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, case_id: str, note: str = "") -> Case:
        """A new, empty case in `root` (made if missing, refused if it holds a
        case already). `case_id` is a pseudonym, letters and digits."""
        root = Path(root)
        if (root / CASE_FILE).exists():
            raise FileExistsError(f"{root} already holds a case")
        case = cls(root=root, case_id=bids.label(case_id), created=_now(),
                   created_by=_user(), note=note)
        root.mkdir(parents=True, exist_ok=True)
        bids.write_dataset_description(root, f"onset case {case.case_id}")
        (root / "derivatives" / "onset").mkdir(parents=True, exist_ok=True)
        case.record("created the case", note or "")
        return case

    @classmethod
    def open(cls, root: str | Path) -> Case:
        root = Path(root)
        path = root / CASE_FILE
        if not path.exists():
            raise FileNotFoundError(f"{root} is not a case: it has no {CASE_FILE}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(root=root, case_id=str(data["case_id"]), created=data.get("created", ""),
                   created_by=data.get("created_by", ""), note=data.get("note", ""),
                   recordings=[Recording.from_json(r) for r in data.get("recordings", [])],
                   steps=dict(data.get("steps", {})), log=list(data.get("log", [])))

    @staticmethod
    def is_case(folder: str | Path) -> bool:
        return (Path(folder) / CASE_FILE).exists()

    def save(self) -> Path:
        data = {"schema": SCHEMA, "case_id": self.case_id, "created": self.created,
                "created_by": self.created_by, "note": self.note,
                "recordings": [r.to_json() for r in self.recordings],
                "steps": self.steps, "log": self.log}
        path = self.root / CASE_FILE
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        partial.replace(path)
        return path

    # -- the audit log and the steps -------------------------------------------------------
    def record(self, action: str, detail: str = "", by: str = "") -> None:
        """Append to the log and save. The log is never rewritten."""
        self.log.append({"at": _now(), "by": by or _user(), "action": action,
                         "detail": detail})
        self.save()

    def mark_step(self, step: str, done: bool = True, note: str = "", by: str = "") -> None:
        if step not in {key for key, _, _ in STEPS}:
            raise ValueError(f"no step called {step!r}")
        self.steps[step] = {"done": bool(done), "by": by or _user(), "at": _now(),
                            "note": note}
        self.record(f"{'completed' if done else 'reopened'} step {step}", note, by)

    def step_done(self, step: str) -> bool:
        return bool(self.steps.get(step, {}).get("done"))

    # -- recordings ---------------------------------------------------------------------
    @property
    def subject(self) -> str:
        return self.case_id

    def next_run(self, session: str = "implant01") -> str:
        used = {int(r.run) for r in self.recordings if r.session == session
                and str(r.run).isdigit()}
        return f"{(max(used) + 1) if used else 1:02d}"

    def where(self, recording: Recording | str) -> bids.BidsRun:
        rec = self.recording(recording) if isinstance(recording, str) else recording
        return bids.BidsRun(self.root, self.subject, rec.session, rec.task, rec.run)

    def recording(self, run: str, session: str | None = None) -> Recording:
        for rec in self.recordings:
            if rec.run == run and (session is None or rec.session == session):
                return rec
        raise KeyError(f"no run {run} in case {self.case_id}")

    def add_recording(self, rec: Recording) -> None:
        if any(r.run == rec.run and r.session == rec.session for r in self.recordings):
            raise ValueError(f"run {rec.run} of {rec.session} is in the case already")
        self.recordings.append(rec)
        self.record("added a recording",
                    f"run {rec.run}: {rec.source_name} ({rec.source_format}), "
                    f"{rec.n_channels} channels, {rec.sfreq:g} Hz, {rec.duration_s:.0f} s, "
                    f"sha256 {rec.source_sha256[:12]}…")

    def has_source(self, sha256: str) -> Recording | None:
        """The recording already converted from a file with this checksum."""
        return next((r for r in self.recordings if r.source_sha256 == sha256), None)

    # -- channels and marks ------------------------------------------------------------------
    def channels(self, rec: Recording | str) -> pd.DataFrame:
        return bids.read_channels(self.where(rec))

    def save_channels(self, rec: Recording | str, table: pd.DataFrame, by: str = "") -> None:
        """Write a changed channels.tsv (types, good/bad, why) and log what changed."""
        where = self.where(rec)
        before = bids.read_channels(where).set_index("name")
        after = table.set_index("name")
        changes = []
        for name in after.index:
            if name not in before.index:
                continue
            for column in ("type", "status", "status_description"):
                old, new = str(before.at[name, column]), str(after.at[name, column])
                if old != new:
                    changes.append(f"{name} {column} {old} → {new}")
        table.to_csv(where.path("channels.tsv"), sep="\t", index=False, na_rep="n/a")
        run = rec if isinstance(rec, str) else rec.run
        self.record("changed channels", f"run {run}: " + ("; ".join(changes) or "no change"),
                    by)

    def marks(self, rec: Recording | str) -> pd.DataFrame:
        return bids.read_events(self.where(rec))

    def save_marks(self, rec: Recording | str, marks: pd.DataFrame, by: str = "") -> None:
        """Write the run's events.tsv and log how many of each kind were added
        or removed."""
        where = self.where(rec)
        before = bids.read_events(where)
        bids.write_events(where, marks)
        after = bids.read_events(where)

        def keys(frame):
            return {(round(float(r.onset), 3), str(r.trial_type), str(r.channels))
                    for r in frame.itertuples()}

        added, removed = keys(after) - keys(before), keys(before) - keys(after)
        run = rec if isinstance(rec, str) else rec.run
        self.record("changed marks", f"run {run}: {len(added)} added, {len(removed)} removed",
                    by)

    @property
    def derivatives(self) -> Path:
        return self.root / "derivatives" / "onset"
