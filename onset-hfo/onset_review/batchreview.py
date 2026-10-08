"""Many recordings, analysed alike, into one table: Qt-free.

Everything else in the window analyses one recording at a time. This runs a
list of them -- cached archive windows, and files of the reader's own -- one
after the other with the same settings, and writes one table with a row per
recording: how many events each detector accepted, the busiest channel and
its interval, whether it stands out from the rest, the channels tied with
it, how many contacts the quality stage set aside, and, where the recording
carries expert markings, how this window's detections score against them
(`onset_review.thisrecording`).

Three rules make the table something to rely on:

* **Analysed alike.** Every recording gets the settings of the window that
  started the batch -- detectors, band, threshold, preprocessing, the
  quality stage -- and the batch folder records them (``settings.json``).
* **Files are confirmed once, for all of them.** A file's own channel types
  are usually wrong (most clinical exports call every channel scalp EEG), and
  a batch cannot stop to ask about each file; so the reader states one rule
  -- every channel SEEG, or ECoG, except names like ECG* or EMG* -- and each
  file's channels are typed by it, exactly as ``--all-channels-as`` does on
  the command line. The rule is in the settings and in every row.
* **A failure is a row, not a stop.** A recording that cannot be read or
  analysed gets its reason in the table, and the batch goes on.

Each recording's ranking is written beside the table (``findings/``), which
is what adding a row to *Your cohort* measures from (`onset_review.yourstudy`).
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["FileRule", "BatchItem", "request_for", "summarise", "run_batch", "load_batch",
           "COLUMNS", "DEFAULT_EXCEPTIONS"]

#: Channels a clinical export usually carries besides the brain's, and the
#: type each pattern gets.
DEFAULT_EXCEPTIONS = (("ECG*", "ecg"), ("EKG*", "ecg"), ("EMG*", "emg"), ("EOG*", "eog"),
                      ("DC*", "misc"), ("TRIG*", "stim"), ("STI*", "stim"), ("PHOTIC*", "misc"),
                      ("SPO2*", "misc"), ("PULSE*", "misc"), ("RESP*", "misc"))

COLUMNS = ("recording", "source", "subject", "window_s", "channels", "events", "leader",
           "leader_rate_per_min", "leader_ci", "stands_out", "n_tied", "tied",
           "quality_set_aside", "expert_f1", "expert_rho", "seconds", "error")


@dataclass(frozen=True)
class FileRule:
    """How a file's channels are typed, and which part of it is analysed."""

    all_as: str = "seeg"
    exceptions: tuple[tuple[str, str], ...] = DEFAULT_EXCEPTIONS
    t_start: float = 0.0
    t_stop: float = 60.0
    line_freq: float = 50.0

    def channel_types(self, names) -> tuple[tuple[str, str], ...]:
        typed = {}
        for name in names:
            kind = self.all_as
            for pattern, other in self.exceptions:
                if fnmatch.fnmatch(str(name).upper(), pattern.upper()):
                    kind = other
                    break
            typed[str(name)] = kind
        return tuple(sorted(typed.items()))

    def describe(self) -> str:
        names = ", ".join(p for p, _ in self.exceptions)
        return (f"every channel {self.all_as.upper()}, except names like {names}; "
                f"{self.t_start:g}–{self.t_stop:g} s; mains {self.line_freq:g} Hz")


@dataclass(frozen=True)
class BatchItem:
    """One recording to analyse: a cached archive window, or a file."""

    dataset: str = ""
    subject: str = ""
    run: str = "01"
    task: str | None = None
    t_start: float = 0.0
    t_stop: float = 60.0
    path: Path | None = None

    @property
    def is_file(self) -> bool:
        return self.path is not None

    def label(self) -> str:
        if self.is_file:
            return Path(self.path).name
        return f"{self.dataset} · {self.subject} · run-{self.run} · " \
               f"{self.t_start:g}–{self.t_stop:g} s"

    @classmethod
    def from_cached(cls, row: dict) -> BatchItem:
        task = row.get("task")
        task = None if task is None or (isinstance(task, float) and np.isnan(task)) else task
        return cls(dataset=str(row["dataset"]), subject=str(row["subject"]),
                   run=str(row.get("run") or "01"), task=task,
                   t_start=float(row["t_start"]), t_stop=float(row["t_stop"]))


def request_for(item: BatchItem, template, rule: FileRule):
    """The request for `item`, analysed as `template` (a ReviewRequest) was.
    A file's channels are typed by `rule`."""
    from onset_review.session import ReviewRequest

    analysis = dict(detectors=tuple(template.detectors), band=template.band,
                    threshold_sd=template.threshold_sd, with_spikes=template.with_spikes,
                    preprocess=template.preprocess, check_quality=template.check_quality)
    if item.is_file:
        from onset_hfo.io import channel_overview

        names = channel_overview(item.path)["name"]
        return ReviewRequest(dataset="", subject=Path(item.path).stem, path=Path(item.path),
                             t_start=float(rule.t_start), t_stop=float(rule.t_stop),
                             line_freq=float(rule.line_freq),
                             channel_types=rule.channel_types(names), **analysis)
    return ReviewRequest(dataset=item.dataset, subject=item.subject, run=item.run,
                         task=item.task, t_start=item.t_start, t_stop=item.t_stop, **analysis)


def summarise(session, seconds: float = float("nan")) -> dict:
    """One recording's row."""
    from onset_review import thisrecording

    request = session.request
    by_detector: dict[str, int] = {}
    for event in session.events:
        if event.accepted:
            by_detector[event.detector] = by_detector.get(event.detector, 0) + 1
    leader = dict(session.leader or {})
    set_aside = 0
    if session.quality is not None and "good" in session.quality:
        set_aside = int((~session.quality["good"].astype(bool)).sum())
    row = {
        "recording": request.label(), "source": "file" if request.path else request.dataset,
        "subject": request.subject, "window_s": f"{request.t_start:g}–{request.t_stop:g}",
        "channels": len(session.raw.ch_names),
        "events": ", ".join(f"{d} {n}" for d, n in sorted(by_detector.items())) or "none",
        "leader": leader.get("leader", ""),
        "leader_rate_per_min": leader.get("leader_rate_per_min", np.nan),
        "leader_ci": "–".join(f"{v:g}" for v in leader.get("leader_ci", [])),
        "stands_out": bool(leader.get("distinguishable", False)),
        "n_tied": len(session.candidates), "tied": ", ".join(session.candidates),
        "quality_set_aside": set_aside, "expert_f1": np.nan, "expert_rho": np.nan,
        "seconds": round(float(seconds), 1), "error": "",
    }
    scores = thisrecording.score_window(session)
    if scores is not None and len(scores):
        first = scores.iloc[0]
        row["expert_f1"] = round(float(first["f1"]), 3)
        row["expert_rho"] = round(float(first["rank_rho"]), 3) \
            if np.isfinite(first["rank_rho"]) else np.nan
    return row


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "recording"


@dataclass
class BatchResult:
    rows: pd.DataFrame
    folder: Path
    settings: dict = field(default_factory=dict)


def run_batch(items, template, rule: FileRule | None = None, folder: str | Path | None = None,
              progress=None, should_stop=None, load=None) -> BatchResult:
    """Analyse each of `items` as `template` was, one after another. Writes
    ``summary.csv``, ``summary.md``, ``settings.json`` and ``findings/`` into
    `folder` (a new folder under the state directory by default) as it goes,
    so a batch stopped halfway has its finished rows on disk."""
    if load is None:
        from onset_review.session import load_session as load
    rule = rule or FileRule()
    items = list(items)
    folder = Path(folder) if folder is not None else _default_folder()
    (folder / "findings").mkdir(parents=True, exist_ok=True)
    settings = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "detectors": list(template.detectors), "band": template.band,
        "threshold_sd": template.threshold_sd, "check_quality": template.check_quality,
        "preprocess": _plain(template.preprocess), "file_rule": rule.describe(),
        "recordings": [item.label() for item in items],
    }
    (folder / "settings.json").write_text(json.dumps(settings, indent=1, default=str) + "\n")
    rows: list[dict] = []
    for index, item in enumerate(items):
        if should_stop is not None and should_stop():
            for rest in items[index:]:
                rows.append({**_empty(rest), "error": "stopped before it ran"})
            break
        if progress is not None:
            progress(index, len(items), item.label())
        started = time.monotonic()
        try:
            request = request_for(item, template, rule)
            session = load(request)
            row = summarise(session, time.monotonic() - started)
            name = f"{index + 1:03d}-{_slug(row['recording'])}.csv"
            session.findings.to_csv(folder / "findings" / name, index=False)
            row["findings_file"] = name
            row["analysed_seconds"] = float(session.span[1] - session.span[0]) \
                if session.span else float(session.raw.times[-1])
            row["channel_names"] = "|".join(session.raw.ch_names)
            del session
        except Exception as error:      # noqa: BLE001 - a failure is a row
            row = {**_empty(item), "error": f"{type(error).__name__}: {error}",
                   "seconds": round(time.monotonic() - started, 1)}
        rows.append(row)
        _write(folder, rows, settings)
    _write(folder, rows, settings)
    return BatchResult(pd.DataFrame(rows), folder, settings)


def _empty(item: BatchItem) -> dict:
    row = {column: "" for column in COLUMNS}
    row.update(recording=item.label(), source="file" if item.is_file else item.dataset,
               subject=Path(item.path).stem if item.is_file else item.subject)
    return row


def _plain(value):
    import dataclasses

    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def _default_folder() -> Path:
    from onset_review.applog import log_dir

    base = log_dir().parent / "batches" if log_dir().name == "logs" else \
        log_dir() / "batches"
    return base / time.strftime("batch-%Y%m%d-%H%M%S")


def _write(folder: Path, rows: list[dict], settings: dict) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "summary.csv", index=False)
    done = frame[frame.get("error", "") == ""] if "error" in frame else frame
    lines = [f"# Batch — {len(frame)} recordings", "",
             f"{settings['when']} · detectors {', '.join(settings['detectors'])} · "
             f"{settings['band'].replace('_', ' ')} · threshold "
             f"{settings['threshold_sd'] if settings['threshold_sd'] is not None else 'each detector’s default'}"
             f" · quality stage {'on' if settings['check_quality'] else 'off'}", "",
             f"Files: {settings['file_rule']}.", "",
             f"{len(done)} analysed, {len(frame) - len(done)} not.", "",
             "| recording | events | busiest | rate /min | stands out | tied | set aside |"
             " error |", "|---|---|---|---:|---|---:|---:|---|"]
    for row in rows:
        rate = row.get("leader_rate_per_min", "")
        rate = f"{rate:g}" if isinstance(rate, (int, float)) and np.isfinite(rate) else ""
        lines.append(f"| {row.get('recording', '')} | {row.get('events', '')} | "
                     f"{row.get('leader', '')} | {rate} | "
                     f"{'yes' if row.get('stands_out') is True else 'no' if row.get('leader') else ''}"
                     f" | {row.get('n_tied', '')} | {row.get('quality_set_aside', '')} | "
                     f"{row.get('error', '')} |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_batch(folder: str | Path) -> BatchResult:
    folder = Path(folder)
    rows = pd.read_csv(folder / "summary.csv").fillna("")
    settings = json.loads((folder / "settings.json").read_text())
    return BatchResult(rows, folder, settings)


def findings_for(result: BatchResult, row: dict) -> pd.DataFrame | None:
    name = row.get("findings_file")
    if not name:
        return None
    path = Path(result.folder) / "findings" / str(name)
    return pd.read_csv(path) if path.exists() else None


def with_template(template, **changes):
    """`template` with some analysis settings changed (tests, and the batch
    window's "Use the defaults" choice)."""
    return replace(template, **changes)
