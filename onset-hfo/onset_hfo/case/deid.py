"""Is anything in this case that could name the patient? Qt-free.

Conversion leaves the header fields that identify a patient behind
(`onset_hfo.case.bridges`), but text written by people travels with the
data: a mark typed as "J. Smith pushed button", a source file called
``SMITH_JOHN_2024-03-02.edf``, a note with a record number. This check reads
every field of a case that carries outside text --

* the case's note, each recording's source name and note, and each step's
  note (`case.json`),
  and the source named in each conversion report;
* every mark's kind, value and channels (``*_events.tsv``);
* every channel's name and description (``*_channels.tsv``);
* the recording sidecars' text fields (``*_ieeg.json``);
* contact names (``*_electrodes.tsv``);
* the log's details (it copies some of the above);

-- and reports what looks identifying: a date (a birth date or a day of
admission in a name), a run of six or more digits (a record number), an
e-mail address, a phone number, and any of the names the reader types in
when running the check. Those names are matched and never stored.

It is a check, not a guarantee: a name nobody typed, in a form no pattern
knows, passes. `redact` replaces what was found with "[removed]" in place
and logs that it did, without the text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["Finding", "check_case", "redact", "PATTERNS", "REMOVED"]

REMOVED = "[removed]"
PATTERNS = {
    "a date": re.compile(
        r"(?<!\d)(?:(?:19|20)\d{2}[-_/.](?:0?[1-9]|1[0-2])[-_/.](?:0?[1-9]|[12]\d|3[01])"
        r"|(?:0?[1-9]|[12]\d|3[01])[-_/.](?:0?[1-9]|1[0-2])[-_/.](?:19|20)?\d{2})(?!\d)"),
    "a long number (a record number?)": re.compile(
        r"(?<![A-Za-z0-9])(?<!\d\.)\d{6,}(?![A-Za-z0-9])(?!\.\d)"),
    "an e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "a phone number": re.compile(
        r"(?<![\w+])(?:\+?\d{1,3}[\s.-]*)?\(?\d{3}\)?[\s.-]+\d{3}[\s.-]+\d{4}(?!\d)"),
}
#: Log entries whose detail copies outside text (a file's name, a note, a
#: contact's name). The rest are the software's own words and numbers --
#: checksums, run folders named by date and time -- and are not scanned.
_LOG_ACTIONS_WITH_OUTSIDE_TEXT = {"created the case", "added a recording",
                                  "imported contact positions", "changed channels",
                                  "set the soz contacts", "planned an electrode on the template"}


@dataclass(frozen=True)
class Finding:
    file: str           # relative to the case root
    where: str          # e.g. "run 01 source name", "mark at 120.0 s value"
    what: str           # which pattern, or "a name you gave"
    excerpt: str        # the matched text with all but its edges masked
    key: tuple          # how `redact` finds the field again

    def describe(self) -> str:
        return f"{self.file} — {self.where}: {self.what} ({self.excerpt})"


def _mask(text: str) -> str:
    text = str(text)
    if len(text) <= 2:
        return "*" * len(text)
    return text[0] + "*" * (len(text) - 2) + text[-1]


def _scan(text, names) -> list[tuple[str, str]]:
    text = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    if not text:
        return []
    hits = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            hits.append((label, match.group(0)))
    lowered = text.lower()
    for name in names:
        name = str(name).strip()
        if len(name) >= 2 and re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])",
                                         lowered):
            hits.append(("a name you gave", name))
    return hits


def _rel(case, path: Path) -> str:
    try:
        return str(Path(path).relative_to(case.root))
    except ValueError:
        return str(path)


def _fields(case):
    """Every (file, where, key, text) of outside text in the case."""
    yield "case.json", "the case's note", ("case", "note"), case.note
    for rec in case.recordings:
        yield "case.json", f"run {rec.run} source name", ("recording", rec.run, "source_name"), \
            rec.source_name
        yield "case.json", f"run {rec.run} note", ("recording", rec.run, "note"), rec.note
        where = case.where(rec)
        events = where.path("events.tsv")
        if events.exists():
            frame = pd.read_csv(events, sep="\t", dtype=str, keep_default_na=False)
            for i, row in frame.iterrows():
                for column in ("trial_type", "value", "channels"):
                    if column in frame.columns:
                        yield (_rel(case, events), f"mark at {row.get('onset', '?')} s {column}",
                               ("events", rec.run, i, column), row[column])
        channels = where.path("channels.tsv")
        if channels.exists():
            frame = pd.read_csv(channels, sep="\t", dtype=str, keep_default_na=False)
            for i, row in frame.iterrows():
                for column in ("name", "status_description", "description"):
                    if column in frame.columns:
                        yield (_rel(case, channels), f"channel {row.get('name', '?')} {column}",
                               ("channels", rec.run, i, column), row[column])
        sidecar = where.path("ieeg.json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            for key, value in data.items():
                if isinstance(value, str):
                    yield _rel(case, sidecar), f"sidecar {key}", ("sidecar", rec.run, key), value
    conversion = Path(case.derivatives) / "conversion"
    for path in sorted(conversion.glob("*_conversion.json")) if conversion.exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        yield _rel(case, path), "conversion report source", ("conversion_json", path.name,
                                                               "source"), data.get("source", "")
    for path in sorted(conversion.glob("*_conversion.md")) if conversion.exists() else []:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.startswith("- **Source:**"):
                yield _rel(case, path), "conversion report source line", \
                    ("conversion_md", path.name, lineno), line.split("SHA-256")[0]
    from onset_hfo.case.electrodes import electrodes_path

    path = electrodes_path(case)
    if path.exists():
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for i, row in frame.iterrows():
            yield _rel(case, path), f"contact {row['name']}", ("electrodes", i, "name"), row["name"]
    for step, done in case.steps.items():
        yield "case.json", f"step {step} note", ("step", step), done.get("note", "")
    for i, entry in enumerate(case.log):
        action = str(entry.get("action", ""))
        if action in _LOG_ACTIONS_WITH_OUTSIDE_TEXT or action.startswith(
                ("completed step", "reopened step")):
            yield "case.json", f"log entry {i + 1} ({entry.get('action', '')})", ("log", i), \
                entry.get("detail", "")


def check_case(case, names=()) -> list[Finding]:
    """What in the case looks identifying. `names` are matched, not kept."""
    found = []
    for file, where, key, text in _fields(case):
        for what, matched in _scan(text, names):
            found.append(Finding(file, where, what, _mask(matched), key))
    return found


def _clean(text: str, names) -> str:
    text = str(text)
    for pattern in PATTERNS.values():
        text = pattern.sub(REMOVED, text)
    for name in names:
        name = str(name).strip()
        if len(name) >= 2:
            text = re.sub(rf"(?i)(?<![a-z]){re.escape(name)}(?![a-z])", REMOVED, text)
    return text


def redact(case, findings: list[Finding], names=(), by: str = "") -> int:
    """Replace what `findings` point at with "[removed]", in place. Log
    entries are rewritten and marked redacted (the log's chain keeps their
    original hashes, so the change shows; `Case.verify_log`). Returns how many
    fields changed."""
    keys = {f.key for f in findings}
    changed = 0
    for rec in case.recordings:
        for attr in ("source_name", "note"):
            if ("recording", rec.run, attr) in keys:
                setattr(rec, attr, _clean(getattr(rec, attr), names))
                changed += 1
    for step, done in case.steps.items():
        if ("step", step) in keys:
            done["note"] = _clean(done.get("note", ""), names)
            changed += 1
    if ("case", "note") in keys:
        case.note = _clean(case.note, names)
        changed += 1
    for rec in case.recordings:
        where = case.where(rec)
        for kind, suffix in (("events", "events.tsv"), ("channels", "channels.tsv")):
            path = where.path(suffix)
            wanted = [k for k in keys if k[0] == kind and k[1] == rec.run]
            if not wanted or not path.exists():
                continue
            frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            for _kind, _run, i, column in wanted:
                frame.at[i, column] = _clean(frame.at[i, column], names)
                changed += 1
            frame.to_csv(path, sep="\t", index=False)
        wanted = [k for k in keys if k[0] == "sidecar" and k[1] == rec.run]
        if wanted:
            path = where.path("ieeg.json")
            data = json.loads(path.read_text(encoding="utf-8"))
            for _kind, _run, key in wanted:
                data[key] = _clean(data[key], names)
                changed += 1
            path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    conversion = Path(case.derivatives) / "conversion"
    for key in keys:
        if key[0] == "conversion_json":
            path = conversion / key[1]
            data = json.loads(path.read_text(encoding="utf-8"))
            data[key[2]] = _clean(data.get(key[2], ""), names)
            path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            changed += 1
        elif key[0] == "conversion_md":
            path = conversion / key[1]
            lines = path.read_text(encoding="utf-8").splitlines()
            head, sep, tail = lines[key[2]].partition("SHA-256")
            lines[key[2]] = _clean(head, names) + sep + tail
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            changed += 1
    for key in keys:
        if key[0] == "log":
            entry = case.log[key[1]]
            entry["detail"] = _clean(entry.get("detail", ""), names)
            entry["redacted"] = True
            changed += 1
    case.save()
    if changed:
        case.record("redacted identifying text",
                    f"{changed} field(s) in {len({f.file for f in findings})} file(s); the text "
                    "is not kept", by)
    return changed
