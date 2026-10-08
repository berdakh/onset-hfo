"""A whole analysis in one file, to reopen later or on another machine: Qt-free.

Until now an analysis was spread across this machine: the recording and its
settings in the window, the verdicts in the reads folder, the scripts in the
editor's tabs, the console's variables in memory, the cohort and re-runs
beside the assistant's settings. A project (``.onsetproj``, a zip) gathers
them:

* **project.json** -- the request (recording, window, band, detectors,
  threshold, preprocessing, quality settings), the page that was open, the
  editor's tabs with their text, and the console's command history;
* **read.json** -- the verdicts and the findings paragraph;
* **variables.pkl** -- what the console made, when there is any (a pickle,
  so opening one asks first);
* **study/** -- your cohort and your re-runs of the study;
* **recording/** -- the recording itself, for a file of your own, when asked
  for. An archive window needs no copy: the request names it, and it is
  fetched or found in the cache as any other. A file left out is named by
  path and by its SHA-256, so opening the project somewhere else can say
  whether the file found there is the same one.

Opening one restores what it can and says what it could not. Nothing in it
is a new number: the analysis is run again from the request, by the same
pipeline, so a project reopened is the same analysis rather than a copy of
its results. The console history comes back as a script in the editor, not
as commands run: what was run in another session is that session's record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SUFFIX", "request_to_dict", "request_from_dict", "save_project", "open_project",
           "ProjectState", "restore_read", "restore_study", "sha256"]

SUFFIX = ".onsetproj"
SCHEMA = 1


# -- the request, as JSON ----------------------------------------------------------------
def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def request_to_dict(request) -> dict:
    return _jsonable(request)


def request_from_dict(data: dict):
    """A ReviewRequest from `request_to_dict`'s output. Unknown keys (a newer
    version's) are left out rather than refused."""
    from onset_hfo.config import PreprocessConfig
    from onset_review.session import ReviewRequest

    known = {f.name: f for f in dataclasses.fields(ReviewRequest)}
    values = {}
    for name, value in data.items():
        if name not in known:
            continue
        if name == "preprocess" and isinstance(value, dict):
            fields = {f.name for f in dataclasses.fields(PreprocessConfig)}
            value = PreprocessConfig(**{k: _tuples(v) for k, v in value.items() if k in fields})
        elif name in ("path", "electrodes_path"):
            value = Path(value) if value else None
        else:
            value = _tuples(value)
        values[name] = value
    return ReviewRequest(**values)


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# -- saving ------------------------------------------------------------------------------------
def save_project(path: str | Path, *, request=None, read=None, page: str = "",
                 editor_tabs: list[dict] | None = None, console_log: list[str] | None = None,
                 variables: dict | None = None, include_recording: bool = False,
                 note: str = "") -> dict:
    """Write a project. `editor_tabs` are {title, path, text}; `variables`
    the console's own names. Returns what went in and what was left out."""
    from onset_review import __version__, yourstudy
    from onset_review import variables as workspace

    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    manifest: dict = {"schema": SCHEMA, "onset_review": __version__,
                      "saved": time.strftime("%Y-%m-%d %H:%M"), "note": note, "page": page,
                      "request": request_to_dict(request) if request is not None else None,
                      "editor": list(editor_tabs or []), "console_log": list(console_log or []),
                      "recording": None, "electrodes": None, "variables": [],
                      "variables_left_out": {}, "study": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(SUFFIX + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        if read is not None:
            bundle.writestr("read.json", json.dumps(read.to_json(), indent=1))
        if variables:
            with tempfile.TemporaryDirectory() as scratch:
                pkl, kept, skipped = workspace.save_workspace(variables,
                                                              Path(scratch) / "variables.pkl")
                if kept:
                    bundle.write(pkl, "variables.pkl")
                manifest["variables"] = kept
                manifest["variables_left_out"] = skipped
        folder = yourstudy.state_dir()
        for name in yourstudy.COHORT_FILES:
            if (folder / name).is_file():
                bundle.write(folder / name, f"study/{name}")
                manifest["study"].append(name)
        for key, attribute in (("recording", "path"), ("electrodes", "electrodes_path")):
            source = getattr(request, attribute, None) if request is not None else None
            if source is None:
                continue
            source = Path(source)
            entry = {"path": str(source), "name": source.name,
                     "sha256": sha256(source) if source.is_file() else None,
                     "included": False}
            # Coordinates are small and always travel; a recording only when asked.
            if source.is_file() and (include_recording or key == "electrodes"):
                bundle.write(source, f"{key}/{source.name}")
                entry["included"] = True
            manifest[key] = entry
        bundle.writestr("project.json", json.dumps(manifest, indent=1, default=str))
    partial.replace(path)
    manifest["path"] = str(path)
    return manifest


# -- opening -----------------------------------------------------------------------------------
@dataclass
class ProjectState:
    """What a project holds, unpacked. `request` is ready to analyse;
    `missing` says what could not be put back."""

    path: Path
    manifest: dict
    request: object = None
    read: object = None
    page: str = ""
    editor_tabs: list = field(default_factory=list)
    console_log: list = field(default_factory=list)
    variables_file: Path | None = None
    folder: Path | None = None
    missing: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def open_project(path: str | Path, folder: str | Path | None = None,
                 locate=None) -> ProjectState:
    """Unpack a project into `folder` (beside it by default) and rebuild its
    request. A recording that was not included is looked for at its saved
    path, then by `locate(entry)` -- the window asks the reader -- and a file
    found there is checked against the saved SHA-256."""
    from onset_review.adjudication import Adjudication

    path = Path(path)
    folder = Path(folder) if folder is not None else path.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as bundle:
        manifest = json.loads(bundle.read("project.json"))
        for member in bundle.namelist():
            target = (folder / member).resolve()
            if not str(target).startswith(str(folder.resolve())):
                continue        # a member that would land outside the folder
            if member.endswith("/"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
    state = ProjectState(path=path, manifest=manifest, folder=folder,
                         page=str(manifest.get("page") or ""),
                         editor_tabs=list(manifest.get("editor") or []),
                         console_log=list(manifest.get("console_log") or []))
    if (folder / "read.json").exists():
        state.read = Adjudication.from_json(json.loads((folder / "read.json").read_text()))
    if (folder / "variables.pkl").exists():
        state.variables_file = folder / "variables.pkl"
    data = manifest.get("request")
    if data:
        for key, attribute in (("recording", "path"), ("electrodes", "electrodes_path")):
            entry = manifest.get(key)
            if not entry:
                continue
            found = _find(entry, folder / key, locate if key == "recording" else None, state)
            data[attribute] = str(found) if found is not None else None
            if found is None and key == "recording":
                state.missing.append(f"the recording {entry.get('name')} "
                                     f"(saved at {entry.get('path')})")
        if not (manifest.get("recording") and data.get("path") is None):
            state.request = request_from_dict(data)
    if manifest.get("variables_left_out"):
        state.notes.append("Left out when saved: " + ", ".join(
            f"{k} ({why})" for k, why in manifest["variables_left_out"].items()))
    return state


def _find(entry: dict, included_dir: Path, locate, state: ProjectState) -> Path | None:
    inside = included_dir / str(entry.get("name"))
    if entry.get("included") and inside.is_file():
        return inside
    candidates = [Path(str(entry.get("path")))]
    for candidate in candidates:
        if candidate.is_file():
            return _checked(candidate, entry, state)
    if locate is not None:
        chosen = locate(entry)
        if chosen:
            return _checked(Path(chosen), entry, state)
    return None


def _checked(candidate: Path, entry: dict, state: ProjectState) -> Path:
    wanted = entry.get("sha256")
    if wanted and sha256(candidate) != wanted:
        state.notes.append(f"{candidate.name} is not the file the project was saved with "
                           "(its contents differ): the analysis will be of this file.")
    return candidate


def restore_read(request, read) -> Path | None:
    """Put the project's verdicts where the window looks for them. A read
    already on this machine for the same window is kept beside it as
    ``…-before-<time>.json``. Returns that backup's path, or None."""
    from onset_review import adjudication

    target = adjudication.path_for(request)
    backup = None
    if target.exists():
        backup = target.with_name(f"{target.stem}-before-{time.strftime('%Y%m%d-%H%M%S')}.json")
        shutil.copy2(target, backup)
    adjudication.save(request, read)
    return backup


def restore_study(state: ProjectState) -> Path | None:
    """Your cohort and re-runs from the project. What this machine had is
    moved to ``study/before-<time>/`` first. Returns that folder, or None."""
    from onset_review import yourstudy

    names = list(state.manifest.get("study") or [])
    if not names:
        return None
    folder = yourstudy.state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    backup = None
    existing = [folder / n for n in yourstudy.COHORT_FILES if (folder / n).is_file()]
    if existing:
        backup = folder / f"before-{time.strftime('%Y%m%d-%H%M%S')}"
        backup.mkdir()
        for file in existing:
            shutil.move(str(file), backup / file.name)
    for name in names:
        source = state.folder / "study" / name
        if source.is_file():
            shutil.copy2(source, folder / name)
    return backup
