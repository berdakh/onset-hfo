"""Any continuous recording on OpenNeuro, fetched the way scikit-learn fetches a dataset.

`onset_hfo.datasets` serves the archives this project studies, each described
by hand in `onset_hfo.config.DATASETS`. This module serves the rest of
OpenNeuro: give it a dataset id and it lists the recordings, describes the
dataset, and fetches a window of any recording into the same `Recording` the
rest of the package works with::

    from onset_hfo import openneuro

    openneuro.describe_openneuro("ds004100")          # name, licence, authors, README
    runs = openneuro.list_openneuro("ds004100")       # one row per recording
    bunch = openneuro.fetch_openneuro("ds004100", subject="sub-HUP060",
                                      t_start=0, t_stop=60)
    bunch.data, bunch.times, bunch.ch_names           # arrays, as sklearn gives them
    bunch.recording                                   # the package's Recording

Like `sklearn.datasets`:

* everything downloaded is kept under a data home (`get_data_home`, the
  ``ONSET_HFO_DATA`` environment variable, or the ``data_home`` argument) and
  read from there the next time, with no network;
* ``download_if_missing=False`` refuses to download;
* `clear_data_home` deletes it all;
* `fetch_openneuro` returns a `Bunch`: a dict whose keys are also attributes,
  with ``DESCR`` describing where the data came from.

Only the window asked for is downloaded where the format allows it:
- **BrainVision** (multiplexed) and **EDF/BDF**: by byte range;
- **EEGLAB, FIF and other formats**: whole files, up to ``max_mb``.

The archive's sidecars come with the window, so the channel types, the bad
channels, the mains frequency, the events and the electrode positions are the
dataset's own:
- ``channels.tsv``;
- ``events.tsv``;
- ``*_ieeg.json``;
- ``electrodes.tsv`` and ``coordsystem.json``.

They are found by BIDS inheritance, so a sidecar at the dataset's top level
counts.
"""

from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["fetch_openneuro", "list_openneuro", "describe_openneuro", "catalogue",
           "build_catalogue", "allow_network", "Cancelled", "CHUNK_BYTES", "get_data_home", "clear_data_home", "Bunch",
           "DATA_HOME_ENV", "SIGNAL_FORMATS", "CATALOGUE"]

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATA_HOME_ENV = "ONSET_HFO_DATA"
#: Signal files OpenNeuro holds, by extension: (format, can a window be cut by byte range).
SIGNAL_FORMATS = {".vhdr": ("BrainVision", True), ".edf": ("EDF", True),
                  ".bdf": ("BDF", True), ".set": ("EEGLAB", False),
                  ".fif": ("FIF", False), ".nwb": ("NWB", False), ".mefd": ("MEF3", False)}
#: Files whose readers this package cannot use for a remote recording.
_NOT_FETCHABLE = {".nwb": "NWB needs pynwb, which this package does not use",
                  ".mefd": "a MEF3 recording is a directory of many files; download it "
                           "whole with the OpenNeuro client and open it with File → Open"}
_ENTITIES = ("sub", "ses", "task", "acq", "run")
_MODALITIES = ("ieeg", "eeg", "meg")


class Bunch(dict):
    """A dict whose keys are also attributes, as `sklearn.utils.Bunch` is."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return list(self.keys())


#: The OpenNeuro datasets with continuous EEG or iEEG, bundled so a dataset can be
#: found by name without the network; `build_catalogue` makes it, `catalogue` reads it.
CATALOGUE = Path(__file__).with_name("openneuro_catalogue.csv")
CATALOGUE_COLUMNS = ["dataset_id", "name", "modalities", "subjects", "license", "doi"]


# -- the network, when asked for ---------------------------------------------------------------
class allow_network:
    """Lift ``ONSET_HFO_OFFLINE`` for the calls inside, then put it back.

    The desktop app runs offline unless started with ``--allow-fetch``, so it
    never reaches for an archive unasked. Listing or downloading from the
    OpenNeuro dialog *is* the asking, so the dialog runs its calls inside this.
    """

    def __enter__(self):
        import os

        from onset_hfo.datasets import OFFLINE_ENV

        self._previous = os.environ.pop(OFFLINE_ENV, None)
        return self

    def __exit__(self, *_):
        import os

        from onset_hfo.datasets import OFFLINE_ENV

        if self._previous is not None:
            os.environ[OFFLINE_ENV] = self._previous
        return False


# -- the catalogue ----------------------------------------------------------------------------
def catalogue(modality: str | None = None, search: str | None = None,
              data_home=None) -> pd.DataFrame:
    """The OpenNeuro datasets with continuous EEG or iEEG: ``dataset_id``,
    ``name``, ``modalities`` (e.g. "ieeg" or "eeg,ieeg"), ``subjects``,
    ``license`` and ``doi``.

    Read from a refreshed copy in the data home when `build_catalogue` has made
    one there, else from the copy bundled with the package. `modality` keeps
    datasets with that kind of recording; `search` keeps those whose id or name
    contains every word of it (any case)."""
    refreshed = get_data_home(data_home) / "openneuro_catalogue.csv"
    path = refreshed if refreshed.exists() else CATALOGUE
    if not path.exists():
        table = pd.DataFrame(columns=CATALOGUE_COLUMNS)
    else:
        table = pd.read_csv(path, dtype={"name": str, "modalities": str, "license": str,
                                         "doi": str}, keep_default_na=False)
    if modality:
        table = table[table["modalities"].str.split(",").apply(lambda m: modality in m)]
    for word in str(search or "").lower().split():
        table = table[table["dataset_id"].str.lower().str.contains(word, regex=False)
                      | table["name"].str.lower().str.contains(word, regex=False)]
    return table.reset_index(drop=True)


def _catalogue_row(dataset_id: str) -> dict | None:
    """One dataset's row, from two small requests: the top of its listing (for
    its subjects and kinds of recording) and its ``dataset_description.json``."""
    import urllib.parse

    from onset_hfo import datasets

    # From the first subject's files, not the dataset's: "derivatives/" sorts
    # before "sub-" and can fill the first thousand keys on its own.
    query = {"list-type": "2", "prefix": f"{dataset_id}/sub-", "max-keys": "1000"}
    xml = datasets._http_get(f"{BUCKET}/?{urllib.parse.urlencode(query)}",
                             retries=2).decode("utf-8", "replace")
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    kinds = sorted({part for key in keys for part in key.split("/")[2:-1]
                    if part in ("eeg", "ieeg")})
    if not kinds:
        return None
    query = {"list-type": "2", "prefix": f"{dataset_id}/", "delimiter": "/",
             "max-keys": "1000"}
    tops = datasets._http_get(f"{BUCKET}/?{urllib.parse.urlencode(query)}",
                              retries=2).decode("utf-8", "replace")
    subjects = len(re.findall(r"<Prefix>[^<]*/sub-[^<]*/</Prefix>", tops))
    try:
        meta = json.loads(datasets._http_get(f"{BUCKET}/{dataset_id}/dataset_description.json",
                                             retries=2))
    except Exception:       # noqa: BLE001 - a dataset without one still counts
        meta = {}
    name = " ".join(str(meta.get("Name", "")).split())
    return {"dataset_id": dataset_id, "name": name or dataset_id,
            "modalities": ",".join(kinds), "subjects": subjects,
            "license": str(meta.get("License", "") or ""),
            "doi": str(meta.get("DatasetDOI", "") or "").replace("doi:", "").strip()}


def build_catalogue(out=None, workers: int = 16, progress=print) -> pd.DataFrame:
    """Survey every dataset on OpenNeuro and keep those with EEG or iEEG.

    About three small requests per dataset, a few minutes in all. Writes the
    table to `out` (default: the data home, where `catalogue` will prefer it
    to the bundled copy) and returns it."""
    import concurrent.futures as futures
    import urllib.parse

    from onset_hfo import datasets

    ids, token = [], None
    while True:
        query = {"list-type": "2", "delimiter": "/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        xml = datasets._http_get(f"{BUCKET}/?{urllib.parse.urlencode(query)}").decode()
        ids += re.findall(r"<Prefix>(ds\d{6})/</Prefix>", xml)
        more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not more:
            break
        token = more.group(1)
    progress(f"{len(ids)} datasets on OpenNeuro; surveying them")
    rows, failed = [], []
    with futures.ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(_catalogue_row, i): i for i in ids}
        for n, job in enumerate(futures.as_completed(jobs), 1):
            try:
                row = job.result()
            except Exception:       # noqa: BLE001 - counted, not fatal
                failed.append(jobs[job])
                continue
            if row:
                rows.append(row)
            if n % 200 == 0:
                progress(f"{n} of {len(ids)} surveyed, {len(rows)} with EEG or iEEG")
    table = pd.DataFrame(rows, columns=CATALOGUE_COLUMNS).sort_values("dataset_id")
    out = Path(out) if out else get_data_home() / "openneuro_catalogue.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    progress(f"{len(table)} datasets with EEG or iEEG written to {out}"
             + (f"; {len(failed)} could not be read: {', '.join(sorted(failed)[:10])}"
                if failed else ""))
    return table.reset_index(drop=True)


# -- the data home ----------------------------------------------------------------------------
def get_data_home(data_home=None) -> Path:
    """Where downloads are kept: `data_home`, else ``$ONSET_HFO_DATA``, else
    the package's data cache (``~/onset_hfo/data/openneuro`` by default)."""
    import os

    from onset_hfo.config import DATA_CACHE

    home = Path(data_home or os.environ.get(DATA_HOME_ENV) or Path(DATA_CACHE) / "openneuro")
    home = home.expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def clear_data_home(data_home=None) -> None:
    """Delete everything this module has downloaded."""
    shutil.rmtree(get_data_home(data_home), ignore_errors=True)


# -- reaching the archive ---------------------------------------------------------------------
def _get(path: str, byte_range=None) -> bytes:
    import urllib.parse

    from onset_hfo import datasets

    # S3 reads a bare "+" in a path as a space.
    return datasets._http_get(f"{BUCKET}/{urllib.parse.quote(path)}", byte_range=byte_range)


def _check_id(dataset_id: str) -> str:
    dataset_id = str(dataset_id).strip()
    if not re.fullmatch(r"ds\d{6}", dataset_id):
        raise ValueError(f"{dataset_id!r} is not an OpenNeuro dataset id (ds followed by six "
                         "digits, as in ds004100)")
    return dataset_id


def _keys(dataset_id: str, data_home=None, refresh: bool = False,
          download_if_missing: bool = True) -> dict[str, int]:
    """Every file in the dataset and its size, from one paginated S3 listing,
    kept in the data home so the next call needs no network."""
    import urllib.parse

    from onset_hfo import datasets

    dataset_id = _check_id(dataset_id)
    cached = get_data_home(data_home) / dataset_id / "listing.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())
    if not download_if_missing:
        raise OSError(f"{dataset_id} has not been listed into {cached.parent} yet, and "
                      "download_if_missing is False")
    found: dict[str, int] = {}
    token = None
    while True:
        query = {"list-type": "2", "prefix": f"{dataset_id}/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        xml = datasets._http_get(f"{BUCKET}/?{urllib.parse.urlencode(query)}").decode(
            "utf-8", "replace")
        for key, size in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml,
                                    flags=re.DOTALL):
            found[key.split("/", 1)[1]] = int(size)
        more = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not more:
            break
        token = more.group(1)
    if not found:
        raise ValueError(f"OpenNeuro has no dataset {dataset_id} (the listing is empty)")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(found))
    return found


def _text(dataset_id: str, name: str, data_home=None, download_if_missing=True) -> str | None:
    """A small file from the dataset, kept in the data home."""
    path = get_data_home(data_home) / dataset_id / "files" / name
    if path.exists():
        return path.read_text("utf-8", "replace")
    if not download_if_missing:
        return None
    try:
        blob = _get(f"{dataset_id}/{name}")
    except RuntimeError:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob.decode("utf-8", "replace")


def _entities(name: str) -> dict[str, str]:
    stem = Path(name).name
    return {k: m.group(1) for k in _ENTITIES
            for m in [re.search(rf"(?:^|_){k}-([A-Za-z0-9]+)", stem)] if m}


# -- describing and listing -------------------------------------------------------------------
def describe_openneuro(dataset_id: str, data_home=None, download_if_missing=True) -> Bunch:
    """What the dataset is and how to cite it: its name, authors, licence and
    DOI from ``dataset_description.json``, its README, and its subjects."""
    dataset_id = _check_id(dataset_id)
    keys = _keys(dataset_id, data_home, download_if_missing=download_if_missing)
    meta = json.loads(_text(dataset_id, "dataset_description.json", data_home,
                            download_if_missing) or "{}")
    readme_name = next((k for k in ("README", "README.md", "README.txt") if k in keys), None)
    readme = (_text(dataset_id, readme_name, data_home, download_if_missing) or "") \
        if readme_name else ""
    subjects = sorted({k.split("/")[0] for k in keys if k.startswith("sub-")})
    recordings = list_openneuro(dataset_id, data_home=data_home,
                                download_if_missing=download_if_missing)
    authors = [str(a).strip() for a in meta.get("Authors", []) if str(a).strip()]
    doi = str(meta.get("DatasetDOI", "") or "").replace("doi:", "").strip()
    name = str(meta.get("Name", dataset_id))
    who = ", ".join(authors[:6]) + (" et al" if len(authors) > 6 else "")
    citation = (f"{who}. " if who else "") + f"{name.rstrip('.')}. OpenNeuro {dataset_id}" \
        + (f". doi:{doi}" if doi else "")
    descr = "\n".join(filter(None, [
        f"{name} (OpenNeuro {dataset_id})",
        f"Licence: {meta.get('License', 'not stated')}",
        f"Authors: {', '.join(authors)}" if authors else "",
        f"DOI: {doi}" if doi else "",
        f"{len(subjects)} subjects, {len(recordings)} recordings "
        f"({', '.join(sorted(set(recordings['format'])))})" if len(recordings) else
        f"{len(subjects)} subjects, no continuous recordings this module can read",
        "", readme.strip()[:4000]]))
    return Bunch(dataset_id=dataset_id, name=name, authors=authors,
                 license=meta.get("License", ""), doi=doi, citation=citation,
                 readme=readme, subjects=subjects, recordings=recordings,
                 url=f"https://openneuro.org/datasets/{dataset_id}", DESCR=descr)


def list_openneuro(dataset_id: str, subject: str | None = None, modality: str | None = None,
                   data_home=None, refresh: bool = False,
                   download_if_missing: bool = True) -> pd.DataFrame:
    """One row per continuous recording in the dataset.

    Columns: ``subject``, ``session``, ``task``, ``acq``, ``run``,
    ``modality`` (ieeg, eeg or meg), ``format``, ``size_mb``, ``windowed``
    (whether a window can be fetched without the whole file), and ``path``
    (the file in the archive).
    """
    keys = _keys(dataset_id, data_home, refresh, download_if_missing)
    rows = []
    for key, size in keys.items():
        parts = key.split("/")
        if not parts[0].startswith("sub-"):
            continue
        folder_mod = parts[-2] if len(parts) >= 2 else ""
        name = parts[-1]
        mefd = next((p for p in parts if p.endswith(".mefd")), None)
        if mefd:
            # A MEF3 recording is a directory; list it once, by its name.
            if parts.index(mefd) != len(parts) - 2 or not name.endswith(".tmet"):
                continue
            name, folder_mod = mefd, parts[parts.index(mefd) - 1]
            key = "/".join(parts[:parts.index(mefd) + 1])
        suffix = next((e for e in SIGNAL_FORMATS if name.endswith(e)), None)
        if suffix is None or folder_mod not in _MODALITIES:
            continue
        if not name[: -len(suffix)].endswith(f"_{folder_mod}"):
            continue
        if suffix == ".vhdr":   # the signal is the .eeg beside it
            size = keys.get(key[: -len(".vhdr")] + ".eeg",
                            keys.get(key[: -len(".vhdr")] + ".dat", size))
        elif suffix == ".set":
            size += keys.get(key[: -len(".set")] + ".fdt", 0)
        found = _entities(name)
        rows.append({"subject": f"sub-{found['sub']}" if "sub" in found else parts[0],
                     "session": f"ses-{found['ses']}" if "ses" in found else "",
                     "task": found.get("task", ""), "acq": found.get("acq", ""),
                     "run": found.get("run", ""), "modality": folder_mod,
                     "format": SIGNAL_FORMATS[suffix][0],
                     "size_mb": round(size / 1e6, 1),
                     "windowed": SIGNAL_FORMATS[suffix][1], "path": key})
    columns = ["subject", "session", "task", "acq", "run", "modality", "format", "size_mb",
               "windowed", "path"]
    table = pd.DataFrame(rows, columns=columns)
    if subject is not None:
        table = table[table["subject"] == _sub(subject)]
    if modality is not None:
        table = table[table["modality"] == modality]
    return table.sort_values(["subject", "session", "task", "acq", "run", "path"]) \
        .reset_index(drop=True)


def _sub(subject: str) -> str:
    subject = str(subject)
    return subject if subject.startswith("sub-") else f"sub-{subject}"


# -- fetching ---------------------------------------------------------------------------------
#: The signal is downloaded in pieces of this size, so a download can say how far
#: it has got and can be stopped between pieces.
CHUNK_BYTES = 4 * 1024 * 1024


class Cancelled(RuntimeError):
    """The download was stopped before it finished; nothing of it is kept."""


class _Transfer:
    """The signal's bytes, fetched in `CHUNK_BYTES` pieces, with progress.

    `progress(fraction, message)` is called after each piece; `should_stop()`
    is asked before each, and a True answer raises `Cancelled`."""

    def __init__(self, progress=None, should_stop=None):
        self.progress, self.should_stop = progress, should_stop
        self.total = self.done = 0

    def expect(self, n_bytes: int) -> None:
        self.total += max(0, int(n_bytes))
        self._tell()

    def get(self, key: str, first: int, last: int) -> bytes:
        """Bytes `first`..`last` (inclusive) of `key`."""
        parts, pos = [], first
        while pos <= last:
            if self.should_stop is not None and self.should_stop():
                raise Cancelled("the download was stopped before it finished")
            end = min(last, pos + CHUNK_BYTES - 1)
            blob = _get(key, (pos, end))
            if len(blob) > end - pos + 1:
                # A server that ignores Range sends the whole file every time:
                # take the window from it once rather than once per piece.
                whole = blob[first:last + 1]
                self._add(len(whole) - sum(len(p) for p in parts))
                return whole
            parts.append(blob)
            self._add(len(blob))
            if len(blob) < end - pos + 1:       # the file ended early
                break
            pos = end + 1
        return b"".join(parts)

    def _add(self, n_bytes: int) -> None:
        self.done += n_bytes
        self._tell()

    def _tell(self) -> None:
        if self.progress is not None:
            fraction = min(1.0, self.done / self.total) if self.total else 0.0
            self.progress(fraction, f"{self.done / 1e6:.1f} of {self.total / 1e6:.1f} MB")


def fetch_openneuro(dataset_id: str, subject: str | None = None, *,
                    session: str | None = None, task: str | None = None,
                    acq: str | None = None, run: str | None = None,
                    modality: str | None = None, t_start: float = 0.0,
                    t_stop: float | None = 60.0, data_home=None,
                    download_if_missing: bool = True, max_mb: float = 500.0,
                    verbose: bool = True, progress=None, should_stop=None) -> Bunch:
    """A window of one recording, downloaded once and kept in the data home.

    The recording is the first in `list_openneuro` order that matches the
    given entities; every entity left out matches anything, and the one
    chosen is in the result (``subject``, ``session``, ``task``, ``acq``,
    ``run``, ``file``). ``t_stop=None`` fetches to the end of the recording.

    Returns a `Bunch` with:

    - ``recording``: the package's `Recording` (an MNE Raw with provenance),
      which the pipeline, the reviewer and every study take as it is;
    - ``data`` (channels x samples, volts), ``times`` (seconds in the
      original recording), ``ch_names``, ``sfreq``;
    - the dataset's own sidecars: ``channels``, ``events``, ``electrodes``
      (DataFrames or None), ``coordsystem`` and ``sidecar`` (dicts), and the
      subject's row of ``participants.tsv`` as ``participant``;
    - ``line_freq``, ``license``, ``citation``, ``url``, ``local_path`` and
      ``DESCR``.

    `progress(fraction, message)` hears how far the download has got, after
    each `CHUNK_BYTES` piece; `should_stop()` is asked before each piece, and
    True stops it with `Cancelled`, keeping nothing of the window.
    """
    dataset_id = _check_id(dataset_id)
    table = list_openneuro(dataset_id, data_home=data_home,
                           download_if_missing=download_if_missing)
    chosen = _choose(table, dataset_id, subject=subject, session=session, task=task,
                     acq=acq, run=run, modality=modality)
    path = chosen["path"]
    suffix = next(e for e in SIGNAL_FORMATS if path.endswith(e))
    if suffix in _NOT_FETCHABLE:
        raise ValueError(f"{path}: {_NOT_FETCHABLE[suffix]}")
    if t_stop is not None and t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")
    window = f"{t_start:g}-{'end' if t_stop is None else f'{t_stop:g}'}s"
    folder = get_data_home(data_home) / dataset_id / Path(path[: -len(suffix)]).name / window
    meta_path = folder / "window.json"
    if not meta_path.exists():
        if not download_if_missing:
            raise OSError(f"{path} [{window}] is not in {folder.parent}, and "
                          "download_if_missing is False")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            _download(dataset_id, chosen, suffix, t_start, t_stop, folder, data_home, max_mb,
                      verbose, _Transfer(progress, should_stop))
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)   # never leave half a window
            raise
    elif verbose:
        print(f"[onset-hfo] using {folder}")
    return _load(folder, json.loads(meta_path.read_text()), data_home, verbose)


def _download(dataset_id, chosen, suffix, t_start, t_stop, folder: Path, data_home, max_mb,
              verbose, transfer: _Transfer) -> None:
    path = chosen["path"]
    stem = path[: -len(suffix)]
    keys = _keys(dataset_id, data_home)
    if verbose:
        window = f"{t_start:g}-{'end' if t_stop is None else f'{t_stop:g}'} s"
        print(f"[onset-hfo] fetching {path} [{window}] from OpenNeuro {dataset_id}")
    if suffix == ".vhdr":
        local, offset, length = _brainvision_window(dataset_id, path, keys, t_start, t_stop,
                                                    folder, max_mb, transfer)
    elif suffix in (".edf", ".bdf"):
        local, offset, length = _edf_window(dataset_id, path, suffix, t_start, t_stop, folder,
                                            max_mb, transfer)
    else:
        local, offset, length = _whole_file(dataset_id, path, suffix, keys, folder, max_mb,
                                            transfer)
    sidecars = _fetch_sidecars(dataset_id, stem, chosen["modality"], keys, folder, data_home)
    # Written last: its presence is what marks the window as complete.
    (folder / "window.json").write_text(json.dumps({
        "dataset_id": dataset_id, "path": path, "local": local.name, "offset": offset,
        "length": length, "t_start": t_start, "t_stop": t_stop,
        **{k: chosen[k] for k in ("subject", "session", "task", "acq", "run", "modality",
                                  "format")},
        "sidecars": sidecars}, indent=1))


def _choose(table: pd.DataFrame, dataset_id: str, **wanted) -> dict:
    if not len(table):
        raise ValueError(f"{dataset_id} has no continuous recording in a format this module "
                         f"reads ({', '.join(f for f, _ in SIGNAL_FORMATS.values())})")
    rows = table
    for column, value in wanted.items():
        if value is None:
            continue
        value = str(value)
        if column == "subject":
            value = _sub(value)
        elif column == "session" and value and not value.startswith("ses-"):
            value = f"ses-{value}"
        elif column == "run" and value.isdigit():
            rows = rows[rows["run"].str.lstrip("0") == value.lstrip("0")]
            continue
        rows = rows[rows[column] == value]
        if not len(rows):
            have = sorted(set(table[column]) - {""})
            raise ValueError(f"no recording in {dataset_id} with {column}={value!r}; it has "
                             f"{', '.join(have[:20]) or 'none'}"
                             f"{' …' if len(have) > 20 else ''}")
    return rows.iloc[0].to_dict()


def _brainvision_window(dataset_id, path, keys, t_start, t_stop, folder: Path, max_mb,
                        transfer: _Transfer):
    from onset_hfo import datasets

    header_text = _get(f"{dataset_id}/{path}").decode("utf-8", "replace")
    header = datasets._parse_vhdr(header_text)
    data_name = header.get("DataFile", Path(path).with_suffix(".eeg").name)
    data_key = str(Path(path).parent / data_name)
    total = keys.get(data_key)
    frame = header["n_channels"] * header["bytes_per_sample"]
    sfreq = header["sfreq"]
    n_samples = (total // frame) if total else None
    first = int(round(t_start * sfreq))
    last = int(round(t_stop * sfreq)) if t_stop is not None else n_samples
    if n_samples is not None:
        last = min(last, n_samples)
    if last is None or last <= first:
        raise ValueError(f"the window starts after the recording ends "
                         f"({(n_samples or 0) / sfreq:g} s)")
    _check_size((last - first) * frame, max_mb, path)
    transfer.expect((last - first) * frame)
    payload = transfer.get(f"{dataset_id}/{data_key}", first * frame, last * frame - 1)
    usable = (len(payload) // frame) * frame
    (folder / data_name).write_bytes(payload[:usable])
    marker = header.get("MarkerFile", Path(data_name).with_suffix(".vmrk").name)
    (folder / marker).write_text(datasets._MINIMAL_VMRK.format(data_file=data_name), "utf-8")
    local = folder / Path(path).name
    local.write_text(header_text, "utf-8")
    return local, first / sfreq, usable // frame / sfreq


def _edf_window(dataset_id, path, suffix, t_start, t_stop, folder: Path, max_mb,
                transfer: _Transfer):
    """The data records of a remote EDF/BDF that cover the window, as a short
    file of the same format. Records are whole seconds or so, so the window
    is cut exactly after reading."""
    width = 3 if suffix == ".bdf" else 2
    key = f"{dataset_id}/{path}"
    head = _get(key, (0, 255))
    header_bytes = int(head[184:192].decode().strip())
    n_records = int(head[236:244].decode().strip())
    record_s = float(head[244:252].decode().strip())
    n_signals = int(head[252:256].decode().strip())
    discontinuous = head[192:197].decode(errors="replace").startswith("EDF+D")
    header = _get(key, (0, header_bytes - 1))
    offset = 256 + n_signals * 216
    samples = [int(header[offset + 8 * i: offset + 8 * i + 8].decode().strip())
               for i in range(n_signals)]
    record_bytes = width * sum(samples)
    if discontinuous or n_records < 0 or record_s <= 0:
        # Discontinuous records are not equally spaced in time: no byte range
        # is a time window, so the file is read whole.
        size = header_bytes + max(n_records, 0) * record_bytes
        _check_size(size, max_mb, path)
        transfer.expect(size)
        local = folder / Path(path).name
        local.write_bytes(transfer.get(key, 0, size - 1))
        return local, 0.0, None
    first = max(0, int(np.floor(t_start / record_s)))
    last = n_records if t_stop is None else min(n_records, int(np.ceil(t_stop / record_s)))
    if last <= first:
        raise ValueError(f"the window starts after the recording ends "
                         f"({n_records * record_s:g} s)")
    _check_size((last - first) * record_bytes, max_mb, path)
    transfer.expect((last - first) * record_bytes)
    data = transfer.get(key, header_bytes + first * record_bytes,
                        header_bytes + last * record_bytes - 1)
    count = len(data) // record_bytes
    header = bytearray(header)
    header[236:244] = f"{count:<8d}".encode()
    local = folder / Path(path).name
    local.write_bytes(bytes(header) + data[:count * record_bytes])
    return local, first * record_s, count * record_s


def _whole_file(dataset_id, path, suffix, keys, folder: Path, max_mb, transfer: _Transfer):
    companions = [path]
    if suffix == ".set" and path[: -4] + ".fdt" in keys:
        companions.append(path[: -4] + ".fdt")
    _check_size(sum(keys.get(p, 0) for p in companions), max_mb, path,
                whole=True)
    transfer.expect(sum(keys.get(p, 0) for p in companions))
    for name in companions:
        size = keys.get(name, 0)
        blob = transfer.get(f"{dataset_id}/{name}", 0, size - 1) if size \
            else _get(f"{dataset_id}/{name}")
        (folder / Path(name).name).write_bytes(blob)
    return folder / Path(path).name, 0.0, None


def _check_size(n_bytes: int, max_mb: float, path: str, whole: bool = False) -> None:
    if n_bytes / 1e6 > max_mb:
        if whole:
            raise ValueError(f"{path} can only be downloaded whole, and it is "
                             f"{n_bytes / 1e6:.0f} MB, more than max_mb={max_mb:g}. "
                             "Raise max_mb to fetch it.")
        raise ValueError(f"The window asked for is {n_bytes / 1e6:.0f} MB of {path}, more "
                         f"than max_mb={max_mb:g}. Ask for a shorter window or raise max_mb.")


def _sidecar_key(keys, stem: str, suffix: str) -> str | None:
    """The most specific file named ``*_{suffix}`` that applies to `stem`
    under BIDS inheritance: in its folder or any above it, with entities
    that are a subset of the recording's."""
    mine = _entities(stem)
    folders = {str(Path(stem).parent)}
    parent = Path(stem).parent
    while str(parent) not in (".", ""):
        parent = parent.parent
        folders.add("" if str(parent) == "." else str(parent))
    best, best_score = None, -1
    for key in keys:
        if not key.endswith(f"_{suffix}") and Path(key).name != suffix:
            continue
        folder = str(Path(key).parent)
        folder = "" if folder == "." else folder
        if folder not in folders:
            continue
        theirs = _entities(key)
        if any(mine.get(k) != v for k, v in theirs.items()):
            continue
        score = len(theirs) * 10 + folder.count("/") + (folder != "")
        if score > best_score:
            best, best_score = key, score
    return best


def _fetch_sidecars(dataset_id, stem, modality, keys, folder: Path,
                    data_home=None) -> dict[str, str]:
    """The window's sidecars, saved beside it; returns which were found."""
    found = {}
    for suffix, name in (("channels.tsv", "channels.tsv"), ("events.tsv", "events.tsv"),
                         (f"{modality}.json", "sidecar.json"),
                         ("electrodes.tsv", "electrodes.tsv"),
                         ("coordsystem.json", "coordsystem.json")):
        key = _sidecar_key(keys, stem, suffix)
        if key is None and suffix in ("electrodes.tsv", "coordsystem.json"):
            # Electrodes are per session and often carry a space- entity the
            # recording has not; take the first in the recording's folder.
            here = str(Path(stem).parent)
            key = next((k for k in sorted(keys) if k.startswith(here + "/")
                        and k.endswith(f"_{suffix}")), None)
        if key is None:
            continue
        try:
            (folder / name).write_bytes(_get(f"{dataset_id}/{key}"))
            found[name] = key
        except RuntimeError:
            continue
    participants = _text(dataset_id, "participants.tsv", data_home)
    if participants:
        (folder / "participants.tsv").write_text(participants, "utf-8")
    return found


#: A label that is about a seizure. Without one, and outside an ictal task, an
#: "onset" or "start" in events.tsv is a stimulus or a task block, not a seizure.
_SEIZURE_WORD = re.compile(r"\bsz\b|seiz|ictal", re.IGNORECASE)


def _marks_seizures(events: pd.DataFrame | None, task: str) -> bool:
    """Whether this recording's events can be read for seizure markers: the
    task is ictal, or some marker names a seizure."""
    if _SEIZURE_WORD.search(str(task or "")):
        return True
    if events is None or "trial_type" not in events:
        return False
    return bool(events["trial_type"].astype(str).str.contains(_SEIZURE_WORD).any())


def _read_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t", na_values=["n/a"], keep_default_na=True)
    except Exception:       # noqa: BLE001 - an unreadable sidecar is a missing one
        return None


def _load(folder: Path, meta: dict, data_home, verbose: bool) -> Bunch:
    from onset_hfo import datasets
    from onset_hfo.io import open_recording

    dataset_id = meta["dataset_id"]
    channels = _read_tsv(folder / "channels.tsv")
    events = _read_tsv(folder / "events.tsv")
    electrodes = _read_tsv(folder / "electrodes.tsv")
    sidecar = json.loads((folder / "sidecar.json").read_text()) \
        if (folder / "sidecar.json").exists() else {}
    coordsystem = json.loads((folder / "coordsystem.json").read_text()) \
        if (folder / "coordsystem.json").exists() else {}
    line_freq = sidecar.get("PowerLineFrequency")
    try:
        line_freq = float(line_freq)
    except (TypeError, ValueError):
        line_freq = None
    types, bads = {}, []
    if channels is not None and "name" in channels:
        if "type" in channels:
            types = {str(n): datasets._mne_type(str(t))
                     for n, t in zip(channels["name"], channels["type"], strict=False)}
        if "status" in channels:
            bads = [str(n) for n, s in zip(channels["name"], channels["status"], strict=False)
                    if str(s).lower() == "bad"]

    offset = float(meta["offset"])
    start = max(0.0, float(meta["t_start"]) - offset)
    stop = None if meta["t_stop"] is None else float(meta["t_stop"]) - offset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = open_recording(folder / meta["local"], t_start=start, t_stop=stop,
                                subject=meta["subject"], line_freq=line_freq or 50.0,
                                channel_types=types, run=meta["run"] or "01",
                                task=meta["task"] or meta["modality"])
    raw = record.raw
    bads = [b for b in bads if b in raw.ch_names]
    raw.info["bads"] = bads
    record.t_offset = offset + start
    description = describe_openneuro(dataset_id, data_home)
    seizure_onset, seizure_offset, kind = (
        datasets._seizure_times(events, with_kind=True)
        if _marks_seizures(events, meta["task"]) else (None, None, "none"))
    record.source = f"openneuro:{dataset_id}"
    record.dataset_id = dataset_id
    record.citation = description.citation
    record.events, record.channels, record.bads = events, channels, bads
    record.seizure = (seizure_onset, seizure_offset)
    record.seizure_marker = kind
    record.marked_contacts = datasets.parse_marked_contacts(events, raw.ch_names,
                                                            record.seizure)
    record.line_freq = line_freq or 50.0
    if events is not None:
        datasets._attach_annotations(raw, events, record.t_offset)
    record.notes = [
        f"OpenNeuro {dataset_id}: {meta['path']}",
        f"window {record.t_offset:g}–{record.t_offset + record.duration:g} s of the "
        "original recording; all times reported by this pipeline are in "
        "original-recording seconds",
        (f"mains frequency {line_freq:g} Hz, from the dataset's {meta['modality']}.json"
         if line_freq else "mains frequency not stated by the dataset; 50 Hz assumed — "
         "check it, since notching the wrong one leaves the interference in place"),
        ("channel types and bad channels from the dataset's channels.tsv"
         if channels is not None else
         "no channels.tsv: channel types are the file's own, which may be wrong"),
        f"licence: {description.license or 'not stated'}; cite: {description.citation}",
    ]
    participant = {}
    table = _read_tsv(folder / "participants.tsv")
    if table is not None and "participant_id" in table:
        row = table[table["participant_id"] == meta["subject"]]
        if len(row):
            participant = {k: (None if pd.isna(v) else v)
                           for k, v in row.iloc[0].to_dict().items()}
    if verbose:
        print(f"[onset-hfo] loaded {meta['subject']} {meta['path'].rsplit('/', 1)[-1]}: "
              f"{len(raw.ch_names)} channels, {record.duration:.1f} s @ {record.sfreq:g} Hz")
    return Bunch(
        recording=record, raw=raw, data=raw.get_data(), times=raw.times + record.t_offset,
        ch_names=list(raw.ch_names), sfreq=float(raw.info["sfreq"]),
        channels=channels, events=events, electrodes=electrodes, coordsystem=coordsystem,
        sidecar=sidecar, participant=participant, line_freq=line_freq,
        dataset_id=dataset_id, subject=meta["subject"], session=meta["session"],
        task=meta["task"], acq=meta["acq"], run=meta["run"], modality=meta["modality"],
        format=meta["format"], file=meta["path"], local_offset=offset,
        t_start=record.t_offset,
        t_stop=record.t_offset + record.duration, license=description.license,
        citation=description.citation, url=f"{description.url}/file-display/"
        + meta["path"].replace("/", ":"), local_path=folder / meta["local"],
        DESCR=f"{meta['path']}, {record.t_offset:g}–{record.t_offset + record.duration:g} s\n\n"
        + description.DESCR)

