"""The session as a workspace: every object it holds, named, typed and sized.

What MATLAB's Workspace and Spyder's Variable Explorer give a user is not
the data -- they had that -- but the *inventory*: a list that says what
exists, how big it is, and what is in it, before anything is opened. This
module is that inventory for a `ReviewSession`, Qt-free so the window and
the tests read the same list. The widget that shows it, and the windows that
open each entry, are in `onset_review.workspace`.

Nothing here computes anything new. Every entry is an object the session
already carries; the signal in volts is `raw`'s own array, fetched only when
an entry is opened or exported, because a minute of intracranial EEG is
five million numbers and the list must not cost that to draw.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Variable", "variables", "console_variables", "session_names", "describe",
           "as_table", "as_array", "as_tree", "export", "save_workspace", "load_workspace",
           "WORKSPACE_SUFFIXES", "GROUPS"]

#: The order the groups are listed in, and what each is.
GROUPS = (
    ("Signal", "the recording as analysed: the samples, their times, the contacts"),
    ("Results", "what the detectors found and how the channels rank"),
    ("Quality", "which contacts and seconds were usable, and the ICA record"),
    ("Anatomy", "where the contacts are and what the surgeon removed"),
    ("Your read", "the verdicts given at this window"),
    ("Setup", "the request that produced all of the above, step by step"),
    ("Console", "names you made in the Python console"),
)

#: Arrays up to this many elements get a min/max in their summary; beyond
#: it, the summary says the shape and the dtype and leaves the numbers to the
#: window that opens them.
SUMMARY_LIMIT = 20_000_000

#: Items shown of a long list or dict before "… N more".
PREVIEW = 6


@dataclass
class Variable:
    """One entry in the workspace list."""

    name: str
    kind: str
    size: str
    summary: str
    group: str
    about: str = ""
    #: The object itself, or None when it is produced on demand by `load`.
    value: object = None
    #: Produces the object when it is not held: the signal array, say.
    load: object = field(default=None, repr=False, compare=False)

    def get(self):
        if self.value is None and self.load is not None:
            self.value = self.load()
        return self.value

    @property
    def missing(self) -> bool:
        return self.value is None and self.load is None


# -- describing one object ----------------------------------------------------
def _short(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _shape(shape) -> str:
    return " × ".join(f"{int(n):,}" for n in shape) if len(shape) else "scalar"


def _preview_list(items) -> str:
    items = list(items)
    head = ", ".join(_short(str(x), 24) for x in items[:PREVIEW])
    rest = len(items) - PREVIEW
    return head + (f", … {rest:,} more" if rest > 0 else "")


def _is_event(value) -> bool:
    return is_dataclass(value) and type(value).__name__ == "Event"


def describe(value) -> tuple[str, str, str]:
    """(kind, size, summary) for any object the session holds."""
    if value is None:
        return "None", "", "not available for this recording"
    if isinstance(value, pd.DataFrame):
        columns = _preview_list(value.columns)
        return "DataFrame", f"{len(value):,} × {value.shape[1]}", f"columns: {columns}"
    if isinstance(value, pd.Series):
        return "Series", f"{len(value):,}", _preview_list(value.tolist())
    if isinstance(value, np.ndarray):
        kind = f"ndarray {value.dtype}"
        if value.size == 0:
            return kind, _shape(value.shape), "empty"
        if value.size <= SUMMARY_LIMIT and np.issubdtype(value.dtype, np.number):
            lo, hi = float(np.nanmin(value)), float(np.nanmax(value))
            return kind, _shape(value.shape), f"min {lo:.4g} · max {hi:.4g}"
        return kind, _shape(value.shape), f"{value.size:,} values"
    if hasattr(value, "info") and hasattr(value, "get_data") and hasattr(value, "ch_names"):
        info = value.info
        n_times = int(getattr(value, "n_times", 0))
        sfreq = float(info["sfreq"])
        return ("Raw (MNE)", f"{len(value.ch_names)} ch × {n_times:,}",
                f"{sfreq:g} Hz · {n_times / sfreq:.1f} s · volts")
    if isinstance(value, (list, tuple)):
        if value and all(_is_event(x) for x in value):
            accepted = sum(1 for x in value if getattr(x, "accepted", True))
            detectors = sorted({getattr(x, "detector", "") for x in value})
            return ("list[Event]", f"{len(value):,}",
                    f"{accepted:,} accepted · " + ", ".join(d for d in detectors if d))
        inner = type(value[0]).__name__ if value else ""
        kind = f"{type(value).__name__}[{inner}]" if inner else type(value).__name__
        return kind, f"{len(value):,}", _preview_list(value) if value else "empty"
    if isinstance(value, dict):
        pairs = (f"{k}: {_short(str(v), 20)}" for k, v in value.items())
        return "dict", f"{len(value):,}", _preview_list(pairs) if value else "empty"
    if is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        pairs = (f"{f.name}={_short(repr(getattr(value, f.name)), 18)}" for f in fields)
        return type(value).__name__, f"{len(fields)} fields", _preview_list(pairs)
    if isinstance(value, bool):
        return "bool", "", str(value)
    if isinstance(value, int):
        return "int", "", f"{value:,}"
    if isinstance(value, float):
        return "float", "", "nan" if math.isnan(value) else f"{value:g}"
    if isinstance(value, str):
        return "str", f"{len(value):,}", _short(value) if value else "empty"
    if isinstance(value, Path):
        return "Path", "", str(value)
    return type(value).__name__, "", _short(repr(value))


# -- the inventory -------------------------------------------------------------
def _entry(name: str, value, group: str, about: str, load=None) -> Variable:
    probe = value if load is None else load_shape(load)
    kind, size, summary = describe(probe)
    if load is not None:
        # Described from the shape alone; the array itself waits.
        kind, size, summary = probe
    return Variable(name, kind, size, summary, group, about, value, load)


def load_shape(load) -> tuple[str, str, str]:
    """`load` carries its own description on `load.describe`."""
    return load.describe


class _Lazy:
    """A loader that knows what it will produce without producing it."""

    def __init__(self, produce, describe: tuple[str, str, str]):
        self._produce = produce
        self.describe = describe

    def __call__(self):
        return self._produce()


def console_variables(names: dict) -> list[Variable]:
    """The names a console made, as workspace entries."""
    return [_entry(name, value, "Console", "Made in the Python console.")
            for name, value in sorted(names.items())]


def variables(session, console: dict | None = None) -> list[Variable]:
    """Every object the session holds, in reading order, grouped; then the
    names `console` (the console's own) adds."""
    extra = console_variables(console or {})
    if session is None:
        return extra
    return _session_variables(session) + extra


def session_names(session) -> dict:
    """Every name the Workspace lists for `session`, bound to its object: the
    console's view of the same workspace, so a name read in one is the name
    typed in the other.

    `signal` is the samples `raw` holds, read-only, rather than a copy: a
    copy of a long span is gigabytes, and writing into it would change the
    analysis under the panels. `np.array(signal)` makes a copy to change.
    """
    if session is None:
        return {}
    names = {}
    for variable in _session_variables(session):
        if variable.name == "signal":
            names["signal"] = _signal_view(session.raw)
        else:
            names[variable.name] = variable.get()
    return names


def _signal_view(raw) -> np.ndarray:
    held = getattr(raw, "_data", None)
    if getattr(raw, "preload", False) and isinstance(held, np.ndarray):
        view = held.view()
    else:
        view = raw.get_data()
    view.flags.writeable = False
    return view


def _session_variables(session) -> list[Variable]:
    request = session.request
    raw = session.raw
    out: list[Variable] = []

    def add(name, value, group, about, load=None):
        out.append(_entry(name, value, group, about, load))

    # Signal
    add("raw", raw, "Signal",
        "MNE's Raw object: the window as it was analysed, after preprocessing, "
        "in volts. Open it to see the samples channel by channel.")
    if raw is not None:
        n_ch, n_times = len(raw.ch_names), int(raw.n_times)
        sfreq = float(raw.info["sfreq"])
        add("signal", None, "Signal",
            "The same samples as a channels × time array, in volts. Fetched when "
            "opened or exported.",
            load=_Lazy(lambda: raw.get_data(),
                       ("ndarray float64", _shape((n_ch, n_times)),
                        f"{n_ch * n_times:,} values · fetched when opened")))
        add("times", None, "Signal",
            "The time of each sample, in original-recording seconds.",
            load=_Lazy(lambda: raw.times + float(session.t_offset),
                       ("ndarray float64", _shape((n_times,)),
                        f"{float(session.t_offset):g} to "
                        f"{float(session.t_offset) + (n_times - 1) / sfreq:.3f} s")))
        add("channels", list(raw.ch_names), "Signal",
            "The channel names after re-referencing: bipolar pairs, or contacts.")
    add("sfreq", float(session.sfreq), "Signal", "Sampling rate after any resampling, Hz.")
    add("montage", session.montage, "Signal", "The reference scheme in force.")
    add("t_offset", float(session.t_offset), "Signal",
        "Where this window starts in the original recording, seconds.")
    add("span", tuple(session.span), "Signal",
        "What was analysed, in original-recording seconds; the trace shows a "
        "slice of it when a longer span was asked for.")

    # Results
    add("events", list(session.events), "Results",
        "Every detection from every detector, accepted or rejected, with the "
        "measurements that decided it.")
    add("findings", session.findings, "Results",
        "One row per channel: rate, its interval, and the annotators' counts. "
        "The table the Ranking shows.")
    add("leader", dict(session.leader), "Results",
        "The busiest channel and whether the data can tell it apart from the next.")
    add("candidates", list(session.candidates), "Results",
        "The channels statistically tied with the busiest: the set, not a winner.")
    add("expert", list(session.expert), "Results",
        "The archive's expert markings in this window; empty for an imported file.")
    add("reviewed_channels", list(session.reviewed_channels), "Results",
        "The channels the archive's annotators looked at at all.")
    add("clean_seconds", dict(session.clean_seconds), "Results",
        "Channel → seconds actually analysed: what every rate was divided by.")

    # Quality
    add("quality", session.quality, "Quality",
        "One row per channel: the quality measurements and the verdict.")
    add("segments", session.segments, "Quality",
        "One row per channel per segment: which seconds were usable.")
    add("ica", session.ica, "Quality",
        "The ICA stage's record when it ran: components, scores, what was removed.")

    # Anatomy
    add("electrodes", session.electrodes, "Anatomy",
        "Measured contact coordinates from an electrodes.tsv, when there is one.")
    add("resection", session.resection, "Anatomy",
        "The surgeon's resected zone for this subject, when the archive ships "
        "a clinical sheet.")

    # Your read
    add("read", session.read, "Your read",
        "Your verdicts on events and contacts at this window, with your name "
        "and the draft findings paragraph.")

    # Setup
    add("request", request, "Setup",
        "Everything that was asked for: subject, window, band, detectors, "
        "threshold, preprocessing choices.")
    add("steps", list(session.steps), "Setup",
        "What was done to the signal, in order, as the report prints it.")
    add("notes", list(session.notes), "Setup", "Dataset notes carried with the window.")
    add("citation", session.citation, "Setup", "How to cite the data.")
    add("recording", session.recording, "Setup",
        "The fetched slice itself, as the pipeline received it: channel table, "
        "bad channels, ground truth, dataset id.")
    return out


# -- views of one object --------------------------------------------------------
def as_table(value) -> pd.DataFrame | None:
    """The object as a table, where there is a natural one; else None."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, (list, tuple)) and value and all(_is_event(x) for x in value):
        from onset_hfo.detectors.base import events_to_frame

        return events_to_frame(list(value))
    if isinstance(value, dict) and value and all(
            not isinstance(v, (dict, list, tuple)) or is_dataclass(v) is False and
            isinstance(v, (int, float, str, bool)) for v in value.values()):
        return pd.DataFrame({"key": list(value.keys()), "value": list(value.values())})
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(x, (int, float, str, bool)) for x in value):
        return pd.DataFrame({"value": list(value)})
    return None


def as_array(value) -> np.ndarray | None:
    """The object as a numeric array, where it is one; else None."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "get_data") and hasattr(value, "ch_names"):
        return value.get_data()
    return None


def as_tree(value, depth: int = 0, limit: int = 200) -> list[tuple[str, str, list]]:
    """The object as (key, value text, children) rows, for a tree view.
    Dataclasses and dicts open into their fields; lists into their items."""
    if depth > 4:
        return []
    rows: list[tuple[str, str, list]] = []

    def leaf(name, item):
        kind, size, summary = describe(item)
        children = as_tree(item, depth + 1, limit) if _opens(item) else []
        text = summary if not children else f"{kind} · {size}".strip(" ·")
        if item is None:
            # In a field list None is a value -- "use the default", usually --
            # not something missing from the recording.
            text = "None"
        rows.append((str(name), text, children))

    if is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            leaf(f.name, getattr(value, f.name))
    elif isinstance(value, dict):
        for key, item in list(value.items())[:limit]:
            leaf(key, item)
        if len(value) > limit:
            rows.append((f"… {len(value) - limit:,} more", "", []))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(list(value)[:limit]):
            leaf(index, item)
        if len(value) > limit:
            rows.append((f"… {len(value) - limit:,} more", "", []))
    return rows


def _opens(item) -> bool:
    return (isinstance(item, (dict, list, tuple)) and len(item) > 0) or (
        is_dataclass(item) and not isinstance(item, type))


# -- export ----------------------------------------------------------------------
def _plain(value):
    """JSON-ready: dataclasses to dicts, arrays to lists, paths to strings."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_dict()
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return repr(value)


def export_suffix(value) -> str:
    """The file type an object is written as: .csv for tables, .npy for
    arrays, .txt for text, .json for the rest."""
    if as_table(value) is not None:
        return ".csv"
    if as_array(value) is not None:
        return ".npy"
    if isinstance(value, str):
        return ".txt"
    return ".json"


def export(value, path: str | Path) -> Path:
    """Write `value` beside the name chosen, in the type `export_suffix`
    gives; a path with another suffix gets that suffix replaced."""
    path = Path(path).with_suffix(export_suffix(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    table = as_table(value)
    if table is not None:
        table.to_csv(path, index=False)
        return path
    array = as_array(value)
    if array is not None:
        np.save(path, array)
        return path
    if isinstance(value, str):
        path.write_text(value)
        return path
    path.write_text(json.dumps(_plain(value), indent=2, default=repr))
    return path


# -- saving and loading a workspace ------------------------------------------------
#: What a workspace can be saved as: (suffix, what it holds).
WORKSPACE_SUFFIXES = (
    (".pkl", "everything Python can pickle; open again in this software or Python"),
    (".npz", "arrays and numbers only; NumPy's own format"),
    (".mat", "arrays, numbers, text and tables; opens in MATLAB"),
)


def _numeric(value) -> np.ndarray | None:
    """`value` as a plain numeric array, or None if it is not one."""
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return np.asarray(value)
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value
    if isinstance(value, (list, tuple)) and value and all(
            isinstance(x, (bool, int, float, np.integer, np.floating)) for x in value):
        return np.asarray(value)
    return None


def _for_matlab(value):
    """`value` as scipy.io.savemat writes it, or None if it has no MATLAB form."""
    array = _numeric(value)
    if array is not None:
        return array
    if isinstance(value, str):
        return value
    if isinstance(value, pd.DataFrame):
        return {str(column): (value[column].to_numpy() if value[column].dtype != object
                              else value[column].astype(str).to_numpy(dtype=object))
                for column in value.columns}
    if isinstance(value, pd.Series):
        return value.to_numpy()
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, str) for x in value):
        return np.asarray(value, dtype=object)
    if isinstance(value, dict) and value and all(
            isinstance(k, str) and k.isidentifier() for k in value):
        inner = {k: _for_matlab(v) for k, v in value.items()}
        return inner if all(v is not None for v in inner.values()) else None
    return None


def save_workspace(names: dict, path: str | Path) -> tuple[Path, list[str], dict[str, str]]:
    """Write `names` to `path`, in the format its suffix names (see
    `WORKSPACE_SUFFIXES`). Returns (path, the names written, {name: why
    not} for the ones the format cannot hold). Nothing is written when
    nothing can be."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {s for s, _ in WORKSPACE_SUFFIXES}:
        raise ValueError(f"a workspace is saved as {', '.join(s for s, _ in WORKSPACE_SUFFIXES)}"
                         f", not {suffix or 'a file without a suffix'}")
    kept: dict = {}
    skipped: dict[str, str] = {}
    if suffix == ".pkl":
        return _pickle_workspace(names, path)
    for name, value in names.items():
        if suffix == ".npz":
            array = _numeric(value)
            if array is None:
                skipped[name] = "not an array or a number; save as .pkl to keep it"
                continue
            kept[name] = array
        else:
            if not name.isidentifier() or len(name) > 63:
                skipped[name] = "not a name MATLAB accepts"
                continue
            converted = _for_matlab(value)
            if converted is None:
                skipped[name] = "has no MATLAB form; save as .pkl to keep it"
                continue
            kept[name] = converted
    if not kept:
        return path, [], skipped
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".npz":
        np.savez(path, **kept)
    else:
        from scipy.io import savemat

        savemat(path, kept, long_field_names=True, oned_as="row")
    return path, list(kept), skipped


def _pickle_workspace(names: dict, path: Path) -> tuple[Path, list[str], dict[str, str]]:
    """All of `names` in one pickle; if one of them cannot be pickled, each
    is tried on its own and the ones that fail are left out. Tried whole
    first so an array of gigabytes is not pickled twice."""
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = pickle.dumps(dict(names), protocol=pickle.HIGHEST_PROTOCOL)
        skipped: dict[str, str] = {}
        kept = list(names)
    except Exception:      # noqa: BLE001 - narrowed below, name by name
        good, skipped = {}, {}
        for name, value in names.items():
            try:
                pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as error:      # noqa: BLE001 - reported, not raised
                skipped[name] = f"cannot be pickled ({type(error).__name__})"
                continue
            good[name] = value
        kept = list(good)
        if not good:
            return path, [], skipped
        payload = pickle.dumps(good, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(payload)
    return path, kept, skipped


def load_workspace(path: str | Path) -> dict:
    """The names a saved workspace holds. A .pkl file runs code when it is
    read, as every pickle does: open only one you made or trust."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        import pickle

        with open(path, "rb") as handle:
            loaded = pickle.load(handle)      # noqa: S301 - the reader chose the file
        if not isinstance(loaded, dict):
            return {path.stem: loaded}
        return {str(k): v for k, v in loaded.items()}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as held:
            return {name: (held[name].item() if held[name].ndim == 0 else held[name])
                    for name in held.files}
    if suffix == ".mat":
        from scipy.io import loadmat

        loaded = loadmat(path, squeeze_me=True, simplify_cells=True)
        return {k: v for k, v in loaded.items() if not k.startswith("__")}
    if suffix == ".npy":
        return {path.stem: np.load(path, allow_pickle=False)}
    raise ValueError(f"cannot load a workspace from {suffix or 'a file without a suffix'}: "
                     "use .pkl, .npz, .mat or .npy")
