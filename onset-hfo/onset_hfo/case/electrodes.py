"""Where a case's contacts are, on the template brain. Qt-free.

Without the patient's own CT and MRI, positions come from one of two places,
and every row says which (`source`):

* **imported** -- a planning or navigation system's export in a template
  space (MNI152, or FreeSurfer's fsaverage/MNI305, moved to MNI152 by
  `onset_hfo.case.atlas.fsaverage_to_mni152`);
* **planned** -- placed here: a depth electrode as a straight line from its
  target (the deepest contact) towards its entry point at the electrode's
  contact spacing; a strip or grid as a flat sheet from its first contact
  along two directions.

Positions are stored in MNI152NLin2009cAsym millimetres, as BIDS-iEEG
(``sub-*/ses-implant01/ieeg/*_space-MNI152NLin2009cAsym_electrodes.tsv``
with its ``coordsystem.json``), each with its probable atlas label. They are
template positions: approximate, and labelled so wherever they are shown.
How far off a straight-line plan is, against real implants on the same
template, is measured in `docs/TEMPLATE_MAP.md`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.case.atlas import ATLAS_SPACE, fsaverage_to_mni152

__all__ = ["ELECTRODE_COLUMNS", "SPACES", "read_coordinate_file", "plan_depth", "plan_sheet",
           "add_labels", "electrodes_path", "load_electrodes", "save_electrodes",
           "channel_positions", "contacts_of", "TEMPLATE_NOTE", "merge",
           "PATIENT_NOTE", "CAPTIONS", "positions_kind", "positions_note"]

ELECTRODE_COLUMNS = ("name", "x", "y", "z", "size", "group", "source", "space_from",
                     "label", "label_how", "label_mm")
#: Spaces a file can be in, as offered when importing.
SPACES = ("MNI152", "fsaverage")
TEMPLATE_NOTE = ("Template positions: approximate, not this patient's anatomy; atlas labels "
                 "are probable structures, not findings.")
#: The same, for contacts localised in the patient's own MRI (`onset_hfo.case.imaging`).
PATIENT_NOTE = ("Contacts from the patient's own MRI: placed in MNI by registering the "
                "whole head (an affine, then a non-linear warp inside the brain), so atlas "
                "labels are still probable structures, not findings.")
#: The short form figures carry, by where the positions came from.
CAPTIONS = {"template": "template positions, approximate",
            "patient": "the patient's MRI, registered to MNI",
            "mixed": "the patient's MRI and template positions"}
_NAME_COLUMNS = ("name", "label", "electrode", "contact", "channel", "ch_name")


def _frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(ELECTRODE_COLUMNS))
    for column in ("x", "y", "z", "label_mm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _group_of(name: str) -> str:
    match = re.match(r"^(.*?)(\d+)$", str(name).strip())
    return match.group(1) if match else str(name).strip()


def read_coordinate_file(path, space: str = "MNI152") -> pd.DataFrame:
    """A table of contact positions (TSV, CSV or whitespace; a name column and
    x, y, z), in `space`. Metres are recognised (every value under 1) and
    turned into millimetres; fsaverage is moved to MNI152. Rows without a
    position (BIDS writes "n/a" for scalp and ECG channels) are left out."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {', '.join(SPACES)}")
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    sep = "\t" if "\t" in text.splitlines()[0] else None
    table = pd.read_csv(path, sep=sep, engine="python", na_values=["n/a", "N/A", "nan"],
                        encoding="utf-8-sig")
    table.columns = [str(c).strip().lower() for c in table.columns]
    name = next((c for c in _NAME_COLUMNS if c in table.columns), None)
    if name is None or not {"x", "y", "z"} <= set(table.columns):
        raise ValueError(f"{path.name}: need a name column and x, y, z columns")
    table = table.dropna(subset=["x", "y", "z"])
    xyz = table[["x", "y", "z"]].astype(float).to_numpy()
    if not len(xyz):
        raise ValueError(f"{path.name} has no contact with a position")
    if np.nanmax(np.abs(xyz)) < 1.0:
        xyz = xyz * 1000.0
    if space == "fsaverage":
        xyz = fsaverage_to_mni152(xyz)
    rows = [{"name": str(n).strip(), "x": p[0], "y": p[1], "z": p[2],
             "size": table["size"].iloc[i] if "size" in table else np.nan,
             "group": _group_of(n), "source": "imported", "space_from": space,
             "label": "", "label_how": "", "label_mm": np.nan}
            for i, (n, p) in enumerate(zip(table[name], xyz, strict=True))]
    return _frame(rows)


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("the two points are the same")
    return vector / norm


def plan_depth(group: str, target, entry, n_contacts: int, spacing_mm: float,
               first: int = 1) -> pd.DataFrame:
    """A depth electrode: contact `first` at `target` (the deepest point), the
    rest every `spacing_mm` towards `entry`, in MNI152 mm."""
    if n_contacts < 1 or spacing_mm <= 0:
        raise ValueError("need at least one contact and a positive spacing")
    target = np.asarray(target, dtype=float)
    step = _unit(np.asarray(entry, dtype=float) - target) * float(spacing_mm)
    rows = [{"name": f"{group}{first + i}", "x": p[0], "y": p[1], "z": p[2], "size": np.nan,
             "group": group, "source": "planned", "space_from": "MNI152", "label": "",
             "label_how": "", "label_mm": np.nan}
            for i, p in enumerate(target + np.outer(np.arange(n_contacts), step))]
    return _frame(rows)


def plan_sheet(group: str, origin, along, across, rows: int, columns: int,
               spacing_mm: float, first: int = 1) -> pd.DataFrame:
    """A strip (one row) or grid, flat: contact `first` at `origin`, numbered
    along a row towards `along`, rows stepping towards `across` (made square
    to the row). The template cortex is curved and a grid is not; a sheet is
    as approximate as it looks."""
    if rows < 1 or columns < 1 or spacing_mm <= 0:
        raise ValueError("need at least one row and column and a positive spacing")
    origin = np.asarray(origin, dtype=float)
    u = _unit(np.asarray(along, dtype=float) - origin)
    v = np.asarray(across, dtype=float) - origin
    v = v - (v @ u) * u
    v = _unit(v) if rows > 1 else np.zeros(3)
    out = []
    for r in range(rows):
        for c in range(columns):
            p = origin + spacing_mm * (c * u + r * v)
            out.append({"name": f"{group}{first + r * columns + c}", "x": p[0], "y": p[1],
                        "z": p[2], "size": np.nan, "group": group, "source": "planned",
                        "space_from": "MNI152", "label": "", "label_how": "",
                        "label_mm": np.nan})
    return _frame(out)


def add_labels(frame: pd.DataFrame, atlas) -> pd.DataFrame:
    """Each contact's probable atlas structure."""
    frame = frame.copy()
    found = atlas.labels(frame[["x", "y", "z"]].to_numpy(float)) if len(frame) else []
    frame["label"] = [f.label for f in found]
    frame["label_how"] = [f.how for f in found]
    frame["label_mm"] = [f.distance_mm for f in found]
    return frame


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """`new` replaces contacts of the same name; the rest are kept."""
    if existing is None or not len(existing):
        return new.reset_index(drop=True)
    names = set(new["name"].str.upper())
    kept = existing[~existing["name"].str.upper().isin(names)]
    return pd.concat([kept, new], ignore_index=True)[list(ELECTRODE_COLUMNS)]


# -- the case's file ------------------------------------------------------------------------
def electrodes_path(case, session: str = "implant01") -> Path:
    folder = Path(case.root) / f"sub-{case.subject}" / f"ses-{session}" / "ieeg"
    return folder / f"sub-{case.subject}_ses-{session}_space-{ATLAS_SPACE}_electrodes.tsv"


def load_electrodes(case) -> pd.DataFrame:
    path = electrodes_path(case)
    if not path.exists():
        return _frame([])
    frame = pd.read_csv(path, sep="\t", na_values=["n/a"], keep_default_na=False,
                        dtype={"name": str, "group": str, "label": str})
    for column in ELECTRODE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan if column in ("x", "y", "z", "label_mm", "size") else ""
    frame[["label", "label_how", "source", "space_from", "group"]] = frame[
        ["label", "label_how", "source", "space_from", "group"]].fillna("")
    return frame[list(ELECTRODE_COLUMNS)]


def positions_kind(frame: pd.DataFrame) -> str:
    """"patient" when every contact came from the patient's own imaging,
    "mixed" when some did, "template" otherwise."""
    if frame is None or not len(frame) or "source" not in frame:
        return "template"
    patient = frame["source"].astype(str) == "patient"
    return "patient" if patient.all() else "mixed" if patient.any() else "template"


def positions_note(frame: pd.DataFrame) -> str:
    kind = positions_kind(frame)
    if kind == "patient":
        return PATIENT_NOTE
    if kind == "mixed":
        return f"{PATIENT_NOTE} The others: {TEMPLATE_NOTE[0].lower()}{TEMPLATE_NOTE[1:]}"
    return TEMPLATE_NOTE


def save_electrodes(case, frame: pd.DataFrame, action: str, detail: str = "",
                    by: str = "") -> Path:
    path = electrodes_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[list(ELECTRODE_COLUMNS)]
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a", float_format="%.2f")
    sources = sorted(set(frame["source"])) if len(frame) else []
    coordsystem = {
        "iEEGCoordinateSystem": ATLAS_SPACE, "iEEGCoordinateUnits": "mm",
        "iEEGCoordinateSystemDescription": "ICBM 152 Nonlinear Asymmetrical 2009c template",
        "iEEGCoordinateProcessingDescription": (
            "Positions from: " + ", ".join(sources) + ". " + positions_note(frame))}
    coord_path = path.with_name(path.name.replace("_electrodes.tsv", "_coordsystem.json"))
    coord_path.write_text(json.dumps(coordsystem, indent=1) + "\n")
    case.record(action, detail or f"{len(frame)} contact(s) now placed", by)
    return path


# -- positions of analysed channels ----------------------------------------------------------
def contacts_of(channel: str) -> list[str]:
    """The contacts a channel is made of: "LA1-LA2" -> ["LA1", "LA2"]."""
    return [part.strip().upper() for part in str(channel).split("-") if part.strip()]


def channel_positions(frame: pd.DataFrame, channels) -> pd.DataFrame:
    """Each channel's position: a contact's own, a bipolar pair's midpoint,
    nothing when a contact is not placed. Columns channel, x, y, z, placed."""
    where = {str(n).strip().upper(): (x, y, z) for n, x, y, z in
             zip(frame["name"], frame["x"], frame["y"], frame["z"], strict=True)} \
        if len(frame) else {}
    rows = []
    for channel in channels:
        parts = contacts_of(channel)
        points = [where[p] for p in parts if p in where]
        if parts and len(points) == len(parts):
            x, y, z = np.mean(np.asarray(points, dtype=float), axis=0)
            rows.append({"channel": channel, "x": x, "y": y, "z": z, "placed": True})
        else:
            rows.append({"channel": channel, "x": np.nan, "y": np.nan, "z": np.nan,
                         "placed": False})
    return pd.DataFrame(rows, columns=["channel", "x", "y", "z", "placed"])
