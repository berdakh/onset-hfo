"""How far is a straight-line plan from where depth contacts really are? ds004100.

The Map step lets a clinician without the patient's CT and MRI place a depth
electrode on the template brain as a straight line from its target (the
deepest contact) towards its entry point, a contact every few millimetres
(`onset_hfo.case.electrodes.plan_depth`). The question here is how much that
model loses, measured against real implants on the same template: the
OpenNeuro dataset ds004100 (HUP), whose SEEG contacts were localised on each
patient's imaging and published in fsaverage space.

Per depth shaft (contacts named by a prefix and a number, at least
`MIN_CONTACTS` with positions), the plan is given the best inputs a person
could have -- the real deepest contact as the target and the real last
contact as the entry direction -- and only one number to choose, the
spacing:

* **its own spacing** (the shaft's median distance between neighbouring
  contacts on the template): what the straight line alone costs;
* **one spacing for every shaft** (the median over all shafts);
* **3.5 mm** and **5 mm**, two common physical pitches -- what using the
  catalogue value costs on a template, where a brain warped to the average
  is not the patient's size.

Each planned contact is compared with the real one: the distance, and whether
the atlas (`onset_hfo.case.atlas`) gives both the same probable structure.
The real positions' own labels are reported too: how much an atlas label can
carry even when the position is right. A person choosing the target and entry
on a template adds error this study does not measure.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["run_study", "shafts", "summarise", "MIN_CONTACTS", "PITCHES"]

DATASET = "ds004100"
MIN_CONTACTS = 4
PITCHES = (3.5, 5.0)
_BUCKET = "https://s3.amazonaws.com/openneuro.org"


def _files() -> list[str]:
    import urllib.parse

    from onset_hfo.datasets import _http_get

    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": f"{DATASET}/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        text = _http_get(f"{_BUCKET}?{urllib.parse.urlencode(query)}").decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", text)
        found = re.search(r"<NextContinuationToken>([^<]*)<", text)
        if not found:
            break
        token = found.group(1)
    return sorted(k for k in keys if k.endswith("acq-seeg_space-fsaverage_electrodes.tsv"))


def shafts(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Contacts grouped by shaft (name prefix), ordered by number, keeping
    shafts with at least `MIN_CONTACTS` placed contacts."""
    out = {}
    parsed = frame.assign(
        shaft=frame["name"].str.extract(r"^(.*?)(\d+)$")[0],
        number=pd.to_numeric(frame["name"].str.extract(r"^(.*?)(\d+)$")[1], errors="coerce"))
    parsed = parsed.dropna(subset=["shaft", "number", "x", "y", "z"])
    for shaft, part in parsed.groupby("shaft"):
        part = part.sort_values("number").drop_duplicates("number")
        if len(part) >= MIN_CONTACTS:
            out[str(shaft)] = part.reset_index(drop=True)
    return out


def _spacing(part: pd.DataFrame) -> float:
    xyz = part[["x", "y", "z"]].to_numpy(float)
    steps = np.diff(part["number"].to_numpy(float))
    gaps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    single = gaps[steps == 1]
    return float(np.median(single)) if len(single) else float("nan")


def _plan(part: pd.DataFrame, spacing: float) -> np.ndarray:
    xyz = part[["x", "y", "z"]].to_numpy(float)
    direction = xyz[-1] - xyz[0]
    direction = direction / np.linalg.norm(direction)
    offsets = part["number"].to_numpy(float) - part["number"].iloc[0]
    return xyz[0] + np.outer(offsets * spacing, direction)


def run_study(out, atlas=None, progress=print) -> dict:
    from onset_hfo.case.atlas import Atlas
    from onset_hfo.case.electrodes import read_coordinate_file
    from onset_hfo.datasets import _http_get

    atlas = atlas or Atlas.load()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "electrodes"
    cache.mkdir(exist_ok=True)
    per_shaft = []
    groups = {}
    for key in _files():
        subject = key.split("/")[1]
        path = cache / Path(key).name
        if not path.exists():
            path.write_bytes(_http_get(f"{_BUCKET}/{key}"))
        frame = read_coordinate_file(path, space="fsaverage")
        found = shafts(frame)
        groups[subject] = found
        for shaft, part in found.items():
            per_shaft.append({"subject": subject, "shaft": shaft, "n": len(part),
                              "spacing_mm": _spacing(part)})
        progress(f"{subject}: {len(found)} shaft(s)")
    shaft_table = pd.DataFrame(per_shaft)
    common = float(shaft_table["spacing_mm"].median())
    variants = {"own": None, "common": common, **{f"{p:g}mm": p for p in PITCHES}}
    rows = []
    for row in shaft_table.itertuples():
        part = groups[row.subject][row.shaft]
        real = part[["x", "y", "z"]].to_numpy(float)
        real_labels = atlas.labels(real)
        xyz = real - real[0]
        direction = (real[-1] - real[0]) / np.linalg.norm(real[-1] - real[0])
        off_line = np.linalg.norm(xyz - np.outer(xyz @ direction, direction), axis=1)
        planned = {name: _plan(part, row.spacing_mm if value is None else value)
                   for name, value in variants.items()}
        planned_labels = {name: atlas.labels(points) for name, points in planned.items()}
        for i in range(len(part)):
            item = {"subject": row.subject, "shaft": row.shaft,
                    "contact": str(part["name"].iloc[i]), "index": i,
                    "from_tip_mm": float(np.linalg.norm(real[i] - real[0])),
                    "off_line_mm": float(off_line[i]),
                    "label": real_labels[i].label, "label_how": real_labels[i].how}
            for name, points in planned.items():
                item[f"error_{name}_mm"] = float(np.linalg.norm(points[i] - real[i]))
                mine = planned_labels[name][i]
                item[f"same_label_{name}"] = (mine.how, mine.label) == (
                    real_labels[i].how, real_labels[i].label)
            rows.append(item)
    contacts = pd.DataFrame(rows)
    shaft_table["max_off_line_mm"] = contacts.groupby(["subject", "shaft"])[
        "off_line_mm"].max().reindex(list(zip(shaft_table["subject"], shaft_table["shaft"],
                                              strict=True))).to_numpy()
    shaft_table.to_csv(out / "per_shaft.csv", index=False)
    contacts.to_csv(out / "per_contact.csv", index=False)
    summary = summarise(contacts, shaft_table, variants)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def summarise(contacts: pd.DataFrame, shaft_table: pd.DataFrame, variants: dict) -> dict:
    after_tip = contacts[contacts["index"] > 0]
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 80)]
    out = {"when": time.strftime("%Y-%m-%d"), "dataset": DATASET,
           "patients": int(contacts["subject"].nunique()), "shafts": int(len(shaft_table)),
           "contacts": int(len(contacts)),
           "spacing_mm": {"median": float(shaft_table["spacing_mm"].median()),
                          "p10": float(shaft_table["spacing_mm"].quantile(0.1)),
                          "p90": float(shaft_table["spacing_mm"].quantile(0.9))},
           "off_line_mm": {"median": float(contacts["off_line_mm"].median()),
                           "p90": float(contacts["off_line_mm"].quantile(0.9)),
                           "shaft_max_median": float(shaft_table["max_off_line_mm"].median())},
           "common_spacing_mm": variants["common"],
           "real_labels": {k: float(v) for k, v in
                           contacts["label_how"].value_counts(normalize=True).items()},
           "variants": {}}
    for name in variants:
        # A shaft with no two neighbouring contacts has no spacing of its own,
        # so no "own" plan: left out of that variant rather than counted a miss.
        defined = after_tip[after_tip[f"error_{name}_mm"].notna()]
        error = defined[f"error_{name}_mm"]
        by_distance = {}
        for lo, hi in bands:
            part = defined[(defined["from_tip_mm"] >= lo) & (defined["from_tip_mm"] < hi)]
            if len(part):
                by_distance[f"{lo}-{hi} mm"] = {
                    "n": int(len(part)),
                    "median": float(part[f"error_{name}_mm"].median()),
                    "p90": float(part[f"error_{name}_mm"].quantile(0.9))}
        out["variants"][name] = {
            "n": int(len(defined)),
            "median_mm": float(error.median()), "p90_mm": float(error.quantile(0.9)),
            "within_5mm": float((error <= 5).mean()),
            "same_label": float(defined[f"same_label_{name}"].mean()),
            "by_distance_from_tip": by_distance}
    return out


if __name__ == "__main__":      # pragma: no cover - a network run
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/template_ds004100")
    print(json.dumps(run_study(target), indent=1, default=str))
