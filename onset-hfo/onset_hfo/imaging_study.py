"""Does registering the patient's MRI to MNI put their contacts in the right place?

The Map step carries contacts localised in a patient's own MRI to MNI with an
affine registration of their head (`onset_hfo.case.imaging`). This study asks
that of real patients: OpenNeuro ds003688 (Berezutskaya et al. 2022, UMC
Utrecht) gives each patient's T1-weighted MRI and their contacts in its space
(ACPC millimetres, ``IntendedFor`` that T1).

There are no MNI positions to compare with, so the check is anatomical:

* **Named medial temporal contacts.** In this archive's naming, a depth shaft
  called ``AR``/``AL`` is aimed at the amygdala and ``AHR``/``AHL``/``PHR``/
  ``PHL`` at the anterior or posterior hippocampus, on the side the last letter
  says (an assumption read from the names, not a field of the dataset). Their
  deepest two contacts should be labelled that structure on that side (in it,
  or within the atlas's 5 mm "near").
* **The brain.** The share of all contacts inside the template's brain or
  within 5 mm of it (surface grids sit on the cortex, a few millimetres out).
* **The registration's own check**: the intensity correlation with the
  template inside its brain.

Each is reported four ways: the affine registration ("registered"), the
affine and then the non-linear warp in full ("warped"), the warp faded in
from the template brain's edge as the Map step applies it ("blended",
`imaging.blend_warp`), and the shortcut of reading the patient's ACPC
millimetres as if they were MNI ("identity"). Run:
``python -m onset_hfo.imaging_study [out]`` (about 30 MB of MRI per patient,
downloaded and deleted; half a minute of registration each).
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "ds003688"
_BUCKET = "https://s3.amazonaws.com/openneuro.org"
#: Shaft name (without the contact number) -> the structure it is aimed at.
TARGETS = {"AR": "right amygdala", "AL": "left amygdala",
           "AHR": "right hippocampus", "AHL": "left hippocampus",
           "PHR": "right hippocampus", "PHL": "left hippocampus"}
NEAR_BRAIN_MM = 5.0

__all__ = ["run_study", "summarise", "target_of", "agrees", "TARGETS"]


def _get(path: str) -> bytes:
    from onset_hfo.datasets import _http_get

    return _http_get(f"{_BUCKET}/{DATASET}/{path}")


def _list(prefix: str = "") -> list[str]:
    import urllib.parse

    from onset_hfo.datasets import _http_get

    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": f"{DATASET}/{prefix}", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        text = _http_get(f"{_BUCKET}?{urllib.parse.urlencode(query)}").decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", text)
        found = re.search(r"<NextContinuationToken>([^<]*)<", text)
        if not found:
            return [k.split("/", 1)[1] for k in keys]
        token = found.group(1)


def target_of(name: str) -> tuple[str, int] | None:
    """The structure a contact's shaft is aimed at, and the contact's number."""
    found = re.fullmatch(r"([A-Za-z]+)(\d+)", str(name).strip())
    if not found or found.group(1).upper() not in TARGETS:
        return None
    return TARGETS[found.group(1).upper()], int(found.group(2))


def agrees(label, target: str) -> bool:
    """`label` (an `AtlasLabel`) names `target`, in it or near it."""
    return label.how in ("in", "near") and label.label == target


def _brain_depth(points: np.ndarray, mask_image) -> np.ndarray:
    """Millimetres inside (positive) or outside (negative) the template brain."""
    from scipy import ndimage

    mask = np.asarray(mask_image.get_fdata()) > 0.5
    zooms = np.asarray(mask_image.header.get_zooms()[:3], dtype=float)
    inside = ndimage.distance_transform_edt(mask, sampling=zooms)
    outside = ndimage.distance_transform_edt(~mask, sampling=zooms)
    voxels = np.rint(np.c_[points, np.ones(len(points))]
                     @ np.linalg.inv(mask_image.affine).T)[:, :3].astype(int)
    valid = np.all((voxels >= 0) & (voxels < np.array(mask.shape)), axis=1)
    depth = np.full(len(points), -np.inf)
    v = voxels[valid].T
    depth[valid] = np.where(mask[tuple(v)], inside[tuple(v)], -outside[tuple(v)])
    return depth


def _subjects() -> dict[str, dict]:
    keys = _list("")
    found: dict[str, dict] = {}
    for key in keys:
        subject = key.split("/")[0]
        if not subject.startswith("sub-"):
            continue
        entry = found.setdefault(subject, {"t1": None, "electrodes": []})
        if key.endswith("_T1w.nii.gz") and "/anat/" in key:
            entry["t1"] = entry["t1"] or key
        elif key.endswith("_electrodes.tsv"):
            entry["electrodes"].append(key)
    return {s: e for s, e in sorted(found.items()) if e["t1"] and e["electrodes"]}


def run_study(out, subjects=None, progress=print) -> dict:
    import nibabel as nib

    from onset_hfo.case import imaging
    from onset_hfo.case.atlas import Atlas, fetch_atlas

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not Atlas.available():
        fetch_atlas()
    if not imaging.template_available():
        imaging.fetch_template()
    atlas = Atlas.load()
    mask = imaging._template("brain_mask")
    listed = _subjects()
    if subjects:
        listed = {s: e for s, e in listed.items() if s in set(subjects)}
    patient_rows, contact_rows = [], []
    scratch = Path(tempfile.mkdtemp(prefix="onset-imaging-"))
    for subject, entry in listed.items():
        started = time.monotonic()
        base = {"subject": subject, "t1": Path(entry["t1"]).name}
        try:
            tables = []
            for key in entry["electrodes"]:
                table = pd.read_csv(io.BytesIO(_get(key)), sep="\t", na_values=["n/a"])
                table = table.dropna(subset=["x", "y", "z"])
                acq = re.search(r"acq-([a-zA-Z0-9]+)", key)
                tables.append(table.assign(acq=acq.group(1) if acq else ""))
            contacts = pd.concat(tables, ignore_index=True)
            path = scratch / "t1.nii.gz"
            path.write_bytes(_get(entry["t1"]))
            image = nib.load(str(path))
            to_mni, warp, check_warped = imaging.register_image_warped(image)
            check = imaging.registration_check(
                imaging._as_float(image), imaging._as_float(imaging._template("head")),
                np.linalg.inv(to_mni), mask)
            path.unlink()
        except Exception as error:      # noqa: BLE001 - a failure is a row
            patient_rows.append({**base, "status": f"{type(error).__name__}: {error}"[:200]})
            continue
        native = contacts[["x", "y", "z"]].to_numpy(float)
        affine = imaging.native_to_mni(native, to_mni)
        ways = {"registered": affine, "warped": warp.apply(affine),
                "blended": imaging.blend_warp(affine, warp), "identity": native}
        depths = {way: _brain_depth(points, mask) for way, points in ways.items()}
        rows = []
        for i, contact in enumerate(contacts.itertuples()):
            aimed = target_of(contact.name)
            row = {"subject": subject, "name": contact.name, "acq": contact.acq,
                   "target": aimed[0] if aimed else "", "number": aimed[1] if aimed else np.nan}
            for way, points in ways.items():
                row[f"{way}_x"], row[f"{way}_y"], row[f"{way}_z"] = np.round(points[i], 2)
                row[f"{way}_depth_mm"] = round(float(depths[way][i]), 2)
                if aimed:
                    label = atlas.label(points[i])
                    row[f"{way}_label"] = label.label
                    row[f"{way}_how"] = label.how
                    row[f"{way}_agrees"] = agrees(label, aimed[0])
            rows.append(row)
        contact_rows += rows
        frame = pd.DataFrame(rows)
        deepest = frame[frame["number"].isin([1, 2])] if "number" in frame else frame.iloc[:0]
        scale = np.linalg.svd(to_mni[:3, :3], compute_uv=False)
        patient_rows.append({
            **base, "status": "ok", "n_contacts": len(frame),
            "correlation": round(check["correlation"], 3),
            "correlation_warped": round(check_warped["correlation"], 3),
            "warp_shift_mm": round(float(np.median(np.linalg.norm(
                ways["warped"] - affine, axis=1))), 2),
            "brain_covered": round(check["brain_covered"], 3),
            "scale_min": round(float(scale.min()), 3), "scale_max": round(float(scale.max()), 3),
            **{f"near_brain_{way}": round(float(np.mean(depths[way] > -NEAR_BRAIN_MM)), 3)
               for way in ways},
            "n_targeted": len(deepest),
            **({f"agree_{way}": int(deepest[f"{way}_agrees"].sum()) for way in ways}
               if len(deepest) else {}),
            "seconds": round(time.monotonic() - started, 1)})
        progress(f"{subject}: correlation {check['correlation']:.2f}, near the brain "
                 f"{patient_rows[-1]['near_brain_registered']:.0%} registered vs "
                 f"{patient_rows[-1]['near_brain_warped']:.0%} warped vs "
                 f"{patient_rows[-1]['near_brain_blended']:.0%} blended vs "
                 f"{patient_rows[-1]['near_brain_identity']:.0%} as-is"
                 + (f", targeted {patient_rows[-1]['agree_registered']} affine, "
                    f"{patient_rows[-1]['agree_warped']} warped, "
                    f"{patient_rows[-1]['agree_blended']} blended, "
                    f"{patient_rows[-1]['agree_identity']} as-is of {len(deepest)}" if len(deepest)
                    else ""))
        pd.DataFrame(patient_rows).to_csv(out / "per_patient.csv", index=False)
        pd.DataFrame(contact_rows).to_csv(out / "per_contact.csv", index=False)
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(out / "per_patient.csv", index=False)
    pd.DataFrame(contact_rows).to_csv(out / "per_contact.csv", index=False)
    summary = summarise(patients, pd.DataFrame(contact_rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def summarise(patients: pd.DataFrame, contacts: pd.DataFrame) -> dict:
    from onset_hfo.case.imaging import REGISTRATION_PIPELINE, TEMPLATE_NAME

    ok = patients[patients["status"] == "ok"] if len(patients) else patients
    targeted = contacts[contacts["number"].isin([1, 2])] if len(contacts) else contacts
    out = {"dataset": DATASET, "when": time.strftime("%Y-%m-%d"),
           "template": TEMPLATE_NAME, "pipeline": list(REGISTRATION_PIPELINE),
           "patients": int(len(ok)), "patients_failed": int(len(patients) - len(ok)),
           "contacts": int(ok["n_contacts"].sum()) if len(ok) else 0}
    if len(ok):
        out["correlation"] = {"median": float(ok["correlation"].median()),
                              "min": float(ok["correlation"].min()),
                              "max": float(ok["correlation"].max())}
        out["scale"] = {"min": float(ok["scale_min"].min()), "max": float(ok["scale_max"].max())}
        out["correlation_warped"] = {"median": float(ok["correlation_warped"].median()),
                                     "min": float(ok["correlation_warped"].min()),
                                     "max": float(ok["correlation_warped"].max())}
        out["warp_shift_mm_median"] = float(ok["warp_shift_mm"].median())
        for way in ("registered", "warped", "blended", "identity"):
            column = contacts[f"{way}_depth_mm"]
            out[f"near_brain_{way}"] = {
                "contacts": float(np.mean(column > -NEAR_BRAIN_MM)),
                "patients_median": float(ok[f"near_brain_{way}"].median())}
    if len(targeted):
        out["targeted"] = {"patients": int(targeted["subject"].nunique()),
                           "contacts": int(len(targeted))}
        for way in ("registered", "warped", "blended", "identity"):
            agree = targeted[f"{way}_agrees"].astype(bool)
            out["targeted"][way] = {
                "agree": int(agree.sum()), "share": float(agree.mean()),
                "by_target": {t: f"{int(g[f'{way}_agrees'].astype(bool).sum())}/{len(g)}"
                              for t, g in targeted.groupby("target")},
                "labels": targeted[f"{way}_label"].value_counts().head(8).to_dict()}
    return out


if __name__ == "__main__":      # pragma: no cover - a long network run
    import os
    import sys

    os.environ.pop("ONSET_HFO_OFFLINE", None)
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/imaging_ds003688")
    print(json.dumps(run_study(target), indent=1, default=str))
