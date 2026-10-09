"""The MRI-to-MNI registration on patients it was not tuned on: three more archives.

`onset_hfo.imaging_study` measured the registration on OpenNeuro ds003688 (UMC
Utrecht), and the warp's 10 mm fade (`imaging.WARP_FADE_MM`) was chosen on
those same patients. This study asks the same questions of three archives
from other hospitals, with nothing changed:

* **ds004473** (Rockhill et al., OHSU; 8 sEEG patients): each patient's T1,
  their contacts in its scanner space (``space-ScanRAS``, metres), and their
  own FreeSurfer segmentation (``aparc+aseg.mgz``). The structure that
  segmentation gives a contact is the truth.
* **ds004696** (Ojeda Valencia et al. 2023, Mayo Clinic; 8 sEEG patients): the
  contacts in the patient's own space, each with the label of the patient's
  own FreeSurfer segmentation (``Destrieux_label_text``), and the authors'
  positions in MNI152NLin6Sym. The archive gives the defaced T1
  (``*_T1w_deFaced.nii``), not the ``IntendedFor`` file, and five of the eight
  patients' positions are in an ACPC-aligned copy of it; the frame check below
  finds out whether they fit.
* **ds005574** ("Podcast" ECoG, Zada et al., NYU; 9 patients, surface grids):
  the contacts in the patient's own space and the authors' positions in
  MNI152NLin2009aSym.

Fixed before the run:

* **Primary, deep.** Of the contacts the patient's own segmentation puts in
  the hippocampus or amygdala, the share our atlas labels that structure on
  that side (in it or within its 5 mm "near") after the affine alone, the
  warp in full, and the warp faded in over 10 mm as the Map step applies it.
* **Primary, surface.** The share of all contacts within 5 mm of the
  template brain, and the patients whose share falls by more than 5 points
  from the affine's with each warp.
* **Secondary.** Every deep structure the atlas names (thalamus, caudate,
  putamen, pallidum, accumbens too); the distance to the authors' own MNI
  positions where there are some (those use another version of MNI152, so a
  few millimetres of it is the templates' difference, not an error).
* **A frame mismatch excludes a patient**: the affine positions a median of
  more than 15 mm from the authors' MNI positions, or (ds004473) fewer than
  half of the contacts inside the patient's own segmented brain.

Run ``python -m onset_hfo.imaging_replication [out]``; about 10-60 MB of MRI
per patient, downloaded and deleted.
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

from onset_hfo.imaging_study import NEAR_BRAIN_MM, _brain_depth, agrees

_BUCKET = "https://s3.amazonaws.com/openneuro.org"
WAYS = ("registered", "warped", "blended", "identity")
FRAME_MM = 15.0
#: FreeSurfer's colour table, for the deep structures the atlas also names.
FREESURFER_DEEP = {10: "left thalamus", 11: "left caudate", 12: "left putamen",
                   13: "left pallidum", 17: "left hippocampus", 18: "left amygdala",
                   26: "left accumbens", 49: "right thalamus", 50: "right caudate",
                   51: "right putamen", 52: "right pallidum", 53: "right hippocampus",
                   54: "right amygdala", 58: "right accumbens"}
PRIMARY = ("hippocampus", "amygdala")

__all__ = ["run_replication", "summarise", "freesurfer_name", "ARCHIVES"]


def _get(path: str) -> bytes:
    import urllib.parse

    from onset_hfo.datasets import _http_get

    # S3 reads a bare "+" in a path as a space (``aparc+aseg.mgz``).
    return _http_get(f"{_BUCKET}/{urllib.parse.quote(path)}")


def _table(path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get(path)), sep="\t", na_values=["n/a"])


def freesurfer_name(text) -> str:
    """A FreeSurfer structure name (``Left-Hippocampus``, ``Right_Thalamus_Proper``)
    in the atlas's words (``left hippocampus``), or "" for anything else."""
    words = re.split(r"[-_ ]+", str(text).strip().lower())
    if len(words) < 2 or words[0] not in ("left", "right"):
        return ""
    structure = {"thalamus": "thalamus", "caudate": "caudate", "putamen": "putamen",
                 "pallidum": "pallidum", "hippocampus": "hippocampus",
                 "amygdala": "amygdala", "accumbens": "accumbens"}.get(words[1], "")
    return f"{words[0]} {structure}" if structure else ""


# -- the three archives -----------------------------------------------------------------------
def _ds004473():
    ids = ["sub-1", "sub-2", "sub-5", "sub-6", "sub-9", "sub-10", "sub-11", "sub-12"]
    for subject in ids:
        def load(scratch, subject=subject):
            import nibabel as nib

            table = _table(f"ds004473/{subject}/ieeg/{subject}_space-ScanRAS_electrodes.tsv")
            table = table.dropna(subset=["x", "y", "z"])
            native = table[["x", "y", "z"]].to_numpy(float) * 1000.0
            path = scratch / "seg.mgz"
            path.write_bytes(_get(f"ds004473/derivatives/freesurfer-7.3.2/{subject}/mri/"
                                  "aparc+aseg.mgz"))
            seg = nib.load(str(path))
            values = np.asarray(seg.dataobj)
            ijk = np.rint(np.c_[native, np.ones(len(native))]
                          @ np.linalg.inv(seg.affine).T)[:, :3].astype(int)
            valid = np.all((ijk >= 0) & (ijk < np.array(values.shape)), axis=1)
            codes = np.zeros(len(native), dtype=int)
            codes[valid] = values[tuple(ijk[valid].T)]
            path.unlink()
            t1 = scratch / "t1.nii.gz"
            t1.write_bytes(_get(f"ds004473/{subject}/anat/{subject}_T1w.nii.gz"))
            return {"names": table["name"].astype(str).tolist(), "native": native,
                    "truth": [FREESURFER_DEEP.get(int(c), "") for c in codes],
                    "in_own_brain": float(np.mean(codes > 0)), "t1": t1, "author_mni": None}
        yield "ds004473", subject, load


def _ds004696():
    mni = None
    for n in range(1, 9):
        subject = f"sub-{n:02d}"

        def load(scratch, subject=subject, n=n):
            nonlocal mni
            table = _table(f"ds004696/{subject}/ses-ieeg01/ieeg/"
                           f"{subject}_ses-ieeg01_electrodes.tsv").dropna(subset=["x", "y", "z"])
            if mni is None:
                mni = _table("ds004696/derivatives/MNI/"
                             "sub-all_ses-ieeg01_space-MNI152NLin6Sym_electrodes.tsv")
            theirs = mni[mni["sub_code"].astype(int) == n].set_index("name")
            author = theirs.reindex(table["name"])[["x", "y", "z"]].to_numpy(float)
            t1 = scratch / "t1.nii"
            t1.write_bytes(_get(f"ds004696/derivatives/freesurfer/{subject}/"
                                f"{subject}_ses-mri01_T1w_deFaced.nii"))
            return {"names": table["name"].astype(str).tolist(),
                    "native": table[["x", "y", "z"]].to_numpy(float),
                    "truth": [freesurfer_name(t) for t in table["Destrieux_label_text"]],
                    "in_own_brain": np.nan, "t1": t1, "author_mni": author}
        yield "ds004696", subject, load


def _ds005574():
    for n in range(1, 10):
        subject = f"sub-{n:02d}"

        def load(scratch, subject=subject):
            native = _table(f"ds005574/{subject}/ieeg/{subject}_space-Other_electrodes.tsv")
            native = native.dropna(subset=["x", "y", "z"])
            theirs = _table(f"ds005574/{subject}/ieeg/"
                            f"{subject}_space-MNI152NLin2009aSym_electrodes.tsv").set_index("name")
            author = theirs.reindex(native["name"])[["x", "y", "z"]].to_numpy(float)
            t1 = scratch / "t1.nii.gz"
            t1.write_bytes(_get(f"ds005574/{subject}/anat/{subject}_T1w.nii.gz"))
            return {"names": native["name"].astype(str).tolist(),
                    "native": native[["x", "y", "z"]].to_numpy(float),
                    "truth": [""] * len(native), "in_own_brain": np.nan, "t1": t1,
                    "author_mni": author}
        yield "ds005574", subject, load


ARCHIVES = {"ds004473": _ds004473, "ds004696": _ds004696, "ds005574": _ds005574}


# -- the run ----------------------------------------------------------------------------------
def run_replication(out, archives=None, progress=print) -> dict:
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
    scratch = Path(tempfile.mkdtemp(prefix="onset-imaging-rep-"))
    patient_rows, contact_rows = [], []
    for archive in archives or ARCHIVES:
        for dataset, subject, load in ARCHIVES[archive]():
            started = time.monotonic()
            base = {"dataset": dataset, "subject": subject}
            try:
                found = load(scratch)
                image = nib.load(str(found["t1"]))
                to_mni, warp, check_warped = imaging.register_image_warped(image)
                check = imaging.registration_check(
                    imaging._as_float(image), imaging._as_float(imaging._template("head")),
                    np.linalg.inv(to_mni), mask)
                found["t1"].unlink()
            except Exception as error:      # noqa: BLE001 - a failure is a row
                patient_rows.append({**base, "status": f"{type(error).__name__}: {error}"[:200]})
                progress(f"{dataset} {subject}: {patient_rows[-1]['status']}")
                continue
            native = found["native"]
            affine = imaging.native_to_mni(native, to_mni)
            ways = {"registered": affine, "warped": warp.apply(affine),
                    "blended": imaging.blend_warp(affine, warp), "identity": native}
            depths = {way: _brain_depth(points, mask) for way, points in ways.items()}
            author = found["author_mni"]
            rows = []
            for i, name in enumerate(found["names"]):
                truth = found["truth"][i]
                row = {"dataset": dataset, "subject": subject, "name": name, "truth": truth}
                for way, points in ways.items():
                    row[f"{way}_depth_mm"] = round(float(depths[way][i]), 2)
                    if author is not None and np.isfinite(author[i]).all():
                        row[f"{way}_to_author_mm"] = round(
                            float(np.linalg.norm(points[i] - author[i])), 2)
                    if truth:
                        label = atlas.label(points[i])
                        row[f"{way}_label"] = label.label
                        row[f"{way}_how"] = label.how
                        row[f"{way}_agrees"] = agrees(label, truth)
                rows.append(row)
            frame = pd.DataFrame(rows)
            to_author = (float(frame["registered_to_author_mm"].median())
                         if "registered_to_author_mm" in frame else np.nan)
            mismatch = (to_author > FRAME_MM) or (found["in_own_brain"] < 0.5)
            contact_rows += [{**r, "excluded": bool(mismatch)} for r in rows]
            deep = frame[frame["truth"] != ""]
            primary = deep[deep["truth"].str.endswith(PRIMARY)] if len(deep) else deep
            patient_rows.append({
                **base, "status": "frame mismatch" if mismatch else "ok",
                "n_contacts": len(frame),
                "correlation": round(check["correlation"], 3),
                "correlation_warped": round(check_warped["correlation"], 3),
                "warp_shift_mm": round(float(np.median(np.linalg.norm(
                    ways["warped"] - affine, axis=1))), 2),
                "in_own_brain": found["in_own_brain"],
                "registered_to_author_mm": round(to_author, 2),
                **{f"{way}_to_author_mm": round(float(frame[f"{way}_to_author_mm"].median()), 2)
                   for way in WAYS if f"{way}_to_author_mm" in frame},
                **{f"near_brain_{way}": round(float(np.mean(depths[way] > -NEAR_BRAIN_MM)), 3)
                   for way in WAYS},
                "n_deep": len(deep), "n_primary": len(primary),
                **({f"agree_{way}": int(primary[f"{way}_agrees"].sum()) for way in WAYS}
                   if len(primary) else {}),
                "seconds": round(time.monotonic() - started, 1)})
            last = patient_rows[-1]
            progress(f"{dataset} {subject}: {last['status']}, correlation "
                     f"{last['correlation']:.2f}, to authors {to_author:.1f} mm, near brain "
                     + " / ".join(f"{last[f'near_brain_{w}']:.0%}" for w in WAYS)
                     + (", hippocampus/amygdala " + " / ".join(
                         str(last[f"agree_{w}"]) for w in WAYS) + f" of {len(primary)}"
                        if len(primary) else ""))
            pd.DataFrame(patient_rows).to_csv(out / "per_patient.csv", index=False)
            pd.DataFrame(contact_rows).to_csv(out / "per_contact.csv", index=False)
    patients = pd.DataFrame(patient_rows)
    contacts = pd.DataFrame(contact_rows)
    patients.to_csv(out / "per_patient.csv", index=False)
    contacts.to_csv(out / "per_contact.csv", index=False)
    summary = summarise(patients, contacts)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def _agreement(rows: pd.DataFrame) -> dict:
    out = {"contacts": int(len(rows)), "patients": int(rows["subject"].nunique())
           if len(rows) else 0}
    for way in WAYS:
        if not len(rows):
            continue
        agree = rows[f"{way}_agrees"].astype(bool)
        out[way] = {"agree": int(agree.sum()), "share": round(float(agree.mean()), 3),
                    "by_structure": {s: f"{int(g[f'{way}_agrees'].astype(bool).sum())}/{len(g)}"
                                     for s, g in rows.groupby(
                                         rows["truth"].str.split(" ").str[-1])}}
    return out


def summarise(patients: pd.DataFrame, contacts: pd.DataFrame) -> dict:
    from onset_hfo.case.imaging import WARP_FADE_MM

    out = {"when": time.strftime("%Y-%m-%d"), "fade_mm": WARP_FADE_MM,
           "excluded": patients.loc[patients["status"] != "ok",
                                    ["dataset", "subject", "status"]].to_dict("records")}
    ok = patients[patients["status"] == "ok"]
    used = contacts[~contacts["excluded"].astype(bool)] if len(contacts) else contacts
    for name, part in [("all", None), *[(d, d) for d in sorted(ok["dataset"].unique())]]:
        p = ok if part is None else ok[ok["dataset"] == part]
        c = used if part is None else used[used["dataset"] == part]
        block = {"patients": int(len(p)), "contacts": int(p["n_contacts"].sum()),
                 "correlation_median": float(p["correlation"].median()),
                 "correlation_warped_median": float(p["correlation_warped"].median()),
                 "warp_shift_mm_median": float(p["warp_shift_mm"].median())}
        for way in WAYS:
            block[f"near_brain_{way}"] = round(float(np.mean(
                c[f"{way}_depth_mm"] > -NEAR_BRAIN_MM)), 3)
            if way != "registered":
                drop = p[f"near_brain_{way}"] < p["near_brain_registered"] - 0.05
                block[f"patients_worse_{way}"] = p.loc[drop, "subject"].tolist()
            column = f"{way}_to_author_mm"
            if column in c and c[column].notna().any():
                block[f"to_author_mm_{way}"] = round(float(c[column].median()), 2)
        deep = c[c["truth"].fillna("") != ""]
        block["primary"] = _agreement(deep[deep["truth"].str.endswith(PRIMARY)])
        block["deep"] = _agreement(deep)
        out[name] = block
    return out


if __name__ == "__main__":      # pragma: no cover - a long network run
    import os
    import sys

    os.environ.pop("ONSET_HFO_OFFLINE", None)
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/imaging_replication")
    print(json.dumps(run_replication(target, sys.argv[2:] or None), indent=1, default=str))
