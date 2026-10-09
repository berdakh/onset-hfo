"""The patient's own imaging: contacts in their MRI, and from there to MNI. Qt-free.

Template positions (`onset_hfo.case.electrodes`) are what a case has without
imaging. When the patient's T1-weighted MRI is to hand, and the contacts have
been localised in it -- by the clinic's planning or localisation software
(ACPC or scanner coordinates, as BIDS ``space-T1w``/``space-ACPC``), or here
with MNE's contact locator on a CT aligned to the MRI -- the case can use
them:

* the T1 is copied into the case (``sub-*/ses-implant01/anat/``);
* it is **registered to the MNI152 template** with MNE's volume registration
  (translation, rigid, then affine; dipy), and the transform is kept
  (``derivatives/onset/imaging/t1_to_mni.json``), with the
  intensity correlation with the template inside its brain as a check;
* contacts in the patient's space are kept as they are
  (``*_space-T1w_electrodes.tsv``) **and** carried to MNI by that transform,
  so atlas labels, the combined map and the glass brain use them, marked
  ``source = patient``: the patient's anatomy, placed in MNI by an affine.

An affine registration aligns the head as a whole; it does not bend one
brain into another, so a contact near cortex can land a few millimetres off
the template's cortex. The label is still read as "probably". The contacts on
the patient's own MRI are not approximate, and `draw_on_mri` shows them there.

A CT with the contacts visible, without positions, can be localised with
MNE's iEEG locator (`mne_gui_addons.locate_ieeg`, the ``imaging`` extra):
`ct_to_t1` aligns the CT to the T1 (rigid) and resamples it into the T1's
grid, `open_locator` opens the locator on it, and `positions_from_locator`
turns what was placed back into the patient's scanner coordinates.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["t1_path", "add_t1", "register_t1", "register_image", "fetch_template",
           "template_available", "TEMPLATE_FILES", "load_registration", "native_to_mni",
           "import_native", "native_electrodes_path", "load_native", "draw_on_mri",
           "ct_to_t1", "ct_in_t1_path", "open_locator", "tkr_to_scanner",
           "positions_from_locator", "registration_check", "REGISTRATION_PIPELINE",
           "locator_available", "registration_check_saved"]

REGISTRATION_PIPELINE = ("translation", "rigid", "affine")
#: The template registration aligns to: the MNI152 head *with* its skull, as a
#: patient's T1 has one (a skull-stripped template shrinks the head onto the
#: brain), in the atlas's own space, and that template's brain mask.
TEMPLATE_NAME = "MNI152NLin2009cAsym T1w, 2 mm (TemplateFlow)"
TEMPLATE_FILES = {"head": "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz",
                  "brain_mask": "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"}
#: Voxel size, in mm, each registration step works at: coarse first.
ZOOMS = {"translation": 6.0, "rigid": 4.0, "affine": 3.0}


def _anat_folder(case, session: str = "implant01") -> Path:
    return Path(case.root) / f"sub-{case.subject}" / f"ses-{session}" / "anat"


def t1_path(case) -> Path | None:
    folder = _anat_folder(case)
    hits = sorted(folder.glob("*_T1w.nii*")) if folder.exists() else []
    return hits[0] if hits else None


def add_t1(case, path, by: str = "") -> Path:
    """Copy the patient's T1 into the case (BIDS ``anat``)."""
    source = Path(path)
    suffix = ".nii.gz" if source.name.endswith(".nii.gz") else ".nii"
    folder = _anat_folder(case)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"sub-{case.subject}_ses-implant01_T1w{suffix}"
    shutil.copyfile(source, target)
    case.record("added the patient's MRI", f"T1-weighted, {target.name}; faces are not "
                "removed -- the case folder must be kept as identifiable", by)
    return target


def template_dir() -> Path:
    from onset_hfo.case.atlas import atlas_dir

    return atlas_dir()


def template_available(folder=None) -> bool:
    folder = Path(folder or template_dir())
    return all((folder / name).exists() for name in TEMPLATE_FILES.values())


def fetch_template(folder=None, progress=None) -> Path:
    """Download the MNI152 head and its brain mask once (about 1.8 MB, from
    TemplateFlow, beside the atlas). A network call."""
    from onset_hfo.case.atlas import _BASE
    from onset_hfo.datasets import _http_get

    folder = Path(folder or template_dir())
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(TEMPLATE_FILES.values()):
        path = folder / name
        if not path.exists():
            partial = path.with_suffix(".part")
            partial.write_bytes(_http_get(_BASE + name))
            partial.replace(path)
        if progress is not None:
            progress((i + 1) / len(TEMPLATE_FILES))
    return folder


def _template(part: str = "head"):
    """The MNI152 head registration aligns a patient's head to, or its brain mask."""
    import nibabel as nib

    path = template_dir() / TEMPLATE_FILES[part]
    if not path.exists():
        raise FileNotFoundError("the MNI152 template is not here yet: fetch it first "
                                "(about 1.8 MB)")
    return nib.load(str(path))


def _as_float(image):
    import nibabel as nib

    data = np.asarray(image.get_fdata(), dtype=np.float32)
    return nib.Nifti1Image(data, image.affine)


def register_image(moving) -> tuple[np.ndarray, dict]:
    """Register a T1-weighted image (nibabel) to MNI152: the 4x4 that carries
    its scanner millimetres to MNI millimetres, and `registration_check`."""
    import mne

    moving = _as_float(moving)
    static = _as_float(_template("head"))
    reg, _ = mne.transforms.compute_volume_registration(
        moving, static, pipeline=REGISTRATION_PIPELINE, zooms=dict(ZOOMS), verbose=False)
    check = registration_check(moving, static, reg, _template("brain_mask"))
    return np.linalg.inv(np.asarray(reg, dtype=float)), check


def register_t1(case, progress=None, by: str = "") -> np.ndarray:
    """Register the case's T1 to MNI152 (affine); keep and return the 4x4
    that carries the patient's scanner millimetres to MNI millimetres."""
    import nibabel as nib

    path = t1_path(case)
    if path is None:
        raise FileNotFoundError("the case has no MRI: add the patient's T1 first")
    to_mni, check = register_image(nib.load(str(path)))
    folder = Path(case.derivatives) / "imaging"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "t1_to_mni.json").write_text(json.dumps({
        "patient_to_mni": to_mni.tolist(), "template": TEMPLATE_NAME,
        "pipeline": list(REGISTRATION_PIPELINE), "zooms_mm": ZOOMS, "t1": path.name,
        "check": check}, indent=1) + "\n")
    case.record("registered the MRI to MNI", f"affine; intensity correlation with the template "
                f"{check['correlation']:.2f} inside its brain, "
                f"{check['brain_covered']:.0%} of the brain covered", by)
    return to_mni


def registration_check(moving, static, reg, mask=None) -> dict:
    """How well the registered head matches the template: the correlation of
    their intensities inside the template's brain (`mask`, or the template's
    brightest voxels), and the share of that brain the head covers, after
    resampling into the template's grid. A T1 and the template correlate well
    when aligned and poorly when a step of the registration went wrong (a head
    that missed, scaled onto the wrong outline, a wrong axis); a value well
    below that of other patients means: look at it."""
    import mne

    moved = mne.transforms.apply_volume_registration(moving, static, reg, verbose=False)
    template = np.asarray(static.get_fdata(), dtype=float)
    head = np.asarray(moved.get_fdata(), dtype=float)
    if mask is not None:
        brain = np.asarray(mask.get_fdata()) > 0.5
    else:
        brain = template > 0.2 * np.percentile(template[template > 0], 98)
    inside = head[brain]
    covered = float(np.mean(inside > 0.1 * np.percentile(head[head > 0], 98))) \
        if (head > 0).any() else 0.0
    if np.std(inside) > 0:
        correlation = float(np.corrcoef(template[brain], inside)[0, 1])
    else:
        correlation = 0.0
    return {"correlation": correlation, "brain_covered": covered}


def registration_check_saved(case) -> dict:
    path = Path(case.derivatives) / "imaging" / "t1_to_mni.json"
    return json.loads(path.read_text()).get("check", {}) if path.exists() else {}


def load_registration(case) -> np.ndarray | None:
    path = Path(case.derivatives) / "imaging" / "t1_to_mni.json"
    if not path.exists():
        return None
    return np.asarray(json.loads(path.read_text())["patient_to_mni"], dtype=float)


def native_to_mni(points, to_mni: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (np.c_[points, np.ones(len(points))] @ np.asarray(to_mni).T)[:, :3]


# -- contacts in the patient's space ---------------------------------------------------------
def native_electrodes_path(case) -> Path:
    folder = Path(case.root) / f"sub-{case.subject}" / "ses-implant01" / "ieeg"
    return folder / f"sub-{case.subject}_ses-implant01_space-T1w_electrodes.tsv"


def load_native(case) -> pd.DataFrame:
    path = native_electrodes_path(case)
    if not path.exists():
        return pd.DataFrame(columns=["name", "x", "y", "z"])
    return pd.read_csv(path, sep="\t", na_values=["n/a"], dtype={"name": str})


def import_native(case, path, atlas=None, by: str = "", what: str = "") -> pd.DataFrame:
    """Contacts in the patient's own MRI space (millimetres, scanner/ACPC), from
    a file or a table with name, x, y, z: kept as they are, and carried to MNI
    by the case's registration, which must exist. Returns the MNI rows,
    `source = patient`."""
    from onset_hfo.case.electrodes import (
        add_labels,
        load_electrodes,
        merge,
        read_coordinate_file,
        save_electrodes,
    )

    to_mni = load_registration(case)
    if to_mni is None:
        raise ValueError("register the patient's MRI to MNI first")
    if isinstance(path, pd.DataFrame):
        native = path.copy()
        native["name"] = native["name"].astype(str).str.strip()
        what = what or f"{len(native)} contact(s)"
    else:
        native = read_coordinate_file(path, space="MNI152")  # read as-is, no conversion
        what = what or f"{len(native)} contact(s) from {Path(path).name}"
    keep = native[["name", "x", "y", "z"]]
    target = native_electrodes_path(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(target, sep="\t", index=False, float_format="%.2f")
    (target.with_name(target.name.replace("_electrodes.tsv", "_coordsystem.json"))
     .write_text(json.dumps({"iEEGCoordinateSystem": "Other", "iEEGCoordinateUnits": "mm",
                             "iEEGCoordinateSystemDescription":
                                 "the patient's own T1-weighted MRI (scanner/ACPC mm)",
                             "IntendedFor": str(t1_path(case).relative_to(case.root))
                             if t1_path(case) else ""}, indent=1) + "\n"))
    from onset_hfo.case.electrodes import ELECTRODE_COLUMNS, _group_of

    mni = native.copy()
    mni[["x", "y", "z"]] = native_to_mni(native[["x", "y", "z"]].to_numpy(float), to_mni)
    for column in ELECTRODE_COLUMNS:
        if column not in mni:
            mni[column] = np.nan if column in ("size", "label_mm") else ""
    mni["group"] = [_group_of(n) for n in mni["name"]]
    mni["source"] = "patient"
    mni["space_from"] = "T1w (registered, affine)"
    mni = mni[list(ELECTRODE_COLUMNS)]
    if atlas is not None:
        mni = add_labels(mni, atlas)
    merged = merge(load_electrodes(case), mni)
    save_electrodes(case, merged, "imported contact positions",
                    f"{what} in the patient's MRI, registered to MNI", by)
    return mni


def draw_on_mri(figure, case, contacts: list[str] | None = None, centre: str | None = None):
    """The patient's own T1 in three planes through `centre` (a contact),
    with the contacts drawn where they are. Returns nilearn's display."""
    import nibabel as nib
    from nilearn import plotting

    from onset_review.studycharts import SERIES

    path = t1_path(case)
    native = load_native(case)
    if path is None or not len(native):
        return None
    if contacts:
        native = native[native["name"].str.upper().isin({c.upper() for c in contacts})]
    image = nib.load(str(path))
    cut = None
    if centre is not None:
        row = native[native["name"].str.upper() == centre.upper()]
        if len(row):
            cut = row[["x", "y", "z"]].to_numpy(float)[0]
    display = plotting.plot_anat(image, figure=figure, cut_coords=cut, draw_cross=False,
                                 annotate=True, dim=-0.3, colorbar=False)
    display.add_markers(native[["x", "y", "z"]].to_numpy(float), marker_color=SERIES[1],
                        marker_size=12)
    return display


# -- CT localisation with MNE's locator -------------------------------------------------------
def locator_available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) is not None
               for name in ("mne_gui_addons", "pyvistaqt"))


def ct_in_t1_path(case) -> Path:
    return Path(case.derivatives) / "imaging" / "ct_in_t1.nii.gz"


def ct_to_t1(case, ct_path, by: str = ""):
    """The patient's CT, rigidly registered to their T1 and resampled into its
    grid -- what the locator shows to place contacts on. Kept in the case's
    derivatives and returned."""
    import mne
    import nibabel as nib

    t1 = t1_path(case)
    if t1 is None:
        raise FileNotFoundError("the case has no MRI: add the patient's T1 first")
    ct = _as_float(nib.load(str(ct_path)))
    t1_image = _as_float(nib.load(str(t1)))
    reg, _ = mne.transforms.compute_volume_registration(
        ct, t1_image, pipeline="rigid", zooms=dict(rigid=4.0), verbose=False)
    aligned = mne.transforms.apply_volume_registration(ct, t1_image, reg, verbose=False)
    target = ct_in_t1_path(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    nib.save(aligned, str(target))
    case.record("aligned the CT to the MRI", f"rigid, {Path(ct_path).name} -> {target.name}",
                by)
    return aligned


def open_locator(base_image, channels, show: bool = True):
    """MNE's iEEG contact locator on `base_image` (the CT in the T1's grid),
    for the contacts named in `channels`. Contacts placed there are written
    into the returned `info` as they are marked; `positions_from_locator`
    reads them back. Needs the ``imaging`` extra (mne-gui-addons, pyvistaqt)."""
    import mne
    from mne_gui_addons import locate_ieeg

    info = mne.create_info([str(c) for c in channels], 1000.0, "seeg")
    head_is_mri = mne.transforms.Transform("head", "mri")       # identity
    gui = locate_ieeg(info, head_is_mri, base_image, show=show)
    _camera_shim(gui)
    return gui, info


def _camera_shim(gui) -> None:
    """mne-gui-addons 0.1 (its only release) asks MNE's 3D renderer for
    ``set_camera(reset_camera=False)``, which MNE 1.13 no longer takes, and
    fails whenever it moves to a contact already placed. Drop the argument
    when the renderer does not know it; nothing else changes."""
    import inspect

    renderer = getattr(gui, "_renderer", None)
    original = getattr(renderer, "set_camera", None)
    if original is None or "reset_camera" in inspect.signature(original).parameters:
        return

    def set_camera(*args, reset_camera=None, **kwargs):
        return original(*args, **kwargs)

    renderer.set_camera = set_camera


def tkr_to_scanner(points_m, image) -> np.ndarray:
    """MNE's locator gives positions in the image's surface RAS ("tkr",
    metres); the patient's scanner coordinates (mm) are what everything else
    here uses. The surface RAS is the one MNE derives, through nibabel's
    MGH header, from the image's shape and affine."""
    from nibabel.freesurfer.mghformat import MGHImage

    shape = tuple(int(n) for n in image.shape[:3])
    header = MGHImage(np.broadcast_to(np.float32(0), shape), np.asarray(image.affine)).header
    to_scanner = header.get_vox2ras() @ np.linalg.inv(header.get_vox2ras_tkr())
    points = np.atleast_2d(np.asarray(points_m, dtype=float)) * 1000.0
    return (np.c_[points, np.ones(len(points))] @ to_scanner.T)[:, :3]


def positions_from_locator(info, t1_image) -> pd.DataFrame:
    """The contacts placed in the locator (on an `mne.Info` montage, surface
    RAS in metres, the head frame being the MRI's) as patient scanner
    millimetres, ready for `import_native`. Contacts not placed are left out."""
    montage = info.get_montage()
    if montage is None:
        return pd.DataFrame(columns=["name", "x", "y", "z"])
    positions = montage.get_positions()["ch_pos"]
    names = [n for n, p in positions.items() if np.all(np.isfinite(p))]
    if not names:
        return pd.DataFrame(columns=["name", "x", "y", "z"])
    scanner = tkr_to_scanner(np.array([positions[n] for n in names]), t1_image)
    return pd.DataFrame({"name": names, "x": scanner[:, 0], "y": scanner[:, 1],
                         "z": scanner[:, 2]})
