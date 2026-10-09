"""Which structure a template position probably lies in. Qt-free, no nibabel.

Without the patient's own CT and MRI a contact's position is known only on a
template brain, and only as well as it was placed there (imported from a
planning system, or planned on the template; `onset_hfo.case.electrodes`).
This module gives such a position an **atlas label**: the Harvard-Oxford
cortical (lateralised) and subcortical atlases at a 25% probability
threshold, in MNI152NLin2009cAsym space at 1 mm, fetched once from
TemplateFlow (about 0.6 MB) and kept.

A label is a *probable* structure, never a fact, and the label says how it
was found:

* **in** -- the position is inside a deep structure (hippocampus, amygdala,
  thalamus, ...) or a cortical region;
* **near** -- it is in white matter or between labels, and the nearest
  labelled structure is within `NEAR_MM`; the distance is given;
* **white matter** / **outside the brain** -- neither.

Positions in fsaverage (MNI305) space are moved to MNI152 by FreeSurfer's
published linear transform first; the residual between MNI152 variants is a
few millimetres, inside what a template position can claim anyway.
"""

from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Atlas", "read_nifti", "fsaverage_to_mni152", "mni152_to_fsaverage", "atlas_dir",
           "fetch_atlas", "ATLAS_FILES", "ATLAS_NAME", "ATLAS_SPACE", "NEAR_MM",
           "CORTICAL_REGIONS", "SUBCORTICAL_LABELS", "AtlasLabel"]

ATLAS_SPACE = "MNI152NLin2009cAsym"
ATLAS_NAME = "Harvard-Oxford (cortical, lateralised; subcortical), 25% threshold"
_BASE = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/"
ATLAS_FILES = {
    "cortical": "tpl-MNI152NLin2009cAsym_res-01_atlas-HOCPAL_desc-th25_dseg.nii.gz",
    "subcortical": "tpl-MNI152NLin2009cAsym_res-01_atlas-HOSPA_desc-th25_dseg.nii.gz",
}
#: How far a position may be from a labelled structure and still be "near" it.
NEAR_MM = 5.0

#: Harvard-Oxford cortical regions in FSL's order. In the lateralised atlas,
#: value 2k-1 is region k on the left and 2k on the right.
CORTICAL_REGIONS = (
    "frontal pole", "insular cortex", "superior frontal gyrus", "middle frontal gyrus",
    "inferior frontal gyrus, pars triangularis", "inferior frontal gyrus, pars opercularis",
    "precentral gyrus", "temporal pole", "superior temporal gyrus, anterior",
    "superior temporal gyrus, posterior", "middle temporal gyrus, anterior",
    "middle temporal gyrus, posterior", "middle temporal gyrus, temporo-occipital",
    "inferior temporal gyrus, anterior", "inferior temporal gyrus, posterior",
    "inferior temporal gyrus, temporo-occipital", "postcentral gyrus",
    "superior parietal lobule", "supramarginal gyrus, anterior",
    "supramarginal gyrus, posterior", "angular gyrus", "lateral occipital cortex, superior",
    "lateral occipital cortex, inferior", "intracalcarine cortex", "frontal medial cortex",
    "supplementary motor cortex", "subcallosal cortex", "paracingulate gyrus",
    "cingulate gyrus, anterior", "cingulate gyrus, posterior", "precuneus",
    "cuneus", "frontal orbital cortex", "parahippocampal gyrus, anterior",
    "parahippocampal gyrus, posterior", "lingual gyrus", "temporal fusiform cortex, anterior",
    "temporal fusiform cortex, posterior", "temporo-occipital fusiform cortex",
    "occipital fusiform gyrus", "frontal operculum", "central operculum",
    "parietal operculum", "planum polare", "Heschl's gyrus", "planum temporale",
    "supracalcarine cortex", "occipital pole",
)
#: Harvard-Oxford subcortical values -> (hemisphere, structure, kind). Kind
#: "deep" is a structure worth naming; "cortex", "white" and "ventricle" are
#: the atlas's coarse classes, used only when nothing finer applies.
SUBCORTICAL_LABELS = {
    1: ("left", "cerebral white matter", "white"), 2: ("left", "cerebral cortex", "cortex"),
    3: ("left", "lateral ventricle", "ventricle"), 4: ("left", "thalamus", "deep"),
    5: ("left", "caudate", "deep"), 6: ("left", "putamen", "deep"),
    7: ("left", "pallidum", "deep"), 8: ("", "brainstem", "deep"),
    9: ("left", "hippocampus", "deep"), 10: ("left", "amygdala", "deep"),
    11: ("left", "accumbens", "deep"), 12: ("right", "cerebral white matter", "white"),
    13: ("right", "cerebral cortex", "cortex"), 14: ("right", "lateral ventricle", "ventricle"),
    15: ("right", "thalamus", "deep"), 16: ("right", "caudate", "deep"),
    17: ("right", "putamen", "deep"), 18: ("right", "pallidum", "deep"),
    19: ("right", "hippocampus", "deep"), 20: ("right", "amygdala", "deep"),
    21: ("right", "accumbens", "deep"),
}

#: FreeSurfer's MNI305 (fsaverage) -> MNI152 linear transform, millimetres
#: (FreeSurfer wiki, "CoordinateSystems").
_MNI305_TO_152 = np.array([[1.0022, 0.0071, -0.0177, 0.0528],
                           [-0.0146, 0.9990, 0.0187, -1.5457],
                           [0.0129, 0.0076, 1.0027, -1.2913],
                           [0.0, 0.0, 0.0, 1.0]])


def fsaverage_to_mni152(points) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (np.c_[points, np.ones(len(points))] @ _MNI305_TO_152.T)[:, :3]


def mni152_to_fsaverage(points) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return (np.c_[points, np.ones(len(points))] @ np.linalg.inv(_MNI305_TO_152).T)[:, :3]


_NIFTI_TYPES = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2",
                768: "u4"}


def read_nifti(path) -> tuple[np.ndarray, np.ndarray]:
    """A NIfTI-1 volume (gzipped or not): (data indexed [i, j, k], the 4x4
    voxel-to-world matrix from its sform, or its pixel sizes when it has none).
    Enough for an atlas; not a general reader."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if struct.unpack("<i", raw[:4])[0] == 348:
        e = "<"
    elif struct.unpack(">i", raw[:4])[0] == 348:
        e = ">"
    else:
        raise ValueError(f"{path.name} is not a NIfTI-1 file")
    dim = struct.unpack(e + "8h", raw[40:56])
    code = struct.unpack(e + "h", raw[70:72])[0]
    if code not in _NIFTI_TYPES:
        raise ValueError(f"{path.name}: data type {code} is not supported")
    offset = int(struct.unpack(e + "f", raw[108:112])[0])
    slope, inter = struct.unpack(e + "2f", raw[112:120])
    sform_code = struct.unpack(e + "h", raw[254:256])[0]
    shape = tuple(int(n) for n in dim[1:4])
    dtype = np.dtype(e + _NIFTI_TYPES[code])
    count = int(np.prod(shape))
    data = np.frombuffer(raw[offset:offset + count * dtype.itemsize], dtype=dtype)
    data = data.reshape(shape[::-1]).T
    if slope not in (0.0, 1.0) or inter != 0.0:
        data = data * slope + inter
    if sform_code > 0:
        affine = np.vstack([np.array(struct.unpack(e + "12f", raw[280:328])).reshape(3, 4),
                            [0, 0, 0, 1]])
    else:
        pix = struct.unpack(e + "8f", raw[76:108])
        affine = np.diag([pix[1], pix[2], pix[3], 1.0])
    return np.ascontiguousarray(data), affine


def atlas_dir() -> Path:
    named = os.environ.get("ONSET_ATLAS_DIR")
    if named:
        return Path(named).expanduser()
    return Path.home() / ".local" / "share" / "onset-review" / "atlas"


def fetch_atlas(folder=None, progress=None) -> Path:
    """Download the two atlas volumes once (about 0.6 MB). A network call."""
    from onset_hfo.datasets import _http_get

    folder = Path(folder or atlas_dir())
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(ATLAS_FILES.values()):
        path = folder / name
        if not path.exists():
            blob = _http_get(_BASE + name)
            path.write_bytes(blob)
        if progress is not None:
            progress((i + 1) / len(ATLAS_FILES))
    return folder


@dataclass(frozen=True)
class AtlasLabel:
    label: str          # e.g. "left hippocampus"
    how: str            # "in", "near", "white matter", "outside the brain"
    distance_mm: float  # 0 when in; to the nearest labelled voxel when near

    def text(self) -> str:
        if self.how == "in":
            return f"probably {self.label}"
        if self.how == "near":
            return f"probably near {self.label} ({self.distance_mm:.0f} mm)"
        return self.how


class Atlas:
    """The two volumes, and a label for any MNI152 position."""

    def __init__(self, cortical: np.ndarray, subcortical: np.ndarray, affine: np.ndarray):
        if cortical.shape != subcortical.shape:
            raise ValueError("the two atlas volumes differ in shape")
        self.cortical = np.asarray(cortical).astype(np.int16)
        self.subcortical = np.asarray(subcortical).astype(np.int16)
        self.affine = np.asarray(affine, dtype=float)
        self._inverse = np.linalg.inv(self.affine)
        self.voxel_mm = float(np.abs(np.linalg.det(self.affine[:3, :3])) ** (1 / 3))
        deep = np.isin(self.subcortical, [k for k, v in SUBCORTICAL_LABELS.items()
                                          if v[2] == "deep"])
        self._named = (self.cortical > 0) | deep
        self._brain = (self.subcortical > 0) | (self.cortical > 0)

    @classmethod
    def load(cls, folder=None) -> Atlas:
        folder = Path(folder or atlas_dir())
        paths = {k: folder / v for k, v in ATLAS_FILES.items()}
        missing = [p.name for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "The atlas is not on this machine (" + ", ".join(missing) + "). Fetch it "
                "once with onset_hfo.case.atlas.fetch_atlas(), or the button on the Map step.")
        cortical, affine = read_nifti(paths["cortical"])
        subcortical, _ = read_nifti(paths["subcortical"])
        return cls(cortical, subcortical, affine)

    @staticmethod
    def available(folder=None) -> bool:
        folder = Path(folder or atlas_dir())
        return all((folder / v).exists() for v in ATLAS_FILES.values())

    # -- labels ----------------------------------------------------------------------------------
    def _voxel(self, point) -> tuple[int, int, int] | None:
        ijk = np.round(self._inverse @ np.r_[np.asarray(point, dtype=float), 1.0])[:3]
        ijk = ijk.astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(self.cortical.shape)):
            return None
        return tuple(int(v) for v in ijk)

    def _name(self, ijk) -> str:
        sub = int(self.subcortical[ijk])
        if sub in SUBCORTICAL_LABELS and SUBCORTICAL_LABELS[sub][2] == "deep":
            side, structure, _kind = SUBCORTICAL_LABELS[sub]
            return f"{side} {structure}".strip()
        value = int(self.cortical[ijk])
        if value > 0:
            region = CORTICAL_REGIONS[(value - 1) // 2]
            side = "left" if (value - 1) % 2 == 0 else "right"
            return f"{side} {region}"
        return ""

    def label(self, point_mni152) -> AtlasLabel:
        ijk = self._voxel(point_mni152)
        if ijk is None:
            return AtlasLabel("", "outside the brain", float("nan"))
        name = self._name(ijk)
        if name:
            return AtlasLabel(name, "in", 0.0)
        reach = int(np.ceil(NEAR_MM / self.voxel_mm))
        lo = [max(0, c - reach) for c in ijk]
        hi = [min(n, c + reach + 1) for c, n in zip(ijk, self.cortical.shape, strict=True)]
        block = self._named[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if block.any():
            hits = np.argwhere(block) + np.array(lo)
            world = (np.c_[hits, np.ones(len(hits))] @ self.affine.T)[:, :3]
            gaps = np.linalg.norm(world - np.asarray(point_mni152, dtype=float), axis=1)
            best = int(np.argmin(gaps))
            if gaps[best] <= NEAR_MM:
                return AtlasLabel(self._name(tuple(hits[best])), "near", float(gaps[best]))
        sub = int(self.subcortical[ijk])
        if sub in SUBCORTICAL_LABELS and SUBCORTICAL_LABELS[sub][2] in ("white", "ventricle"):
            side, structure, _ = SUBCORTICAL_LABELS[sub]
            return AtlasLabel(f"{side} {structure}", "white matter" if
                              SUBCORTICAL_LABELS[sub][2] == "white" else "ventricle",
                              float("nan"))
        if self._brain[ijk]:
            return AtlasLabel("", "white matter", float("nan"))
        return AtlasLabel("", "outside the brain", float("nan"))

    def labels(self, points_mni152) -> list[AtlasLabel]:
        return [self.label(p) for p in np.atleast_2d(np.asarray(points_mni152, dtype=float))]

    def regions(self) -> list[str]:
        """Every structure the atlas names, for the planner's target list."""
        names = [f"{side} {r}" for r in CORTICAL_REGIONS for side in ("left", "right")]
        names += [f"{s} {n}".strip() for s, n, k in SUBCORTICAL_LABELS.values() if k == "deep"]
        return sorted(set(names))

    def centroid(self, name: str) -> np.ndarray:
        """The centre of a named structure, in MNI152 mm (a planner's target)."""
        side, _, structure = name.partition(" ")
        mask = None
        for value, (s, n, k) in SUBCORTICAL_LABELS.items():
            if k == "deep" and f"{s} {n}".strip() == name:
                mask = self.subcortical == value
        if mask is None and structure in CORTICAL_REGIONS:
            k = CORTICAL_REGIONS.index(structure)
            mask = self.cortical == (2 * k + 1 + (side == "right"))
        if mask is None or not mask.any():
            raise KeyError(f"the atlas has no structure called {name!r}")
        ijk = np.argwhere(mask).mean(axis=0)
        return (self.affine @ np.r_[ijk, 1.0])[:3]

    def silhouette(self, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The brain's outline seen along `axis` (0: from the side, 2: from
        above): the two world coordinates of the grid and a 0/1 image, for
        drawing contacts over."""
        shown = self._brain.any(axis=axis).astype(float)
        keep = [a for a in range(3) if a != axis]
        coords = []
        for a in keep:
            n = self._brain.shape[a]
            index = np.zeros((n, 4))
            index[:, a] = np.arange(n)
            index[:, 3] = 1
            coords.append((index @ self.affine.T)[:, a])
        return coords[0], coords[1], shown
