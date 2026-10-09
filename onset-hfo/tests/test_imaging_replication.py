"""The registration replication on three more archives: its offline parts.

What has to hold: FreeSurfer's structure names, in either spelling, become the
atlas's words and nothing else does; an S3 path with a "+" is asked for as
written; and the summary counts the primary (hippocampus and amygdala) and
deep contacts per way, leaves out a patient whose frames did not match, and
names the patients a warp took off the brain.
"""

from __future__ import annotations

import pandas as pd

from onset_hfo import imaging_replication as rep


def test_freesurfer_names_become_the_atlas_words():
    assert rep.freesurfer_name("Left-Hippocampus") == "left hippocampus"
    assert rep.freesurfer_name("Right_Amygdala") == "right amygdala"
    assert rep.freesurfer_name("Left-Thalamus-Proper") == "left thalamus"
    assert rep.freesurfer_name("Right-Accumbens-area") == "right accumbens"
    for other in ("Right_Cerebral_White_Matter", "lh_G&S_cingul-Ant", "WM_hypointensities",
                  "Left_Lateral_Ventricle", "", None):
        assert rep.freesurfer_name(other) == ""
    assert set(rep.FREESURFER_DEEP.values()) <= {
        f"{side} {s}" for side in ("left", "right")
        for s in ("thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala",
                  "accumbens")}


def test_a_plus_in_an_s3_path_is_sent_encoded(monkeypatch):
    asked = []
    monkeypatch.setattr("onset_hfo.datasets._http_get", lambda url: asked.append(url) or b"")
    rep._get("ds004473/derivatives/sub-1/mri/aparc+aseg.mgz")
    assert asked[0].endswith("/mri/aparc%2Baseg.mgz")


def _contact(subject, truth, agrees, depth, excluded=False):
    row = {"dataset": "dsX", "subject": subject, "name": "c", "truth": truth,
           "excluded": excluded}
    for way in rep.WAYS:
        row[f"{way}_depth_mm"] = depth[way]
        if truth:
            row[f"{way}_agrees"] = agrees[way]
    return row


def test_the_summary_counts_each_way_and_leaves_out_a_mismatched_frame():
    good = {"registered": True, "warped": True, "blended": True, "identity": False}
    miss = {"registered": False, "warped": True, "blended": True, "identity": False}
    on = {w: 2.0 for w in rep.WAYS}
    off = {**on, "warped": -20.0}
    contacts = pd.DataFrame([
        _contact("sub-1", "left hippocampus", good, on),
        _contact("sub-1", "right amygdala", miss, on),
        _contact("sub-1", "left putamen", miss, on),
        _contact("sub-1", "", {}, off),
        _contact("sub-2", "left hippocampus", good, on, excluded=True)])
    patients = pd.DataFrame([
        {"dataset": "dsX", "subject": "sub-1", "status": "ok", "n_contacts": 4, "n_primary": 2,
         "agree_registered": 1, "agree_warped": 2, "agree_blended": 2, "agree_identity": 0,
         "correlation": 0.5, "correlation_warped": 0.7, "warp_shift_mm": 3.0,
         "near_brain_registered": 1.0, "near_brain_warped": 0.75, "near_brain_blended": 1.0,
         "near_brain_identity": 1.0},
        {"dataset": "dsX", "subject": "sub-2", "status": "frame mismatch", "n_contacts": 1}])
    summary = rep.summarise(patients, contacts)
    assert summary["excluded"] == [{"dataset": "dsX", "subject": "sub-2",
                                    "status": "frame mismatch"}]
    block = summary["all"]
    assert block["patients"] == 1 and block["contacts"] == 4
    assert block["primary"]["contacts"] == 2
    assert block["primary"]["registered"]["agree"] == 1
    assert block["primary"]["blended"]["agree"] == 2
    assert block["primary"]["blended"]["by_structure"] == {"amygdala": "1/1",
                                                          "hippocampus": "1/1"}
    assert block["deep"]["contacts"] == 3 and block["deep"]["warped"]["agree"] == 3
    assert block["near_brain_warped"] == 0.75 and block["near_brain_blended"] == 1.0
    assert block["patients_worse_warped"] == ["dsX/sub-1"]
    assert block["patients_worse_blended"] == []
    assert summary["dsX"]["primary"]["contacts"] == 2
    assert summary["low_check"] == [] and summary["primary_by_check"]["flagged"]["contacts"] == 0
    assert summary["primary_by_check"]["not_flagged"] == {
        "patients": 1, "contacts": 2, "registered": 1, "warped": 2, "blended": 2, "identity": 0}


def test_surface_ras_plus_the_centre_is_scanner_ras(tmp_path):
    import nibabel as nib
    import numpy as np

    # A conformed (LIA, 1 mm) volume with its centre away from the origin, as FreeSurfer has it.
    affine = np.array([[-1.0, 0, 0, 40.0], [0, 0, 1.0, -70.0], [0, -1.0, 0, 55.0], [0, 0, 0, 1]])
    image = nib.MGHImage(np.zeros((64, 64, 64), dtype=np.float32), affine)
    path = tmp_path / "t1.mgz"
    nib.save(image, str(path))
    voxel = np.array([10.0, 20.0, 30.0, 1.0])
    scanner = (image.affine @ voxel)[:3]
    surface = (image.header.get_vox2ras_tkr() @ voxel)[:3]
    assert np.allclose(surface + rep.surface_centre(path), scanner, atol=1e-4)
