"""The patient's own imaging: their MRI registered to MNI, their contacts carried there.

What has to hold: a head that is the template moved by a known rotation and
shift registers back to the template, so a point in the patient's scanner
space lands within a voxel or two of where it is in MNI, and the overlap
check says the registration is good; one 25 mm off scores far lower; contacts in the patient's space are kept as they are
(BIDS ``space-T1w``) and enter the case's MNI table marked as the patient's;
importing without a registration is refused; and the surface-RAS positions
MNE's locator gives are turned back into scanner millimetres exactly as MNE
derives them, whatever the image's orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")
pytest.importorskip("dipy")

from onset_hfo.case import imaging  # noqa: E402
from onset_hfo.case.electrodes import load_electrodes  # noqa: E402
from onset_hfo.case.model import Case  # noqa: E402

#: The patient's head is the template, turned 8 degrees about z and shifted.
ANGLE = np.deg2rad(8.0)
MOVE = np.array([[np.cos(ANGLE), -np.sin(ANGLE), 0, 9.0],
                 [np.sin(ANGLE), np.cos(ANGLE), 0, -6.0],
                 [0, 0, 1, 5.0],
                 [0, 0, 0, 1]])
#: MNI points well inside the brain: hippocampus, insula, motor cortex, thalamus.
MNI_POINTS = np.array([[-28.0, -22, -14], [38, 4, 2], [-36, -20, 56], [10, -18, 6]])


@pytest.fixture(scope="module", autouse=True)
def offline_template(tmp_path_factory):
    """The template the registration aligns to, without the download: nilearn's
    bundled MNI152 and its brain mask, under the fetched files' names."""
    from nilearn import datasets

    folder = tmp_path_factory.mktemp("template")
    nib.save(datasets.load_mni152_template(resolution=2),
             str(folder / imaging.TEMPLATE_FILES["head"]))
    nib.save(datasets.load_mni152_brain_mask(resolution=2),
             str(folder / imaging.TEMPLATE_FILES["brain_mask"]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ONSET_ATLAS_DIR", str(folder))
        assert imaging.template_available()
        yield folder


def _patient_head(tmp_path, flip=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    template = imaging._template()
    data = np.asarray(template.get_fdata(), dtype=np.float32)
    if flip:
        data = data[::-1].copy()
    image = nib.Nifti1Image(data, MOVE @ template.affine)
    path = tmp_path / "t1.nii.gz"
    nib.save(image, path)
    return path


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("imaging")
    case = Case.create(tmp_path / "case", "P70")
    imaging.add_t1(case, _patient_head(tmp_path), by="tester")
    to_mni = imaging.register_t1(case, by="tester")
    return case, to_mni


def test_a_moved_head_registers_back_to_the_template(registered):
    case, to_mni = registered
    patient_points = imaging.native_to_mni(MNI_POINTS, MOVE)   # MNI -> patient
    back = imaging.native_to_mni(patient_points, to_mni)
    assert np.linalg.norm(back - MNI_POINTS, axis=1).max() < 3.0, "within a voxel or two"
    assert np.allclose(imaging.load_registration(case), to_mni)
    check = (case.derivatives / "imaging" / "t1_to_mni.json").read_text()
    assert '"correlation"' in check and "affine" in check
    assert imaging.t1_path(case).name == "sub-P70_ses-implant01_T1w.nii.gz"
    entries = [e["action"] for e in case.log]
    assert "added the patient's MRI" in entries and "registered the MRI to MNI" in entries


def test_the_check_tells_a_good_registration_from_a_wrong_one(registered, tmp_path):
    case, _ = registered
    import json

    good = json.loads((case.derivatives / "imaging" / "t1_to_mni.json").read_text())["check"]
    assert good["correlation"] > 0.9 and good["brain_covered"] > 0.95
    template = imaging._as_float(imaging._template())
    head = imaging._as_float(nib.load(str(imaging.t1_path(case))))
    missed = MOVE.copy()
    missed[:3, 3] += [0.0, 20.0, -15.0]                         # 25 mm off
    wrong = imaging.registration_check(head, template, missed)  # reg is the inverse of to_mni
    assert wrong["correlation"] < good["correlation"] - 0.3


def test_contacts_in_the_patients_space_are_kept_and_carried_to_mni(registered, tmp_path):
    case, to_mni = registered
    native = imaging.native_to_mni(MNI_POINTS, MOVE)
    table = pd.DataFrame({"name": ["LH1", "RI1", "LM1", "RT1"], "x": native[:, 0],
                          "y": native[:, 1], "z": native[:, 2]})
    path = tmp_path / "contacts.tsv"
    table.to_csv(path, sep="\t", index=False)
    mni = imaging.import_native(case, path, by="tester")
    kept = imaging.load_native(case)
    assert np.allclose(kept[["x", "y", "z"]].to_numpy(float), native, atol=0.01)
    assert imaging.native_electrodes_path(case).name.endswith("_space-T1w_electrodes.tsv")
    assert np.linalg.norm(mni[["x", "y", "z"]].to_numpy(float) - MNI_POINTS, axis=1).max() < 3.0
    stored = load_electrodes(case)
    assert set(stored["source"]) == {"patient"} and len(stored) == 4
    coordsystem = imaging.native_electrodes_path(case).with_name(
        "sub-P70_ses-implant01_space-T1w_coordsystem.json").read_text()
    assert "anat/sub-P70_ses-implant01_T1w.nii.gz" in coordsystem


def test_importing_without_a_registration_is_refused(tmp_path):
    case = Case.create(tmp_path / "case", "P71")
    path = tmp_path / "c.tsv"
    path.write_text("name\tx\ty\tz\nA1\t1\t2\t3\n")
    with pytest.raises(ValueError, match="register"):
        imaging.import_native(case, path)
    with pytest.raises(FileNotFoundError, match="no MRI"):
        imaging.register_t1(case)


@pytest.mark.parametrize("axes", [((0, -1.0), (2, 1.0), (1, -1.0)),     # LIA, FreeSurfer's
                                  ((0, 1.0), (1, 1.0), (2, 1.0)),       # RAS
                                  ((0, -1.0), (1, -1.0), (2, 1.0))])    # LPS
def test_locator_positions_become_scanner_millimetres(axes):
    from nibabel.freesurfer.mghformat import MGHImage

    shape = (90, 100, 80)
    affine = np.zeros((4, 4))
    for column, (row, sign) in enumerate(axes):
        affine[row, column] = sign * (1.0 + 0.1 * column)
    affine[:3, 3] = [12.0, -30.0, 25.0]
    affine[3, 3] = 1
    image = nib.Nifti1Image(np.zeros(shape, np.float32), affine)
    header = MGHImage(np.zeros(shape, np.float32), affine).header
    voxels = np.array([[10, 20, 30, 1], [45, 50, 40, 1], [80, 5, 70, 1]], dtype=float)
    tkr_m = (voxels @ header.get_vox2ras_tkr().T)[:, :3] / 1000.0
    scanner = (voxels @ affine.T)[:, :3]
    assert np.allclose(imaging.tkr_to_scanner(tkr_m, image), scanner, atol=1e-6)


def test_a_contact_placed_in_mnes_locator_comes_back_where_it_is():
    """The whole CT path through the real locator: needs the ``imaging`` extra
    and a display with OpenGL (CI has neither; it is run under Xvfb)."""
    pytest.importorskip("mne_gui_addons")
    pytest.importorskip("pyvistaqt")
    import os

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("the locator's 3D view needs OpenGL; run under Xvfb")
    from qtpy.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    shape = (64, 64, 64)
    ct = np.zeros(shape, np.float32)
    ct[16:48, 16:48, 16:48] = 100.0
    bright = {"LA1": (30, 32, 32), "LA2": (36, 32, 32)}
    for voxel in bright.values():
        ct[voxel] = 3000.0            # one voxel: the locator's snap lands on it exactly
    affine = np.array([[-2.0, 0, 0, 60], [0, 0, 2, -70], [0, -2, 0, 60], [0, 0, 0, 1]])
    image = nib.Nifti1Image(ct, affine)
    gui, info = imaging.open_locator(image, ["LA1", "LA2", "LA3"], show=False)
    try:
        for name, voxel in bright.items():
            scanner = affine @ np.r_[voxel, 1.0]
            tkr = gui._vox_ras_t @ np.linalg.inv(gui._vox_scan_ras_t) @ scanner
            gui._set_ras(tkr[:3])
            gui.mark_channel(name)
        gui._go_to_ch(gui._ch_list_model.index(0, 0))   # back to a placed contact
    finally:
        gui.close()
    found = imaging.positions_from_locator(info, image).set_index("name")
    assert list(found.index) == ["LA1", "LA2"], "LA3 was not placed"
    for name, voxel in bright.items():
        expected = (affine @ np.r_[voxel, 1.0])[:3]
        assert np.linalg.norm(found.loc[name, ["x", "y", "z"]].to_numpy(float) - expected) < 0.01


@pytest.fixture
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_the_map_step_takes_the_patients_mri_and_contacts(qapp, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from onset_hfo.case.electrodes import PATIENT_NOTE, TEMPLATE_NOTE
    from onset_review.casewindow import CaseWindow

    case = Case.create(tmp_path / "case", "P72")
    window = CaseWindow(case, reader="dr test")
    try:
        assert window.show_step("map")
        page = window.map_page
        assert "No MRI" in page.imaging_status.text()
        assert not page.native_button.isEnabled() and not page.on_mri_button.isEnabled()
        assert page.note.text() == TEMPLATE_NOTE
        page.add_mri(_patient_head(tmp_path / "head"), wait=True)
        assert "registered to MNI" in page.imaging_status.text()
        assert page.native_button.isEnabled()
        native = imaging.native_to_mni(MNI_POINTS, MOVE)
        table = pd.DataFrame({"name": ["LH1", "RI1", "LM1", "RT1"], "x": native[:, 0],
                              "y": native[:, 1], "z": native[:, 2]})
        page.import_native(table, what="4 test contacts")
        assert page.note.text() == PATIENT_NOTE
        assert "4 contact(s) in the MRI's space" in page.imaging_status.text()
        assert page.on_mri_button.isEnabled()
        sources = {page.contacts.item(r, 3).text() for r in range(page.contacts.rowCount())}
        assert sources == {"patient"}
        assert window.map_page._positions == "patient"
        page.contacts.selectRow(2)
        dialog = page.show_on_mri(show=False)
        assert dialog.centre.currentText() == "LM1" and dialog.figure.axes
        dialog.close()
        told = []
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda *a, **k: told.append(a[2])))
        monkeypatch.setattr(imaging, "locator_available", lambda: False)
        assert page.ct_dialog() is None and "imaging extra" in told[-1]
    finally:
        window.close()
