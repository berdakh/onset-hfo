"""The Map step: contacts on the template, atlas labels, and the combined map.

What has to hold: a NIfTI atlas is read without nibabel and a position gets
the structure it is in, the nearest one within 5 mm (with the distance), or
says white matter / outside; fsaverage positions move to MNI152 and back;
coordinate files in metres or millimetres, with BIDS "n/a" rows, read the
same; a planned shaft puts contact 1 at the target and the rest along the
line at the spacing; a grid is square; the case keeps its contacts as
BIDS-iEEG with a coordinate system that says they are template positions;
the clinician's zone is stored and logged; a bipolar channel sits at its
pair's midpoint; the combined table puts both analyses, place and zone on
one row; the planner study's arithmetic is right on a shaft whose answer is
known; and the window does it all.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from onset_hfo.case import atlas as atlas_module
from onset_hfo.case import electrodes as el
from onset_hfo.case.atlas import Atlas, fsaverage_to_mni152, mni152_to_fsaverage, read_nifti
from onset_hfo.case.model import Case

VOXEL = 2.0
ORIGIN = (-40.0, -40.0, -40.0)
SHAPE = (41, 41, 41)


def _write_nifti(path, data: np.ndarray, voxel=VOXEL, origin=ORIGIN):
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *data.shape, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 4)          # int16
    struct.pack_into("<h", header, 72, 16)
    struct.pack_into("<8f", header, 76, 1.0, voxel, voxel, voxel, 1, 1, 1, 1)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<2f", header, 112, 1.0, 0.0)
    struct.pack_into("<2h", header, 252, 0, 1)     # qform 0, sform 1
    rows = [(voxel, 0, 0, origin[0]), (0, voxel, 0, origin[1]), (0, 0, voxel, origin[2])]
    for i, row in enumerate(rows):
        struct.pack_into("<4f", header, 280 + 16 * i, *row)
    header[344:348] = b"n+1\0"
    blob = bytes(header) + b"\0" * 4 + data.astype("<i2").T.tobytes()
    path.write_bytes(gzip.compress(blob))


def _ijk(point):
    return tuple(int(round((p - o) / VOXEL)) for p, o in zip(point, ORIGIN, strict=True))


@pytest.fixture(scope="module")
def atlas_folder(tmp_path_factory):
    """A brain-shaped ball of left white matter with a left hippocampus block
    (subcortical 9) at (-20, -20, -10) and a left precentral gyrus block
    (cortical 13) at (-30, 0, 20)."""
    folder = tmp_path_factory.mktemp("atlas")
    grid = np.stack(np.meshgrid(*[np.arange(n) * VOXEL + o for n, o in zip(SHAPE, ORIGIN,
                                                                             strict=True)],
                                indexing="ij"), axis=-1)
    inside = np.linalg.norm(grid, axis=-1) <= 36
    sub = np.where(inside, 1, 0)
    hippo = np.all(np.abs(grid - np.array([-20, -20, -10])) <= 4, axis=-1)
    sub[hippo] = 9
    cortex = np.zeros(SHAPE, dtype=int)
    gyrus = np.all(np.abs(grid - np.array([-30, 0, 20])) <= 4, axis=-1)
    cortex[gyrus] = 13
    _write_nifti(folder / atlas_module.ATLAS_FILES["subcortical"], sub)
    _write_nifti(folder / atlas_module.ATLAS_FILES["cortical"], cortex)
    return folder


@pytest.fixture(scope="module")
def atlas(atlas_folder):
    return Atlas.load(atlas_folder)


def test_a_nifti_is_read_with_its_affine(atlas_folder):
    data, affine = read_nifti(atlas_folder / atlas_module.ATLAS_FILES["subcortical"])
    assert data.shape == SHAPE
    assert np.allclose(affine[:3, 3], ORIGIN) and np.allclose(np.diag(affine)[:3], VOXEL)
    assert data[_ijk((-20, -20, -10))] == 9 and data[_ijk((0, 0, 0))] == 1
    assert data[_ijk((-40, -40, -40))] == 0


def test_a_position_gets_the_structure_it_is_in_or_near(atlas):
    assert atlas.label((-20, -20, -10)).text() == "probably left hippocampus"
    assert atlas.label((-30, 0, 20)).label == "left precentral gyrus"
    near = atlas.label((-20, -20, -2))            # 4 mm above the top voxel, at -6
    assert near.how == "near" and near.label == "left hippocampus"
    assert near.distance_mm == pytest.approx(4.0, abs=0.01)
    assert "near left hippocampus (4 mm)" in near.text()
    assert atlas.label((10, 10, 0)).how == "white matter"
    assert atlas.label((0, 0, 39)).how == "outside the brain"
    assert atlas.label((500, 0, 0)).how == "outside the brain"
    assert np.allclose(atlas.centroid("left hippocampus"), (-20, -20, -10))
    with pytest.raises(KeyError):
        atlas.centroid("left nowhere")
    a, b, image = atlas.silhouette(2)
    assert image.shape == (SHAPE[0], SHAPE[1]) and a[0] == ORIGIN[0] and b[-1] == 40.0


def test_fsaverage_and_mni152_round_trip():
    points = np.array([[-26.0, -20.0, -14.0], [40.0, 10.0, 50.0]])
    moved = fsaverage_to_mni152(points)
    assert np.all(np.abs(moved - points) < 3.0), "a few millimetres apart"
    assert np.allclose(mni152_to_fsaverage(moved), points)


def test_coordinate_files_in_metres_or_mm_read_the_same(tmp_path):
    mm = tmp_path / "mm.tsv"
    mm.write_text("name\tx\ty\tz\tsize\nLA1\t-20\t-20\t-10\tn/a\nLA2\t-25\t-20\t-10\tn/a\n"
                  "EKG1\tn/a\tn/a\tn/a\tn/a\n")
    metres = tmp_path / "m.csv"
    metres.write_text("label,x,y,z\nLA1,-0.020,-0.020,-0.010\nLA2,-0.025,-0.020,-0.010\n")
    a = el.read_coordinate_file(mm)
    b = el.read_coordinate_file(metres)
    assert list(a["name"]) == ["LA1", "LA2"] and list(a["group"]) == ["LA", "LA"]
    assert np.allclose(a[["x", "y", "z"]], b[["x", "y", "z"]])
    assert set(a["source"]) == {"imported"}
    fs = el.read_coordinate_file(mm, space="fsaverage")
    assert np.allclose(fs[["x", "y", "z"]], fsaverage_to_mni152(a[["x", "y", "z"]]))
    bad = tmp_path / "bad.tsv"
    bad.write_text("name\tx\ty\nA1\t1\t2\n")
    with pytest.raises(ValueError, match="x, y, z"):
        el.read_coordinate_file(bad)


def test_a_planned_shaft_and_grid_have_the_right_geometry():
    shaft = el.plan_depth("LA", (-20, -20, -10), (-70, -20, -10), 6, 3.5)
    xyz = shaft[["x", "y", "z"]].to_numpy()
    assert list(shaft["name"]) == [f"LA{i}" for i in range(1, 7)]
    assert np.allclose(xyz[0], (-20, -20, -10))
    assert np.allclose(np.linalg.norm(np.diff(xyz, axis=0), axis=1), 3.5)
    assert np.allclose(xyz[:, 1:], [-20, -10]) and xyz[-1, 0] == pytest.approx(-37.5)
    grid = el.plan_sheet("G", (0, 0, 60), (10, 0, 60), (3, 10, 62), 2, 3, 10.0)
    points = grid[["x", "y", "z"]].to_numpy()
    assert list(grid["name"]) == [f"G{i}" for i in range(1, 7)]
    assert np.allclose(points[1] - points[0], (10, 0, 0))
    row = points[3] - points[0]
    assert np.linalg.norm(row) == pytest.approx(10.0) and row @ (points[1] - points[0]) == \
        pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        el.plan_depth("X", (0, 0, 0), (0, 0, 0), 4, 3.5)


def test_the_case_keeps_its_contacts_and_zone(tmp_path, atlas):
    case = Case.create(tmp_path / "case", "P50")
    planned = el.add_labels(el.plan_depth("LA", (-20, -20, -10), (-60, -20, -10), 4, 5.0), atlas)
    el.save_electrodes(case, planned, "planned an electrode on the template")
    path = el.electrodes_path(case)
    assert path.name == "sub-P50_ses-implant01_space-MNI152NLin2009cAsym_electrodes.tsv"
    coords = json.loads(path.with_name(path.name.replace("electrodes.tsv",
                                                         "coordsystem.json")).read_text())
    assert coords["iEEGCoordinateSystem"] == "MNI152NLin2009cAsym"
    assert "not this patient's anatomy" in coords["iEEGCoordinateProcessingDescription"]
    back = el.load_electrodes(case)
    assert list(back["name"]) == ["LA1", "LA2", "LA3", "LA4"]
    assert back.loc[0, "label"] == "left hippocampus" and back.loc[0, "label_how"] == "in"
    moved = el.plan_depth("LA", (-21, -20, -10), (-60, -20, -10), 2, 5.0)
    merged = el.merge(back, moved)
    assert len(merged) == 4 and merged.set_index("name").at["LA1", "x"] == -21
    case.set_zone(["la1", "LA2 "], by="dr test")
    again = Case.open(case.root)
    assert again.zone("soz") == ["LA1", "LA2"] and again.zones["soz"]["by"] == "dr test"
    assert again.log[-1]["action"] == "set the soz contacts"
    where = el.channel_positions(back, ["LA1-LA2", "LA3", "LB1-LB2"])
    assert np.allclose(where.loc[0, ["x", "y", "z"]].astype(float), (-22.5, -20, -10))
    assert list(where["placed"]) == [True, True, False]


def _results(case):
    inter = SimpleNamespace(table=pd.DataFrame({
        "channel": ["LA1-LA2", "LA2-LA3", "LB1-LB2"], "rate_per_min": [9.0, 4.0, 1.0],
        "rate_ci_low": [7.0, 3.0, 0.5], "rate_ci_high": [11.0, 5.0, 2.0],
        "tied": [True, False, False], "segments_tied": [3, 1, 0],
        "segments_analysed": [3, 3, 3]}))
    ictal = SimpleNamespace(combined=pd.DataFrame({
        "channel": ["LA1-LA2", "LB1-LB2", "LC1-LC2"], "median_ei": [1.0, 0.2, 0.0],
        "seizures_high": [2, 0, 0], "seizures": [2, 2, 2]}),
        consistent=lambda: ["LA1-LA2"])
    return inter, ictal


def test_the_combined_table_puts_both_analyses_place_and_zone_together(tmp_path, monkeypatch,
                                                                        atlas):
    from onset_review import caseictal, caseinterictal, casemap

    case = Case.create(tmp_path / "case", "P51")
    el.save_electrodes(case, el.add_labels(
        el.plan_depth("LA", (-20, -20, -10), (-60, -20, -10), 4, 5.0), atlas), "planned")
    case.set_zone(["LA1", "LA2"])
    inter, ictal = _results(case)
    monkeypatch.setattr(caseinterictal, "latest_result", lambda c: inter)
    monkeypatch.setattr(caseictal, "latest_ictal", lambda c: ictal)
    table = casemap.combined_table(case)
    assert list(table["channel"])[0] == "LA1-LA2"
    assert set(table["channel"]) == {"LA1-LA2", "LA2-LA3", "LB1-LB2", "LC1-LC2"}
    row = table.set_index("channel").loc["LA1-LA2"]
    assert row["rate_per_min"] == 9.0 and row["median_ei"] == 1.0 and row["soz"]
    assert row["placed"] and row["where"].startswith("left hippocampus")
    assert table.set_index("channel").at["LA2-LA3", "soz"], "either contact in the zone"
    assert not table.set_index("channel").at["LB1-LB2", "placed"]
    assert pd.isna(table.set_index("channel").at["LC1-LC2", "rate_per_min"])
    found = casemap.agreement(table, ictal.consistent())
    assert found["measures"]["interictal"]["top"] == "LA1-LA2"
    assert found["measures"]["interictal"]["top_in_soz"]
    assert found["measures"]["interictal"]["auc"] == pytest.approx(1.0)
    assert found["measures"]["ictal"]["auc"] == pytest.approx(1.0)
    assert found["both"] == ["LA1-LA2"]
    text = casemap.statement(table, found)
    assert "Highest interictal rate: LA1-LA2 (in the marked onset zone)" in text
    assert "Singled out by both: LA1-LA2." in text and "2 of 4 channels placed" in text


def test_the_planner_study_arithmetic_on_a_known_shaft(atlas):
    from onset_hfo import template_study as ts

    # A straight shaft at 5 mm, and one bent 3 mm sideways at its far end.
    straight = pd.DataFrame({"name": [f"A{i}" for i in range(1, 7)],
                             "x": [-20 - 5 * i for i in range(6)], "y": -20.0, "z": -10.0})
    bent = straight.assign(name=[f"B{i}" for i in range(1, 7)],
                           y=[-20, -20, -20, -20, -20, -17.0])
    both = pd.concat([straight, bent, pd.DataFrame({"name": ["C1", "C2"], "x": [0, 1],
                                                    "y": 0, "z": 0})], ignore_index=True)
    found = ts.shafts(both)
    assert set(found) == {"A", "B"}, "two contacts are too few for a shaft"
    assert ts._spacing(found["A"]) == pytest.approx(5.0)
    assert np.allclose(ts._plan(found["A"], 5.0), found["A"][["x", "y", "z"]])
    plan = ts._plan(found["A"], 3.5)
    assert np.linalg.norm(plan[-1] - found["A"][["x", "y", "z"]].to_numpy()[-1]) == \
        pytest.approx(7.5)


@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_the_window_places_labels_zones_and_maps(qapp, tmp_path, monkeypatch, atlas_folder):
    from qtpy.QtCore import Qt

    from onset_review import caseictal, caseinterictal
    from onset_review.casewindow import CaseWindow, PlanDialog

    monkeypatch.setenv("ONSET_ATLAS_DIR", str(atlas_folder))
    case = Case.create(tmp_path / "case", "P52")
    inter, ictal = _results(case)
    monkeypatch.setattr(caseinterictal, "latest_result", lambda c: inter)
    monkeypatch.setattr(caseictal, "latest_ictal", lambda c: ictal)
    window = CaseWindow(case, reader="dr test")
    try:
        assert window.show_step("map")
        page = window.map_page
        assert page.atlas is not None and not page.atlas_button.isVisible()
        coords = tmp_path / "plan.tsv"
        coords.write_text("name\tx\ty\tz\nLB1\t-30\t0\t20\nLB2\t-35\t0\t20\n")
        page.import_file(coords)
        dialog = PlanDialog(page.atlas, page)
        dialog.group.setText("la")
        choose = dialog.findChildren(type(dialog.kind))[1]
        choose.setCurrentIndex(choose.findData("left hippocampus"))
        assert [round(v) for v in dialog.point("first")] == [-20, -20, -10]
        for box, value in zip(dialog.points["second"], (-60, -20, -10), strict=True):
            box.setValue(value)
        dialog.count.setValue(4)
        page.add_planned(dialog.plan())
        names = [page.contacts.item(r, 0).text() for r in range(page.contacts.rowCount())]
        assert names == ["LB1", "LB2", "LA1", "LA2", "LA3", "LA4"]
        assert page.contacts.item(2, 4).text() == "left hippocampus"
        assert page.contacts.item(0, 4).text() == "left precentral gyrus"
        for r in (2, 3):
            page.contacts.item(r, 1).setCheckState(Qt.Checked)
        assert page.save_zone() == ["LA1", "LA2"]
        assert case.zone("soz") == ["LA1", "LA2"]
        assert "Highest interictal rate: LA1-LA2 (in the marked onset zone)" in \
            page.statement.text()
        assert page.combined.rowCount() == 4
        page.contacts.selectRow(1)
        assert page.remove_selected() == ["LB2"]
        page.view.setCurrentIndex(1)
        page.done_button.click()
        assert case.step_done("map")
        actions = [entry["action"] for entry in case.log]
        assert "imported contact positions" in actions
        assert "planned an electrode on the template" in actions
        assert "removed contact positions" in actions
    finally:
        window.close()
