"""Where the contacts are drawn, and the claim that drawing makes.

This file is mostly about one thing. Neither archive this project reads ships
electrode coordinates, so the 3D view's positions are inferred from electrode
names -- and a picture of contacts on a brain is read as an implantation plan
unless it works hard not to be. The tests that matter here are the ones that
pin the *provenance* of every coordinate and the words that go with it, not the
ones that check arithmetic.

Qt-free throughout: `anatomy` imports no Qt, which is what lets the claim be
tested without a display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_review.anatomy import (
    REGIONS,
    SHAFT_PITCH_M,
    brain_surface,
    electrode_layout,
    layout_caption,
    parse_channel,
)
from onset_review.session import ReviewRequest, session_from_recording


@pytest.fixture(scope="module")
def review(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


# -- reading an electrode name ---------------------------------------------

@pytest.mark.parametrize("name,shaft,index,hemisphere", [
    ("AR1", "AR", 1, "R"),
    ("AHR12", "AHR", 12, "R"),
    ("PHL3", "PHL", 3, "L"),
    ("IAR5", "IAR", 5, "R"),
    ("G8", "G", 8, "?"),
])
def test_channel_names_split_the_way_the_site_writes_them(name, shaft, index,
                                                          hemisphere):
    got_shaft, got_index, got_hemisphere, _ = parse_channel(name)
    assert (got_shaft, got_index, got_hemisphere) == (shaft, index, hemisphere)


def test_an_unrecognised_electrode_is_placed_rather_than_dropped():
    """Refusing to place one would quietly remove it from a view of all of them."""
    shaft, index, _, region = parse_channel("ZZQ4")
    assert (shaft, index) == ("ZZQ", 4)
    assert region == "unmapped"


def test_every_region_entry_has_a_label_and_a_target():
    for stem, entry in REGIONS.items():
        assert entry["label"], stem
        assert len(entry["target"]) == 3, stem
        assert all(abs(float(v)) < 0.12 for v in entry["target"]), stem


# -- the layout ------------------------------------------------------------

def test_the_layout_has_one_row_per_channel(review):
    layout = electrode_layout(review)
    assert sorted(layout["channel"]) == sorted(review.findings["channel"])


def test_the_layout_is_in_rank_order(review):
    layout = electrode_layout(review)
    assert list(layout["rank"]) == sorted(layout["rank"])


def test_contacts_along_a_shaft_are_spaced_in_order(review):
    """Contact numbering is spatially ordered; the drawing has to be too."""
    layout = electrode_layout(review)
    for _, rows in layout.groupby("shaft"):
        if len(rows) < 2 or set(rows["source"]) != {"inferred"}:
            continue
        rows = rows.sort_values("index")
        steps = np.diff(np.abs(rows["x"].to_numpy()))
        gaps = np.diff(rows["index"].to_numpy())
        assert np.allclose(steps, gaps * SHAFT_PITCH_M, atol=1e-9)


def test_left_and_right_shafts_land_on_opposite_sides():
    """The hemisphere letter is the only side information these names carry."""
    from onset_review.anatomy import _inferred_position

    right = _inferred_position("AHR", 1)
    left = _inferred_position("AHL", 1)
    assert right[0] > 0 > left[0]
    assert right[0] == pytest.approx(-left[0])


def test_a_bipolar_channel_is_drawn_between_its_two_contacts(review):
    """A pair localises to between its contacts, not to one of them."""
    from onset_review.anatomy import _inferred_position

    layout = electrode_layout(review).set_index("channel")
    for channel, row in layout.iterrows():
        if "-" not in channel or row["source"] != "inferred":
            continue
        a, b = channel.split("-")[:2]
        pa = _inferred_position(*parse_channel(a)[:2])
        pb = _inferred_position(*parse_channel(b)[:2])
        assert row["x"] == pytest.approx((pa[0] + pb[0]) / 2)
        break
    else:
        pytest.skip("no bipolar inferred channel in this recording")


# -- provenance, which is the whole point ----------------------------------

def test_inferred_positions_are_labelled_inferred(review):
    """The synthetic recording has no electrodes.tsv, like both real archives."""
    layout = electrode_layout(review)
    assert set(layout["source"]) == {"inferred"}


def test_the_caption_refuses_to_call_a_schematic_anatomy(review):
    caption = layout_caption(electrode_layout(review))
    assert "SCHEMATIC" in caption
    assert "not this patient's anatomy" in caption


def test_measured_coordinates_are_used_and_declared_when_present(review):
    """The seam a site with its own BIDS data comes in through.

    No dataset here has an `electrodes.tsv`, so without this test the path
    would never run and would rot. Coordinates are given in millimetres, as
    BIDS writes them, to exercise the unit conversion too.
    """
    channels = list(review.findings["channel"])
    contacts = sorted({c for channel in channels for c in channel.split("-")})
    electrodes = pd.DataFrame({
        "name": contacts,
        "x": np.linspace(-40.0, 40.0, len(contacts)),     # millimetres
        "y": np.zeros(len(contacts)),
        "z": np.full(len(contacts), 10.0),
    })
    layout = electrode_layout(review, electrodes=electrodes)
    assert set(layout["source"]) == {"archive"}
    assert np.allclose(layout["z"], 0.010)                # mm -> m
    assert np.abs(layout["x"]).max() <= 0.040
    caption = layout_caption(layout)
    assert "measured coordinates" in caption.lower()
    assert "SCHEMATIC" not in caption


def test_a_mixed_layout_is_reported_as_partly_schematic(review):
    """A midpoint is only as trustworthy as its worse end."""
    channels = list(review.findings["channel"])
    one = channels[0].split("-")[0]
    electrodes = pd.DataFrame({"name": [one], "x": [10.0], "y": [0.0], "z": [0.0]})
    layout = electrode_layout(review, electrodes=electrodes)
    assert set(layout["source"]) == {"inferred"}
    caption = layout_caption(layout)
    assert "SCHEMATIC" in caption


def test_the_resection_is_carried_when_the_dataset_has_one(review):
    """And the zone is `unknown`, not `spared`, when it does not."""
    layout = electrode_layout(review, resection=None)
    assert set(layout["zone"]) == {"unknown"}


def test_an_unparseable_resection_does_not_take_the_view_down(review):
    class Broken:
        @property
        def resected(self):
            raise RuntimeError("clinical sheet unreadable")

    layout = electrode_layout(review, resection=Broken())
    assert len(layout) == len(review.findings)
    assert set(layout["zone"]) == {"unknown"}


# -- the reference surface -------------------------------------------------

def test_the_head_surface_is_head_shaped():
    vertices, faces = brain_surface()
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    assert faces.max() < len(vertices)
    span = vertices.max(axis=0) - vertices.min(axis=0)
    # Longer front-to-back than side-to-side, and shorter top-to-bottom than
    # either: that is what makes the orientation readable without axis labels.
    assert span[1] > span[0] > span[2]
    assert 0.10 < span[0] < 0.20


def test_the_layout_is_empty_rather_than_failing_with_no_findings(review):
    import dataclasses

    empty = dataclasses.replace(review, findings=pd.DataFrame())
    layout = electrode_layout(empty)
    assert layout.empty
    assert "No channels" in layout_caption(layout)
