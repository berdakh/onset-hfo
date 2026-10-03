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
    """Contact numbering is spatially ordered; the drawing has to be too.

    Evenly spaced rather than spaced at the nominal 5 mm: a long shaft is
    scaled down to fit inside the head, so the pitch is a property of the shaft
    and only the evenness is a property of every shaft.
    """
    layout = electrode_layout(review)
    checked = 0
    for _, rows in layout.groupby("shaft"):
        if len(rows) < 3 or set(rows["source"]) != {"inferred"}:
            continue
        rows = rows.sort_values("index")
        points = rows[["x", "y", "z"]].to_numpy()
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.diff(
            rows["index"].to_numpy())
        # Within a few percent of each other, not identical: the outermost
        # contact of a shaft aimed at the skull gets pulled back inside the
        # reference head, which shortens its last step by a fraction of a
        # millimetre. Order and near-even spacing are what the drawing
        # promises; exact pitch is not, and is not known for these electrodes.
        assert np.allclose(steps, np.median(steps), rtol=0.05)
        assert steps.min() > 0
        # And the contacts march away from the first one rather than doubling
        # back, which is the property a reviewer actually reads off the shaft.
        outward = np.linalg.norm(points - points[0], axis=1)
        assert np.all(np.diff(outward) > 0)
        checked += 1
    assert checked, "no multi-contact shaft to check"


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
    """Whichever of the two captions applies, it denies the same thing."""
    from onset_review.anatomy import NOT_ANATOMY

    caption = layout_caption(electrode_layout(review))
    assert NOT_ANATOMY in caption
    assert caption.split("—")[0].strip() in ("SCHEMATIC LAYOUT", "MONTAGE DIAGRAM")


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
    from onset_review.anatomy import NOT_ANATOMY

    layout = electrode_layout(review, electrodes=electrodes)
    assert set(layout["source"]) == {"inferred"}
    assert NOT_ANATOMY in layout_caption(layout)


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


# -- layouts that are not depth electrodes in a named structure ------------
#
# The regression these guard is specific and was found by sweeping the other
# recordings rather than by thinking: on the subdural dataset not one electrode
# name matches the table, so every contact landed on the single `UNKNOWN`
# target and seventy of them from a dozen electrodes drew as one blob a few
# millimetres across. That is worse than useless -- a tight cluster of activity
# is a finding, and this one was an artifact of having no information at all.

def _ecog_layout():
    """A subdural montage: strips and a grid, no name in the region table."""
    import pandas as pd

    channels, ranks = [], []
    # Real ds003029 shaft names, none of which is in REGIONS. "G" is left out
    # on purpose: it *is* in the table, as a subdural grid, so it would take
    # the mapped path and this fixture is about the unmapped one.
    for shaft, n in (("PST", 4), ("ATT", 8), ("MLT", 32), ("SF", 6)):
        for i in range(1, n):
            channels.append(f"{shaft}{i}-{shaft}{i + 1}")
            ranks.append(len(channels))
    return pd.DataFrame({
        "channel": channels, "rank": ranks,
        "n_events": [1] * len(channels), "rate_per_min": [1.0] * len(channels),
        "reviewed": [False] * len(channels),
    })


class _Fake:
    def __init__(self, findings):
        self.findings = findings
        self.candidates = []
        self.reviewed_channels = []


def test_unmapped_shafts_are_spread_rather_than_piled_up():
    layout = electrode_layout(_Fake(_ecog_layout()))
    assert set(layout["region"]) == {"unmapped"}
    centres = layout.groupby("shaft")[["x", "y", "z"]].mean().to_numpy()
    gaps = [np.linalg.norm(a - b)
            for i, a in enumerate(centres) for b in centres[i + 1:]]
    # Every pair of electrodes at least a centimetre apart: enough to tell them
    # apart on screen, which is the only claim being made.
    assert min(gaps) > 0.010, f"shafts collapsed together: {min(gaps):.4f} m"


def test_unmapped_positions_are_stable_between_runs():
    """`hash()` is salted per process; an electrode that moved every run would
    make two screenshots of the same analysis disagree."""
    from onset_review.anatomy import _inferred_position

    first = _inferred_position("PST", 3, 8)
    assert _inferred_position("PST", 3, 8) == first
    assert _inferred_position("ATT", 3, 8) != first


def test_contacts_along_an_unmapped_shaft_stay_in_order():
    from onset_review.anatomy import _inferred_position

    points = np.array([_inferred_position("MLT", i, 32) for i in range(1, 33)])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.allclose(steps, steps[0], atol=1e-9)      # evenly spaced
    assert steps[0] > 0


def test_every_contact_is_drawn_inside_the_head():
    """A contact floating outside the reference surface reads as a mistake.

    Capping shaft length is not enough on its own: a 16-contact electrode aimed
    at a lateral structure still reaches past the skull, which is how one
    subject ended up with half its contacts outside.
    """
    from onset_review.anatomy import HEAD_AXES_M, HEAD_FILL

    axes = np.asarray(HEAD_AXES_M)
    for findings in (_ecog_layout(),
                     pd.DataFrame({"channel": [f"OTL{i}-OTL{i + 1}"
                                               for i in range(1, 16)],
                                   "rank": list(range(1, 16)),
                                   "n_events": [1] * 15,
                                   "rate_per_min": [1.0] * 15,
                                   "reviewed": [False] * 15})):
        layout = electrode_layout(_Fake(findings))
        radius = np.linalg.norm(layout[["x", "y", "z"]].to_numpy() / axes, axis=1)
        assert radius.max() <= HEAD_FILL + 1e-9, f"{radius.max():.3f}"


def test_a_long_shaft_is_scaled_rather_than_marching_out():
    from onset_review.anatomy import MAX_SHAFT_LENGTH_M, _pitch

    assert _pitch(8, MAX_SHAFT_LENGTH_M) == SHAFT_PITCH_M       # fits as it is
    assert _pitch(32, MAX_SHAFT_LENGTH_M) < SHAFT_PITCH_M       # scaled down
    assert _pitch(32, MAX_SHAFT_LENGTH_M) * 31 == pytest.approx(
        MAX_SHAFT_LENGTH_M)


def test_a_wholly_unmapped_montage_is_not_called_anatomy():
    caption = layout_caption(electrode_layout(_Fake(_ecog_layout())))
    assert "MONTAGE DIAGRAM" in caption
    assert "no anatomy at all" in caption


def test_a_measured_position_is_never_moved_to_fit_the_head():
    """Containment is for the schematic. A measured contact goes where it was
    measured, even if that is outside a cartoon skull."""
    import pandas as pd

    findings = _ecog_layout().head(3)
    contacts = sorted({c for ch in findings["channel"] for c in ch.split("-")})
    electrodes = pd.DataFrame({"name": contacts,
                               "x": [500.0] * len(contacts),   # mm, absurd
                               "y": [0.0] * len(contacts),
                               "z": [0.0] * len(contacts)})
    layout = electrode_layout(_Fake(findings), electrodes=electrodes)
    assert set(layout["source"]) == {"archive"}
    assert layout["x"].max() == pytest.approx(0.5)
