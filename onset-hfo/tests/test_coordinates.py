"""Electrode coordinates supplied by a reader, and what is claimed about them.

Two kinds of test here. The first kind is parsing: every export dialog writes
a different file and refusing all but one of them would mean the feature is
never used. The second kind, and the one that matters, is about claims — a
coordinate file that places half the contacts, or that is read in the wrong
units, produces a confident picture of the wrong head, and the only defence is
that the software says so before it draws anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from onset_review import coordinates
from onset_review.anatomy import NOT_ANATOMY, electrode_layout, layout_caption


@dataclass
class _Session:
    """Enough of a session for `electrode_layout` and `contacts_of`."""

    findings: pd.DataFrame
    candidates: list = field(default_factory=list)
    reviewed_channels: list = field(default_factory=list)


def _session(channels=("AL1-AL2", "AL2-AL3", "BR1-BR2", "BR2-BR3")) -> _Session:
    return _Session(findings=pd.DataFrame({
        "channel": list(channels),
        "rank": range(1, len(channels) + 1),
        "n_events": [10, 8, 6, 4][:len(channels)],
        "rate_per_min": [10.0, 8.0, 6.0, 4.0][:len(channels)],
        "reviewed": [True] * len(channels),
    }))


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- what a contact is ------------------------------------------------------


def test_the_contacts_are_the_pairs_members_not_the_channel_names():
    """The mistake that makes a perfectly good coordinate file look like it
    matches nothing: a file names `AL1`, the window names `AL1-AL2`."""
    names = coordinates.contacts_of(_session())
    assert set(names) == {"AL1", "AL2", "AL3", "BR1", "BR2", "BR3"}


# -- parsing ----------------------------------------------------------------


def test_a_bids_electrodes_tsv_is_read(tmp_path):
    path = _write(tmp_path, "electrodes.tsv",
                  "name\tx\ty\tz\tsize\n"
                  "AL1\t-30.0\t-20.0\t-15.0\t2\n"
                  "AL2\t-35.0\t-20.0\t-15.0\t2\n")
    read = coordinates.read_coordinates(path, _session())
    assert read.usable
    assert read.units == "millimetres"
    assert list(read.frame["name"]) == ["AL1", "AL2"]
    assert read.matched == ("AL1", "AL2")


def test_metres_and_millimetres_are_told_apart_by_magnitude(tmp_path):
    """A head is tenths of a metre, never tens. Getting this wrong puts every
    contact a thousand times too far out."""
    millimetres = _write(tmp_path, "mm.tsv", "name\tx\ty\tz\nAL1\t-30\t-20\t-15\n")
    metres = _write(tmp_path, "m.tsv", "name\tx\ty\tz\nAL1\t-0.030\t-0.020\t-0.015\n")
    assert coordinates.read_coordinates(millimetres, _session()).units == "millimetres"
    assert coordinates.read_coordinates(metres, _session()).units == "metres"


def test_a_comma_separated_file_works_too(tmp_path):
    path = _write(tmp_path, "export.csv", "name,x,y,z\nAL1,-30,-20,-15\n")
    assert coordinates.read_coordinates(path, _session()).usable


def test_the_column_names_an_export_dialog_actually_writes(tmp_path):
    """`label` and the anatomical axis letters, which some planning systems
    use instead of name/x/y/z."""
    path = _write(tmp_path, "planning.csv",
                  "Label,R,A,S\nAL1,-30,-20,-15\nAL2,-35,-20,-15\n")
    read = coordinates.read_coordinates(path, _session())
    assert read.usable
    assert read.matched == ("AL1", "AL2")


def test_a_headerless_four_column_file_is_recognised(tmp_path):
    path = _write(tmp_path, "plain.txt", "AL1 -30 -20 -15\nAL2 -35 -20 -15\n")
    read = coordinates.read_coordinates(path, _session())
    assert read.usable
    assert read.matched == ("AL1", "AL2")


def test_names_are_matched_without_regard_to_case(tmp_path):
    path = _write(tmp_path, "lower.csv", "name,x,y,z\nal1,-30,-20,-15\n")
    assert coordinates.read_coordinates(path, _session()).matched == ("AL1",)


def test_a_contact_that_was_never_localised_is_dropped_not_placed_at_zero(tmp_path):
    """BIDS writes `n/a` for an electrode nobody localised. Reading those as
    the origin puts a cluster at the centre of the head that looks like a
    finding and is an artifact of a missing value."""
    path = _write(tmp_path, "gaps.tsv",
                  "name\tx\ty\tz\nAL1\t-30\t-20\t-15\nAL2\tn/a\tn/a\tn/a\n")
    read = coordinates.read_coordinates(path, _session())
    assert list(read.frame["name"]) == ["AL1"]
    assert "AL2" in read.unplaced


def test_a_repeated_contact_is_taken_once(tmp_path):
    path = _write(tmp_path, "dupes.csv",
                  "name,x,y,z\nAL1,-30,-20,-15\nAL1,99,99,99\n")
    read = coordinates.read_coordinates(path, _session())
    assert len(read.frame) == 1
    assert float(read.frame.iloc[0]["x"]) == -30.0


# -- claims -----------------------------------------------------------------


def test_a_file_matching_nothing_is_refused_rather_than_applied(tmp_path):
    """Worse than no file: it would place nothing and leave a view that looks
    like it was given coordinates."""
    path = _write(tmp_path, "other.csv", "name,x,y,z\nZZ1,1,2,3\nZZ2,4,5,6\n")
    read = coordinates.read_coordinates(path, _session())
    assert not read.usable
    assert "match" in read.summary()


def test_a_partial_match_says_how_partial_and_names_what_is_missing(tmp_path):
    path = _write(tmp_path, "half.csv",
                  "name,x,y,z\nAL1,-30,-20,-15\nAL2,-35,-20,-15\n")
    read = coordinates.read_coordinates(path, _session())
    assert read.usable
    assert read.coverage == pytest.approx(2 / 6)
    summary = read.summary()
    assert "2 of 6 contacts placed" in summary
    assert "stay schematic" in summary
    assert "BR1" in summary


def test_names_in_the_file_that_are_not_in_the_recording_are_counted(tmp_path):
    path = _write(tmp_path, "extra.csv",
                  "name,x,y,z\nAL1,-30,-20,-15\nQQ9,1,2,3\n")
    read = coordinates.read_coordinates(path, _session())
    assert read.unused == ("QQ9",)
    assert "not in this recording" in read.summary()


def test_a_missing_axis_is_a_refusal_that_says_which(tmp_path):
    path = _write(tmp_path, "flat.csv", "name,x,y\nAL1,-30,-20\n")
    read = coordinates.read_coordinates(path, _session())
    assert not read.usable
    assert "No z column" in read.summary()


def test_a_file_with_no_names_is_a_refusal_that_says_so(tmp_path):
    path = _write(tmp_path, "anon.csv", "x,y,z\n-30,-20,-15\n")
    read = coordinates.read_coordinates(path, _session())
    assert not read.usable
    assert "No column of contact names" in read.summary()


def test_a_file_that_is_not_there_is_a_refusal_not_a_crash(tmp_path):
    read = coordinates.read_coordinates(tmp_path / "nope.tsv", _session())
    assert not read.usable
    assert "does not exist" in read.summary()


def test_a_file_that_is_not_a_table_is_a_refusal_not_a_crash(tmp_path):
    path = _write(tmp_path, "notes.txt", "")
    read = coordinates.read_coordinates(path, _session())
    assert not read.usable
    assert read.summary()


# -- what the view then says ------------------------------------------------


def test_coordinates_replace_the_schematic_layout_and_the_caption(tmp_path):
    session = _session()
    before = layout_caption(electrode_layout(session))
    assert NOT_ANATOMY in before

    path = _write(tmp_path, "electrodes.tsv", "name\tx\ty\tz\n" + "".join(
        f"{name}\t{-30 if name.startswith('A') else 30}\t{-20 + i * 4}\t-15\n"
        for i, name in enumerate(coordinates.contacts_of(session))))
    read = coordinates.read_coordinates(path, session)
    layout = electrode_layout(session, electrodes=read.frame)
    assert set(layout["source"]) == {"archive"}
    after = layout_caption(layout, origin="the file you supplied")
    assert NOT_ANATOMY not in after
    assert "the file you supplied" in after


def test_the_caption_does_not_claim_the_dataset_supplied_them(tmp_path):
    """The default wording says "the dataset's own"; a reviewer's file is not
    the dataset, and a false provenance claim in the caption of the view whose
    job is to say what is known is the worst place for one."""
    from onset_review.anatomy import ARCHIVE_ORIGIN

    session = _session()
    path = _write(tmp_path, "mine.csv", "name,x,y,z\n" + "".join(
        f"{name},{-30 if name.startswith('A') else 30},{-20 + i * 4},-15\n"
        for i, name in enumerate(coordinates.contacts_of(session))))
    layout = electrode_layout(
        session, electrodes=coordinates.read_coordinates(path, session).frame)
    mine = layout_caption(layout, origin="the coordinate file you supplied")
    assert ARCHIVE_ORIGIN not in mine
    assert "you supplied" in mine


def test_a_measured_position_settles_which_side_a_contact_is_on(tmp_path):
    """Plenty of real electrode names carry no L or R. A layout built from
    coordinates should not keep saying "side unknown" about something it has
    just been told."""
    session = _session(channels=("P1-P2", "P2-P3", "Q1-Q2"))
    path = _write(tmp_path, "sides.csv", "name,x,y,z\n"
                  "P1,-30,-20,-15\nP2,-35,-20,-15\nP3,-40,-20,-15\n"
                  "Q1,30,-20,-15\nQ2,35,-20,-15\n")
    read = coordinates.read_coordinates(path, session)
    layout = electrode_layout(session, electrodes=read.frame)
    sides = dict(zip(layout["channel"], layout["hemisphere"], strict=True))
    assert sides["P1-P2"] == "L"
    assert sides["Q1-Q2"] == "R"
    # And without coordinates it stays honest about not knowing.
    plain = electrode_layout(session)
    assert set(plain["hemisphere"]) == {"?"}


# -- left and right ----------------------------------------------------------

_RAS = ("name,x,y,z\n"
        "AL1,-30,-20,-15\nAL2,-35,-20,-15\nAL3,-40,-20,-15\n"
        "BR1,30,-20,-15\nBR2,35,-20,-15\nBR3,40,-20,-15\n")


def test_a_file_whose_sides_match_its_names_raises_no_question(tmp_path):
    read = coordinates.read_coordinates(_write(tmp_path, "ras.csv", _RAS), _session())
    assert read.side_checked == 2
    assert read.side_conflicts == () and not read.mirrored
    assert read.side_warning() == ""
    assert "disagree" not in read.summary().lower()


def test_a_mirrored_file_is_called_mirrored_before_it_is_drawn(tmp_path):
    """LPS read as RAS puts every contact on the wrong side, and the picture
    it draws looks exactly as finished as a correct one."""
    flipped = "\n".join(
        line if line.startswith("name") else
        ",".join([line.split(",")[0], str(-float(line.split(",")[1])),
                  *line.split(",")[2:]])
        for line in _RAS.strip().splitlines()) + "\n"
    read = coordinates.read_coordinates(_write(tmp_path, "lps.csv", flipped),
                                        _session())
    assert read.usable, "the reviewer decides; the file is not silently refused"
    assert read.mirrored
    assert read.side_conflicts == (("AL", "L", "R"), ("BR", "R", "L"))
    text = read.summary()
    assert "LEFT AND RIGHT DISAGREE on 2 of 2" in text
    assert "MIRRORED" in text and "LPS" in text


def test_one_shaft_on_the_wrong_side_is_named_and_not_called_a_mirror(tmp_path):
    text = _RAS + "CL1,20,0,0\nCL2,25,0,0\nDR1,45,10,0\nDR2,50,10,0\n"
    read = coordinates.read_coordinates(_write(tmp_path, "one.csv", text), _session())
    assert read.side_conflicts == (("CL", "L", "R"),)
    assert read.side_checked == 4 and not read.mirrored
    assert "CL is named left and lies on the right" in read.summary()


def test_the_side_check_skips_what_it_cannot_judge(tmp_path):
    """No side in the name, or a shaft on the midline: neither is evidence."""
    conflicts, checked = coordinates.laterality_audit(
        ["P1", "P2", "ML1", "ML2", "AR1"], [30, 35, 1.0, 2.0, 0.030 * 1000])
    assert conflicts == () and checked == 1
    # Metres are scaled before the midline margin is applied.
    conflicts, checked = coordinates.laterality_audit(
        ["AL1", "AL2"], [0.030, 0.035], units="metres")
    assert conflicts == (("AL", "L", "R"),) and checked == 1


def test_coordinates_do_not_overrule_a_side_the_name_states(tmp_path):
    """Settling the conflict in the file's favour would erase the only sign
    that the file is mirrored."""
    session = _session()
    flipped = _RAS.replace(",-", ",+").replace(",3", ",-3").replace(",4", ",-4") \
        .replace(",+", ",")
    read = coordinates.read_coordinates(_write(tmp_path, "f.csv", flipped), session)
    assert read.mirrored
    layout = electrode_layout(session, electrodes=read.frame)
    sides = dict(zip(layout["channel"], layout["hemisphere"], strict=True))
    assert sides["AL1-AL2"] == "L" and sides["BR1-BR2"] == "R"
