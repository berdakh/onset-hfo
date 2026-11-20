"""Electrode coordinates a hospital actually has, from a file a reader picks.

The 3D view says "schematic" because neither public archive ships
`electrodes.tsv`, and `onset_review.anatomy` is careful to keep saying so. In
a hospital that caveat is false by accident rather than in principle: every
implanted patient has coordinates, from a post-implant CT coregistered to the
planning MRI. They are simply in the surgical planning system rather than in
the archive someone downloaded.

This module is the door between the two. Give it the file that system exports
-- a BIDS `electrodes.tsv`, or a CSV out of anything else -- and the 3D view
stops being a montage diagram and starts being that patient's head.

Three things it is careful about, because all three are ways to produce a
confident-looking picture of the wrong thing:

**Names.** A coordinate file whose contact names do not match the recording's
is worse than no file: it places *some* contacts and silently leaves the rest
schematic, which is a mixed picture nobody can read. So the match is counted
and reported before anything is applied, and a file matching nothing is
refused rather than applied to an empty set.

**Units.** BIDS puts iEEG coordinates in millimetres and the unit in a
companion JSON, which a file exported on its own does not carry. Magnitude
tells them apart -- a head is tenths of a metre, never tens -- and the reader
is told which way it was read.

**What the coordinates are relative to.** Nothing here can check that, and it
does not pretend to: a file in scanner space, in MNI space or in the
planning system's own frame all look identical. The view draws the contacts
relative to each other, which is what the clinical question needs, and the
caption keeps saying the surface is a reference shape rather than this
patient's cortex.

Qt-free. `onset_review.window` opens the dialog; everything decided is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Coordinates", "read_coordinates", "contacts_of", "FILE_FILTER",
           "NAME_COLUMNS", "AXIS_COLUMNS"]

#: What the name column may be called. BIDS says `name`; export dialogs say
#: everything else. Matched case-insensitively, first hit wins, in this order
#: so that a file with both `name` and `label` uses the BIDS one.
NAME_COLUMNS = ("name", "label", "electrode", "contact", "channel", "ch_name")

#: What each axis column may be called, in the same way. `R`/`A`/`S` are the
#: anatomical axis letters some planning systems use instead of x/y/z.
AXIS_COLUMNS = {
    "x": ("x", "pos_x", "x_mm", "r", "x_coord"),
    "y": ("y", "pos_y", "y_mm", "a", "y_coord"),
    "z": ("z", "pos_z", "z_mm", "s", "z_coord"),
}

#: For the open dialog.
FILE_FILTER = ("Electrode coordinates (*.tsv *.csv *.txt);;"
               "BIDS electrodes (*electrodes.tsv);;All files (*)")

#: Above this, coordinates are taken to be millimetres. A human head is about
#: 0.2 m across, so anything over a metre cannot be metres; anything under is
#: taken to be metres already.
METRES_CEILING = 1.0


@dataclass
class Coordinates:
    """A coordinate file, read and checked against a recording."""

    #: `name`, `x`, `y`, `z` -- the shape `anatomy` already consumes, in the
    #: units the file carried.
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    path: Path | None = None
    units: str = ""                      #: "millimetres" or "metres"
    #: "template" (MNI, fsaverage or Talairach, per a BIDS coordsystem.json
    #: beside the file), "patient" (a scanner or surgical space named there),
    #: or "unknown" when nothing says.
    space: str = "unknown"
    matched: tuple = ()                  #: contacts in both the file and the recording
    unplaced: tuple = ()                 #: contacts in the recording but not the file
    unused: tuple = ()                   #: contacts in the file but not the recording
    problems: tuple = ()                 #: why it cannot be used, if it cannot
    #: Shafts whose name says one side and whose coordinates lie on the other,
    #: as `(shaft, side the name says, side the coordinates say)`.
    side_conflicts: tuple = ()
    #: How many shafts had a side in their name and a position off the midline,
    #: i.e. how many the check above could be made on.
    side_checked: int = 0

    @property
    def usable(self) -> bool:
        return bool(len(self.frame)) and not self.problems and bool(self.matched)

    @property
    def coverage(self) -> float:
        total = len(self.matched) + len(self.unplaced)
        return (len(self.matched) / total) if total else 0.0

    @property
    def mirrored(self) -> bool:
        """Most shafts that could be checked lie on the side their name denies.

        One shaft on the wrong side is a naming quirk or a midline electrode.
        Most of them is a file in a left-handed frame -- LPS, which is what
        DICOM and several planning systems write -- being read as RAS, and
        the whole implant drawn mirrored."""
        return (self.side_checked >= 2
                and len(self.side_conflicts) * 2 > self.side_checked)

    def side_warning(self) -> str:
        """What the reviewer must be told about left and right, or ""."""
        if not self.side_conflicts:
            return ""
        listed = "; ".join(f"{shaft} is named {_SIDE[named]} and lies on the "
                           f"{_SIDE[placed]}"
                           for shaft, named, placed in self.side_conflicts[:8])
        more = (f"; and {len(self.side_conflicts) - 8} more"
                if len(self.side_conflicts) > 8 else "")
        if self.mirrored:
            return (f"LEFT AND RIGHT DISAGREE on {len(self.side_conflicts)} of "
                    f"{self.side_checked} electrodes: {listed}{more}. This "
                    "software reads x as increasing toward the patient's "
                    "right (RAS). A file written the other way round (LPS, as "
                    "DICOM and several planning systems export) is drawn "
                    "MIRRORED, every contact on the wrong side. Check the "
                    "file's orientation before using these positions. If the "
                    "trailing L or R in these names does not mean a side, the "
                    "names are the thing to distrust instead.")
        return (f"Left and right disagree on {len(self.side_conflicts)} of "
                f"{self.side_checked} electrodes: {listed}{more}. Check these "
                "against the implantation plan before reading a side off the "
                "picture.")

    def summary(self) -> str:
        """One honest sentence, whatever the outcome."""
        if self.problems:
            return " ".join(self.problems)
        if not self.matched:
            return ("None of the names in this file match the contacts in this "
                    "recording, so nothing can be placed from it.")
        total = len(self.matched) + len(self.unplaced)
        text = (f"{len(self.matched)} of {total} contacts placed from "
                f"{self.path.name if self.path else 'the file'}, read as "
                f"{self.units}.")
        if self.unplaced:
            shown = ", ".join(self.unplaced[:6])
            more = f" and {len(self.unplaced) - 6} more" if len(self.unplaced) > 6 else ""
            text += (f" {len(self.unplaced)} contact(s) are not in the file and "
                     f"stay schematic: {shown}{more}.")
        if self.unused:
            text += (f" {len(self.unused)} name(s) in the file are not in this "
                     f"recording and were ignored.")
        warning = self.side_warning()
        if warning:
            text += " " + warning
        return text


_SIDE = {"L": "left", "R": "right"}

#: Closer to the midline than this, in millimetres, a shaft's mean position
#: says nothing about its side: interhemispheric and midline electrodes exist.
MIDLINE_MM = 3.0


def laterality_audit(names, xs, units: str = "millimetres") -> tuple[tuple, int]:
    """Compare the side each electrode's name states with the side its
    coordinates lie on. Returns `(conflicts, n_checked)`.

    The name is read the one way this software reads names: a trailing `L` or
    `R` on the letters (`AHL3` is left). An electrode whose name gives no side
    is not checked, and neither is one within `MIDLINE_MM` of the midline. The
    position is read as RAS, x increasing to the patient's right, which is
    what every other part of this software assumes of a coordinate file and
    what nothing in the file can confirm.
    """
    from onset_review.anatomy import parse_channel

    scale = 1.0 if units == "millimetres" else 1000.0
    by_shaft: dict = {}
    for name, x in zip(names, xs, strict=True):
        try:
            value = float(x) * scale
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        shaft, _, side, _ = parse_channel(str(name).upper())
        if side in _SIDE:
            by_shaft.setdefault((shaft, side), []).append(value)
    conflicts, checked = [], 0
    for (shaft, side), values in sorted(by_shaft.items()):
        centre = float(np.mean(values))
        if abs(centre) < MIDLINE_MM:
            continue
        checked += 1
        placed = "L" if centre < 0 else "R"
        if placed != side:
            conflicts.append((shaft, side, placed))
    return tuple(conflicts), checked


def contacts_of(session) -> list:
    """Every physical contact the recording has, upper-cased.

    A bipolar channel is a pair, and it is the *pair's* contacts that a
    coordinate file names -- so `SA1-SA2` contributes `SA1` and `SA2`.
    Matching on the channel name instead is the mistake that makes a perfectly
    good coordinate file look like it matches nothing.
    """
    from onset_review.anatomy import electrode_layout

    layout = electrode_layout(session)
    names: list = []
    for column in ("contact_a", "contact_b"):
        if column in layout.columns:
            names.extend(str(value).upper() for value in layout[column]
                         if value and str(value).lower() != "nan")
    if not names:
        names = [str(ch).upper() for ch in getattr(session, "reviewed_channels", [])]
    seen, out = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def read_coordinates(path, session=None) -> Coordinates:
    """Read a coordinate file and say what it would place. Never raises.

    Reading is separate from applying on purpose: the reviewer is shown the
    match before anything on screen changes, because "47 of 64 contacts
    placed, the rest stay schematic" is a thing to decide about rather than
    discover afterwards.
    """
    path = Path(path)
    frame, problems = _read_table(path)
    if problems:
        return Coordinates(path=path, problems=tuple(problems))

    frame, problems = _normalise(frame)
    if problems:
        return Coordinates(path=path, problems=tuple(problems))

    units = ("millimetres"
             if float(np.abs(frame[["x", "y", "z"]].to_numpy()).max()) > METRES_CEILING
             else "metres")

    wanted = contacts_of(session) if session is not None else []
    have = {str(name).upper() for name in frame["name"]}
    matched = tuple(name for name in wanted if name in have)
    unplaced = tuple(name for name in wanted if name not in have)
    unused = tuple(sorted(have - set(wanted))) if wanted else ()
    conflicts, checked = laterality_audit(frame["name"], frame["x"], units)
    return Coordinates(frame=frame, path=path, units=units, matched=matched,
                       unplaced=unplaced, unused=unused, space=coordinate_space(path),
                       side_conflicts=conflicts, side_checked=checked)


#: Coordinate systems a BIDS sidecar names that are a template, not a head.
TEMPLATE_SPACES = ("MNI", "FSAVERAGE", "TALAIRACH", "ICBM")


def coordinate_space(path) -> str:
    """"template", "patient" or "unknown", from a BIDS ``*coordsystem.json``
    beside the file, when there is one. A file alone says nothing about its
    space, and nothing here guesses."""
    import json

    path = Path(path)
    sidecars = sorted(path.parent.glob("*coordsystem.json"))
    for sidecar in sidecars:
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key, value in payload.items():
            if "CoordinateSystem" in str(key) and isinstance(value, str):
                named = value.upper()
                if any(token in named for token in TEMPLATE_SPACES):
                    return "template"
                return "patient"
    return "unknown"


# --------------------------------------------------------------------------


def _read_table(path: Path):
    """Whatever delimiter the file happens to use, or a reason it cannot be read."""
    if not path.exists():
        return pd.DataFrame(), [f"{path} does not exist."]
    try:
        # `sep=None` with the Python engine sniffs tab, comma or semicolon,
        # which between them covers every export anyone has handed over.
        frame = pd.read_csv(path, sep=None, engine="python",
                            comment="#", skip_blank_lines=True)
    except Exception as error:          # noqa: BLE001 - any parse failure is one answer
        return pd.DataFrame(), [f"Could not read {path.name}: {error}"]
    if frame.empty:
        return pd.DataFrame(), [f"{path.name} has no rows."]
    # A headerless four-column file reads as one row of column names; the
    # giveaway is that three of the four "names" parse as numbers.
    if _looks_headerless(frame):
        try:
            frame = pd.read_csv(path, sep=None, engine="python", header=None,
                                comment="#", skip_blank_lines=True)
            frame.columns = ["name", "x", "y", "z"][:frame.shape[1]] + list(
                frame.columns[4:])
        except Exception as error:      # noqa: BLE001
            return pd.DataFrame(), [f"Could not read {path.name}: {error}"]
    return frame, []


def _looks_headerless(frame: pd.DataFrame) -> bool:
    numeric = 0
    for column in list(frame.columns)[1:4]:
        try:
            float(str(column))
        except ValueError:
            continue
        numeric += 1
    return numeric >= 3


def _normalise(frame: pd.DataFrame):
    """Down to `name`, `x`, `y`, `z`, or a reason it cannot be."""
    lookup = {str(column).strip().lower(): column for column in frame.columns}

    def pick(options):
        for option in options:
            if option in lookup:
                return lookup[option]
        return None

    name_column = pick(NAME_COLUMNS)
    axes = {axis: pick(options) for axis, options in AXIS_COLUMNS.items()}
    missing = [axis for axis, column in axes.items() if column is None]
    if name_column is None:
        return pd.DataFrame(), [
            "No column of contact names. One of "
            + ", ".join(NAME_COLUMNS) + " is needed; this file has "
            + ", ".join(str(c) for c in frame.columns) + "."]
    if missing:
        return pd.DataFrame(), [
            f"No {'/'.join(missing)} column. This file has "
            + ", ".join(str(c) for c in frame.columns) + "."]

    out = pd.DataFrame({
        "name": [str(value).strip().upper() for value in frame[name_column]],
        **{axis: pd.to_numeric(frame[column], errors="coerce")
           for axis, column in axes.items()},
    })
    # A contact with no position is not a contact at position zero. BIDS uses
    # `n/a` for electrodes that were not localised, and reading those as the
    # origin puts a cluster at the centre of the head that looks like a finding.
    out = out[np.isfinite(out[["x", "y", "z"]].to_numpy()).all(axis=1)]
    out = out[out["name"].astype(bool)]
    out = out.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(), [
            "Every row in this file is missing a coordinate."]
    return out, []


@dataclass
class ImportAudit:
    """A coordinate file checked against the contacts a case recorded.

    The single-recording import has always shown its match before placing
    anything. The case import did not look: any file was merged into the
    case's contacts, so another patient's export -- same electrode names, as
    SEEG names are -- placed cleanly, and this patient's rates were drawn on
    that patient's geometry.
    """

    matched: tuple = ()          #: recorded and in the file
    unplaced: tuple = ()         #: recorded, not in the file
    unused: tuple = ()           #: in the file, never recorded in this case
    #: Electrodes whose contact count differs: `(electrode, in file, recorded)`.
    count_mismatch: tuple = ()
    side_conflicts: tuple = ()
    side_checked: int = 0
    n_recorded: int = 0

    @property
    def mirrored(self) -> bool:
        return (self.side_checked >= 2
                and len(self.side_conflicts) * 2 > self.side_checked)

    @property
    def refused(self) -> str:
        """Why the file cannot be this case's, or ""."""
        if self.n_recorded and not self.matched:
            return ("None of the contact names in this file is a contact this "
                    "case recorded, so nothing can be placed from it. It is "
                    "another patient's file, or its names are written "
                    "differently (LA01 for LA1, say).")
        return ""

    @property
    def doubtful(self) -> bool:
        """Something a person should decide about before the positions are used."""
        return bool(self.unused or self.count_mismatch or self.side_conflicts)

    def summary(self) -> str:
        if self.refused:
            return self.refused
        if not self.n_recorded:
            return ("No recording in this case yet, so the names in the file "
                    "could not be checked against its contacts.")
        parts = [f"{len(self.matched)} of {self.n_recorded} recorded contacts "
                 f"are in the file."]
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} recorded contact(s) are not in "
                         f"it and stay unplaced: {_some(self.unplaced)}.")
        if self.unused:
            parts.append(f"{len(self.unused)} name(s) in the file were NEVER "
                         f"RECORDED in this case: {_some(self.unused)}. A file "
                         f"from another patient looks like this.")
        if self.count_mismatch:
            parts.append("Contact counts differ: " + "; ".join(
                f"{shaft} has {in_file} in the file and {recorded} recorded"
                for shaft, in_file, recorded in self.count_mismatch[:8]) + ".")
        sides = Coordinates(side_conflicts=self.side_conflicts,
                            side_checked=self.side_checked).side_warning()
        if sides:
            parts.append(sides)
        return " ".join(parts)


def _some(names, limit: int = 8) -> str:
    names = list(names)
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more"
                                       if len(names) > limit else "")


def audit_import(names, xs, recorded) -> ImportAudit:
    """Check a coordinate file's contacts against the ones a case recorded.

    `names` and `xs` are the file's contact names and x in millimetres (RAS);
    `recorded` is every intracranial contact name in the case's recordings.
    Names are compared without regard to case and otherwise exactly: `LA01`
    is not quietly taken for `LA1`, it is reported.
    """
    from onset_review.anatomy import parse_channel

    have = [str(n).strip().upper() for n in names]
    wanted = list(dict.fromkeys(str(n).strip().upper() for n in recorded))
    matched = tuple(n for n in wanted if n in set(have))
    unplaced = tuple(n for n in wanted if n not in set(have))
    unused = tuple(sorted(set(have) - set(wanted))) if wanted else ()

    def per_shaft(group) -> dict:
        counts: dict = {}
        for name in group:
            shaft = parse_channel(name)[0]
            counts[shaft] = counts.get(shaft, 0) + 1
        return counts

    in_file, in_case = per_shaft(set(have)), per_shaft(wanted)
    mismatch = tuple((shaft, in_file[shaft], in_case[shaft])
                     for shaft in sorted(set(in_file) & set(in_case))
                     if in_file[shaft] != in_case[shaft]) if wanted else ()
    conflicts, checked = laterality_audit(have, list(xs), "millimetres")
    return ImportAudit(matched=matched, unplaced=unplaced, unused=unused,
                       count_mismatch=mismatch, side_conflicts=conflicts,
                       side_checked=checked, n_recorded=len(wanted))
