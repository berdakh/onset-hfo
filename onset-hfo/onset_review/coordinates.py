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
    matched: tuple = ()                  #: contacts in both the file and the recording
    unplaced: tuple = ()                 #: contacts in the recording but not the file
    unused: tuple = ()                   #: contacts in the file but not the recording
    problems: tuple = ()                 #: why it cannot be used, if it cannot

    @property
    def usable(self) -> bool:
        return bool(len(self.frame)) and not self.problems and bool(self.matched)

    @property
    def coverage(self) -> float:
        total = len(self.matched) + len(self.unplaced)
        return (len(self.matched) / total) if total else 0.0

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
        return text


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
    return Coordinates(frame=frame, path=path, units=units, matched=matched,
                       unplaced=unplaced, unused=unused)


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
