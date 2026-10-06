"""The flat map: every contact as a dot, coloured by what it measured.

The figure a paper prints is a per-contact count map -- each electrode
contact a dot, the dot's colour its rate, a colour scale beside it, the
clinical region marked. The 3D view already has the geometry; this module is
the same layout made flat, so a page can draw it with a colour bar and a
reviewer can save it as a figure, and so the assistant can be handed the
numbers behind it.

Qt-free on purpose. The frame, the projection, the summary and the table the
assistant reads are all computed here and tested without a display; the
panel in :mod:`onset_review.mapview` only draws.

**What is drawn is only as true as the coordinates.** Positions come from
:func:`onset_review.anatomy.electrode_layout`: measured when the recording
has an ``electrodes.tsv`` or a reviewer supplied one, inferred from the
electrode names otherwise, and the caption says which. A map of inferred
positions shows which shafts are active and in what order along them; it is
not anatomy, and the panel labels it schematic for that reason. No rendered
cortex: these archives ship no MRI, and a surface nobody measured would be a
picture of nothing.

**What the assistant is told.** The table saved beside the analysis carries
each channel's shaft, side, region, position, rate and rank, and whether the
positions are measured or schematic. It does not carry the resection. That
is shown to the reviewer, in rings; the model is never told what the surgeon
removed, because an answer that said "the leader was resected" would read as
an opinion about surgery, which nothing in this product is allowed to have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from onset_review import anatomy

__all__ = ["MEASURES", "VIEWS", "map_frame", "project", "head_outline",
           "contacts_table", "where_summary"]

#: (key, label, colour-bar label). "primary" is the detector the window ranks
#: by, from the findings table; the others are counted from the event list
#: on this window; "expert" is the archive annotators' own marks.
MEASURES = (
    ("primary", "Rate (ranking detector)", "events / min"),
    ("rms", "Rate (RMS detector)", "events / min"),
    ("line_length", "Rate (line-length detector)", "events / min"),
    ("spike", "Discharges", "spikes / min"),
    ("expert", "Expert markings", "marked events / min"),
    ("rank", "Rank", "rank (1 = busiest)"),
)

#: Name -> (horizontal, vertical) axes of the projection, as signed indices
#: into (x, y, z): right-positive, anterior-positive, superior-positive. A
#: negative index mirrors the axis so the view is as seen from that side.
VIEWS = {
    "Top": ((0, 1.0), (1, 1.0)),       # x right, y up: looking down, nose up
    "Right": ((1, 1.0), (2, 1.0)),     # anterior to the right, superior up
    "Left": ((1, -1.0), (2, 1.0)),     # anterior to the left
    "Front": ((0, 1.0), (2, 1.0)),
    "Back": ((0, -1.0), (2, 1.0)),
}


def map_frame(session, resection=None, electrodes=None,
              measure: str = "primary") -> pd.DataFrame:
    """The layout frame with a ``value`` column for the chosen measure.

    Rates other than the ranking detector's are counted from the window's
    accepted events, over the analysed span, so they are this window's and
    nothing else's. Expert markings exist only on the channels the annotators
    reviewed; the rest are zero, and ``reviewed`` says which is which.
    """
    frame = anatomy.electrode_layout(session, resection=resection, electrodes=electrodes)
    if frame.empty:
        frame["value"] = pd.Series(dtype=float)
        return frame
    if measure in ("primary", "rate_per_min"):
        frame["value"] = frame["rate_per_min"].astype(float)
    elif measure == "rank":
        frame["value"] = frame["rank"].astype(float)
    else:
        minutes = _minutes(session)
        counts: dict[str, int] = {}
        if measure == "expert":
            events = list(getattr(session, "expert", []) or [])
        else:
            events = [e for e in getattr(session, "events", []) or []
                      if getattr(e, "accepted", True) and e.detector == measure]
        for event in events:
            counts[str(event.channel)] = counts.get(str(event.channel), 0) + 1
        frame["value"] = [counts.get(str(c), 0) / minutes if minutes else 0.0
                          for c in frame["channel"]]
    return frame


def _minutes(session) -> float:
    span = getattr(session, "span", None)
    try:
        return max(float(span[1]) - float(span[0]), 1e-9) / 60.0
    except (TypeError, IndexError):
        return 1.0


def project(frame: pd.DataFrame, view: str = "Top") -> tuple[np.ndarray, np.ndarray]:
    """The (horizontal, vertical) coordinates of each row in `view`."""
    (h_axis, h_sign), (v_axis, v_sign) = VIEWS[view]
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float) if len(frame) else np.zeros((0, 3))
    return xyz[:, h_axis] * h_sign, xyz[:, v_axis] * v_sign


def head_outline(view: str = "Top") -> tuple[np.ndarray, np.ndarray]:
    """The reference head the 3D view draws, flattened to this view's plane:
    the convex outline of its hull, as a closed curve. A cartoon on purpose,
    like the surface it comes from."""
    from scipy.spatial import ConvexHull

    vertices, _faces = anatomy.brain_surface()
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    (h_axis, h_sign), (v_axis, v_sign) = VIEWS[view]
    flat = np.column_stack([points[:, h_axis] * h_sign, points[:, v_axis] * v_sign])
    hull = ConvexHull(flat)
    ring = np.append(hull.vertices, hull.vertices[0])
    return flat[ring, 0], flat[ring, 1]


# --------------------------------------------------------------------------
# What the assistant is handed
# --------------------------------------------------------------------------

#: Columns of the table saved beside the analysis for the assistant. No zone.
CONTACT_COLUMNS = ["channel", "shaft", "hemisphere", "region", "x", "y", "z",
                   "rank", "rate_per_min", "n_events", "source"]


def contacts_table(session, resection=None, electrodes=None) -> pd.DataFrame:
    """One row per channel with where it is and what it measured, for
    ``contacts.csv`` in the saved analysis. The resection is left out on
    purpose (see the module docstring)."""
    frame = anatomy.electrode_layout(session, resection=resection, electrodes=electrodes)
    if frame.empty:
        return pd.DataFrame(columns=CONTACT_COLUMNS)
    out = frame[CONTACT_COLUMNS].copy()
    out["hemisphere"] = out["hemisphere"].map(
        {"L": "left", "R": "right"}).fillna("unknown")
    return out.reset_index(drop=True)


def where_summary(contacts: pd.DataFrame, top: int = 5) -> dict:
    """The sentence the map says, with the counts behind it. The same
    function the saved analysis uses for the assistant, so the page and the
    model never disagree about where the activity is."""
    from onset_hfo.store import summarise_contacts

    return summarise_contacts(contacts, top=top)
