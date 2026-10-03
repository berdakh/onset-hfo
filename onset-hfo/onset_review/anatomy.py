"""Where the contacts are, and how much of that we actually know.

This module exists to answer one question honestly: *can we draw this
patient's electrodes in three dimensions?*

For the two archives this project reads, the answer is **no, not from measured
coordinates.** Neither `ds003498` nor `ds003029` ships `electrodes.tsv` or
`coordsystem.json`; there is no stereotactic position for any contact in
either. Drawing contacts at invented coordinates on a rendered brain would
produce a picture indistinguishable from a real implantation plan, which is
the one thing a prototype in a clinical setting must not do.

What the archives *do* carry is the recording site's own electrode names, and
those are not arbitrary. `AHR3` is the third contact of the right anterior
hippocampal depth electrode, in the convention Fedele et al. used, and a
neurophysiologist reads it that way without thinking. Two further facts are
structural rather than conventional: contacts along a depth electrode are
numbered in spatial order from the mesial tip outward -- which the
preprocessing already relies on when it builds bipolar pairs -- and the
trailing `L`/`R` is the hemisphere.

So this module builds a **schematic layout**: one shaft per electrode, aimed
at the structure its name claims, contacts spaced along it in order. Every row
it returns carries a `source` saying which of the two it is:

``archive``
    real coordinates, read from the dataset's own `electrodes.tsv`. No dataset
    here has one; the path exists and is tested so that a site pointing this
    software at its own BIDS data gets its real implantation.
``inferred``
    position derived from the channel name by the table below. Anatomically
    *plausible* and useful for seeing which shafts are hot and whether they sit
    inside the resection. **Not this patient's anatomy**, and everything that
    renders it is required to say so.

The distinction is carried in the data rather than left to the interface,
because an interface element can be closed, cropped out of a screenshot, or
forgotten, and a column cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ContactLayout", "electrode_layout", "parse_channel", "brain_surface",
           "REGIONS", "SHAFT_PITCH_M", "layout_caption", "NOT_ANATOMY"]

#: The one phrase every caption over an inferred layout carries, whichever of
#: the two it is. A single invariant rather than two wordings, so the interface
#: and the tests cannot drift into disagreeing about what is being denied.
NOT_ANATOMY = "not this patient's anatomy"

#: Spacing between neighbouring contacts along a depth electrode, in metres.
#: 5 mm is the usual pitch for the mesial-temporal depth electrodes this cohort
#: was implanted with. It sets the *scale* of the schematic, not any patient's
#: geometry.
SHAFT_PITCH_M = 0.005

#: Electrode-name prefix (hemisphere letter stripped) -> what it is, and where
#: that structure sits, as an approximate MNI target in metres.
#:
#: The coordinates are textbook locations of the structures, not measurements
#: of anybody. `x` is given for the right hemisphere and mirrored for the left.
#: `aim` is the unit direction from the target out toward the skull, i.e. the
#: direction of increasing contact number, which is how these electrodes are
#: implanted: laterally, through the temporal bone, tip mesial.
REGIONS: dict[str, dict] = {
    "A":   {"label": "amygdala", "target": (0.024, -0.004, -0.020)},
    "AH":  {"label": "anterior hippocampus", "target": (0.026, -0.018, -0.022)},
    "H":   {"label": "hippocampus", "target": (0.028, -0.026, -0.020)},
    "PH":  {"label": "posterior hippocampus", "target": (0.028, -0.036, -0.014)},
    "EC":  {"label": "entorhinal cortex", "target": (0.022, -0.010, -0.030)},
    "E":   {"label": "entorhinal cortex", "target": (0.022, -0.010, -0.030)},
    "IA":  {"label": "anterior insula", "target": (0.034, 0.012, 0.002)},
    "IP":  {"label": "posterior insula", "target": (0.036, -0.018, 0.006)},
    "I":   {"label": "insula", "target": (0.035, -0.004, 0.004)},
    "T":   {"label": "temporal neocortex", "target": (0.048, -0.020, -0.018)},
    "TP":  {"label": "temporal pole", "target": (0.034, 0.012, -0.030)},
    "P":   {"label": "posterior temporal", "target": (0.040, -0.048, -0.004)},
    "PL":  {"label": "lateral parietal", "target": (0.040, -0.058, 0.030)},
    "PM":  {"label": "medial parietal", "target": (0.010, -0.058, 0.034)},
    "OT":  {"label": "occipitotemporal", "target": (0.034, -0.058, -0.016)},
    "O":   {"label": "occipital", "target": (0.024, -0.078, 0.000)},
    "F":   {"label": "frontal", "target": (0.030, 0.038, 0.014)},
    "C":   {"label": "cingulate", "target": (0.008, 0.008, 0.028)},
    "G":   {"label": "subdural grid", "target": (0.052, -0.010, 0.020)},
}

#: What a region is called when the prefix matches nothing in the table. Named
#: rather than guessed: an unrecognised electrode gets a position and a label
#: that both say "we do not know what this is".
UNKNOWN = {"label": "unmapped", "target": (0.045, 0.000, 0.040)}

#: Where unmapped shafts are laid out: on an arc over the lateral surface,
#: one bearing per shaft. They used to share `UNKNOWN["target"]`, which on a
#: subdural dataset -- where no name matches the table -- piled seventy
#: contacts from ten electrodes into one blob a few millimetres across. That is
#: worse than useless: it reads as a tight cluster of activity, which is a
#: finding, and it is an artifact of having no information at all.
#:
#: Spreading them claims nothing new. Which shaft a contact is on, and its
#: order along that shaft, are both real; only the bearing is arbitrary, and it
#: is derived from the name so it never moves between runs.
UNMAPPED_RADIUS_M = 0.055
UNMAPPED_SPAN_DEG = 150.0
#: Longest any shaft is drawn, in metres, mapped or not. At the 5 mm
#: depth-electrode pitch a 31-contact grid marches 155 mm and a 16-contact
#: depth electrode 75 mm -- both straight out of a 144 mm head, which is how
#: half of one subject's contacts ended up floating outside the reference
#: surface. Long shafts are scaled to fit instead.
#:
#: Nothing is lost: the contact pitch is not known for any of these electrodes,
#: and what the drawing is for is which contacts share a shaft and their order
#: along it. Shafts short enough to fit keep the real 5 mm spacing.
MAX_SHAFT_LENGTH_M = 0.045
#: Unmapped shafts lie over the surface rather than boring through it, so they
#: have a little more room.
UNMAPPED_MAX_LENGTH_M = 0.060

#: Half-axes of the reference head, in metres. Shared by `brain_surface` and by
#: the containment below so the drawing and the hull cannot disagree about
#: where the head is.
HEAD_AXES_M = (0.072, 0.092, 0.062)
#: How far out a contact may be drawn, as a fraction of those half-axes.
#: Capping shaft length is not enough on its own -- a shaft aimed at a lateral
#: structure still reaches past the skull -- and a contact floating outside the
#: head reads as a mistake rather than as the schematic it is.
HEAD_FILL = 0.94


@dataclass(frozen=True)
class ContactLayout:
    """One electrode contact, placed."""

    contact: str
    shaft: str
    index: int                  #: 1-based position along the shaft
    hemisphere: str             #: "L", "R" or "?"
    region: str
    position: tuple[float, float, float]
    source: str                 #: "archive" or "inferred"


def parse_channel(name: str) -> tuple[str, int, str, str]:
    """Split a contact name into (shaft, index, hemisphere, region label).

    The convention, which holds across `ds003498` and is the one the
    preprocessing already assumes: letters then digits, the digits counting
    outward from the mesial tip, and a trailing `L` or `R` on the letters
    naming the hemisphere.

    A name that does not fit still gets an answer -- the whole alphabetic part
    becomes the shaft and the region is `unmapped` -- because refusing to place
    an unexpected electrode would silently drop channels from a view whose
    whole job is to show all of them.
    """
    text = str(name).strip()
    digits = ""
    while text and text[-1].isdigit():
        digits = text[-1] + digits
        text = text[:-1]
    shaft = text or str(name)
    index = int(digits) if digits else 1

    letters = shaft.upper().rstrip("'")
    hemisphere = letters[-1] if letters[-1:] in ("L", "R") else "?"
    stem = letters[:-1] if hemisphere != "?" else letters
    region = REGIONS.get(stem, REGIONS.get(letters, UNKNOWN))["label"]
    return shaft, index, hemisphere, region


def _inferred_position(shaft: str, index: int,
                       n_contacts: int = 1) -> tuple[float, float, float]:
    """Place contact `index` of `shaft` in the schematic head.

    The shaft runs from its structure's target outward along +x (toward the
    skull on that side), which is the direction a mesial-temporal depth
    electrode is inserted along and therefore the direction contact numbers
    increase in. Everything about this is a convention being drawn, not a
    position being reported.
    """
    letters = shaft.upper().rstrip("'")
    hemisphere = letters[-1] if letters[-1:] in ("L", "R") else "R"
    stem = letters[:-1] if letters[-1:] in ("L", "R") else letters
    region = REGIONS.get(stem, REGIONS.get(letters, UNKNOWN))

    if region is UNKNOWN:
        return _unmapped_position(stem, index, hemisphere, n_contacts)

    x, y, z = region["target"]
    x = x + (index - 1) * _pitch(n_contacts, MAX_SHAFT_LENGTH_M)
    if hemisphere == "L":
        x = -x
    # A small deterministic splay on z keeps two electrodes aimed at the same
    # structure (an `A` and an `AH` shaft, say) from drawing on top of each
    # other. Derived from the name so it never moves between runs.
    splay = (_bearing(stem) % 5 - 2) * 0.0022
    return (float(x), float(y), float(z + splay))


def _pitch(n_contacts: int, longest: float) -> float:
    """Spacing to draw a shaft of `n_contacts` at, so it fits inside the head."""
    gaps = max(1, int(n_contacts) - 1)
    return min(SHAFT_PITCH_M, longest / gaps)


def _bearing(stem: str) -> int:
    """A stable small integer for a shaft name.

    `hash()` is salted per process, so using it would move every unmapped
    electrode between runs of the same analysis. CRC32 is stable, cheap, and
    nothing here needs it to be anything more.
    """
    import zlib

    return zlib.crc32(stem.encode("utf-8")) % 997


def _unmapped_position(stem: str, index: int, hemisphere: str,
                       n_contacts: int = 1) -> tuple[float, float, float]:
    """Lay an unrecognised electrode along its own bearing over the surface.

    Used when a shaft's name matches nothing in `REGIONS`, which on a subdural
    dataset is every shaft. Each one gets an angle of its own, and its contacts
    march along the tangent at that angle -- which is how a strip or a grid row
    actually runs. The arc is centred on the right unless the name says left,
    for no better reason than that something has to be chosen; the caption says
    as much.
    """
    import math

    angle = math.radians(-UNMAPPED_SPAN_DEG / 2
                         + (_bearing(stem) / 997.0) * UNMAPPED_SPAN_DEG)
    side = -1.0 if hemisphere == "L" else 1.0
    # Start on the surface at that bearing, then step along the tangent so
    # consecutive contacts stay consecutive.
    x = side * UNMAPPED_RADIUS_M * math.cos(angle)
    y = UNMAPPED_RADIUS_M * 1.25 * math.sin(angle)
    step = (index - 1) * _pitch(n_contacts, UNMAPPED_MAX_LENGTH_M)
    return (float(x - side * step * math.sin(angle)),
            float(y + step * math.cos(angle)),
            float(0.018 + ((_bearing(stem) % 7) - 3) * 0.004))


def _contained(point) -> tuple[float, float, float]:
    """Pull a schematic position back inside the reference head.

    Only ever shrinks, and only along the ray from the centre, so the bearing
    of a shaft and the order of contacts along it both survive. Real
    coordinates never reach here: a measured position is drawn where it was
    measured, even if that is outside a cartoon skull.
    """
    axes = np.asarray(HEAD_AXES_M, dtype=float)
    scaled = np.asarray(point, dtype=float) / axes
    radius = float(np.linalg.norm(scaled))
    if radius <= HEAD_FILL or radius == 0.0:
        return tuple(float(v) for v in point)
    return tuple(float(v) for v in np.asarray(point) * (HEAD_FILL / radius))


def _from_electrodes_tsv(frame: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Real coordinates out of a BIDS `electrodes.tsv`, in metres.

    BIDS writes iEEG electrode positions in millimetres by convention, and the
    units are in the companion `coordsystem.json` rather than the table, so the
    magnitude is used to tell them apart: a head is tenths of a metre, never
    tens. Guessing wrong puts every contact a thousand times too far out, which
    is at least obvious; silently mixing the two would not be.
    """
    needed = {"name", "x", "y", "z"}
    if frame is None or not needed <= set(frame.columns):
        return {}
    positions = {}
    for row in frame.itertuples():
        try:
            point = np.array([float(row.x), float(row.y), float(row.z)])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(point).all():
            continue
        if np.abs(point).max() > 1.0:          # millimetres
            point = point / 1000.0
        positions[str(row.name).upper()] = tuple(float(v) for v in point)
    return positions


def electrode_layout(session, resection=None, electrodes: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """One row per bipolar channel, placed, ranked and zoned.

    A bipolar channel is a *pair* of contacts, so it is drawn at their midpoint
    and both contacts are named in the row. That is not a cosmetic choice: it
    is the honest geometry of the measurement, and a view that pinned a bipolar
    rate to one of its two contacts would be claiming a localisation the
    montage cannot support.

    Columns: `channel`, `contact_a`, `contact_b`, `shaft`, `hemisphere`,
    `region`, `x`/`y`/`z`, `rank`, `n_events`, `rate_per_min`, `tied` (in the
    statistically tied set), `zone` (`resected`/`partial`/`spared`/unknown),
    `reviewed`, and `source`.
    """
    findings = getattr(session, "findings", None)
    if findings is None or findings.empty:
        return pd.DataFrame(columns=["channel", "contact_a", "contact_b", "shaft",
                                     "hemisphere", "region", "x", "y", "z",
                                     "rank", "n_events", "rate_per_min", "tied",
                                     "zone", "reviewed", "source"])

    measured = _from_electrodes_tsv(electrodes) if electrodes is not None else {}
    tied = set(getattr(session, "candidates", []) or [])
    zones = _zones_for(session, resection)

    # Each shaft's length, so an unmapped one can be scaled to fit the head
    # rather than marching out of it.
    sizes: dict[str, int] = {}
    for channel in findings["channel"]:
        for contact in _contacts_of(str(channel)):
            shaft, index, _, _ = parse_channel(contact)
            sizes[shaft] = max(sizes.get(shaft, 0), index)

    rows = []
    for record in findings.to_dict("records"):
        channel = str(record["channel"])
        contacts = _contacts_of(channel)
        places, sources = [], []
        for contact in contacts:
            if contact.upper() in measured:
                places.append(measured[contact.upper()])
                sources.append("archive")
            else:
                shaft, index, _, _ = parse_channel(contact)
                places.append(_contained(
                    _inferred_position(shaft, index, sizes.get(shaft, 1))))
                sources.append("inferred")
        centre = np.mean(np.asarray(places, dtype=float), axis=0)
        shaft, index, hemisphere, region = parse_channel(contacts[0])
        rows.append({
            "channel": channel,
            "contact_a": contacts[0],
            "contact_b": contacts[-1] if len(contacts) > 1 else "",
            "shaft": shaft,
            "index": index,
            "hemisphere": hemisphere,
            "region": region,
            "x": float(centre[0]), "y": float(centre[1]), "z": float(centre[2]),
            "rank": int(record.get("rank", 0) or 0),
            "n_events": int(record.get("n_events", 0) or 0),
            "rate_per_min": float(record.get("rate_per_min", 0.0) or 0.0),
            "tied": channel in tied,
            "zone": zones.get(channel, "unknown"),
            "reviewed": bool(record.get("reviewed", False)),
            # "inferred" wins a mixed pair: a midpoint is only as trustworthy
            # as its worse end.
            "source": "archive" if set(sources) == {"archive"} else "inferred",
        })
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _contacts_of(channel: str) -> list[str]:
    """The physical contacts behind a channel name (`AR1-AR2` -> both)."""
    parts = [p for p in str(channel).split("-") if p]
    return parts if parts else [str(channel)]


def _zones_for(session, resection) -> dict[str, str]:
    """Each channel's position relative to the resection, when that is known.

    Guarded end to end: the clinical sheet is a separate download, only
    `ds003498` has one, and a 3D view that refuses to draw because a surgical
    sidecar is missing would be useless on every other dataset.
    """
    if resection is None:
        return {}
    try:
        from onset_hfo.clinical import classify_channels

        table = classify_channels(list(session.findings["channel"]), resection)
    except Exception:
        return {}
    if table is None or table.empty or "zone" not in table.columns:
        return {}
    return {str(r.channel): str(r.zone) for r in table.itertuples()}


def layout_caption(layout: pd.DataFrame) -> str:
    """The sentence that has to sit under any drawing of this layout.

    Returned from here rather than written into the panel so that the window,
    the exported report and any future view all carry the same words, and so
    that the claim changes automatically the day a dataset arrives with real
    coordinates.
    """
    if layout.empty:
        return "No channels to place."
    sources = set(layout["source"])
    if sources == {"archive"}:
        return ("Contact positions are the dataset's own measured coordinates "
                "(electrodes.tsv).")
    if "archive" in sources:
        n = int((layout["source"] == "inferred").sum())
        return (f"Measured coordinates, except for {n} channel(s) placed from "
                f"their electrode names. Those are schematic.")
    unmapped = float((layout["region"] == UNKNOWN["label"]).mean())
    if unmapped > 0.5:
        return (f"MONTAGE DIAGRAM — {NOT_ANATOMY}, and no anatomy at all. This "
                "archive ships no electrode coordinates, and these electrode "
                "names match no structure this software knows, so the only "
                "real information here is which contacts share a shaft and "
                "their order along it. Shafts are fanned out so they can be "
                "told apart; their positions and sides mean nothing.")
    return (f"SCHEMATIC LAYOUT — {NOT_ANATOMY}. This archive ships no electrode "
            "coordinates, so each shaft is drawn at the textbook location of "
            "the structure its name claims, contacts in order along it. Use it "
            "to see which shafts are active and how they sit relative to the "
            "resection; do not read a position off it.")


def brain_surface(n_theta: int = 48, n_phi: int = 32):
    """A schematic brain hull: vertices and triangular faces, in metres.

    Deliberately an obvious cartoon rather than a rendered cortex. A realistic
    surface under schematic contacts would imply the contacts are registered to
    it, and they are not. This is a smoothed ellipsoid of roughly adult-head
    proportion with a sagittal cleft and a flattened base -- enough to read
    left/right, front/back and up/down at a glance, which is all a reference
    surface is doing here.

    Returned as plain arrays so the renderer can be anything.
    """
    theta = np.linspace(0, np.pi, n_phi)            # polar, 0 = top
    phi = np.linspace(0, 2 * np.pi, n_theta)        # azimuth
    theta, phi = np.meshgrid(theta, phi, indexing="ij")

    # Wider than tall, longer than wide, as a head is.
    ax, ay, az = HEAD_AXES_M
    x = ax * np.sin(theta) * np.cos(phi)
    y = ay * np.sin(theta) * np.sin(phi)
    z = az * np.cos(theta)

    # Occiput fuller than the forehead, and the underside flattened where the
    # skull base is, so the orientation is readable without an axis label.
    y = y - 0.010 * (z / az) ** 2
    z = np.where(z < 0, z * (0.72 + 0.28 * np.abs(y) / ay), z)
    # The interhemispheric fissure: a shallow crease on the midline.
    x = x * (1.0 - 0.18 * np.exp(-((x / ax) ** 2) / 0.02) * (z > 0))

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            a = i * n_theta + j
            b = a + 1
            c = a + n_theta
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return vertices, np.asarray(faces, dtype=int)
