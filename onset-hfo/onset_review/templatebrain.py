"""Contacts on nilearn's template brain: the glass brain and a 3D view. Qt-free.

When `nilearn` is installed (it comes with the review extra), the Map step
and the report draw a case's contacts the way iEEG papers do: on nilearn's
**glass brain**, the MNI152 template seen from the left, from above and from
the right, every contact as a dot sized by its interictal rate and shaded by
its ictal index, the onset-zone contacts ringed. `write_3d_view` writes the
same contacts as a rotatable 3D page (nilearn's `view_markers`, self-contained,
no network) to open in a browser. Nilearn also ships fsaverage5's surface,
which `onset_review.anatomy.template_surface` falls back to when MNE's
fsaverage has not been fetched.

Without nilearn everything here returns False or None and the callers keep
their own drawing. The positions are template positions in every case, and
the captions say so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["available", "draw_glass", "write_3d_view", "GLASS_CAPTION", "glass_caption"]

GLASS_CAPTION = ("glass brain (MNI152, nilearn) · template positions, approximate\n"
                 "size: rate · shade: ictal index · ring: onset zone")


_3D_TITLES = {"template": "Template positions, approximate — not this patient's anatomy",
              "patient": "The patient's MRI positions, registered to MNI, shown on "
                         "the template",
              "mixed": "The patient's MRI positions and template positions, shown on the "
                       "template"}


def glass_caption(kind: str = "template") -> str:
    """The caption, saying where the positions came from (`electrodes.CAPTIONS`)."""
    from onset_hfo.case.electrodes import CAPTIONS

    return GLASS_CAPTION.replace(CAPTIONS["template"], CAPTIONS.get(kind, CAPTIONS["template"]))


def available() -> bool:
    return importlib.util.find_spec("nilearn") is not None


def _placed(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["placed"].astype(bool)] if len(table) else table


def draw_glass(figure, table: pd.DataFrame, rect=(0.0, 0.0, 1.0, 1.0)):
    """The placed channels on the glass brain, in `rect` of `figure`
    (left, bottom, width, height in figure fractions). Returns nilearn's
    display, or None when there is nothing to draw or no nilearn."""
    placed = _placed(table)
    if not available() or not len(placed):
        return None
    from nilearn import plotting

    from onset_review.studycharts import SERIES

    coords = placed[["x", "y", "z"]].to_numpy(float)
    rate = placed["rate_per_min"].fillna(0.0).to_numpy(float)
    top = max(float(rate.max()), 1e-9)
    sizes = 8.0 + 50.0 * rate / top
    shade = placed["median_ei"].fillna(0.0).clip(0, 1).to_numpy(float)
    display = plotting.plot_markers(shade, coords, node_size=sizes, node_cmap="viridis",
                                    node_vmin=0.0, node_vmax=1.0, display_mode="lzr",
                                    figure=figure, axes=list(rect), colorbar=False,
                                    alpha=0.85)
    soz = placed["soz"].astype(bool).to_numpy()
    if soz.any():
        # A ring, not a dot: nilearn hands `marker_color` to matplotlib as `c`,
        # so the face is made transparent there, one colour per marker.
        display.add_markers(coords[soz], marker_color=[(0.0, 0.0, 0.0, 0.0)] * int(soz.sum()),
                            marker_size=sizes[soz] + 14, edgecolors=SERIES[1],
                            linewidths=0.9)
    return display


def write_3d_view(path, table: pd.DataFrame, positions: str = "template") -> Path | None:
    """The placed channels as a rotatable 3D page, labelled with each
    channel's name, rate, index and probable structure. None when there is
    nothing to show or no nilearn."""
    placed = _placed(table)
    if not available() or not len(placed):
        return None
    from nilearn import plotting

    from onset_review.studycharts import MUTED, SERIES

    labels = []
    for row in placed.itertuples():
        rate = "–" if pd.isna(row.rate_per_min) else f"{row.rate_per_min:.1f}/min"
        index = "–" if pd.isna(row.median_ei) else f"{row.median_ei:.2f}"
        labels.append(f"{row.channel}: rate {rate}, index {index}"
                      + (f", probably {row.where}" if row.where else "")
                      + (" — onset zone" if row.soz else ""))
    rate = placed["rate_per_min"].fillna(0.0).to_numpy(float)
    sizes = 4.0 + 8.0 * rate / max(float(rate.max()), 1e-9)
    colours = [SERIES[1] if s else (SERIES[0] if (e or 0) >= 0.3 else MUTED)
               for s, e in zip(placed["soz"], placed["median_ei"].fillna(0.0), strict=True)]
    view = plotting.view_markers(placed[["x", "y", "z"]].to_numpy(float),
                                 marker_color=colours, marker_size=list(np.round(sizes, 1)),
                                 marker_labels=labels,
                                 title=_3D_TITLES.get(positions, _3D_TITLES["template"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view.save_as_html(str(path))
    return path
