"""The figures' shared style: two hues, ink, and quiet axes. No Qt.

Every chart the window draws -- the case steps, the batch, the comparison,
the report -- takes its colours from here, so they read as one set. Two hues
only, validated for colour-vision deficiency and for contrast on the page's
white: the window's indigo accent for the first series and an orange for the
second. Everything else is ink: text in the text colours, grid and axes
recessive, no title inside the figure when the page heading above it is the
title.
"""

from __future__ import annotations

from onset_hfo.chartstyle import SERIES

__all__ = ["SERIES", "DETECTOR_LABELS", "MARK", "MUTED", "CURSOR", "GRID"]

#: A mark that is not a series (a recording's own point, a threshold): ink,
#: not a third hue, so it is not mistaken for one.
MARK = "#1D1D1F"
MUTED = "#6E6E73"
#: The moment and the "has changed" mark on a magma heatmap or map (the ictal
#: progression): a cyan that stands out on magma's black and on its yellow.
CURSOR = "#22D3EE"
GRID = "#E5E5EA"

DETECTOR_LABELS = {"rms": "RMS energy", "line_length": "Line length",
                   "hilbert": "Hilbert envelope", "energy": "Short-time energy"}


def _quiet(axis) -> None:
    """Recessive axes: no top or right spine, grey left and bottom, a light
    horizontal grid, muted tick labels."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=3)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


