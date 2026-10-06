"""The 3D view: ranked contacts in a head, with the resection shown.

What a reviewer gets from this that no table gives them is *adjacency*. A
findings table says `AR1-AR2` and `AR2-AR3` lead; it does not say they are two
neighbouring contacts of one shaft, which is a completely different finding
from two leaders on opposite sides of the head. Shafts drawn as shafts make
that immediate.

Rendered with Matplotlib rather than OpenGL or a dedicated 3D toolkit. The
scene is fifty spheres and a hull, so there is nothing here that needs a GPU,
and in exchange it renders identically on a laptop, over X11 forwarding, and
headless into a PNG for the documentation and the test suite. A 3D view that
fails to start on a hospital machine with an unexceptional graphics driver
would be worse than no 3D view.

**The honesty constraint is the design constraint.** `anatomy.py` explains it:
neither archive ships electrode coordinates, so the positions are inferred from
the electrode names. This panel therefore carries `layout_caption` permanently,
draws its reference surface as an obvious cartoon rather than a rendered
cortex, and titles itself "schematic" whenever any contact is inferred. None of
that is optional decoration -- a picture of contacts on a brain is read as an
implantation plan unless it works hard not to be.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme
from onset_review.anatomy import ARCHIVE_ORIGIN, UNKNOWN, electrode_layout, layout_caption
from onset_review.theme import card

__all__ = ["BrainPanel", "VIEWS", "ZONE_EDGES"]

#: Preset camera angles, as (elevation, azimuth). A reviewer wants "show me the
#: right side" far more often than they want to find it by dragging, and the
#: mesial-temporal shafts this cohort carries are only legible from a handful
#: of angles.
VIEWS = {
    "Right": (8, 0),
    "Left": (8, 180),
    "Top": (88, -90),
    "Front": (6, -90),
    "Back": (6, 90),
    "Oblique": (22, -52),
}

#: Ring colour by position relative to the surgeon's resection. Drawn as an
#: outline rather than a fill so it cannot be confused with the rate colour,
#: which is the thing the fill means.
ZONE_EDGES = {
    "resected": "#11a011",
    "partial": "#e08a00",
    "spared": "#303030",
    "unknown": "#9a9a9a",
}


class BrainPanel(QWidget):
    """Contacts in three dimensions, coloured by rate and numbered by rank."""

    channelPicked = Signal(str)

    def __init__(self, session, resection=None, electrodes=None, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._resection = resection
        #: Where the measured coordinates, if any, came from. Said in the
        #: caption, because "the dataset's own coordinates" is false the moment
        #: a reviewer supplies a file of their own.
        self._origin = ARCHIVE_ORIGIN
        self.layout_frame = electrode_layout(session, resection=resection,
                                             electrodes=electrodes)

        self.colour_by = QComboBox()
        self.colour_by.addItem("Colour by rate", "rate_per_min")
        self.colour_by.addItem("Colour by rank", "rank")
        self.labels = QCheckBox("Rank numbers")
        self.labels.setChecked(True)
        self.labels.setToolTip("Number the busiest contacts in rank order.")
        self.shafts = QCheckBox("Shafts")
        self.shafts.setChecked(True)
        self.shafts.setToolTip(
            "Join the contacts of each electrode. Two leaders on one shaft is "
            "a different finding from two leaders across the head.")
        self.resection_only = QCheckBox("Resection")
        self.resection_only.setChecked(bool(
            (self.layout_frame["zone"] != "unknown").any()
            if not self.layout_frame.empty else False))
        self.resection_only.setEnabled(self.resection_only.isChecked())
        self.resection_only.setToolTip(
            "Ring each contact by whether the surgeon removed it"
            if self.resection_only.isEnabled()
            else "No resection record for this dataset")

        bar = QHBoxLayout()
        bar.addWidget(self.colour_by)
        bar.addWidget(self.labels)
        bar.addWidget(self.shafts)
        bar.addWidget(self.resection_only)
        bar.addSpacing(8)
        # A combo rather than one button per angle. Six buttons set a minimum
        # width of about 900 px for this panel, which in a docked column is a
        # floor for the whole window; the trace loses whatever this takes.
        bar.addWidget(QLabel("View"))
        self.view = QComboBox()
        for name in VIEWS:
            self.view.addItem(name, name)
        self.view.setCurrentText("Oblique")
        self.view.currentTextChanged.connect(self.set_view)
        bar.addWidget(self.view)
        bar.addStretch(1)

        # `add_axes` over the whole canvas rather than `add_subplot` with a
        # constrained layout: Matplotlib reserves a subplot's margins for tick
        # labels a 3D axes with its frame off never draws, and the scene ends
        # up occupying about a third of the canvas.
        self.headline = QLabel()
        self.headline.setWordWrap(True)
        # Maximum, not Minimum: the two captions take the height their text
        # needs and no more, so spare height goes to the scene. With Minimum
        # the caption grew to fill a page and the scene sat in a strip.
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.headline.setStyleSheet(
            f"font-size:9pt;padding:1px 4px;color:{theme.current().text};")

        # Matplotlib does not see the Qt stylesheet, so a dark window would
        # otherwise hold one bright white rectangle. Colours come from the same
        # palette as everything else.
        tokens = theme.current()
        # A short default figure, not because the scene wants to be short --
        # it takes the whole dock when there is room -- but because `figsize`
        # is where the canvas's size *hint* comes from, and that hint is the
        # height the scroll area lays the panel out at. A 4.2 in hint made the
        # panel 554 px tall inside a 144 px dock on a laptop screen, so the
        # visible slice was all chrome and no scene.
        self.figure = Figure(figsize=(5.2, 2.6), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(240)
        # Low enough to live in a docked column beneath two tables. The panel
        # is worth more space than this and says so by being floatable: one
        # drag gives it a window.
        self.canvas.setMinimumHeight(150)
        self.axes = self.figure.add_axes((-0.05, -0.14, 0.99, 1.26),
                                         projection="3d")
        self.axes.set_facecolor(tokens.surface)
        self._colorbar = None
        self._points = None

        self.caption = QLabel(layout_caption(self.layout_frame,
                                             self._origin))
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        schematic = (not self.layout_frame.empty
                     and "inferred" in set(self.layout_frame["source"]))
        self.caption.setStyleSheet(card("warn" if schematic else "info"))

        # The contents go in a scroll area rather than straight on the panel.
        # Docked, this is the tallest thing in the right-hand column -- a row
        # of controls, a wrapped headline, a 3D scene and a wrapped caption --
        # and a main window takes its minimum height from the sum of its
        # column, so without this the window could not shrink to a laptop
        # screen. See `theme.scrolled`.
        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addLayout(bar)
        # The scene first, the two lines of text under it as a figure caption.
        # Reading order aside, this is what the panel looks like when it is
        # squeezed: the scroll area shows the top of the body, so whatever is
        # first is what a reviewer sees. With the headline above, a short dock
        # showed a row of controls, a line of text and then blank canvas --
        # which looks like a panel that failed to draw rather than one that
        # needs more room.
        box.addWidget(self.canvas, 1)
        box.addWidget(self.headline)
        box.addWidget(self.caption)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))

        self.colour_by.currentIndexChanged.connect(self.redraw)
        for toggle in (self.labels, self.shafts, self.resection_only):
            toggle.stateChanged.connect(self.redraw)
        self.canvas.mpl_connect("pick_event", self._picked)

        self._elev, self._azim = VIEWS["Oblique"]
        if not self.layout_frame.empty:
            # Open facing the hemisphere the busiest contact is in.
            self._azim = -40 if float(
                self.layout_frame.iloc[0]["x"]) >= 0 else -140
        self.redraw()

    def set_electrodes(self, electrodes, origin: str = ARCHIVE_ORIGIN) -> None:
        """Re-place every contact from a coordinate table, and redraw.

        Rebuilding the layout rather than nudging the points: which hemisphere
        a contact is on, which shaft it belongs to and whether it is inside the
        resection are all derived alongside its position, and a view that moved
        the dots without re-deriving the rest would be drawing a different
        patient's geometry with this one's labels.
        """
        self._origin = origin
        self.layout_frame = electrode_layout(self._session,
                                             resection=self._resection,
                                             electrodes=electrodes)
        self.caption.setText(layout_caption(self.layout_frame, self._origin))
        schematic = (not self.layout_frame.empty
                     and "inferred" in set(self.layout_frame["source"]))
        self.caption.setStyleSheet(card("warn" if schematic else "info"))
        if not self.layout_frame.empty:
            self._azim = -40 if float(
                self.layout_frame.iloc[0]["x"]) >= 0 else -140
        self.redraw()

    # -- drawing -----------------------------------------------------------
    def set_view(self, name: str) -> None:
        self._elev, self._azim = VIEWS.get(name, VIEWS["Oblique"])
        self.axes.view_init(elev=self._elev, azim=self._azim)
        self.canvas.draw_idle()

    def redraw(self) -> None:
        """Rebuild the scene. Cheap enough at this size to not bother diffing."""
        frame = self.layout_frame
        self.axes.clear()
        self.axes.set_facecolor(theme.current().surface)
        if self._colorbar is not None:
            try:
                self._colorbar.ax.remove()
            except Exception:
                pass
            self._colorbar = None

        self._draw_head()
        if frame.empty:
            self.axes.set_title("No channels to place")
            self.canvas.draw_idle()
            return

        if self.shafts.isChecked():
            self._draw_shafts(frame)

        field = str(self.colour_by.currentData() or "rate_per_min")
        values = frame[field].to_numpy(dtype=float)
        if field == "rank":
            # Rank 1 is the interesting end, so invert it before colouring:
            # otherwise the busiest contact is drawn in the "nothing here"
            # colour and the quietest glows.
            values = float(values.max()) + 1.0 - values
        sizes = 42.0 + 330.0 * _unit(frame["rate_per_min"].to_numpy(dtype=float))
        edges = ([ZONE_EDGES.get(z, ZONE_EDGES["unknown"]) for z in frame["zone"]]
                 if self.resection_only.isChecked()
                 else theme.current().text_muted)

        self._points = self.axes.scatter(
            frame["x"], frame["y"], frame["z"], c=values, s=sizes,
            cmap="inferno", edgecolors=edges,
            linewidths=1.8 if self.resection_only.isChecked() else 0.6,
            depthshade=False, picker=6, zorder=5)

        if self.labels.isChecked():
            self._draw_ranks(frame)

        tokens = theme.current()
        cax = self.figure.add_axes((0.93, 0.18, 0.018, 0.56))
        bar = self.figure.colorbar(self._points, cax=cax)
        bar.ax.yaxis.set_tick_params(color=tokens.text_muted,
                                     labelcolor=tokens.text_muted)
        bar.outline.set_edgecolor(tokens.separator)
        bar.set_label("events / min" if field == "rate_per_min"
                      else "rank (1 = busiest)", fontsize=8,
                      color=tokens.text_muted)
        bar.ax.tick_params(labelsize=7)
        if field == "rank":
            top = float(frame["rank"].max())
            ticks = sorted({1.0, top})
            bar.set_ticks([top + 1.0 - t for t in ticks])
            bar.set_ticklabels([f"{int(t)}" for t in ticks])
        self._colorbar = bar

        self._finish_axes(frame)
        self.canvas.draw_idle()

    def _draw_head(self) -> None:
        from onset_review.anatomy import brain_surface

        vertices, faces = brain_surface()
        self._head_bounds = (vertices.min(axis=0), vertices.max(axis=0))
        # Faint edges as well as a translucent fill. A smooth low-alpha surface
        # with no edges renders as a flat disc from every angle, which tells a
        # reviewer nothing about which way they are looking.
        tokens = theme.current()
        self.axes.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces,
            color=tokens.separator, alpha=0.14 if tokens.dark else 0.10,
            linewidth=0.12, edgecolor=tokens.text_muted, shade=True, zorder=0)

    def _draw_shafts(self, frame) -> None:
        for _, rows in frame.groupby("shaft"):
            rows = rows.sort_values("index")
            if len(rows) < 2:
                continue
            self.axes.plot(rows["x"], rows["y"], rows["z"],
                           color=theme.current().text_muted,
                           linewidth=1.0, alpha=0.55, zorder=3)

    def _draw_ranks(self, frame) -> None:
        """Number the contacts a reviewer is actually going to look at.

        Only the leading handful, and only those: numbering all forty-three
        makes a cloud of text no one can read, and the ranks that matter are
        the ones at the top of the table anyway. Tied contacts are marked
        rather than numbered in isolation, because the tie is the finding.
        """
        from matplotlib import patheffects

        shown = frame.nsmallest(min(8, len(frame)), "rank")
        tokens = theme.current()
        # The halo is the panel's own background, not white: on a dark theme a
        # white outline round dark text makes a bright smear.
        halo = [patheffects.withStroke(linewidth=2.6,
                                       foreground=tokens.surface)]
        for position, row in enumerate(shown.itertuples()):
            text = f"{row.rank}" + ("*" if row.tied else "")
            # Alternate the label above and below the contact. Contacts 5 mm
            # apart on one shaft project to within a few pixels of each other,
            # and a column of numbers on top of one another is worse than none.
            lift = 0.0075 if position % 2 == 0 else -0.0085
            self.axes.text(row.x, row.y, row.z + lift, text, fontsize=9,
                           fontweight="bold", color=tokens.text, ha="center",
                           va="center", zorder=6, path_effects=halo)

    def _finish_axes(self, frame) -> None:
        axes = self.axes
        axes.view_init(elev=self._elev, azim=self._azim)
        low, high = getattr(self, "_head_bounds", (None, None))
        if low is not None:
            pad = 0.004
            axes.set_xlim(low[0] - pad, high[0] + pad)
            axes.set_ylim(low[1] - pad, high[1] + pad)
            axes.set_zlim(low[2] - pad, high[2] + pad)
            extent = (high - low)
            axes.set_box_aspect(tuple(extent / extent.max()))
        try:                      # Matplotlib >= 3.8: fill the axes box
            axes.set_proj_type("persp", focal_length=0.72)
        except (TypeError, ValueError, AttributeError):
            pass
        axes.set_xlabel("L — R", fontsize=8, labelpad=-6)
        axes.set_ylabel("post — ant", fontsize=8, labelpad=-6)
        axes.set_zlabel("inf — sup", fontsize=8, labelpad=-6)
        for ticks in (axes.set_xticks, axes.set_yticks, axes.set_zticks):
            ticks([])
        axes.grid(False)
        try:
            axes.set_axis_off()
        except Exception:
            pass

        tied = int(frame["tied"].sum())
        leader = frame.iloc[0]
        title = (f"{leader.channel} leads at {leader.rate_per_min:g}/min "
                 f"({leader.region}, {'right' if leader.hemisphere == 'R' else 'left' if leader.hemisphere == 'L' else 'side unknown'})")
        if tied > 1:
            title += f" — {tied} contacts tied (*)"
        if "inferred" in set(frame["source"]):
            # Match the caption's own verdict rather than saying "schematic"
            # over a caption that says "no anatomy at all": two different
            # strengths of warning on one panel reads as one of them being
            # boilerplate.
            unmapped = float((frame["region"] == UNKNOWN["label"]).mean())
            title = ("Montage diagram · " if unmapped > 0.5
                     else "Schematic layout · ") + title
        self.headline.setText(title)

        if self.resection_only.isChecked():
            from matplotlib.lines import Line2D

            present = [z for z in ("resected", "partial", "spared")
                       if (frame["zone"] == z).any()]
            handles = [Line2D([], [], marker="o", linestyle="none",
                              markerfacecolor="none", markeredgewidth=1.8,
                              markeredgecolor=ZONE_EDGES[z], label=z)
                       for z in present]
            if handles:
                # On the figure, not the axes: a 3D axes legend is positioned
                # against the axes box, which now extends off the canvas.
                legend = self.figure.legend(
                    handles=handles, loc="lower left", fontsize=7, frameon=False,
                    title="surgeon removed", title_fontsize=7,
                    bbox_to_anchor=(0.01, 0.02),
                    labelcolor=theme.current().text_muted)
                legend.get_title().set_color(theme.current().text_muted)

    # -- interaction --------------------------------------------------------
    def _picked(self, event) -> None:
        if self._points is None or event.artist is not self._points:
            return
        indices = list(getattr(event, "ind", []) or [])
        if not indices:
            return
        channel = str(self.layout_frame.iloc[int(indices[0])]["channel"])
        self.channelPicked.emit(channel)

    def highlight(self, channel: str) -> None:
        """Turn the camera to a channel chosen somewhere else in the window.

        Rotating to face the contact rather than drawing a marker on it: in a
        projection, a highlighted point behind the head is indistinguishable
        from one in front of it, and the reviewer would be looking at the wrong
        hemisphere without knowing.
        """
        frame = self.layout_frame
        row = frame[frame["channel"] == channel]
        if row.empty:
            return
        self._azim = 0 if float(row.iloc[0]["x"]) >= 0 else 180
        self._elev = 8
        self.axes.view_init(elev=self._elev, azim=self._azim)
        self.canvas.draw_idle()


def _unit(values: np.ndarray) -> np.ndarray:
    """Scale to 0–1 for marker size, flat when every value is the same."""
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 0:
        return np.full(values.shape, 0.4)
    return (values - float(values.min())) / span
