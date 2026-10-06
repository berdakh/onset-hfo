"""The Map page: contacts flat, coloured by rate, with a colour scale.

The picture a paper prints and the 3D view does not: every contact a dot,
coloured by what it measured, a colour bar beside it, the leading contacts
numbered, the resection ringed, and a button that saves it as a PNG. The
selected channel -- picked in the findings table, the event list, the trace
or here -- is drawn with a halo, so a reviewer looking at one event can see
where it is without leaving the page.

Everything drawn comes from :mod:`onset_review.contactmap`, which is Qt-free
and tested without a display; this widget only draws. The caption under the
figure is :func:`onset_review.anatomy.layout_caption`, the same sentence the
3D view and the report carry, and it says plainly when the positions are
schematic. No rendered cortex, for the reason given in `contactmap`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onset_review import contactmap, theme
from onset_review.anatomy import ARCHIVE_ORIGIN, layout_caption
from onset_review.brainview import ZONE_EDGES
from onset_review.theme import card

__all__ = ["ContactMapPanel", "WHERE_QUESTION"]

#: What the "Ask where" button asks the assistant. One question, fixed, so
#: that the map and the model are talking about the same thing.
WHERE_QUESTION = "Where on the head is the activity, and on how many electrodes?"


class ContactMapPanel(QWidget):
    """Contacts flat, coloured by a measure, one channel highlighted."""

    #: A channel the reviewer clicked on the map.
    channelPicked = Signal(str)
    #: The reviewer wants the assistant asked this, about this map.
    askRequested = Signal(str)

    def __init__(self, session, resection=None, electrodes=None, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._resection = resection
        self._electrodes = electrodes
        self._origin = ARCHIVE_ORIGIN
        self._highlighted: str | None = None
        self._points = None
        self._colorbar = None
        self.frame = contactmap.map_frame(session, resection=resection,
                                          electrodes=electrodes)

        self.measure = QComboBox()
        self.measure.setObjectName("onset_map_measure")
        for key, label, _unit in contactmap.MEASURES:
            if key == "expert" and not list(getattr(session, "expert", []) or []):
                continue
            self.measure.addItem(label, key)
        self.view = QComboBox()
        self.view.setObjectName("onset_map_view")
        for name in contactmap.VIEWS:
            self.view.addItem(name, name)
        self.labels = QCheckBox("Rank numbers")
        self.labels.setChecked(True)
        self.rings = QCheckBox("Resection")
        has_zones = bool((self.frame["zone"] != "unknown").any()) if not self.frame.empty else False
        self.rings.setChecked(has_zones)
        self.rings.setEnabled(has_zones)
        self.rings.setToolTip("Ring each contact by whether the surgeon removed it"
                              if has_zones else "No resection record for this dataset")
        self.save = QPushButton("Save as PNG…")
        self.save.setObjectName("onset_map_save")
        self.save.setToolTip("The figure as drawn, at print resolution")
        self.save.clicked.connect(lambda _=False: self.save_png())
        self.ask = QPushButton("Ask the assistant where")
        self.ask.setObjectName("onset_map_ask")
        self.ask.setToolTip(WHERE_QUESTION)
        self.ask.clicked.connect(lambda _=False: self.askRequested.emit(WHERE_QUESTION))

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("Colour by"))
        bar.addWidget(self.measure)
        bar.addWidget(QLabel("View"))
        bar.addWidget(self.view)
        bar.addWidget(self.labels)
        bar.addWidget(self.rings)
        bar.addStretch(1)
        bar.addWidget(self.ask)
        bar.addWidget(self.save)

        tokens = theme.current()
        self.figure = Figure(figsize=(6.4, 5.2), facecolor=tokens.surface, dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Short on purpose: a docked column's minimum is the sum of its panels'
        # minimums, and this panel shares a tab stack with the 3D view. The
        # figure takes whatever height the page gives it.
        self.canvas.setMinimumHeight(150)
        self.canvas.setMinimumWidth(240)
        self.axes = self.figure.add_axes((0.04, 0.05, 0.80, 0.90))
        self.canvas.mpl_connect("pick_event", self._picked)

        self.headline = QLabel("")
        self.headline.setObjectName("onset_map_headline")
        self.headline.setWordWrap(True)
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.headline.setStyleSheet(f"font-size:10pt;padding:1px 4px;color:{tokens.text};")
        self.caption = QLabel("")
        self.caption.setObjectName("onset_map_caption")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addLayout(bar)
        box.addWidget(self.canvas, 1)
        box.addWidget(self.headline)
        box.addWidget(self.caption)
        # Scrolled, like every other panel that can be docked: the toolbar
        # row and the caption must not put a floor under the whole window.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))

        for widget in (self.measure, self.view):
            widget.currentIndexChanged.connect(lambda _i: self.redraw())
        for widget in (self.labels, self.rings):
            widget.stateChanged.connect(lambda _s: self.redraw())
        self.redraw()

    # -- data ---------------------------------------------------------------
    def set_electrodes(self, electrodes, origin: str = ARCHIVE_ORIGIN) -> None:
        """Measured coordinates arrived (from the dataset or a file): place
        the contacts from them and say where they came from."""
        self._electrodes = electrodes
        self._origin = origin
        self.redraw()

    def measure_key(self) -> str:
        return str(self.measure.currentData() or "primary")

    # -- drawing --------------------------------------------------------------
    def redraw(self) -> None:
        tokens = theme.current()
        self.frame = contactmap.map_frame(self._session, resection=self._resection,
                                          electrodes=self._electrodes,
                                          measure=self.measure_key())
        frame = self.frame
        self.axes.clear()
        self.axes.set_facecolor(tokens.surface)
        if self._colorbar is not None:
            try:
                self._colorbar.ax.remove()
            except Exception:       # noqa: BLE001
                pass
            self._colorbar = None
        view = str(self.view.currentData() or "Top")

        outline_h, outline_v = contactmap.head_outline(view)
        self.axes.plot(outline_h, outline_v, color=tokens.separator, linewidth=1.2, zorder=1)
        self.axes.fill(outline_h, outline_v, color=tokens.surface_alt, alpha=0.5, zorder=0)

        if frame.empty:
            self.axes.set_title("No channels to place", color=tokens.text_muted)
            self.caption.setText(layout_caption(frame, self._origin))
            self.caption.setStyleSheet(card("info"))
            self._finish(view)
            self.canvas.draw_idle()
            return

        h, v = contactmap.project(frame, view)
        values = frame["value"].to_numpy(dtype=float)
        key = self.measure_key()
        if key == "rank":
            values = float(values.max()) + 1.0 - values
        sizes = 40.0 + 300.0 * _unit(values)
        edges = ([ZONE_EDGES.get(z, ZONE_EDGES["unknown"]) for z in frame["zone"]]
                 if self.rings.isChecked() else tokens.text_muted)
        self._points = self.axes.scatter(
            h, v, c=values, s=sizes, cmap="inferno", edgecolors=edges,
            linewidths=1.8 if self.rings.isChecked() else 0.6, picker=6, zorder=5)

        if self._highlighted and (frame["channel"] == self._highlighted).any():
            i = int(np.flatnonzero((frame["channel"] == self._highlighted).to_numpy())[0])
            self.axes.scatter([h[i]], [v[i]], s=float(sizes[i]) * 3.2, facecolors="none",
                              edgecolors=tokens.accent, linewidths=2.2, zorder=6)
            self.axes.annotate(self._highlighted, (h[i], v[i]), xytext=(8, 8),
                               textcoords="offset points", fontsize=9, fontweight="bold",
                               color=tokens.accent, zorder=7)

        if self.labels.isChecked():
            self._draw_ranks(frame, h, v)

        unit = next(u for k, _l, u in contactmap.MEASURES if k == key)
        cax = self.figure.add_axes((0.87, 0.15, 0.025, 0.6))
        bar = self.figure.colorbar(self._points, cax=cax)
        bar.ax.yaxis.set_tick_params(color=tokens.text_muted, labelcolor=tokens.text_muted)
        bar.outline.set_edgecolor(tokens.separator)
        bar.set_label(unit, fontsize=8, color=tokens.text_muted)
        bar.ax.tick_params(labelsize=7)
        if key == "rank":
            top = float(frame["rank"].max())
            bar.set_ticks([top, 1.0])
            bar.set_ticklabels([f"{int(top)}", "1"])
        self._colorbar = bar

        self._finish(view)
        self.caption.setText(layout_caption(frame, self._origin))
        self.caption.setStyleSheet(card("warn" if "SCHEMATIC" in self.caption.text()
                                        or "DIAGRAM" in self.caption.text() else "info"))
        self._say_headline(frame)
        self.canvas.draw_idle()

    def _draw_ranks(self, frame, h, v) -> None:
        from matplotlib import patheffects

        tokens = theme.current()
        halo = [patheffects.withStroke(linewidth=2.6, foreground=tokens.surface)]
        order = np.argsort(frame["rank"].to_numpy())[:8]
        for position, i in enumerate(order):
            row = frame.iloc[int(i)]
            text = f"{int(row['rank'])}" + ("*" if bool(row["tied"]) else "")
            lift = 7 if position % 2 == 0 else -9
            self.axes.annotate(text, (h[i], v[i]), xytext=(0, lift),
                               textcoords="offset points", fontsize=9, fontweight="bold",
                               color=tokens.text, ha="center", va="center", zorder=6,
                               path_effects=halo)

    def _finish(self, view: str) -> None:
        tokens = theme.current()
        self.axes.set_aspect("equal", adjustable="datalim")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for side in self.axes.spines.values():
            side.set_visible(False)
        schematic = (not self.frame.empty) and (self.frame["source"] != "archive").any()
        title = f"{view} view" + (" — schematic layout" if schematic else "")
        self.axes.set_title(title, fontsize=9, color=tokens.text_muted, loc="left")

    def _say_headline(self, frame) -> None:
        sides = frame["hemisphere"].map({"L": "left", "R": "right"}).fillna("unknown")
        summary = contactmap.where_summary(frame.assign(hemisphere=sides))
        text = summary.get("summary", "")
        if self._highlighted and (frame["channel"] == self._highlighted).any():
            row = frame[frame["channel"] == self._highlighted].iloc[0]
            side = {"L": "left", "R": "right"}.get(str(row["hemisphere"]), "side unknown")
            where = f"{row['region']}, {side}" if row["region"] else side
            text = (f"<b>{self._highlighted}</b> — rank {int(row['rank'])}, "
                    f"{float(row['rate_per_min']):.1f}/min, shaft {row['shaft']} ({where}). "
                    + text)
        self.headline.setText(text)

    # -- interaction ----------------------------------------------------------
    def _picked(self, event) -> None:
        if self._points is None or event.artist is not self._points:
            return
        indices = list(getattr(event, "ind", []) or [])
        if not indices:
            return
        channel = str(self.frame.iloc[int(indices[0])]["channel"])
        self.highlight(channel)
        self.channelPicked.emit(channel)

    def highlight(self, channel: str) -> None:
        """Halo a channel chosen anywhere in the window."""
        if not channel or channel == self._highlighted:
            return
        self._highlighted = str(channel)
        self.redraw()

    def save_png(self, path: str | Path | None = None) -> Path | None:
        """Write the figure as drawn. Asks where, unless told."""
        if path is None:
            subject = getattr(getattr(self._session, "recording", None), "subject", "map")
            suggested = f"{subject}-contact-map-{self.measure_key()}.png"
            chosen, _ = QFileDialog.getSaveFileName(self, "Save the map", suggested,
                                                    "PNG image (*.png)")
            if not chosen:
                return None
            path = chosen
        path = Path(path)
        self.figure.savefig(path, dpi=200, facecolor=self.figure.get_facecolor())
        return path


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span <= 0:
        return np.full(values.shape, 0.4)
    return (values - float(values.min())) / span
