"""A study page's chart in a window of its own, that answers the pointer.

The pages print their figures as pictures, which is what a page should do:
the picture and the table beside it come from the same frame and cannot
disagree. This window draws the same figure with the same function
(`studycharts.draw_sweep`, `studycharts.draw_outcome`) on a live canvas, so
that a reader can ask it things:

* hover a dot or a point: the patient or the detector, and the value;
* click a patient's dot: their window opens on the Recording page, when one
  is cached on this machine;
* on the sweep, drag the threshold: a line follows it, and every detector's
  numbers at that threshold are read out under the chart, from the
  committed sweep -- the thresholds that were measured, nothing between them
  invented.

Zoom and pan are matplotlib's own toolbar. Nothing here changes a number.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from onset_review import studycharts, theme

__all__ = ["ChartWindow"]


class ChartWindow(QWidget):
    """`kind` is "sweep" or "outcome"; `data` what the page drew from."""

    #: A patient's dot was clicked: their subject id.
    subjectClicked = Signal(str)

    def __init__(self, kind: str, data: dict, title: str = "", parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        if kind not in ("sweep", "outcome"):
            raise ValueError(f"no chart called {kind!r}")
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle(title or "Chart")
        self.kind, self.data = kind, data
        self.figure = Figure(figsize=(9, 4), dpi=100, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setObjectName("onset_chart_canvas")
        self.readout = QLabel("")
        self.readout.setObjectName("onset_chart_readout")
        self.readout.setWordWrap(True)
        self.readout.setTextFormat(Qt.RichText)
        self.hint = QLabel("")
        self.hint.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
        self.hint.setWordWrap(True)
        box = QVBoxLayout(self)
        box.setContentsMargins(theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING)
        box.addWidget(NavigationToolbar2QT(self.canvas, self))
        box.addWidget(self.canvas, 1)
        self.slider = None
        self._thresholds: list[float] = []
        self._cursor = None
        if kind == "sweep":
            self._draw_sweep()
            row = QHBoxLayout()
            row.addWidget(QLabel("Threshold"))
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setObjectName("onset_chart_threshold")
            self.slider.setRange(0, max(0, len(self._thresholds) - 1))
            self.slider.valueChanged.connect(self.set_threshold_index)
            row.addWidget(self.slider, 1)
            box.addLayout(row)
            self.hint.setText("Drag the threshold to read every detector's numbers at it; "
                              "hover a point for its value. Only measured thresholds are "
                              "offered.")
        else:
            self._draw_outcome()
            self.hint.setText("Hover a dot for the patient; click it to open their window "
                              "when one is cached here.")
        box.addWidget(self.readout)
        box.addWidget(self.hint)
        self._tip = None
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.canvas.mpl_connect("button_press_event", self._click)
        self.resize(1000, 620)
        if self.slider is not None and self._thresholds:
            chosen = data.get("start_threshold")
            index = (int(np.argmin([abs(t - chosen) for t in self._thresholds]))
                     if chosen is not None else 0)
            self.slider.setValue(index)
            self.set_threshold_index(index)

    # -- drawing ----------------------------------------------------------------
    def _draw_sweep(self) -> None:
        axis = self.figure.add_subplot(1, 1, 1)
        self.axis = axis
        d = self.data
        self.artists = studycharts.draw_sweep(self.figure, axis, d["frame"], d["band"],
                                              d["metric"], d.get("best"),
                                              marks=d.get("marks", ()), yours=d.get("yours"))
        part = d["frame"][d["frame"]["band"] == d["band"]]
        self._thresholds = sorted(float(t) for t in part["threshold_sd"].unique())
        self._cursor = axis.axvline(self._thresholds[0] if self._thresholds else 0,
                                    color=studycharts.MUTED, linewidth=1, linestyle=":")
        self.figure.tight_layout()

    def _draw_outcome(self) -> None:
        d = self.data
        self.dots = studycharts.draw_outcome(self.figure, d["cohort"], d.get("groups"),
                                             highlight=d.get("highlight"), yours=d.get("yours"))
        self.figure.tight_layout(w_pad=2.0)

    # -- the threshold -----------------------------------------------------------
    @property
    def threshold(self) -> float | None:
        if self.slider is None or not self._thresholds:
            return None
        return self._thresholds[self.slider.value()]

    def set_threshold_index(self, index: int) -> None:
        if not self._thresholds:
            return
        index = max(0, min(index, len(self._thresholds) - 1))
        threshold = self._thresholds[index]
        self._cursor.set_xdata([threshold, threshold])
        self.canvas.draw_idle()
        self.readout.setText(self.readout_at(threshold))

    def readout_at(self, threshold: float) -> str:
        """Every detector's measured numbers at `threshold`, as the sweep has them."""
        frame, band = self.data["frame"], self.data["band"]
        rows = frame[(frame["band"] == band) & (frame["threshold_sd"] == threshold)]
        parts = [f"<b>{threshold:g} SD</b>"]
        for row in rows.sort_values("detector").itertuples():
            name = studycharts.DETECTOR_LABELS.get(row.detector, row.detector)
            values = []
            for column, label in (("precision", "precision"), ("recall", "recall"),
                                  ("f1", "F1"), ("rank_rho", "ρ"),
                                  ("detections", "per 60 s")):
                value = getattr(row, column, float("nan"))
                if value == value:      # not NaN
                    values.append(f"{label} {value:.0f}" if column == "detections"
                                  else f"{label} {value:.2f}")
            parts.append(f"{name}: " + " · ".join(values))
        if len(parts) == 1:
            parts.append("not measured in this band")
        return "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)

    # -- the pointer ---------------------------------------------------------------
    def describe_at(self, event) -> str:
        """What is under the pointer, or ""."""
        if self.kind == "sweep":
            metric = self.data["metric"]
            for (point, mark) in self.artists["marks"]:
                if point.contains(event)[0]:
                    return (f"This window, {mark['detector']}: {mark['value']:.2f} at "
                            f"{mark['threshold_sd']:g} SD")
            for group, suffix in (("lines", ""), ("yours", " (your re-run)")):
                for detector, (line, rows) in self.artists[group].items():
                    hit, info = line.contains(event)
                    if hit and len(info.get("ind", [])):
                        row = rows.iloc[int(info["ind"][0])]
                        return (f"{studycharts.DETECTOR_LABELS.get(detector, detector)}"
                                f"{suffix}: {row[metric]:.3f} at {row['threshold_sd']:g} SD")
            return ""
        for dots, subjects, values, label in self.dots:
            hit, info = dots.contains(event)
            if hit and len(info.get("ind", [])):
                at = int(info["ind"][0])
                return f"{subjects[at]} — {label}: {values[at]:.2f} of the tied set resected"
        return ""

    def subject_at(self, event) -> str:
        if self.kind != "outcome":
            return ""
        for dots, subjects, _values, label in self.dots:
            hit, info = dots.contains(event)
            if hit and len(info.get("ind", [])) and "your cohort" not in label:
                return subjects[int(info["ind"][0])]
        return ""

    def _hover(self, event) -> None:
        if event.inaxes is None:
            return
        text = self.describe_at(event)
        if self._tip is not None:
            self._tip.remove()
            self._tip = None
        if text:
            self._tip = event.inaxes.annotate(
                text, (event.xdata, event.ydata), xytext=(12, 12), textcoords="offset points",
                fontsize=9, color=studycharts.TEXT,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": studycharts.GRID})
        self.canvas.draw_idle()

    def _click(self, event) -> None:
        subject = self.subject_at(event) if event.inaxes is not None else ""
        if subject:
            self.subjectClicked.emit(subject)
