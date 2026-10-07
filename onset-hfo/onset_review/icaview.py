"""The Components panel: what ICA found, for the reviewer to choose from.

ICA here is experimental and removes nothing by itself. This panel shows
every component the stage fitted -- its share of the variance, MNE's muscle
and ECG scores, how much of its power is above 40 Hz, which contacts it
loads on, its time course and its spectrum -- and lets the reviewer tick the
ones to remove. Ticking changes nothing until **Remove ticked and
re-analyse**, which hands the choice to the Preprocessing panel's Apply, so
the removal is a preprocessing choice recorded in the steps and the report
like any other.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme
from onset_review.theme import card

__all__ = ["ComponentsPanel"]

COLUMNS = ("remove", "component", "variance", "muscle", "ecg", "above 40 Hz", "loads on")


class ComponentsPanel(QWidget):
    """The ICA components, scored, with one drawn close up."""

    #: The reviewer asked for these component indices to be removed.
    removeRequested = Signal(tuple)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._record = getattr(session, "ica", None)
        tokens = theme.current()

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("onset_components")
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(220)
        self.table.itemSelectionChanged.connect(self._draw_selected)

        self.figure = Figure(figsize=(4.4, 2.2), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(150)
        self.canvas.setMinimumWidth(240)
        self.time_axis = self.figure.add_axes((0.08, 0.6, 0.90, 0.36))
        self.spectrum_axis = self.figure.add_axes((0.08, 0.12, 0.40, 0.36))
        self.loading_axis = self.figure.add_axes((0.56, 0.12, 0.42, 0.36))

        self.remove_button = QPushButton("Remove ticked and re-analyse")
        self.remove_button.setObjectName("onset_components_remove")
        self.remove_button.clicked.connect(lambda _=False: self._request())
        self.status = theme.muted("")
        self.status.setObjectName("onset_components_status")
        bar = QHBoxLayout()
        bar.addWidget(self.remove_button)
        bar.addWidget(self.status, 1)

        self.caption = QLabel("")
        self.caption.setObjectName("onset_components_caption")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addWidget(self.table)
        box.addWidget(self.canvas, 1)
        box.addLayout(bar)
        box.addWidget(self.caption)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))
        self.populate()

    # -- the table ------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self._record) and int(self._record.get("n_components", 0)) > 0

    def populate(self) -> None:
        self.table.setRowCount(0)
        if not self.available:
            self.table.setVisible(False)
            self.canvas.setVisible(False)
            self.remove_button.setEnabled(False)
            self.status.setText("")
            self.caption.setText(
                "ICA is off. It is an experimental stage: turn it on under "
                "<b>Preprocessing</b> and press Apply, and the components it finds "
                "appear here, scored, for you to choose from. Nothing is removed "
                "until you choose.")
            self.caption.setStyleSheet(card("info"))
            return
        rec = self._record
        n = int(rec["n_components"])
        muscle = list(rec.get("muscle_scores") or [])
        ecg = list(rec.get("ecg_scores") or [])
        share = list(rec.get("variance_share") or [])
        hf = list(rec.get("hf_share") or [])
        loadings = np.asarray(rec.get("loadings"))
        names = list(rec.get("channels") or [])
        excluded = set(int(i) for i in rec.get("excluded") or [])
        suggested = set(rec.get("suggested_muscle") or []) | set(rec.get("suggested_ecg") or [])
        self.table.setRowCount(n)
        for i in range(n):
            tick = QTableWidgetItem("")
            tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            tick.setCheckState(Qt.Checked if i in excluded else Qt.Unchecked)
            self.table.setItem(i, 0, tick)
            label = f"{i}" + (" (removed)" if i in excluded else "") \
                + (" · suggested" if i in suggested and i not in excluded else "")
            self.table.setItem(i, 1, QTableWidgetItem(label))
            self.table.setItem(i, 2, QTableWidgetItem(
                f"{100 * share[i]:.1f} %" if i < len(share) else "—"))
            self.table.setItem(i, 3, QTableWidgetItem(
                f"{muscle[i]:.2f}" if i < len(muscle) else "—"))
            self.table.setItem(i, 4, QTableWidgetItem(
                f"{abs(ecg[i]):.2f}" if i < len(ecg) else "—"))
            self.table.setItem(i, 5, QTableWidgetItem(
                f"{100 * hf[i]:.0f} %" if i < len(hf) else "—"))
            top = ""
            if loadings.ndim == 2 and loadings.shape[1] > i and names:
                order = np.argsort(-np.abs(loadings[:, i]))[:3]
                top = ", ".join(names[j] for j in order if j < len(names))
            self.table.setItem(i, 6, QTableWidgetItem(top))
        self.table.setVisible(True)
        self.canvas.setVisible(True)
        self.remove_button.setEnabled(True)
        self.table.selectRow(0)
        self._say()

    def ticked(self) -> tuple[int, ...]:
        out = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item is not None and item.checkState() == Qt.Checked:
                out.append(i)
        return tuple(out)

    def _request(self) -> None:
        self.removeRequested.emit(self.ticked())

    def _say(self) -> None:
        rec = self._record
        parts = [f"ICA ({rec['method']}, {rec['n_components']} components, seed {rec['seed']}) "
                 f"fitted on {len(rec.get('channels') or [])} channels."]
        if not rec.get("muscle_scored", bool(rec.get("muscle_scores"))):
            parts.append("<b>Not scored for muscle</b>: MNE's score needs electrode positions "
                         "and this recording carries none; the share above 40 Hz is the guide.")
        elif rec.get("suggested_muscle"):
            parts.append("MNE's muscle scorer suggests "
                         + ", ".join(str(i) for i in rec["suggested_muscle"]) + ".")
        else:
            parts.append("MNE's muscle scorer suggests no component.")
        if rec.get("ecg_channel"):
            parts.append(f"ECG scored against {rec['ecg_channel']}"
                         + (": " + ", ".join(str(i) for i in rec["suggested_ecg"])
                            if rec.get("suggested_ecg") else ": none") + ".")
        if rec.get("excluded"):
            parts.append("<b>Removed: " + ", ".join(str(i) for i in rec["excluded"])
                         + " (your choice, in the steps and the report).</b>")
        else:
            parts.append("<b>Nothing removed.</b>")
        parts.append("ICA can take real HFO energy out with the artefact; the literature "
                     "is split on it for HFO work. Look at a component's time course and "
                     "spectrum before removing it, and at what the removal does to the "
                     "ranking after.")
        parts.extend(n for n in (rec.get("notes") or []) if not n.startswith("Not scored for muscle"))
        self.caption.setText(" ".join(parts))
        self.caption.setStyleSheet(card("warn"))
        self.status.setText("Tick components, then remove and re-analyse.")

    # -- one component close up ---------------------------------------------
    def selected(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return int(rows[0].row()) if rows else None

    def _draw_selected(self) -> None:
        index = self.selected()
        if index is None or not self.available:
            return
        self.draw(index)

    def draw(self, index: int) -> None:
        tokens = theme.current()
        rec = self._record
        sources = np.asarray(rec["sources"], dtype=float)
        sfreq = float(rec["sfreq"])
        if index < 0 or index >= sources.shape[0]:
            return
        for axis in (self.time_axis, self.spectrum_axis, self.loading_axis):
            axis.clear()
            axis.set_facecolor(tokens.surface)
            axis.tick_params(labelsize=6, colors=tokens.text_muted)
            for side in axis.spines.values():
                side.set_color(tokens.separator)
        trace = sources[index]
        seconds = np.arange(trace.size) / sfreq
        # Up to ten seconds at full resolution; the whole window decimated.
        step = max(1, trace.size // 20000)
        self.time_axis.plot(seconds[::step], trace[::step], lw=0.5, color=tokens.text)
        self.time_axis.set_title(f"component {index}: time course", fontsize=7,
                                 color=tokens.text_muted, loc="left")
        self.time_axis.set_xlim(0, float(seconds[-1]) if seconds.size else 1)
        n = int(min(trace.size, 8192))
        spectrum = np.abs(np.fft.rfft(trace[:n] - trace[:n].mean())) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        keep = freqs > 0.5
        self.spectrum_axis.loglog(freqs[keep], np.maximum(spectrum[keep], 1e-12),
                                  lw=0.7, color=tokens.accent)
        self.spectrum_axis.set_title("spectrum", fontsize=7, color=tokens.text_muted, loc="left")
        loadings = np.asarray(rec["loadings"], dtype=float)
        names = list(rec.get("channels") or [])
        if loadings.ndim == 2 and loadings.shape[1] > index:
            weights = loadings[:, index]
            order = np.argsort(-np.abs(weights))[:12]
            self.loading_axis.barh(range(len(order)), weights[order], color=tokens.accent)
            self.loading_axis.set_yticks(range(len(order)))
            self.loading_axis.set_yticklabels([names[j] if j < len(names) else str(j)
                                               for j in order], fontsize=6)
            self.loading_axis.invert_yaxis()
            self.loading_axis.set_title("loads on", fontsize=7, color=tokens.text_muted,
                                        loc="left")
        self.canvas.draw_idle()
