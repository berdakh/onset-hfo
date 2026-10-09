"""File → Compare with: two analyses side by side, as a window.

The left side (A) is the window on screen; the right (B) is the same
recording analysed with other settings, or an analysis kept as a project.
The comparison itself is `onset_review.comparison`; this shows it: what
differs, a few sentences, every channel's rate under each on one chart with
the ranking beside it, the events found by both or by one, and the verdicts
that differ. B can be opened in the main window, and the comparison saved as
Markdown.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from onset_review import comparison, theme

__all__ = ["CompareWindow", "OtherSettingsDialog"]


class OtherSettingsDialog(QDialog):
    """The settings B is analysed with: this window's, with a reference, a
    band or a threshold changed."""

    def __init__(self, request, parent=None):
        super().__init__(parent)
        from onset_hfo.config import PreprocessConfig
        from onset_hfo.preprocess import REFERENCES, effective_reference

        self.setWindowTitle("Compare with other settings")
        self.request = request
        preprocess = request.preprocess or PreprocessConfig()
        self.reference = QComboBox()
        for name in REFERENCES:
            self.reference.addItem(name.replace("_", " "), name)
        self.reference.setCurrentIndex(self.reference.findData(effective_reference(preprocess)))
        self.grid = QLineEdit(", ".join(f"{g}:{c}" for g, c in preprocess.grid_columns))
        self.grid.setPlaceholderText("for the Laplacian: G:8")
        self.band = QComboBox()
        self.band.addItem("ripple (80–250 Hz)", "ripple")
        self.band.addItem("fast ripple (250–500 Hz)", "fast_ripple")
        self.band.setCurrentIndex(max(0, self.band.findData(request.band)))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 20.0)
        self.threshold.setSingleStep(0.5)
        self.threshold.setSpecialValueText("each detector's default")
        self.threshold.setSuffix(" SD")
        self.threshold.setValue(float(request.threshold_sd or 0.0))
        form = QFormLayout(self)
        form.addRow(QLabel("B is this recording analysed again with these settings; "
                           "everything else stays as it is in this window."))
        form.addRow("Reference", self.reference)
        form.addRow("Grid columns", self.grid)
        form.addRow("Band", self.band)
        form.addRow("Threshold", self.threshold)
        self.problem = QLabel("")
        self.problem.setStyleSheet(f"color:{theme.current().bad};")
        form.addRow(self.problem)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Analyse and compare")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def other(self):
        """The request for B, or raises ValueError on grid columns it cannot read."""
        from onset_hfo.config import PreprocessConfig
        from onset_hfo.preprocess import parse_grid_columns

        preprocess = self.request.preprocess or PreprocessConfig()
        scheme = self.reference.currentData()
        preprocess = dataclasses.replace(
            preprocess, reference=scheme, bipolar=scheme == "bipolar",
            average_reference=scheme == "average",
            grid_columns=parse_grid_columns(self.grid.text()) if scheme == "laplacian" else ())
        threshold = float(self.threshold.value()) or None
        return dataclasses.replace(self.request, preprocess=preprocess,
                                   band=self.band.currentData(), threshold_sd=threshold)

    def _accept(self) -> None:
        try:
            self.other()
        except ValueError as error:
            self.problem.setText(str(error))
            return
        self.accept()


def _table(headers) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _cell(value, fmt="{:g}") -> QTableWidgetItem:
    item = QTableWidgetItem()
    if isinstance(value, float) and value != value:
        item.setText("–")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        item.setData(Qt.DisplayRole, float(fmt.format(value)) if fmt != "{:+g}" else
                     float(value))
        if fmt == "{:+g}":
            item.setText(fmt.format(value))
    else:
        item.setText("yes" if value is True else "" if value is False else str(value))
    return item


class CompareWindow(QWidget):
    """Two analyses lined up."""

    #: B's request, to open in the main window.
    openRequested = Signal(object)

    def __init__(self, a, b, a_label: str = "A", b_label: str = "B", parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Compare two analyses")
        self.resize(1180, 760)
        self.a, self.b = a, b
        self.result = comparison.compare(a, b, a_label, b_label)
        c = self.result
        tokens = theme.current()
        changed = ", ".join(f"{name} {right}" for name, _left, right in c.settings)
        who = QLabel(f"<b>{c.a_label}</b> — {a.request.label()} (the window)<br>"
                     f"<b>{c.b_label}</b> — {b.request.label()}"
                     + (f" — with {changed}" if changed else ""))
        who.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary = QLabel("\n".join(c.sentences))
        self.summary.setObjectName("onset_compare_summary")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # -- ranking: a chart and a table ---------------------------------------------
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(4.6, 4.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setMinimumSize(320, 300)
        comparison.draw_rates(figure, figure.add_subplot(111), c)
        unit = c.unit
        self.ranking = _table([unit.capitalize(), f"Rank {c.a_label}", f"Rank {c.b_label}",
                               "Moved", f"Rate {c.a_label}", f"Rate {c.b_label}",
                               f"Tied {c.a_label}", f"Tied {c.b_label}"])
        self.ranking.setObjectName("onset_compare_ranking")
        rows = c.ranking.to_dict("records")
        self.ranking.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = (row[unit], row["rank_a"], row["rank_b"], row["moved"], row["rate_a"],
                      row["rate_b"], bool(row["tied_a"]), bool(row["tied_b"]))
            formats = ("", "{:g}", "{:g}", "{:+g}", "{:.1f}", "{:.1f}", "", "")
            for col, (value, fmt) in enumerate(zip(values, formats, strict=True)):
                self.ranking.setItem(r, col, _cell(value, fmt or "{:g}"))
        self.ranking.setSortingEnabled(True)
        self.ranking.sortItems(1, Qt.AscendingOrder)       # A's ranking, to begin with
        rank_split = QSplitter(Qt.Horizontal)
        rank_split.addWidget(self.canvas)
        rank_split.addWidget(self.ranking)
        rank_split.setSizes([460, 620])

        # -- events and verdicts -----------------------------------------------------------
        self.events = _table(["Detector", f"Accepted under {c.a_label}",
                              f"Accepted under {c.b_label}", "Found by both",
                              f"Only {c.a_label}", f"Only {c.b_label}"])
        self.events.setObjectName("onset_compare_events")
        rows = c.events.to_dict("records")
        self.events.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for col, key in enumerate(("detector", "in_a", "in_b", "both", "only_a",
                                       "only_b")):
                self.events.setItem(r, col, _cell(row[key]))
        events_page = QWidget()
        events_box = QVBoxLayout(events_page)
        note = QLabel("Over the time both analysed. Two events are the same when they overlap "
                      f"in time (within {comparison.TOLERANCE_S * 1000:g} ms) on the same "
                      + ("contact." if unit == "contact" else "channel."))
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
        events_box.addWidget(note)
        events_box.addWidget(self.events, 1)
        self.verdicts = _table(["Channel", f"Verdict {c.a_label}", f"Verdict {c.b_label}"])
        self.verdicts.setObjectName("onset_compare_verdicts")
        rows = c.verdicts.to_dict("records")
        self.verdicts.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for col, key in enumerate(("channel", "verdict_a", "verdict_b")):
                self.verdicts.setItem(r, col, _cell(row[key]))
        verdict_page = QWidget()
        verdict_box = QVBoxLayout(verdict_page)
        verdict_note = QLabel("Channels judged differently in the two reads." if rows else
                              "The two reads judge every channel alike (or judge none).")
        verdict_note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
        verdict_box.addWidget(verdict_note)
        verdict_box.addWidget(self.verdicts, 1)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(rank_split, f"Ranking, by {unit}")
        self.tabs.addTab(events_page, "Events")
        self.tabs.addTab(verdict_page, "Verdicts")

        self.open_button = QPushButton(f"Open {c.b_label} in the window")
        self.open_button.setToolTip("Replace the main window with this analysis")
        self.open_button.clicked.connect(lambda _=False: self.openRequested.emit(self.b.request))
        self.save_button = QPushButton("Save as Markdown…")
        self.save_button.clicked.connect(lambda _=False: self.save_dialog())
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)

        box = QVBoxLayout(self)
        box.addWidget(who)
        box.addWidget(self.summary)
        box.addWidget(self.tabs, 1)
        box.addLayout(buttons)

    def markdown(self) -> str:
        return comparison.to_markdown(self.result, self.a.request, self.b.request)

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.markdown(), encoding="utf-8")
        return path

    def save_dialog(self) -> Path | None:
        from onset_review.files import current_folder

        path, _ = QFileDialog.getSaveFileName(self, "Save the comparison",
                                              str(Path(current_folder()) / "comparison.md"),
                                              "Markdown (*.md)")
        return self.save(path) if path else None
