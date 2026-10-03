"""The dockable panels: what a reviewer reads beside the trace.

Five panels, each a plain `QWidget` that can be docked, floated, closed and
reopened. They have one rule in common and it is the reason this file is as dull
as it is: **a panel renders, it does not compute.** Every number comes from
`onset_review.session` or `onset_review.trends`, which are importable without
Qt and covered by tests. If a panel ever needs a quantity that is not there, the
quantity goes there first.

The layout follows Persyst's reading order rather than a signal processor's.
A reviewer opens the trend, finds the busy patch, clicks it, lands on the
signal, and only then consults the table. So the trend sits above the trace,
the tables sit beside it, and every panel emits where it wants the trace to go
instead of drawing its own copy of the signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_review import trends
from onset_review.session import ReviewSession

__all__ = ["FindingsPanel", "EventsPanel", "TrendsPanel", "AgreementPanel",
           "ProvenancePanel", "DataFrameModel"]

#: Columns renamed for reading. A clinician should never have to learn that
#: `mean_prominence_db` is how far the oscillation rises above the background.
HEADERS = {
    "rank": "#",
    "channel": "Channel",
    "n_events": "Events",
    "rate_per_min": "Rate /min",
    "rate_ci_low": "95% low",
    "rate_ci_high": "95% high",
    "mean_amplitude_uv": "Ampl µV",
    "mean_frequency_hz": "Freq Hz",
    "mean_duration_ms": "Dur ms",
    "mean_prominence_db": "Prom dB",
    "n_with_spike": "On spike",
    "reviewed": "Expert looked",
    "expert_n": "Expert events",
    "expert_rate_per_min": "Expert /min",
    "t_local": "Time s",
    "t_file": "File time s",
    "kind": "Kind",
    "detector": "Detector",
    "duration_ms": "Dur ms",
    "amplitude_uv": "Ampl µV",
    "frequency_hz": "Freq Hz",
    "prominence_db": "Prom dB",
    "n_peaks": "Peaks",
    "with_spike": "On spike",
    "n_detector": "Detector",
    "n_expert": "Expert",
    "matched": "Agree",
    "detector_only": "Only ours",
    "expert_only": "Only theirs",
    "sensitivity": "Sensitivity",
    "precision": "Precision",
}

#: Columns shown as a percentage rather than a fraction.
_PERCENT = {"sensitivity", "precision"}
#: Columns shown without decimals.
_INTEGER = {"rank", "n_events", "n_with_spike", "expert_n", "n_peaks",
            "n_detector", "n_expert", "matched", "detector_only", "expert_only"}


class DataFrameModel(QAbstractTableModel):
    """A read-only table model over a DataFrame, formatted for reading.

    One model for all four tables because the alternative -- a model per panel
    -- is four places to fix the same number-formatting bug. Row highlighting is
    delegated to a callable so each panel decides what deserves attention
    without subclassing.
    """

    def __init__(self, frame: pd.DataFrame, columns: list[str] | None = None,
                 highlight=None, parent=None):
        super().__init__(parent)
        self._highlight = highlight
        self.set_frame(frame, columns)

    def set_frame(self, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
        self.beginResetModel()
        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            frame = frame[keep] if keep else frame
        self._frame = frame.reset_index(drop=True)
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._frame.shape[1]

    def column_name(self, column: int) -> str:
        return str(self._frame.columns[column])

    def row_value(self, row: int, name: str):
        if name not in self._frame.columns or not 0 <= row < len(self._frame):
            return None
        return self._frame.iloc[row][name]

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            name = str(self._frame.columns[section])
            return HEADERS.get(name, name.replace("_", " "))
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        name = str(self._frame.columns[index.column()])

        if role == Qt.DisplayRole:
            return _format(value, name)
        if role == Qt.TextAlignmentRole:
            numeric = isinstance(value, (int, float, np.integer, np.floating))
            return int(Qt.AlignRight | Qt.AlignVCenter) if numeric and not isinstance(
                value, (bool, np.bool_)) else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self._highlight is not None:
            colour = self._highlight(self._frame.iloc[index.row()])
            return QColor(colour) if colour else None
        if role == Qt.ToolTipRole:
            return _format(value, name)
        return None


def _format(value, name: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "—"
    if name in _PERCENT:
        return f"{float(value) * 100:.0f}%"
    if name in _INTEGER:
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}" if abs(float(value)) < 1000 else f"{float(value):.0f}"
    return str(value)


def _table_view() -> QTableView:
    """A table configured for reading rather than editing.

    `SelectRows` and `SingleSelection` together are what make the arrow keys
    walk the recording: every selection change is one event or one channel, so
    the panel can translate it into a trace position without guessing which of
    several selected rows was meant.
    """
    view = QTableView()
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.setSelectionMode(QAbstractItemView.SingleSelection)
    view.setEditTriggers(QAbstractItemView.NoEditTriggers)
    view.setAlternatingRowColors(True)
    view.setSortingEnabled(True)
    view.verticalHeader().setVisible(False)
    view.verticalHeader().setDefaultSectionSize(20)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(True)
    return view


class FindingsPanel(QWidget):
    """Channels ranked by rate, with the interval that says whether to believe it.

    The two colourings are the panel's whole argument. A channel in the
    statistically tied set is tinted, because naming one busiest channel out of
    an overlapping set is the mistake this project measured and documented. A
    channel the archive's annotators never looked at is greyed, because a
    detection there is unjudged rather than wrong and a reviewer comparing
    columns has to see which rows the comparison is defined on.
    """

    channelPicked = Signal(str)

    COLUMNS = ["rank", "channel", "n_events", "rate_per_min", "rate_ci_low",
               "rate_ci_high", "mean_frequency_hz", "mean_duration_ms",
               "mean_prominence_db", "n_with_spike", "reviewed", "expert_n",
               "expert_rate_per_min"]

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self._session = session
        tied = set(session.candidates)
        reviewed = set(session.reviewed_channels)
        has_expert = session.has_expert

        def highlight(row):
            if row.get("channel") in tied:
                return "#fff3cd"
            if has_expert and row.get("channel") not in reviewed:
                return "#f4f4f4"
            return None

        self.model = DataFrameModel(session.findings, self.COLUMNS, highlight)
        self.view = _table_view()
        self.view.setModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self._emit)

        caption = QLabel(
            "Tinted rows are statistically tied with the busiest channel"
            + (". Grey rows were never reviewed by the archive's annotators, "
               "so a detection there is unjudged rather than wrong."
               if has_expert else "."))
        caption.setWordWrap(True)
        caption.setStyleSheet("color:#555;font-size:11px;padding:2px 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.view)
        layout.addWidget(caption)

    def _emit(self) -> None:
        rows = self.view.selectionModel().selectedRows()
        if not rows:
            return
        channel = self.model.row_value(rows[0].row(), "channel")
        if channel:
            self.channelPicked.emit(str(channel))

    def select_channel(self, channel: str) -> None:
        matches = self.model.frame.index[self.model.frame["channel"] == channel]
        if len(matches):
            self.view.selectRow(int(matches[0]))
            self.view.scrollTo(self.model.index(int(matches[0]), 0))


class EventsPanel(QWidget):
    """Every event in time order: the list the arrow keys walk.

    The filters exist because the three questions a reviewer actually asks of
    this list are different lists. "Show me the ripples on the busiest channel"
    and "show me everything riding a discharge" and "show me what the filter
    threw away and why" are each one combo box apart, and a single
    unfilterable table of five hundred rows answers none of them.
    """

    eventPicked = Signal(float, str)

    COLUMNS = ["t_local", "channel", "kind", "duration_ms", "amplitude_uv",
               "frequency_hz", "prominence_db", "n_peaks", "with_spike",
               "reject_reason"]

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self._session = session
        self._all = trends.event_table(session, include_rejected=True)

        self.kind = QComboBox()
        self.kind.addItems(["All kinds", "Ripples only", "Fast ripples only",
                            "Discharges only"])
        self.channel = QComboBox()
        self.channel.addItem("All channels")
        self.channel.addItems(list(session.findings["channel"])
                              if not session.findings.empty else [])
        self.on_spike = QCheckBox("Riding a discharge")
        self.rejected = QCheckBox("Show rejected")
        self.rejected.setToolTip(
            "Candidates the artifact filter removed, with the reason. Useful "
            "for asking why the detector did not mark something you can see.")

        bar = QHBoxLayout()
        for widget in (self.kind, self.channel, self.on_spike, self.rejected):
            bar.addWidget(widget)
        bar.addStretch(1)
        self.count = QLabel()
        self.count.setStyleSheet("color:#555;font-size:11px;")
        bar.addWidget(self.count)

        self.model = DataFrameModel(
            self._all, self.COLUMNS,
            lambda row: "#f8e0e0" if not row.get("accepted", True) else None)
        self.view = _table_view()
        self.view.setModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self._emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(3)
        layout.addLayout(bar)
        layout.addWidget(self.view)

        for widget in (self.kind, self.channel):
            widget.currentIndexChanged.connect(self.refilter)
        for widget in (self.on_spike, self.rejected):
            widget.stateChanged.connect(self.refilter)
        self.refilter()

    def refilter(self) -> None:
        frame = self._all
        if not self.rejected.isChecked():
            frame = frame[frame["accepted"]]
        wanted = {1: "ripple", 2: "fast ripple", 3: "spike"}.get(
            self.kind.currentIndex())
        if wanted:
            frame = frame[frame["kind"] == wanted]
        if self.channel.currentIndex() > 0:
            frame = frame[frame["channel"] == self.channel.currentText()]
        if self.on_spike.isChecked():
            frame = frame[frame["with_spike"]]
        self.model.set_frame(frame, self.COLUMNS)
        self.count.setText(f"{len(frame)} of {len(self._all)} events")

    def _emit(self) -> None:
        rows = self.view.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        t = self.model.row_value(row, "t_local")
        channel = self.model.row_value(row, "channel")
        if t is not None:
            self.eventPicked.emit(float(t), str(channel or ""))

    def step(self, delta: int) -> None:
        """Move the selection by `delta` rows, which is what the toolbar does."""
        if not self.model.rowCount():
            return
        rows = self.view.selectionModel().selectedRows()
        current = rows[0].row() if rows else -1
        target = min(max(current + delta, 0), self.model.rowCount() - 1)
        self.view.selectRow(target)
        self.view.scrollTo(self.model.index(target, 0))


class TrendsPanel(QWidget):
    """The trend: one row per channel, one column per time bin, colour for rate.

    This is the panel borrowed most directly from how Persyst is actually used.
    A reviewer does not read sixty seconds of forty-three channels trace-first;
    they read the trend, find the patch, and click into it. For HFO work the
    trend does a second job for free: a window whose activity is three bright
    cells and four hundred empty ones *looks* like what it is, which no ranked
    table of rates manages to convey.

    The source selector is the familiarisation feature. Switching between the
    detector's trend and the annotators' trend on the same axes shows a new
    reviewer where the two sources agree and, more usefully, where they do not.
    """

    cellPicked = Signal(float, str)

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        self._session = session
        self._pg = pg
        self._channels = (list(session.findings["channel"])
                          if not session.findings.empty else [])

        self.source = QComboBox()
        self.source.addItem("Detector", "detector")
        if session.has_expert:
            self.source.addItem("Expert markings", "expert")
        self.bin_s = QComboBox()
        for seconds in (1.0, 2.0, 5.0, 10.0):
            self.bin_s.addItem(f"{seconds:g} s bins", seconds)
        self.bin_s.setCurrentIndex(2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Trend of"))
        bar.addWidget(self.source)
        bar.addWidget(self.bin_s)
        bar.addStretch(1)
        self.hint = QLabel("Click a cell to take the trace there.")
        self.hint.setStyleSheet("color:#555;font-size:11px;")
        bar.addWidget(self.hint)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Time in window", units="s")
        self.plot.invertY(True)
        self.plot.setMouseEnabled(x=True, y=False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.scene().sigMouseClicked.connect(self._clicked)

        self.bar = pg.ColorBarItem(colorMap=pg.colormap.get("inferno"),
                                   label="events / bin", interactive=False)
        self.bar.setImageItem(self.image, insert_in=self.plot.getPlotItem())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(3)
        layout.addLayout(bar)
        layout.addWidget(self.plot)

        self.source.currentIndexChanged.connect(self.redraw)
        self.bin_s.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def redraw(self) -> None:
        bin_s = float(self.bin_s.currentData() or trends.DEFAULT_BIN_S)
        source = str(self.source.currentData() or "detector")
        matrix = trends.rate_matrix(self._session, bin_s=bin_s, source=source)
        if matrix.empty:
            self.image.clear()
            return
        self._matrix = matrix
        self._bin_s = bin_s
        # pyqtgraph images are column-major: (x, y) = (time, channel).
        self.image.setImage(np.asarray(matrix.to_numpy(), dtype=float).T,
                            autoLevels=False)
        self.image.setRect(0.0, 0.0, matrix.shape[1] * bin_s, matrix.shape[0])
        top = float(matrix.to_numpy().max()) or 1.0
        self.bar.setLevels((0.0, top))

        # At most a dozen or so labels: a 43-channel trend in a 230 px dock has
        # room for nothing like one label per row, and pyqtgraph will happily
        # draw them all on top of each other rather than thin them out.
        axis = self.plot.getAxis("left")
        step = max(1, int(np.ceil(len(matrix.index) / 14)))
        axis.setTicks([[(i + 0.5, name) for i, name in enumerate(matrix.index)
                        if i % step == 0]])
        # Built from the default font rather than QFont("", 7): an empty
        # family name reaches pyqtgraph's axis painter as a font Qt cannot
        # resolve, and `AxisItem.generateDrawSpecs` segfaults measuring text
        # with it. Copying the application font and shrinking it cannot.
        ticks = QFont(self.font())
        ticks.setPointSize(max(6, ticks.pointSize() - 2))
        axis.setStyle(tickFont=ticks)
        self.plot.setXRange(0.0, matrix.shape[1] * bin_s, padding=0.0)
        # Half a row of slack at each end, or the first and last channel labels
        # are clipped by the plot's own edge -- and the first is the busiest
        # channel, which is the one a reviewer looks for.
        self.plot.setYRange(-0.5, matrix.shape[0] + 0.5, padding=0.0)

    def _clicked(self, event) -> None:
        if getattr(self, "_matrix", None) is None:
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(event.scenePos())
        row = int(np.floor(point.y()))
        if not 0 <= row < len(self._matrix.index):
            return
        channel = str(self._matrix.index[row])
        # Centre the trace on the bin the reviewer clicked, not its left edge.
        t = max(0.0, float(point.x()) - self._bin_s / 2.0)
        self.cellPicked.emit(t, channel)


class AgreementPanel(QWidget):
    """The detector against the archive's annotators, on the window on screen.

    The point of this panel is not the percentages, which a single minute of one
    patient cannot pin down. It is the two "only" columns: where the detector
    marks something nobody marked, and where the annotators marked something it
    missed. Those are the rows worth opening on the trace, and they are how a
    reviewer learns in an afternoon what the detector is and is not for.
    """

    channelPicked = Signal(str)

    COLUMNS = ["channel", "n_expert", "n_detector", "matched", "detector_only",
               "expert_only", "sensitivity", "precision"]

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        summary = trends.agreement_summary(session)
        self.statement = QLabel(summary.get("statement") or
                                summary.get("reason", ""))
        self.statement.setWordWrap(True)
        self.statement.setStyleSheet(
            "padding:6px;background:#eef4fb;border:1px solid #cfe0f0;font-size:11px;")

        self.model = DataFrameModel(trends.agreement(session), self.COLUMNS)
        self.view = _table_view()
        self.view.setModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self._emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.statement)
        layout.addWidget(self.view)

    def _emit(self) -> None:
        rows = self.view.selectionModel().selectedRows()
        if rows:
            channel = self.model.row_value(rows[0].row(), "channel")
            if channel:
                self.channelPicked.emit(str(channel))


class ProvenancePanel(QTextBrowser):
    """What was done to the signal, in the order it was done.

    Every number on screen was produced by a chain of choices -- which channels
    were kept, where the notch went, which threshold fired -- and a reviewer who
    cannot see that chain is being asked to trust it. The pipeline already logs
    each step; this panel is the one place in the interface that simply shows
    them, unedited.
    """

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setHtml(self._html(session))

    @staticmethod
    def _html(session: ReviewSession) -> str:
        request = session.request
        rows = [
            ("Dataset", f"{request.dataset} · {request.subject} · run-{request.run}"),
            ("Window", f"{request.t_start:g}–{request.t_stop:g} s of the "
                       f"original recording"),
            ("Band", request.band_label()),
            ("Detector", ", ".join(request.detectors)),
            ("Threshold", "each detector's measured default"
                          if request.threshold_sd is None
                          else f"{request.threshold_sd:g} robust SD"),
            ("Sampling rate", f"{session.sfreq:g} Hz"),
            ("Montage", f"{session.montage}, {len(session.findings)} channels"),
        ]
        parts = ["<style>body{font:12px sans-serif;color:#222}"
                 "td{padding:1px 6px 1px 0;vertical-align:top}"
                 "th{text-align:left;padding-right:8px;color:#555;"
                 "font-weight:normal;white-space:nowrap}"
                 "li{margin-bottom:3px}h3{font-size:12px;margin:12px 0 4px}"
                 "</style><table>"]
        parts += [f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in rows]
        parts.append("</table>")
        parts.append("<h3>Preprocessing, in order</h3><ol>")
        parts += [f"<li>{step}</li>" for step in session.steps]
        parts.append("</ol>")
        if session.notes:
            parts.append("<h3>Dataset notes</h3><ul>")
            parts += [f"<li>{note}</li>" for note in session.notes]
            parts.append("</ul>")
        if session.citation:
            parts.append(f"<h3>Source</h3><p>{session.citation}</p>")
        parts.append(
            "<h3>Status</h3><p><b>Research prototype — not a medical "
            "device.</b> Not CE-marked, not FDA-cleared, not validated for "
            "clinical use. The cohort evidence, including what it fails to "
            "show, is in <code>docs/EVALUATION.md</code> and "
            "<code>docs/LIMITATIONS.md</code>.</p>")
        return "".join(parts)
