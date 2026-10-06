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
from qtpy.QtGui import QColor, QFont, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_review import adjudication, theme, trends
from onset_review.session import ReviewSession
from onset_review.theme import card

__all__ = ["FindingsPanel", "EventsPanel", "TrendsPanel", "AgreementPanel",
           "ProvenancePanel", "DataFrameModel", "MIN_TABLE_HEIGHT"]

#: Columns renamed for reading. A clinician should never have to learn that
#: `mean_prominence_db` is how far the oscillation rises above the background.
HEADERS = {
    "verdict": "My read",
    "my_read": "My read",
    "judged": "Judged",
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

#: Long form for the columns whose heading cannot carry their meaning. Shown
#: as a tooltip on the header; see `DataFrameModel.headerData`.
HEADER_TIPS = {
    "verdict": "Your verdict on this event. A/D/U while this panel has focus.",
    "my_read": "Your verdict on this contact",
    "judged": "How many of this contact's ranked events you have given a "
              "verdict on — the same events the Events column counts",
    "rate_per_min": "The detector's rate, over the seconds actually analysed "
                    "on this contact",
    "reviewed": "Whether the archive's own annotators looked at this contact "
                "at all. They did not review every one, and a detection on a "
                "contact they skipped is unjudged rather than wrong.",
}

#: The reader's verdicts as they appear in a table cell. Short on purpose:
#: "A real event" repeated down four hundred rows is a wall of text, and the
#: column is narrow because the measurements beside it are what the reader is
#: judging against.
SHORT_VERDICTS = {"agree": "real", "disagree": "not real", "unsure": "unsure"}

def _indicate(view, model, name: str) -> None:
    """Point the sort arrow at the column the rows are already ordered by."""
    column = model.column_index(name)
    if column >= 0:
        view.horizontalHeader().setSortIndicator(column, Qt.AscendingOrder)


def _compact(button):
    """A button sized for a row that sits under a table rather than beside it.

    The verdict rows are two more rows in the right-hand column, and that
    column's height is what decides how small the whole window may be. At the
    platform's default button metrics the two of them cost 70 px of minimum
    window height, which is most of a laptop screen's margin.
    """
    button.setAutoDefault(False)
    button.setStyleSheet("padding:1px 8px;font-size:11px;")
    button.setMaximumHeight(22)
    return button


def _attributed(panel) -> bool:
    """True once the read has a reader's name on it, asking for one if not.

    A verdict is refused rather than recorded anonymously. That looks strict
    for a research prototype and it is the right strictness: the whole value
    of a recorded judgement is that someone can be asked about it afterwards,
    and a file full of opinions with no name against them cannot be used for
    anything -- not a report, not a second read, not a disagreement.

    Asked once per window, prefilled from the login name, and never asked
    again once answered.
    """
    read = panel._session.read
    if read.reader:
        return True
    ask = getattr(panel, "reader_prompt", None)
    name = (ask() if callable(ask) else "").strip()
    if not name:
        return False
    read.reader = name
    return True


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
        #: (column name, ascending). By name rather than by index so that it
        #: survives a rebuild that adds or removes a column.
        self._order = None
        self.set_frame(frame, columns)

    def set_frame(self, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
        self.beginResetModel()
        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            frame = frame[keep] if keep else frame
        self._frame = self._ordered(frame).reset_index(drop=True)
        self._cache()
        self.endResetModel()

    def _cache(self) -> None:
        """The frame as plain Python rows, built once per reset.

        `data` is called once per cell per layout pass, and a view sizing
        its columns asks for every cell of every row. At one `iat` on an
        Arrow-backed frame per call that was twenty seconds for a 500-row
        events list, on the GUI thread, at every window open; a list lookup
        is microseconds.
        """
        frame = self._frame
        self._names = [str(c) for c in frame.columns]
        self._cells = frame.to_numpy(dtype=object).tolist() if len(frame) else []
        self._colours: dict[int, object] = {}

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._order is None or frame.empty:
            return frame
        name, ascending = self._order
        if name not in frame.columns:
            return frame
        # `stable` so that re-sorting by a coarse column -- the reader's
        # verdict, say -- leaves each group in the order it already had,
        # which for the events list is time.
        return frame.sort_values(name, ascending=ascending, kind="stable")

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        """Sort by a column, which is what the clickable headers promised.

        `QTableView.setSortingEnabled` draws the arrow and calls this; the
        base class's implementation does nothing, so until this existed every
        header in the window was a control that moved and changed nothing.
        """
        if not 0 <= column < self._frame.shape[1]:
            return
        self._order = (str(self._frame.columns[column]),
                       order == Qt.AscendingOrder)
        self.beginResetModel()
        self._frame = self._ordered(self._frame).reset_index(drop=True)
        self._cache()
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

    def column_index(self, name: str) -> int:
        """Where `name` is drawn, or -1. Not the same as its place in a panel's
        `COLUMNS`: a column the frame does not carry is dropped, and every
        column after it shifts."""
        columns = list(self._frame.columns)
        return columns.index(name) if name in columns else -1

    def row_value(self, row: int, name: str):
        if name not in self._frame.columns or not 0 <= row < len(self._frame):
            return None
        return self._frame.iloc[row][name]

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return str(section + 1) if role == Qt.DisplayRole else None
        name = str(self._frame.columns[section])
        if role == Qt.DisplayRole:
            return HEADERS.get(name, name.replace("_", " "))
        # A short header is readable and a long one is explanatory; a tooltip
        # is how a column gets to be both. Only the columns whose meaning is
        # not in their name have one.
        if role == Qt.ToolTipRole:
            return HEADER_TIPS.get(name)
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if not (0 <= row < len(self._cells) and 0 <= col < len(self._names)):
            return None
        value = self._cells[row][col]
        name = self._names[col]

        if role == Qt.DisplayRole:
            return _format(value, name)
        if role == Qt.TextAlignmentRole:
            numeric = isinstance(value, (int, float, np.integer, np.floating))
            return int(Qt.AlignRight | Qt.AlignVCenter) if numeric and not isinstance(
                value, (bool, np.bool_)) else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self._highlight is not None:
            # One call per row, not per cell: the highlight reads the whole
            # row and answers the same for every column of it.
            if row not in self._colours:
                self._colours[row] = self._highlight(self._frame.iloc[row])
            colour = self._colours[row]
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


#: Floor for a docked table, in pixels: a header plus about three rows. Qt
#: divides a dock column by the widgets' minimum sizes, so a panel below with a
#: large minimum (the 3D view) otherwise takes the column and leaves the tables
#: showing a single row. `resizeDocks` does not win that argument; a minimum
#: does.
#:
#: Two rows rather than the six this started at, because the same minimums Qt
#: uses to divide the column it also sums to decide how small the window may
#: get: three tables in one column at six rows each cost 270 px of minimum
#: window height, which is the difference between fitting a 768 px laptop
#: screen and not. The *preferred* heights in `window.decorate` are unchanged,
#: so a large screen looks exactly as it did.
#:
#: This floor used to carry a second job -- stopping the 3D view's own large
#: minimum from eating the whole column -- which is why it started at six
#: rows. That view is in a scroll area now and no longer has a large minimum,
#: so the floor only has to be enough to see that a table is a table.
MIN_TABLE_HEIGHT = 60


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
    # Size columns from a sample of rows, not every row: in the resize-to-
    # contents mode Qt re-measures every cell of every column on each layout
    # pass, and a 500-row events list is 5,000 measurements a pass. A hundred
    # rows is enough to find the widest value of a column that is formatted
    # the same all the way down.
    view.horizontalHeader().setResizeContentsPrecision(100)
    view.horizontalHeader().setStretchLastSection(True)
    view.setMinimumHeight(MIN_TABLE_HEIGHT)
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
    #: (channel, verdict) after the reader judges a contact; "" means taken
    #: back. Connected to the autosave by the window.
    channelJudged = Signal(str, str)

    COLUMNS = ["rank", "channel", "my_read", "judged", "n_events",
               "rate_per_min", "rate_ci_low", "rate_ci_high",
               "mean_frequency_hz", "mean_duration_ms", "mean_prominence_db",
               "n_with_spike", "reviewed", "expert_n", "expert_rate_per_min"]

    #: Verdict -> (button text, tooltip). No single-letter keys here: the
    #: letters belong to the event list, where they are pressed hundreds of
    #: times, and a contact is judged once.
    CHANNEL_KEYS = (
        ("accept", "Count it", "This contact's events are worth counting"),
        ("ignore", "Ignore it", "Exclude this contact from the reading — a "
                                "popping electrode, or signal you do not "
                                "trust. The rate stays on screen; your verdict "
                                "goes in the report beside it."),
        ("unsure", "Cannot tell", "Recorded as undecided rather than left "
                                  "blank, which is a different thing"),
    )

    #: As on `EventsPanel`.
    reader_prompt = None

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self._session = session
        tied = set(session.candidates)
        reviewed = set(session.reviewed_channels)
        has_expert = session.has_expert

        def highlight(row):
            # The reader's own verdict outranks every automatic tint: once
            # someone has said "ignore this contact", that is the most
            # important thing about the row.
            if row.get("my_read") == adjudication.CHANNEL_LABELS["ignore"]:
                return theme.current().bad_surface
            if row.get("channel") in tied:
                return theme.current().highlight
            if has_expert and row.get("channel") not in reviewed:
                return theme.current().surface_alt
            return None

        self.model = DataFrameModel(self._with_read(), self.COLUMNS, highlight)
        self.view = _table_view()
        self.view.setModel(self.model)
        _indicate(self.view, self.model, "rank")
        self.view.selectionModel().selectionChanged.connect(self._emit)

        self.buttons = {}
        bar = QHBoxLayout()
        bar.setSpacing(4)
        bar.addWidget(QLabel("This contact:"))
        for verdict, text, tip in self.CHANNEL_KEYS:
            button = _compact(QPushButton(text))
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _=False, verdict=verdict: self.judge_channel(verdict))
            bar.addWidget(button)
            self.buttons[verdict] = button
        self.undo = _compact(QPushButton("Clear"))
        self.undo.setToolTip("Take this verdict back")
        self.undo.clicked.connect(lambda _=False: self.judge_channel(""))
        bar.addWidget(self.undo)
        bar.addStretch(1)

        caption = QLabel(
            "Tinted rows are statistically tied with the busiest channel"
            + (". Grey rows were never reviewed by the archive's annotators, "
               "so a detection there is unjudged rather than wrong."
               if has_expert else "."))
        caption.setWordWrap(True)
        caption.setStyleSheet(
            f"color:{theme.current().text_muted};font-size:9pt;padding:2px 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.view)
        layout.addLayout(bar)
        layout.addWidget(caption)

    # -- the reader's verdicts ---------------------------------------------
    def _with_read(self) -> pd.DataFrame:
        """The findings table with the reader's two columns added.

        `judged` is a fraction rather than a confirmed rate on purpose. A rate
        computed from a partly-judged channel is a confirmed count divided by
        the whole window, which understates every channel the reader has not
        finished; "12 of 51" says what is actually known.

        Counted over the same events as the `Events` column beside it --
        accepted, from the detector the ranking comes from -- and not over
        everything on the contact. The two columns sit next to each other and
        a reader will read them as a pair, so "0 of 102" beside "51" would be
        a question rather than an answer.
        """
        read = self._session.read
        primary = self._session.request.primary
        progress = read.progress([e for e in self._session.events
                                  if e.accepted and e.detector == primary])
        frame = self._session.findings.copy()
        if frame.empty:
            frame["my_read"] = []
            frame["judged"] = []
            return frame
        frame["my_read"] = [
            adjudication.CHANNEL_LABELS.get(read.channel_verdict(str(ch)), "")
            for ch in frame["channel"]]
        frame["judged"] = [
            (lambda p: f"{p['judged']} of {p['total']}" if p else "—")(
                progress.get(str(ch)))
            for ch in frame["channel"]]
        return frame

    def selected_channel(self) -> str:
        rows = self.view.selectionModel().selectedRows()
        if not rows:
            return ""
        return str(self.model.row_value(rows[0].row(), "channel") or "")

    def judge_channel(self, verdict: str) -> str:
        """Record `verdict` on the selected contact; "" takes it back."""
        channel = self.selected_channel()
        if not channel:
            return ""
        read = self._session.read
        if verdict and not _attributed(self):
            return ""
        if verdict:
            read.judge_channel(channel, verdict, reader=read.reader)
        else:
            read.clear_channel(channel)
        self.channelJudged.emit(channel, verdict)
        self.refresh(keep=channel)
        return channel

    def refresh(self, keep: str = "") -> None:
        self.model.set_frame(self._with_read(), self.COLUMNS)
        if keep:
            self.select_channel(keep)

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
    #: The selected event's verdict key. Carried as the key rather than as a
    #: time and a channel so that the detail view and the reader's verdict are
    #: talking about the same event by construction.
    eventKeyPicked = Signal(str)
    #: (event key, verdict) after the reader judges one. An empty verdict
    #: means they took it back. The window connects this to the autosave.
    judged = Signal(str, str)

    #: "My read" first. In a list of four hundred rows the column a reader is
    #: filling in is the one they scan down, and a column they have to scroll
    #: sideways to see is a column they will not keep up to date.
    #: `key` is last and hidden: it is how a selected row becomes the event it
    #: stands for, and it has to live in the model rather than in a frame
    #: beside it, or sorting the table would silently misalign the two.
    COLUMNS = ["verdict", "t_local", "channel", "kind", "duration_ms",
               "amplitude_uv", "frequency_hz", "prominence_db", "n_peaks",
               "with_spike", "reject_reason", "key"]

    #: Verdict -> (button text, keyboard key, tooltip). Single letters, and
    #: scoped to this panel rather than the window, because `a` is MNE's own
    #: annotation-mode key on the trace and stealing it would break the
    #: browser underneath us.
    KEYS = (
        ("agree", "&Real", "A", "This is a real event (A)"),
        ("disagree", "&Not real", "D", "Artifact, ringing, or not an "
                                       "oscillation (D)"),
        ("unsure", "&Unsure", "U", "Genuinely ambiguous — recorded as such, "
                                   "and counted as neither (U)"),
    )

    #: Set by `window.decorate` to a callable that asks who is reviewing and
    #: returns the name, or "" if they declined. Left as None in tests and in
    #: any caller that has already named the reader.
    reader_prompt = None

    def __init__(self, session: ReviewSession, parent=None):
        super().__init__(parent)
        self._session = session
        self._all = trends.event_table(session, include_rejected=True)
        self._shown = self._all

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
        self.count.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
        bar.addWidget(self.count)

        self.model = DataFrameModel(self._all, self.COLUMNS, self._tint)
        self.view = _table_view()
        self.view.setModel(self.model)
        self.view.selectionModel().selectionChanged.connect(self._emit)

        # The reader's own row of controls, under the table and against the
        # list it acts on. Buttons as well as keys: the keys are how the work
        # is actually done, and the buttons are how someone finds out the keys
        # exist.
        self.buttons = {}
        self.shortcuts = []
        verdicts = QHBoxLayout()
        verdicts.setSpacing(4)
        verdicts.addWidget(QLabel("This event:"))
        for verdict, text, key, tip in self.KEYS:
            button = _compact(QPushButton(text))
            button.setToolTip(tip)
            button.clicked.connect(
                lambda _=False, verdict=verdict: self.judge(verdict))
            verdicts.addWidget(button)
            self.buttons[verdict] = button
            self._bind(key, lambda verdict=verdict: self.judge(verdict))
        self.undo = _compact(QPushButton("Clear"))
        self.undo.setToolTip("Take this verdict back (Backspace)")
        self.undo.clicked.connect(lambda _=False: self.judge(""))
        verdicts.addWidget(self.undo)
        self._bind("Backspace", lambda: self.judge(""))
        self.note = _compact(QPushButton("Note…"))
        self.note.setToolTip("Say why, in a sentence (N)")
        self.note.clicked.connect(self._write_note)
        verdicts.addWidget(self.note)
        self._bind("N", self._write_note)
        verdicts.addStretch(1)
        self.progress = QLabel()
        self.progress.setStyleSheet(
            f"color:{theme.current().text_muted};font-size:9pt;")
        verdicts.addWidget(self.progress)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(3)
        layout.addLayout(bar)
        layout.addWidget(self.view)
        layout.addLayout(verdicts)

        for widget in (self.kind, self.channel):
            widget.currentIndexChanged.connect(self.refilter)
        for widget in (self.on_spike, self.rejected):
            widget.stateChanged.connect(self.refilter)
        self.refilter()
        # After `refilter`, so the frame carries every column and the index is
        # the one actually drawn. The list arrives in time order, so that is
        # where the arrow goes: an indicator on a column the data is not
        # ordered by is a statement the table is not making.
        _indicate(self.view, self.model, "t_local")

    # -- the reader's verdicts ---------------------------------------------
    def _bind(self, key: str, slot) -> None:
        """A single-letter shortcut that only fires while this panel has focus.

        `QShortcut` rather than `QPushButton.setShortcut`, because a button's
        shortcut is window-wide and cannot be scoped: binding `A` that way
        takes MNE's annotation-mode key away from the trace browser this
        window is built on. Scoped here, the letters belong to the event list
        while the reader is working it and to MNE everywhere else.
        """
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(slot)
        self.shortcuts.append(shortcut)

    def _tint(self, row):
        """Row colour: the reader's verdict first, the filter's second.

        Deliberately that order. Once someone has said an event is not real,
        that is the more important fact about the row than whether the
        artifact filter happened to agree.
        """
        tokens = theme.current()
        verdict = str(row.get("verdict") or "")
        if verdict == SHORT_VERDICTS["disagree"]:
            return tokens.bad_surface
        if verdict:
            return tokens.highlight
        return tokens.bad_surface if not row.get("accepted", True) else None

    def _verdicts_for(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the reader's column. Short words, because the column is narrow
        and "A real event" at 400 rows is a wall of text."""
        read = self._session.read
        frame = frame.copy()
        frame["verdict"] = [SHORT_VERDICTS.get(read.verdict_of(str(key)), "")
                            for key in frame.get("key", [])]
        return frame

    def selected_key(self) -> str:
        rows = self.view.selectionModel().selectedRows()
        if not rows:
            return ""
        return str(self.model.row_value(rows[0].row(), "key") or "")

    def judge(self, verdict: str, advance: bool = True) -> str:
        """Record `verdict` on the selected event; "" takes it back.

        Advancing to the next row afterwards is what makes this usable at the
        scale it has to work at: four hundred events is four hundred key
        presses, and reaching for the arrow key between each pair doubles that
        for no reason.
        """
        key = self.selected_key()
        if not key:
            return ""
        read = self._session.read
        if verdict and not _attributed(self):
            return ""
        if verdict:
            read.judge_event(key, verdict, reader=read.reader)
        else:
            read.clear_event(key)
        self.judged.emit(key, verdict)
        row = self.view.selectionModel().selectedRows()[0].row()
        self.refilter(keep_row=row)
        if advance:
            self.step(+1)
        return key

    def _write_note(self) -> None:
        """A sentence saying why. Optional, and the reason a disagreement is
        worth anything to the next person who reads it."""
        from qtpy.QtWidgets import QInputDialog

        key = self.selected_key()
        if not key or not _attributed(self):
            return
        read = self._session.read
        existing = read.events.get(key)
        text, ok = QInputDialog.getText(
            self, "Note on this event", "Why?",
            text=existing.note if existing else "")
        if not ok:
            return
        verdict = read.verdict_of(key) or "unsure"
        read.judge_event(key, verdict, reader=read.reader, note=text.strip())
        self.judged.emit(key, verdict)
        self.refilter(keep_row=self.view.selectionModel().selectedRows()[0].row()
                      if self.view.selectionModel().selectedRows() else -1)

    def step_unjudged(self, delta: int = 1) -> bool:
        """Jump to the next event with no verdict on it. False when none left."""
        read = self._session.read
        rows = self.view.selectionModel().selectedRows()
        current = rows[0].row() if rows else -1
        order = (range(current + 1, self.model.rowCount()) if delta >= 0
                 else range(current - 1, -1, -1))
        for row in order:
            if not read.verdict_of(str(self.model.row_value(row, "key") or "")):
                self.view.selectRow(row)
                self.view.scrollTo(self.model.index(row, 0))
                return True
        return False

    def refilter(self, keep_row: int = -1) -> None:
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
        # `_shown` is the filtered frame in its natural order, which the
        # progress figures count over; the model may hold it in a different
        # order once the reader has clicked a header, which is why a row is
        # turned back into an event through the model and not through this.
        self._shown = self._verdicts_for(frame).reset_index(drop=True)
        self.model.set_frame(self._shown, self.COLUMNS)
        hidden = self.model.column_index("key")
        if hidden >= 0:
            self.view.setColumnHidden(hidden, True)
        self.count.setText(f"{len(frame)} of {len(self._all)} events")
        self._show_progress()
        # Refiltering rebuilds the model, which drops the selection. After a
        # verdict that would send the reader back to the top of the list.
        if 0 <= keep_row < self.model.rowCount():
            self.view.selectRow(keep_row)

    def _show_progress(self) -> None:
        read = self._session.read
        total = int(self._all["accepted"].sum()) if len(self._all) else 0
        done = sum(1 for key in self._all.loc[self._all["accepted"], "key"]
                   if read.verdict_of(str(key))) if total else 0
        self.progress.setText(f"{done} of {total} judged" if total else "")

    def _emit(self) -> None:
        rows = self.view.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        t = self.model.row_value(row, "t_local")
        channel = self.model.row_value(row, "channel")
        key = self.model.row_value(row, "key")
        if t is not None:
            self.eventPicked.emit(float(t), str(channel or ""))
        if key:
            self.eventKeyPicked.emit(str(key))

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
        self.hint.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
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
        self.statement.setStyleSheet(card("info"))

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
