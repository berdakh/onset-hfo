"""File → New case / Open case: one patient's recordings, step by step.

A case (`onset_hfo.case`) is a BIDS-iEEG folder with the work done on it.
This window walks it: a bar of steps on the left -- Import, Channels &
electrodes, Annotate, then the steps later phases bring -- and the step's
page beside it.

* **Import** converts a clinical file into the case through its bridge:
  what the file is, its channels typed as the reader confirms them, the
  marks it carries mapped to the case's vocabulary, what identifying fields
  it had (left behind); then a conversion report.
* **Channels & electrodes** sets each channel's type and marks contacts
  bad, with why; every change goes into the case's log.
* **Annotate** draws hours of recording at a glance (sampled, and says so)
  with seizures, sleep and artefacts as tracks; marks are added from a form,
  or drawn on the trace in MNE's own browser, which opens at the time chosen
  with the case's vocabulary ready; closing it brings its marks back. Any
  minute opens in the review window, analysed.
* **Segments** chooses the stretches to analyse by rule (sleep stage,
  distance from seizures, artefacts, flat stretches; `onset_hfo.case.segments`).
* **Preprocess** sets how every segment is analysed.
* **Interictal** analyses each segment by the review window's own path and
  pools them (`onset_review.caseinterictal`).
* **Ictal onset** computes the Epileptogenicity Index of every marked seizure
  and how consistently each channel leads (`onset_review.caseictal`).
* **Review** opens each segment in the review window and counts the verdicts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_hfo.case import annotations as marks
from onset_hfo.case import bids, bridges, overview
from onset_hfo.case.model import STEPS, Case
from onset_review import theme

__all__ = ["CaseWindow", "ConvertDialog", "clock", "MNE_TYPE_CHOICES"]

#: The types a channel can be given, in MNE's spelling.
MNE_TYPE_CHOICES = ("seeg", "ecog", "eeg", "ecg", "emg", "eog", "misc", "stim")
TRACE_SECONDS = 20.0


def clock(seconds: float) -> str:
    """Seconds from the recording's start as h:mm:ss.s."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
    return label


class _Job(QThread):
    """A function run off the window's thread, with progress."""

    progressed = Signal(float)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.result = None
        self.error = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.result = self.function(progress=self.progressed.emit,
                                        should_stop=lambda: self._stop)
        except Exception as error:      # noqa: BLE001 - shown, not raised
            self.error = error


# -- converting a file ----------------------------------------------------------------------
class ConvertDialog(QDialog):
    """What a file is and how it will be converted, before anything is written."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Convert {Path(path).name} into the case")
        self.resize(820, 700)
        self.path = Path(path)
        self.info = bridges.inspect(self.path)
        info = self.info
        lines = [f"<b>{self.path.name}</b> — {info['format']}, {info['n_channels']} channels "
                 f"at {info['sfreq']:g} Hz, {info['duration_s'] / 3600:.2f} h"
                 + (f", starting at {info['start_time_of_day']}"
                    if info["start_time_of_day"] else "")]
        if info["identifying"]:
            lines.append("Left behind (identifying): "
                         + ", ".join(sorted(info["identifying"].values())) + ".")
        else:
            lines.append("The file carries no patient name, identifier or date.")
        summary = QLabel("<br>".join(lines))
        summary.setWordWrap(True)

        channels = info["channels"]
        self.table = QTableWidget(len(channels), 4)
        self.table.setObjectName("onset_convert_channels")
        self.table.setHorizontalHeaderLabels(["Channel", "In the file", "Convert as", "Bad"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.combos: dict[str, QComboBox] = {}
        self.bad: dict[str, QCheckBox] = {}
        for r, row in enumerate(channels.itertuples()):
            self.table.setItem(r, 0, QTableWidgetItem(str(row.name)))
            self.table.setItem(r, 1, QTableWidgetItem(str(row.declared)))
            combo = QComboBox()
            combo.addItems(MNE_TYPE_CHOICES)
            combo.setCurrentText(str(row.suggested) if row.suggested in MNE_TYPE_CHOICES
                                 else "misc")
            self.table.setCellWidget(r, 2, combo)
            self.combos[str(row.name)] = combo
            check = QCheckBox()
            self.table.setCellWidget(r, 3, check)
            self.bad[str(row.name)] = check
        quick = QHBoxLayout()
        quick.addWidget(QLabel("Every brain channel:"))
        for kind, text in (("seeg", "SEEG (depth)"), ("ecog", "ECoG (grids, strips)")):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, kind=kind: self.set_brain_type(kind))
            quick.addWidget(button)
        quick.addStretch(1)

        mapped = info["marks"]
        self.marks_table = QTableWidget(len(mapped), 3)
        self.marks_table.setHorizontalHeaderLabels(["Start", "Text in the file", "Taken as"])
        self.marks_table.verticalHeader().setVisible(False)
        self.marks_table.horizontalHeader().setStretchLastSection(True)
        for r, row in enumerate(mapped.itertuples()):
            for c, value in enumerate((clock(row.onset), row.value, row.trial_type)):
                self.marks_table.setItem(r, c, QTableWidgetItem(str(value)))
        self.mains = QComboBox()
        self.mains.addItem("50 Hz", 50.0)
        self.mains.addItem("60 Hz", 60.0)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText("the recording reference, as the montage sheet says")
        form = QFormLayout()
        form.addRow("Mains", self.mains)
        form.addRow("Reference", self.reference)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Convert")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(summary)
        box.addWidget(QLabel("<b>Channels</b> — a clinical export usually calls every channel "
                             "scalp EEG; say what each is."))
        box.addLayout(quick)
        box.addWidget(self.table, 3)
        box.addWidget(QLabel(f"<b>Marks in the file</b> — {len(mapped)}; the text is kept, the "
                             "kind is what the analysis reads."))
        box.addWidget(self.marks_table, 1)
        box.addLayout(form)
        box.addWidget(buttons)

    def set_brain_type(self, kind: str) -> None:
        for combo in self.combos.values():
            if combo.currentText() in ("seeg", "ecog", "eeg"):
                combo.setCurrentText(kind)

    def channel_types(self) -> dict:
        return {name: combo.currentText() for name, combo in self.combos.items()}

    def bad_channels(self) -> list[str]:
        return [name for name, check in self.bad.items() if check.isChecked()]


# -- the pages -------------------------------------------------------------------------------
class _RecordingPicker(QComboBox):
    def fill(self, case: Case) -> None:
        current = self.currentData()
        self.blockSignals(True)
        self.clear()
        for rec in case.recordings:
            self.addItem(f"run {rec.run} — {rec.source_name} ({rec.duration_s / 3600:.2f} h)",
                         rec.run)
        if current is not None and self.findData(current) >= 0:
            self.setCurrentIndex(self.findData(current))
        self.blockSignals(False)


class ImportPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self._job = None
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("onset_case_recordings")
        self.table.setHorizontalHeaderLabels(["Run", "From", "Format", "Rate", "Channels",
                                              "Length", "Converted"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_report)
        self.add_button = QPushButton("Add a recording…")
        self.add_button.setObjectName("onset_case_add")
        self.add_button.clicked.connect(lambda _=False: self.add_dialog())
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.status = _muted("Each recording is converted from the clinical system's file "
                             "into the case; the file itself is never changed.")
        self.report = QTextBrowser()
        self.report.setObjectName("onset_case_report")
        row = QHBoxLayout()
        row.addWidget(self.add_button)
        row.addWidget(self.progress, 1)
        box = QVBoxLayout(self)
        box.addLayout(row)
        box.addWidget(self.status)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.report)
        split.setSizes([260, 360])
        box.addWidget(split, 1)

    def refresh(self) -> None:
        case = self.window_.case
        self.table.setRowCount(len(case.recordings))
        for r, rec in enumerate(case.recordings):
            values = (rec.run, rec.source_name, rec.source_format, f"{rec.sfreq:g} Hz",
                      str(rec.n_channels), clock(rec.duration_s),
                      f"{rec.converted_at.replace('T', ' ')} by {rec.converted_by}")
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))
        if case.recordings and not self.table.selectedItems():
            self.table.selectRow(len(case.recordings) - 1)

    def show_report(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        case = self.window_.case
        if not rows or rows[0].row() >= len(case.recordings):
            return
        rec = case.recordings[rows[0].row()]
        path = case.derivatives / "conversion" / f"{case.where(rec).stem}_conversion.md"
        self.report.setMarkdown(path.read_text(encoding="utf-8") if path.exists()
                                else "No conversion report for this recording.")

    def add_dialog(self):
        from onset_hfo.io import file_filter

        path, _ = QFileDialog.getOpenFileName(self, "Add a recording to the case",
                                              str(Path.home()), file_filter())
        if not path:
            return None
        try:
            dialog = ConvertDialog(path, self)
        except Exception as error:      # noqa: BLE001 - a file that cannot be read is said
            QMessageBox.warning(self, "Add a recording", str(error))
            return None
        if dialog.exec() != QDialog.Accepted:
            return None
        return self.convert(path, dialog.channel_types(), dialog.bad_channels(),
                            float(dialog.mains.currentData()),
                            dialog.reference.text().strip() or "n/a")

    def convert(self, path, channel_types, bad=(), line_freq=50.0, reference="n/a",
                wait: bool = False):
        from onset_review.workers import track

        case = self.window_.case

        def work(progress, should_stop):
            return bridges.convert(path, case, channel_types=channel_types, bad=bad,
                                   line_freq=line_freq, reference=reference,
                                   progress=progress, should_stop=should_stop)

        self._job = _Job(work, self)
        self._job.progressed.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._job.finished.connect(self._converted)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.add_button.setEnabled(False)
        self.status.setText(f"Converting {Path(path).name}…")
        track(self._job).start()
        if wait:
            self._job.wait()
            self._converted()
        return self._job

    def _converted(self) -> None:
        job = self._job
        if job is None or job.isRunning():
            return
        self._job = None
        self.progress.setVisible(False)
        self.add_button.setEnabled(True)
        if job.error is not None:
            self.status.setText(f"Not converted: {job.error}")
            return
        report = job.result
        self.status.setText(f"Converted {report.source} as run {report.run} in "
                            f"{report.seconds:.0f} s.")
        self.refresh()
        self.table.selectRow(len(self.window_.case.recordings) - 1)
        self.changed.emit()


class ChannelsPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self.picker = _RecordingPicker()
        self.picker.currentIndexChanged.connect(lambda _i: self.load())
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("onset_case_channels")
        self.table.setHorizontalHeaderLabels(["Channel", "Type", "Bad", "Why"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.save_button = QPushButton("Save changes")
        self.save_button.clicked.connect(lambda _=False: self.save())
        self.done_button = QPushButton("Channels checked")
        self.done_button.setToolTip("Mark this step done for the case")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("channels"))
        self.status = _muted("Types decide what is analysed: only SEEG and ECoG contacts are. "
                             "A contact marked bad is left out of every analysis, with the "
                             "reason you give.")
        top = QHBoxLayout()
        top.addWidget(QLabel("Recording"))
        top.addWidget(self.picker, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addWidget(self.status)
        box.addWidget(self.table, 1)
        box.addLayout(buttons)
        self._rows: list[dict] = []

    def refresh(self) -> None:
        self.picker.fill(self.window_.case)
        self.load()

    def load(self) -> None:
        run = self.picker.currentData()
        if run is None:
            self.table.setRowCount(0)
            return
        frame = self.window_.case.channels(run)
        self._rows = frame.to_dict("records")
        self.table.setRowCount(len(frame))
        for r, row in enumerate(self._rows):
            self.table.setItem(r, 0, QTableWidgetItem(str(row["name"])))
            combo = QComboBox()
            combo.addItems(MNE_TYPE_CHOICES)
            combo.setCurrentText(bids.MNE_TYPES.get(str(row["type"]).upper(), "misc"))
            self.table.setCellWidget(r, 1, combo)
            check = QCheckBox()
            check.setChecked(str(row["status"]).lower() == "bad")
            self.table.setCellWidget(r, 2, check)
            why = str(row.get("status_description") or "")
            self.table.setItem(r, 3, QTableWidgetItem("" if why == "n/a" else why))

    def edited(self) -> pd.DataFrame:
        rows = []
        for r, row in enumerate(self._rows):
            kind = self.table.cellWidget(r, 1).currentText()
            bad = self.table.cellWidget(r, 2).isChecked()
            why = (self.table.item(r, 3).text().strip() if self.table.item(r, 3) else "")
            rows.append({**row, "type": bids.BIDS_TYPES.get(kind, "MISC"),
                         "status": "bad" if bad else "good",
                         "status_description": why or "n/a"})
        return pd.DataFrame(rows, columns=list(bids.CHANNEL_COLUMNS))

    def save(self) -> None:
        run = self.picker.currentData()
        if run is None:
            return
        self.window_.case.save_channels(run, self.edited())
        self.status.setText(f"Saved; the change is in the case's log ({time.strftime('%H:%M')}).")
        self.changed.emit()


class AnnotatePage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self._job = None
        self.trace = None           # the MNE browser, while it is open
        self._trace_raw = None
        self.picker = _RecordingPicker()
        self.picker.currentIndexChanged.connect(lambda _i: self.load())
        self.draw_button = QPushButton("Draw the overview")
        self.draw_button.setObjectName("onset_case_draw")
        self.draw_button.setToolTip("Hours of recording at a glance, sampled: drawn once and "
                                    "kept with the case")
        self.draw_button.clicked.connect(lambda _=False: self.draw_overview())
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(9, 2.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(220)
        self.canvas.mpl_connect("button_press_event", self._clicked)
        self.overview_note = _muted("")
        self.time = QDoubleSpinBox()
        self.time.setObjectName("onset_case_time")
        self.time.setDecimals(1)
        self.time.setRange(0.0, 1e7)
        self.time.setSuffix(" s")
        self.time.valueChanged.connect(lambda _v: self._show_time())
        self.time_label = QLabel("")
        self.trace_button = QPushButton("Open the trace here")
        self.trace_button.setObjectName("onset_case_trace")
        self.trace_button.setToolTip("MNE's browser at this time, every channel, the case's "
                                     "marks shown: press A to mark, drag to draw; closing it "
                                     "brings the marks back to the case")
        self.trace_button.clicked.connect(lambda _=False: self.open_trace())
        self.review_button = QPushButton("Analyse this minute")
        self.review_button.setToolTip("Open the minute from this time in the review window, "
                                      "analysed")
        self.review_button.clicked.connect(lambda _=False: self.open_review())
        self.marks_table = QTableWidget(0, 7)
        self.marks_table.setObjectName("onset_case_marks")
        self.marks_table.setHorizontalHeaderLabels(["Kind", "Start", "Duration", "Channels",
                                                    "Text", "From", "By"])
        self.marks_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.marks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.marks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.marks_table.horizontalHeader().setStretchLastSection(True)
        self.marks_table.itemDoubleClicked.connect(lambda _item: self._go_to_mark())
        self.kind = QComboBox()
        for kind, what in marks.KINDS.items():
            self.kind.addItem(kind, kind)
            self.kind.setItemData(self.kind.count() - 1, what, Qt.ToolTipRole)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 86400.0)
        self.duration.setDecimals(1)
        self.duration.setSuffix(" s")
        self.note = QLineEdit()
        self.note.setPlaceholderText("note (optional)")
        self.add_button = QPushButton("Add at this time")
        self.add_button.clicked.connect(lambda _=False: self.add_mark())
        self.delete_button = QPushButton("Remove")
        self.delete_button.clicked.connect(lambda _=False: self.remove_selected())
        self.done_button = QPushButton("Marks complete")
        self.done_button.setToolTip("Mark this step done for the case")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("annotate"))

        top = QHBoxLayout()
        top.addWidget(QLabel("Recording"))
        top.addWidget(self.picker, 1)
        top.addWidget(self.draw_button)
        top.addWidget(self.progress)
        at = QHBoxLayout()
        at.addWidget(QLabel("Time"))
        at.addWidget(self.time)
        at.addWidget(self.time_label, 1)
        at.addWidget(self.trace_button)
        at.addWidget(self.review_button)
        add = QHBoxLayout()
        add.addWidget(self.kind)
        add.addWidget(QLabel("lasting"))
        add.addWidget(self.duration)
        add.addWidget(self.note, 1)
        add.addWidget(self.add_button)
        add.addWidget(self.delete_button)
        add.addWidget(self.done_button)
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addWidget(self.canvas)
        box.addWidget(self.overview_note)
        box.addLayout(at)
        box.addWidget(self.marks_table, 1)
        box.addLayout(add)
        self._marks = pd.DataFrame(columns=list(bids.EVENT_COLUMNS))
        self._overview = None

    # -- the recording ------------------------------------------------------------
    @property
    def run(self):
        return self.picker.currentData()

    def refresh(self) -> None:
        self.picker.fill(self.window_.case)
        self.load()

    def load(self) -> None:
        if self.run is None:
            self._marks = pd.DataFrame(columns=list(bids.EVENT_COLUMNS))
            self._fill_marks()
            return
        rec = self.window_.case.recording(self.run)
        self.time.setMaximum(max(0.0, rec.duration_s))
        self._marks = self.window_.case.marks(self.run)
        self._overview = overview.cached_envelope(self.window_.case, rec, compute=False)
        self._fill_marks()
        self._draw()
        self._show_time()

    def _start_of_day(self) -> str:
        try:
            sidecar = json.loads(self.window_.case.where(self.run).path("ieeg.json").read_text())
        except (OSError, ValueError):
            return ""
        return str(sidecar.get("RecordingStartTimeOfDay") or "")

    def _show_time(self) -> None:
        t = float(self.time.value())
        text = f"+{clock(t)} from the start"
        start = self._start_of_day()
        if start:
            h, m, s = (float(x) for x in start.split(":"))
            total = (h * 3600 + m * 60 + s + t) % 86400
            text += f" · {int(total // 3600):02d}:{int(total % 3600 // 60):02d} on the clock"
        self.time_label.setText(text)

    # -- the overview -----------------------------------------------------------------
    def draw_overview(self, wait: bool = False):
        from onset_review.workers import track

        if self.run is None:
            return None
        case, rec = self.window_.case, self.window_.case.recording(self.run)

        def work(progress, should_stop):
            return overview.cached_envelope(case, rec, progress=progress,
                                            should_stop=should_stop)

        self._job = _Job(work, self)
        self._job.progressed.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._job.finished.connect(self._drawn)
        self.progress.setVisible(True)
        self.draw_button.setEnabled(False)
        track(self._job).start()
        if wait:
            self._job.wait()
            self._drawn()
        return self._job

    def _drawn(self) -> None:
        job = self._job
        if job is None or job.isRunning():
            return
        self._job = None
        self.progress.setVisible(False)
        self.draw_button.setEnabled(True)
        if job.error is not None:
            self.overview_note.setText(f"No overview: {job.error}")
            return
        self._overview = job.result
        self._draw()

    def _draw(self) -> None:
        from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

        self.figure.clear()
        axis = self.figure.add_axes([0.09, 0.32, 0.89, 0.62])
        tracks = self.figure.add_axes([0.09, 0.1, 0.89, 0.18], sharex=axis)
        _quiet(axis)
        rec = self.window_.case.recording(self.run) if self.run is not None else None
        seconds = rec.duration_s if rec else 3600.0
        # The axis in the unit the recording's length reads best in.
        unit, scale = (("hours", 3600.0) if seconds >= 7200 else
                       ("minutes", 60.0) if seconds >= 120 else ("seconds", 1.0))
        self._scale = scale
        frame = self._overview
        if frame is not None and len(frame):
            axis.plot(frame["t"] / scale, frame["line_length"], color=SERIES[0], linewidth=0.8)
            axis.set_ylabel("line length, µV/s\n(busiest tenth)", fontsize=7, color=MUTED)
            flat = frame[frame["flat_share"] > 0.5]
            for t in flat["t"]:
                axis.axvspan(t / scale, (t + frame.attrs.get("block_s", 30)) / scale,
                             color=MUTED, alpha=0.25, linewidth=0)
            coverage = frame.attrs.get("coverage", 0)
            self.overview_note.setText(
                f"Sampled: {frame.attrs.get('sample_s', 2):g} s of every "
                f"{frame.attrs.get('block_s', 30):g} s ({coverage:.0%} of the recording), "
                f"over {frame.attrs.get('n_contacts', '?')} contacts. Grey: most contacts flat. "
                "Click to choose a time.")
        else:
            axis.text(0.5, 0.5, "Draw the overview to see the recording at a glance",
                      transform=axis.transAxes, ha="center", va="center", color=MUTED,
                      fontsize=9)
            self.overview_note.setText("")
        _quiet(tracks)
        tracks.set_yticks([0.5, 1.5, 2.5])
        tracks.set_yticklabels(["artefact", "sleep", "seizure"], fontsize=7)
        tracks.set_ylim(0, 3)
        stage_level = {"sleep-W": 0.95, "sleep-REM": 0.75, "sleep-N1": 0.55, "sleep-N2": 0.35,
                       "sleep-N3": 0.15}
        for row in self._marks.itertuples():
            start, length = float(row.onset) / scale, max(float(row.duration), 1.0) / scale
            kind = str(row.trial_type)
            if kind.startswith("seizure"):
                tracks.axvspan(start, start + length, ymin=2 / 3, ymax=1, color=SERIES[1],
                               alpha=0.8, linewidth=0)
            elif kind in stage_level:
                level = 1 + stage_level[kind]
                tracks.plot([start, start + length], [level, level], color=SERIES[0],
                            linewidth=2)
            elif kind == "artefact":
                tracks.axvspan(start, start + length, ymin=0, ymax=1 / 3, color=MUTED,
                               alpha=0.6, linewidth=0)
        self._cursor = axis.axvline(float(self.time.value()) / scale, color=MUTED,
                                    linewidth=1.2, alpha=0.7)
        axis.set_xlim(0, max(seconds / scale, 1e-3))
        tracks.set_xlabel(f"{unit} from the start", fontsize=7, color=MUTED)
        axis.yaxis.grid(True, color=GRID, linewidth=0.6)
        axis.tick_params(labelbottom=False)
        self.canvas.draw_idle()

    def _clicked(self, event) -> None:
        if event.xdata is None:
            return
        self.time.setValue(max(0.0, float(event.xdata) * getattr(self, "_scale", 3600.0)))
        if getattr(self, "_cursor", None) is not None:
            self._cursor.set_xdata([event.xdata, event.xdata])
            self.canvas.draw_idle()

    # -- marks ----------------------------------------------------------------------
    def _fill_marks(self) -> None:
        frame = self._marks.sort_values("onset").reset_index(drop=True)
        self._marks = frame
        self.marks_table.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples()):
            values = (row.trial_type, clock(row.onset), f"{float(row.duration):g} s",
                      "all" if str(row.channels) in ("n/a", "") else str(row.channels),
                      "" if str(row.value) == "n/a" else str(row.value), str(row.source),
                      "" if str(row.by) == "n/a" else str(row.by))
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if c == 0:
                    item.setToolTip(marks.label_of(str(row.trial_type)))
                self.marks_table.setItem(r, c, item)

    def _save_marks(self) -> None:
        self.window_.case.save_marks(self.run, self._marks)
        self._marks = self.window_.case.marks(self.run)
        self._fill_marks()
        self._draw()
        self.changed.emit()

    def add_mark(self, kind: str | None = None, onset: float | None = None,
                 duration: float | None = None, note: str | None = None) -> dict | None:
        if self.run is None:
            return None
        row = marks.new_mark(float(self.time.value()) if onset is None else onset,
                             kind or self.kind.currentData(),
                             float(self.duration.value()) if duration is None else duration,
                             value=self.note.text().strip() if note is None else note,
                             source="reader", by=self.window_.reader())
        self._marks = pd.concat([self._marks, pd.DataFrame([row])], ignore_index=True)
        self.note.clear()
        self._save_marks()
        return row

    def remove_selected(self) -> int:
        rows = sorted({i.row() for i in self.marks_table.selectedIndexes()}, reverse=True)
        if not rows:
            return 0
        self._marks = self._marks.drop(index=rows).reset_index(drop=True)
        self._save_marks()
        return len(rows)

    def _go_to_mark(self) -> None:
        rows = self.marks_table.selectionModel().selectedRows()
        if rows:
            self.time.setValue(float(self._marks.iloc[rows[0].row()]["onset"]))
            self._draw()

    # -- the trace, in MNE's browser ----------------------------------------------------
    def open_trace(self, show: bool = True):
        """MNE's browser at the chosen time, the case's marks as annotations,
        the case's vocabulary in its annotation list."""
        if self.run is None:
            return None
        raw = bids.read_run(self.window_.case.where(self.run), preload=False)
        start = max(0.0, float(self.time.value()) - TRACE_SECONDS / 4)
        browser = raw.plot(start=start, duration=TRACE_SECONDS,
                           n_channels=min(30, len(raw.ch_names)), show=show, block=False,
                           title=f"Case {self.window_.case.case_id}, run {self.run}")
        self._seed_vocabulary(browser)
        self.trace, self._trace_raw = browser, raw
        closed = getattr(browser, "gotClosed", None)
        if closed is not None:
            closed.connect(self.take_marks_from_trace)
        return browser

    @staticmethod
    def _seed_vocabulary(browser) -> None:
        dock = getattr(getattr(browser, "mne", None), "fig_annotation", None)
        if dock is None:
            return
        try:
            have = set(browser.mne.new_annotation_labels) | \
                set(browser.mne.inst.annotations.description)
            for kind in marks.KINDS:
                if kind not in have:
                    dock._add_description(kind)
            browser.mne.current_description = "seizure-onset"
            dock.description_cmbx.setCurrentText("seizure-onset")
        except Exception:       # noqa: BLE001 - the browser's internals; marks still work
            pass

    def take_marks_from_trace(self) -> int:
        """The browser's annotations are the run's marks now: what was added
        there is added (as the reader's), what was removed is removed, and a
        mark that did not move keeps who made it and its text."""
        raw = self._trace_raw
        if raw is None or self.run is None:
            return 0
        before = {(round(float(r.onset), 3), str(r.trial_type)): r._asdict()
                  for r in self._marks.itertuples(index=False)}
        rows = []
        for item in raw.annotations:
            text = str(item["description"])
            kind = text if text in marks.KINDS else marks.kind_of(text)
            key = (round(float(item["onset"]), 3), kind)
            if key in before and abs(float(before[key]["duration"]) - float(item["duration"])) \
                    < 1e-3:
                rows.append(before[key])
                continue
            rows.append(marks.new_mark(float(item["onset"]), kind, float(item["duration"]),
                                       tuple(item.get("ch_names", ()) or ()),
                                       value="" if text == kind else text, source="reader",
                                       by=self.window_.reader()))
        changed = len(rows) != len(before) or any(
            (round(float(r["onset"]), 3), str(r["trial_type"])) not in before for r in rows)
        self._trace_raw, self.trace = None, None
        if changed:
            self._marks = pd.DataFrame(rows, columns=list(bids.EVENT_COLUMNS))
            self._save_marks()
        return len(rows)

    def review_request(self):
        from onset_review.session import ReviewRequest

        case, rec = self.window_.case, self.window_.case.recording(self.run)
        where = case.where(rec)
        channels = case.channels(rec)
        sidecar = json.loads(where.path("ieeg.json").read_text())
        start = float(self.time.value())
        # A minute from the chosen time; near the end, the last minute.
        stop = min(rec.duration_s, start + 60.0)
        start = max(0.0, min(start, stop - 60.0))
        types = tuple((str(r.name), bids.MNE_TYPES.get(str(r.type).upper(), "misc"))
                      for r in channels.itertuples())
        exclude = tuple(str(r.name) for r in channels.itertuples() if str(r.status) == "bad")
        from onset_hfo.config import PreprocessConfig

        return ReviewRequest(dataset="", subject=case.case_id, run=rec.run, path=where.header,
                             t_start=start, t_stop=stop, channel_types=types,
                             line_freq=float(sidecar.get("PowerLineFrequency") or 50.0),
                             preprocess=PreprocessConfig(exclude=exclude) if exclude else None)

    def open_review(self):
        if self.run is None:
            return None
        request = self.review_request()
        self.window_.openRequested.emit(request)
        return request


# -- phase 2: segments, the analysis, the interictal run, review ------------------------------
def _hours(seconds: float) -> str:
    return "–" if seconds == float("inf") else f"{seconds / 3600:.1f} h"


class SegmentsPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        from onset_hfo.case import segments as segments_module

        self.window_ = window
        self.stages = {}
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Sleep stages"))
        for stage in segments_module.STAGE_NAMES:
            check = QCheckBox(stage.replace("sleep-", ""))
            check.setChecked(stage in segments_module.SegmentRule().stages)
            self.stages[stage] = check
            stage_row.addWidget(check)
        stage_row.addWidget(_muted("(none ticked: any time)"))
        stage_row.addStretch(1)
        self.from_seizure = QDoubleSpinBox()
        self.from_seizure.setRange(0.0, 48.0)
        self.from_seizure.setSingleStep(0.5)
        self.from_seizure.setSuffix(" h")
        self.from_seizure.setValue(1.0)
        self.segment_min = QDoubleSpinBox()
        self.segment_min.setRange(1.0, 60.0)
        self.segment_min.setSuffix(" min")
        self.segment_min.setValue(5.0)
        self.total_min = QDoubleSpinBox()
        self.total_min.setRange(1.0, 1440.0)
        self.total_min.setSuffix(" min")
        self.total_min.setValue(30.0)
        self.avoid_flat = QCheckBox("Away from stretches the overview found flat")
        self.avoid_flat.setChecked(True)
        form = QFormLayout()
        form.addRow(stage_row)
        form.addRow("At least, from any seizure", self.from_seizure)
        form.addRow("Each segment", self.segment_min)
        form.addRow("In all", self.total_min)
        form.addRow(self.avoid_flat)
        self.choose_button = QPushButton("Choose segments")
        self.choose_button.setObjectName("onset_case_choose")
        self.choose_button.clicked.connect(lambda _=False: self.choose())
        self.done_button = QPushButton("Segments chosen")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("segments"))
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("onset_case_segments")
        self.table.setHorizontalHeaderLabels(["Segment", "Run", "From", "To", "Stage",
                                              "Nearest seizure"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.choose_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        box = QVBoxLayout(self)
        box.addWidget(_muted("Interictal rates are read from stretches chosen by rule: by sleep "
                             "stage, away from seizures, artefacts and flat recording. The "
                             "defaults are starting points; the report states the rule."))
        box.addLayout(form)
        box.addLayout(buttons)
        box.addWidget(self.summary)
        box.addWidget(self.table, 1)

    def rule(self):
        from onset_hfo.case.segments import SegmentRule

        return SegmentRule(
            stages=tuple(s for s, check in self.stages.items() if check.isChecked()),
            min_from_seizure_s=float(self.from_seizure.value()) * 3600,
            segment_s=float(self.segment_min.value()) * 60,
            total_s=float(self.total_min.value()) * 60,
            avoid_flat=self.avoid_flat.isChecked())

    def refresh(self) -> None:
        from onset_hfo.case.segments import describe_choice, load_segments

        frame, rule, info = load_segments(self.window_.case)
        if rule is not None:
            for stage, check in self.stages.items():
                check.setChecked(stage in rule.stages)
            self.from_seizure.setValue(rule.min_from_seizure_s / 3600)
            self.segment_min.setValue(rule.segment_s / 60)
            self.total_min.setValue(rule.total_s / 60)
            self.avoid_flat.setChecked(rule.avoid_flat)
            self.summary.setText(describe_choice(frame, info, rule))
        self._fill(frame)

    def _fill(self, frame) -> None:
        self.table.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples()):
            values = (row.segment, row.run, clock(row.start), clock(row.stop),
                      str(row.stage).replace("sleep-", ""),
                      _hours(float(row.nearest_seizure_s)))
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))

    def choose(self):
        from onset_hfo.case import segments as segments_module

        case = self.window_.case
        recordings = []
        for rec in case.recordings:
            flat = []
            cached = overview.cached_envelope(case, rec, compute=False)
            if cached is not None and len(cached):
                block = float(cached.attrs.get("block_s", 30.0))
                flat = [(float(t), float(t) + block)
                        for t in cached.loc[cached["flat_share"] > 0.5, "t"]]
            recordings.append({"run": rec.run, "duration": rec.duration_s,
                               "marks": case.marks(rec), "flat": flat})
        rule = self.rule()
        frame, info = segments_module.choose(recordings, rule)
        segments_module.save_segments(case, frame, rule, info)
        self.summary.setText(segments_module.describe_choice(frame, info, rule))
        self._fill(frame)
        self.changed.emit()
        return frame


class AnalysisSettingsPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        from onset_hfo.detectors import HFO_DETECTORS
        from onset_hfo.preprocess import REFERENCES

        self.window_ = window
        self.reference = QComboBox()
        for name in REFERENCES:
            self.reference.addItem(name, name)
        self.grid = QLineEdit()
        self.grid.setPlaceholderText("for the Laplacian, without positions: G:8")
        self.band = QComboBox()
        self.band.addItem("ripple (80–250 Hz)", "ripple")
        self.band.addItem("fast ripple (250–500 Hz)", "fast_ripple")
        self.detectors = {}
        detector_row = QHBoxLayout()
        for name in HFO_DETECTORS:
            check = QCheckBox(name.replace("_", " "))
            self.detectors[name] = check
            detector_row.addWidget(check)
        detector_row.addStretch(1)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 20.0)
        self.threshold.setSingleStep(0.5)
        self.threshold.setSuffix(" SD")
        self.threshold.setSpecialValueText("each detector's default")
        self.quality = QCheckBox("Quality stage: set aside bad contacts and seconds")
        self.spikes = QCheckBox("Note which events ride on a spike")
        self.describe = _muted("")
        form = QFormLayout()
        form.addRow("Reference", self.reference)
        form.addRow("Grid columns", self.grid)
        form.addRow("Band", self.band)
        form.addRow("Detectors (the first ticked ranks)", detector_row)
        form.addRow("Threshold", self.threshold)
        form.addRow(self.quality)
        form.addRow(self.spikes)
        self.save_button = QPushButton("Save the settings")
        self.save_button.clicked.connect(lambda _=False: self.save())
        self.done_button = QPushButton("Settings set")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("preprocess"))
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        box = QVBoxLayout(self)
        box.addWidget(_muted("How every segment of the case is analysed: the same settings for "
                             "all of them, written into the results. The rest of the "
                             "preprocessing (filters, notch at the recording's mains) is the "
                             "project's measured default."))
        box.addLayout(form)
        box.addWidget(self.describe)
        box.addLayout(buttons)
        box.addStretch(1)

    def refresh(self) -> None:
        from onset_hfo.config import PreprocessConfig
        from onset_hfo.preprocess import effective_reference
        from onset_review.caseinterictal import template_for

        template = template_for(self.window_.case)
        preprocess = template.preprocess or PreprocessConfig()
        self.reference.setCurrentIndex(max(0, self.reference.findData(
            effective_reference(preprocess))))
        self.grid.setText(", ".join(f"{g}:{c}" for g, c in preprocess.grid_columns))
        self.band.setCurrentIndex(max(0, self.band.findData(template.band)))
        for name, check in self.detectors.items():
            check.setChecked(name in template.detectors)
        self.threshold.setValue(float(template.threshold_sd or 0.0))
        self.quality.setChecked(bool(template.check_quality))
        self.spikes.setChecked(bool(template.with_spikes))
        self._describe(template)

    def _describe(self, template) -> None:
        from onset_hfo.config import PreprocessConfig
        from onset_hfo.preprocess import describe

        sentence, _warnings = describe(template.preprocess or PreprocessConfig(),
                                       template.band_hz, 2000.0)
        self.describe.setText("Each segment will " + sentence + ".")

    def template(self):
        import dataclasses

        from onset_hfo.config import PreprocessConfig
        from onset_hfo.preprocess import parse_grid_columns
        from onset_review.caseinterictal import template_for

        base = template_for(self.window_.case)
        scheme = self.reference.currentData()
        preprocess = dataclasses.replace(
            base.preprocess or PreprocessConfig(), reference=scheme,
            bipolar=scheme == "bipolar", average_reference=scheme == "average",
            grid_columns=parse_grid_columns(self.grid.text()) if scheme == "laplacian" else ())
        detectors = tuple(n for n, c in self.detectors.items() if c.isChecked())
        if not detectors:
            raise ValueError("Tick at least one detector.")
        return dataclasses.replace(base, preprocess=preprocess, band=self.band.currentData(),
                                   detectors=detectors,
                                   threshold_sd=float(self.threshold.value()) or None,
                                   check_quality=self.quality.isChecked(),
                                   with_spikes=self.spikes.isChecked())

    def save(self):
        from onset_review.caseinterictal import save_template

        try:
            template = self.template()
        except ValueError as error:
            self.describe.setText(str(error))
            return None
        save_template(self.window_.case, template, by=self.window_.reader())
        self._describe(template)
        self.changed.emit()
        return template


class InterictalPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self._job = None
        self.result = None
        self.run_button = QPushButton("Run over the segments")
        self.run_button.setObjectName("onset_case_run")
        self.run_button.clicked.connect(lambda _=False: self.run())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self._job and self._job.stop())
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.what = _muted("")
        self.statement = QLabel("")
        self.statement.setObjectName("onset_case_statement")
        self.statement.setWordWrap(True)
        self.statement.setTextInteractionFlags(Qt.TextSelectableByMouse)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5.0, 4.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(360)
        self.table = QTableWidget(0, 8)
        self.table.setObjectName("onset_case_pooled")
        self.table.setHorizontalHeaderLabels(["Rank", "Channel", "Events", "Minutes",
                                              "Rate /min", "Interval", "Tied",
                                              "Segments tied"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.done_button = QPushButton("Interictal done")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("interictal"))
        self.table.verticalHeader().setVisible(False)
        top = QHBoxLayout()
        top.addWidget(self.run_button)
        top.addWidget(self.stop_button)
        top.addWidget(self.progress, 1)
        top.addStretch(1)
        top.addWidget(self.done_button)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(self.table)
        split.setSizes([480, 620])
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addWidget(self.what)
        box.addWidget(self.statement)
        box.addWidget(split, 1)

    def refresh(self) -> None:
        from onset_hfo.case.segments import load_segments
        from onset_review.caseinterictal import latest_result, template_for

        case = self.window_.case
        frame, _rule, _info = load_segments(case)
        template = template_for(case)
        self.run_button.setEnabled(len(frame) > 0)
        self.what.setText(
            (f"{len(frame)} segment(s), {frame['duration'].sum() / 60:.0f} minutes"
             if len(frame) else "No segments chosen yet: choose them on the Segments step")
            + f" · detectors {', '.join(template.detectors)} (ranked by "
              f"{template.primary}) · {template.band_label()} · threshold "
            + (f"{template.threshold_sd:g} SD" if template.threshold_sd else "each "
               "detector's default")
            + ". Each segment is analysed as the review window would; analysed segments are "
              "kept, so a second run costs only what is new.")
        if self.result is None:
            self.result = latest_result(case)
        self._show()

    def run(self, wait: bool = False):
        from onset_hfo.case.segments import load_segments
        from onset_review.caseinterictal import run_interictal
        from onset_review.workers import track

        case = self.window_.case
        frame, _rule, _info = load_segments(case)
        if not len(frame):
            return None

        def work(progress, should_stop):
            return run_interictal(case, frame, progress=progress, should_stop=should_stop)

        self._job = _Job(work, self)
        self._job.progressed.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._job.finished.connect(self._ran)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        track(self._job).start()
        if wait:
            self._job.wait()
            self._ran()
        return self._job

    def _ran(self) -> None:
        job = self._job
        if job is None or job.isRunning():
            return
        self._job = None
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if job.error is not None:
            self.statement.setText(f"Not finished: {job.error}")
            return
        self.result = job.result
        self._show()
        self.changed.emit()

    def _show(self) -> None:
        from onset_review.caseinterictal import draw_pooled

        result = self.result
        self.figure.clear()
        if result is None:
            self.statement.setText("")
            self.table.setRowCount(0)
            self.canvas.draw_idle()
            return
        self.statement.setText(result.statement() + f" (Results: {result.folder.name}.)")
        draw_pooled(self.figure, self.figure.add_subplot(111), result)
        self.canvas.draw_idle()
        rows = result.table.to_dict("records")
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            rank = "" if pd.isna(row["rank"]) else str(int(row["rank"]))
            rate = "–" if pd.isna(row["rate_per_min"]) else f"{row['rate_per_min']:.2f}"
            values = (rank, row["channel"], str(int(row["n_events"])),
                      f"{row['minutes']:.1f}", rate,
                      f"{row['rate_ci_low']:.2f}–{row['rate_ci_high']:.2f}",
                      "yes" if row["tied"] else "",
                      f"{int(row['segments_tied'])} of {int(row['segments_analysed'])}")
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))


class IctalPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self._job = None
        self.result = None
        self.run_button = QPushButton("Run over the seizures")
        self.run_button.setObjectName("onset_case_run_ictal")
        self.run_button.clicked.connect(lambda _=False: self.run())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self._job and self._job.stop())
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.done_button = QPushButton("Ictal done")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("ictal"))
        self.what = _muted("")
        self.seizures = QTableWidget(0, 5)
        self.seizures.setObjectName("onset_case_seizures")
        self.seizures.setHorizontalHeaderLabels(["Seizure", "Run", "Onset", "Marked as",
                                                 "Status"])
        self.seizures.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.seizures.verticalHeader().setVisible(False)
        self.seizures.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.seizures.horizontalHeader().setStretchLastSection(True)
        self.seizures.setMaximumHeight(150)
        self.statement = QLabel("")
        self.statement.setObjectName("onset_case_ictal_statement")
        self.statement.setWordWrap(True)
        self.statement.setTextInteractionFlags(Qt.TextSelectableByMouse)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5.0, 4.6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(360)
        self.table = QTableWidget(0, 5)
        self.table.setObjectName("onset_case_ictal")
        self.table.setHorizontalHeaderLabels(["Rank", "Channel", "Median index",
                                              "Seizures at or above 0.3",
                                              "Median change (s)"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        top = QHBoxLayout()
        top.addWidget(self.run_button)
        top.addWidget(self.stop_button)
        top.addWidget(self.progress, 1)
        top.addStretch(1)
        top.addWidget(self.done_button)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(self.table)
        split.setSizes([480, 620])
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addWidget(self.what)
        box.addWidget(self.seizures)
        box.addWidget(self.statement)
        box.addWidget(split, 1)

    def refresh(self) -> None:
        from onset_hfo.ictal import IctalSettings
        from onset_review.caseictal import latest_ictal, seizures_of

        case = self.window_.case
        frame = seizures_of(case)
        usable = int(frame["usable"].sum()) if len(frame) else 0
        self.run_button.setEnabled(usable > 0)
        self.what.setText(
            (f"{len(frame)} seizure(s) marked, {usable} with an electrographic onset to "
             "analyse" if len(frame) else "No seizure is marked yet: mark each seizure's "
             "electrographic onset on the Annotate step")
            + ". Per seizure, per channel: the Epileptogenicity Index — "
            + IctalSettings().describe()
            + ". A research measure; it needs the electrographic onset marked by a person.")
        if self.result is None:
            self.result = latest_ictal(case)
        self._show_seizures(frame)
        self._show()

    def _show_seizures(self, frame) -> None:
        states = {}
        if self.result is not None and len(self.result.seizures):
            states = {(str(r.run), round(float(r.onset), 3)): str(r.status)
                      for r in self.result.seizures.itertuples()}
        self.seizures.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples()):
            state = states.get((str(row.run), round(float(row.onset), 3)))
            if state is None:
                state = "to analyse" if row.usable else f"left out: {row.note}"
            elif state == "ok":
                state = "analysed"
            values = (row.seizure, row.run, clock(row.onset), row.marker, state)
            for c, value in enumerate(values):
                self.seizures.setItem(r, c, QTableWidgetItem(str(value)))

    def run(self, wait: bool = False):
        from onset_review.caseictal import run_ictal
        from onset_review.workers import track

        case = self.window_.case

        def work(progress, should_stop):
            return run_ictal(case, progress=progress, should_stop=should_stop)

        self._job = _Job(work, self)
        self._job.progressed.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._job.finished.connect(self._ran)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        track(self._job).start()
        if wait:
            self._job.wait()
            self._ran()
        return self._job

    def _ran(self) -> None:
        from onset_review.caseictal import seizures_of

        job = self._job
        if job is None or job.isRunning():
            return
        self._job = None
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if job.error is not None:
            self.statement.setText(f"Not finished: {job.error}")
            return
        self.result = job.result
        self._show_seizures(seizures_of(self.window_.case))
        self._show()
        self.changed.emit()

    def _show(self) -> None:
        from onset_review.caseictal import draw_ictal

        result = self.result
        self.figure.clear()
        if result is None:
            self.statement.setText("")
            self.table.setRowCount(0)
            self.canvas.draw_idle()
            return
        self.statement.setText(result.statement() + f" (Results: {result.folder.name}.)")
        if len(result.combined):
            draw_ictal(self.figure, self.figure.add_subplot(111), result)
        self.canvas.draw_idle()
        rows = result.combined.to_dict("records")
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            change = "–" if pd.isna(row["median_change_s"]) else f"{row['median_change_s']:+.1f}"
            values = (str(int(row["rank"])), row["channel"], f"{row['median_ei']:.2f}",
                      f"{int(row['seizures_high'])} of {int(row['seizures'])}", change)
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))


class ReviewPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("onset_case_review")
        self.table.setHorizontalHeaderLabels(["Segment", "Run", "From", "Stage",
                                              "Accepted events", "Verdicts recorded"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemDoubleClicked.connect(lambda _item: self.open_selected())
        self.open_button = QPushButton("Open in the review window")
        self.open_button.clicked.connect(lambda _=False: self.open_selected())
        self.done_button = QPushButton("Review done")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("review"))
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        buttons.addWidget(self.done_button)
        box = QVBoxLayout(self)
        box.addWidget(_muted("Each segment opens in the review window, analysed as it was in "
                             "the pooled run; the verdicts given there are counted here."))
        box.addWidget(self.table, 1)
        box.addLayout(buttons)
        self._frame = pd.DataFrame()

    def refresh(self) -> None:
        from onset_hfo.case.segments import load_segments
        from onset_review import adjudication
        from onset_review.caseinterictal import latest_result, segment_request, template_for

        case = self.window_.case
        frame, _rule, _info = load_segments(case)
        self._frame = frame
        result = latest_result(case)
        counts = {}
        if result is not None and len(result.per_segment):
            counts = result.per_segment.groupby("segment")["n_events"].sum().to_dict()
        template = template_for(case)
        self.table.setRowCount(len(frame))
        for r, row in enumerate(frame.itertuples()):
            read = adjudication.load(segment_request(case, str(row.run), float(row.start),
                                                     float(row.stop), template))
            verdicts = len(read.events) + len(read.channels)
            values = (row.segment, row.run, clock(row.start),
                      str(row.stage).replace("sleep-", ""),
                      str(int(counts.get(row.segment, 0))) if counts else "–", str(verdicts))
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))

    def open_selected(self):
        from onset_review.caseinterictal import segment_request, template_for

        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._frame):
            return None
        row = self._frame.iloc[rows[0].row()]
        request = segment_request(self.window_.case, str(row["run"]), float(row["start"]),
                                  float(row["stop"]), template_for(self.window_.case))
        self.window_.openRequested.emit(request)
        return request


class LaterPage(QWidget):
    def __init__(self, title: str, phase: int, parent=None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        label = QLabel(f"<b>{title}</b> arrives in phase {phase} of the clinic edition.")
        label.setWordWrap(True)
        box.addWidget(label)
        box.addWidget(_muted("Until then the review window does this one recording window at a "
                             "time: choose a time on the Annotate step and Analyse this minute."))
        box.addStretch(1)


# -- the window -------------------------------------------------------------------------------
class CaseWindow(QMainWindow):
    """One case, step by step."""

    #: A request to open in the review window.
    openRequested = Signal(object)

    def __init__(self, case: Case, reader: str = "", parent=None):
        super().__init__(parent)
        self.case = case
        self._reader = reader
        self.setWindowTitle(f"Case {case.case_id} — onset-review")
        self.resize(1280, 820)
        self.steps = QListWidget()
        self.steps.setObjectName("onset_case_steps")
        self.steps.setFixedWidth(220)
        self.pages = QStackedWidget()
        self.import_page = ImportPage(self)
        self.channels_page = ChannelsPage(self)
        self.annotate_page = AnnotatePage(self)
        self.segments_page = SegmentsPage(self)
        self.settings_page = AnalysisSettingsPage(self)
        self.interictal_page = InterictalPage(self)
        self.ictal_page = IctalPage(self)
        self.review_page = ReviewPage(self)
        own = {"import": self.import_page, "channels": self.channels_page,
               "annotate": self.annotate_page, "segments": self.segments_page,
               "preprocess": self.settings_page, "interictal": self.interictal_page,
               "ictal": self.ictal_page, "review": self.review_page}
        self.page_for: dict[str, QWidget] = {}
        for key, title, phase in STEPS:
            page = own.get(key) or LaterPage(title, phase)
            self.page_for[key] = page
            self.pages.addWidget(page)
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, key)
            if key not in own:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip(f"Arrives in phase {phase}")
            self.steps.addItem(item)
        self.steps.currentRowChanged.connect(self._show_step)
        for page in own.values():
            page.changed.connect(self.refresh)
        self.header = QLabel("")
        self.header.setObjectName("onset_case_header")
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body = QWidget()
        right = QVBoxLayout(body)
        right.addWidget(self.header)
        right.addWidget(self.pages, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.steps)
        split.addWidget(body)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self.refresh()
        self.steps.setCurrentRow(0)

    def reader(self) -> str:
        if self._reader:
            return self._reader
        from onset_review.adjudication import reader_name

        return reader_name() or ""

    def show_step(self, key: str) -> bool:
        for row in range(self.steps.count()):
            item = self.steps.item(row)
            if item.data(Qt.UserRole) == key and item.flags() & Qt.ItemIsEnabled:
                self.steps.setCurrentRow(row)
                return True
        return False

    def current_step(self) -> str:
        item = self.steps.currentItem()
        return item.data(Qt.UserRole) if item is not None else ""

    def _show_step(self, row: int) -> None:
        item = self.steps.item(row)
        if item is None:
            return
        page = self.page_for[item.data(Qt.UserRole)]
        self.pages.setCurrentWidget(page)
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()

    def complete(self, step: str) -> None:
        self.case.mark_step(step, True, by=self.reader())
        self.refresh()

    def refresh(self) -> None:
        case = self.case
        hours = sum(r.duration_s for r in case.recordings) / 3600
        self.header.setText(f"<b>Case {case.case_id}</b> — {len(case.recordings)} recording(s), "
                            f"{hours:.1f} h · {case.root}")
        for row in range(self.steps.count()):
            item = self.steps.item(row)
            key = item.data(Qt.UserRole)
            title = next(t for k, t, _p in STEPS if k == key)
            item.setText(("✓ " if case.step_done(key) else "    ") + title)

    def closeEvent(self, event) -> None:      # noqa: N802
        page = self.annotate_page
        if page.trace is not None:
            try:
                page.trace.close()
            except Exception:       # noqa: BLE001
                pass
        super().closeEvent(event)


def new_case_dialog(parent=None) -> Case | None:
    """Ask for a folder and a pseudonym; make the case."""
    folder = QFileDialog.getExistingDirectory(parent, "Where the case will live (a new or "
                                              "empty folder)", str(Path.home()))
    if not folder:
        return None
    name, ok = QInputDialog.getText(parent, "New case", "The case's pseudonym — never the "
                                    "patient's name (letters and digits, e.g. P017):")
    if not ok or not name.strip():
        return None
    try:
        return Case.create(Path(folder), name.strip())
    except (FileExistsError, ValueError) as error:
        QMessageBox.warning(parent, "New case", str(error))
        return None


def open_case_dialog(parent=None) -> Case | None:
    folder = QFileDialog.getExistingDirectory(parent, "Open a case", str(Path.home()))
    if not folder:
        return None
    try:
        return Case.open(folder)
    except FileNotFoundError as error:
        QMessageBox.warning(parent, "Open a case", str(error))
        return None
