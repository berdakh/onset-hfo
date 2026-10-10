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
* **Map** places the contacts on the template brain (imported from a planning
  system, or planned as shafts, strips and grids), labels them with an atlas
  as *probable* structures, takes the clinician's onset zone, and puts the
  interictal rate, the ictal index and the zone side by side
  (`onset_hfo.case.electrodes`, `onset_review.casemap`).
* **Report** checks the case is ready (analyses run and reviewed,
  de-identification checked, the audit log intact), takes reviewers'
  sign-offs against the content they approve, and produces numbered PDF
  versions (`onset_review.casereport`, `onset_hfo.case.deid`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from qtpy.QtCore import QSize, Qt, QThread, Signal
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


class PlanDialog(QDialog):
    """Place an electrode on the template: a depth shaft from its target to its
    entry, or a strip or grid from its first contact. Points are typed in MNI
    millimetres or taken from an atlas structure's centre."""

    def __init__(self, atlas=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plan an electrode on the template")
        self.atlas = atlas
        self.group = QLineEdit()
        self.group.setPlaceholderText("the electrode's name, e.g. LA")
        self.kind = QComboBox()
        for label, key in (("Depth (SEEG)", "depth"), ("Strip", "strip"), ("Grid", "grid")):
            self.kind.addItem(label, key)
        self.points = {}
        form = QFormLayout()
        form.addRow("Electrode", self.group)
        form.addRow("Kind", self.kind)
        for key, title in (("first", "Target / first contact"), ("second", "Entry / along"),
                           ("third", "Across (grids)")):
            row = QHBoxLayout()
            boxes = []
            for axis in "xyz":
                box = QDoubleSpinBox()
                box.setRange(-120.0, 120.0)
                box.setDecimals(1)
                box.setSuffix(f" {axis}")
                boxes.append(box)
                row.addWidget(box)
            choose = QComboBox()
            choose.addItem("or a structure's centre…", "")
            if atlas is not None:
                for name in atlas.regions():
                    choose.addItem(name, name)
            choose.currentIndexChanged.connect(
                lambda _i, c=choose, b=boxes: self._from_structure(c, b))
            row.addWidget(choose, 1)
            self.points[key] = boxes
            form.addRow(title, row)
        self.count = QDoubleSpinBox()
        self.count.setDecimals(0)
        self.count.setRange(1, 64)
        self.count.setValue(8)
        self.rows = QDoubleSpinBox()
        self.rows.setDecimals(0)
        self.rows.setRange(1, 16)
        self.rows.setValue(1)
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.5, 20.0)
        self.spacing.setValue(5.0)
        self.spacing.setSuffix(" mm on the template")
        self.spacing.setToolTip("On real implants warped to the template the spacing was "
                                "5.0–6.4 mm for most shafts, and planning at a smaller "
                                "catalogue pitch put contacts a median 10 mm off "
                                "(docs/TEMPLATE_MAP.md).")
        form.addRow("Contacts (per row)", self.count)
        form.addRow("Rows (grids)", self.rows)
        form.addRow("Spacing", self.spacing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(_muted("Depth: contact 1 at the target (the deepest point), the rest "
                             "towards the entry. Strip or grid: contact 1 at the first point, "
                             "numbered along the second, rows towards the third; flat, where "
                             "the cortex is not. Template positions are approximate."))
        box.addLayout(form)
        box.addWidget(buttons)

    def _from_structure(self, choose, boxes) -> None:
        name = choose.currentData()
        if not name or self.atlas is None:
            return
        for box, value in zip(boxes, self.atlas.centroid(name), strict=True):
            box.setValue(float(value))

    def point(self, key: str):
        return [b.value() for b in self.points[key]]

    def plan(self):
        from onset_hfo.case.electrodes import plan_depth, plan_sheet

        group = self.group.text().strip().upper()
        if not group:
            raise ValueError("name the electrode")
        kind = self.kind.currentData()
        n = int(self.count.value())
        if kind == "depth":
            return plan_depth(group, self.point("first"), self.point("second"), n,
                              self.spacing.value())
        rows = 1 if kind == "strip" else int(self.rows.value())
        return plan_sheet(group, self.point("first"), self.point("second"),
                          self.point("third"), rows, n, self.spacing.value())


class MapPage(QWidget):
    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self._job = None
        self.atlas = None
        self.import_button = QPushButton("Import coordinates…")
        self.import_button.clicked.connect(lambda _=False: self.import_dialog())
        self.plan_button = QPushButton("Plan an electrode…")
        self.plan_button.clicked.connect(lambda _=False: self.plan_dialog())
        self.remove_button = QPushButton("Remove the selected")
        self.remove_button.clicked.connect(lambda _=False: self.remove_selected())
        self.atlas_button = QPushButton("Fetch the atlas (0.6 MB)")
        self.atlas_button.clicked.connect(lambda _=False: self.fetch_atlas())
        self.zone_button = QPushButton("Save the onset zone")
        self.zone_button.setToolTip("The contacts ticked in the Onset zone column: the seizure "
                                    "onset zone as you judge it, for the agreement below.")
        self.zone_button.clicked.connect(lambda _=False: self.save_zone())
        self.view = QComboBox()
        self.view.addItem("from above", "top")
        self.view.addItem("from the side", "side")
        self.view.currentIndexChanged.connect(lambda _i: self._draw())
        self.view3d_button = QPushButton("Open in 3D")
        self.view3d_button.setToolTip("The placed channels on the template in a rotatable 3D "
                                      "page (nilearn), opened in your browser; written to "
                                      "derivatives/onset/map/.")
        self.view3d_button.clicked.connect(lambda _=False: self.open_3d())
        self.done_button = QPushButton("Map done")
        self.done_button.clicked.connect(lambda _=False: self.window_.complete("map"))
        top = QHBoxLayout()
        for widget in (self.import_button, self.plan_button, self.remove_button,
                       self.atlas_button):
            top.addWidget(widget)
        top.addStretch(1)
        top.addWidget(self.done_button)
        imaging_row = self._imaging_row()
        from onset_hfo.case.electrodes import TEMPLATE_NOTE

        self.note = _muted(TEMPLATE_NOTE)
        self.contacts = QTableWidget(0, 5)
        self.contacts.setObjectName("onset_case_contacts")
        self.contacts.setHorizontalHeaderLabels(["Contact", "Onset zone", "x, y, z (mm)",
                                                 "From", "Probably"])
        self.contacts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.contacts.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.contacts.verticalHeader().setVisible(False)
        self.contacts.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.contacts.horizontalHeader().setStretchLastSection(True)
        left = QWidget()
        left_box = QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.addWidget(self.contacts, 1)
        zone_row = QHBoxLayout()
        zone_row.addWidget(self.zone_button)
        zone_row.addStretch(1)
        left_box.addLayout(zone_row)
        self.statement = QLabel("")
        self.statement.setObjectName("onset_case_map_statement")
        self.statement.setWordWrap(True)
        self.statement.setTextInteractionFlags(Qt.TextSelectableByMouse)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(7.0, 3.4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        self.combined = QTableWidget(0, 8)
        self.combined.setObjectName("onset_case_combined")
        self.combined.setHorizontalHeaderLabels(["Channel", "Rate /min", "Interval", "Tied",
                                                 "Index", "Seizures ≥ 0.3", "Probably",
                                                 "Onset zone"])
        self.combined.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.combined.verticalHeader().setVisible(False)
        self.combined.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.combined.horizontalHeader().setStretchLastSection(True)
        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.setContentsMargins(0, 0, 0, 0)
        view_row = QHBoxLayout()
        view_row.addWidget(self.statement, 1)
        view_row.addWidget(self.view)
        view_row.addWidget(self.view3d_button)
        right_box.addLayout(view_row)
        right_box.addWidget(self.canvas, 1)
        right_box.addWidget(self.combined, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([420, 760])
        box = QVBoxLayout(self)
        box.addLayout(top)
        box.addLayout(imaging_row)
        box.addWidget(self.note)
        box.addWidget(split, 1)
        self._table = pd.DataFrame()
        self._names: list[str] = []
        self._locator = None

    def _imaging_row(self) -> QVBoxLayout:
        """The patient's own imaging: their MRI, registered to MNI; contacts in
        it from a file or placed on their CT (`onset_hfo.case.imaging`)."""
        self.mri_button = QPushButton("Add the patient's MRI…")
        self.mri_button.setToolTip("Their T1-weighted MRI (NIfTI). It is registered to the "
                                   "MNI152 template (affine, then a non-linear warp inside "
                                   "the brain) so contacts in it can be mapped.")
        self.mri_button.clicked.connect(lambda _=False: self.mri_dialog())
        self.native_button = QPushButton("Import contacts in the MRI's space…")
        self.native_button.setToolTip("Positions in the patient's own MRI (scanner or ACPC "
                                      "millimetres), as planning or localisation software "
                                      "exports them (BIDS space-T1w).")
        self.native_button.clicked.connect(lambda _=False: self.native_dialog())
        self.ct_button = QPushButton("Place contacts on a CT…")
        self.ct_button.setToolTip("The CT with the electrodes in, aligned to the MRI, opened "
                                  "in MNE's contact locator (the imaging extra).")
        self.ct_button.clicked.connect(lambda _=False: self.ct_dialog())
        self.take_button = QPushButton("Use the placed contacts")
        self.take_button.setToolTip("Bring the contacts placed in the locator into the case.")
        self.take_button.clicked.connect(lambda _=False: self.take_located())
        self.take_button.setVisible(False)
        self.on_mri_button = QPushButton("Show on the MRI")
        self.on_mri_button.setToolTip("The contacts on the patient's own MRI, in three planes.")
        self.on_mri_button.clicked.connect(lambda _=False: self.show_on_mri())
        self.imaging_status = QLabel("")
        self.imaging_status.setObjectName("onset_case_imaging")
        self.imaging_status.setWordWrap(True)
        self.imaging_status.setStyleSheet(
            f"color:{theme.current().text_muted};font-size:9pt;")
        row = QHBoxLayout()
        for widget in (self.mri_button, self.native_button, self.ct_button, self.take_button,
                       self.on_mri_button):
            row.addWidget(widget)
        row.addStretch(1)
        column = QVBoxLayout()
        column.addLayout(row)
        column.addWidget(self.imaging_status)
        return column

    def _show_imaging(self, electrodes) -> None:
        from onset_hfo.case import imaging

        case = self.window_.case
        t1 = imaging.t1_path(case)
        registered = imaging.load_registration(case) is not None
        native = imaging.load_native(case)
        self.native_button.setEnabled(registered)
        self.ct_button.setEnabled(t1 is not None)
        self.on_mri_button.setEnabled(t1 is not None and len(native) > 0)
        self.take_button.setVisible(self._locator is not None)
        if t1 is None:
            status = "No MRI of this patient: contacts are placed on the template."
        elif not registered:
            status = "MRI added, not yet registered to MNI."
        else:
            check = imaging.registration_check_saved(case)
            correlation = check.get("correlation", float("nan"))
            warped = (Path(case.derivatives) / "imaging" / imaging.WARP_FILE).exists()
            how = "affine, then non-linear" if warped else "affine"
            status = (f"MRI registered to MNI ({how}; correlation with the template "
                      f"{correlation:.2f} after the affine). "
                      f"{len(native)} contact(s) in the MRI's space.")
            if correlation < imaging.CHECK_WARN:
                status += (" The correlation is low: the registration may have missed. "
                           "Check the contacts on the MRI before relying on the map.")
        self.imaging_status.setText(status)
        from onset_hfo.case.electrodes import positions_kind, positions_note

        self._positions = positions_kind(electrodes)
        self.note.setText(positions_note(electrodes))

    def add_mri(self, path, wait: bool = False):
        """Copy the MRI into the case and register it to MNI, off the window's thread."""
        from onset_hfo.case import imaging

        imaging.add_t1(self.window_.case, path, by=self.window_.reader())
        return self.register_mri(wait=wait)

    def register_mri(self, wait: bool = False):
        from onset_hfo.case import imaging

        case, reader = self.window_.case, self.window_.reader()
        def work(progress, should_stop):
            if not imaging.template_available():
                imaging.fetch_template()      # the MNI152 head, about 1.8 MB, once
            return imaging.register_t1(case, by=reader)

        self._job = _Job(work, self)
        self._job.finished.connect(self._registered)
        self.mri_button.setEnabled(False)
        self.imaging_status.setText("Registering the MRI to MNI…")
        self._job.start()
        if wait:
            self._job.wait()
            self._registered()
        return self._job

    def _registered(self) -> None:
        job, self._job = self._job, None
        self.mri_button.setEnabled(True)
        if job is None:
            return
        if job.error is not None:
            QMessageBox.warning(self, "Register the MRI", f"Not registered: {job.error}")
        self.refresh()
        self.changed.emit()

    def mri_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "The patient's T1-weighted MRI",
                                              str(Path.home()),
                                              "NIfTI (*.nii *.nii.gz);;All files (*)")
        if not path:
            return None
        try:
            return self.add_mri(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Add the patient's MRI", str(error))
            return None

    def import_native(self, path_or_table, what: str = ""):
        from onset_hfo.case import imaging

        frame = imaging.import_native(self.window_.case, path_or_table, self._load_atlas(),
                                      by=self.window_.reader(), what=what)
        self.refresh()
        self.changed.emit()
        return frame

    def native_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Contact positions in the patient's MRI",
                                              str(Path.home()),
                                              "Coordinates (*.tsv *.csv *.txt);;All files (*)")
        if not path:
            return None
        try:
            return self.import_native(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Import contacts", str(error))
            return None

    def open_locator(self, ct_path, show: bool = True):
        """Align the CT to the MRI and open MNE's locator on it for the case's contacts."""
        from onset_hfo.case import imaging

        aligned = imaging.ct_to_t1(self.window_.case, ct_path, by=self.window_.reader())
        gui, info = imaging.open_locator(aligned, self._names, show=show)
        self._locator = (gui, info, aligned)
        self.take_button.setVisible(True)
        return gui

    def ct_dialog(self):
        from onset_hfo.case import imaging

        if not imaging.locator_available():
            QMessageBox.information(
                self, "Place contacts on a CT",
                "Placing contacts on a CT uses MNE's contact locator, which is not "
                "installed. Install the imaging extra:\n\n"
                "    pip install \"onset-hfo[imaging]\"\n\n"
                "Or import positions your localisation software exported, with "
                "\"Import contacts in the MRI's space…\".")
            return None
        if imaging.load_registration(self.window_.case) is None:
            QMessageBox.information(self, "Place contacts on a CT",
                                    "Add the patient's MRI first: the CT is aligned to it, "
                                    "and it is what carries the contacts to MNI.")
            return None
        if not self._names:
            QMessageBox.information(self, "Place contacts on a CT",
                                    "The case has no intracranial channels to place yet.")
            return None
        path, _ = QFileDialog.getOpenFileName(self, "The patient's CT with the electrodes",
                                              str(Path.home()),
                                              "NIfTI (*.nii *.nii.gz);;All files (*)")
        if not path:
            return None
        try:
            return self.open_locator(path)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Place contacts on a CT", str(error))
            return None

    def take_located(self):
        from onset_hfo.case import imaging

        if self._locator is None:
            return None
        gui, info, aligned = self._locator
        placed = imaging.positions_from_locator(info, aligned)
        if not len(placed):
            QMessageBox.information(self, "Use the placed contacts",
                                    "No contact has been placed in the locator yet.")
            return None
        frame = self.import_native(placed, what=f"{len(placed)} contact(s) placed on the CT "
                                   "with MNE's locator")
        self._locator = None
        try:
            gui.close()
        except RuntimeError:
            pass
        self.take_button.setVisible(False)
        return frame

    def show_on_mri(self, show: bool = True):
        dialog = MriDialog(self.window_.case, self._selected_names(), self)
        if show:
            dialog.show()
        return dialog

    def _selected_names(self) -> list[str]:
        rows = sorted({i.row() for i in self.contacts.selectionModel().selectedRows()}) \
            if self.contacts.selectionModel() else []
        return [self._names[r] for r in rows if r < len(self._names)]

    # -- data ------------------------------------------------------------------------------------
    def _load_atlas(self):
        from onset_hfo.case.atlas import Atlas

        if self.atlas is None and Atlas.available():
            try:
                self.atlas = Atlas.load()
            except (OSError, ValueError):
                self.atlas = None
        self.atlas_button.setVisible(self.atlas is None)
        return self.atlas

    def _contact_names(self, electrodes) -> list[str]:
        names = []
        for rec in self.window_.case.recordings:
            try:
                channels = self.window_.case.channels(rec)
            except (OSError, ValueError):
                continue
            for row in channels.itertuples():
                if str(row.type).upper() in ("SEEG", "ECOG", "EEG", "DBS"):
                    name = str(row.name).strip().upper()
                    if name not in names:
                        names.append(name)
        for name in electrodes["name"].astype(str).str.upper():
            if name not in names:
                names.append(name)
        return names

    def refresh(self) -> None:
        from onset_hfo.case.electrodes import load_electrodes

        case = self.window_.case
        self._load_atlas()
        electrodes = load_electrodes(case)
        self._names = self._contact_names(electrodes)
        self._show_imaging(electrodes)
        placed = {str(r.name).upper(): r for r in electrodes.itertuples()}
        zone = set(case.zone("soz"))
        self.contacts.blockSignals(True)
        self.contacts.setRowCount(len(self._names))
        for r, name in enumerate(self._names):
            row = placed.get(name)
            where = (f"{row.x:.1f}, {row.y:.1f}, {row.z:.1f}" if row is not None else "–")
            source = row.source if row is not None else "not placed"
            probably = ""
            if row is not None and row.label_how:
                probably = (row.label if row.label_how == "in" else
                            f"near {row.label} ({row.label_mm:.0f} mm)"
                            if row.label_how == "near" else row.label_how)
            for c, value in zip((0, 2, 3, 4), (name, where, source, probably), strict=True):
                self.contacts.setItem(r, c, QTableWidgetItem(value))
            tick = QTableWidgetItem("")
            tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            tick.setCheckState(Qt.Checked if name in zone else Qt.Unchecked)
            self.contacts.setItem(r, 1, tick)
        self.contacts.blockSignals(False)
        self._show_combined()

    def _show_combined(self) -> None:
        from onset_review.caseictal import latest_ictal
        from onset_review.casemap import agreement, combined_table, statement

        case = self.window_.case
        table = combined_table(case)
        self._table = table
        ictal = latest_ictal(case)
        found = agreement(table, ictal.consistent() if ictal else [])
        self.statement.setText(statement(table, found) if len(table) else
                               "Run the Interictal or Ictal onset step to map its results.")
        rows = table.to_dict("records")
        self.combined.setRowCount(len(rows))
        for r, row in enumerate(rows):
            rate = "–" if pd.isna(row["rate_per_min"]) else f"{row['rate_per_min']:.2f}"
            ci = "" if pd.isna(row["rate_ci_low"]) else \
                f"{row['rate_ci_low']:.2f}–{row['rate_ci_high']:.2f}"
            ei = "–" if pd.isna(row["median_ei"]) else f"{row['median_ei']:.2f}"
            high = "" if pd.isna(row["seizures"]) else \
                f"{int(row['seizures_high'])} of {int(row['seizures'])}"
            values = (row["channel"], rate, ci, "yes" if row["tied"] else "", ei, high,
                      row["where"] or ("not placed" if not row["placed"] else ""),
                      "yes" if row["soz"] else "")
            for c, value in enumerate(values):
                self.combined.setItem(r, c, QTableWidgetItem(str(value)))
        self._draw()

    def _draw(self) -> None:
        from onset_review import templatebrain
        from onset_review.casemap import draw_combined

        glass = templatebrain.available()
        placed = bool(len(self._table) and self._table["placed"].any())
        self.view.setVisible(not glass)
        self.view3d_button.setVisible(glass)
        self.view3d_button.setEnabled(placed)
        if len(self._table):
            draw_combined(self.figure, self._table, self.atlas,
                          view=self.view.currentData() or "top",
                          positions=getattr(self, "_positions", "template"))
        else:
            self.figure.clear()
        self.canvas.draw_idle()

    # -- actions ---------------------------------------------------------------------------------
    def write_3d(self):
        from onset_review.templatebrain import write_3d_view

        target = Path(self.window_.case.derivatives) / "map" / "contacts-3d.html"
        return write_3d_view(target, self._table, getattr(self, "_positions", "template"))

    def open_3d(self):
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        path = self.write_3d()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        return path

    def _store(self, frame, action: str, detail: str) -> None:
        from onset_hfo.case.electrodes import add_labels, load_electrodes, merge, save_electrodes

        if self._load_atlas() is not None:
            frame = add_labels(frame, self.atlas)
        merged = merge(load_electrodes(self.window_.case), frame)
        save_electrodes(self.window_.case, merged, action, detail, by=self.window_.reader())
        self.refresh()
        self.changed.emit()

    def import_file(self, path, space: str = "MNI152"):
        from onset_hfo.case.electrodes import read_coordinate_file

        frame = read_coordinate_file(path, space)
        self._store(frame, "imported contact positions",
                    f"{len(frame)} contact(s) from {Path(path).name} ({space})")
        return frame

    def import_dialog(self):
        from onset_hfo.case.electrodes import SPACES

        path, _ = QFileDialog.getOpenFileName(self, "Contact positions in a template space",
                                              str(Path.home()),
                                              "Coordinates (*.tsv *.csv *.txt);;All files (*)")
        if not path:
            return None
        space, ok = QInputDialog.getItem(self, "Which space", "The file's coordinates are in:",
                                         list(SPACES), 0, False)
        if not ok:
            return None
        try:
            return self.import_file(path, space)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Import coordinates", str(error))
            return None

    def add_planned(self, frame):
        group = str(frame["group"].iloc[0]) if len(frame) else ""
        self._store(frame, "planned an electrode on the template",
                    f"{group}: {len(frame)} contact(s) from "
                    f"({frame['x'].iloc[0]:.1f}, {frame['y'].iloc[0]:.1f}, "
                    f"{frame['z'].iloc[0]:.1f}) mm")
        return frame

    def plan_dialog(self):
        dialog = PlanDialog(self._load_atlas(), self)
        if dialog.exec() != QDialog.Accepted:
            return None
        try:
            return self.add_planned(dialog.plan())
        except ValueError as error:
            QMessageBox.warning(self, "Plan an electrode", str(error))
            return None

    def remove_selected(self) -> list[str]:
        from onset_hfo.case.electrodes import load_electrodes, save_electrodes

        rows = sorted({i.row() for i in self.contacts.selectionModel().selectedRows()}) \
            if self.contacts.selectionModel() else []
        names = {self._names[r] for r in rows if r < len(self._names)}
        electrodes = load_electrodes(self.window_.case)
        keep = electrodes[~electrodes["name"].str.upper().isin(names)]
        if len(keep) == len(electrodes):
            return []
        save_electrodes(self.window_.case, keep, "removed contact positions",
                        ", ".join(sorted(names)), by=self.window_.reader())
        self.refresh()
        return sorted(names)

    def ticked(self) -> list[str]:
        return [self._names[r] for r in range(self.contacts.rowCount())
                if self.contacts.item(r, 1) is not None
                and self.contacts.item(r, 1).checkState() == Qt.Checked]

    def save_zone(self) -> list[str]:
        contacts = self.ticked()
        self.window_.case.set_zone(contacts, "soz", by=self.window_.reader())
        self._show_combined()
        self.changed.emit()
        return contacts

    def fetch_atlas(self, wait: bool = False):
        from onset_hfo.case.atlas import fetch_atlas

        self._job = _Job(lambda progress, should_stop: fetch_atlas(progress=progress), self)
        self._job.finished.connect(self._fetched)
        self.atlas_button.setEnabled(False)
        self._job.start()
        if wait:
            self._job.wait()
            self._fetched()
        return self._job

    def _fetched(self) -> None:
        from onset_hfo.case.electrodes import add_labels, load_electrodes, save_electrodes

        job, self._job = self._job, None
        self.atlas_button.setEnabled(True)
        if job is None:
            return
        if job.error is not None:
            QMessageBox.warning(self, "Fetch the atlas", f"Not fetched: {job.error}")
            return
        if self._load_atlas() is not None:
            electrodes = load_electrodes(self.window_.case)
            if len(electrodes):
                save_electrodes(self.window_.case, add_labels(electrodes, self.atlas),
                                "labelled contacts with the atlas",
                                f"{len(electrodes)} contact(s)", by=self.window_.reader())
        self.refresh()


class MriDialog(QDialog):
    """The contacts on the patient's own MRI, in three planes through one of them."""

    def __init__(self, case, selected: list[str] | None = None, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from onset_hfo.case import imaging

        self.setWindowTitle("Contacts on the patient's MRI")
        self.resize(980, 460)
        self.case = case
        native = imaging.load_native(case)
        self.centre = QComboBox()
        self.centre.setObjectName("onset_case_mri_centre")
        for name in native["name"].astype(str):
            self.centre.addItem(name)
        if selected:
            index = self.centre.findText(selected[0], Qt.MatchFixedString)
            if index >= 0:
                self.centre.setCurrentIndex(index)
        self.centre.currentIndexChanged.connect(lambda _i: self.draw())
        self.figure = Figure(figsize=(9.6, 3.8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        row = QHBoxLayout()
        row.addWidget(QLabel("Through"))
        row.addWidget(self.centre)
        row.addStretch(1)
        box = QVBoxLayout(self)
        box.addLayout(row)
        box.addWidget(self.canvas, 1)
        box.addWidget(_muted("Where the contacts were localised in this patient's MRI; "
                             "the atlas names on the Map step come from the template."))
        self.draw()

    def draw(self) -> None:
        from onset_hfo.case import imaging

        self.figure.clear()
        imaging.draw_on_mri(self.figure, self.case, centre=self.centre.currentText() or None)
        self.canvas.draw_idle()


class ReportPage(QWidget):
    """Report: what must hold first, the de-identification check, sign-off,
    and the numbered versions (`onset_review.casereport`)."""

    changed = Signal()

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self.findings = []
        self.checks = QTableWidget(0, 3)
        self.checks.setObjectName("onset_case_checks")
        self.checks.setHorizontalHeaderLabels(["Before a report", "Met", "Detail"])
        self.checks.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.checks.verticalHeader().setVisible(False)
        self.checks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.checks.horizontalHeader().setStretchLastSection(True)
        self.checks.setMaximumHeight(190)
        self.names = QLineEdit()
        self.names.setPlaceholderText("the patient's names, comma-separated — looked for, "
                                     "never stored")
        self.check_button = QPushButton("Check de-identification")
        self.check_button.clicked.connect(lambda _=False: self.check())
        self.redact_button = QPushButton("Redact what was found")
        self.redact_button.setEnabled(False)
        self.redact_button.clicked.connect(lambda _=False: self.redact())
        self.found = QTableWidget(0, 3)
        self.found.setObjectName("onset_case_deid")
        self.found.setHorizontalHeaderLabels(["Where", "Looks like", "Text (masked)"])
        self.found.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.found.verticalHeader().setVisible(False)
        self.found.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.found.horizontalHeader().setStretchLastSection(True)
        self.found.setMaximumHeight(150)
        self.role = QLineEdit()
        self.role.setPlaceholderText("your role, e.g. consultant neurophysiologist")
        self.statement = QLineEdit()
        from onset_review.casereport import SIGNOFF_STATEMENT

        self.statement.setText(SIGNOFF_STATEMENT)
        self.sign_button = QPushButton("Sign off")
        self.sign_button.clicked.connect(lambda _=False: self.sign())
        self.signoffs = QTableWidget(0, 4)
        self.signoffs.setObjectName("onset_case_signoffs")
        self.signoffs.setHorizontalHeaderLabels(["By", "Role", "At", "For this content"])
        self.signoffs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.signoffs.verticalHeader().setVisible(False)
        self.signoffs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.signoffs.horizontalHeader().setStretchLastSection(True)
        self.signoffs.setMaximumHeight(130)
        self.produce_button = QPushButton("Produce the report")
        self.produce_button.setObjectName("onset_case_produce")
        self.produce_button.clicked.connect(lambda _=False: self.produce())
        self.versions = QTableWidget(0, 5)
        self.versions.setObjectName("onset_case_versions")
        self.versions.setHorizontalHeaderLabels(["Version", "Made", "By", "Signed off",
                                                 "Content"])
        self.versions.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.versions.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.versions.verticalHeader().setVisible(False)
        self.versions.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.versions.horizontalHeader().setStretchLastSection(True)
        self.versions.itemDoubleClicked.connect(lambda _item: self.open_selected())
        self.status = QLabel("")
        self.status.setObjectName("onset_case_report_status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        deid_row = QHBoxLayout()
        deid_row.addWidget(self.names, 1)
        deid_row.addWidget(self.check_button)
        deid_row.addWidget(self.redact_button)
        sign_row = QHBoxLayout()
        sign_row.addWidget(self.role, 1)
        sign_row.addWidget(self.sign_button)
        produce_row = QHBoxLayout()
        produce_row.addWidget(self.produce_button)
        produce_row.addStretch(1)
        box = QVBoxLayout(self)
        box.addWidget(self.checks)
        box.addWidget(_muted("De-identification: the case's notes, source names, marks, "
                             "channel descriptions, sidecars, contact names and log are "
                             "searched for dates, record numbers, e-mail addresses, phone "
                             "numbers and the names you type."))
        box.addLayout(deid_row)
        box.addWidget(self.found)
        box.addWidget(_muted("A sign-off is for the content as it is now — every result, "
                             "setting, contact position and zone the report draws on. Change "
                             "any of them and it is superseded."))
        box.addWidget(self.statement)
        box.addLayout(sign_row)
        box.addWidget(self.signoffs)
        box.addLayout(produce_row)
        box.addWidget(self.status)
        box.addWidget(self.versions, 1)

    def refresh(self) -> None:
        from onset_review.casereport import readiness, report_versions, signoff_status

        case = self.window_.case
        rows = readiness(case)
        self.checks.setRowCount(len(rows))
        for r, (what, ok, detail) in enumerate(rows):
            for c, value in enumerate((what, "✓" if ok else "✗", detail)):
                self.checks.setItem(r, c, QTableWidgetItem(value))
        ready = all(ok for _w, ok, _d in rows)
        self.produce_button.setEnabled(ready)
        self.sign_button.setText(f"Sign off as {self.window_.reader() or '…'}")
        status = signoff_status(case)
        self.signoffs.setRowCount(len(status))
        for r, s in enumerate(status):
            values = (s["by"], s.get("role", ""), s["at"].replace("T", " "),
                      "yes" if s["current"] else "superseded")
            for c, value in enumerate(values):
                self.signoffs.setItem(r, c, QTableWidgetItem(value))
        versions = report_versions(case)
        self.versions.setRowCount(len(versions))
        for r, v in enumerate(versions):
            values = (f"v{v['version']}", v["made_at"].replace("T", " "), v["made_by"],
                      "yes" if v["signed"] else "no", v["fingerprint"][:12])
            for c, value in enumerate(values):
                self.versions.setItem(r, c, QTableWidgetItem(value))
        current = any(s["current"] for s in status)
        self.status.setText(
            ("Ready. " if ready else "Not ready: " + "; ".join(
                w for w, ok, _d in rows if not ok) + ". ")
            + ("Signed off for this content." if current else
               "Not signed off for this content: a report made now is a draft."))

    def _names(self) -> list[str]:
        return [n.strip() for n in self.names.text().replace(";", ",").split(",") if n.strip()]

    def check(self) -> list:
        from onset_hfo.case.deid import check_case

        case = self.window_.case
        names = self._names()
        self.findings = check_case(case, names)
        self.found.setRowCount(len(self.findings))
        for r, f in enumerate(self.findings):
            for c, value in enumerate((f"{f.file} — {f.where}", f.what, f.excerpt)):
                self.found.setItem(r, c, QTableWidgetItem(value))
        self.redact_button.setEnabled(bool(self.findings))
        case.record("checked de-identification",
                    f"{len(self.findings)} finding(s); {len(names)} name(s) looked for",
                    self.window_.reader())
        self.refresh()
        return self.findings

    def redact(self) -> int:
        from onset_hfo.case.deid import redact

        changed = redact(self.window_.case, self.findings, self._names(),
                         by=self.window_.reader())
        self.check()
        self.changed.emit()
        return changed

    def sign(self) -> dict | None:
        from onset_review.casereport import sign_off

        try:
            entry = sign_off(self.window_.case, self.window_.reader(), self.role.text(),
                             self.statement.text())
        except ValueError as error:
            QMessageBox.warning(self, "Sign off", str(error))
            return None
        self.refresh()
        return entry

    def produce(self):
        from onset_review.casereport import html_to_pdf, produce

        try:
            result = produce(self.window_.case, self.window_.reader(), render_pdf=html_to_pdf)
        except ValueError as error:
            QMessageBox.warning(self, "Produce the report", str(error))
            return None
        self.refresh()
        self.status.setText(self.status.text() + f" Made v{result.version}: "
                            f"{result.pdf.name if result.pdf else result.html.name}.")
        self.window_.refresh()
        self.changed.emit()
        return result

    def open_selected(self):
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        from onset_review.casereport import report_versions

        rows = self.versions.selectionModel().selectedRows() if self.versions.selectionModel() \
            else []
        versions = report_versions(self.window_.case)
        if not rows or rows[0].row() >= len(versions):
            return None
        folder = Path(versions[rows[0].row()]["folder"])
        target = next(iter(sorted(folder.glob("*.pdf"))), folder / "report.html")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        return target


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
        self.map_page = MapPage(self)
        self.report_page = ReportPage(self)
        own = {"import": self.import_page, "channels": self.channels_page,
               "annotate": self.annotate_page, "segments": self.segments_page,
               "preprocess": self.settings_page, "interictal": self.interictal_page,
               "ictal": self.ictal_page, "review": self.review_page,
               "map": self.map_page, "report": self.report_page}
        self.page_for: dict[str, QWidget] = {}
        for key, title, phase in STEPS:
            page = own.get(key) or LaterPage(title, phase)
            self.page_for[key] = page
            self.pages.addWidget(page)
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, key)
            item.setSizeHint(QSize(0, 28))
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
        from onset_review import credit

        self.statusBar().showMessage(credit())
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
