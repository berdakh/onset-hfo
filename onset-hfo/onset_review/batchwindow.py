"""File → Analyse many recordings: the batch, as a window.

A list of recordings -- cached archive windows, picked from the same list as
Home's, and files of the reader's own -- analysed one after another in the
background with the settings of the window that opened it, into one table
(`onset_review.batchreview`). The rule that types a file's channels is
stated once, above the list, because a batch cannot stop to ask about each
file. Stop ends the batch between recordings; what finished is kept. A row
can be opened in the main window, or added to *Your cohort* with the
resection and outcome the reader enters. *Across the batch* reads the finished
table down its columns (`onset_review.batchresults`): each recording's
busiest channel on one chart, each detector's agreement with the experts
across the recordings that have markings, and the recordings that differ
from the rest.
"""

from __future__ import annotations

import time
from pathlib import Path

from qtpy.QtCore import Qt, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
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
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from onset_review import batchresults, batchreview, theme

__all__ = ["BatchWindow"]

#: The table's columns: (key, heading).
SHOWN = (("recording", "Recording"), ("events", "Accepted events"), ("leader", "Busiest"),
         ("leader_rate_per_min", "Rate /min"), ("leader_ci", "Interval"),
         ("stands_out", "Stands out"), ("n_tied", "Tied"), ("quality_set_aside", "Set aside"),
         ("expert_f1", "F1 vs experts"), ("expert_rho", "ρ vs experts"),
         ("seconds", "Took (s)"), ("error", "Not analysed because"))


class BatchResultsView(QWidget):
    """*Across the batch*: what the table says read down its columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("onset_batch_across")
        self.across: batchresults.Across | None = None
        self._canvases: list = []
        tokens = theme.current()
        body = QWidget()
        self._box = QVBoxLayout(body)
        self._box.setContentsMargins(8, 8, 8, 8)
        self._box.setSpacing(10)
        self.summary = QLabel("Run the batch; what it says across its recordings shows here.")
        self.summary.setObjectName("onset_batch_summary")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._box.addWidget(self.summary)
        self.leaders_title = QLabel("<b>The busiest channel in each recording</b>")
        self.leaders_note = QLabel("A filled dot stands out from the rest; an open one is "
                                   "tied with others and is only the busiest.")
        self.agreement_title = QLabel("<b>Agreement with the experts, by detector</b>")
        self.agreement_note = QLabel("Each dot is one recording with expert markings; the bar "
                                     "is the median.")
        self.outliers_title = QLabel("<b>Recordings that differ from the rest</b>")
        for note in (self.leaders_note, self.agreement_note):
            note.setWordWrap(True)
            note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
        self.leaders_host = QVBoxLayout()
        self.agreement_host = QVBoxLayout()
        self.outliers = QTableWidget(0, 5)
        self.outliers.setObjectName("onset_batch_outliers")
        self.outliers.setHorizontalHeaderLabels(["Recording", "Measure", "Value",
                                                 "Batch median", "Robust z"])
        self.outliers.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.outliers.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.outliers.horizontalHeader().setStretchLastSection(True)
        self.outliers_note = QLabel("")
        self.outliers_note.setWordWrap(True)
        self.outliers_note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
        for widget in (self.leaders_title, self.leaders_note):
            self._box.addWidget(widget)
        self._box.addLayout(self.leaders_host)
        for widget in (self.agreement_title, self.agreement_note):
            self._box.addWidget(widget)
        self._box.addLayout(self.agreement_host)
        for widget in (self.outliers_title, self.outliers_note, self.outliers):
            self._box.addWidget(widget)
        self._box.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._sections(False)

    def _sections(self, shown: bool) -> None:
        for widget in (self.leaders_title, self.leaders_note, self.agreement_title,
                       self.agreement_note, self.outliers_title, self.outliers_note,
                       self.outliers):
            widget.setVisible(shown)

    @staticmethod
    def _clear(host) -> None:
        while host.count():
            item = host.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _canvas(self, host, width: float, height: float, draw):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._clear(host)
        figure = Figure(figsize=(width, height), dpi=100)
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumHeight(int(height * 100))
        draw(figure, figure.add_subplot(111))
        host.addWidget(canvas)
        self._canvases.append(canvas)
        return canvas

    def show_result(self, result) -> batchresults.Across:
        across = batchresults.across(result.rows, getattr(result, "scores", None))
        self.across = across
        self.summary.setText("\n".join(across.sentences))
        self._canvases = []
        have = len(across.leaders) > 0
        self._sections(have)
        if not have:
            self._clear(self.leaders_host)
            self._clear(self.agreement_host)
            return across
        height = 1.1 + 0.32 * len(across.leaders)
        self.leaders_canvas = self._canvas(
            self.leaders_host, 7.5, height,
            lambda figure, axis: batchresults.draw_leaders(figure, axis, across.leaders))
        marked = len(across.scores) > 0
        self.agreement_note.setText(
            "Each dot is one recording with expert markings; the bar is the median."
            if marked else "None of these recordings carries expert markings.")
        if marked:
            self.agreement_canvas = self._canvas(
                self.agreement_host, 6.0, 2.8,
                lambda figure, axis: batchresults.draw_agreement(figure, axis,
                                                                 across.scores))
        else:
            self._clear(self.agreement_host)
        rows = across.outliers.to_dict("records")
        self.outliers.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("recording", "measure", "value", "median", "z")):
                self.outliers.setItem(r, c, QTableWidgetItem(str(row[key])))
        self.outliers.setVisible(bool(rows))
        self.outliers_note.setText(
            f"More than {batchresults.OUTLIER_Z:g} robust z (from the median absolute "
            "deviation) from the batch's median, on events per minute, the share of contacts "
            "set aside, the busiest channel's rate, or the agreement with the experts. "
            + ("None does." if across.can_flag and not rows else
               f"Needs {batchresults.MIN_FOR_OUTLIERS} analysed recordings; this batch has "
               f"{len(across.leaders)}." if not across.can_flag else
               "Different is not wrong: open the recording and look."))
        return across


class _BatchWorker(QThread):
    progressed = Signal(int, int, str)

    def __init__(self, items, template, rule, folder, parent=None):
        super().__init__(parent)
        self._items, self._template, self._rule, self._folder = items, template, rule, folder
        self._stop = False
        self.result = None
        self.error: str | None = None

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.result = batchreview.run_batch(
                self._items, self._template, self._rule, self._folder,
                progress=lambda i, n, label: self.progressed.emit(i, n, label),
                should_stop=lambda: self._stop)
        except Exception as error:      # noqa: BLE001 - reported in the window
            self.error = f"{type(error).__name__}: {error}"


class _CachedPicker(QDialog):
    """Pick cached windows, with a filter by patient."""

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add cached windows")
        self.resize(560, 520)
        self.frame = frame.reset_index(drop=True)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter, e.g. sub-0 or 0–60")
        self.filter.textChanged.connect(self._filter)
        self.list = QListWidget()
        for index, row in self.frame.iterrows():
            item = BatchItemRow.cached_label(row)
            entry = QListWidgetItem(item)
            entry.setFlags(entry.flags() | Qt.ItemIsUserCheckable)
            entry.setCheckState(Qt.Unchecked)
            entry.setData(Qt.UserRole, int(index))
            self.list.addItem(entry)
        all_box = QCheckBox("All shown")
        all_box.toggled.connect(self._all)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(self.filter)
        box.addWidget(self.list, 1)
        box.addWidget(all_box)
        box.addWidget(buttons)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _all(self, on: bool) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def chosen(self) -> list:
        return [batchreview.BatchItem.from_cached(self.frame.iloc[self.list.item(i).data(
                Qt.UserRole)].to_dict()) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]


class BatchItemRow:
    @staticmethod
    def cached_label(row) -> str:
        return (f"{row['dataset']} · {row['subject']} · run-{row.get('run', '01')} · "
                f"{float(row['t_start']):g}–{float(row['t_stop']):g} s")


class BatchWindow(QWidget):
    """The list, the rule for files, Run/Stop, and the table."""

    #: A request to open in the main window (a row's recording).
    openRequested = Signal(object)

    def __init__(self, template, cached=None, paths=None, folder=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Analyse many recordings")
        self.resize(1180, 760)
        self.template = template
        self._cached = cached
        self.base_folder = Path(folder) if folder else None
        self.items: list[batchreview.BatchItem] = []
        self.result: batchreview.BatchResult | None = None
        self._worker = None
        tokens = theme.current()

        # -- what to analyse ----------------------------------------------------
        self.list = QListWidget()
        self.list.setObjectName("onset_batch_items")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        def button(text, tip, slot):
            out = QPushButton(text)
            out.setToolTip(tip)
            out.clicked.connect(lambda _=False: slot())
            return out

        add_cached = button("Add cached windows…", "Archive windows already on this machine",
                            self.add_cached_dialog)
        add_files = button("Add files…", "Recordings of your own, typed by the rule below",
                           self.add_files_dialog)
        remove = button("Remove", "Take the selected recordings off the list",
                        self.remove_selected)
        list_box = QVBoxLayout()
        list_box.addWidget(QLabel("<b>Recordings</b>"))
        list_box.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        for widget in (add_cached, add_files, remove):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        self.reopen_button = QPushButton("Open a batch…")
        self.reopen_button.setObjectName("onset_batch_reopen")
        self.reopen_button.setToolTip("A batch run before: its table, its charts, and its rows "
                                      "to open or add to your cohort")
        self.reopen_menu = QMenu(self.reopen_button)
        self.reopen_menu.aboutToShow.connect(self._fill_reopen_menu)
        self.reopen_button.setMenu(self.reopen_menu)
        buttons.addWidget(self.reopen_button)
        list_box.addLayout(buttons)

        # -- how --------------------------------------------------------------------
        self.settings_line = QLabel(self._describe(template))
        self.settings_line.setWordWrap(True)
        self.all_as = QComboBox()
        self.all_as.addItem("SEEG", "seeg")
        self.all_as.addItem("ECoG", "ecog")
        self.exceptions = QLineEdit(", ".join(p for p, _ in batchreview.DEFAULT_EXCEPTIONS))
        self.exceptions.setToolTip("Channels named like these are not analysed as brain: "
                                   "ECG/EKG as ECG, EMG, EOG, the rest as other")
        self.t_start = QDoubleSpinBox()
        self.t_stop = QDoubleSpinBox()
        for spin in (self.t_start, self.t_stop):
            spin.setRange(0.0, 86400.0)
            spin.setDecimals(0)
            spin.setSuffix(" s")
        self.t_stop.setValue(60.0)
        self.mains = QComboBox()
        self.mains.addItem("50 Hz", 50.0)
        self.mains.addItem("60 Hz", 60.0)
        rule = QFormLayout()
        rule.addRow("Files: every channel", self.all_as)
        rule.addRow("except names like", self.exceptions)
        window = QHBoxLayout()
        window.addWidget(self.t_start)
        window.addWidget(QLabel("to"))
        window.addWidget(self.t_stop)
        window.addStretch(1)
        rule.addRow("window of each file", window)
        rule.addRow("mains", self.mains)
        rule_note = QLabel("A file's own channel types are usually wrong; this rule types every "
                           "file's channels instead of the confirmation each file gets when "
                           "opened alone. Check one file with File → Open a file first.")
        rule_note.setWordWrap(True)
        rule_note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
        how = QVBoxLayout()
        how.addWidget(QLabel("<b>How</b>"))
        how.addWidget(self.settings_line)
        how.addLayout(rule)
        how.addWidget(rule_note)
        how.addStretch(1)
        top = QHBoxLayout()
        top.addLayout(list_box, 3)
        top.addLayout(how, 2)
        top_host = QWidget()
        top_host.setLayout(top)

        # -- running ---------------------------------------------------------------------
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("onset_batch_run")
        self.run_button.clicked.connect(lambda _=False: self.run())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self.stop())
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("Add recordings, then Run.")
        self.status.setObjectName("onset_batch_status")
        self.status.setWordWrap(True)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.stop_button)
        run_row.addWidget(self.progress, 1)

        # -- the table ----------------------------------------------------------------------
        self.table = QTableWidget(0, len(SHOWN))
        self.table.setObjectName("onset_batch_table")
        self.table.setHorizontalHeaderLabels([h for _, h in SHOWN])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemDoubleClicked.connect(lambda _item: self.open_selected())
        self.open_button = button("Open in the window", "Open the selected recording in the "
                                  "main window, analysed the same way", self.open_selected)
        self._open_tip = self.open_button.toolTip()
        self.cohort_button = button("Add to your cohort…", "Measure the selected recording "
                                    "the way the Outcome study measures a patient",
                                    self.add_to_cohort_dialog)
        self.folder_button = button("Open the folder", "summary.csv, summary.md, settings.json "
                                    "and each recording's ranking",
                                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(
                                        str(self.result.folder))) if self.result else None)
        for widget in (self.open_button, self.cohort_button, self.folder_button):
            widget.setEnabled(False)
        result_row = QHBoxLayout()
        for widget in (self.open_button, self.cohort_button, self.folder_button):
            result_row.addWidget(widget)
        result_row.addStretch(1)
        bottom = QWidget()
        bottom_box = QVBoxLayout(bottom)
        bottom_box.setContentsMargins(0, 0, 0, 0)
        bottom_box.addLayout(run_row)
        bottom_box.addWidget(self.status)
        table_page = QWidget()
        table_box = QVBoxLayout(table_page)
        table_box.setContentsMargins(0, 0, 0, 0)
        table_box.addWidget(self.table, 1)
        table_box.addLayout(result_row)
        self.across_view = BatchResultsView()
        self.tabs = QTabWidget()
        self.tabs.setObjectName("onset_batch_tabs")
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(table_page, "Table")
        self.tabs.addTab(self.across_view, "Across the batch")
        self.tabs.setTabToolTip(1, "Each recording's busiest channel on one chart, each "
                                   "detector's agreement with the experts, and the "
                                   "recordings that differ from the rest")
        bottom_box.addWidget(self.tabs, 1)

        split = QSplitter(Qt.Vertical)
        split.addWidget(top_host)
        split.addWidget(bottom)
        split.setSizes([300, 460])
        box = QVBoxLayout(self)
        box.addWidget(split)
        if paths:
            self.add_paths(paths)

    @staticmethod
    def _describe(request, when: str = "") -> str:
        threshold = (f"{request.threshold_sd:g} SD" if request.threshold_sd is not None
                     else "each detector's own threshold")
        lead = (f"Analysed alike on {when}:" if when
                else "Analysed alike, as the window that opened this:")
        return (f"{lead} detectors {', '.join(request.detectors)}; {request.band_label()}; "
                f"{threshold}; preprocessing "
                f"{'as set there' if request.preprocess else 'the defaults'}; "
                f"quality stage {'on' if request.check_quality else 'off'}.")

    # -- a batch run before ----------------------------------------------------------------
    def _fill_reopen_menu(self) -> None:
        self.reopen_menu.clear()
        recent = batchreview.recent_batches()
        for folder, settings in recent:
            count = len(settings.get("recordings") or [])
            action = self.reopen_menu.addAction(
                f"{settings.get('when', '?')} — {count} recording(s) — {folder.name}")
            action.setToolTip(str(folder))
            action.triggered.connect(lambda _=False, f=folder: self.open_batch(f))
        if recent:
            self.reopen_menu.addSeparator()
        choose = self.reopen_menu.addAction("Choose a batch folder…")
        choose.triggered.connect(lambda _=False: self.open_batch_dialog())

    def open_batch_dialog(self):
        start = str(self.base_folder or Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Open a batch", start)
        return self.open_batch(folder) if folder else None

    def open_batch(self, folder):
        """Show a batch written before: its table and its charts, and, when it
        kept how each recording was asked for, rows that open and join the
        cohort as they did when it ran."""
        if self._worker is not None and self._worker.isRunning():
            self.status.setText("Wait for this batch to finish, or stop it, first.")
            return None
        folder = Path(folder)
        if not (folder / "summary.csv").exists() or not (folder / "settings.json").exists():
            self.status.setText(f"{folder} is not a batch folder: it has no summary.csv and "
                                "settings.json.")
            return None
        result = batchreview.load_batch(folder)
        again = batchreview.reopenable(result)
        self.list.clear()
        self.items = []
        if again is not None:
            items, template, rule = again
            self.template = template
            self._set_rule(rule)
            self.add_items(items)
        else:
            for label in result.settings.get("recordings") or []:
                self.list.addItem(QListWidgetItem(str(label)))
        when = str(result.settings.get("when", ""))
        if again is not None:
            self.settings_line.setText(self._describe(self.template, when))
        self.show_result(result)
        if again is None:
            self.open_button.setEnabled(False)
            self.open_button.setToolTip("This batch was written before batches kept how each "
                                        "recording was asked for; run it again to open its rows")
        batchreview.remember_batch(folder)
        self.status.setText(f"The batch of {when}, from {folder}. "
                            + self.status.text().split("; written to")[0] + ".")
        return result

    def _set_rule(self, rule: batchreview.FileRule) -> None:
        self.all_as.setCurrentIndex(max(0, self.all_as.findData(rule.all_as)))
        self.exceptions.setText(", ".join(p for p, _ in rule.exceptions))
        self.t_start.setValue(rule.t_start)
        self.t_stop.setValue(rule.t_stop)
        self.mains.setCurrentIndex(max(0, self.mains.findData(rule.line_freq)))

    # -- the list ------------------------------------------------------------------------
    def add_items(self, items) -> int:
        known = {item.label() for item in self.items}
        added = 0
        for item in items:
            if item.label() in known:
                continue
            self.items.append(item)
            entry = QListWidgetItem(("📄 " if item.is_file else "") + item.label())
            entry.setToolTip(str(item.path) if item.is_file else item.label())
            self.list.addItem(entry)
            known.add(item.label())
            added += 1
        self.status.setText(f"{len(self.items)} recording(s) to analyse.")
        return added

    def add_paths(self, paths) -> int:
        return self.add_items(batchreview.BatchItem(path=Path(p)) for p in paths)

    def add_cached_dialog(self) -> int:
        frame = self._cached() if callable(self._cached) else self._cached
        if frame is None or not len(frame):
            self.status.setText("No windows are cached on this machine.")
            return 0
        picker = _CachedPicker(frame, self)
        if picker.exec() != QDialog.Accepted:
            return 0
        return self.add_items(picker.chosen())

    def add_files_dialog(self) -> int:
        from onset_hfo.io import supported_suffixes

        patterns = " ".join(f"*{s}" for s in supported_suffixes())
        chosen, _ = QFileDialog.getOpenFileNames(self, "Add recordings", str(Path.home()),
                                                 f"Recordings ({patterns});;All files (*)")
        return self.add_paths(chosen) if chosen else 0

    def remove_selected(self) -> None:
        rows = sorted({self.list.row(i) for i in self.list.selectedItems()}, reverse=True)
        for row in rows:
            self.list.takeItem(row)
            del self.items[row]
        self.status.setText(f"{len(self.items)} recording(s) to analyse.")

    def rule(self) -> batchreview.FileRule:
        names = [n.strip() for n in self.exceptions.text().split(",") if n.strip()]
        defaults = dict(batchreview.DEFAULT_EXCEPTIONS)
        exceptions = tuple((n, defaults.get(n, "misc")) for n in names)
        return batchreview.FileRule(all_as=self.all_as.currentData(), exceptions=exceptions,
                                    t_start=float(self.t_start.value()),
                                    t_stop=float(self.t_stop.value()),
                                    line_freq=float(self.mains.currentData()))

    # -- running ---------------------------------------------------------------------------
    def folder(self) -> Path:
        stamp = time.strftime("onset-batch-%Y%m%d-%H%M%S")
        return (self.base_folder / stamp) if self.base_folder else Path.home() / stamp

    def run(self, wait: bool = False) -> None:
        if not self.items or (self._worker is not None and self._worker.isRunning()):
            return
        from onset_review.workers import track

        self._worker = _BatchWorker(list(self.items), self.template, self.rule(), self.folder(),
                                    self)
        self._worker.progressed.connect(self._progressed)
        self._worker.finished.connect(self._finished)
        self.progress.setRange(0, len(self.items))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        track(self._worker).start()
        if wait:
            self._worker.wait()
            self._finished()

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.status.setText("Stopping after the recording being analysed…")

    def _progressed(self, index: int, total: int, label: str) -> None:
        self.progress.setValue(index)
        self.status.setText(f"Analysing {label} ({index + 1} of {total})…")

    def _finished(self) -> None:
        worker = self._worker
        if worker is None or worker.isRunning():
            return
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setVisible(False)
        if worker.error:
            self.status.setText(f"The batch failed: {worker.error}")
            return
        if worker.result is None:
            return
        self.result = worker.result
        self.show_result(self.result)
        batchreview.remember_batch(self.result.folder)

    def show_result(self, result: batchreview.BatchResult) -> None:
        self.result = result
        rows = result.rows.to_dict("records")
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (key, _heading) in enumerate(SHOWN):
                value = row.get(key, "")
                if isinstance(value, float):
                    text = "" if value != value else (f"{value:.2f}" if key.startswith("expert")
                                                      else f"{value:g}")
                elif isinstance(value, bool):
                    text = "yes" if value else "no"
                else:
                    text = str(value)
                cell = QTableWidgetItem(text)
                if key == "error" and text:
                    cell.setForeground(Qt.red)
                self.table.setItem(r, c, cell)
        done = sum(1 for row in rows if not row.get("error"))
        self.status.setText(f"{done} of {len(rows)} analysed; written to {result.folder}.")
        try:
            self.across_view.show_result(result)
        except Exception as error:      # noqa: BLE001 - the table stands without it
            self.across_view.summary.setText(f"Could not read the batch across: {error}")
        for widget in (self.open_button, self.cohort_button, self.folder_button):
            widget.setEnabled(True)
        self.open_button.setToolTip(self._open_tip)

    # -- a row ------------------------------------------------------------------------------
    def _selected(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows:
            return rows[0].row()
        return 0 if self.table.rowCount() == 1 else None

    def open_selected(self):
        index = self._selected()
        if index is None or self.result is None or index >= len(self.items):
            return None
        request = batchreview.request_for(self.items[index], self.template, self.rule())
        self.openRequested.emit(request)
        return request

    def add_to_cohort_dialog(self):
        index = self._selected()
        if index is None or self.result is None:
            return None
        row = self.result.rows.iloc[index].to_dict()
        if row.get("error"):
            self.status.setText("That recording was not analysed, so it cannot be measured.")
            return None
        from onset_review.studypages import CohortDialog

        channels = [c for c in str(row.get("channel_names", "")).split("|") if c]
        dialog = CohortDialog.for_channels(str(row.get("subject", "")), channels, self)
        if dialog.exec() != QDialog.Accepted:
            return None
        return self.add_to_cohort(index, *dialog.values())

    def add_to_cohort(self, index: int, outcome: str, resected, label: str = ""):
        from onset_review import yourstudy

        row = self.result.rows.iloc[index].to_dict()
        try:
            entry = yourstudy.add_batch_row(row, batchreview.findings_for(self.result, row),
                                            self.result.settings, outcome, resected, label)
        except ValueError as error:
            self.status.setText(str(error))
            return None
        self.status.setText(f"{entry['label']} is in your cohort: {entry['n_candidates']} tied "
                            f"channel(s), {entry['candidates_resected']:.0%} inside the "
                            "resection you entered. The Outcome page shows it.")
        return entry

    def closeEvent(self, event) -> None:      # noqa: N802
        self.stop()
        super().closeEvent(event)
