"""File → Open from OpenNeuro…: any public recording, by dataset id.

The dialog lists a dataset's continuous recordings (`onset_hfo.openneuro`),
says what the dataset is and under which licence, and downloads the window
asked for. The window then goes through the same confirmation as File → Open
a file, with the dataset's own channel types and mains frequency filled in,
so the reviewer still sees and confirms what will be analysed.

Listing and downloading run off the window's thread: a listing is a few
seconds and a window can be a minute on a slow connection.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from onset_review.theme import SPACING, card, muted, plain_buttons

__all__ = ["OpenNeuroDialog", "choose_openneuro", "import_dialog_for"]

#: The catalogue's columns: dataset id, kind of recording, subjects, name.
FIND_COLUMNS = (("dataset_id", "Dataset"), ("modalities", "Kind"), ("subjects", "Subjects"),
                ("name", "Name"))
COLUMNS = (("subject", "Subject"), ("session", "Session"), ("task", "Task"), ("acq", "Acq"),
           ("run", "Run"), ("modality", "Kind"), ("format", "Format"), ("size_mb", "MB"))


class _Call(QThread):
    """One call off the window's thread; its result or error read after `wait`."""

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.result = None
        self.error = None

    def run(self) -> None:
        try:
            self.result = self.function()
        except Exception as error:      # noqa: BLE001 - shown in the dialog
            self.error = f"{type(error).__name__}: {error}"


class OpenNeuroDialog(QDialog):
    """Pick a dataset, a recording and a window; download it."""

    def __init__(self, parent=None, data_home=None, dataset: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Open from OpenNeuro")
        self.setMinimumWidth(820)
        self.data_home = data_home
        #: The fetched `Bunch`, once a window has been downloaded.
        self.bunch = None
        self.table_rows = []

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING)
        layout.addWidget(muted(
            "Find a dataset by name, or type any OpenNeuro id. Listing and downloading "
            "reach openneuro.org, even when this window was started offline: choosing them "
            "here is the asking. Only the window asked for is downloaded where the format "
            "allows it (BrainVision, EDF, BDF); other formats come whole. Everything is kept "
            "on this machine and opened from there next time."))

        # -- finding a dataset: the bundled catalogue, searched as you type ---------------
        find = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search datasets: epilepsy, seizure, sleep, ds0041…")
        self.search.setAccessibleName("Search the OpenNeuro catalogue")
        self.search.textChanged.connect(self.show_catalogue)
        self.kind = QComboBox()
        for label, value in (("iEEG", "ieeg"), ("EEG", "eeg"), ("EEG or iEEG", "")):
            self.kind.addItem(label, value)
        self.kind.setAccessibleName("Kind of recording")
        self.kind.currentIndexChanged.connect(self.show_catalogue)
        find.addWidget(QLabel("Find"))
        find.addWidget(self.search, 1)
        find.addWidget(self.kind)
        layout.addLayout(find)

        self.found = QTableWidget(0, len(FIND_COLUMNS))
        self.found.setHorizontalHeaderLabels([label for _, label in FIND_COLUMNS])
        self.found.verticalHeader().setVisible(False)
        self.found.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.found.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.found.setSelectionMode(QAbstractItemView.SingleSelection)
        self.found.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.found.horizontalHeader().setStretchLastSection(True)
        self.found.setMinimumHeight(170)
        self.found.setAccessibleName("OpenNeuro datasets")
        self.found.setToolTip("Click a dataset to choose it; double-click to list its "
                              "recordings")
        self.found.itemSelectionChanged.connect(self._dataset_chosen)
        self.found.itemDoubleClicked.connect(lambda _item: self.list_recordings())
        layout.addWidget(self.found, 1)
        self.found_count = muted("")
        layout.addWidget(self.found_count)

        row = QHBoxLayout()
        self.dataset = QLineEdit(dataset)
        self.dataset.setPlaceholderText("ds004100")
        self.dataset.setMaximumWidth(200)
        self.dataset.setAccessibleName("OpenNeuro dataset id")
        self.dataset.returnPressed.connect(self.list_recordings)
        self.list_button = QPushButton("List recordings")
        self.list_button.clicked.connect(self.list_recordings)
        row.addWidget(QLabel("Dataset"))
        row.addWidget(self.dataset)
        row.addWidget(self.list_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.about = QLabel("")
        self.about.setWordWrap(True)
        self.about.setTextFormat(Qt.RichText)
        self.about.setOpenExternalLinks(True)
        layout.addWidget(self.about)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(200)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.setAccessibleName("Recordings in the dataset")
        layout.addWidget(self.table, 1)

        form = QFormLayout()
        self.t_start = QDoubleSpinBox()
        self.t_start.setRange(0.0, 1e7)
        self.t_start.setSuffix(" s")
        self.t_stop = QDoubleSpinBox()
        self.t_stop.setRange(1.0, 1e7)
        self.t_stop.setValue(60.0)
        self.t_stop.setSuffix(" s")
        self.max_mb = QDoubleSpinBox()
        self.max_mb.setRange(1.0, 100_000.0)
        self.max_mb.setValue(500.0)
        self.max_mb.setSuffix(" MB")
        self.max_mb.setToolTip("A download larger than this is refused rather than started")
        for box in (self.t_start, self.t_stop, self.max_mb):
            box.setMaximumWidth(160)
        form.addRow("From", self.t_start)
        form.addRow("To", self.t_stop)
        form.addRow("At most", self.max_mb)
        layout.addLayout(form)

        self.status = muted("")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.fetch_button = self.buttons.addButton("Download and open",
                                                   QDialogButtonBox.AcceptRole)
        self.fetch_button.setProperty("primary", True)
        self.fetch_button.setEnabled(False)
        self.buttons.accepted.connect(self.fetch)
        self.buttons.rejected.connect(self.reject)
        plain_buttons(self.buttons)
        layout.addWidget(self.buttons)
        self.show_catalogue()

    # -- the catalogue ---------------------------------------------------------------------
    def show_catalogue(self, *_args) -> int:
        """Fill the dataset table from the catalogue, filtered by the search
        words and the kind of recording. Returns how many are shown."""
        from onset_hfo import openneuro

        table = openneuro.catalogue(modality=self.kind.currentData() or None,
                                    search=self.search.text(), data_home=self.data_home)
        self.catalogue_rows = table.to_dict("records")
        self.found.setRowCount(len(self.catalogue_rows))
        for r, row in enumerate(self.catalogue_rows):
            for c, (key, _label) in enumerate(FIND_COLUMNS):
                text = str(row.get(key, "")).replace("ieeg", "iEEG").replace("eeg", "EEG") \
                    if key == "modalities" else str(row.get(key, ""))
                item = QTableWidgetItem(text)
                if key == "name":
                    item.setToolTip(f"{text}\nLicence: {row.get('license') or 'not stated'}")
                self.found.setItem(r, c, item)
        total = len(openneuro.catalogue(data_home=self.data_home))
        self.found_count.setText(
            f"{len(self.catalogue_rows)} of {total} OpenNeuro datasets with EEG or iEEG. "
            "Not listed? Type its id below.")
        return len(self.catalogue_rows)

    def _dataset_chosen(self) -> None:
        rows = self.found.selectionModel().selectedRows() if self.found.selectionModel() \
            else []
        if rows:
            self.dataset.setText(self.catalogue_rows[rows[0].row()]["dataset_id"])

    # -- work off the window's thread ----------------------------------------------------
    def _call(self, function, waiting: str):
        from onset_hfo.openneuro import allow_network
        from onset_review.workers import run

        self.status.setText(waiting)
        self.setEnabled(False)
        worker = _Call(function)
        try:
            with allow_network():       # asked for here, so not refused as offline
                run(worker)
        finally:
            self.setEnabled(True)
        worker.deleteLater()
        return worker.result, worker.error

    def list_recordings(self) -> bool:
        from onset_hfo import openneuro

        dataset = self.dataset.text().strip()
        about, error = self._call(
            lambda: openneuro.describe_openneuro(dataset, data_home=self.data_home),
            f"Listing {dataset}…")
        if error:
            self.status.setText("")
            self.about.setText(error)
            self.about.setStyleSheet(card("bad"))
            self.show_rows([])
            return False
        self.about.setStyleSheet(card())
        recordings = about.recordings
        formats = ", ".join(sorted(set(recordings["format"]))) if len(recordings) else "none"
        self.about.setText(
            f"<b>{about.name}</b> · licence {about.license or 'not stated'} · "
            f"{len(about.subjects)} subjects, {len(recordings)} recordings ({formats}) · "
            f"<a href='{about.url}'>openneuro.org</a><br>Cite: {about.citation}")
        self.show_rows(recordings.to_dict("records"))
        self.status.setText("" if len(recordings) else
                            "No continuous recording in a format this software reads.")
        return True

    def show_rows(self, rows) -> None:
        self.table_rows = list(rows)
        self.table.setRowCount(len(self.table_rows))
        for r, row in enumerate(self.table_rows):
            for c, (key, _label) in enumerate(COLUMNS):
                value = row.get(key, "")
                item = QTableWidgetItem(f"{value:g}" if isinstance(value, float) else str(value))
                if key == "format" and not row.get("windowed", True):
                    item.setToolTip("Comes whole: this format cannot be read a window at a "
                                    "time")
                self.table.setItem(r, c, item)
        if self.table_rows:
            self.table.selectRow(0)
        self._selection_changed()

    def selected(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() \
            else []
        return self.table_rows[rows[0].row()] if rows else None

    def _selection_changed(self) -> None:
        row = self.selected()
        self.fetch_button.setEnabled(row is not None)
        if row is not None:
            self.status.setText(f"{row['path']}"
                                + ("" if row["windowed"] else
                                   f" — comes whole, {row['size_mb']:g} MB"))

    def fetch(self) -> bool:
        from onset_hfo import openneuro

        row = self.selected()
        if row is None:
            return False
        dataset = self.dataset.text().strip()
        start, stop = float(self.t_start.value()), float(self.t_stop.value())
        bunch, error = self._call(
            lambda: openneuro.fetch_openneuro(
                dataset, row["subject"], session=row["session"] or None,
                task=row["task"] or None, acq=row["acq"] or None, run=row["run"] or None,
                modality=row["modality"], t_start=start, t_stop=stop,
                data_home=self.data_home, max_mb=float(self.max_mb.value()), verbose=False),
            f"Downloading {start:g}–{stop:g} s of {row['path'].rsplit('/', 1)[-1]}…")
        if error:
            self.status.setText(error)
            return False
        self.bunch = bunch
        self.accept()
        return True


def import_dialog_for(bunch, parent=None):
    """The File → Open a file confirmation for a fetched window, with the
    dataset's channel types (its bad channels not analysed), mains frequency,
    window and label filled in."""
    from onset_hfo import datasets
    from onset_review.importer import ImportDialog, _select

    dialog = ImportDialog(bunch.local_path, parent)
    dialog.setWindowTitle(f"Open {bunch.dataset_id} {bunch.subject}")
    # The window file's times start at 0; the label keeps where it sits in the
    # original recording, so a report can be traced back to the archive.
    dialog.subject.setText(f"{bunch.dataset_id} {bunch.subject}"
                           + (f" {bunch.task}" if bunch.task else "")
                           + f" from {bunch.t_start:g} s")
    channels = bunch.channels
    if channels is not None and {"name", "type"} <= set(channels.columns):
        status = channels["status"] if "status" in channels else [""] * len(channels)
        for name, kind, state in zip(channels["name"], channels["type"], status, strict=False):
            mne_kind = datasets._mne_type(str(kind))
            # A channel the dataset marks bad is not analysed, whatever its type:
            # HUP, for one, types its scalp and EKG leads SEEG and marks them bad.
            if str(state).lower() == "bad" or mne_kind not in ("seeg", "ecog", "eeg"):
                mne_kind = "misc"
            dialog.set_channel(str(name), mne_kind)
    if bunch.line_freq in (50.0, 60.0):
        _select(dialog.line_freq, float(bunch.line_freq))
    start = float(bunch.t_start - bunch.local_offset)
    stop = float(bunch.t_stop - bunch.local_offset)
    dialog.t_stop.setMaximum(max(dialog.t_stop.maximum(), stop))
    dialog.t_stop.setValue(stop)
    dialog.t_start.setValue(start)
    return dialog


def choose_openneuro(parent=None, data_home=None):
    """The whole flow: pick and download, then confirm. A request, or None."""
    picker = OpenNeuroDialog(parent, data_home=data_home)
    accepted = picker.exec_() if hasattr(picker, "exec_") else picker.exec()
    if not accepted or picker.bunch is None:
        return None
    dialog = import_dialog_for(picker.bunch, parent)
    accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
    return dialog.request() if accepted else None
