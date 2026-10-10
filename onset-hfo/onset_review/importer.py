"""The import dialog: confirm what the file actually contains, then open it.

There is no conversion step and no proprietary format -- `onset_hfo.io` hands
the file to one of MNE's thirty readers and gets back the same `Recording` the
archive gives. What this dialog adds is the part a parser cannot do.

**Channel type is the whole point of this screen.** The archive declares which
channels are intracranial. A clinical EDF or BrainVision export declares
everything `eeg`, including the one real iEEG file in this repository's own
cache. The pipeline keeps anything typed `eeg`, `ecog` or `seeg` and drops the
rest by name, so an imported scalp recording -- or an iEEG file with its EKG
and DC channels still in -- is analysed anyway and produces rates. Confidently.
The reviewer is therefore shown the file's own answer, told it is usually
wrong, and made to confirm it before a single detector runs.

Everything else here follows from the same principle: say what is not known
rather than assume it. The mains frequency is a choice, not a reading. The
window is short by default because a night of intracranial EEG is tens of
gigabytes and this software analyses a minute. And an imported file brings no
expert markings, no resected zone and no participant record, which the dialog
says before the reviewer finds the panels empty.
"""

from __future__ import annotations

import math
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from onset_hfo.config import BANDS
from onset_hfo.io import (
    channel_overview,
    detect_format,
    file_filter,
    recording_info,
)
from onset_review.session import ReviewRequest
from onset_review.theme import SPACING, card, muted, plain_buttons

__all__ = ["ImportDialog", "choose_file", "TYPE_CHOICES",
           "VISIBLE_ROWS", "VISIBLE_ROWS_MIN"]

#: Contacts the dialog opens showing. A typical SEEG implantation is 50 to 150
#: contacts, so this never shows all of them -- the point is that it reads as a
#: list to be worked down rather than a field to be clicked past.
VISIBLE_ROWS = 14

#: And the fewest it will shrink to when the window is dragged smaller. Below
#: about this the table stops being a list and starts being a peephole.
VISIBLE_ROWS_MIN = 7

#: Types a reviewer can assign. `seeg` and `ecog` are analysed; everything else
#: is dropped in preprocessing, which is the point of offering them.
TYPE_CHOICES = [
    ("Depth electrode (SEEG)", "seeg"),
    ("Subdural grid or strip (ECoG)", "ecog"),
    ("Scalp EEG", "eeg"),
    ("ECG", "ecg"),
    ("EOG", "eog"),
    ("EMG", "emg"),
    ("Trigger / stimulus", "stim"),
    ("Other, not analysed", "misc"),
]


class ImportDialog(QDialog):
    """Confirm the channels, the window and the mains, then build a request."""

    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.format = detect_format(self.path)
        self.setWindowTitle(f"Import {self.path.name}")
        self.setMinimumWidth(760)

        self.error = ""
        try:
            self.overview = channel_overview(self.path)
            self.info = recording_info(self.path)
        except Exception as problem:         # a reader failing is a message,
            self.overview = None             # not a traceback in a clinic
            self.info = {}
            self.error = f"{type(problem).__name__}: {problem}"

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACING)

        banner = QLabel(
            "<b>Research tool — not a medical device.</b> If this file "
            "holds patient data, de-identification, ethics approval and data "
            "governance are yours; nothing here checks them, and nothing is "
            "sent anywhere.")
        banner.setWordWrap(True)
        banner.setStyleSheet(card("warn"))
        layout.addWidget(banner)

        where = muted(
            _short(self.path) + (f"  ·  {self.format.name} via "
                                 f"mne.io.{self.format.reader}" if self.format
                                 else "  ·  unrecognised format"))
        where.setToolTip(str(self.path))   # the full path, for anyone who
        layout.addWidget(where)            # needs to check which file this is

        if self.overview is None:
            failed = QLabel(f"This file could not be read.<br><br>{self.error}")
            failed.setWordWrap(True)
            failed.setStyleSheet(card("bad"))
            layout.addWidget(failed)
            self.table = None
        else:
            # Stretch 1 against the settings group's 0: the channel list is
            # the screen, and a dialog that shows five of fifty contacts
            # invites exactly the unchecked confirmation it exists to prevent.
            layout.addWidget(self._channels_group(), 1)

        layout.addWidget(self._settings_group(), 0)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open
                                        | QDialogButtonBox.Cancel)
        open_button = self.buttons.button(QDialogButtonBox.Open)
        open_button.setProperty("primary", True)
        open_button.setEnabled(self.overview is not None)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        plain_buttons(self.buttons)
        layout.addWidget(self.buttons)
        self._refresh_counts()
        self._size_to_channels()

    def wanted_height(self) -> int:
        """The height that shows `VISIBLE_ROWS` contacts, before any clamp.

        Separate from applying it so the intent can be checked on a machine
        whose display is too small to grant it -- which is every headless one.
        """
        height = self.sizeHint().height()
        if self.table is None:
            return height
        return (height + self._rows_tall(VISIBLE_ROWS)
                - self._rows_tall(VISIBLE_ROWS_MIN))

    def _size_to_channels(self) -> None:
        """Open tall enough to work down the channel list, but not off-screen.

        The dialog's natural height only guarantees the minimum, which is the
        shortest useful table rather than the one worth opening on. So the
        height is asked for explicitly, and then clamped to what the display
        actually has: a dialog whose Open button is below the bottom of a
        laptop screen is worse than a short table.
        """
        from qtpy.QtGui import QGuiApplication

        wanted = self.sizeHint()
        wanted.setHeight(self.wanted_height())
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            wanted.setHeight(min(wanted.height(), int(available.height() * 0.9)))
            wanted.setWidth(min(wanted.width(), int(available.width() * 0.9)))
        self.resize(wanted)

    # -- channels ----------------------------------------------------------
    def _channels_group(self) -> QGroupBox:
        group = QGroupBox("Channels")
        box = QVBoxLayout(group)

        warning = QLabel(
            "<b>Check these before opening.</b> This file declares "
            f"<b>{self._declared_summary()}</b>. Clinical exports usually "
            "declare every channel as scalp EEG whatever they actually hold, "
            "and this software analyses anything marked SEEG or ECoG — so an "
            "unchecked import can produce confident rates for the wrong "
            "channels.")
        warning.setWordWrap(True)
        warning.setStyleSheet(card("warn"))
        box.addWidget(warning)

        tools = QHBoxLayout()
        for label, kind in (("All as SEEG", "seeg"), ("All as ECoG", "ecog"),
                            ("All as scalp EEG", "eeg")):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, k=kind: self.set_all(k))
            tools.addWidget(button)
        tools.addStretch(1)
        self.counts = muted("")
        self.counts.setWordWrap(False)   # "50 of 50 will be analysed" on one
        tools.addWidget(self.counts)     # line, beside the buttons it counts
        box.addLayout(tools)

        self.table = QTableWidget(len(self.overview), 3)
        self.table.setHorizontalHeaderLabels(
            ["Channel", "The file says", "Analyse it as"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        for row, record in enumerate(self.overview.to_dict("records")):
            self.table.setItem(row, 0, QTableWidgetItem(str(record["name"])))
            declared = QTableWidgetItem(str(record["declared"]))
            declared.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 1, declared)
            chooser = QComboBox()
            for label, kind in TYPE_CHOICES:
                chooser.addItem(label, kind)
            _select(chooser, str(record["suggested"]))
            chooser.currentIndexChanged.connect(self._refresh_counts)
            self.table.setCellWidget(row, 2, chooser)
        self.table.resizeColumnsToContents()
        # Tall enough to be a list you work down rather than a formality you
        # scroll past. Stretch alone does not do it: the settings group below
        # has a large size hint, so without a floor of its own the table is
        # squeezed to four rows of fifty. The floor is counted in rows rather
        # than pixels, so it survives a different font or display scaling,
        # and it shrinks for a short montage so an eight-channel strip does
        # not open a dialog two thirds empty.
        self.table.setMinimumHeight(self._rows_tall(VISIBLE_ROWS_MIN))
        box.addWidget(self.table)
        return group

    def _rows_tall(self, rows: int) -> int:
        """The pixel height that shows `rows` of this table, header included."""
        rows = min(int(rows), self.table.rowCount())
        row_height = (self.table.rowHeight(0) if self.table.rowCount()
                      else self.table.verticalHeader().defaultSectionSize())
        return (self.table.horizontalHeader().height()
                + rows * max(row_height, 24)
                + 2 * self.table.frameWidth())

    def _declared_summary(self) -> str:
        counts = self.overview["declared"].value_counts()
        return ", ".join(f"{int(n)} × {kind}" for kind, n in counts.items())

    def set_all(self, kind: str) -> None:
        for row in range(self.table.rowCount()):
            _select(self.table.cellWidget(row, 2), kind)
        self._refresh_counts()

    def set_channel(self, name: str, kind: str) -> bool:
        """Type one channel by name. False if this file has no such channel.

        A saved answer, or a `--channel-type` flag, can outlive the montage it
        was written for; naming a channel that is not here is worth reporting
        and not worth raising over.
        """
        if self.table is None:
            return False
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == name:
                _select(self.table.cellWidget(row, 2), kind)
                self._refresh_counts()
                return True
        return False

    def channel_types(self) -> dict[str, str]:
        if self.table is None:
            return {}
        return {self.table.item(row, 0).text():
                str(self.table.cellWidget(row, 2).currentData())
                for row in range(self.table.rowCount())}

    def _refresh_counts(self) -> None:
        if self.table is None:
            return
        chosen = list(self.channel_types().values())
        analysed = sum(1 for kind in chosen if kind in ("seeg", "ecog"))
        self.counts.setText(f"{analysed} of {len(chosen)} will be analysed")
        button = self.buttons.button(QDialogButtonBox.Open)
        # Opened whatever the sampling rate: a recording too slow for the HFO
        # bands still has its discharges and its signal quality to read, and
        # the band itself is chosen after opening, on the Signal page.
        button.setEnabled(analysed > 0)
        button.setToolTip(
            "" if analysed else
            "Nothing is marked SEEG or ECoG, so there would be nothing to analyse.")

    # -- settings ----------------------------------------------------------
    def _settings_group(self) -> QGroupBox:
        group = QGroupBox("Window and recording")
        form = QFormLayout(group)
        duration = float(self.info.get("duration") or 0.0)
        sfreq = float(self.info.get("sfreq") or 0.0)

        self.subject = QLineEdit(self.path.stem[:40])
        self.subject.setMaximumWidth(320)
        self.subject.setToolTip(
            "A label for the report. Use a study code, not a patient name.")

        # Bounded by the recording itself rather than by a round number: a
        # reviewer asking for 0–60 s of a 12 s file should be corrected by the
        # spin box, not by an exception after the reader has run.
        span = duration or 1e7
        self.t_start = QDoubleSpinBox()
        self.t_start.setRange(0.0, max(0.0, span - 1.0))
        self.t_start.setSuffix(" s")
        self.t_stop = QDoubleSpinBox()
        self.t_stop.setRange(1.0, span)
        self.t_stop.setValue(min(60.0, span))
        self.t_stop.setSuffix(" s")
        self.t_stop.setToolTip(
            "Only this window is read. A night of intracranial EEG is tens of "
            "gigabytes and this software analyses a minute of it.")

        self.line_freq = QComboBox()
        for label, value in (("50 Hz", 50.0), ("60 Hz", 60.0)):
            self.line_freq.addItem(label, value)
        self.line_freq.setMaximumWidth(160)
        self.line_freq.setToolTip(
            "Not read from the file. Notching the wrong one leaves the "
            "interference in place and carves a hole where there was none, "
            "and it never announces itself.")

        for box in (self.t_start, self.t_stop):
            box.setMaximumWidth(160)      # a seconds field as wide as the
                                          # dialog reads as a text entry
        # Keep the two in order as they are typed. An empty window is caught
        # by the reader either way, but being told "0 s is not a window" after
        # a load is a worse way to learn you typed the fields the wrong way
        # round than simply not being able to.
        self.t_start.valueChanged.connect(
            lambda value: self.t_stop.setMinimum(value + 1.0))
        self.t_stop.valueChanged.connect(
            lambda value: self.t_start.setMaximum(max(0.0, value - 1.0)))

        # The band is not asked for here: ripple or fast ripple is a question
        # about the analysis, answered after opening on the Signal page. What
        # the file's rate allows is said, so nobody is surprised later.
        self.band_name = "ripple"
        rate_note = _rate_note(sfreq) if sfreq else ""

        form.addRow("Label", self.subject)
        form.addRow("From", self.t_start)
        form.addRow("To", self.t_stop)
        form.addRow("Mains", self.line_freq)
        if self.info:
            summary = (f"{self.info['n_channels']} channels · "
                       f"{sfreq:g} Hz · {duration:g} s long.  ")
            form.addRow("", muted(summary + rate_note))
        form.addRow("", muted(
            "An imported file brings no expert markings, no resected zone and "
            "no participant record, so the panels that compare against them "
            "will say they are unavailable."))
        return group

    # -- result ------------------------------------------------------------
    def request(self, detectors: tuple[str, ...] = ("rms",)) -> ReviewRequest:
        return ReviewRequest(
            dataset="", subject=self.subject.text().strip() or self.path.stem,
            run="01", task=None,
            t_start=float(self.t_start.value()),
            t_stop=float(self.t_stop.value()),
            detectors=detectors,
            band=self.band_name,
            path=self.path,
            channel_types=tuple(sorted(self.channel_types().items())),
            line_freq=float(self.line_freq.currentData() or 50.0),
        )


def _rate_note(sfreq: float) -> str:
    """What this sampling rate allows, in the words the analysis will use."""
    ripple, fast = (math.ceil(high / 0.45) for _, high in (BANDS.ripple, BANDS.fast_ripple))
    if not BANDS.usable(sfreq, BANDS.ripple):
        return (f"Too slow for HFOs (ripples need at least {ripple} Hz): HFO detection "
                f"will be skipped, and interictal discharges and signal quality "
                f"analysed.")
    if not BANDS.usable(sfreq, BANDS.fast_ripple):
        return (f"Ripples can be analysed; fast ripples need at least {fast} Hz. "
                f"The band is chosen after opening, on the Signal page.")
    return "Ripples or fast ripples: the band is chosen after opening, on the Signal page."


def _short(path: Path) -> str:
    """The last two components, so a deep cache path does not wrap to two lines."""
    return str(path) if len(str(path)) <= 64 else f"…/{path.parent.name}/{path.name}"


def _select(box: QComboBox, value: str) -> None:
    for index in range(box.count()):
        if box.itemData(index) == value:
            box.setCurrentIndex(index)
            return
    box.setCurrentIndex(box.count() - 1)      # "Other, not analysed"


def choose_file(parent=None, start: str | Path | None = None):
    """Pick a recording, confirm it, and return a request. `None` if cancelled."""
    from qtpy.QtWidgets import QFileDialog

    path, _ = QFileDialog.getOpenFileName(
        parent, "Open a recording", str(start or Path.home()), file_filter())
    if not path:
        return None
    dialog = ImportDialog(path, parent)
    accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
    return dialog.request() if accepted else None
