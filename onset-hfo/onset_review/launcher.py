"""The open dialog: choose a patient, a minute, a band, a detector.

What a reviewer picks here is the whole input to the analysis, which is why
`ReviewRequest` is a frozen value object and why this dialog does nothing but
fill one in. A session can then be reproduced from the exported report by
retyping five fields.

Two choices in here are about not wasting a clinician's afternoon:

* **Only cached windows are offered.** The dialog lists what is on disk, read
  from the cache's own metadata. A reviewer on a hospital machine with no route
  to OpenNeuro never discovers mid-click that the thing they chose needs a
  700 MB download, and a cached minute opens in seconds. Fetching is a tick box
  that is off by default and says what it will do.
* **The fast-ripple band is greyed out at 1000 Hz.** A 500 Hz band on a 500 Hz
  Nyquist is not a conservative analysis, it is a meaningless one, and the
  pipeline would refuse it anyway. Refusing it in the dialog, with the reason
  visible, teaches something; refusing it after a two-minute load does not.

Loading runs on a worker thread. Not for elegance: preprocessing and detection
on a 60 s 50-channel slice takes a few seconds, and a frozen window with no
progress bar is indistinguishable from a crashed one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qtpy.QtCore import QEventLoop, QObject, Qt, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QVBoxLayout,
)

from onset_hfo.config import BANDS
from onset_hfo.detectors import HFO_DETECTORS
from onset_review.session import (
    DETECTOR_LABELS,
    ReviewRequest,
    ReviewSession,
    cached_windows,
    load_session,
)

__all__ = ["LauncherDialog", "LoadWorker", "load_with_progress", "choose_request",
           "DEFAULT_DETECTOR"]

#: Selected when the dialog opens. Named rather than "the first one in the
#: registry": the registry is a dict whose order is an implementation detail,
#: and the day someone adds a detector above it the dialog would silently
#: default to something no sweep has been run for.
DEFAULT_DETECTOR = "rms"


class LauncherDialog(QDialog):
    """Pick a window and the analysis to run on it."""

    def __init__(self, cache_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Onset Review — open a recording")
        self.setMinimumWidth(620)
        self._cache_dir = cache_dir
        self._windows = cached_windows(cache_dir)

        self.subject = QComboBox()
        self.window = QComboBox()
        self.band = QComboBox()
        self.detectors = QListWidget()
        self.detectors.setSelectionMode(QListWidget.MultiSelection)
        self.detectors.setMaximumHeight(96)
        for name in HFO_DETECTORS:
            item = QListWidgetItem(DETECTOR_LABELS.get(name, name))
            item.setData(Qt.UserRole, name)
            self.detectors.addItem(item)
            # After `addItem`, not before: an item that is not yet in a list
            # widget has no selection model to record the request, so setting
            # it first is silently a no-op and the dialog opens with nothing
            # chosen.
            item.setSelected(name == DEFAULT_DETECTOR)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1.0, 10.0)
        self.threshold.setSingleStep(0.25)
        self.threshold.setDecimals(2)
        self.threshold.setSpecialValueText("measured default")
        self.threshold.setValue(1.0)
        self.threshold.setSuffix(" robust SD")
        self.threshold.setToolTip(
            "Leave at the minimum to use each detector's own measured "
            "threshold (2.0 SD for the energy detectors, 3.0 for line "
            "length). Raising it detects fewer, more confident events.")

        self.spikes = QCheckBox("Also detect interictal discharges")
        self.spikes.setChecked(True)
        self.spikes.setToolTip(
            "HFOs riding on a discharge are a real and clinically distinct "
            "population, so they are marked rather than filtered out.")
        self.expert = QCheckBox("Overlay the archive's expert markings on open")
        self.expert.setToolTip(
            "Only ds003498 carries them. The best way to get a feel for what "
            "the detector does is to watch it beside the annotators.")

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#555;font-size:11px;")

        form = QFormLayout()
        form.addRow("Patient", self.subject)
        form.addRow("Window", self.window)
        form.addRow("Band", self.band)
        form.addRow("Detector(s)", self.detectors)
        form.addRow("Threshold", self.threshold)

        options = QGroupBox("Options")
        inner = QVBoxLayout(options)
        inner.addWidget(self.spikes)
        inner.addWidget(self.expert)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        banner = QLabel(
            "<b>Research prototype — not a medical device.</b> Public research "
            "recordings only; nothing here is a diagnosis.")
        banner.setWordWrap(True)   # or it is clipped, which is the one label
                                   # in this dialog that must not be

        layout = QVBoxLayout(self)
        layout.addWidget(banner)
        layout.addLayout(form)
        layout.addWidget(options)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.buttons)
        layout.addLayout(row)

        self.subject.currentIndexChanged.connect(self._fill_windows)
        self.window.currentIndexChanged.connect(self._fill_bands)
        self._fill_subjects()

    # -- population ------------------------------------------------------
    def _fill_subjects(self) -> None:
        self.subject.clear()
        if self._windows.empty:
            self.status.setText(
                "No recordings are cached yet. Fetch one first, for example:\n"
                "    python -m onset_hfo.cli fetch --subject sub-01 "
                "--t-start 0 --t-stop 60")
            self.buttons.button(QDialogButtonBox.Open).setEnabled(False)
            return
        for (dataset, subject), rows in self._windows.groupby(["dataset", "subject"]):
            label = f"{subject}  ({dataset}, {len(rows)} window"
            label += "s)" if len(rows) != 1 else ")"
            self.subject.addItem(label, (dataset, subject))
        self._fill_windows()

    def _current_rows(self) -> pd.DataFrame:
        key = self.subject.currentData()
        if key is None or self._windows.empty:
            return self._windows.iloc[0:0]
        dataset, subject = key
        return self._windows[(self._windows["dataset"] == dataset)
                             & (self._windows["subject"] == subject)]

    def _fill_windows(self) -> None:
        self.window.clear()
        for _, row in self._current_rows().iterrows():
            label = (f"{row['t_start']:g}–{row['t_stop']:g} s "
                     f"({row['duration']:g} s, {row['n_channels']} contacts, "
                     f"{row['sfreq']:g} Hz)")
            self.window.addItem(label, row.to_dict())
        self._fill_bands()

    def _fill_bands(self) -> None:
        """Offer only the bands the recording's sampling rate can support."""
        self.band.clear()
        row = self.window.currentData() or {}
        sfreq = float(row.get("sfreq") or 0.0)
        notes = []
        for name in ("ripple", "fast_ripple"):
            low, high = getattr(BANDS, name)
            text = f"{name.replace('_', ' ')} ({low:.0f}–{high:.0f} Hz)"
            if BANDS.usable(sfreq, (low, high)):
                self.band.addItem(text, name)
            else:
                notes.append(
                    f"The {name.replace('_', ' ')} band needs more than "
                    f"{2 * high:.0f} Hz; this recording is {sfreq:.0f} Hz.")
        if row.get("task"):
            notes.append(f"Recorded during {row['task']} activity; the method "
                         f"was developed on interictal sleep.")
        self.status.setText("  ".join(notes))

    # -- result ----------------------------------------------------------
    def request(self) -> ReviewRequest:
        row = self.window.currentData() or {}
        chosen = [item.data(Qt.UserRole) for item in self.detectors.selectedItems()]
        threshold = (None if self.threshold.value() <= self.threshold.minimum()
                     else float(self.threshold.value()))
        return ReviewRequest(
            dataset=str(row.get("dataset", "ds003498")),
            subject=str(row.get("subject", "sub-01")),
            run=str(row.get("run", "01")),
            task=row.get("task"),
            t_start=float(row.get("t_start", 0.0)),
            t_stop=float(row.get("t_stop", 60.0)),
            detectors=tuple(chosen) or ("rms",),
            band=str(self.band.currentData() or "ripple"),
            threshold_sd=threshold,
            with_spikes=self.spikes.isChecked(),
        )

    def overlay_expert(self) -> bool:
        return self.expert.isChecked()


def _warm_imports() -> None:
    """Resolve MNE's lazily-attached submodules before any worker thread runs.

    MNE attaches `io`, `viz` and `filter` through `lazy_loader`, so the first
    attribute access is an import. Letting that first access happen on the load
    thread while the main thread is inside `processEvents` races CPython's
    import machinery against Qt and segfaults -- intermittently, which is the
    worst way for it to fail. Touching them here, on the GUI thread, before the
    worker starts costs about a second once per process and removes the race.
    """
    import mne

    for name in ("io", "viz", "filter", "annotations"):
        getattr(mne, name, None)


class LoadWorker(QThread):
    """Runs `load_session` off the GUI thread and reports where it has got to.

    A `QThread` subclass rather than a worker object moved onto a thread. Both
    are idiomatic; this one is right here because the thread's job is a single
    call that ends, and a subclass's `run` returning *is* the thread finishing.
    The moved-object form keeps an event loop alive until something explicitly
    quits it, which is one more thing to get right for no benefit.

    The progress callback fires on this thread, so it is re-emitted as a signal
    rather than touching a widget -- the one rule that makes a Qt worker safe,
    and the one most easily broken by a convenience call.

    The *result* is a plain attribute rather than a signal. A queued signal is
    delivered by an event loop, and the loop that would deliver it is the one
    `finished` is about to stop, so a result sent that way arrives after the
    caller has given up in roughly one run of four. `wait()` is a memory
    barrier and the thread is done by then, so reading the attribute afterwards
    needs no signal and cannot race.
    """

    progress = Signal(int, str)

    def __init__(self, request: ReviewRequest, cache_dir: Path | None = None,
                 parent=None):
        super().__init__(parent)
        self._request = request
        self._cache_dir = cache_dir
        #: Written on this thread, read by the caller after `wait()`.
        self.session: ReviewSession | None = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            self.session = load_session(
                self._request, cache_dir=self._cache_dir,
                progress=lambda fraction, message: self.progress.emit(
                    int(fraction * 100), message))
        except Exception as error:   # the dialog has to say what went wrong
            self.error = f"{type(error).__name__}: {error}"


class _ProgressRelay(QObject):
    """Receives the worker's progress on the GUI thread and updates the dialog.

    A real `QObject` with real slots, created on the GUI thread, because that is
    what makes the connection below a *queued* one. Connecting the worker's
    signal straight to a lambda that touches the dialog looks equivalent and is
    not: with no receiver object Qt invokes the callable on the emitting thread,
    so the loader ends up mutating widgets while the GUI thread is painting
    them. That corrupts the heap rather than raising, and it surfaces as a
    segfault somewhere else entirely, minutes later, in about one run in three.
    """

    def __init__(self, dialog):
        super().__init__()
        self._dialog = dialog

    @Slot(int, str)
    def update(self, value: int, message: str) -> None:
        self._dialog.setValue(value)
        self._dialog.setLabelText(message)


def load_with_progress(request: ReviewRequest, cache_dir: Path | None = None,
                       parent=None) -> ReviewSession | None:
    """Load a session behind a progress dialog. Returns None if it failed.

    The dialog is deliberately not cancellable. Detection is a few seconds of
    NumPy with no interruption point, so a Cancel button would either lie or
    need the detectors rewritten around a flag; a determinate progress bar that
    names the current stage is the honest version of the same reassurance.
    """
    from qtpy.QtWidgets import QMessageBox

    _warm_imports()
    dialog = QProgressDialog("Loading…", "", 0, 100, parent)
    dialog.setWindowTitle("Onset Review")
    dialog.setCancelButton(None)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    worker = LoadWorker(request, cache_dir)
    relay = _ProgressRelay(dialog)          # lives on this, the GUI, thread
    worker.progress.connect(relay.update, Qt.QueuedConnection)

    # A nested event loop rather than a `processEvents` spin: queued
    # connections are delivered by an event loop, and spinning means the GUI
    # thread is sometimes inside paint code when one arrives.
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    dialog.show()
    worker.start()
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()
    worker.wait()
    dialog.close()

    session, error = worker.session, worker.error
    relay.deleteLater()
    worker.deleteLater()

    if error is not None:
        QMessageBox.critical(parent, "Could not load this window", error)
        return None
    return session


def choose_request(cache_dir: Path | None = None, parent=None):
    """Show the launcher. Returns (request, overlay_expert) or (None, False)."""
    dialog = LauncherDialog(cache_dir, parent)
    if dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec():
        return dialog.request(), dialog.overlay_expert()
    return None, False
