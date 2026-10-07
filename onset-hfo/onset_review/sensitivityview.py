"""The Threshold panel: the leading channels' rates as the threshold rises.

Drawing only; :mod:`onset_review.sensitivity` re-runs the detector. One line
per leading channel, rate against threshold, the window's own threshold
marked, and under it the sentence that says whether the leader survived.
The re-run takes a few seconds, so it waits for the button and runs on a
worker; the panel says what it is doing while it does.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from onset_review import sensitivity, theme
from onset_review.theme import card

__all__ = ["SensitivityPanel"]


class _Worker(QThread):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self.result = None
        self.error = ""

    def run(self) -> None:
        try:
            self.result = sensitivity.compute(self._session)
        except Exception as error:      # noqa: BLE001 - reported, not raised
            self.error = f"{type(error).__name__}: {error}"


class SensitivityPanel(QWidget):
    """Does the ranking survive a stricter threshold? Press to find out."""

    computed = Signal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._worker = None
        self.result = getattr(session, "sensitivity", None)

        self.run_button = QPushButton("Re-test at stricter thresholds")
        self.run_button.setObjectName("onset_sensitivity_run")
        self.run_button.setToolTip(
            "Re-runs the ranking detector at 1.25×, 1.5× and 2× the threshold this "
            "window used, on the same prepared signal: a few seconds. Nothing on the "
            "other pages changes.")
        self.run_button.clicked.connect(lambda _=False: self.run())
        self.status = theme.muted("")
        self.status.setObjectName("onset_sensitivity_status")
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(self.run_button)
        bar.addWidget(self.status, 1)

        tokens = theme.current()
        self.figure = Figure(figsize=(4.4, 2.4), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(240)
        self.canvas.setMinimumHeight(150)
        self.axes = self.figure.add_axes((0.16, 0.2, 0.80, 0.76))

        self.headline = QLabel("")
        self.headline.setObjectName("onset_sensitivity_headline")
        self.headline.setWordWrap(True)
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.headline.setStyleSheet(f"font-size:9pt;padding:1px 4px;color:{tokens.text};")
        self.headline.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.caption = QLabel(
            "The ranking detector re-run at multiples of the threshold this window "
            "used, on the same prepared signal, for the window's leading channels. A "
            "leader that still leads at twice the threshold is robust to the choice; "
            "one that drops out is a threshold choice, not a channel property. Rates "
            "here are over the whole window, so the first point can differ slightly "
            "from the ranking's per-channel clean time.")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.caption.setStyleSheet(card("info"))

        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addLayout(bar)
        box.addWidget(self.canvas, 1)
        box.addWidget(self.headline)
        box.addWidget(self.caption)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))

        if self.result is not None:
            self._show(self.result)
        else:
            self._idle()

    # -- running ------------------------------------------------------------
    def run(self, background: bool = True) -> None:
        """Re-run the detector; on a worker unless told otherwise (tests)."""
        if self._worker is not None and self._worker.isRunning():
            return
        self.run_button.setEnabled(False)
        self.status.setText("Re-running the detector at four thresholds…")
        self._worker = _Worker(self._session, self)
        self._worker.finished.connect(self._done)
        if background:
            self._worker.start()
        else:
            self._worker.run()
            self._done()

    def _done(self) -> None:
        worker = self._worker
        self.run_button.setEnabled(True)
        if worker is None:
            return
        if worker.error or worker.result is None:
            self.status.setText(f"The re-test failed: {worker.error or 'no result'}")
            return
        self.result = worker.result
        # Kept on the session so the assistant can quote it without re-running.
        try:
            self._session.sensitivity = self.result
        except Exception:       # noqa: BLE001 - a frozen session just goes without
            pass
        self._show(self.result)
        self.computed.emit(self.result)

    def _idle(self) -> None:
        self.axes.clear()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.setVisible(False)
        self.headline.setText("")
        self.status.setText("Not run yet: a few seconds, on request.")

    # -- drawing --------------------------------------------------------------
    def _show(self, sens) -> None:
        tokens = theme.current()
        self.axes.clear()
        self.axes.set_facecolor(tokens.surface)
        if not sens.available:
            self.canvas.setVisible(False)
            self.status.setText(sensitivity.describe(sens) or "Nothing to show.")
            self.headline.setText("")
            return
        self.status.setText(f"Re-run in {sens.runtime_s:g} s.")
        palette = [tokens.accent, tokens.warn, tokens.good, tokens.bad, tokens.text]
        for index, channel in enumerate(sens.channels):
            if channel not in sens.rates.index:
                continue
            ys = [float(sens.rates.loc[channel, t]) for t in sens.thresholds]
            self.axes.plot(sens.thresholds, ys, marker="o", markersize=3.5, lw=1.4,
                           color=palette[index % len(palette)], label=channel,
                           alpha=1.0 if index == 0 else 0.85)
        self.axes.axvline(sens.base_threshold, color=tokens.text_muted, lw=0.6,
                          ls=(0, (2, 4)), alpha=0.7)
        self.axes.set_xlabel("threshold (robust SD)", fontsize=8, color=tokens.text_muted)
        self.axes.set_ylabel("events / min", fontsize=8, color=tokens.text_muted)
        self.axes.set_xticks(sens.thresholds)
        self.axes.set_xticklabels([f"{t:g}" for t in sens.thresholds])
        self.axes.tick_params(labelsize=7, colors=tokens.text_muted)
        for side in self.axes.spines.values():
            side.set_color(tokens.separator)
        for spine in ("top", "right"):
            self.axes.spines[spine].set_visible(False)
        self.axes.set_ylim(bottom=0)
        legend = self.axes.legend(fontsize=7, frameon=False, loc="upper right")
        for text in legend.get_texts():
            text.set_color(tokens.text)
        self.canvas.setVisible(True)
        self.canvas.draw_idle()
        self.headline.setText(sensitivity.describe(sens))

    def closeEvent(self, event):      # noqa: N802  (Qt's spelling)
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(30000)
        super().closeEvent(event)
