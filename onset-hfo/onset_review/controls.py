"""Visible controls for the trace: amplitude, time window, channels, scrolling.

`mne-qt-browser` already does all of this from the keyboard, and does it well.
The problem is that nothing on screen says so. A neurophysiologist opening an
unfamiliar program reaches for a gain control with the mouse, does not find
one, and concludes the trace cannot be rescaled -- which is how a reviewer ends
up judging 80 Hz oscillations at whatever gain the program happened to open at.

So this is a thin bar over MNE's own methods. Every control calls the same
`scale_all` / `change_duration` / `change_nchan` / `hscroll` / `vscroll` the
keys call, which matters for two reasons: the two input routes cannot drift
apart, and the readouts can be taken from MNE's own scalebar rather than
recomputed here. A gain readout that disagrees with the scalebar drawn two
centimetres away is worse than no readout.

Nothing in this file decides anything. It is the one module in the package
that is pure interface.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

__all__ = ["TraceControls", "AMPLITUDE_STEP"]

#: One click of the gain buttons. MNE's own keyboard binding uses the same
#: ratio, so a click and a keypress move the trace by the same amount.
AMPLITUDE_STEP = 1.1


def _button(text: str, tooltip: str, slot) -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tooltip)
    button.setAutoRepeat(True)          # holding it keeps scrolling
    button.setAutoRepeatDelay(400)
    button.setAutoRepeatInterval(90)
    button.clicked.connect(slot)
    return button


class TraceControls(QWidget):
    """The bar: gain, seconds on screen, channels on screen, and both scrolls.

    Holds a reference to MNE's figure rather than a copy of its state, and
    re-reads that state after every action. The browser can also be driven from
    the keyboard, the scrollbars and the settings dialog, so any value cached
    here would be wrong as soon as a reviewer touched one of those.
    """

    #: Emitted after anything changes the view, so a caller can resync.
    viewChanged = Signal()

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self._figure = figure
        self._updating = False

        self.gain = QLabel("—")
        self.gain.setMinimumWidth(78)
        self.gain.setAlignment(Qt.AlignCenter)
        self.gain.setToolTip(
            "Height of the scale bar on the trace: the amplitude one division "
            "represents. Taken from MNE's own scalebar, so it cannot disagree "
            "with it.")
        self.gain.setStyleSheet("font-family:monospace;")

        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.2, 600.0)
        self.seconds.setSingleStep(1.0)
        self.seconds.setDecimals(1)
        self.seconds.setSuffix(" s")
        self.seconds.setToolTip(
            "Seconds of signal on screen. Shorter is the only way to see an "
            "80 Hz oscillation as an oscillation rather than a thicker line.")

        self.channels = QSpinBox()
        self.channels.setRange(1, 512)
        self.channels.setSuffix(" ch")
        self.channels.setToolTip("Channels on screen.")

        self.position = QDoubleSpinBox()
        self.position.setRange(0.0, 1e6)
        self.position.setDecimals(1)
        self.position.setSingleStep(1.0)
        self.position.setSuffix(" s")
        self.position.setToolTip(
            "Where the left edge of the screen sits in this window.")

        home = QPushButton("⌂")
        home.setToolTip("Back to the start of the window")
        home.setMaximumWidth(30)
        home.clicked.connect(self.go_home)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(3)

        layout.addWidget(QLabel("Amplitude"))
        layout.addWidget(_button("−", "Smaller traces (key: -)",
                                 lambda: self._scale(1 / AMPLITUDE_STEP)))
        layout.addWidget(self.gain)
        layout.addWidget(_button("+", "Taller traces (key: +)",
                                 lambda: self._scale(AMPLITUDE_STEP)))

        layout.addSpacing(10)
        layout.addWidget(QLabel("Window"))
        layout.addWidget(self.seconds)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Channels"))
        layout.addWidget(self.channels)
        layout.addWidget(_button("▲", "Scroll up through the channels (key: ↑)",
                                 lambda: self._vscroll(-1)))
        layout.addWidget(_button("▼", "Scroll down through the channels (key: ↓)",
                                 lambda: self._vscroll(1)))

        layout.addSpacing(10)
        layout.addWidget(QLabel("At"))
        layout.addWidget(_button("◀◀", "Back one screen (key: Home)",
                                 lambda: self._hscroll("-full")))
        layout.addWidget(_button("◀", "Back (key: ←)",
                                 lambda: self._hscroll("left")))
        layout.addWidget(self.position)
        layout.addWidget(_button("▶", "Forward (key: →)",
                                 lambda: self._hscroll("right")))
        layout.addWidget(_button("▶▶", "Forward one screen (key: End)",
                                 lambda: self._hscroll("+full")))
        layout.addWidget(home)
        layout.addStretch(1)

        # The hint is the widest thing in the bar and the least important, so
        # it is told not to contribute to the minimum width: on a 1366-wide
        # laptop it would otherwise set the floor for the whole window and push
        # the trace off the screen. It shortens instead of clipping the row.
        hint = QLabel("The trace also takes the usual MNE keys; press ? on it.")
        hint.setStyleSheet("color:#666;font-size:11px;")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        hint.setMinimumWidth(0)
        layout.addWidget(hint)

        self.seconds.valueChanged.connect(self._set_seconds)
        self.channels.valueChanged.connect(self._set_channels)
        self.position.valueChanged.connect(self._set_position)
        self.sync()

    # -- reading MNE's state ------------------------------------------------
    @property
    def state(self):
        return getattr(self._figure, "mne", None)

    def sync(self) -> None:
        """Pull every readout from the browser. Safe to call at any time.

        Guarded against re-entry because setting a spin box emits its own
        signal, and an unguarded sync would call back into the browser with
        the value it just reported.
        """
        state = self.state
        if state is None or self._updating:
            return
        self._updating = True
        try:
            self.gain.setText(self._gain_text())
            duration = float(getattr(state, "duration", 10.0))
            xmax = float(getattr(state, "xmax", duration))
            self.seconds.setMaximum(max(duration, xmax))
            self.seconds.setValue(duration)
            # `len`, not `or`: the browser's `ch_names` is a NumPy array and
            # `array or default` asks for its truth value, which raises.
            names = getattr(state, "ch_names", None)
            total = len(names) if names is not None else 1
            self.channels.setMaximum(max(1, total))
            self.channels.setValue(int(getattr(state, "n_channels", 1)))
            self.position.setMaximum(max(0.0, xmax - duration))
            self.position.setValue(float(getattr(state, "t_start", 0.0)))
        finally:
            self._updating = False
        self.viewChanged.emit()

    def _gain_text(self) -> str:
        """MNE's own scalebar text, which is the number drawn on the trace."""
        try:
            texts = self._figure._get_scale_bar_texts()
        except Exception:
            return "—"
        return next((t for t in texts if t), "—")

    # -- driving it ---------------------------------------------------------
    def _call(self, name: str, **kwargs) -> None:
        """Call one of the browser's own view methods, then resync.

        Guarded: a control that cannot do its job must leave the window
        standing. The reviewer still has the keyboard and the scrollbars.
        """
        method = getattr(self._figure, name, None)
        if method is None:
            return
        try:
            method(**kwargs)
        except Exception:
            return
        self.sync()

    def _scale(self, step: float) -> None:
        self._call("scale_all", step=step)

    def _hscroll(self, step) -> None:
        self._call("hscroll", step=step)

    def _vscroll(self, step: int) -> None:
        self._call("vscroll", step=step)

    def go_home(self) -> None:
        """Back to the start of the window, via the position box's own path."""
        self.position.setValue(0.0)

    def _set_seconds(self, value: float) -> None:
        """Change the window length by stepping MNE rather than setting it.

        Going through `change_duration` keeps the scrollbar, the overview bar
        and the decimation in agreement -- all of which read the duration back
        out of the browser after it changes.

        Its `step` is a **fraction of the current duration**, not a number of
        seconds: internally it computes `duration * step`. Passing the
        difference in seconds instead means asking for 12 s while showing 10
        lands on 30, which is what this did until a test caught it.
        """
        state = self.state
        if state is None or self._updating:
            return
        current = float(getattr(state, "duration", 0.0))
        if current <= 0:
            return
        step = (float(value) - current) / current
        if abs(step) > 1e-9:
            self._call("change_duration", step=step)

    def _set_channels(self, value: int) -> None:
        state = self.state
        if state is None or self._updating:
            return
        step = int(value) - int(getattr(state, "n_channels", value))
        if step:
            self._call("change_nchan", step=step)

    def _set_position(self, value: float) -> None:
        state = self.state
        if state is None or self._updating:
            return
        duration = float(getattr(state, "duration", 10.0))
        try:
            self._figure.mne.plt.setXRange(float(value), float(value) + duration,
                                           padding=0.0)
        except Exception:
            return
        self.sync()
