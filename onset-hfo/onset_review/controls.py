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
    QMenu,
    QPushButton,
    QSpinBox,
    QToolButton,
    QWidget,
)

from onset_review import theme

__all__ = ["TraceControls", "AMPLITUDE_STEP", "PAGE_MM", "as_microvolts",
           "paper_speed", "DUPLICATED_TOOLS"]

#: What MNE's own toolbar offers that this row already does, by the text of
#: the browser's actions. Stacked under this row, the browser's bar read as
#: a second, unexplained set of the same controls; so while the trace is on
#: the page that bar is hidden, and only what it alone offers -- annotation
#: mode, the crosshair, the overview bar, MNE's settings and help -- is kept,
#: behind one button at the end of this row. Projectors are left out too:
#: intracranial recordings carry none.
DUPLICATED_TOOLS = frozenset({
    "Show fewer time points", "Show more time points",
    "Show fewer channels", "Show more channels",
    "Reduce amplitude", "Increase amplitude",
    "Show projectors",
})

#: Width of a standard clinical EEG page, in millimetres. Ten seconds at
#: 30 mm/s, which is the pairing every reader has in their hands. It is the
#: constant that lets a window length in seconds be quoted as the paper speed
#: it corresponds to, without this software pretending to know the physical
#: size of anybody's monitor -- which it cannot, since X11 reports a DPI that
#: is wrong as often as it is right.
PAGE_MM = 300.0

#: One click of the gain buttons. MNE's own keyboard binding uses the same
#: ratio, so a click and a keypress move the trace by the same amount.
AMPLITUDE_STEP = 1.1


def as_microvolts(scalebar: str) -> str:
    """MNE's scalebar text in the unit intracranial EEG is read in.

    `0.1 mV` is a correct statement and not one anybody working on a 90 µV
    ripple wants to do arithmetic on. Returns "" for anything it cannot parse
    rather than a guess: a wrong number beside a right one is worse than one
    number.
    """
    import re

    match = re.match(r"\s*([-+]?[\d.]+)\s*([munµ]?)V\s*$", str(scalebar))
    if not match:
        return ""
    try:
        value = float(match.group(1))
    except ValueError:
        return ""
    factor = {"": 1e6, "m": 1e3, "u": 1.0, "µ": 1.0, "n": 1e-3}[match.group(2)]
    microvolts = value * factor
    if not (0 < microvolts < 1e7):
        return ""
    return (f"{microvolts:.0f} µV" if microvolts >= 10
            else f"{microvolts:.1f} µV")


def paper_speed(seconds: float) -> str:
    """The window length as the paper speed a reader would recognise.

    A clinical page is ten seconds across 300 mm, so 300/seconds is the speed
    in mm/s that puts the same amount of signal in front of someone. Quoted as
    an equivalence because that is what it is: nothing here knows how wide the
    monitor is.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    speed = PAGE_MM / seconds
    return f"≈ {speed:.0f} mm/s" if speed >= 1 else f"≈ {speed:.1f} mm/s"


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
            "represents. It is MNE's own scalebar text, converted to "
            "microvolts — the unit intracranial EEG is read in, and the one "
            "nobody judging a 90 µV ripple wants to do arithmetic around. The "
            "number is the same number; only the unit is ours.")
        # Tabular figures: the gain changes by a factor each click and a
        # proportional font makes the readout jitter sideways as it does.
        self.gain.setStyleSheet(
            f"font-family:monospace;color:{theme.current().text};")

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

        # The keys, behind a "?" rather than written across the bar: a
        # sentence there was the widest thing in the row and the least
        # important, and on a 1366-wide laptop it set the floor for the
        # whole window.
        layout.addWidget(theme.help_button(
            "Amplitude: the trace's scale. Window: seconds on screen. Channels: "
            "rows on screen. At: where the window starts. The trace also takes "
            "the usual MNE keys; press ? on it."))

        self.tools = QToolButton()
        self.tools.setObjectName("onset_trace_tools")
        self.tools.setText("Trace tools")
        self.tools.setToolTip("What the browser's own toolbar adds to this row: "
                              "annotation mode, the crosshair, the overview bar, "
                              "MNE's settings and its help")
        self.tools.setAutoRaise(True)
        self.tools.setPopupMode(QToolButton.InstantPopup)
        self.tools.setMenu(self._tools_menu())
        self.tools.setVisible(not self.tools.menu().isEmpty())
        layout.addWidget(self.tools)
        self.show_browser_toolbar(False)

        self.seconds.valueChanged.connect(self._set_seconds)
        self.channels.valueChanged.connect(self._set_channels)
        self.position.valueChanged.connect(self._set_position)
        self.sync()

    # -- reading MNE's state ------------------------------------------------
    @property
    def state(self):
        return getattr(self._figure, "mne", None)

    @property
    def browser_toolbar(self):
        """MNE's own toolbar on the figure, or None off a real browser."""
        return getattr(self.state, "toolbar", None)

    def show_browser_toolbar(self, on: bool) -> None:
        """Hidden while the trace sits under this row, shown when the trace
        is in a window of its own, where this row is not."""
        toolbar = self.browser_toolbar
        if toolbar is not None:
            toolbar.setVisible(bool(on))

    def _tools_menu(self) -> QMenu:
        """The browser's toolbar actions this row does not duplicate, in the
        browser's order, with the overview-bar chooser as a submenu."""
        menu = QMenu(self)
        toolbar = self.browser_toolbar
        if toolbar is None:
            return menu
        overview = getattr(self.state, "overview_menu", None)
        for action in toolbar.actions():
            if action.isSeparator() or action.text() in DUPLICATED_TOOLS:
                continue
            if not action.text():
                # The overview-bar button: a widget on the toolbar, a menu here.
                if overview is not None:
                    overview.setTitle("Overview bar")
                    menu.addMenu(overview)
                continue
            menu.addAction(action)
        return menu

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
            # Converted where it can be, MNE's own text where it cannot: a
            # readout that guessed at a unit it could not parse would be the
            # one thing this label must never be, which is wrong.
            scalebar = self._gain_text()
            self.gain.setText(as_microvolts(scalebar) or scalebar)
            duration = float(getattr(state, "duration", 10.0))
            xmax = float(getattr(state, "xmax", duration))
            self.seconds.setMaximum(max(duration, xmax))
            self.seconds.setValue(duration)
            self.seconds.setToolTip(
                "Seconds of signal on screen — the same amount of signal a "
                f"clinical page holds at {paper_speed(duration).lstrip('≈ ')}, "
                "ten seconds across 300 mm. That is an equivalence, not a "
                "measurement: this software does not know the physical size of "
                "your monitor and does not pretend to.\n\n"
                "Shorter is the only way to see an 80 Hz oscillation as an "
                "oscillation rather than a thicker line.")
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
