"""The Console: Python in this window's memory, as Spyder's IPython console.

For an analysis this software does not do. The console runs in the same
process as the window, so the names in it are the window's own objects --
`raw` is the signal on screen, `findings` the ranking, `session` all of it
-- rather than copies loaded from disk. Whatever is made there shows up in
the Workspace under *Console*; any figure a command draws opens in a window
of its own; and the working directory is the Files pane's current folder,
changed by either.

Two honest costs. A long computation runs in the window's own thread; the
window is let through to answer ten times a second while it runs, so it
repaints and its Stop button (or Esc) interrupts the code as Ctrl+C does in
Spyder -- but one long call inside a library (a large FFT, a file read)
holds it until that call returns. And the console can change the objects the
panels and the report are built from. So every command run in a window is
kept with that window's session and printed, in order, in the exported
report: a number in the report can always be traced to whether a console
touched it.

The console is IPython through `qtconsole` when that is installed (it comes
with the `review` extra), with tab completion, magics and rich tracebacks;
otherwise a plain Python console that says what to install.
"""

from __future__ import annotations

import _thread
import code
import contextlib
import io
import os
import signal
import sys
import threading
import time
import types
import warnings
from pathlib import Path

from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
from qtpy.QtGui import QFont, QFontDatabase, QKeySequence, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["ConsolePanel", "namespace", "user_variables", "connect_panes", "BANNER",
           "RICH_PACKAGES"]

#: What the full console needs beyond the `review` extra's Qt.
RICH_PACKAGES = ("qtconsole", "ipykernel")

#: Names IPython itself puts in the namespace, never the user's.
_IPYTHON_NAMES = {"In", "Out", "get_ipython", "exit", "quit"}


def _session_names(session) -> dict:
    """Every name the Workspace lists, bound to the same object: one
    workspace, seen from the list and from the console."""
    from onset_review.variables import session_names

    if session is None:
        return {"session": None}
    return {"session": session, **session_names(session)}


def namespace(session) -> dict:
    """The names a console starts with: the session's objects and the
    libraries an analysis reaches for. Qt-free."""
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd

    import onset_hfo

    names = {"np": np, "pd": pd, "mne": mne, "plt": plt, "onset_hfo": onset_hfo,
             "Path": Path}
    names.update(_session_names(session))
    return names


def user_variables(ns: dict, injected: dict, hidden=()) -> dict:
    """The names in `ns` the user made: not private, not IPython's (`hidden`
    is what the shell itself added), not a module, and not one the console
    put there -- unless it was reassigned."""
    out = {}
    hidden = set(hidden) | _IPYTHON_NAMES
    for name, value in ns.items():
        if name.startswith("_") or name in hidden:
            continue
        # Modules, functions and classes are left out, as Spyder's Variable
        # Explorer leaves them out by default: `from scipy.signal import
        # welch` makes a name, not a variable.
        if isinstance(value, (types.ModuleType, types.FunctionType,
                              types.BuiltinFunctionType, type)):
            continue
        if name in injected and injected[name] is value:
            continue
        out[name] = value
    return out


def _banner(session) -> str:
    where = session.request.label() if session is not None else "no recording open"
    names = ("every name in the Workspace — raw, signal, times, channels, sfreq, events, "
             "findings, quality, segments, request, read — and session"
             if session is not None else "session (None until a recording is open)")
    return (f"Python {sys.version.split()[0]} in this window's memory — {where}.\n"
            f"Named here: {names}; np, pd, mne, plt, onset_hfo.\n"
            "signal is raw's samples in volts, read-only (np.array(signal) to change a "
            "copy). Figures open in windows of their own. The working directory is the "
            "Files pane's folder.\n"
            "Commands run here are listed in the exported report. Stop (or Esc) "
            "interrupts a long computation, as Ctrl+C does.\n")


BANNER = _banner(None)


def _monospace() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    font.setPointSizeF(max(9.0, font.pointSizeF()))
    return font


def _raised(before):
    """The exception the plain console just printed, if a new one was."""
    after = getattr(sys, "last_value", None)
    return after if after is not None and after is not before else None


def describe_error(error) -> str:
    """An exception as the model is shown it: the last frames and the message."""
    import traceback

    if error is None:
        return ""
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    text = "".join(lines).strip().split("\n")
    return "\n".join(text[-14:])


#: How often the window is let through to answer while console code runs.
PUMP_SECONDS = 0.1
#: How long code runs before the busy strip shows, so a quick command never
#: flickers it.
SHOW_AFTER = 0.4

#: Input that waits while code runs: the window repaints and scrolls, but a
#: click or a key that would start something else goes nowhere until the
#: code finishes or is stopped. The busy strip itself is exempt.
_HELD = {QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick,
         QEvent.KeyPress, QEvent.KeyRelease, QEvent.ContextMenu, QEvent.Drop,
         QEvent.DragEnter, QEvent.DragMove, QEvent.TouchBegin, QEvent.TabletPress}


def _tick_signal() -> int | None:
    """The signal the ticks arrive on: one nothing else here uses."""
    return getattr(signal, "SIGUSR1", None) or getattr(signal, "SIGBREAK", None)


def _on_tick(_signum, _frame) -> None:
    running = _Running.current
    if running is not None:
        running._tick()


def _stop_key(event) -> bool:
    return event.key() == Qt.Key_Escape or (
        event.key() == Qt.Key_C and bool(event.modifiers() & Qt.ControlModifier))


def _widget_of(obj) -> QWidget | None:
    """The widget an event's target belongs to: itself, or for a shortcut or
    an action, the widget that owns it."""
    while obj is not None and not isinstance(obj, QWidget):
        obj = obj.parent()
    return obj


class _Running(QObject):
    """Console code running in the window's own thread, kept answerable.

    The code cannot be moved off the window's thread: it is handed the
    window's own objects, and Qt and matplotlib are not safe to share across
    threads. So while it runs, a small thread asks for a tick ten times a
    second (`_thread.interrupt_main` with a signal of its own -- no signal is
    sent to the process, so no system call is cut short); Python runs the
    tick between two steps of the code, and the tick lets the window repaint
    and answer. A Stop pressed then raises KeyboardInterrupt in the code, at
    the step it was on, which IPython reports as it reports Ctrl+C.

    While code runs, the window's own input waits, all but the busy strip: a
    click elsewhere could start more code, or rebuild a panel the code is
    reading. A dialog or figure window the code itself opens takes input as
    usual, and so does the console while the code waits on `input()`.
    Closing the window stops the code and closes the window once it has
    stopped."""

    #: The one whose code is running now.
    current = None
    #: The signal the tick handler is on: None until tried, 0 when it could not
    #: be (not on the main thread, or the signal is already someone's).
    _signal: int | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active = False
        self.stop_requested = False
        #: Whether the last code to run was stopped.
        self.stopped = False
        self.started = 0.0
        self._done = threading.Event()
        self._ticker: threading.Thread | None = None
        self._in_tick = False
        self._close_after: list = []
        #: Returns the console's own widget while the code waits on input(),
        #: else None: typing must reach it then.
        self.reading = lambda: None
        self.strip = QFrame()
        self.strip.setObjectName("onset_console_busy")
        row = QHBoxLayout(self.strip)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(90, 8)
        self.label = QLabel("Running…")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("onset_console_stop")
        self.stop_button.setToolTip("Interrupt the running code, as Ctrl+C does (Esc)")
        self.stop_button.clicked.connect(self.request_stop)
        row.addWidget(self.bar)
        row.addWidget(self.label, 1)
        row.addWidget(self.stop_button)
        self.strip.hide()

    @classmethod
    def _install(cls) -> int:
        if cls._signal is None:
            number, cls._signal = _tick_signal(), 0
            if number is not None and threading.current_thread() is threading.main_thread():
                try:
                    if signal.getsignal(number) in (signal.SIG_DFL, None):
                        signal.signal(number, _on_tick)
                        cls._signal = int(number)
                except (OSError, ValueError):
                    pass
        return cls._signal

    def begin(self) -> None:
        if self.active:
            return
        self.stop_requested = self.stopped = False
        self.started = time.monotonic()
        number = self._install()
        if not number or QApplication.instance() is None:
            return              # the window waits, as it did before
        self.active = True
        _Running.current = self
        QApplication.instance().installEventFilter(self)
        self._done.clear()
        self._ticker = threading.Thread(target=self._ticks, args=(number,),
                                        name="onset-console-ticks", daemon=True)
        self._ticker.start()

    def _ticks(self, number: int) -> None:
        while not self._done.wait(PUMP_SECONDS):
            _thread.interrupt_main(number)

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        if _Running.current is self:
            _Running.current = None
        self._done.set()
        if self._ticker is not None:
            self._ticker.join(1.0)
            self._ticker = None
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.strip.hide()
        closing, self._close_after = self._close_after, []
        for window in closing:
            QTimer.singleShot(0, window.close)

    @property
    def seconds(self) -> float:
        return time.monotonic() - self.started

    def request_stop(self) -> None:
        if self.active:
            self.stop_requested = True
            self.label.setText("Stopping…")

    def _tick(self) -> None:
        if not self.active or self._in_tick:
            return
        self._in_tick = True
        try:
            seconds = self.seconds
            if seconds >= SHOW_AFTER and self.strip.isHidden() and not self.stop_requested:
                self.strip.show()
            if not self.stop_requested:
                self.label.setText(f"Running… {seconds:.0f} s")
            QApplication.processEvents()
        finally:
            self._in_tick = False
        if self.stop_requested and self.active:
            # Once per press: code that catches it and carries on can be
            # stopped again.
            self.stop_requested = False
            self.stopped = True
            self.label.setText(f"Running… {self.seconds:.0f} s")
            raise KeyboardInterrupt("stopped from the console's Stop button")

    def eventFilter(self, watched, event):      # noqa: N802  (Qt's spelling)
        if not self.active:
            return False
        kind = event.type()
        if kind == QEvent.Close and isinstance(watched, QMainWindow):
            event.ignore()
            self.request_stop()
            if watched not in self._close_after:
                self._close_after.append(watched)
            return True
        if kind not in _HELD and kind not in (QEvent.Shortcut, QEvent.ShortcutOverride):
            return False
        widget = _widget_of(watched)
        if widget is None:
            return False
        if widget is self.strip or self.strip.isAncestorOf(widget):
            return False
        reading = self.reading()
        if reading is not None and (widget is reading or reading.isAncestorOf(widget)):
            if kind == QEvent.KeyPress and _stop_key(event):
                self.request_stop()
                return True
            return False
        if not isinstance(widget.window(), QMainWindow):
            return False        # a dialog or a window the running code opened
        if kind == QEvent.ShortcutOverride:
            # Taken, so the key arrives as a key press, and a menu shortcut
            # does not start something while the code runs.
            event.accept()
            return True
        if kind == QEvent.KeyPress and _stop_key(event):
            self.request_stop()
        return True


class _BasicConsole(QWidget):
    """A plain Python console: what is there when IPython is not."""

    def __init__(self, ns: dict, banner: str, ran, running: _Running | None = None,
                 parent=None):
        super().__init__(parent)
        self.namespace = ns
        self._ran = ran
        self._running = running
        self._interp = code.InteractiveConsole(ns)
        self._buffer: list[str] = []
        self._history: list[str] = []
        self._cursor = 0
        self.output = QPlainTextEdit()
        self.output.setObjectName("onset_console_output")
        self.output.setReadOnly(True)
        self.output.setFont(_monospace())
        self.output.setPlainText(banner)
        self.input = QLineEdit()
        self.input.setObjectName("onset_console_input")
        self.input.setFont(_monospace())
        self.input.setPlaceholderText(">>> type Python and press Enter")
        self.input.returnPressed.connect(self._enter)
        QShortcut(QKeySequence(Qt.Key_Up), self.input, activated=lambda: self._recall(-1))
        QShortcut(QKeySequence(Qt.Key_Down), self.input, activated=lambda: self._recall(1))
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self.output, 1)
        box.addWidget(self.input)

    def _recall(self, step: int) -> None:
        if not self._history:
            return
        self._cursor = max(0, min(len(self._history), self._cursor + step))
        self.input.setText(self._history[self._cursor] if self._cursor < len(self._history)
                           else "")

    def _enter(self) -> None:
        line = self.input.text()
        self.input.clear()
        if line.strip():
            self._history.append(line)
        self._cursor = len(self._history)
        self._write(("... " if self._buffer else ">>> ") + line + "\n")
        self._buffer.append(line)
        before = getattr(sys, "last_value", None)
        more = self._capture(lambda: self._interp.push(line), timed=True)
        if not more:
            source = "\n".join(self._buffer)
            self._buffer = []
            self._ran(source, _raised(before))

    def run(self, source: str) -> None:
        """Run a block as if typed, the result of a last expression echoed."""
        self._write("".join((">>> " if i == 0 else "... ") + line + "\n"
                            for i, line in enumerate(source.splitlines())))
        symbol = "single" if "\n" not in source.strip() else "exec"
        before = getattr(sys, "last_value", None)
        self._capture(lambda: self._interp.runsource(source, "<console>", symbol), timed=True)
        self._ran(source, _raised(before))

    def _capture(self, call, timed: bool = False):
        out = io.StringIO()
        running = self._running if timed else None
        if running is not None:
            running.begin()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                result = call()
        except KeyboardInterrupt:
            # Stopped between compiling and running: nothing of it ran.
            out.write("KeyboardInterrupt\n")
            self._interp.resetbuffer()
            result = False
        finally:
            if running is not None:
                running.end()
        self._write(out.getvalue())
        return result

    def _write(self, text: str) -> None:
        if text:
            self.output.moveCursor(QTextCursor.End)
            self.output.insertPlainText(text)
            self.output.ensureCursorVisible()

    def text(self) -> str:
        return self.output.toPlainText()

    def shutdown(self) -> None:
        pass


class _Kernel:
    """The one in-process IPython kernel. IPython's shell is a singleton, so a
    second kernel would share the first one's namespace and fire its events
    too; one per process, with each console widget a client of it, and the
    events going to whichever console is in use."""

    instance = None

    def __init__(self):
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        from qtconsole.inprocess import QtInProcessKernelManager

        self.manager = QtInProcessKernelManager()
        self.manager.start_kernel(show_banner=False)
        self.shell = self.manager.kernel.shell
        self.client = self.manager.client()
        self.client.start_channels()
        #: The console the next command belongs to.
        self.active = None
        self.shell.events.register("pre_run_cell", self._before)
        self.shell.events.register("post_run_cell", self._after)

    @classmethod
    def get(cls) -> _Kernel:
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def _before(self, *_args) -> None:
        if self.active is not None:
            self.active._before()

    def _after(self, result) -> None:
        if self.active is not None:
            error = result.error_in_exec or result.error_before_exec
            self.active._ran(result.info.raw_cell if result.info is not None else "",
                             error)


class _RichConsole:
    """IPython through qtconsole: a widget of its own on the shared kernel."""

    def __init__(self, ns: dict, banner: str, owner):
        from qtconsole.rich_jupyter_widget import RichJupyterWidget

        self.kernel = _Kernel.get()
        self.kernel.shell.push(ns)
        self.kernel.active = owner
        widget = RichJupyterWidget()
        widget.setObjectName("onset_console_ipython")
        # The widget prints its banner itself when it connects; ours replaces
        # qtconsole's version line, and IPython's own tip follows it.
        widget.banner = banner
        widget.font = _monospace()
        widget.kernel_manager = self.kernel.manager
        widget.kernel_client = self.kernel.client
        self.widget = widget

    @property
    def namespace(self) -> dict:
        return self.kernel.shell.user_ns

    @property
    def hidden(self) -> set:
        """What IPython itself put in the namespace (`open`, `In`, `Out`...)."""
        return set(self.kernel.shell.user_ns_hidden)

    def run(self, source: str) -> None:
        self.widget.execute(source)

    def text(self) -> str:
        return self.widget._control.toPlainText()

    def shutdown(self) -> None:
        # The kernel is the process's, and outlives any one console.
        if self.kernel.active is not None and self.kernel.active._impl is self:
            self.kernel.active = None


class _FigureWindow(QWidget):
    """A figure a console command drew, in a window of its own."""

    def __init__(self, figure, title: str):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        self.setWindowFlags(Qt.Window)
        self.setWindowTitle(title)
        self.figure = figure
        canvas = FigureCanvasQTAgg(figure)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(NavigationToolbar2QT(canvas, self))
        box.addWidget(canvas, 1)
        width, height = (figure.get_size_inches() * figure.dpi).astype(int)
        self.resize(max(480, int(width)), max(360, int(height) + 40))


class ConsolePanel(QWidget):
    """The Console pane. Built empty and started on first use, so a window
    whose console is never opened never pays for IPython."""

    #: The source of a command just run (after it ran).
    executed = Signal(str)
    #: The working directory, when a command changed it.
    folderChanged = Signal(str)

    def __init__(self, session=None, parent=None, prefer: str = "ipython"):
        super().__init__(parent)
        self.session = session
        self.prefer = prefer
        self.kind = ""
        self.missing: tuple[str, ...] = ()
        self.folder: Path | None = None
        self.figure_windows: list[_FigureWindow] = []
        self._impl = None
        self._injected: dict = {}
        self._figures_before: set[int] = set()
        #: A command's source -> what the log keeps instead (a file's text
        #: for the command that ran it).
        self._log_as: dict[str, str] = {}
        #: The last command's error, as `describe_error` puts it; "" after a
        #: command that ran cleanly.
        self.last_error = ""
        from onset_review.consolepanes import HistoryPanel, PlotsPanel

        #: The Plots and History panes: the console's, so they go where it
        #: goes. A host that shows them puts them on screen.
        self.plots = PlotsPanel(open_window=self._open_figure_window,
                                folder=lambda: self.folder or Path.cwd())
        self.history = HistoryPanel(console=self)
        #: Called when a figure arrives: True when the host showed the Plots
        #: pane, so no window of its own is needed.
        self.plots_shown = lambda: False
        self._box = QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(0)
        #: The code running now, its busy strip and its Stop button.
        self.running = _Running(self)
        self.running.reading = self._reading_input
        self._box.addWidget(self.running.strip)
        self._waiting = QLabel("The Python console starts when this pane is first shown.")
        self._waiting.setWordWrap(True)
        self._waiting.setAlignment(Qt.AlignCenter)
        self._waiting.setStyleSheet(f"color:{theme.current().text_muted};")
        self._box.addWidget(self._waiting, 1)
        self.setMinimumHeight(120)

    # -- starting --------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._impl is not None

    def start(self) -> None:
        if self._impl is not None:
            return
        # plt.show() on the window's non-interactive backend would warn on
        # every call; here figures open in windows of their own instead.
        warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")
        ns = namespace(self.session)
        self._injected = dict(ns)
        banner = _banner(self.session)
        impl = None
        if self.prefer == "ipython":
            try:
                impl = _RichConsole(ns, banner, self)
                self.kind = "ipython"
                widget = impl.widget
            except ImportError as error:
                self.missing = tuple(p for p in RICH_PACKAGES
                                     if p in str(error)) or RICH_PACKAGES
        if impl is None:
            note = ""
            if self.missing:
                note = (f"(The full IPython console needs {' and '.join(self.missing)}: "
                        f"pip install {' '.join(self.missing)}, or re-run the installer.)\n")
            impl = _BasicConsole(ns, banner + note, self._ran, self.running)
            self.kind = "basic"
            widget = impl
        self._impl = impl
        self._box.removeWidget(self._waiting)
        self._waiting.deleteLater()
        self._box.addWidget(widget, 1)
        if self.folder is not None:
            self._chdir(self.folder)

    def showEvent(self, event) -> None:      # noqa: N802
        super().showEvent(event)
        if self._impl is None:
            # Not at once: laying the panes out shows and hides this one in a
            # single step, and starting then would start IPython and change
            # the working directory for a pane nobody opened.
            QTimer.singleShot(0, self._start_if_shown)
        elif self.kind == "ipython":
            self._impl.kernel.active = self

    def _start_if_shown(self) -> None:
        if self._impl is None and self.isVisible():
            self.start()

    # -- what it holds -----------------------------------------------------------
    @property
    def namespace(self) -> dict:
        return self._impl.namespace if self._impl is not None else {}

    def user_variables(self) -> dict:
        hidden = getattr(self._impl, "hidden", set())
        return user_variables(self.namespace, self._injected, hidden)

    def visible_names(self) -> dict:
        """What a name typed in the console finds, as the Workspace lists it:
        the session's names and the user's own, without modules and
        functions. Before the console starts, the session's alone."""
        if self._impl is None:
            return _session_names(self.session)
        names = {k: v for k, v in self._injected.items()
                 if not isinstance(v, types.ModuleType) and k in self.namespace}
        names.update(self.user_variables())
        return names

    def clear_user_variables(self) -> None:
        """Forget every name the user made (IPython's %reset, without the
        window's own)."""
        for name in self.user_variables():
            self.namespace.pop(name, None)

    def text(self) -> str:
        return self._impl.text() if self._impl is not None else ""

    def set_session(self, session) -> None:
        """A new window over a new analysis: the session's names point at
        it; everything the user made stays."""
        self.session = session
        if self._impl is None:
            return
        names = _session_names(session)
        self.namespace.update(names)
        self._injected.update(names)
        label = session.request.label() if session is not None else "no recording"
        self._note(f"\n# The window now shows {label}; session, raw, events and the rest "
                   "point at it. Your own names are as you left them.\n")

    def _note(self, text: str) -> None:
        if self.kind == "ipython":
            self._impl.widget._append_plain_text(text, before_prompt=True)
        elif self._impl is not None:
            self._impl._write(text)

    # -- running -------------------------------------------------------------
    def run(self, source: str) -> None:
        """Run `source` as if typed (tests, and anything that sends code here)."""
        self.start()
        if self.kind == "basic":
            self._before()
        else:
            self._impl.kernel.active = self
        self._impl.run(source)

    def run_file(self, path) -> None:
        """Run a script file in this namespace, as Spyder's runfile does: its
        names land here, and the log keeps the file's text, not just its name."""
        import shlex

        path = Path(path).resolve()
        text = path.read_text(encoding="utf-8")
        self.start()
        if self.kind == "ipython":
            command = f"%run -i {shlex.quote(str(path))}"
        else:
            command = (f"exec(compile(open({str(path)!r}, encoding='utf-8').read(), "
                       f"{str(path)!r}, 'exec'))")
        self._log_as[command] = f"# Ran the file {path}\n{text.rstrip()}"
        self.run(command)

    def _reading_input(self):
        if self.kind == "ipython" and getattr(self._impl.widget, "_reading", False):
            return self._impl.widget
        return None

    @property
    def busy(self) -> bool:
        """Whether code is running in the console now."""
        return self.running.active

    def stop(self) -> None:
        """Interrupt the running code, as Ctrl+C does; nothing when none runs."""
        self.running.request_stop()

    def _before(self) -> None:
        import matplotlib.pyplot as plt

        self._figures_before = set(plt.get_fignums())
        if self.kind == "ipython":
            self.running.begin()

    def _ran(self, source: str, error=None) -> None:
        self.running.end()
        stopped = self.running.stopped and isinstance(error, KeyboardInterrupt)
        self.last_error = describe_error(error)
        logged = self._log_as.pop(source.strip(), source)
        if source.strip():
            if stopped:
                logged = (f"{logged.rstrip()}\n# Stopped by the reader after "
                          f"{self.running.seconds:.0f} s: it did not finish.")
            if self.session is not None:
                self.session.console_log.append(logged.rstrip())
        self._show_new_figures(source)
        if source.strip():
            self.history.add(source)
        here = Path.cwd()
        if self.folder is None or here != self.folder:
            self.folder = here
            self.folderChanged.emit(str(here))
        self.executed.emit(source)

    def _show_new_figures(self, source: str = "") -> None:
        """Each figure the command drew goes to the Plots pane; and to a window
        of its own unless the host has the pane on screen."""
        import matplotlib.pyplot as plt

        for number in sorted(set(plt.get_fignums()) - self._figures_before):
            figure = plt.figure(number)
            plt.close(number)
            try:
                self.plots.add(figure, command=source)
            except Exception:       # noqa: BLE001 - a figure that cannot draw still opens
                pass
            if not self.plots_shown():
                self._open_figure_window(figure, f"Figure {number} — Console")
        self._figures_before = set(plt.get_fignums())

    def _open_figure_window(self, figure, title: str) -> None:
        window = _FigureWindow(figure, title)
        window.show()
        self.figure_windows.append(window)

    # -- the working directory --------------------------------------------------
    def set_folder(self, folder) -> None:
        """Make `folder` the working directory -- once the console is running;
        until then it is only remembered, so a window whose console is never
        used never changes directory."""
        folder = Path(folder)
        self.folder = folder
        if self._impl is not None:
            self._chdir(folder)

    @staticmethod
    def _chdir(folder: Path) -> None:
        try:
            if folder.is_dir() and Path.cwd() != folder:
                os.chdir(folder)
        except OSError:
            pass

    # -- leaving -----------------------------------------------------------------
    def close_figures(self) -> None:
        for window in self.figure_windows:
            window.close()
        self.figure_windows = []

    def shutdown(self) -> None:
        self.close_figures()
        if self._impl is not None:
            self._impl.shutdown()


def connect_panes(workspace, files, console, editor=None) -> None:
    """Workspace, Files, Console and Editor as Spyder links them: the
    Workspace lists what the console made and refreshes after each command;
    the console's working directory and the Files pane's folder follow each
    other; the editor runs in the console and opens its dialogs in the
    Files pane's folder."""
    if console is None:
        return
    if editor is not None:
        editor.console = console
        console.history.editor = editor
        if files is not None:
            editor.folder = lambda: files.folder
        console.executed.connect(lambda _source: editor.assistant.sync())
    if workspace is not None:
        workspace.console = console
        console.executed.connect(lambda _source: workspace.refresh())
    if files is not None:
        console.set_folder(files.folder)
        files.folderChanged.connect(console.set_folder)
        console.folderChanged.connect(
            lambda folder: Path(folder) != files.folder and files.set_folder(folder))
