"""Stopping console code: the window answers while it runs, and Stop (or
Esc, or closing the window) interrupts it as Ctrl+C does.

The code runs in the window's own thread, so what is tested is that the
window is let through while it runs: a timer set before the code starts
fires during it and presses Stop, and the code ends at once rather than at
its natural end, with KeyboardInterrupt as its error and the report saying
it was stopped.
"""

from __future__ import annotations

import os
import time

import pytest

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")

from qtpy.QtCore import QEvent, Qt, QTimer  # noqa: E402
from qtpy.QtGui import QKeyEvent  # noqa: E402

from onset_review.console import ConsolePanel  # noqa: E402
from onset_review.session import ReviewRequest, session_from_recording  # noqa: E402

#: Code that would run for a minute if nothing stopped it.
LONG = "import time\nstart = time.time()\nn = 0\nwhile time.time() - start < 60:\n    n += 1\n"


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield qt.QApplication.instance() or qt.QApplication([])


@pytest.fixture
def review(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


@pytest.fixture(params=["ipython", "basic"])
def console(request, qapp, review):
    panel = ConsolePanel(review, prefer=request.param)
    panel.resize(600, 400)
    panel.show()
    panel.start()
    yield panel
    panel.clear_user_variables()
    panel.shutdown()
    panel.close()


def test_stop_interrupts_running_code_at_once(qapp, console, review):
    seen = {}

    def press_stop():
        seen["busy"] = console.busy
        seen["strip"] = console.running.strip.isVisible()
        console.stop()

    QTimer.singleShot(700, press_stop)
    began = time.monotonic()
    console.run(LONG)
    took = time.monotonic() - began
    assert took < 5, f"the code ran {took:.1f} s after Stop"
    assert seen == {"busy": True, "strip": True}
    assert not console.busy and console.running.strip.isHidden()
    assert "KeyboardInterrupt" in console.last_error
    assert console.namespace["n"] > 0          # it did run, up to the stop
    assert review.console_log[-1].endswith("it did not finish.")
    # The console takes the next command as usual.
    console.run("after = 6 * 7")
    assert console.namespace["after"] == 42 and console.last_error == ""
    assert not review.console_log[-1].endswith("it did not finish.")


def test_esc_stops_and_other_input_waits_while_code_runs(qapp, console):
    from qtpy.QtWidgets import QMainWindow, QPushButton

    host = QMainWindow()
    other = QPushButton("elsewhere")
    host.setCentralWidget(other)
    clicked = []
    other.clicked.connect(lambda: clicked.append(True))
    host.show()

    def during():
        other.click()                                # programmatic: not input
        clicked.clear()
        qt.QApplication.sendEvent(other, QKeyEvent(QEvent.KeyPress, Qt.Key_Space,
                                                   Qt.NoModifier))
        qt.QApplication.sendEvent(other, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space,
                                                   Qt.NoModifier))
        qt.QApplication.sendEvent(other, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                                   Qt.NoModifier))

    QTimer.singleShot(500, during)
    began = time.monotonic()
    console.run(LONG)
    assert time.monotonic() - began < 5
    assert clicked == []                 # the key on the other button waited
    assert "KeyboardInterrupt" in console.last_error
    host.close()


def test_a_dialog_the_code_opens_still_takes_input(qapp, console):
    from qtpy.QtWidgets import QDialog, QMainWindow, QPushButton

    host = QMainWindow()
    host.setCentralWidget(console)
    host.show()
    pressed = []
    probe = {}

    def during():
        dialog = QDialog()
        button = QPushButton("OK", dialog)
        button.clicked.connect(lambda: pressed.append(True))
        dialog.show()
        qt.QApplication.sendEvent(button, QKeyEvent(QEvent.KeyPress, Qt.Key_Space,
                                                    Qt.NoModifier))
        qt.QApplication.sendEvent(button, QKeyEvent(QEvent.KeyRelease, Qt.Key_Space,
                                                    Qt.NoModifier))
        probe["dialog"] = list(pressed)
        dialog.close()
        console.stop()

    QTimer.singleShot(500, during)
    console.run(LONG)
    assert probe["dialog"] == [True], "a window the code opened is not held"
    assert "KeyboardInterrupt" in console.last_error
    host.takeCentralWidget()
    host.close()


def test_closing_the_window_stops_the_code_then_closes(qapp, review):
    from qtpy.QtWidgets import QMainWindow

    host = QMainWindow()
    panel = ConsolePanel(review, prefer="ipython")
    host.setCentralWidget(panel)
    host.show()
    panel.start()
    QTimer.singleShot(500, host.close)
    began = time.monotonic()
    panel.run(LONG)
    assert time.monotonic() - began < 5
    assert "KeyboardInterrupt" in panel.last_error
    for _ in range(20):
        qapp.processEvents()
    assert not host.isVisible()
    panel.clear_user_variables()
    panel.shutdown()


def test_quick_code_never_shows_the_strip(qapp, console):
    console.run("quick = sum(range(1000))")
    assert console.namespace["quick"] == 499500
    assert console.running.strip.isHidden() and not console.busy
