"""Closing the window ends the program, whatever is still working.

Reported from a laptop: the window closed, the terminal never got its prompt
back, and the process had to be killed. The assistant had been asked a
question; the local model, on a CPU, was still answering, and the window's
nested event loop -- quit by the close -- then waited for the answer with no
limit. Each test here runs a real application in a process of its own, closes
its window while a worker is busy, and times how long the process takes to
end: a stoppable worker is stopped, and one that cannot stop is left.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent('''
    import sys, time
    from qtpy.QtCore import QThread, QTimer
    from qtpy.QtWidgets import QApplication, QWidget
    from onset_review import workers

    stoppable = sys.argv[1] == "stoppable"

    class Busy(QThread):
        """Sixty seconds of work: a local model answering on a CPU."""
        def __init__(self):
            super().__init__()
            self.stopped = False
            self.error = None
        def stop(self):
            if stoppable:
                self.stopped = True
        def run(self):
            for _ in range(600):
                if self.stopped:
                    return
                time.sleep(0.1)

    app = QApplication([])
    window = QWidget()
    window.show()

    def ask():
        QTimer.singleShot(500, window.close)        # the reader closes the window
        finished = workers.run(Busy())
        print("run returned", finished, flush=True)

    QTimer.singleShot(0, ask)
    code = app.exec_() if hasattr(app, "exec_") else app.exec()
    print("loop ended", flush=True)
    sys.exit(workers.finish(code))
''')


def _run(mode: str) -> tuple[subprocess.CompletedProcess, float]:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-c", SCRIPT, mode], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=50)
    return done, time.monotonic() - start


def test_a_worker_answering_when_the_window_closes_is_stopped_and_the_program_ends():
    done, seconds = _run("stoppable")
    assert done.returncode == 0, done.stderr
    assert "run returned False" in done.stdout, "the close ended the wait, not the answer"
    assert "loop ended" in done.stdout
    assert seconds < 20, f"the process took {seconds:.0f} s to end after its window closed"
    assert "left without waiting" not in done.stderr


def test_a_worker_that_cannot_stop_does_not_keep_the_program_alive():
    done, seconds = _run("unstoppable")
    assert done.returncode == 0, done.stderr
    assert "loop ended" in done.stdout
    assert "left without waiting for it" in done.stderr
    assert seconds < 25, f"the process took {seconds:.0f} s to end after its window closed"


def test_a_worker_that_finishes_is_waited_for():
    from qtpy.QtCore import QThread
    from qtpy.QtWidgets import QApplication

    from onset_review import workers

    QApplication.instance() or QApplication([])

    class Quick(QThread):
        def __init__(self):
            super().__init__()
            self.value = None

        def run(self):
            self.value = 42

    worker = Quick()
    assert workers.run(worker) is True and worker.value == 42
    assert worker not in workers.running()
