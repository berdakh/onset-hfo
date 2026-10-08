"""Background work that never keeps the application alive after its window.

Every slow thing the window does runs on a `QThread`: a question to the local
model, a draft of code, the analysis behind the assistant, a re-run of a
study, a model being pulled. Two rules keep closing the window what it should
be -- the end of the program:

* `run` waits for a worker inside a nested event loop, so the window keeps
  painting. Closing the window quits every event loop, the nested one
  included; `run` then asks the worker to stop (a model request in flight is
  cut off, so the local model stops generating) and waits a few seconds at
  most, rather than waiting for an answer nobody will read. A local model on
  a CPU can take minutes, and the process used to wait for all of them.
* every worker is registered here (`track`), and when the application's loop
  has ended `stop_all` stops whatever is still running. `finish` is what the
  entry point calls last: if a worker cannot stop in time -- an analysis
  mid-computation has nowhere to stop -- the process leaves at once instead
  of hanging the terminal it was started from, or crashing as Python tears a
  running thread down.
"""

from __future__ import annotations

import os
import sys
import weakref

__all__ = ["track", "run", "running", "stop_all", "finish", "STOP_WAIT_MS"]

#: How long a worker asked to stop is given, at most, before it is left.
STOP_WAIT_MS = 3000

_TRACKED: weakref.WeakSet = weakref.WeakSet()


def track(worker):
    """Register `worker` (a QThread) so it is stopped when the program ends."""
    _TRACKED.add(worker)
    return worker


def running() -> list:
    out = []
    for worker in list(_TRACKED):
        try:
            if worker.isRunning():
                out.append(worker)
        except RuntimeError:        # its C++ side is already gone
            continue
    return out


def _stop(worker) -> None:
    stop = getattr(worker, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:      # noqa: BLE001 - stopping is best effort
            pass
    try:
        worker.requestInterruption()
    except RuntimeError:
        pass


def run(worker) -> bool:
    """Start `worker` and wait for it in a nested event loop, the window
    painting meanwhile. True if it finished; False if the loop was ended
    from outside -- the window closing -- in which case the worker was asked
    to stop and given `STOP_WAIT_MS`, and its `error` says it was stopped."""
    from qtpy.QtCore import QEventLoop

    track(worker)
    done = {"finished": False}

    def finished() -> None:
        done["finished"] = True
        loop.quit()

    loop = QEventLoop()
    worker.finished.connect(finished)
    worker.start()
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()
    if done["finished"] or not worker.isRunning():
        worker.wait()
        return True
    _stop(worker)
    worker.wait(STOP_WAIT_MS)
    if hasattr(worker, "error") and not getattr(worker, "error", None):
        try:
            worker.error = "stopped: the window was closed"
        except AttributeError:
            pass
    return False


def stop_all(wait_ms: int = STOP_WAIT_MS) -> list:
    """Ask every running worker to stop and wait for them together, up to
    `wait_ms`. Returns the ones still running."""
    from qtpy.QtCore import QDeadlineTimer

    left = running()
    for worker in left:
        _stop(worker)
    deadline = QDeadlineTimer(wait_ms)
    for worker in left:
        try:
            worker.wait(deadline)
        except (RuntimeError, TypeError):
            worker.wait(max(0, deadline.remainingTime()))
    return running()


def finish(code: int) -> int:
    """The program's last step after the application's loop has ended:
    stop what is still running, and if anything will not stop, leave now --
    a terminal must get its prompt back when the window is closed."""
    left = stop_all()
    if left:
        names = ", ".join(sorted({type(w).__name__ for w in left}))
        print(f"onset-review: closed while still working ({names}); left without "
              "waiting for it.", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code
