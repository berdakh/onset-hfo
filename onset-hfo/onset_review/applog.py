"""The application's own log, and a problem report built from it: Qt-free.

Until now a problem reached the developers as a photograph of a terminal.
The window now keeps a log of its own -- what was opened, every background
job and how long it took or why it stopped, every warning and every uncaught
error with its traceback, and Qt's own complaints -- in a file that rotates
at a megabyte (``onset-review.log`` under ``~/.local/state/onset-review``).
A crash that takes the interpreter down (a segfault in a C extension) is
written by `faulthandler` to ``crash.log`` beside it, which no Python logging
can reach.

*Help → Report a problem* turns that into one file to send: the versions of
everything that matters, the machine, the log, the crash log and the latest
model check, with the home folder written as ``~``. What it does not contain
is any signal: the log records what was opened (a subject, a window, a file
name) and what happened, never data.

The log is for diagnosis, not provenance: the record of an analysis is the
exported report, as before.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

__all__ = ["log_dir", "setup", "about", "report_text", "write_report", "LOG_NAME",
           "CRASH_NAME", "logger"]

LOG_NAME = "onset-review.log"
CRASH_NAME = "crash.log"
#: Rotated at this size, three generations kept: a few weeks of ordinary use.
MAX_BYTES = 1_000_000
BACKUPS = 3

logger = logging.getLogger("onset_review")

_state: dict = {}


def log_dir() -> Path:
    """Where the log lives: ``ONSET_REVIEW_LOG_DIR``; beside a redirected
    settings directory (the tests, a portable install); else the XDG state
    directory."""
    named = os.environ.get("ONSET_REVIEW_LOG_DIR")
    if named:
        return Path(named).expanduser()
    if os.environ.get("ONSET_REVIEW_CONFIG_DIR"):
        return Path(os.environ["ONSET_REVIEW_CONFIG_DIR"]).expanduser() / "logs"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "onset-review"


def setup(level: int = logging.INFO) -> Path | None:
    """Start logging to the file, once per process: the file handler, warnings
    captured, uncaught exceptions logged, and the crash log armed. Returns
    the log's path, or None when the directory cannot be written -- a log is
    never a reason for the window not to open."""
    if _state.get("path") is not None:
        return _state["path"]
    folder = log_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            folder / LOG_NAME, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    logging.captureWarnings(True)
    _state["handler"] = handler
    _state["path"] = folder / LOG_NAME

    previous = sys.excepthook

    def excepthook(kind, value, traceback):
        logger.error("uncaught %s", kind.__name__, exc_info=(kind, value, traceback))
        previous(kind, value, traceback)

    sys.excepthook = excepthook
    try:
        crash = open(folder / CRASH_NAME, "a", encoding="utf-8")      # noqa: SIM115
        crash.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} ---\n")
        crash.flush()
        faulthandler.enable(file=crash, all_threads=True)
        _state["crash"] = crash
    except OSError:
        pass
    logger.info("started %s", _versions_line())
    return _state["path"]


def install_qt_handler() -> None:
    """Qt's own warnings -- "QThread: Destroyed while thread is still
    running", a failed font, a bad stylesheet -- into the log too."""
    try:
        from qtpy.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING,
              QtMsgType.QtCriticalMsg: logging.ERROR, QtMsgType.QtFatalMsg: logging.CRITICAL}
    qt_log = logging.getLogger("qt")

    def handler(kind, _context, message):
        qt_log.log(levels.get(kind, logging.INFO), "%s", message)
        if kind != QtMsgType.QtDebugMsg:
            sys.stderr.write(f"{message}\n")

    qInstallMessageHandler(handler)


# -- what the report says ----------------------------------------------------------
def _version(module: str) -> str:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:      # noqa: BLE001 - not installed, or no metadata
        return "not installed"


def _versions_line() -> str:
    from onset_review import __version__

    return (f"onset-review {__version__}, Python {platform.python_version()}, "
            f"{platform.system()} {platform.release()}")


def about() -> dict:
    """Versions, machine and settings, for a report. No recording data."""
    from onset_review import __version__

    try:
        from onset_hfo.config import PIPELINE_VERSION
    except Exception:      # noqa: BLE001
        PIPELINE_VERSION = "?"     # noqa: N806
    out = {
        "onset-review": __version__, "pipeline": PIPELINE_VERSION,
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpus": os.cpu_count(),
    }
    try:
        from onset_agent.hardware import total_ram_gb

        out["ram_gb"] = round(float(total_ram_gb()), 1)
    except Exception:      # noqa: BLE001
        pass
    for package in ("PySide6-Essentials", "PySide6", "qtpy", "mne", "mne-qt-browser",
                    "numpy", "scipy", "pandas", "matplotlib", "scikit-learn", "qtconsole",
                    "ipykernel"):
        out[package] = _version(package)
    try:
        from qtpy.QtCore import qVersion
        from qtpy.QtWidgets import QApplication

        out["qt"] = qVersion()
        app = QApplication.instance()
        if app is not None:
            out["screens"] = [f"{s.size().width()}x{s.size().height()} "
                              f"@{s.devicePixelRatio():g}" for s in app.screens()]
            out["platform_plugin"] = app.platformName()
    except Exception:      # noqa: BLE001 - no Qt is a fact worth reporting too
        out["qt"] = "not available"
    try:
        from onset_review.assistant_config import config_path, load_defaults

        defaults = load_defaults()
        out["assistant"] = {"kind": getattr(defaults, "kind", ""),
                            "model": getattr(defaults, "model", "")}
        out["settings"] = str(config_path().parent)
    except Exception:      # noqa: BLE001
        pass
    out["log"] = str(log_dir())
    return out


def _redact(text: str) -> str:
    home = str(Path.home())
    return text.replace(home, "~") if home and home != "/" else text


def _tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def latest_model_check() -> Path | None:
    folder = log_dir() / "model-checks"
    found = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) \
        if folder.is_dir() else []
    return found[-1] if found else None


def report_text(lines: int = 200) -> str:
    """The report as one readable text: about, the log's tail, the crash log's."""
    flush()
    parts = ["# Onset Review problem report", time.strftime("%Y-%m-%d %H:%M:%S"), "",
             "## Versions and machine"]
    for key, value in about().items():
        parts.append(f"{key}: {value}")
    parts += ["", f"## The last {lines} lines of the log", _tail(log_dir() / LOG_NAME, lines)]
    crash = _tail(log_dir() / CRASH_NAME, 80)
    if crash.strip():
        parts += ["", "## The crash log (Python's faulthandler)", crash]
    check = latest_model_check()
    if check is not None:
        parts += ["", f"## The latest model check: {check.name} (in the saved report)"]
    return _redact("\n".join(parts)) + "\n"


def flush() -> None:
    handler = _state.get("handler")
    if handler is not None:
        handler.flush()


def write_report(path: str | Path) -> Path:
    """One zip to send: report.txt, the logs, the crash log and the latest
    model check, each with the home folder written as ``~``."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    flush()
    folder = log_dir()
    files = [folder / LOG_NAME] + [folder / f"{LOG_NAME}.{n}" for n in range(1, BACKUPS + 1)]
    files.append(folder / CRASH_NAME)
    check = latest_model_check()
    if check is not None:
        files += [check, check.with_suffix(".md")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("report.txt", report_text())
        for file in files:
            if file.is_file():
                text = file.read_text(encoding="utf-8", errors="replace")
                bundle.writestr(file.name, _redact(text))
    logger.info("problem report written: %s", _redact(str(path)))
    return path
