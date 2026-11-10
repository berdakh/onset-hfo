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
model check, with the home folder written as ``~``, and a few words from the
person on what went wrong. What it does not contain is any signal: the log
records what was opened (a subject, a window, a file name) and what happened,
never data.

**What is shown is what is saved.** The report is built once, as a
:class:`Bundle` -- the exact text of every file -- and both the preview and the
zip are made from that object, so there is no file in the zip that the person
did not have the chance to read, and any file can be left out. `report.txt`
holds no line of any log -- it lists the log files instead -- so unticking the
logs leaves out everything they say, the names of recordings included.
Nothing is sent anywhere: the person sends the file themselves.

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
from functools import partial
from pathlib import Path

from onset_hfo import problemreport
from onset_hfo.problemreport import Bundle

__all__ = ["log_dir", "setup", "about", "report_text", "write_report", "LOG_NAME",
           "CRASH_NAME", "logger", "Bundle", "build_bundle", "write_bundle"]

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

    out = problemreport.about()
    out["onset-review"] = __version__
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


_redact = problemreport.redact


def latest_model_check() -> Path | None:
    folder = log_dir() / "model-checks"
    found = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) \
        if folder.is_dir() else []
    return found[-1] if found else None


def report_text(description: str = "", with_logs: bool = True) -> str:
    """The report as one readable text: what the person said went wrong, the
    versions and the machine, and which log files sit beside it.

    It quotes no log line. The logs name the recordings that were opened, and
    they travel as files of their own that the person can untick; a copy of
    their tail in here would go with the report whatever was unticked.
    `with_logs=False` leaves the list out too, for a command line tool that
    never wrote a log."""
    flush()
    parts = ["# Onset Review problem report", time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    if description.strip():
        parts += ["## What happened, in the reporter's words", description.strip(), ""]
    parts.append("## Versions and machine")
    for key, value in about().items():
        parts.append(f"{key}: {value}")
    if not with_logs:
        return _redact("\n".join(parts)) + "\n"
    present = [file for file in _log_files() if file.is_file()]
    parts += ["", "## Logs",
              "Each is a file of its own beside this one, sent only if it was ticked; "
              "this text quotes none of them."]
    parts += [f"- {file.name}" for file in present] or ["- none on this machine"]
    return _redact("\n".join(parts)) + "\n"


def flush() -> None:
    handler = _state.get("handler")
    if handler is not None:
        handler.flush()


def _log_files() -> list[Path]:
    folder = log_dir()
    files = [folder / LOG_NAME] + [folder / f"{LOG_NAME}.{n}" for n in range(1, BACKUPS + 1)]
    files.append(folder / CRASH_NAME)
    check = latest_model_check()
    if check is not None:
        files += [check, check.with_suffix(".md")]
    return files


def build_bundle(description: str = "", with_logs: bool = True) -> Bundle:
    """The report for this machine, once: `report.txt` and, when `with_logs`,
    every log, the crash log and the latest model check that exist.

    Every file is read and redacted here, not later, so what the caller shows
    is what `write_bundle` writes. `with_logs=False` is for a command line tool
    that has no window log: the versions, the machine and the description.
    """
    flush()
    files = {"report.txt": report_text(description=description, with_logs=with_logs)}
    if with_logs:
        for file in _log_files():
            if file.is_file():
                files[file.name] = _redact(file.read_text(encoding="utf-8", errors="replace"))
    return Bundle(files, with_logs, partial(report_text, with_logs=with_logs))


def write_bundle(bundle: Bundle, path: str | Path) -> Path:
    """Write exactly `bundle` as one zip, and nothing else."""
    path = problemreport.write_bundle(bundle, path)
    logger.info("problem report written: %s (%s)", _redact(str(path)),
                ", ".join(bundle.files))
    return path


def write_report(path: str | Path, description: str = "") -> Path:
    """One zip to send: report.txt, the logs, the crash log and the latest
    model check, each with the home folder written as ``~``."""
    return write_bundle(build_bundle(description), path)
