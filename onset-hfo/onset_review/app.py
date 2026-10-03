"""`onset-review`: the entry point, and the arguments that skip the dialog.

Two ways in, because two people use this. A clinician launches it from the
applications menu, gets the dialog, and never sees an argument. A developer or
a demo runs

    onset-review --subject sub-01 --window 0 60 --band ripple --expert

and lands on the trace in one step. The flags mirror `ReviewRequest` field for
field so that anything reproducible from a report is reproducible from a shell.

Offline is the default. `ONSET_HFO_OFFLINE=1` is set before anything imports the
loader unless `--allow-fetch` was passed, so the software cannot reach for a
700 MB archive on a hospital network without being asked to. That is a
deliberate inversion of the library's default, where fetching is the point.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]

#: Exit codes, so a wrapper script can tell a refusal from a crash.
EXIT_OK, EXIT_CANCELLED, EXIT_NO_QT, EXIT_FAILED = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onset-review",
        description="Review high-frequency oscillations in public intracranial "
                    "EEG. Research prototype — not a medical device.")
    parser.add_argument("--subject", help="e.g. sub-01. Skips the open dialog.")
    parser.add_argument("--dataset", default="ds003498",
                        help="OpenNeuro dataset id (default: %(default)s)")
    parser.add_argument("--run", default="01")
    parser.add_argument("--task", default=None,
                        help="e.g. ictal, for datasets that name a task")
    parser.add_argument("--window", nargs=2, type=float, metavar=("START", "STOP"),
                        default=[0.0, 60.0],
                        help="seconds of the original recording (default: 0 60)")
    parser.add_argument("--band", default="ripple",
                        choices=["ripple", "fast_ripple"])
    parser.add_argument("--detector", action="append", dest="detectors",
                        help="repeatable; default rms")
    parser.add_argument("--threshold-sd", type=float, default=None,
                        help="override each detector's measured threshold")
    parser.add_argument("--no-spikes", action="store_true",
                        help="skip interictal discharge detection")
    parser.add_argument("--expert", action="store_true",
                        help="overlay the archive's expert markings on open")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where cached slices live (default artifacts/data)")
    parser.add_argument("--allow-fetch", action="store_true",
                        help="permit downloading a window that is not cached")
    parser.add_argument("--list", action="store_true",
                        help="print the cached windows and exit")
    parser.add_argument("--export", type=Path, default=None,
                        help="write the review to this path and exit without a "
                             "window (.md or .html); implies --subject")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="save a PNG of the window and exit; for docs and "
                             "for checking a headless install")
    return parser


def _request_from(args) -> object:
    from onset_review.session import ReviewRequest

    return ReviewRequest(
        dataset=args.dataset, subject=args.subject, run=args.run, task=args.task,
        t_start=float(args.window[0]), t_stop=float(args.window[1]),
        detectors=tuple(args.detectors or ("rms",)), band=args.band,
        threshold_sd=args.threshold_sd, with_spikes=not args.no_spikes)


def _print_cached(cache_dir: Path | None) -> int:
    from onset_review.session import cached_windows

    frame = cached_windows(cache_dir)
    if frame.empty:
        print("No cached windows. Fetch one with:\n"
              "  python -m onset_hfo.cli fetch --subject sub-01 "
              "--t-start 0 --t-stop 60")
        return EXIT_FAILED
    columns = ["dataset", "subject", "task", "run", "t_start", "t_stop",
               "sfreq", "n_channels", "bands"]
    # ds003498 names no task, so that column is empty for most rows; printing
    # pandas' NaN there reads as a missing value rather than an absent one.
    print(frame[columns].fillna({"task": "—"}).to_string(index=False))
    print(f"\n{len(frame)} cached window(s). Any of these opens offline.")
    return EXIT_OK


def _export_only(args) -> int:
    """Produce the review document with no display at all.

    Worth having for its own sake -- a batch of reviews is a useful thing to
    generate overnight -- and worth having for the test suite, which can then
    check the whole product end to end on a machine with no Qt installed.
    """
    from onset_review.report import write_review
    from onset_review.session import load_session

    session = load_session(_request_from(args), cache_dir=args.cache_dir,
                           progress=lambda f, m: print(f"[{f:4.0%}] {m}",
                                                       file=sys.stderr))
    written = write_review(session, args.export)
    print(written)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Set before the loader is imported, so the refusal is in place from the
    # first fetch rather than checked later and hopefully honoured.
    if not args.allow_fetch:
        os.environ.setdefault("ONSET_HFO_OFFLINE", "1")

    if args.list:
        return _print_cached(args.cache_dir)
    if args.export is not None:
        if not args.subject:
            print("--export needs --subject", file=sys.stderr)
            return EXIT_FAILED
        return _export_only(args)

    try:
        from qtpy.QtWidgets import QApplication
    except ImportError as error:
        print(f"The review window needs Qt, which is not installed ({error}).\n"
              f"Install it with:  pip install 'onset-hfo[review]'\n"
              f"Or use --export to produce the review document without a "
              f"display.", file=sys.stderr)
        return EXIT_NO_QT

    from onset_review import launcher

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Onset Review")
    app.setApplicationDisplayName("Onset Review")

    if args.subject:
        request, overlay = _request_from(args), args.expert
    else:
        request, overlay = launcher.choose_request(args.cache_dir)
        if request is None:
            return EXIT_CANCELLED

    session = launcher.load_with_progress(request, args.cache_dir)
    if session is None:
        return EXIT_FAILED

    review = _Review(app, request, overlay, args)
    review.open(session)

    if args.screenshot is not None:
        return _screenshot(app, review.parts, args.screenshot)
    return app.exec_() if hasattr(app, "exec_") else app.exec()


class _Review:
    """Owns the one open window, and replaces it when preprocessing changes.

    Changing a filter changes every number in every panel -- the rates, the
    intervals, the candidate set, the trend, the 3D layout, the agreement, the
    assistant's evidence -- so the honest response is to rebuild them all from
    the signal up rather than to refresh some and leave others stale. MNE's
    browser cannot be handed a different recording either, so the window itself
    is replaced.

    What survives is the reviewer's own arrangement: the dock layout is saved
    and restored across the swap, so applying a filter does not cost someone
    the panels they had dragged where they wanted them.
    """

    def __init__(self, app, request, overlay: bool, args):
        self.app = app
        self.request = request
        self.overlay = overlay
        self.args = args
        self.parts = None

    def open(self, session) -> None:
        from onset_review import window

        previous = self.parts
        state = previous.host.saveState() if previous is not None else None

        figure = window.open_trace(session, show_expert=self.overlay,
                                   show=self.args.screenshot is None)
        self.parts = window.decorate(figure, session, show_expert=self.overlay,
                                     on_preprocess=self.reanalyse)
        if state is not None:
            # Restored after the docks exist and before the window is shown, so
            # the reviewer never sees the default arrangement flash past.
            self.parts.host.restoreState(state)
            self.parts.host.resize(previous.host.size())
        self.parts.host.show()
        if previous is not None:
            try:
                previous.figure.close()
            except Exception:
                pass

    def reanalyse(self, preprocess) -> None:
        """Re-run this window under new preprocessing, and replace the view.

        A failure leaves the current window exactly as it was. `load_with_progress`
        has already told the reviewer what went wrong, and the alternative --
        closing a working window because a setting was rejected -- would lose
        them their place for no reason.
        """
        import dataclasses

        from onset_review import launcher

        request = dataclasses.replace(self.request, preprocess=preprocess)
        session = launcher.load_with_progress(request, self.args.cache_dir,
                                              parent=self.parts.host)
        if session is None:
            return
        self.request = request
        self.open(session)


def _screenshot(app, parts, path: Path) -> int:
    """Render the window to a file without ever showing it.

    This is how the documentation screenshots are produced and how a headless
    install is verified: if this succeeds, every panel built, every table
    populated and the trace rendered, which is most of what a smoke test can
    establish without a human looking.
    """
    from qtpy.QtCore import QThread

    parts.host.resize(1680, 980)
    for _ in range(12):      # let the trace's own load thread settle
        app.processEvents()
        QThread.msleep(60)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = parts.host.grab().save(str(path))

    # Close before returning. `mne-qt-browser` loads and downsamples its data on
    # a worker thread, and returning from here with that thread still running
    # tears the interpreter down underneath it -- which segfaults after the
    # screenshot has already been written, so the file looks fine and the exit
    # code does not. Closing the figure stops the thread; the drain gives it
    # somewhere to finish.
    try:
        parts.figure.close()
    except Exception:
        pass
    for _ in range(10):
        app.processEvents()
        QThread.msleep(30)

    print(path if ok else f"could not write {path}",
          file=sys.stdout if ok else sys.stderr)

    # Leave without running interpreter shutdown. Closing above stops the load
    # thread most of the time, but `mne-qt-browser` and PySide6 still race each
    # other during teardown often enough to segfault roughly one run in six --
    # after the file is written, so the screenshot is fine and only the exit
    # code is wrong. That is the worst possible shape for a verification
    # command, which exists to be believed: a flaky non-zero exit on a correct
    # render is indistinguishable from a real failure in CI.
    #
    # Everything this process owns is finished and flushed by here, so there is
    # nothing left for teardown to do. Only this path does it; the interactive
    # window exits through the event loop like any Qt application.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_OK if ok else EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
