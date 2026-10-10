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
import re
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]

#: Exit codes, so a wrapper script can tell a refusal from a crash.
EXIT_OK, EXIT_CANCELLED, EXIT_NO_QT, EXIT_FAILED = 0, 1, 2, 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onset-review",
        description="An AI-assisted research tool for intracranial EEG: "
                    "high-frequency oscillations, interictal discharges and seizure "
                    "onset, in public recordings or your own. Research tool — not a "
                    "medical device.")
    from onset_review import credit

    parser.add_argument("--version", action="version", version=credit())
    parser.add_argument("--subject", help="e.g. sub-01. Skips the open dialog.")
    parser.add_argument("--open", dest="open_path", type=Path, default=None,
                        metavar="FILE",
                        help="a recording on this machine (EDF, BrainVision, "
                             "Persyst, Nihon Kohden, Nicolet, Blackrock, MEF3 "
                             "and the rest of MNE's readers). Opens the import "
                             "dialog so the channel types can be confirmed.")
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
    parser.add_argument("--channel-type", action="append", default=None,
                        dest="channel_types", metavar="NAME=TYPE",
                        help="with --open and --no-confirm: set one channel's "
                             "type, e.g. --channel-type 'AR1=seeg'. Repeatable.")
    parser.add_argument("--all-channels-as", default=None,
                        choices=["seeg", "ecog", "eeg"], metavar="TYPE",
                        help="with --open and --no-confirm: assume every "
                             "channel is this type. State it; do not let a "
                             "clinical export's own answer stand by default.")
    parser.add_argument("--no-confirm", action="store_true",
                        help="with --open: skip the channel-confirmation "
                             "dialog. For scripted runs only — nothing then "
                             "checks that the channels being analysed are "
                             "intracranial.")
    parser.add_argument("--line-freq", type=float, default=50.0,
                        help="mains frequency for --open (default: %(default)s)")
    parser.add_argument("--span", type=float, default=None, metavar="SECONDS",
                        help="analyse this many seconds from the window's "
                             "start, while the trace shows only --window. "
                             "Above 180 s the analysis is streamed in chunks, "
                             "so ten minutes of contacts can be ranked on a "
                             "laptop. Clicking an event outside the loaded "
                             "window loads the minute it is in.")
    parser.add_argument("--reader", default=None, metavar="NAME",
                        help="who is reviewing. Your verdicts are recorded "
                             "against this name and it goes in the exported "
                             "report; without it the window asks before the "
                             "first verdict.")
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
    parser.add_argument("--layout", choices=["pages", "docks"], default="pages",
                        help="'pages' (default) is a sidebar of pages, the shape of "
                             "the results site; 'docks' is every panel docked on "
                             "the trace with three task layouts. The View menu "
                             "switches between them.")
    parser.add_argument("--theme", choices=["auto", "light", "dark"],
                        default="auto",
                        help="follow the desktop's setting, or force one")
    parser.add_argument("--list", action="store_true",
                        help="print the cached windows and exit")
    parser.add_argument("--export", type=Path, default=None,
                        help="write the review to this path and exit without a "
                             "window (.md or .html); implies --subject")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="save a PNG of the window and exit; for docs and "
                             "for checking a headless install")
    # So `onset-review study-001.edf` works, and so the desktop entry can take
    # a file from a file manager. It is the same thing as `--open`.
    parser.add_argument("file", nargs="?", type=Path, default=None,
                        help="the same as --open: a recording on this machine")
    return parser


def _request_from(args) -> object:
    """Build a request from the flags alone, for `--subject` and `--open`."""
    from onset_review.session import ReviewRequest

    t_start, t_stop = float(args.window[0]), float(args.window[1])
    # `--span` is a length on the command line because that is how someone
    # says "ten minutes"; it is stored as absolute bounds because the trace
    # window moves inside the span and a span measured from the trace would
    # slide with it.
    span = float(args.span) if getattr(args, "span", None) else None
    common = dict(
        run=args.run, task=args.task, t_start=t_start, t_stop=t_stop,
        span_start=t_start if span else None,
        span_stop=(t_start + span) if span else None,
        detectors=tuple(args.detectors or ("rms",)), band=args.band,
        threshold_sd=args.threshold_sd, with_spikes=not args.no_spikes)

    if args.open_path is not None:
        return ReviewRequest(
            dataset="", subject=args.subject or args.open_path.stem,
            path=args.open_path, line_freq=float(args.line_freq),
            channel_types=_channel_types(args), **common)

    return ReviewRequest(dataset=args.dataset, subject=args.subject, **common)


def _channel_types(args) -> tuple[tuple[str, str], ...]:
    """The channel types named on the command line, as a request carries them.

    `--all-channels-as` first, then any `--channel-type NAME=TYPE` on top, so
    "all SEEG except the two EKG leads" is two flags rather than forty.
    """
    from onset_hfo.io import channel_overview

    chosen: dict[str, str] = {}
    if args.all_channels_as:
        chosen = {str(name): args.all_channels_as
                  for name in channel_overview(args.open_path)["name"]}
    for pair in args.channel_types or ():
        name, _, kind = pair.partition("=")
        if not kind:
            raise SystemExit(f"--channel-type wants NAME=TYPE, got {pair!r}")
        chosen[name.strip()] = kind.strip()
    return tuple(sorted(chosen.items()))


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
    if args.file is not None and args.open_path is None:
        args.open_path = args.file
    # A saved project opens in the window, not through the import dialog.
    project_path = case_path = None
    if args.open_path is not None and str(args.open_path).endswith(".onsetproj"):
        project_path, args.open_path = Path(args.open_path), None
    elif args.open_path is not None and (Path(args.open_path) / "case.json").exists():
        case_path, args.open_path = Path(args.open_path), None

    # Set before the loader is imported, so the refusal is in place from the
    # first fetch rather than checked later and hopefully honoured.
    if not args.allow_fetch:
        os.environ.setdefault("ONSET_HFO_OFFLINE", "1")

    if args.list:
        return _print_cached(args.cache_dir)

    unconfirmed = (args.open_path is not None
                   and (args.no_confirm or args.export is not None)
                   and not (args.all_channels_as or args.channel_types))
    if unconfirmed:
        # The one refusal in this entry point. A clinical export declares every
        # channel `eeg` and the pipeline analyses anything typed eeg, so a
        # headless run with nothing said about the channels would produce a
        # full report, with rates and a candidate set, for whatever happened to
        # be in the file. Saying "all SEEG" is one flag; being wrong silently
        # has no flag at all.
        print("--open without the confirmation dialog needs the channel types "
              "stated: --all-channels-as seeg (or ecog), and "
              "--channel-type NAME=TYPE for the exceptions.\n"
              "This file declares nothing reliable: most clinical exports call "
              "every channel scalp EEG, and this software analyses whatever is "
              "typed eeg, ecog or seeg.", file=sys.stderr)
        return EXIT_FAILED

    if args.export is not None:
        if not (args.subject or args.open_path):
            print("--export needs --subject or --open", file=sys.stderr)
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

    if QApplication.instance() is None:
        # View → Interface size, remembered: Qt reads it as the application starts.
        from onset_review.guide import apply_interface_scale

        apply_interface_scale()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from onset_review import applog

    applog.setup()
    applog.install_qt_handler()
    app.setApplicationName("Onset Review")
    app.setApplicationDisplayName("Onset Review")

    from onset_review import theme

    chosen = {"light": theme.LIGHT, "dark": theme.DARK}.get(args.theme)
    theme.apply_theme(app, chosen)

    if args.open_path is not None and not args.no_confirm:
        # The dialog, not the flags, is the default route for a file: the
        # channel types it confirms are the one thing no header can be trusted
        # for, and a flag that skips them has to be typed deliberately.
        from onset_review.importer import ImportDialog

        dialog = ImportDialog(args.open_path)
        _prefill(dialog, args)
        if not (dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()):
            return EXIT_CANCELLED
        request = dialog.request(detectors=tuple(args.detectors or ("rms",)))
        overlay = False
    elif args.subject or args.open_path is not None:
        request, overlay = _request_from(args), args.expert
    elif args.layout == "pages":
        # The window first, the data from inside it. The docked arrangement
        # cannot do this: its window *is* the trace, so it needs a recording
        # before it can exist, and keeps the dialog.
        review = _Review(app, None, False, args)
        review.start()
        if project_path is not None:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda: review.open_project(project_path))
        if case_path is not None:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda: review.open_case(case_path))
        return _finish(app.exec_() if hasattr(app, "exec_") else app.exec())
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
    return _finish(app.exec_() if hasattr(app, "exec_") else app.exec())


def _finish(code: int) -> int:
    """After the window: stop what is still working, and never hang the
    terminal waiting for it (`onset_review.workers.finish`)."""
    from onset_review.workers import finish

    return finish(code)


def _prefill(dialog, args) -> None:
    """Carry the flags into the import dialog, so they are a head start.

    A reviewer who typed `--window 0 30 --all-channels-as seeg` has already
    said what the dialog asks; it should open on their answer and still let
    them look at it, rather than discard it and ask again.
    """
    dialog.t_start.setValue(float(args.window[0]))
    dialog.t_stop.setValue(float(args.window[1]))
    for index in range(dialog.line_freq.count()):
        if float(dialog.line_freq.itemData(index)) == float(args.line_freq):
            dialog.line_freq.setCurrentIndex(index)
    if args.band:
        dialog.band_name = str(args.band)
    if args.subject:
        dialog.subject.setText(args.subject)
    if dialog.table is None:
        return
    if args.all_channels_as:
        if args.all_channels_as == "eeg":
            dialog.set_kind("eeg")      # every channel scalp EEG: a scalp recording
        dialog.set_all(args.all_channels_as)
    for pair in args.channel_types or ():
        name, _, kind = pair.partition("=")
        if not dialog.set_channel(name.strip(), kind.strip()):
            print(f"--channel-type {name.strip()!r}: this file has no such "
                  f"channel", file=sys.stderr)


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
        self.mode = getattr(args, "layout", "pages") or "pages"
        #: The window with nothing open, while it is showing.
        self.start_window = None
        #: The Python console and the script editor, carried from window to
        #: window.
        self.console = None
        self.editor = None

    def start(self) -> None:
        """Open on Home with nothing loaded; everything else comes from there."""
        from onset_review import window

        self.start_window = window.decorate_start(
            cached=self.cached_windows, on_open_cached=self.open_cached,
            on_import=self.import_file, on_choose=self.choose_window,
            on_open_path=self.open_path)
        self._install_handlers(self.start_window)
        self.console = self.start_window.panels.get("console")
        if window.fit_to_screen(self.start_window):
            self.start_window.showMaximized()
        else:
            self.start_window.resize(1280, 820)
            self.start_window.show()

    def _host(self):
        """Whatever window is up, for dialogs to be parented to."""
        if self.parts is not None:
            return self.parts.host
        return self.start_window

    def choose_window(self) -> None:
        """File → Open a recording…: the full dialog, then open its choice."""
        from onset_review import launcher

        request, overlay = launcher.choose_request(self.args.cache_dir,
                                                   parent=self._host())
        if request is None:
            return
        session = launcher.load_with_progress(request, self.args.cache_dir,
                                              parent=self._host())
        if session is None:
            return
        self.request, self.overlay = request, overlay
        self.open(session)

    def open(self, session) -> None:
        from onset_review import window

        previous = self.parts
        # A dock arrangement is worth carrying across a rebuild; a saved
        # dock state means nothing to a page window and the other way round.
        same_docks = (previous is not None and previous.pages is None
                      and self.mode == "docks")
        state = previous.host.saveState() if same_docks else None
        page = (previous.pages.current_page()
                if previous is not None and previous.pages is not None else "")
        if previous is None and self.start_window is not None:
            # From the start window, straight to the trace: Home was only
            # ever the way in.
            page = "recording"
        if getattr(self, "_next_page", None):
            page = self._next_page
        held = self._held_case()
        self._attach_read(session)
        import logging

        logging.getLogger("onset_review").info(
            "opened %s (%d channels, %d events)", session.request.label(),
            len(session.raw.ch_names) if session.raw is not None else 0, len(session.events))

        figure = window.open_trace(session, show_expert=self.overlay,
                                   show=self.args.screenshot is None)
        self.parts = window.decorate(figure, session, show_expert=self.overlay,
                                     on_preprocess=self.reanalyse,
                                     on_import=self.import_file,
                                     on_open_path=self.open_path,
                                     on_quality=self.requality,
                                     on_electrodes=self.use_coordinates,
                                     on_window=self.go_to_window,
                                     on_step_window=self.step_window,
                                     on_trace_at=self.trace_at,
                                     mode=self.mode,
                                     cached=self.cached_windows,
                                     on_open_cached=self.open_cached,
                                     on_relayout=self.relayout,
                                     console=self.console, editor=self.editor)
        # One console for the life of the application: a re-analysis or a
        # newly opened recording rebuilds the window, and the console moves
        # into the new one with everything made in it.
        self.console = self.parts.panels.get("console")
        self.editor = self.parts.panels.get("editor")
        if held is not None:
            # Taken into the new window before the old one closes, so it is
            # never closed with it.
            if self.parts.pages is not None:
                self.parts.pages.hold_case(held, show=False)
            else:
                self._keep_apart(held)
        self._install_handlers(self.parts.host)
        if page and self.parts.pages is not None:
            # The page the reviewer was on, after a re-analysis: a filter
            # applied from the Quality page should leave them on it.
            self.parts.pages.show_page(page)
        maximised = False
        if previous is not None and state is None:
            self.parts.host.resize(previous.host.size())
            maximised = previous.host.isMaximized()
        if previous is None and self.start_window is not None:
            # The same window, as far as the reviewer is concerned: the full
            # one takes the start window's size and place before it shows.
            self.parts.host.resize(self.start_window.size())
            self.parts.host.move(self.start_window.pos())
            maximised = self.start_window.isMaximized()
        if state is not None:
            # Restored after the docks exist and before the window is shown, so
            # the reviewer never sees the default arrangement flash past.
            self.parts.host.restoreState(state)
            self.parts.host.resize(previous.host.size())
            maximised = previous.host.isMaximized()
        # MNE chooses the browser's opening size and chooses it large -- wider
        # and taller than a laptop screen. Clamped to the screen's work area
        # here, and opened maximised when even the clamped size was a
        # reduction, which is what someone on a small screen wants anyway.
        # Not on the screenshot path: that one sets its own size, and a
        # maximised window would ignore it.
        oversized = (self.args.screenshot is None
                     and window.fit_to_screen(self.parts.host))
        if oversized or maximised:
            self.parts.host.showMaximized()
        else:
            self.parts.host.show()
        if previous is not None:
            try:
                previous.figure.close()
            except Exception:
                pass
        if self.start_window is not None:
            self.start_window.close()
            self.start_window = None
        # Said out loud, because the alternative is a reviewer noticing later
        # that some of their verdicts are no longer on the screen and having
        # to guess whether the software lost them. It did not: they are in the
        # file, under the settings they were given under.
        if getattr(self, "_orphans", 0):
            self.parts.host.statusBar().showMessage(
                f"{self._orphans} of your verdicts no longer match an event "
                f"in this analysis. They are kept, and come back if you undo "
                f"the change.", 15000)

    def relayout(self, mode: str) -> None:
        """Rebuild the window in the other arrangement, same analysis."""
        if mode == self.mode or self.parts is None:
            return
        self.mode = mode
        self.open(self.parts.session)

    def cached_windows(self):
        """What is on disk, for the Home page's list."""
        from onset_review.launcher import cached_windows

        return cached_windows(self.args.cache_dir)

    def open_cached(self, row: dict) -> None:
        """Open another cached window from the Home page, replacing this one.

        The analysis settings travel: the same detectors, band and threshold
        as the window being left, so that two patients opened one after the
        other were analysed the same way.
        """
        from onset_review import launcher
        from onset_review.session import ReviewRequest

        # With nothing open yet the defaults apply; otherwise the window being
        # left sets them, so two patients opened in a row are analysed alike.
        base = self.request if self.request is not None else ReviewRequest()
        task = row.get("task")
        request = ReviewRequest(
            dataset=str(row.get("dataset", "ds003498")),
            subject=str(row.get("subject", "sub-01")),
            run=str(row.get("run", "01")),
            task=None if task is None or str(task) in ("", "nan", "—") else str(task),
            t_start=float(row.get("t_start", 0.0)),
            t_stop=float(row.get("t_stop", 60.0)),
            detectors=base.detectors, band=base.band,
            threshold_sd=base.threshold_sd, with_spikes=base.with_spikes,
            preprocess=base.preprocess)
        if request == self.request:
            return
        session = launcher.load_with_progress(request, self.args.cache_dir,
                                              parent=self._host())
        if session is None:
            return
        self.request, self.overlay = request, False
        self.open(session)

    def _attach_read(self, session) -> None:
        """Bring the reader's previous verdicts onto this analysis.

        Every path into a window comes through `open`, including the re-runs
        after a preprocessing or quality change, which is exactly when this
        matters: those rebuild every event object, and without this the
        reviewer's work would appear to vanish because the objects it was
        attached to no longer exist.
        """
        from onset_review import adjudication

        stored = adjudication.load(session.request)
        if not stored.reader:
            # Carried from the window being replaced. A reviewer who named
            # themselves and then moved to the next minute is the same person,
            # and being asked again at every window is how a reader learns to
            # click past the question.
            carried = (self.parts.session.read.reader
                       if self.parts is not None else "")
            stored.reader = (carried or (self.args.reader or "")).strip()
        session.read = adjudication.reconcile(stored, session.events)
        self._orphans = len(session.read.orphaned)

    def trace_at(self, t_file: float) -> None:
        """Put the signal around `t_file` under the analysis already on screen.

        Only the trace moves. The ranking, the events, the quality verdicts
        and the reviewer's own read all belong to the analysed span and would
        be wrong to recompute — and recomputing them is what clicking an event
        would otherwise cost on a ten-minute span.
        """
        from onset_review.session import reload_trace

        session = self.parts.session
        loaded = session.request.t_stop - session.request.t_start
        span_start, span_stop = session.span
        start = min(max(span_start, float(t_file) - loaded / 2.0),
                    max(span_start, span_stop - loaded))
        if abs(start - session.request.t_start) < 1e-6:
            return
        host = self.parts.host
        host.statusBar().showMessage(
            f"Loading {start:g}–{start + loaded:g} s…", 4000)
        try:
            moved = reload_trace(session, start, start + loaded,
                                 self.args.cache_dir)
        except Exception as error:          # noqa: BLE001
            host.statusBar().showMessage(
                f"Could not load {start:g}–{start + loaded:g} s: {error}",
                12000)
            return
        self.request = moved.request
        self.open(moved)

    def go_to_window(self, t_start: float, t_stop: float) -> None:
        """Analyse a different stretch of the same recording.

        The question this software exists to make someone ask is whether the
        answer holds in the next minute -- across these twenty patients the
        annotators' own busiest fast-ripple channel is the same channel in only
        7 of 20 when one minute is compared against another of the same
        recording. Making that cost a trip back through the open dialog is
        making it cost more than it is worth.
        """
        t_start = max(0.0, float(t_start))
        t_stop = float(t_stop)
        if t_stop <= t_start:
            return
        self._rerun(t_start=t_start, t_stop=t_stop)

    def step_window(self, direction: int) -> None:
        """The same window length, one window forward or back."""
        length = self.request.t_stop - self.request.t_start
        start = self.request.t_start + direction * length
        if start < 0:
            start = 0.0
        if start == self.request.t_start:
            return
        self.go_to_window(start, start + length)

    def use_coordinates(self, path) -> None:
        """Remember the coordinate file, without re-running the analysis.

        Recorded on the request rather than only on the session because every
        re-analysis builds a new session from the request -- and a reviewer
        who placed their contacts and then widened a notch should not have to
        find the file again.
        """
        import dataclasses

        self.request = dataclasses.replace(self.request, electrodes_path=path)
        from onset_review.session import laplacian_positions

        positions, _name = laplacian_positions(self.request)
        if positions and self.parts is not None and self._ask_reanalyse_for_positions():
            # The Laplacian's neighbours come from positions: the window on
            # screen was referenced without these, so the analysis follows
            # the file rather than the view alone.
            self._rerun()

    def _ask_reanalyse_for_positions(self) -> bool:
        from qtpy.QtWidgets import QMessageBox

        return QMessageBox.question(
            self._host(), "Re-analyse with these positions?",
            "The Laplacian takes each contact's neighbours from the contacts' positions. "
            "This window was referenced without this file's. Re-analyse now so the "
            "neighbours come from it?") == QMessageBox.Yes

    def import_file(self) -> None:
        """Open a recording from this machine, replacing this window.

        Replacing rather than opening a second one: every panel, the trend, the
        3D layout and the assistant's evidence are built from one session, and
        two windows sharing a process would share nothing else. A reviewer who
        wants both runs the command twice.
        """
        from onset_review.importer import choose_file

        self._open_request(choose_file(self._host(), self.args.cache_dir))

    def open_openneuro(self) -> None:
        """File → Open from OpenNeuro…: list a dataset, download a window of
        one recording, confirm it as an imported file, and open it."""
        from onset_review.openneurodialog import choose_openneuro

        self._open_request(choose_openneuro(self._host()))

    # -- many recordings, and whole projects ------------------------------------------------
    def _install_handlers(self, host) -> None:
        """What the File menu's batch and project entries, and the Files pane,
        call: set on the window rather than passed down every builder."""
        host.on_batch = self.batch
        host.on_openneuro = self.open_openneuro
        host.on_save_project = self.save_project
        host.on_open_project = self.open_project
        host.on_new_case = self.new_case
        host.on_open_case = self.open_case
        host.on_compare_settings = self.compare_settings
        host.on_compare_project = self.compare_project

    def batch(self, paths=None):
        """File → Analyse many recordings: the batch window, analysing alike
        with this window's settings (the defaults when nothing is open)."""
        from onset_review.batchwindow import BatchWindow
        from onset_review.files import current_folder
        from onset_review.session import ReviewRequest

        window = getattr(self, "batch_window", None)
        if window is None or not window.isVisible():
            template = (self.parts.session.request if self.parts is not None
                        else ReviewRequest(detectors=("rms", "line_length")))
            window = BatchWindow(template, cached=self.cached_windows, folder=current_folder())
            window.openRequested.connect(self._open_request)
            self.batch_window = window
        if paths:
            window.add_paths(paths)
        window.show()
        window.raise_()
        return window

    # -- cases: one patient, step by step -------------------------------------------------------
    def new_case(self, case=None):
        """File → New case: a folder and a pseudonym, then the case window."""
        from onset_review.casewindow import new_case_dialog

        case = case or new_case_dialog(self._host())
        return self._show_case(case) if case is not None else None

    def open_case(self, folder=None):
        """File → Open case (or a case folder given on the command line)."""
        from onset_hfo.case.model import Case
        from onset_review.casewindow import open_case_dialog

        case = Case.open(folder) if folder is not None else open_case_dialog(self._host())
        return self._show_case(case) if case is not None else None

    def _show_case(self, case):
        """The case on Patient → Case, in this window; in the docked
        arrangement, which has no pages, in a window of its own."""
        from onset_review.casewindow import CaseWindow

        window = CaseWindow(case)
        window.openRequested.connect(self._open_from_case)
        pages = self.parts.pages if self.parts is not None else self.start_window
        if pages is not None and hasattr(pages, "hold_case"):
            pages.hold_case(window)
            return window
        self._keep_apart(window)
        return window

    def _keep_apart(self, window) -> None:
        from qtpy.QtCore import Qt

        window.setParent(None)
        window.setWindowFlags(Qt.Window)
        window.statusBar().show()
        self.case_windows = [w for w in getattr(self, "case_windows", [])
                             if w.isVisible() and w is not window] + [window]
        window.show()

    def _held_case(self):
        """The case the window being replaced holds, to carry into the next."""
        host = (self.parts.pages if self.parts is not None else self.start_window)
        held = getattr(host, "held_case", None)
        if held is None and self.mode == "pages":
            # From the docked arrangement back to pages: its case window.
            apart = [w for w in getattr(self, "case_windows", []) if w.isVisible()]
            held = apart[-1] if apart else None
        return held

    def _open_from_case(self, request) -> None:
        """A recording opened from the case goes to Review; the case stays
        on Patient → Case."""
        self._next_page = "recording"
        try:
            self._open_request(request)
        finally:
            self._next_page = None

    # -- two analyses side by side -----------------------------------------------------------
    def compare_settings(self, request=None):
        """File → Compare with → other settings: this recording analysed again
        as asked, beside the window."""
        from qtpy.QtWidgets import QDialog

        from onset_review import launcher
        from onset_review.comparewindow import OtherSettingsDialog

        if self.parts is None:
            return None
        if request is None:
            dialog = OtherSettingsDialog(self.parts.session.request, self._host())
            if dialog.exec() != QDialog.Accepted:
                return None
            request = dialog.other()
        other = launcher.load_with_progress(request, self.args.cache_dir,
                                            parent=self._host())
        if other is None:
            return None
        return self._show_comparison(other)

    def compare_project(self, path=None):
        """File → Compare with → a saved project: its analysis, run again from
        its request, with its own read, beside the window."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        from onset_review import launcher, project
        from onset_review.files import current_folder

        if self.parts is None:
            return None
        host = self._host()
        if path is None:
            path, _ = QFileDialog.getOpenFileName(host, "Compare with a project",
                                                  str(current_folder()),
                                                  f"Projects (*{project.SUFFIX})")
            if not path:
                return None
        state = project.open_project(path, locate=self._locate_recording)
        if state.request is None:
            QMessageBox.warning(host, "Compare with a project",
                                "The project's recording was not found: "
                                + "; ".join(state.missing or ["nothing to analyse"]) + ".")
            return None
        other = launcher.load_with_progress(state.request, self.args.cache_dir, parent=host)
        if other is None:
            return None
        if state.read is not None:
            other.read = state.read
        return self._show_comparison(other, b_label="B")

    def _locate_recording(self, entry):
        """Ask where a project's recording is, when it is not where it was."""
        from qtpy.QtWidgets import QFileDialog

        from onset_review.files import current_folder

        chosen, _ = QFileDialog.getOpenFileName(
            self._host(), f"Where is {entry.get('name')}? (it was at {entry.get('path')})",
            str(current_folder()))
        return chosen or None

    def _show_comparison(self, other, b_label: str = "B"):
        from onset_review.comparewindow import CompareWindow

        window = CompareWindow(self.parts.session, other, "A", b_label)
        window.openRequested.connect(self._open_request)
        self.compare_windows = [w for w in getattr(self, "compare_windows", [])
                                if w.isVisible()] + [window]
        window.show()
        return window

    def save_project(self, path=None, include_recording=None):
        """File → Save project: this recording and everything done with it."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        from onset_review import project
        from onset_review.files import current_folder

        host = self._host()
        if self.parts is None:
            QMessageBox.information(host, "Save project", "Open a recording first: a project "
                                    "is a recording and what was done with it.")
            return None
        session = self.parts.session
        request = session.request
        if include_recording is None:
            include_recording = False
            if request.path is not None and Path(request.path).is_file():
                size = Path(request.path).stat().st_size / 1e6
                answer = QMessageBox.question(
                    host, "Include the recording?",
                    f"Put {Path(request.path).name} ({size:,.0f} MB) inside the project, so "
                    "it opens on another machine as it is? Without it the project keeps the "
                    "file's location and a checksum, and asks for the file when it is not "
                    "there.", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No)
                if answer == QMessageBox.Cancel:
                    return None
                include_recording = answer == QMessageBox.Yes
        if path is None:
            suggested = current_folder() / f"{request.subject or 'project'}{project.SUFFIX}"
            path, _ = QFileDialog.getSaveFileName(host, "Save the project", str(suggested),
                                                  f"Onset projects (*{project.SUFFIX})")
            if not path:
                return None
        tabs = ([{"title": e.title, "path": str(e.path) if e.path else None,
                  "text": e.toPlainText()} for e in self.editor.editors()]
                if self.editor is not None else [])
        variables = (self.console.user_variables()
                     if self.console is not None and self.console.started else {})
        page = self.parts.pages.current_page() if self.parts.pages is not None else ""
        manifest = project.save_project(path, request=request, read=session.read, page=page,
                                        editor_tabs=tabs, console_log=list(session.console_log),
                                        variables=variables,
                                        include_recording=bool(include_recording))
        import logging

        logging.getLogger("onset_review").info("project saved: %s", manifest["path"])
        left = manifest.get("variables_left_out") or {}
        if host is not None and host.statusBar() is not None:
            host.statusBar().showMessage(
                f"Saved {Path(manifest['path']).name}"
                + (f"; left out {', '.join(left)}" if left else ""), 10000)
        return manifest

    def open_project(self, path=None, ask: bool = True):
        """File → Open project: unpack it, put back the verdicts, the cohort,
        the scripts and the variables, and analyse the recording again from
        its request."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        from onset_review import project
        from onset_review.files import current_folder

        host = self._host()
        if path is None:
            path, _ = QFileDialog.getOpenFileName(host, "Open a project", str(current_folder()),
                                                  f"Onset projects (*{project.SUFFIX})")
            if not path:
                return None

        locate = self._locate_recording if ask else (lambda _entry: None)
        try:
            state = project.open_project(path, locate=locate)
        except Exception as error:      # noqa: BLE001 - a broken file is reported
            QMessageBox.warning(host, "Open project", f"Could not open {Path(path).name}: "
                                f"{error}")
            return None
        if state.request is None:
            QMessageBox.warning(host, "Open project", "This project's recording could not be "
                                "found: " + "; ".join(state.missing or ["no request in it"]))
            return state
        if state.read is not None:
            project.restore_read(state.request, state.read)
        if state.manifest.get("study"):
            replace = True
            if ask:
                replace = QMessageBox.question(
                    host, "Your cohort and re-runs",
                    "The project carries a cohort and re-runs of the study. Use them in place "
                    "of this machine's? This machine's are kept in study/before-… either way.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes
            if replace:
                project.restore_study(state)
        self._open_request(state.request)
        if self.parts is None or self.parts.session.request != state.request:
            return state
        editor = self.parts.panels.get("editor")
        if editor is not None:
            for tab in state.editor_tabs:
                where = Path(tab["path"]) if tab.get("path") else None
                if where is not None and where.is_file() and \
                        where.read_text(encoding="utf-8", errors="replace") == tab.get("text"):
                    editor.open_file(where)
                else:
                    editor.new_file(str(tab.get("text", "")),
                                    title=tab.get("title") or (where.name if where else None))
            if state.console_log:
                editor.new_file("# Commands run in the console when the project was saved.\n"
                                "# They were not run in this session: run a cell to run it.\n\n"
                                + "\n\n# %%\n".join(state.console_log) + "\n",
                                title="Console history")
        console = self.parts.panels.get("console")
        if state.variables_file is not None and console is not None:
            load = True
            if ask:
                load = QMessageBox.question(
                    host, "Console variables",
                    f"Load the {len(state.manifest.get('variables') or [])} console variable(s) "
                    "saved in the project? They are a pickle, which can run code as it is "
                    "read: load them only from a project you trust.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
            if load:
                from onset_review.variables import load_workspace

                console.start()
                console.namespace.update(load_workspace(state.variables_file))
                workspace = self.parts.panels.get("workspace")
                if workspace is not None:
                    workspace.refresh()
        if state.page and self.parts.pages is not None:
            self.parts.pages.show_page(state.page)
        import logging

        logging.getLogger("onset_review").info("project opened: %s", path)
        if state.notes and ask:
            QMessageBox.information(host, "Project opened", "\n\n".join(state.notes))
        return state

    def open_path(self, path) -> None:
        """Open one file named by the Files pane: the same confirmation
        dialog as File → Open a file, without the file chooser before it."""
        from onset_review.importer import ImportDialog

        dialog = ImportDialog(path, self._host())
        accepted = dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()
        self._open_request(dialog.request() if accepted else None)

    def _open_request(self, request) -> None:
        from onset_review import launcher

        if request is None:
            return
        session = launcher.load_with_progress(request, self.args.cache_dir,
                                              parent=self._host())
        if session is None:
            return
        self.request, self.overlay = request, False
        self.open(session)

    def requality(self, check: bool, keep) -> None:
        """Re-run this window with the quality stage on, off, or overruled.

        A separate entry point from `reanalyse` rather than a flag on it,
        because the two changes are different claims. Changing a filter
        changes what the signal *is*; changing this changes which of it was
        worth analysing, and a report has to be able to say which happened.
        """
        self._rerun(check_quality=bool(check), keep_channels=tuple(keep))

    def reanalyse(self, preprocess, band: str | None = None) -> None:
        """Re-run this window under new preprocessing, and replace the view.

        A failure leaves the current window exactly as it was. `load_with_progress`
        has already told the reviewer what went wrong, and the alternative --
        closing a working window because a setting was rejected -- would lose
        them their place for no reason.
        """
        changes = {"preprocess": preprocess}
        if band:
            changes["band"] = band          # chosen on the Signal page
        self._rerun(**changes)

    def _rerun(self, **changes) -> None:
        """Replace the window with the same slice analysed differently."""
        import dataclasses

        from onset_review import launcher

        request = dataclasses.replace(self.request, **changes)
        session = launcher.load_with_progress(request, self.args.cache_dir,
                                              parent=self.parts.host)
        if session is None:
            # The loader has already said what went wrong in its own dialog.
            # This is the sentence that says what it was trying to do, because
            # "could not fetch" without "the minute you asked for" leaves a
            # reviewer wondering what they just lost.
            self.parts.host.statusBar().showMessage(
                f"Still showing {self.request.t_start:g}–"
                f"{self.request.t_stop:g} s: the window you asked for could "
                f"not be loaded.", 12000)
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

    # The documentation size, or the one asked for: ONSET_SCREENSHOT_SIZE=1366x768
    # renders the laptop the clinic actually has.
    width, height = 1680, 980
    asked = os.environ.get("ONSET_SCREENSHOT_SIZE", "")
    if re.fullmatch(r"\d{3,4}x\d{3,4}", asked):
        width, height = (int(part) for part in asked.split("x"))
    parts.host.resize(width, height)
    for _ in range(12):      # let the trace's own load thread settle
        app.processEvents()
        QThread.msleep(60)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages = parts.pages
    if pages is not None:
        # One file per page beside the one asked for, which gets the
        # Recording page: the trace is what a screenshot is for.
        pages.show_page("recording")
        for _ in range(4):
            app.processEvents()
            QThread.msleep(60)
    ok = parts.host.grab().save(str(path))
    if pages is not None and ok:
        # The Analysis page turns Analysis mode on; every other page is taken
        # as the reader has the window, and the mode is left as it was.
        mode = pages.analysis_mode
        for key in pages.page_keys():
            pages.set_analysis_mode(mode or key == "analysis", remember=False)
            pages.show_page(key)
            for _ in range(4):
                app.processEvents()
                QThread.msleep(60)
            extra = path.with_name(f"{path.stem}-{key}{path.suffix}")
            if parts.host.grab().save(str(extra)):
                print(extra)
        pages.set_analysis_mode(mode, remember=False)
        pages.show_page("recording")

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
