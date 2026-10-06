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

    app = QApplication.instance() or QApplication(sys.argv[:1])
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
        return app.exec_() if hasattr(app, "exec_") else app.exec()
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
    for index in range(dialog.band.count()):
        if dialog.band.itemData(index) == args.band:
            dialog.band.setCurrentIndex(index)
    if args.subject:
        dialog.subject.setText(args.subject)
    if dialog.table is None:
        return
    if args.all_channels_as:
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

    def start(self) -> None:
        """Open on Home with nothing loaded; everything else comes from there."""
        from onset_review import window

        self.start_window = window.decorate_start(
            cached=self.cached_windows, on_open_cached=self.open_cached,
            on_import=self.import_file, on_choose=self.choose_window)
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
        self._attach_read(session)

        figure = window.open_trace(session, show_expert=self.overlay,
                                   show=self.args.screenshot is None)
        self.parts = window.decorate(figure, session, show_expert=self.overlay,
                                     on_preprocess=self.reanalyse,
                                     on_import=self.import_file,
                                     on_quality=self.requality,
                                     on_electrodes=self.use_coordinates,
                                     on_window=self.go_to_window,
                                     on_step_window=self.step_window,
                                     on_trace_at=self.trace_at,
                                     mode=self.mode,
                                     cached=self.cached_windows,
                                     on_open_cached=self.open_cached,
                                     on_relayout=self.relayout)
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

    def import_file(self) -> None:
        """Open a recording from this machine, replacing this window.

        Replacing rather than opening a second one: every panel, the trend, the
        3D layout and the assistant's evidence are built from one session, and
        two windows sharing a process would share nothing else. A reviewer who
        wants both runs the command twice.
        """
        from onset_review import launcher
        from onset_review.importer import choose_file

        request = choose_file(self._host(), self.args.cache_dir)
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

    def reanalyse(self, preprocess) -> None:
        """Re-run this window under new preprocessing, and replace the view.

        A failure leaves the current window exactly as it was. `load_with_progress`
        has already told the reviewer what went wrong, and the alternative --
        closing a working window because a setting was rejected -- would lose
        them their place for no reason.
        """
        self._rerun(preprocess=preprocess)

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

    parts.host.resize(1680, 980)
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
        for key in pages.page_keys():
            pages.show_page(key)
            for _ in range(4):
                app.processEvents()
                QThread.msleep(60)
            extra = path.with_name(f"{path.stem}-{key}{path.suffix}")
            if parts.host.grab().save(str(extra)):
                print(extra)
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
