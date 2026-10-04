"""The review window: MNE's trace, with clinical panels docked around it.

The central decision of this product is that the signal viewer is not ours.
`mne-qt-browser` is a mature, fast, keyboard-driven iEEG viewer that
neurophysiologists already know -- it scrolls, it rescales, it butterflies, it
lets a reviewer drag an annotation -- and reimplementing it would produce
something worse that also had to be maintained. It also happens to be a
`QMainWindow`, which means its own window can host dock widgets.

So `decorate` takes the figure MNE hands back and adds to it: a menu bar, a
navigation toolbar, the five panels of `onset_review.panels`, and a status bar
that permanently carries the one sentence a ranked channel list must never be
read without. The result is one window, with one trace in it, and no second
copy of the signal anywhere.

The fallback matters as much as the main path. If a future `mne-qt-browser`
stops being a `QMainWindow`, `decorate` puts the panels in a window of their own
and says so rather than crashing, and `has_dock_host` is what the test suite
checks so the day that happens is a failing test and not a bug report from a
hospital.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QActionGroup
from qtpy.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from onset_review import adjudication, report, theme
from onset_review.assistant import AssistantPanel
from onset_review.brainview import BrainPanel
from onset_review.controls import AMPLITUDE_STEP, TraceControls
from onset_review.dataquality import QualityPanel
from onset_review.panels import (
    AgreementPanel,
    EventsPanel,
    FindingsPanel,
    ProvenancePanel,
    TrendsPanel,
)
from onset_review.patient import PatientPanel
from onset_review.preprocessing import PreprocessPanel
from onset_review.session import BAND_COLOURS, ReviewSession, annotations_for

__all__ = ["decorate", "open_trace", "has_dock_host", "marks_for",
           "fit_to_screen", "work_area",
           "ReviewWindowParts", "MARK_SCOPES"]

#: How much of the detection gets drawn on the trace. This is the single most
#: consequential display choice in the product. A 60 s window of ds003498 holds
#: around five hundred detections and seven hundred expert marks, and MNE draws
#: an annotation as a full-height span across every channel -- so marking
#: everything turns the trace into a barcode and hides the signal it is there
#: to show. Restricting the marks to the channel under review is both readable
#: and the way the work is actually done: pick a channel, look at it, move on.
MARK_SCOPES = {
    "selected": "Marks on the selected channel",
    "all": "Marks on every channel",
    "none": "No marks",
}

#: Seconds of signal on screen when a window opens. Ten is enough context to
#: see whether a marked ripple sits on a discharge and short enough that an
#: 80 Hz oscillation is still a visible oscillation rather than a thickening of
#: the line -- which is the whole reason a reviewer is looking at the trace
#: instead of the rate.
DEFAULT_DURATION = 10.0
#: Channels on screen. More than this and the traces are too compressed to
#: judge an event by eye on a laptop display.
DEFAULT_N_CHANNELS = 12
#: Volts per display unit. 50 µV is the scale the HFO literature plots filtered
#: ripples at; the reviewer rescales with the usual MNE keys from there.
DEFAULT_SCALING = 50e-6

#: Window-wide shortcuts for the reader's verdicts, as `Ctrl`+digit. The
#: events panel also binds the bare letters A/D/U while it has focus; these
#: are the discoverable half of the same pair, and digits rather than letters
#: because `Ctrl+A` is select-all everywhere in the world.
EVENT_SHORTCUTS = {"agree": "1", "disagree": "2", "unsure": "3"}

#: The opening layout, as a fraction of the screen's work area with pixel
#: bounds: the right-hand column of panels, then the trend strip above the
#: trace. Fractions rather than fixed pixels because the fixed pixels were
#: chosen against a 1680-wide window, and on a 1366x768 laptop they gave a
#: 640 px column of panels beside a 700 px trace -- the wrong way round, since
#: the trace is the thing being judged and the panels only describe it.
COLUMN_FRACTION, COLUMN_BOUNDS = 0.34, (380, 640)
TREND_FRACTION, TREND_BOUNDS = 0.22, (150, 260)

#: Vertical shares of the right-hand column: the findings table, the events
#: table, and the tab stack under them. Qt scales them to whatever height the
#: column has, so they are a ratio as much as a size.
COLUMN_SPLIT = (300, 240, 420)


def work_area(widget=None):
    """The usable rectangle of the screen `widget` is on, or None if headless.

    `availableGeometry` rather than `geometry`: the taskbar, dock or top panel
    is exactly the strip a maximised window may not have, and a window sized
    to the full screen height opens with its status bar -- the one carrying the
    not-a-medical-device caveat -- hidden underneath it.
    """
    from qtpy.QtGui import QGuiApplication

    screen = None
    if widget is not None:
        try:
            screen = widget.screen()
        except AttributeError:      # Qt < 5.14 has no QWidget.screen()
            screen = None
    screen = screen or QGuiApplication.primaryScreen()
    return None if screen is None else screen.availableGeometry()


def fit_to_screen(host) -> bool:
    """Bring `host` inside the screen's work area, and say if it did not fit.

    A True return is the caller's cue to open the window maximised instead of
    at this size.

    MNE's browser picks its own opening size, around 1680x1160. That is most
    of a 1920x1200 desktop and larger than a laptop screen in both directions,
    so the window opened with its right-hand edge and its status bar off the
    screen. It also opened *without a maximise button*, which is the same bug
    seen from the window manager's side: a window whose minimum size does not
    fit the work area cannot be maximised into it, so the button is withheld.
    Hence both halves of the fix -- the panels' minimums came down (see
    `theme.scrolled` and `panels.MIN_TABLE_HEIGHT`) so that the window *can*
    fit, and this clamps the opening size so that it *does*.
    """
    area = work_area(host)
    if area is None:
        return False
    want = host.size().expandedTo(host.minimumSizeHint())
    over = want.width() > area.width() or want.height() > area.height()
    host.resize(min(want.width(), area.width()),
                min(want.height(), area.height()))
    # Centred on what is left rather than left where it was: a window that was
    # just made smaller is otherwise still anchored to a corner off the screen.
    size = host.size()
    host.move(area.x() + max(0, area.width() - size.width()) // 2,
              area.y() + max(0, area.height() - size.height()) // 2)
    return over


def _share(extent: int, fraction: float, bounds: tuple) -> int:
    low, high = bounds
    return max(low, min(high, int(extent * fraction)))


class ReviewWindowParts:
    """Handles on everything `decorate` created, for callers and for tests.

    Returned rather than stashed on the figure, because monkey-patching
    attributes onto someone else's widget is how two libraries end up fighting
    over the same name.
    """

    def __init__(self, figure, host: QMainWindow, panels: dict,
                 docks: dict, session: ReviewSession):
        self.figure = figure
        self.host = host
        self.panels = panels
        self.docks = docks
        self.session = session
        #: False when the panels had to go into their own window.
        self.docked = host is figure
        #: Set by `decorate`; what is currently drawn over the signal.
        self.display: _Display | None = None

    def refresh_marks(self) -> None:
        """Redraw the trace's marks for the current scope and channel."""
        _apply_marks(self.figure, self.session, self.display.scope,
                     self.display.channel, self.display.expert)


class _Display:
    """What is currently drawn over the signal.

    Three settings that every panel can change and that all have to agree, so
    they live in one object rather than being read back off the widgets that
    happen to hold them.
    """

    def __init__(self, scope: str, channel: str | None, expert: bool):
        self.scope = scope
        self.channel = channel
        self.expert = expert


def has_dock_host(figure) -> bool:
    """Whether MNE's figure can host our docks, i.e. is still a QMainWindow."""
    return isinstance(figure, QMainWindow)


def open_trace(session: ReviewSession, duration: float = DEFAULT_DURATION,
               n_channels: int = DEFAULT_N_CHANNELS,
               show_expert: bool = False, show: bool = True):
    """Open the session's signal in MNE's Qt browser, annotations and all.

    The scaling, window length and channel count are set here rather than left
    to MNE's defaults because a reviewer opening a patient should see something
    readable immediately; every one of them is adjustable from the keyboard
    afterwards, which is why they are defaults and not settings.
    """
    import matplotlib
    import mne

    matplotlib.use("Agg", force=False)  # keep MNE's own mpl figures off-screen
    mne.viz.set_browser_backend("qt")

    session.raw.set_annotations(
        marks_for(session, scope="selected",
                  channel=session.leader.get("leader"), expert=show_expert))

    figure = session.raw.plot(
        duration=duration, n_channels=min(n_channels, len(session.findings) or 1),
        scalings=dict(seeg=DEFAULT_SCALING, eeg=DEFAULT_SCALING,
                      ecog=DEFAULT_SCALING),
        title=session.request.label(), block=False, show=show, verbose="ERROR")
    _colour_annotations(figure)
    return figure


def marks_for(session: ReviewSession, scope: str = "selected",
              channel: str | None = None, expert: bool = False):
    """The annotations to put on the trace, given what is being looked at.

    Kept separate from `session.annotations_for` because this is a display
    decision and that is a data one: the session always holds every event, and
    this chooses which of them the reviewer currently wants drawn over the
    signal.
    """
    events = list(session.events)
    extra = list(session.expert) if (expert and session.has_expert) else []
    if scope == "none":
        events, extra = [], []
    elif scope == "selected" and channel:
        events = [e for e in events if e.channel == channel]
        extra = [e for e in extra if e.channel == channel]
    annotations = annotations_for(events, session.t_offset)
    if extra:
        annotations = annotations + annotations_for(
            extra, session.t_offset, accepted_only=False, prefix="expert ")
    return annotations


def _colour_annotations(figure) -> None:
    """Give each annotation description its project colour.

    MNE keys annotation colour off the description string, which is exactly why
    `session._label_for` names events by band rather than by detector: the two
    HFO bands have to be distinguishable on the trace, and the expert overlay
    has to be distinguishable from both.
    """
    state = getattr(figure, "mne", None)
    if state is None:
        return
    # MNE assigns a colour to any description it has not seen, so ours are
    # written *after* its own setup runs rather than before, or the first
    # redraw silently replaces them with a rotating palette.
    try:
        figure._setup_annotation_colors()
    except Exception:
        pass
    store = getattr(state, "annotation_segment_colors", None)
    if store is None:
        return
    for description, colour in BAND_COLOURS.items():
        store[description] = colour
    # Each region re-reads the dict, so existing ones have to be told to; a
    # region created before our colours were written keeps MNE's until asked.
    for region in getattr(state, "regions", []):
        if getattr(region, "description", None) in BAND_COLOURS:
            try:
                region.update_color()
            except Exception:
                pass


def goto(figure, t: float, channel: str | None = None,
         session: ReviewSession | None = None) -> None:
    """Scroll the trace so `t` is on screen, and `channel` with it.

    Centring rather than left-aligning: a reviewer clicking an event wants to
    see what led into it as much as the event itself, and an event pinned to the
    left edge of the screen has no history.

    Written against `figure.mne` and the plot's own range because
    `mne-qt-browser` exposes no public "go here" call. Both touches are guarded:
    failing to scroll must never take the window down, and a reviewer who sees
    the trace not move will scroll it themselves.
    """
    state = getattr(figure, "mne", None)
    if state is None:
        return
    duration = float(getattr(state, "duration", DEFAULT_DURATION))
    xmax = float(getattr(state, "xmax", t + duration))
    start = min(max(0.0, float(t) - duration / 2.0), max(0.0, xmax - duration))
    try:
        figure.mne.plt.setXRange(start, start + duration, padding=0.0)
    except Exception:
        return

    if channel and session is not None:
        names = list(getattr(state, "ch_names", []))
        if channel in names:
            row = names.index(channel)
            on_screen = int(getattr(state, "n_channels", DEFAULT_N_CHANNELS))
            top = min(max(0, row - on_screen // 2),
                      max(0, len(names) - on_screen))
            try:
                # `+ 1` because that is MNE's own convention, and getting it
                # wrong is not a cosmetic error: the browser recomputes
                # `n_channels` as `round(y1 - y0 - 1)` whenever the range
                # changes, so a range of exactly `on_screen` tells it to show
                # one channel fewer. Every click on an event cost a channel,
                # and after a dozen the trace was empty and MNE raised inside
                # its own redraw. Showing N channels spans N + 1 units.
                figure.mne.plt.setYRange(top, top + on_screen + 1, padding=0.0)
            except Exception:
                pass


def decorate(figure, session: ReviewSession, show_expert: bool = False,
             on_preprocess=None, on_import=None,
             on_quality=None) -> ReviewWindowParts:
    """Add the menus, the toolbar, the panels and the caveat to MNE's window.

    `on_preprocess` is called with a new `PreprocessConfig` when the reviewer
    applies one. `on_quality` is called with (check_quality, keep_channels)
    when they change which contacts are analysed. `on_import` is called with
    no arguments when they ask to open a recording from this machine. All
    three are callbacks rather than something
    this module does itself, because re-running the analysis means fetching,
    detecting and rebuilding every panel -- which is the entry point's job, and
    keeps this file free of the loader and the progress dialog.
    """
    host = figure if has_dock_host(figure) else QMainWindow()
    display = _Display("selected", session.leader.get("leader"), show_expert)
    host.setWindowTitle(f"Onset Review — {session.request.label()}")

    panels = {
        "trends": TrendsPanel(session),
        "controls": TraceControls(figure),
        "findings": FindingsPanel(session),
        "events": EventsPanel(session),
        "patient": PatientPanel(session),
        "brain": BrainPanel(session, resection=session.resection,
                            electrodes=session.electrodes),
        "assistant": AssistantPanel(session),
        "preprocess": PreprocessPanel(session),
        "quality": QualityPanel(session),
        "agreement": AgreementPanel(session),
        "provenance": ProvenancePanel(session),
    }
    docks = {}

    def dock(key: str, title: str, area, widget: QWidget,
             tip: str = "") -> QDockWidget:
        # Short titles: Qt puts the window title on the tab, and six tabbed
        # docks in a 640 px column elide into "P...", "Where the contac...",
        # "Detector vs ...". The sentence goes in the tooltip instead.
        item = QDockWidget(title, host)
        item.setObjectName(f"dock_{key}")
        if tip:
            item.setToolTip(tip)
        item.setWidget(widget)
        item.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
                             | Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        host.addDockWidget(area, item)
        docks[key] = item
        return item

    # The trend goes above the trace because that is the order it is read in,
    # and the trace controls go directly under it, against the trace they drive.
    dock("trends", "Trend — rate per channel over time",
         Qt.TopDockWidgetArea, panels["trends"])
    controls = dock("controls", "Trace controls", Qt.TopDockWidgetArea,
                    panels["controls"])
    controls.setFeatures(QDockWidget.DockWidgetMovable
                         | QDockWidget.DockWidgetFloatable)
    # Split vertically, or Qt puts two docks in the same area side by side and
    # the trend loses half its width to a bar that wants one row.
    host.splitDockWidget(docks["trends"], controls, Qt.Vertical)
    dock("findings", "Findings — channels ranked by rate",
         Qt.RightDockWidgetArea, panels["findings"])
    dock("events", "Events", Qt.RightDockWidgetArea, panels["events"])
    # Agreement and provenance are reference rather than working views, so they
    # share a tab stack and start behind the panels a reviewer uses minute to
    # minute.
    # The 3D view, the agreement table and the provenance share one tab stack
    # under the tables. All three answer "what am I actually looking at" rather
    # than being read minute to minute, and any of them in a column of its own
    # takes its width from the trace -- which is the one thing on screen that
    # has to stay wide enough to judge an oscillation on. A reviewer who wants
    # the 3D view big floats it out of the dock, which is one drag.
    who = dock("patient", "Patient", Qt.RightDockWidgetArea, panels["patient"],
               "Who this recording belongs to, as far as the archive says")
    brain = dock("brain", "Contacts", Qt.RightDockWidgetArea, panels["brain"],
                 "Where the contacts are, ranked and relative to the resection")
    accord = dock("agreement", "Agreement", Qt.RightDockWidgetArea,
                  panels["agreement"],
                  "This detector against the archive's own annotators")
    prov = dock("provenance", "Provenance", Qt.RightDockWidgetArea,
                panels["provenance"], "How this was produced, step by step")
    helper = dock("assistant", "Assistant", Qt.RightDockWidgetArea,
                  panels["assistant"],
                  "Ask about this window; answers cite it or refuse")
    # Preprocessing sits with provenance rather than with the working views:
    # it is the other half of the same question. "How this was produced" says
    # what was done; the tab next to it is where a reviewer changes it.
    prep = dock("preprocess", "Preprocessing", Qt.RightDockWidgetArea,
                panels["preprocess"],
                "What is done to the signal before any detector sees it")
    # And beside it, the stage that decides what the preprocessed signal is
    # fit for. The two belong together: one says what was done to the signal,
    # the other says which of it was worth analysing.
    fit = dock("quality", "Quality", Qt.RightDockWidgetArea,
               panels["quality"],
               "Data quality — which contacts and which seconds were "
               "analysed, which were only flagged, and why")
    host.tabifyDockWidget(who, brain)
    host.tabifyDockWidget(brain, accord)
    host.tabifyDockWidget(accord, prov)
    host.tabifyDockWidget(prov, prep)
    host.tabifyDockWidget(prep, fit)
    host.tabifyDockWidget(fit, helper)
    brain.raise_()
    # A trend squeezed to a strip is unreadable, and Qt will squeeze it unless
    # the widget itself says otherwise; `resizeDocks` alone loses to the
    # central widget's own size policy.
    panels["trends"].setMinimumHeight(100)
    panels["controls"].setFixedHeight(panels["controls"].sizeHint().height())
    area = work_area(host)
    width = area.width() if area is not None else 1680
    height = area.height() if area is not None else 1050
    host.resizeDocks([docks["trends"]],
                     [_share(height, TREND_FRACTION, TREND_BOUNDS)],
                     Qt.Vertical)
    column = [docks["findings"], docks["events"], brain, helper, prep, fit, who]
    host.resizeDocks(column,
                     [_share(width, COLUMN_FRACTION, COLUMN_BOUNDS)]
                     * len(column), Qt.Horizontal)
    # Vertical shares for the right-hand column. Without these the 3D view's
    # own minimum height wins the whole column and the two tables above it are
    # left showing one row each.
    host.resizeDocks([docks["findings"], docks["events"], brain],
                     list(COLUMN_SPLIT), Qt.Vertical)

    parts = ReviewWindowParts(figure, host, panels, docks, session)
    parts.display = display
    _wire(figure, host, panels, session, display, parts)
    _status(host, session)
    _menus(figure, host, panels, docks, session, display, parts,
           on_import=on_import)
    _set_reader_status(host, session)
    if on_quality is not None:
        panels["quality"].applied.connect(on_quality)
    else:
        panels["quality"].apply.setEnabled(False)
        panels["quality"].apply.setToolTip(
            "Re-analysis is not available in this window")
    if on_preprocess is not None:
        panels["preprocess"].applied.connect(on_preprocess)
    else:
        panels["preprocess"].apply.setEnabled(False)
        panels["preprocess"].apply.setToolTip(
            "Re-analysis is not available in this window")

    _apply_marks(figure, session, display.scope, display.channel, display.expert)
    # Open on the channel whose marks are being shown. Landing on channel one
    # of forty-three while the marks belong to the busiest one is how a
    # reviewer concludes the overlay is broken.
    if display.channel:
        _goto_channel(figure, session, display.channel)
        panels["findings"].select_channel(display.channel)
    return parts


def _wire(figure, host, panels: dict, session: ReviewSession,
          display: _Display, parts: ReviewWindowParts) -> None:
    """Make every panel's selection move the one trace, and the marks with it.

    The second half is what makes the default mark scope workable: choosing a
    channel anywhere re-marks the trace for that channel, so "selected channel
    only" never means "the channel I picked three clicks ago".
    """
    def select(channel: str, t: float | None = None) -> None:
        if channel:
            display.channel = channel
            parts.refresh_marks()
        if t is None:
            _goto_channel(figure, session, channel)
        else:
            goto(figure, t, channel, session)
        # The control bar reads its values out of the browser rather than
        # keeping its own, so anything that moves the view has to tell it to
        # look again -- otherwise its "At" box says where the reviewer was
        # before they clicked.
        panels["controls"].sync()

    panels["events"].eventPicked.connect(lambda t, channel: select(channel, t))
    panels["trends"].cellPicked.connect(lambda t, channel: select(channel, t))
    panels["findings"].channelPicked.connect(lambda channel: select(channel))
    panels["agreement"].channelPicked.connect(lambda channel: select(channel))
    panels["brain"].channelPicked.connect(lambda channel: select(channel))
    # A citation names a time in the archive's seconds, which is what a report
    # quotes; the trace runs from zero, so the offset comes off here.
    panels["assistant"].evidencePicked.connect(
        lambda channel, t_file: select(channel, float(t_file) - session.t_offset))
    # ...and the 3D view turns to face whatever was chosen elsewhere, so the
    # three views never disagree about which contact is under discussion.
    panels["findings"].channelPicked.connect(panels["brain"].highlight)
    # Picking a channel in the agreement table should move the findings table
    # with it, so the two never disagree about what is selected.
    panels["agreement"].channelPicked.connect(panels["findings"].select_channel)

    # The reader's verdicts. There is no save button: a reviewer who has
    # worked through three hundred events and lost them to a crash will not
    # use this software again, and the write is a few kilobytes of JSON.
    def keep_read(*_) -> None:
        try:
            adjudication.save(session.request, session.read)
        except OSError as error:
            # Said once, on the status bar, rather than in a modal that
            # interrupts a reader mid-list. Losing the verdicts silently is
            # the thing not to do; stopping the review is also not the thing
            # to do.
            host.statusBar().showMessage(
                f"Could not save your read: {error}", 10000)

    panels["events"].judged.connect(keep_read)
    panels["findings"].channelJudged.connect(keep_read)
    # Judging an event changes how much of its channel has been judged, which
    # the findings table shows. Without this the progress column goes stale
    # the moment the reader starts working.
    panels["events"].judged.connect(
        lambda *_: panels["findings"].refresh(
            keep=panels["findings"].selected_channel()))


def _goto_channel(figure, session: ReviewSession, channel: str) -> None:
    """Jump to a channel's first event, or just scroll to the channel.

    A channel with no events still gets scrolled to: "there is nothing here" is
    an answer a reviewer is entitled to see on the trace.
    """
    events = sorted((e for e in session.hfo_events if e.channel == channel),
                    key=lambda e: e.start)
    t = (events[0].start - session.t_offset) if events else 0.0
    goto(figure, t, channel, session)


def _status(host: QMainWindow, session: ReviewSession) -> None:
    """Pin the caveat to the status bar for as long as the window is open.

    This is the interface's one piece of editorialising and it is deliberate.
    A ranked list of channels is what a reviewer will act on and it looks
    identical whether or not the data supports it, so the sentence saying which
    of those two it is does not get to be somewhere a reviewer might not look.
    """
    label = QLabel(session.caveat())
    label.setWordWrap(False)
    distinguishable = session.leader.get("distinguishable")
    # The one place the design is not allowed to be quiet. A caveat rendered
    # in the muted secondary colour would read as a footnote, which is exactly
    # what it must not be.
    label.setStyleSheet(
        "padding:2px 8px;font-size:9pt;"
        + (f"color:{theme.current().good};" if distinguishable
           else f"color:{theme.current().bad};font-weight:700;"))
    label.setObjectName("onset_caveat")
    # `addWidget` reparents the label onto the status bar, which then owns it,
    # so no reference is kept here. Naming it is how a caller or a test finds
    # it again without an attribute smuggled onto someone else's widget.
    host.statusBar().addWidget(label, 1)

    # Whose read this is, on the right, permanently. The caveat says what the
    # software thinks; this says who has signed up to it so far, and it is not
    # something to have to open a menu to check.
    reader = QLabel()
    reader.setObjectName("onset_reader")
    reader.setStyleSheet(
        f"padding:2px 8px;font-size:9pt;color:{theme.current().text_muted};")
    host.statusBar().addPermanentWidget(reader)


def _menus(figure, host: QMainWindow, panels: dict, docks: dict,
           session: ReviewSession, display: _Display,
           parts: ReviewWindowParts, on_import=None) -> None:
    """Menus and a toolbar, in the vocabulary of the task rather than the code."""
    menubar = host.menuBar()

    file_menu = menubar.addMenu("&Review")
    opener = file_menu.addAction("&Open a file…")
    opener.setShortcut("Ctrl+O")
    opener.setEnabled(on_import is not None)
    opener.setToolTip(
        "Read a recording from this machine through MNE — EDF, BrainVision, "
        "Persyst, Nihon Kohden, Nicolet, Blackrock, MEF3 and the rest. This "
        "window is replaced." if on_import is not None else
        "Opening another recording is not available in this window")
    if on_import is not None:
        opener.triggered.connect(lambda _=False: on_import())
    file_menu.addSeparator()
    file_menu.addAction("&Export review…", lambda: _export(host, session))
    file_menu.addSeparator()
    file_menu.addAction("&Close window", host.close)

    view = menubar.addMenu("&View")
    for dock_widget in docks.values():
        view.addAction(dock_widget.toggleViewAction())
    view.addSeparator()
    group = QActionGroup(host)
    group.setExclusive(True)
    for scope, text in MARK_SCOPES.items():
        action = view.addAction(text)
        action.setCheckable(True)
        action.setChecked(scope == display.scope)
        group.addAction(action)
        action.triggered.connect(
            lambda checked, scope=scope: (setattr(display, "scope", scope),
                                          parts.refresh_marks()))
    view.addSeparator()
    expert = view.addAction("Overlay &expert markings")
    expert.setCheckable(True)
    expert.setChecked(display.expert and session.has_expert)
    expert.setEnabled(session.has_expert)
    expert.setToolTip("Show the archive annotators' own HFO marks on the trace, "
                      "alongside the detector's" if session.has_expert else
                      "This recording carries no expert markings")
    expert.toggled.connect(
        lambda on: (setattr(display, "expert", bool(on)), parts.refresh_marks()))

    view.addSeparator()
    _window_actions(view, host)

    navigate = menubar.addMenu("&Navigate")
    navigate.addAction("&Next event", lambda: panels["events"].step(+1))
    navigate.addAction("&Previous event", lambda: panels["events"].step(-1))
    navigate.addSeparator()
    navigate.addAction("Busiest &channel",
                       lambda: _goto_channel(figure, session,
                                             session.leader.get("leader") or ""))
    navigate.addSeparator()
    controls = panels["controls"]
    navigate.addAction("Taller traces", lambda: controls._scale(AMPLITUDE_STEP))
    navigate.addAction("Smaller traces",
                       lambda: controls._scale(1 / AMPLITUDE_STEP))
    navigate.addAction("Back to the start of the window", controls.go_home)

    _read_menu(menubar, host, panels, session)

    help_menu = menubar.addMenu("&Help")
    help_menu.addAction("What am I looking at?", lambda: _about(host, session))
    help_menu.addAction("Keyboard shortcuts (MNE trace)",
                        lambda: _shortcuts(figure, host))


def _name_mnes_widgets(host: QMainWindow) -> None:
    """Give MNE's own dock and toolbar an `objectName`, so state can be saved.

    `QMainWindow.saveState` keys everything by object name and warns on stderr
    about each widget that has none -- and MNE's annotation dock and tool bar
    have none. Naming them here both silences that and makes the saved state
    complete, so restoring a layout puts MNE's widgets back too rather than
    leaving them wherever they happened to be.
    """
    from qtpy.QtWidgets import QToolBar

    for widget in host.findChildren(QDockWidget) + host.findChildren(QToolBar):
        if not widget.objectName():
            widget.setObjectName(f"mne_{type(widget).__name__}")


def _ask_reader(host: QMainWindow, session: ReviewSession) -> str:
    """Who is reviewing. Asked once, before the first verdict is recorded.

    Prefilled from the login name because typing it is friction and the
    machine usually knows -- but confirmed by a person, because `root` in a
    container is not a clinician and a report that quietly attributes a
    judgement to a login name is worse than one with a blank where the name
    should be.
    """
    from qtpy.QtWidgets import QInputDialog

    name, ok = QInputDialog.getText(
        host, "Who is reviewing?",
        "Your verdicts are recorded against this name, with the time you gave "
        "them.\nIt goes in the exported report.",
        text=session.read.reader or adjudication.reader_name())
    name = name.strip() if ok else ""
    if name:
        session.read.reader = name
        try:
            adjudication.save(session.request, session.read)
        except OSError:
            pass
        _set_reader_status(host, session)
    return name


def _set_reader_status(host: QMainWindow, session: ReviewSession) -> None:
    label = host.statusBar().findChild(QLabel, "onset_reader")
    if label is None:
        return
    read = session.read
    counts = read.counts()
    if not read.reader and not counts["judged"]:
        label.setText("")
        return
    label.setText(f"Read by {read.reader or 'nobody named'} — "
                  f"{counts['judged']} judged")


def _read_menu(menubar, host: QMainWindow, panels: dict,
               session: ReviewSession) -> None:
    """The reader's own menu: say who you are, and say what you think.

    A menu of its own rather than entries scattered through the others,
    because recording a judgement is a different activity from navigating or
    changing the display, and because it is the one activity in this window
    whose output has the reader's name on it.

    The shortcuts here are `Ctrl`-modified and work anywhere in the window;
    the bare letters on the events panel do the same thing while that panel
    has focus. Two sets on purpose: the bare letters are how the work is
    actually done, and a modified key is what someone finds by looking.
    """
    events, findings = panels["events"], panels["findings"]
    menu = menubar.addMenu("&Read")

    who = menu.addAction("&Who is reviewing…")
    who.setShortcut("Ctrl+Shift+R")
    who.setToolTip("The name your verdicts are recorded against")
    who.triggered.connect(lambda _=False: _ask_reader(host, session))
    menu.addSeparator()

    for verdict, text, key, tip in events.KEYS:
        action = menu.addAction(text.replace("&", "") + " — this event")
        action.setShortcut(f"Ctrl+{EVENT_SHORTCUTS[verdict]}")
        action.setToolTip(tip)
        action.triggered.connect(
            lambda _=False, verdict=verdict: events.judge(verdict))
    undo = menu.addAction("Clear this event's verdict")
    undo.setShortcut("Ctrl+Backspace")
    undo.triggered.connect(lambda _=False: events.judge(""))
    note = menu.addAction("&Note on this event…")
    note.triggered.connect(lambda _=False: events._write_note())
    menu.addSeparator()

    nxt = menu.addAction("Next &unjudged event")
    nxt.setShortcut("Ctrl+J")
    nxt.setToolTip("Skip to the next event you have not given a verdict on")
    nxt.triggered.connect(
        lambda _=False: events.step_unjudged(+1) or host.statusBar().showMessage(
            "Nothing left unjudged in this list.", 4000))
    menu.addSeparator()

    for verdict, text, tip in findings.CHANNEL_KEYS:
        action = menu.addAction(f"{text} — this contact")
        action.setToolTip(tip)
        action.triggered.connect(
            lambda _=False, verdict=verdict: findings.judge_channel(verdict))
    menu.addSeparator()

    about = menu.addAction("Note on this &window…")
    about.setToolTip("What you concluded, in your own words. It goes in the "
                     "report.")
    about.triggered.connect(lambda _=False: _window_note(host, session))

    prompt = lambda: _ask_reader(host, session)      # noqa: E731
    events.reader_prompt = prompt
    findings.reader_prompt = prompt
    for panel in (events, findings):
        signal = (panel.judged if panel is events else panel.channelJudged)
        signal.connect(lambda *_: _set_reader_status(host, session))


def _window_note(host: QMainWindow, session: ReviewSession) -> None:
    """The sentence a reader writes at the end: what they concluded."""
    from qtpy.QtWidgets import QInputDialog

    text, ok = QInputDialog.getMultiLineText(
        host, "Note on this window",
        "Your conclusion, in your own words. It is exported with the review.",
        session.read.note)
    if not ok:
        return
    session.read.note = text.strip()
    try:
        adjudication.save(session.request, session.read)
    except OSError as error:
        host.statusBar().showMessage(f"Could not save your read: {error}", 10000)


def _window_actions(view, host: QMainWindow) -> None:
    """Fit, maximise and full screen, in the View menu.

    The window manager's own buttons are the usual way to do this, and they
    are kept -- but they are not reliable here. A window larger than the work
    area gets its maximise button withheld by some window managers, and under
    a tiling or a minimal one there is no title bar to put a button on. A menu
    entry with a shortcut works in every case, and "fit to this screen" is
    something no title bar offers at all: it is the one to reach for after
    moving the window to a second monitor, or after a dock layout has pushed
    the window wider than the screen.

    `triggered` rather than `toggled` for the two checkable entries, so that
    re-syncing the ticks when the menu opens does not itself maximise the
    window.

    The fourth entry, restoring the default dock layout, is here for the same
    reason: it is the way back from an arrangement that no longer fits.
    """
    # Captured before the reviewer can drag anything, which is the point: the
    # docks are rearrangeable and a dragged-out panel is easy to lose. Qt has
    # no undo for a dock drag, so the default has to be kept somewhere.
    _name_mnes_widgets(host)
    default_layout = host.saveState()
    restore = view.addAction("&Restore the default layout")
    restore.setToolTip("Put the panels back where they started")
    restore.triggered.connect(
        lambda _=False: host.restoreState(default_layout))

    shrink = view.addAction("&Fit the window to this screen")
    shrink.setShortcut("Ctrl+0")
    shrink.setToolTip("Resize the window to fit the screen it is on, and "
                      "centre it")
    # `showNormal` first: it restores the geometry the window had before it
    # was maximised, which would otherwise undo the resize.
    shrink.triggered.connect(lambda _=False: (host.showNormal(),
                                              fit_to_screen(host)))

    big = view.addAction("Ma&ximise window")
    big.setShortcut("Ctrl+Shift+M")
    big.setCheckable(True)
    big.triggered.connect(
        lambda on: host.showMaximized() if on else host.showNormal())

    whole = view.addAction("F&ull screen")
    whole.setShortcut("F11")
    whole.setCheckable(True)

    def set_full(on: bool) -> None:
        if on:
            # Remembered, because leaving full screen with `showNormal` would
            # otherwise un-maximise a window that was maximised on the way in.
            host.setProperty("onset_was_maximised", host.isMaximized())
            host.showFullScreen()
        elif host.property("onset_was_maximised"):
            host.showMaximized()
        else:
            host.showNormal()

    whole.triggered.connect(lambda on: set_full(bool(on)))

    def sync() -> None:
        big.setChecked(host.isMaximized())
        whole.setChecked(host.isFullScreen())

    view.aboutToShow.connect(sync)
    sync()


def _apply_marks(figure, session: ReviewSession, scope: str,
                 channel: str | None, expert: bool) -> None:
    """Redraw the trace's marks without reloading a sample of signal.

    Replacing the annotations on the instance and asking the browser to redraw
    them is enough; the data itself never changes, so switching scope or
    toggling the expert overlay is instant even on a long window.
    """
    annotations = marks_for(session, scope, channel, expert)
    session.raw.set_annotations(annotations)
    state = getattr(figure, "mne", None)
    if state is not None and getattr(state, "inst", None) is not None:
        state.inst.set_annotations(annotations)
    try:
        figure._draw_annotations()
    except Exception:
        try:
            figure._redraw()
        except Exception:
            return
    _colour_annotations(figure)


def _export(host: QMainWindow, session: ReviewSession) -> None:
    request = session.request
    suggested = (f"review_{request.subject}_{request.t_start:g}-"
                 f"{request.t_stop:g}s_{request.band}.md")
    path, _ = QFileDialog.getSaveFileName(
        host, "Export review", str(Path.home() / suggested),
        "Markdown (*.md);;Web page (*.html)")
    if not path:
        return
    try:
        written = report.write_review(session, path)
    except OSError as error:
        QMessageBox.warning(host, "Export failed", str(error))
        return
    QMessageBox.information(host, "Review exported", f"Written to {written}")


def _about(host: QMainWindow, session: ReviewSession) -> None:
    summary = session.summary()
    QMessageBox.information(
        host, "What am I looking at?",
        f"<p><b>{summary['window']}</b>, {summary['band']}, "
        f"{summary['montage']} montage, {summary['channels']} channels.</p>"
        f"<p>The detector marked <b>{summary['accepted']}</b> events "
        f"({summary['rejected']} candidates were rejected as artifact) and "
        f"<b>{summary['spikes']}</b> interictal discharges. The archive's "
        f"annotators marked {summary['expert_events']} events on "
        f"{summary['n_reviewed']} channels of this window.</p>"
        f"<p>{session.caveat()}</p>"
        "<p>Read the <b>Trend</b> first: find a bright patch, click it, and the "
        "trace goes there. <b>Findings</b> ranks channels by rate and tints the "
        "ones that cannot be told apart from the busiest. <b>Events</b> walks "
        "the window one event at a time.</p>"
        "<p><b>Research prototype — not a medical device.</b> Not CE-marked, "
        "not FDA-cleared, not validated for clinical use.</p>")


def _shortcuts(figure, host: QMainWindow) -> None:
    """Hand off to MNE's own help window, which lists the trace's keys."""
    try:
        figure._create_help_fig()
        return
    except Exception:
        pass
    QMessageBox.information(
        host, "Keyboard shortcuts",
        "<p>The trace is <code>mne-qt-browser</code>. Press "
        "<code>?</code> with the trace focused for its full list; the ones used "
        "most here are:</p>"
        "<ul><li><code>←</code> / <code>→</code> — scroll in time</li>"
        "<li><code>↑</code> / <code>↓</code> — scroll through channels</li>"
        "<li><code>+</code> / <code>-</code> — change the amplitude scale</li>"
        "<li><code>a</code> — annotation mode</li>"
        "<li><code>b</code> — butterfly view</li></ul>")
