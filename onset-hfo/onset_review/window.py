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

from onset_review import report, theme
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
                figure.mne.plt.setYRange(top, top + on_screen, padding=0.0)
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
    fit = dock("quality", "Data quality", Qt.RightDockWidgetArea,
               panels["quality"],
               "Which contacts and which seconds were analysed, and why not")
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
    panels["trends"].setMinimumHeight(170)
    panels["controls"].setFixedHeight(panels["controls"].sizeHint().height())
    host.resizeDocks([docks["trends"]], [260], Qt.Vertical)
    host.resizeDocks([docks["findings"], docks["events"], brain, helper, prep,
                      fit, who], [640] * 7, Qt.Horizontal)
    # Vertical shares for the right-hand column. Without these the 3D view's
    # own minimum height wins the whole column and the two tables above it are
    # left showing one row each.
    host.resizeDocks([docks["findings"], docks["events"], brain],
                     [300, 240, 420], Qt.Vertical)

    parts = ReviewWindowParts(figure, host, panels, docks, session)
    parts.display = display
    _wire(figure, host, panels, session, display, parts)
    _menus(figure, host, panels, docks, session, display, parts,
           on_import=on_import)
    _status(host, session)
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

    help_menu = menubar.addMenu("&Help")
    help_menu.addAction("What am I looking at?", lambda: _about(host, session))
    help_menu.addAction("Keyboard shortcuts (MNE trace)",
                        lambda: _shortcuts(figure, host))


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
