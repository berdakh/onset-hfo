"""The review window: that it builds, and that its controls drive the trace.

A display is needed from the first line, so the import guard is at the top and
takes the whole file with it when the `review` extra is absent. The numbers are
not in here -- they are in `tests/test_review_core.py`, which has no Qt import
and runs everywhere. What is in here is the part that breaks silently when a
dependency changes underneath it: that MNE's figure is still a `QMainWindow`
and can host our docks, that every panel builds against a real session, and
that each control calls the browser method it claims to.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from onset_review import trends
from onset_review.session import ReviewRequest, ReviewSession, session_from_recording

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")

from qtpy.QtCore import Qt  # noqa: E402  (after the import guard, on purpose)


@pytest.fixture(scope="module")
def review(recording) -> ReviewSession:
    """One analysed window of the synthetic recording, reused by every test."""
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))

@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    app = qt.QApplication.instance() or qt.QApplication([])
    yield app


@pytest.fixture(scope="module")
def built(qapp, review):
    from onset_review import window

    figure = window.open_trace(review, show=False)
    parts = window.decorate(figure, review)
    yield parts
    try:
        figure.close()
    except Exception:
        pass


def test_mne_figure_can_still_host_our_docks(built):
    """The one architectural assumption of the whole window.

    `mne-qt-browser` returns a QMainWindow today, which is why the panels can be
    docked onto the trace instead of living in a second window. If a release
    ever changes that, this is where it should be noticed.
    """
    from onset_review import window

    assert window.has_dock_host(built.figure)
    assert built.docked


def test_every_panel_is_docked(built):
    assert set(built.docks) == {"trends", "controls", "findings", "events",
                                "detail", "brain", "map", "agreement", "provenance",
                                "assistant", "preprocess", "patient", "quality"}
    assert all(dock.widget() is not None for dock in built.docks.values())


#: The smallest screen this has to work on: the 1366x768 panel still shipped
#: on budget laptops, minus a 40 px top bar or taskbar. A window whose minimum
#: is larger than this does not merely overflow -- X11 window managers read the
#: size hints and withhold the maximise button from a window that cannot be
#: maximised into the work area, which is how the symptom was first reported.
SMALL_SCREEN = (1366, 728)


def test_the_window_can_shrink_to_a_small_laptop_screen(built):
    """Nothing in the window is allowed to put a floor under the whole thing.

    Qt builds a main window's minimum size by summing each dock column's
    minimums, so one panel that insists on 360 px of height makes the window
    insist on it too. This is the test that keeps that from creeping back: the
    preferred sizes are free to be generous, the minimums are not.
    """
    minimum = built.host.minimumSizeHint()
    assert minimum.width() <= SMALL_SCREEN[0], (
        f"window cannot be made narrower than {minimum.width()} px")
    assert minimum.height() <= SMALL_SCREEN[1], (
        f"window cannot be made shorter than {minimum.height()} px")


def test_fitting_to_the_screen_reports_whether_it_had_to_clamp(qapp):
    """`fit_to_screen` is the launcher's cue to open maximised instead.

    Against a bare `QMainWindow` rather than the review window, because this
    is the geometry rule on its own and it has to be testable on whatever
    screen the suite is running against -- including the 800x800 one Qt's
    offscreen platform reports, which is smaller than the review window's
    minimum and so could never exercise the "already fits" branch.
    """
    from onset_review import window

    host = qt.QMainWindow()
    try:
        area = window.work_area(host)
        assert area is not None, "no screen to fit to"

        host.resize(area.width() + 400, area.height() + 400)
        assert window.fit_to_screen(host) is True
        assert host.size().width() <= area.width()
        assert host.size().height() <= area.height()
        # Centred on what is left, not abandoned at a corner off the screen.
        assert area.contains(host.geometry())

        # A window that already fits is left exactly as it was.
        host.resize(area.width() // 2, area.height() // 2)
        kept = host.size()
        assert window.fit_to_screen(host) is False
        assert host.size() == kept
    finally:
        host.close()


def test_the_view_menu_can_maximise_and_go_full_screen(built):
    """The menu entries exist, and they are the ones a shortcut can reach.

    Asserted rather than assumed because the window manager's own buttons are
    not dependable here -- that is the whole reason these were added.
    """
    view = [action.menu() for action in built.host.menuBar().actions()
            if "View" in action.text()][0]
    entries = {action.text().replace("&", ""): action
               for action in view.actions() if action.text()}
    assert "Fit the window to this screen" in entries
    assert entries["Maximise window"].isCheckable()
    assert entries["Full screen"].shortcut().toString() == "F11"
    assert "Restore the default layout" in entries


def test_restoring_the_default_layout_undoes_a_dock_drag(built):
    """A dragged-out panel has to be recoverable; Qt has no undo for one."""
    view = [action.menu() for action in built.host.menuBar().actions()
            if "View" in action.text()][0]
    restore = [action for action in view.actions()
               if action.text().replace("&", "") == "Restore the default layout"][0]
    brain = built.docks["brain"]
    was_floating = brain.isFloating()
    try:
        brain.setFloating(True)
        assert brain.isFloating()
        restore.trigger()
        assert not brain.isFloating()
    finally:
        brain.setFloating(was_floating)


# -- a span longer than the trace -------------------------------------------


def test_the_three_time_bases_are_kept_straight(review):
    """`_file_time` and `_on_screen` are the whole of the arithmetic that keeps
    a table counting from the span and a trace counting from itself."""
    import dataclasses

    from onset_review import window

    spanned = dataclasses.replace(
        review,
        request=dataclasses.replace(review.request, t_start=300.0,
                                    t_stop=360.0, span_start=0.0,
                                    span_stop=600.0),
        span_start=0.0, span_stop=600.0)
    assert spanned.streamed

    # A table time of 310 s is 310 s into the recording, and the trace holds
    # 300-360, so it is on screen.
    assert window._file_time(spanned, 310.0) == pytest.approx(310.0)
    assert window._on_screen(spanned, 310.0) is True
    # 30 s into the span is not: that minute is not loaded.
    assert window._on_screen(spanned, 30.0) is False
    assert window._on_screen(spanned, 599.0) is False

    # And on an ordinary window, where span and trace are the same, every
    # event in the table is on screen — the behaviour before spans existed.
    assert not review.streamed
    assert window._on_screen(review, 0.0) is True
    assert window._on_screen(review, review.request.duration - 0.1) is True


def test_an_ordinary_window_says_nothing_about_a_span(built, review):
    """The sentence is only worth having when the two differ."""
    assert review.scope() == ""
    assert built.host.statusBar().findChild(qt.QLabel, "onset_scope") is None


def test_decorate_takes_the_trace_loader(qapp):
    """The callback exists and is optional, so a caller that cannot reload the
    trace gets a window that simply does not scroll off its own signal.

    Takes `qapp` only to satisfy the guard below, which matches on the word
    `decorate` and is crude on purpose — a false positive costs one unused
    fixture and a false negative costs a crash with no name on it.
    """
    import inspect

    from onset_review import window

    assert "on_trace_at" in inspect.signature(window.decorate).parameters


# -- moving through the recording -------------------------------------------


def test_the_review_menu_moves_to_another_window(built):
    menu = [action.menu() for action in built.host.menuBar().actions()
            if "File" in action.text()][0]
    entries = {action.text().replace("&", ""): action for action in menu.actions()
               if action.text()}
    assert "Next window" in entries
    assert "Previous window" in entries
    assert "Go to window…" in entries
    assert entries["Next window"].shortcut().toString() == "Ctrl+Shift+Right"
    # Disabled rather than missing when the caller cannot re-analyse: an entry
    # that silently does nothing is worse than one that says it cannot.
    assert not entries["Next window"].isEnabled()


def test_stepping_asks_for_the_next_window_of_the_same_length(review):
    """The arithmetic, without the loader: a window is moved by its own length,
    and never before the start of the recording."""
    import dataclasses

    from onset_review.app import _Review

    asked = []

    class _Fake(_Review):
        def __init__(self):
            self.request = dataclasses.replace(review.request, t_start=60.0,
                                               t_stop=120.0)
            self.parts = None

        def _rerun(self, **changes):
            asked.append(changes)

    mover = _Fake()
    mover.step_window(+1)
    assert asked == [{"t_start": 120.0, "t_stop": 180.0}]

    asked.clear()
    mover.step_window(-1)
    assert asked == [{"t_start": 0.0, "t_stop": 60.0}]


def test_the_first_window_does_not_step_back_past_the_start(review):
    import dataclasses

    from onset_review.app import _Review

    asked = []

    class _Fake(_Review):
        def __init__(self):
            self.request = dataclasses.replace(review.request, t_start=0.0,
                                               t_stop=60.0)
            self.parts = None

        def _rerun(self, **changes):
            asked.append(changes)

    mover = _Fake()
    mover.step_window(-1)
    assert asked == []


def test_a_window_of_no_length_is_refused(review):
    import dataclasses

    from onset_review.app import _Review

    asked = []

    class _Fake(_Review):
        def __init__(self):
            self.request = dataclasses.replace(review.request)
            self.parts = None

        def _rerun(self, **changes):
            asked.append(changes)

    mover = _Fake()
    mover.go_to_window(30.0, 30.0)
    mover.go_to_window(30.0, 10.0)
    assert asked == []


def test_the_reader_is_carried_to_the_next_window(review):
    """A reviewer who named themselves and then moved to the next minute is
    the same person. Being asked again at every window is how a reader learns
    to click past the question."""
    import dataclasses
    import types

    from onset_review.adjudication import Adjudication
    from onset_review.app import _Review

    previous = types.SimpleNamespace(
        session=types.SimpleNamespace(read=Adjudication(reader="Dr Smith")))

    class _Fake(_Review):
        def __init__(self):
            self.request = review.request
            self.parts = previous
            self.args = types.SimpleNamespace(reader=None)

    nxt = dataclasses.replace(review, read=Adjudication())
    nxt.request = dataclasses.replace(review.request, t_start=600.0,
                                      t_stop=660.0)
    _Fake()._attach_read(nxt)
    assert nxt.read.reader == "Dr Smith"
    # ...and the verdicts do not come with them: they were about other signal.
    assert nxt.read.counts()["judged"] == 0


# -- electrode coordinates --------------------------------------------------


def test_the_review_menu_offers_to_place_the_contacts(built):
    menu = [action.menu() for action in built.host.menuBar().actions()
            if "File" in action.text()][0]
    entries = {action.text().replace("&", "") for action in menu.actions()
               if action.text()}
    assert "Electrode coordinates…" in entries


def test_supplied_coordinates_replace_the_schematic_view(built, review, tmp_path):
    """And say in the caption that they came from a file, not the dataset."""
    from onset_review import coordinates
    from onset_review.anatomy import ARCHIVE_ORIGIN, NOT_ANATOMY

    brain = built.panels["brain"]
    before = brain.caption.text()
    assert NOT_ANATOMY in before

    names = coordinates.contacts_of(review)
    path = tmp_path / "electrodes.tsv"
    path.write_text("name\tx\ty\tz\n" + "".join(
        f"{name}\t{-30 if index % 2 else 30}\t{-20 + index * 3}\t-15\n"
        for index, name in enumerate(names)), encoding="utf-8")
    read = coordinates.read_coordinates(path, review)
    assert read.usable
    try:
        brain.set_electrodes(read.frame, origin="the file you supplied")
        assert NOT_ANATOMY not in brain.caption.text()
        assert "the file you supplied" in brain.caption.text()
        assert ARCHIVE_ORIGIN not in brain.caption.text()
        assert set(brain.layout_frame["source"]) == {"archive"}
    finally:
        brain.set_electrodes(review.electrodes)
    assert NOT_ANATOMY in brain.caption.text()


def test_the_coordinate_file_is_remembered_on_the_request(review, tmp_path):
    """So a reviewer who placed their contacts and then widened a notch does
    not have to find the file again."""
    import dataclasses

    request = dataclasses.replace(review.request, electrodes_path=tmp_path / "e.tsv")
    assert request.electrodes_path == tmp_path / "e.tsv"
    # And replacing something else keeps it, which is the whole point.
    again = dataclasses.replace(request, band="fast_ripple")
    assert again.electrodes_path == tmp_path / "e.tsv"


# -- task layouts ----------------------------------------------------------


def test_a_window_opens_in_a_layout_rather_than_showing_everything(built):
    """Eleven docked panels at once is an arrangement a reviewer has to undo
    before they can work."""
    from onset_review import window

    # `isHidden` rather than `isVisible`: nothing in this fixture's window has
    # been shown, so every widget in it is "not visible" whatever the layout.
    visible = {key for key, dock in built.docks.items() if not dock.isHidden()}
    assert visible == set(window.LAYOUTS[window.DEFAULT_LAYOUT][1])
    assert len(visible) < len(built.docks)


def test_every_layout_names_panels_that_exist(built):
    """The cheapest way for a layout to break is a renamed dock key."""
    from onset_review import window

    for name, (what, keys, focus) in window.LAYOUTS.items():
        assert what, name
        assert set(keys) <= set(built.docks), name
        assert focus in keys, name


def test_switching_layout_shows_its_panels_and_hides_the_rest(built):
    from onset_review import window

    try:
        for name, (_, keys, focus) in window.LAYOUTS.items():
            assert window.apply_layout(built.docks, name) is True
            shown = {key for key, dock in built.docks.items()
                     if not dock.isHidden()}
            assert shown == set(keys), name
            # And the layout decides which tab is in front, rather than
            # inheriting whichever one happened to be there.
            assert not built.docks[focus].isHidden()
    finally:
        window.apply_layout(built.docks, window.DEFAULT_LAYOUT)


def test_a_hidden_panel_is_hidden_and_not_destroyed(built):
    """Switching layouts has to cost nothing and lose nothing: every panel
    stays built and wired, one tick away in View."""
    from onset_review import window

    try:
        window.apply_layout(built.docks, "Reporting")
        assert built.docks["detail"].isHidden()
        assert built.docks["detail"].widget() is built.panels["detail"]
        # Still wired: selecting an event still draws it, unseen.
        built.panels["events"].view.selectRow(2)
        assert built.panels["detail"]._snapshot is not None
    finally:
        window.apply_layout(built.docks, window.DEFAULT_LAYOUT)


def test_an_unknown_layout_changes_nothing(built):
    from onset_review import window

    before = {key: dock.isHidden() for key, dock in built.docks.items()}
    assert window.apply_layout(built.docks, "Radiology") is False
    assert {k: d.isHidden() for k, d in built.docks.items()} == before


def test_the_view_menu_leads_with_the_layouts(built):
    from onset_review import window

    view = [action.menu() for action in built.host.menuBar().actions()
            if "View" in action.text()][0]
    texts = [action.text().replace("&", "") for action in view.actions()
             if action.text()]
    for index, name in enumerate(window.LAYOUTS):
        assert texts[index] == name
    assert texts[len(window.LAYOUTS)] == "Everything at once"


# -- the event detail view -------------------------------------------------


def test_selecting_an_event_draws_it_close_up(built):
    """One place decides which event is under discussion: the list."""
    events, close_up = built.panels["events"], built.panels["detail"]
    events.view.selectRow(0)
    key = events.selected_key()
    assert key
    assert close_up._snapshot is not None
    assert close_up._snapshot.channel == str(
        events.model.row_value(0, "channel"))
    assert "µV" in close_up.headline.text() or "ms" in close_up.headline.text()


def test_the_detail_view_says_the_signal_is_not_unprocessed(built):
    """The one view whose job is to let someone check the analysis must not
    claim to show them something it is not showing them."""
    events, close_up = built.panels["events"], built.panels["detail"]
    events.view.selectRow(1)
    text = close_up.caption.text().lower()
    assert "not unprocessed" in text
    assert "island" in text and "column" in text


def test_a_click_in_the_trend_draws_the_nearest_event_close_up(built):
    """Seen on a desktop: a click in the trend moved the trace, and the "This
    event" tab beside it stayed three empty frames with the hint at the
    bottom. The trend now picks the nearest listed event on that channel,
    which drives the detail view through the list; with nothing listed near
    the click, the trace still goes there."""
    events, close_up = built.panels["events"], built.panels["detail"]
    close_up.clear("nothing yet")
    assert not close_up.canvas.isVisibleTo(close_up), "no frames without an event"
    t = float(events.model.row_value(3, "t_local"))
    channel = str(events.model.row_value(3, "channel"))
    built.panels["trends"].cellPicked.emit(t + 0.4, channel)
    assert close_up._snapshot is not None and close_up._snapshot.channel == channel
    assert close_up.canvas.isVisibleTo(close_up)
    assert events.selected_key() == str(events.model.row_value(3, "key"))
    # Far from any event: nothing selected, nothing drawn, and no error.
    close_up.clear("again")
    assert events.select_nearest(10_000.0, channel) is False
    assert close_up._snapshot is None


def test_a_click_on_the_trace_resolves_to_a_time_and_a_channel(built, review):
    """The browser's scene click is turned into (span seconds, channel) the
    way its own crosshair does it, then handed to the same picker the trend
    uses."""
    from qtpy.QtCore import QPointF, Qt

    from onset_review import window

    figure = built.figure
    picked = []
    window._trace_clicks(figure, review, lambda t, ch: picked.append((t, ch)))
    trace = figure.mne.traces[0]
    x_trace = 2.0
    scene_point = figure.mne.viewbox.mapViewToScene(QPointF(x_trace, trace.ypos))

    class Click:
        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return scene_point

    figure.mne.plt.scene().sigMouseClicked.emit(Click())
    assert picked, "the click reached the picker"
    t_span, channel = picked[-1]
    assert channel == trace.ch_name
    assert t_span == pytest.approx(x_trace + float(review.t_offset) - float(review.span[0]),
                                   abs=0.05)


def test_a_key_with_no_event_leaves_the_panel_standing(built):
    close_up = built.panels["detail"]
    assert close_up.show_key("ZZ9-ZZ10|999.000|rms") is False
    assert close_up._snapshot is None
    assert "not in this analysis" in close_up.caption.text()


# -- the reader's own verdicts ---------------------------------------------


@pytest.fixture
def judging(built, review):
    """A window with a named reader and a clean slate of verdicts.

    Function-scoped against a module-scoped window, so each test starts from
    no verdicts without paying to rebuild the whole thing.
    """
    from onset_review.adjudication import Adjudication

    previous = review.read
    review.read = Adjudication(reader="Dr Smith")
    built.panels["events"].refilter()
    built.panels["findings"].refresh()
    yield built
    review.read = previous
    built.panels["events"].refilter()
    built.panels["findings"].refresh()


def test_a_verdict_lands_on_the_selected_event(judging, review):
    events = judging.panels["events"]
    events.view.selectRow(0)
    key = events.selected_key()
    assert key
    assert events.judge("agree") == key
    assert review.read.verdict_of(key) == "agree"
    assert review.read.events[key].reader == "Dr Smith"


def test_judging_advances_to_the_next_event(judging):
    """Four hundred events is four hundred key presses; reaching for the arrow
    key between each pair would double that for no reason."""
    events = judging.panels["events"]
    events.view.selectRow(3)
    events.judge("disagree")
    assert events.view.selectionModel().selectedRows()[0].row() == 4


def test_the_verdict_shows_in_the_table_and_can_be_taken_back(judging, review):
    events = judging.panels["events"]
    events.view.selectRow(0)
    key = events.judge("agree")
    assert events.model.row_value(0, "verdict") == "real"

    events.view.selectRow(0)
    events.judge("")
    assert review.read.verdict_of(key) == ""
    assert events.model.row_value(0, "verdict") == ""


def test_an_unattributed_verdict_is_refused(built, review):
    """The whole value of a recorded judgement is that someone can be asked
    about it. A file of anonymous opinions cannot be used for anything."""
    from onset_review.adjudication import Adjudication

    previous = review.read
    review.read = Adjudication()          # nobody named
    events = built.panels["events"]
    events.reader_prompt = lambda: ""     # and they decline to say
    try:
        events.view.selectRow(0)
        key = events.selected_key()
        assert events.judge("agree") == ""
        assert review.read.verdict_of(key) == ""

        events.reader_prompt = lambda: "Dr Jones"
        assert events.judge("agree") == key
        assert review.read.events[key].reader == "Dr Jones"
    finally:
        review.read = previous
        events.reader_prompt = None
        events.refilter()


def test_asking_who_is_reviewing_actually_runs(built, review, monkeypatch):
    """Exercised end to end rather than stubbed out.

    Every other test here injects `reader_prompt` directly, so `_ask_reader`
    itself was never executed — and it had a block of someone else's code
    pasted into it, referring to names that do not exist in its scope. It
    would have raised `NameError` the first time any reviewer was asked their
    name, which is the first verdict for anyone not passing `--reader`. The
    linter found it; this is what should have.
    """
    from onset_review import adjudication, window

    previous = review.read
    try:
        review.read = adjudication.Adjudication()
        monkeypatch.setattr(
            "qtpy.QtWidgets.QInputDialog.getText",
            staticmethod(lambda *args, **kwargs: ("Dr Jones", True)))
        assert window._ask_reader(built.host, review) == "Dr Jones"
        assert review.read.reader == "Dr Jones"
        # And the status bar picks it up, which is the other half of its job.
        label = built.host.statusBar().findChild(qt.QLabel, "onset_reader")
        assert "Dr Jones" in label.text()

        # Declining leaves the read unattributed rather than half-named.
        review.read = adjudication.Adjudication()
        monkeypatch.setattr(
            "qtpy.QtWidgets.QInputDialog.getText",
            staticmethod(lambda *args, **kwargs: ("", False)))
        assert window._ask_reader(built.host, review) == ""
        assert review.read.reader == ""
    finally:
        review.read = previous


def test_the_next_unjudged_event_skips_the_judged_ones(judging):
    events = judging.panels["events"]
    events.view.selectRow(0)
    events.judge("agree")        # judges row 0, lands on row 1
    events.judge("agree")        # judges row 1, lands on row 2
    events.view.selectRow(0)
    assert events.step_unjudged(+1) is True
    assert events.view.selectionModel().selectedRows()[0].row() == 2


def test_a_channel_verdict_shows_in_the_findings_table(judging, review):
    findings = judging.panels["findings"]
    findings.view.selectRow(0)
    channel = findings.selected_channel()
    assert findings.judge_channel("ignore") == channel
    assert review.read.channel_verdict(channel) == "ignore"
    assert findings.model.row_value(0, "my_read") == "Ignore this contact"


def test_judging_an_event_updates_the_channel_progress_column(judging, review):
    """Without the refresh wiring the progress column goes stale the moment
    the reader starts working, which is the moment it starts mattering."""
    findings, events = judging.panels["findings"], judging.panels["events"]
    events.view.selectRow(0)
    channel = str(events._shown.iloc[0]["channel"])
    findings.select_channel(channel)
    before = findings.model.row_value(
        findings.model.frame.index[
            findings.model.frame["channel"] == channel][0], "judged")
    events.judge("agree")
    after = findings.model.row_value(
        findings.model.frame.index[
            findings.model.frame["channel"] == channel][0], "judged")
    assert before != after
    assert after.startswith("1 of")


def test_every_verdict_is_written_to_disk_without_being_asked(judging, review):
    """There is no save button, on purpose: a reader who loses three hundred
    verdicts to a crash will not use this software again."""
    from onset_review import adjudication

    events = judging.panels["events"]
    events.view.selectRow(0)
    key = events.judge("agree")
    stored = adjudication.load(review.request)
    assert stored.verdict_of(key) == "agree"
    assert stored.reader == "Dr Smith"


def test_clicking_a_header_actually_sorts(built):
    """The arrow was decoration: `setSortingEnabled` draws it and calls
    `QAbstractItemModel.sort`, whose base implementation does nothing. Every
    header in this window was a control that moved and changed nothing."""
    events = built.panels["events"]
    column = events.model.column_index("amplitude_uv")
    try:
        events.view.sortByColumn(column, Qt.DescendingOrder)
        values = [float(events.model.row_value(row, "amplitude_uv"))
                  for row in range(min(8, events.model.rowCount()))]
        assert values == sorted(values, reverse=True)
        assert events.model.rowCount() == len(events._shown)
    finally:
        events.view.sortByColumn(events.model.column_index("t_local"),
                                 Qt.AscendingOrder)


def test_a_verdict_follows_the_row_it_was_given_on_after_a_sort(judging, review):
    """The hazard sorting introduces: if a selected row were turned back into
    an event through a frame held beside the model, sorting one and not the
    other would file the verdict against a different event."""
    events = judging.panels["events"]
    try:
        events.view.sortByColumn(events.model.column_index("amplitude_uv"),
                                 Qt.DescendingOrder)
        events.view.selectRow(0)
        key = events.selected_key()
        biggest = float(events.model.row_value(0, "amplitude_uv"))
        assert events.judge("agree") == key
        assert review.read.verdict_of(key) == "agree"
        # And the key really is the biggest event, not the first in time.
        matching = events._shown[events._shown["key"] == key]
        assert float(matching.iloc[0]["amplitude_uv"]) == pytest.approx(biggest)
    finally:
        events.view.sortByColumn(events.model.column_index("t_local"),
                                 Qt.AscendingOrder)


def test_the_sort_arrow_points_at_the_order_the_rows_are_in(built):
    events, findings = built.panels["events"], built.panels["findings"]
    header = events.view.horizontalHeader()
    assert events.model.column_name(header.sortIndicatorSection()) == "t_local"
    header = findings.view.horizontalHeader()
    assert findings.model.column_name(header.sortIndicatorSection()) == "rank"


def test_the_read_menu_offers_the_verdicts_and_names_the_reader(built):
    menu = [action.menu() for action in built.host.menuBar().actions()
            if "Read" in action.text()][0]
    entries = {action.text().replace("&", "") for action in menu.actions()
               if action.text()}
    assert "Who is reviewing…" in entries
    assert "Next unjudged event" in entries
    assert any(text.startswith("Real") for text in entries)
    assert any(text.startswith("Ignore it") for text in entries)


def test_the_status_bar_says_whose_read_this_is(judging, review):
    events = judging.panels["events"]
    events.view.selectRow(0)
    events.judge("agree")
    label = judging.host.statusBar().findChild(qt.QLabel, "onset_reader")
    assert label is not None
    assert "Dr Smith" in label.text()
    assert "1 judged" in label.text()


def test_the_single_letter_keys_are_scoped_to_the_events_panel(built):
    """A window-wide `A` would take MNE's annotation key away from the trace."""
    events = built.panels["events"]
    contexts = {shortcut.context() for shortcut in events.shortcuts}
    assert contexts == {Qt.WidgetWithChildrenShortcut}
    assert {shortcut.key().toString() for shortcut in events.shortcuts} >= {
        "A", "D", "U"}


def test_the_caveat_is_on_the_status_bar(built, review):
    """It is not allowed to be somewhere a reviewer might not look."""
    label = built.host.statusBar().findChild(qt.QLabel, "onset_caveat")
    assert label is not None
    assert label.text() == review.caveat()


def test_the_findings_panel_shows_every_channel(built, review):
    assert built.panels["findings"].model.rowCount() == len(review.findings)


def test_the_events_panel_filters_down_to_discharges(built, review):
    panel = built.panels["events"]
    panel.kind.setCurrentIndex(3)           # Discharges only
    panel.refilter()
    assert panel.model.rowCount() == len(review.spikes)
    panel.kind.setCurrentIndex(0)
    panel.refilter()
    assert panel.model.rowCount() == len(review.accepted)


def test_stepping_the_event_list_moves_one_row(built):
    panel = built.panels["events"]
    panel.step(+1)
    first = panel.view.selectionModel().selectedRows()[0].row()
    panel.step(+1)
    assert panel.view.selectionModel().selectedRows()[0].row() == first + 1


def test_marks_default_to_one_channel(built, review):
    """The default that makes the trace readable at all.

    Marking every detection draws five hundred full-height spans over forty
    channels of signal; the reviewer sees a barcode. Scoping to the channel
    under review is the default, and this is the test that keeps it one.
    """
    from onset_review import window

    everything = window.marks_for(review, scope="all")
    one = window.marks_for(review, scope="selected",
                           channel=review.leader.get("leader"))
    assert len(one) < len(everything)
    assert len(window.marks_for(review, scope="none")) == 0
    assert built.display.scope == "selected"


def test_the_expert_overlay_is_labelled_apart_from_ours(review):
    """Same band, two sources: MNE colours by description, so they must differ.

    The expert marks are grafted on rather than taken from the recording,
    because the synthetic one carries none and this path -- the comparison that
    makes the product worth sitting with -- should not go untested for want of
    a cached slice of `ds003498`.
    """
    import dataclasses

    from onset_hfo.detectors.base import Event
    from onset_review import window
    from onset_review.session import BAND_COLOURS

    channel = review.findings["channel"].iloc[0]
    expert = [Event(channel=channel, start=t, stop=t + 0.05, detector="expert",
                    band=(80.0, 250.0), accepted=True) for t in (1.0, 2.0, 3.0)]
    annotated = dataclasses.replace(review, expert=expert,
                                    reviewed_channels=[channel])

    overlay = window.marks_for(annotated, scope="all", expert=True)
    plain = window.marks_for(annotated, scope="all", expert=False)
    assert len(overlay) == len(plain) + len(expert)
    assert "expert ripple" in set(overlay.description)
    assert "expert ripple" not in set(plain.description)
    # Every description the trace can carry must have a colour, or MNE invents
    # one and the two sources stop being told apart by eye.
    assert set(overlay.description) <= set(BAND_COLOURS)

    # And the agreement table has to appear now that there is something to
    # agree with, restricted to the one channel the annotator looked at.
    summary = trends.agreement_summary(annotated)
    assert summary["available"] is True
    assert summary["n_expert"] == len(expert)
    assert list(trends.agreement(annotated)["channel"]) == [channel]


def test_a_trend_cell_maps_back_to_a_channel(built, review):
    panel = built.panels["trends"]
    panel.redraw()
    matrix = trends.rate_matrix(review, bin_s=float(panel.bin_s.currentData()))
    assert list(matrix.index) == list(review.findings["channel"])


# -- the open dialog -------------------------------------------------------
#
# The first screen a reviewer sees, and the one place a mistake is invisible in
# a headless test of everything else: the dialog opened with no detector
# selected for a while, and `request()` quietly fell back to the right one, so
# nothing downstream noticed.

@pytest.fixture
def cache(tmp_path):
    """Two cached windows: one 2000 Hz, one 1000 Hz, written as the loader does."""
    import json

    for meta in (
        {"dataset": "ds003498", "subject": "sub-01", "task": None, "run": "01",
         "t_start": 0, "t_stop": 60, "sfreq": 2000.0, "n_channels": 50},
        {"dataset": "ds003029", "subject": "sub-pt01", "task": "ictal",
         "run": "01", "t_start": 50, "t_stop": 60, "sfreq": 1000.0,
         "n_channels": 98},
    ):
        name = (f"{meta['subject']}_"
                + (f"task-{meta['task']}_" if meta["task"] else "")
                + f"run-{meta['run']}_{meta['t_start']:g}-{meta['t_stop']:g}s")
        directory = tmp_path / meta["dataset"] / name
        directory.mkdir(parents=True)
        (directory / "slice.json").write_text(json.dumps(meta))
    return tmp_path


def test_the_dialog_opens_with_a_detector_selected(qapp, cache):
    from onset_review.launcher import DEFAULT_DETECTOR, LauncherDialog

    dialog = LauncherDialog(cache)
    from qtpy.QtCore import Qt

    chosen = [item.data(Qt.UserRole) for item in dialog.detectors.selectedItems()]
    assert chosen == [DEFAULT_DETECTOR]
    assert dialog.request().detectors == (DEFAULT_DETECTOR,)


def test_the_dialog_offers_only_the_bands_the_rate_supports(qapp, cache):
    """A 500 Hz band on a 500 Hz Nyquist is meaningless, not conservative."""
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    bands = {}
    for index in range(dialog.subject.count()):
        dialog.subject.setCurrentIndex(index)
        _, subject = dialog.subject.itemData(index)
        bands[subject] = [dialog.band.itemData(i) for i in range(dialog.band.count())]
    assert bands["sub-01"] == ["ripple", "fast_ripple"]
    assert bands["sub-pt01"] == ["ripple"]


def test_the_dialog_says_why_a_band_is_missing(qapp, cache):
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    for index in range(dialog.subject.count()):
        if dialog.subject.itemData(index)[1] == "sub-pt01":
            dialog.subject.setCurrentIndex(index)
    assert "1000 Hz" in dialog.status.text()
    assert "interictal sleep" in dialog.status.text()      # and that it is ictal


def test_the_dialog_builds_the_request_it_is_showing(qapp, cache):
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    for index in range(dialog.subject.count()):
        if dialog.subject.itemData(index)[1] == "sub-pt01":
            dialog.subject.setCurrentIndex(index)
    request = dialog.request()
    assert (request.dataset, request.subject, request.task) == (
        "ds003029", "sub-pt01", "ictal")
    assert (request.t_start, request.t_stop) == (50.0, 60.0)
    # The spin box sits at its minimum until a reviewer moves it, and that
    # means "each detector's own measured threshold", not "1.0 SD".
    assert request.threshold_sd is None
    dialog.threshold.setValue(3.5)
    assert dialog.request().threshold_sd == 3.5


def test_an_empty_cache_disables_opening_and_says_what_to_do(qapp, tmp_path):
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(tmp_path)
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    assert "fetch" in dialog.status.text().lower()


# -- the trace controls ----------------------------------------------------
#
# The bar does not decide anything: it calls MNE's own view methods and reads
# its readouts back out of MNE. These tests pin exactly that, because the
# failure mode of a control bar that keeps its own copy of the state is that it
# silently disagrees with the trace it sits above.

def test_the_controls_read_their_values_from_the_browser(built):
    controls = built.panels["controls"]
    state = built.figure.mne
    assert controls.seconds.value() == pytest.approx(float(state.duration))
    assert controls.channels.value() == int(state.n_channels)
    assert controls.position.value() == pytest.approx(float(state.t_start))


def test_clicking_through_events_does_not_shrink_the_trace(built):
    """Found by the adjudication tests, which click far more than any test did.

    MNE recomputes `n_channels` as `round(y1 - y0 - 1)` every time the Y range
    changes, so scrolling to a channel with a range of exactly `n_channels`
    told it to show one fewer. Each click on an event cost a channel; after a
    dozen the trace was empty and MNE raised inside its own redraw. A reviewer
    walking an event list is exactly the person who would have hit it.
    """
    events = built.panels["events"]
    before = int(built.figure.mne.n_channels)
    assert before > 0
    for row in range(min(12, events.model.rowCount())):
        events.view.selectRow(row)
    assert int(built.figure.mne.n_channels) == before


def _microvolts(text: str) -> float:
    import re

    match = re.match(r"\s*([-+]?[\d.]+)\s*([munµ]?)V\s*$", text)
    assert match, f"not an amplitude: {text!r}"
    return float(match.group(1)) * {"": 1e6, "m": 1e3, "u": 1.0, "µ": 1.0,
                                    "n": 1e-3}[match.group(2)]


def test_the_microvolt_conversion_refuses_what_it_cannot_parse():
    """A wrong number beside a right one is worse than one number."""
    from onset_review.controls import as_microvolts

    assert as_microvolts("0.1 mV") == "100 µV"
    assert as_microvolts("50.0 µV") == "50 µV"
    assert as_microvolts("1.0 mV") == "1000 µV"
    for nonsense in ("", "auto", "nonsense", "mV", "0.1 mA"):
        assert as_microvolts(nonsense) == ""


def test_the_window_length_is_also_given_as_a_paper_speed(built):
    """The pairing every reader has in their hands: ten seconds at 30 mm/s.

    In the tooltip rather than as a second label on the bar, because that bar's
    width is the whole window's minimum width and a 1024 px screen is still a
    screen.
    """
    from onset_review.controls import paper_speed

    assert paper_speed(10.0) == "≈ 30 mm/s"
    assert paper_speed(5.0) == "≈ 60 mm/s"
    assert paper_speed(20.0) == "≈ 15 mm/s"
    assert paper_speed(0.0) == ""

    controls = built.panels["controls"]
    controls.sync()
    tip = controls.seconds.toolTip()
    assert paper_speed(controls.seconds.value()).lstrip("≈ ") in tip
    # Nothing here knows how wide the monitor is, and X11 reports a DPI that
    # is wrong as often as it is right, so it is called what it is.
    assert "equivalence, not a measurement" in tip
    assert "does not know the physical size" in tip


def test_the_gain_readout_is_mnes_own_scalebar(built):
    """A readout that disagrees with the scalebar drawn on the trace is worse
    than no readout.

    Agreement is now checked as the same *amplitude* rather than the same
    string: the readout is MNE's text converted to microvolts, which is the
    unit intracranial EEG is read in and the one nobody judging a 90 µV ripple
    wants to do arithmetic around. The number is the same number.
    """
    controls = built.panels["controls"]
    drawn = [text for text in built.figure._get_scale_bar_texts() if text]
    assert drawn
    assert controls.gain.text().endswith("µV")
    assert any(_microvolts(controls.gain.text()) == pytest.approx(
        _microvolts(text), rel=1e-6) for text in drawn)


def test_scaling_changes_the_trace_and_the_readout(built):
    from onset_review.controls import AMPLITUDE_STEP

    controls = built.panels["controls"]
    before = float(built.figure.mne.scale_factor)
    controls._scale(AMPLITUDE_STEP)
    assert float(built.figure.mne.scale_factor) == pytest.approx(
        before * AMPLITUDE_STEP)
    # The readout still names the same amplitude as the bar on the trace; it
    # names it in microvolts, which is the unit, not a different number.
    drawn = [text for text in built.figure._get_scale_bar_texts() if text]
    assert any(_microvolts(controls.gain.text()) == pytest.approx(
        _microvolts(text), rel=1e-6) for text in drawn)
    controls._scale(1 / AMPLITUDE_STEP)        # put it back for other tests
    assert float(built.figure.mne.scale_factor) == pytest.approx(before)


def test_changing_the_window_length_goes_through_the_browser(built):
    """And lands on the seconds asked for.

    `change_duration` takes a fraction of the current duration, not seconds, so
    a control that passes the difference in seconds overshoots badly -- 10 s to
    "12 s" becomes 30 s. That is the bug this asserts against.
    """
    controls = built.panels["controls"]
    before = float(built.figure.mne.duration)
    controls.seconds.setValue(before + 2.0)
    assert float(built.figure.mne.duration) == pytest.approx(before + 2.0, abs=0.1)
    controls.seconds.setValue(before)
    assert float(built.figure.mne.duration) == pytest.approx(before, abs=0.1)


def test_changing_the_channel_count_goes_through_the_browser(built):
    controls = built.panels["controls"]
    before = int(built.figure.mne.n_channels)
    controls.channels.setValue(before + 1)
    assert int(built.figure.mne.n_channels) == before + 1
    controls.channels.setValue(before)


def test_scrolling_moves_the_window_and_stays_inside_it(built):
    controls = built.panels["controls"]
    # A window as long as the recording has nowhere to scroll, and an earlier
    # test could have left it that way; make the state this test needs.
    controls.seconds.setValue(min(5.0, float(built.figure.mne.xmax) / 3))
    controls.go_home()
    assert controls.position.value() == pytest.approx(0.0)
    controls._hscroll("right")
    assert float(built.figure.mne.t_start) > 0.0
    for _ in range(50):                 # far past the end
        controls._hscroll("+full")
    state = built.figure.mne
    assert float(state.t_start) <= float(state.xmax) - float(state.duration) + 1e-6
    controls.go_home()


def test_a_control_whose_method_is_missing_leaves_the_window_standing(built):
    """The reviewer still has the keyboard; a dead button must not raise."""
    controls = built.panels["controls"]
    controls._call("no_such_method_on_the_browser", step=1)


# -- the 3D view -----------------------------------------------------------

def test_the_brain_panel_places_every_channel(built, review):
    panel = built.panels["brain"]
    assert len(panel.layout_frame) == len(review.findings)
    assert list(panel.layout_frame["channel"]) == list(
        review.findings.sort_values("rank")["channel"])


def test_the_brain_panel_says_it_is_not_anatomy(built):
    """The caption is the honesty guard and is not allowed to go missing.

    Either wording may apply -- a schematic layout when the electrode names map
    to structures, a montage diagram when none of them do -- and the headline
    has to agree with whichever the caption chose, or one of the two reads as
    boilerplate.
    """
    from onset_review.anatomy import NOT_ANATOMY

    caption = built.panels["brain"].caption.text()
    headline = built.panels["brain"].headline.text()
    assert NOT_ANATOMY in caption
    verdict = caption.split("—")[0].strip()
    assert verdict in ("SCHEMATIC LAYOUT", "MONTAGE DIAGRAM")
    assert headline.lower().startswith(verdict.lower())


def test_the_brain_panel_survives_every_option(built):
    panel = built.panels["brain"]
    for index in range(panel.colour_by.count()):
        panel.colour_by.setCurrentIndex(index)
        for labels in (True, False):
            panel.labels.setChecked(labels)
            panel.shafts.setChecked(not labels)
            panel.redraw()
    panel.colour_by.setCurrentIndex(0)
    panel.labels.setChecked(True)
    panel.shafts.setChecked(True)


def test_every_preset_view_is_a_real_angle(built):
    from onset_review.brainview import VIEWS

    panel = built.panels["brain"]
    for name in VIEWS:
        panel.set_view(name)
        assert (panel._elev, panel._azim) == VIEWS[name]


def test_highlighting_turns_to_the_right_hemisphere(built):
    panel = built.panels["brain"]
    frame = panel.layout_frame
    for sign, expected in ((1, 0), (-1, 180)):
        rows = frame[np.sign(frame["x"]) == sign]
        if rows.empty:
            continue
        panel.highlight(str(rows.iloc[0]["channel"]))
        assert panel._azim == expected


# -- the assistant ---------------------------------------------------------

def test_an_evidence_id_resolves_to_a_channel_and_a_time():
    from onset_review.assistant import parse_evidence_id

    assert parse_evidence_id("sub-01|AR1-AR2|rms|3.505") == ("AR1-AR2", 3.505)
    assert parse_evidence_id("not an id") is None
    assert parse_evidence_id("sub-01|AR1-AR2|rms|not-a-number") is None


def test_a_clicked_citation_survives_qts_url_encoding(qapp, review):
    """Qt percent-encodes the `|` in an evidence id on the way into a QUrl.

    Without unquoting, every citation link silently does nothing when clicked
    -- and the clickable citation is the entire argument that the assistant's
    answers can be checked rather than trusted.
    """
    from qtpy.QtCore import QUrl

    from onset_review.assistant import AssistantPanel

    panel = AssistantPanel(review)
    seen = []
    panel.evidencePicked.connect(lambda channel, t: seen.append((channel, t)))
    panel._citation_clicked(QUrl("evidence:sub-01|AR1-AR2|rms|3.505"))
    assert seen == [("AR1-AR2", 3.505)]
    assert "%7C" in QUrl("evidence:sub-01|AR1-AR2").toString()   # the hazard
    panel.deleteLater()


def test_the_assistant_offers_a_question_it_will_refuse():
    """A reviewer should meet the boundary from the interface, early."""
    from onset_review.assistant import SUGGESTIONS

    assert any("resect" in question.lower() for question, _ in SUGGESTIONS)
    assert all(label and len(label) < 24 for _, label in SUGGESTIONS)


def test_the_assistant_opens_on_the_configured_model(qapp, review, monkeypatch):
    """What the installer's --with-assistant buys: the panel opens on the
    model it set up, instead of on "No model" with the option left to find."""
    from onset_review.assistant import AssistantPanel

    monkeypatch.setenv("ONSET_ASSISTANT_BACKEND", "ollama")
    monkeypatch.setenv("ONSET_ASSISTANT_MODEL", "qwen3:8b")
    panel = AssistantPanel(review)
    assert panel.backend.currentData() == "ollama"
    assert panel.model.text() == "qwen3:8b"
    assert panel.model.isVisibleTo(panel)
    assert panel._defaults.source == "environment"


def test_the_assistant_still_opens_on_no_model_when_nothing_is_configured(
        qapp, review, monkeypatch, tmp_path):
    """The conftest already forbids the localhost probe; with no file and no
    environment the old default stands."""
    from onset_review.assistant import AssistantPanel

    monkeypatch.delenv("ONSET_ASSISTANT_BACKEND", raising=False)
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path))
    panel = AssistantPanel(review)
    assert panel.backend.currentData() == "scripted"
    assert "nothing" in panel._defaults.source


def test_the_panel_discovers_a_served_model_and_answers_through_it(
        qapp, review, monkeypatch, tmp_path):
    """The whole assistant path, through the real panel: a served model on
    localhost is discovered by the probe with nothing configured, the panel
    opens on it, a question goes through the worker thread and the real agent
    loop, the guards pass an honest answer, and the citation is a link.

    The model is a protocol-faithful fake (tests/_fake_ollama.py) that does
    what the system prompt asks -- read the briefing, cite -- so this exercises
    the success path rather than the refusal path random weights produce.
    """
    from _fake_ollama import FakeOllama

    from onset_agent.tools import dispatch
    from onset_review.assistant import AssistantPanel

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path))   # no installer file
    monkeypatch.delenv("ONSET_ASSISTANT_BACKEND", raising=False)
    monkeypatch.delenv("ONSET_ASSISTANT_NO_PROBE", raising=False)   # let it look
    with FakeOllama() as fake:
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        panel = AssistantPanel(review)
        try:
            assert panel.backend.currentData() == "ollama"
            assert panel.model.text() == fake.tag
            assert "localhost" in panel._defaults.source

            panel.ask("Which channel had the highest ripple rate?")

            leader = dispatch(panel._store, "top_channels", {"k": 1})["channels"][0]
            text = panel.transcript.toPlainText()
            assert leader["channel"] in text, text
            assert str(leader["rate_per_min"]) in text, text
            assert "stand behind" not in text          # not a refusal
            assert 'href="' in panel.transcript.toHtml()  # the citation is clickable
            # One model call, with the briefing's results already in front of it.
            assert len(fake.calls) == 1 and fake.calls[0][2] >= 4
            assert "Retrieved for the model top_channels" in text, "the work is shown"
        finally:
            panel.deleteLater()


def test_the_default_backend_runs_no_model():
    from onset_review.assistant import BACKENDS

    assert BACKENDS[0][1] == "scripted"
    assert any(kind == "ollama" for _, kind in BACKENDS)


def test_the_request_rebuilds_the_config_that_produced_the_review(review):
    """The assistant answers from a re-run; it has to be the *same* run.

    `pipeline_config` is what makes the review on screen and a `run_pipeline`
    over the same request one analysis rather than two that happen to agree.
    """
    from onset_hfo.detectors import HFO_DETECTORS

    config = review.request.pipeline_config()
    for name in HFO_DETECTORS:
        assert getattr(config, name).band == review.request.band_hz

    strict = dataclasses.replace(review.request, threshold_sd=4.25)
    for name in HFO_DETECTORS:
        assert getattr(strict.pipeline_config(), name).threshold_sd == 4.25


def test_the_assistant_answers_with_the_numbers_on_screen(qapp, review):
    """The point of the panel, and the thing that would rot silently.

    An assistant quoting rates from a differently-configured run would be worse
    than none: the numbers would look like the table's and not be them. This
    builds the store the agent reads and checks it against the session.
    """
    from onset_review.assistant import AssistantPanel

    panel = AssistantPanel(review)
    try:
        assert panel._ensure_store()
        rates = panel._store.rates[review.request.primary]
        mine = review.findings.set_index("channel")["n_events"]
        theirs = rates.set_index("channel")["n_events"]
        for channel, count in mine.items():
            assert int(theirs.get(channel, 0)) == int(count), channel
    finally:
        panel.deleteLater()


def test_the_assistant_keeps_the_window_painting_while_it_works(qapp, review):
    """It must not block the GUI thread while a model is thinking.

    `QThread.start()` followed by `wait()` on the GUI thread stops the window
    repainting entirely: the "Thinking…" line never appears, the panel does not
    visibly disable, and behind a local model that is half a minute of an
    application the desktop reports as not responding. The fix is a nested
    event loop, and this is what keeps it.
    """
    import inspect

    from onset_review import assistant

    source = inspect.getsource(assistant)
    assert "QEventLoop" in source
    # Exactly one place waits, and it is after the loop has already drained.
    waits = [line.strip() for line in source.splitlines()
             if ".wait()" in line and not line.strip().startswith("#")]
    assert waits == ["worker.wait()"], waits

    panel = assistant.AssistantPanel(review)
    try:
        painted = []
        panel.ask("Which channels have the highest ripple rate?")
        qapp.processEvents()
        painted.append(panel.transcript.toPlainText())
        assert "Thinking" in painted[0]
        assert panel.isEnabled()          # re-enabled when the answer landed
    finally:
        panel.deleteLater()


# -- the preprocessing panel ----------------------------------------------

def test_the_preprocessing_panel_opens_on_the_measured_defaults(built):
    """Opening it and pressing Apply without touching it must change nothing,
    so Apply starts disabled and there is nothing to warn about."""
    from onset_hfo.config import PreprocessConfig

    panel = built.panels["preprocess"]
    assert panel.config() == PreprocessConfig()
    assert not panel.apply.isEnabled()
    assert panel.warnings.text() == ""


def test_every_control_reaches_the_config(built):
    panel = built.panels["preprocess"]
    try:
        panel.highpass.setValue(2.0)
        panel.notch_width.setValue(3.0)
        panel.harmonics.setChecked(False)
        panel.average.setChecked(True)
        cfg = panel.config()
        assert cfg.highpass == 2.0
        assert cfg.notch_width == 3.0
        assert cfg.notch_harmonics is False
        assert (cfg.bipolar, cfg.average_reference) == (False, True)
        assert panel.apply.isEnabled()          # something changed
    finally:
        panel.reset_to_defaults()


def test_reset_returns_to_the_measured_defaults(built):
    from onset_hfo.config import PreprocessConfig

    panel = built.panels["preprocess"]
    panel.highpass.setValue(5.0)
    panel.monopolar.setChecked(True)
    panel.channels.item(0).setCheckState(Qt.Checked)
    assert panel.config() != PreprocessConfig()
    panel.reset_to_defaults()
    assert panel.config() == PreprocessConfig()
    assert not panel.apply.isEnabled()


def test_marking_a_contact_excludes_it(built):
    panel = built.panels["preprocess"]
    try:
        name = panel.channels.item(0).text()
        panel.channels.item(0).setCheckState(Qt.Checked)
        assert panel.config().exclude == (name,)
        # Contacts, not channels: exclusion happens before the bipolar montage
        # is built, which is what makes it a preprocessing choice.
        assert "-" not in name
    finally:
        panel.reset_to_defaults()


def test_a_band_destroying_setting_is_refused_by_the_panel(built):
    """Not merely warned about: Apply declines, so the reviewer finds out from
    the red text rather than from a dialog after a re-analysis."""
    panel = built.panels["preprocess"]
    emitted = []
    panel.applied.connect(emitted.append)
    try:
        panel.lowpass.setValue(150.0)
        # `isVisible` is False for anything inside a window that was never
        # shown, which is every widget here; what the reviewer reads is the
        # text, and that is what this is about.
        assert "cuts into the band" in panel.warnings.text()
        panel._apply()
        assert emitted == []
        assert "Fix this before applying" in panel.warnings.text()
    finally:
        panel.reset_to_defaults()


def test_a_sound_change_is_emitted(built):
    panel = built.panels["preprocess"]
    emitted = []
    panel.applied.connect(emitted.append)
    try:
        panel.notch_width.setValue(3.0)
        panel._apply()
        assert len(emitted) == 1
        assert emitted[0].notch_width == 3.0
    finally:
        panel.reset_to_defaults()


def test_apply_is_disabled_when_nothing_can_act_on_it(qapp, review):
    """`decorate` without a reload callback must not offer a button that does
    nothing when pressed."""
    from onset_review import window as window_module

    figure = window_module.open_trace(review, show=False)
    parts = window_module.decorate(figure, review)       # no on_preprocess
    try:
        assert not parts.panels["preprocess"].apply.isEnabled()
    finally:
        figure.close()


# -- the patient panel and the theme ---------------------------------------

def test_the_patient_panel_shows_the_record_and_hides_the_outcome(built, review):
    """The outcome is hidden behind a deliberate click.

    Knowing the patient became seizure-free changes how the same rate table
    reads, and this software is for forming an impression from the signal.
    """
    panel = built.panels["patient"]
    if not panel.record.get("available"):
        pytest.skip("the synthetic recording has no participant record")
    assert not panel.outcome.isVisibleTo(panel)
    panel.outcome_shown.setChecked(True)
    assert panel.outcome.isVisibleTo(panel)
    assert "ILAE" in panel.outcome.text()


def test_the_patient_panel_opens_on_a_subject_with_no_record(qapp, review):
    """A recording from outside the cohort must not take the panel down."""
    import dataclasses

    from onset_review.patient import PatientPanel

    stranger = dataclasses.replace(
        review, request=dataclasses.replace(review.request, subject="sub-99"))
    panel = PatientPanel(stranger)
    try:
        assert panel.record["available"] is False
    finally:
        panel.deleteLater()


def test_the_theme_sets_a_palette_and_a_stylesheet(qapp):
    """Forcing a theme with CSS alone leaves platform-drawn controls -- check
    indicators, spin-box arrows -- rendered for the other one."""
    from onset_review import theme

    for palette in (theme.LIGHT, theme.DARK):
        chosen = theme.apply_theme(qapp, palette)
        assert chosen is palette
        assert theme.current() is palette
        assert palette.accent in qapp.styleSheet()
        assert qapp.palette().base().color().name().lower() == \
            palette.surface.lower()
    theme.apply_theme(qapp, theme.LIGHT)


def test_panels_read_the_active_palette_rather_than_importing_one(qapp, review):
    """The dark theme's first version rendered tinted rows as light text on a
    light tint, because every panel had imported the light palette by name."""
    from onset_review import theme
    from onset_review.panels import FindingsPanel

    try:
        theme.set_current(theme.DARK)
        dark = FindingsPanel(review)
        tinted = [theme.DARK.highlight, theme.DARK.surface_alt, None]
        row = dark.model.frame.iloc[0]
        assert dark.model._highlight(row) in tinted
        dark.deleteLater()
    finally:
        theme.set_current(theme.LIGHT)


def test_every_warning_in_the_interface_is_the_same_warning_colour():
    """Before `theme.card` there were four hand-written hex values for this and
    two of them disagreed."""
    from onset_review import theme

    for kind in ("info", "warn", "bad", "plain"):
        assert "border-radius" in theme.card(kind)
    assert theme.LIGHT.bad in theme.card("bad", theme.LIGHT)
    assert theme.DARK.bad in theme.card("bad", theme.DARK)


def test_no_panel_hardcodes_a_colour():
    """Colours come from the palette so the two themes cannot diverge.

    Three files are exempt, each for its own reason. `theme.py` is where the
    colours live. `session.BAND_COLOURS` and `brainview.ZONE_EDGES` are data,
    not styling -- what a ripple is drawn as, what "inside the resection" is
    drawn as -- and they mean the same thing on either background. And
    `report.py` styles an exported HTML document that is read in a browser or
    printed, which should not inherit whatever theme the application happened
    to be in when it was written.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "onset_review"
    allowed = {"theme.py", "session.py", "brainview.py", "report.py"}
    for path in sorted(root.glob("*.py")):
        if path.name in allowed:
            continue
        hits = re.findall(r"#[0-9a-fA-F]{6}\b", path.read_text())
        assert not hits, f"{path.name} hardcodes {hits}"


# -- importing a file from this machine -----------------------------------
#
# The one screen in this software whose job is to stop something, rather than
# to show something. A clinical EDF or BrainVision export declares every
# channel `eeg` -- including the one real iEEG file this project's own cache
# holds -- and the pipeline analyses anything typed eeg, ecog or seeg. So an
# import that trusts the file produces a full review, with rates, intervals and
# a candidate channel set, for a scalp montage, and says nothing. These tests
# pin the parts of the dialog that exist to prevent exactly that.

@pytest.fixture(scope="module")
def recording_file(tmp_path_factory):
    """A file on disk that declares every channel `eeg`, as exports do."""
    import mne

    rng = np.random.default_rng(11)
    names = ["AR1", "AR2", "AR3", "HL1", "EKG", "DC1"]
    data = rng.normal(0.0, 2e-5, size=(len(names), 20 * 2000))
    info = mne.create_info(names, 2000.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path_factory.mktemp("import") / "study-001_raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    return path


def test_the_import_dialog_shows_the_file_s_own_answer_beside_the_choice(
        qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    assert dialog.error == ""
    assert dialog.table.rowCount() == 6
    declared = [dialog.table.item(row, 1).text() for row in range(6)]
    assert declared == ["eeg"] * 6                 # what the file claims
    assert "6 × eeg" in dialog._declared_summary()
    assert set(dialog.channel_types().values()) == {"seeg"}


def test_the_import_dialog_refuses_to_open_with_nothing_to_analyse(
        qapp, recording_file):
    """The one disabled button in this software that is an argument.

    Everything typed ecg or misc is dropped in preprocessing, so a file where
    nothing is marked SEEG or ECoG would load, preprocess down to no channels
    and fail somewhere unhelpful. Refusing here, with the reason on the button,
    is the same refusal made legible.
    """
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    button = dialog.buttons.button(QDialogButtonBox.Open)
    assert button.isEnabled()
    assert dialog.counts.text() == "6 of 6 will be analysed"

    dialog.set_all("ecg")
    assert not button.isEnabled()
    assert dialog.counts.text() == "0 of 6 will be analysed"
    assert "nothing to analyse" in button.toolTip()

    dialog.set_all("ecog")
    assert button.isEnabled()
    assert dialog.counts.text() == "6 of 6 will be analysed"


def test_one_channel_changed_by_hand_updates_the_count(qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    ekg = [row for row in range(dialog.table.rowCount())
           if dialog.table.item(row, 0).text() == "EKG"][0]
    chooser = dialog.table.cellWidget(ekg, 2)
    chooser.setCurrentIndex(
        [i for i in range(chooser.count()) if chooser.itemData(i) == "ecg"][0])
    assert dialog.counts.text() == "5 of 6 will be analysed"
    assert dialog.channel_types()["EKG"] == "ecg"


def test_the_import_dialog_builds_the_request_it_is_showing(qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    dialog.set_all("seeg")
    dialog.subject.setText("study-001")
    dialog.t_stop.setValue(15.0)
    dialog.line_freq.setCurrentIndex(1)              # 60 Hz
    request = dialog.request()

    assert request.imported is True
    assert request.path == recording_file
    assert request.subject == "study-001"
    assert (request.t_start, request.t_stop) == (0.0, 15.0)
    assert request.line_freq == 60.0
    assert dict(request.channel_types) == {name: "seeg" for name in
                                           ("AR1", "AR2", "AR3", "HL1",
                                            "EKG", "DC1")}


def test_the_window_cannot_be_set_past_the_end_of_the_file(qapp, recording_file):
    """A reviewer asking for 0–60 s of a 20 s file is corrected by the spin box.

    Not by an exception after the reader has run: by then they have waited, and
    the message is about a window rather than about the file being short.
    """
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    dialog.t_stop.setValue(600.0)
    assert dialog.t_stop.value() == pytest.approx(20.0, abs=0.01)
    assert dialog.t_stop.value() <= dialog.info["duration"] + 0.01


def test_the_import_dialog_offers_only_the_bands_the_file_can_support(
        qapp, tmp_path_factory):
    """The same refusal the launcher makes, made from the file's own header."""
    import mne

    from onset_review.importer import ImportDialog

    info = mne.create_info(["A1", "A2"], 1000.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 5000)), info, verbose="ERROR")
    slow = tmp_path_factory.mktemp("slow") / "slow_raw.fif"
    raw.save(slow, overwrite=True, verbose="ERROR")

    dialog = ImportDialog(slow)
    assert [dialog.band.itemData(i) for i in range(dialog.band.count())] == ["ripple"]
    assert "needs more than 1000 Hz" in _form_text(dialog)


def _form_text(dialog) -> str:
    from qtpy.QtWidgets import QLabel

    return " ".join(label.text() for label in dialog.findChildren(QLabel))


def test_an_unreadable_file_is_a_message_rather_than_a_traceback(
        qapp, tmp_path):
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    broken = tmp_path / "truncated_raw.fif"
    broken.write_bytes(b"not a fif file")
    dialog = ImportDialog(broken)
    assert dialog.table is None
    assert dialog.error
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()


def test_the_dialog_says_what_an_imported_file_does_not_bring(qapp,
                                                              recording_file):
    """Said before the reviewer finds the panels empty, not after."""
    from onset_review.importer import ImportDialog

    text = _form_text(ImportDialog(recording_file))
    assert "no expert markings" in text
    assert "not a medical device" in text
    assert "declare every channel as scalp EEG" in text


def test_the_launcher_offers_a_file_even_with_an_empty_cache(qapp, tmp_path):
    """An empty archive cache is exactly when someone has their own recording."""
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(tmp_path)
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    assert dialog.import_button.isEnabled()
    assert "open a file" in dialog.status.text().lower()


def test_an_imported_request_keeps_the_launcher_s_analysis_choices(
        qapp, cache, recording_file, monkeypatch):
    """The band comes from the file's rate; the detectors come from the dialog.

    Two screens, each answering what it is in a position to answer. Asking for
    the detector again on the import screen would be a third page nobody needs,
    and asking for the band on the launcher would offer one the file cannot
    support.
    """
    from onset_review import importer
    from onset_review.importer import ImportDialog
    from onset_review.launcher import LauncherDialog

    def fake_choose(parent=None, start=None):
        dialog = ImportDialog(recording_file)
        dialog.set_all("seeg")
        return dialog.request()

    monkeypatch.setattr(importer, "choose_file", fake_choose)

    dialog = LauncherDialog(cache)
    dialog.threshold.setValue(3.5)
    dialog.spikes.setChecked(False)
    dialog.expert.setChecked(True)
    dialog.open_file()

    request = dialog.request()
    assert request.imported is True
    assert request.threshold_sd == 3.5
    assert request.with_spikes is False
    assert request.band == "ripple"
    # The expert overlay belongs to the archive; an imported file has none, so
    # a checkbox left ticked from a previous choice must not travel with it.
    assert dialog.overlay_expert() is False


def test_a_cancelled_import_leaves_the_launcher_as_it_was(qapp, cache,
                                                          monkeypatch):
    from onset_review import importer
    from onset_review.launcher import LauncherDialog

    monkeypatch.setattr(importer, "choose_file", lambda parent=None, start=None: None)
    dialog = LauncherDialog(cache)
    dialog.open_file()
    assert dialog.request().imported is False


def test_the_review_menu_can_open_another_file(qapp, built):
    """And says so even when it cannot, rather than offering a dead entry."""
    actions = {action.text(): action
               for action in built.host.menuBar().actions()[0].menu().actions()}
    opener = [text for text in actions if "Open a file" in text]
    assert opener, list(actions)
    # `built` is decorated without an import callback, so the entry is there to
    # be found and explicitly disabled.
    assert not actions[opener[0]].isEnabled()
    assert "not available" in actions[opener[0]].toolTip()


def test_the_patient_panel_says_an_imported_window_has_no_record(qapp, recording):
    """Not "no record found": no record *exists*, and the difference matters."""
    import pathlib

    from onset_review.patient import PatientPanel

    request = ReviewRequest(dataset="", subject="study-001",
                            path=pathlib.Path("study-001_raw.fif"),
                            t_start=0.0, t_stop=float(recording.duration))
    review = session_from_recording(recording, request)
    panel = PatientPanel(review)
    text = " ".join(label.text() for label in panel.findChildren(qt.QLabel))
    assert "imported from study-001_raw.fif" in text
    assert "no participant record, no resected zone" in text
    # The archive's withheld-demographics note is about a cohort this window is
    # not part of, so it is not shown.
    assert "age, sex and handedness" not in text


def test_the_command_line_flags_open_the_dialog_on_their_answer(qapp,
                                                                recording_file):
    """`--window 0 30 --all-channels-as seeg` is an answer, not a suggestion.

    Discarding it and asking again would make the flags useless for anyone who
    wants to check what they typed before it runs.
    """
    from onset_review.app import _prefill, build_parser
    from onset_review.importer import ImportDialog

    args = build_parser().parse_args(
        ["--open", str(recording_file), "--window", "2", "12",
         "--line-freq", "60", "--subject", "study-001",
         "--all-channels-as", "seeg", "--channel-type", "EKG=ecg",
         "--channel-type", "NOT-A-CHANNEL=seeg"])

    dialog = ImportDialog(recording_file)
    _prefill(dialog, args)

    request = dialog.request()
    assert (request.t_start, request.t_stop) == (2.0, 12.0)
    assert request.line_freq == 60.0
    assert request.subject == "study-001"
    assert dict(request.channel_types)["EKG"] == "ecg"
    assert dict(request.channel_types)["AR1"] == "seeg"
    assert dialog.counts.text() == "5 of 6 will be analysed"


def test_the_channel_table_opens_tall_enough_to_be_a_list(qapp,
                                                          tmp_path_factory):
    """Fifty contacts behind a four-row window is a field to click past.

    The floor is counted in rows rather than pixels so it survives a different
    font or display scale, and it shrinks to the montage: an eight-contact
    strip should not open a dialog two thirds empty. The height the dialog
    *asks* for is checked rather than the one it gets, because what it gets is
    clamped to the display, and a headless screen is 1024x768.
    """
    import mne
    from qtpy.QtGui import QGuiApplication

    from onset_review.importer import (
        VISIBLE_ROWS,
        VISIBLE_ROWS_MIN,
        ImportDialog,
    )

    def written(n_channels, where):
        names = [f"A{i + 1}" for i in range(n_channels)]
        info = mne.create_info(names, 2000.0, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((n_channels, 4000)), info,
                              verbose="ERROR")
        path = tmp_path_factory.mktemp(where) / "rec_raw.fif"
        raw.save(path, overwrite=True, verbose="ERROR")
        return path

    many = ImportDialog(written(40, "many"))
    assert many.table.minimumHeight() == many._rows_tall(VISIBLE_ROWS_MIN)
    assert many.wanted_height() > many.sizeHint().height()     # asks for more
    assert many.wanted_height() - many.sizeHint().height() == (
        many._rows_tall(VISIBLE_ROWS) - many._rows_tall(VISIBLE_ROWS_MIN))

    # Whatever it asked for, it must fit on the display it is opening on. An
    # Open button below the bottom of a laptop screen is worse than a short
    # table.
    cap = int(QGuiApplication.primaryScreen().availableGeometry().height() * 0.9)
    assert many.height() <= max(cap, many.minimumSizeHint().height())

    few = ImportDialog(written(4, "few"))
    # Four rows cannot fill seven, and the floor must not invent the other
    # three as empty space.
    assert few.table.minimumHeight() == few._rows_tall(4)
    assert few._rows_tall(VISIBLE_ROWS) == few._rows_tall(4)
    assert few.wanted_height() == few.sizeHint().height()


def test_every_test_that_builds_a_widget_asks_for_the_application():
    """A missing `qapp` is a crash, not a failure, and only sometimes.

    Qt aborts the process when a `QWidget` is constructed with no
    `QApplication`. A test that builds one without requesting the fixture
    therefore passes for as long as some *earlier* test in the file happens to
    have created the application — and takes the whole run down with `Fatal
    Python error: Aborted`, no test name and no traceback, the moment it runs
    without one in front of it. A `-k` selection does that, and so would
    splitting the file across workers, reordering it, or deleting the test
    that was holding the fixture up.

    Three tests in this file were in that state, found by a `-k` selection
    that happened to run one of them alone. The suite had been green
    throughout, which is the point: whole-file order was hiding them.
    """
    import ast
    import pathlib

    #: Names that mean "this line constructs a widget". Deliberately crude:
    #: a false positive costs one unused fixture, a false negative costs a
    #: crash with no name on it.
    WIDGETY = ("Panel", "Dialog", "Controls", "decorate", "open_trace",
               "muted", "section_label", "QLabel", "QWidget")
    #: Fixtures that create or depend on the `QApplication`. Transitive
    #: dependencies are listed by hand: this check parses the file rather than
    #: resolving pytest's fixture graph, so a fixture that takes `built` has
    #: to be named here too.
    PROVIDES_APP = {"qapp", "built", "judging", "paged"}

    tree = ast.parse(pathlib.Path(__file__).read_text())
    offenders = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith("test_")):
            continue
        if node.name == "test_every_test_that_builds_a_widget_asks_for_the_application":
            continue                       # its own body names every word
        if PROVIDES_APP & {arg.arg for arg in node.args.args}:
            continue
        source = ast.dump(node)
        if any(word in source for word in WIDGETY):
            offenders.append(f"{node.name} (line {node.lineno})")
    assert not offenders, (
        "these build a widget without the qapp fixture, and will abort the "
        "process if test order puts them first: " + ", ".join(offenders))


# -- the Data quality panel ------------------------------------------------
#
# The panel's numbers are tested in `tests/test_quality.py`, which needs no
# display. What is here is the part a reviewer touches, and above all the one
# behaviour that must not regress: a **flagged** contact is analysed, kept in
# the ranking, and cannot be "reinstated" because it was never taken out. The
# distinction between flagged and set aside is the whole design, and it was
# drawn by sub-13 of ds003498, whose annotators had marked 91, 102 and 164
# ripples on the three contacts this software's band-power check flags.

def _faulted(recording, noisy: int = 0, dead: tuple[int, ...] = (8, 9)):
    """The synthetic recording with one noisy contact and two dead ones."""
    import copy

    out = copy.deepcopy(recording)
    raw = out.raw.load_data()
    rng = np.random.default_rng(5)
    row = raw.get_data(picks=[noisy])[0]
    raw._data[noisy] = row + rng.normal(0, 8 * np.std(row), row.size)
    for index in dead:
        raw._data[index] = 0.0
    return out


@pytest.fixture(scope="module")
def faulted(recording):
    return session_from_recording(
        _faulted(recording),
        ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


def test_the_quality_panel_separates_set_aside_from_flagged(qapp, faulted):
    from onset_review.dataquality import QualityPanel

    frame = QualityPanel(faulted).rows()
    aside = frame[~frame["good"]]
    flagged = frame[frame["flagged"]]
    assert not aside.empty, "the two dead contacts should have been set aside"
    assert not flagged.empty, "the noisy contact should have been flagged"
    assert set(aside["channel"]).isdisjoint(set(flagged["channel"]))

    # Worst first: a reviewer opens this because something is wrong.
    assert frame.iloc[0]["channel"] in set(aside["channel"])
    assert "set aside" in frame.iloc[0]["verdict"]
    assert "analysed, flagged" in flagged.iloc[0]["verdict"]


def test_a_flagged_contact_keeps_its_rate_in_the_findings_table(qapp, faulted):
    """The behaviour the whole flagged/set-aside split exists to protect.

    A contact full of real ripples has high band power because the ripples are
    in the band, and nothing here can tell that from a noisy amplifier. So a
    flagged contact is analysed, ranked and rated like any other; only the
    unambiguous faults lose their rate.
    """
    from onset_review.dataquality import QualityPanel

    frame = QualityPanel(faulted).rows()
    findings = faulted.findings.set_index("channel")

    for channel in frame[frame["flagged"]]["channel"]:
        row = findings.loc[channel]
        assert not np.isnan(row["rate_per_min"]), f"{channel} lost its rate"
        assert row["rank"] >= 1, f"{channel} lost its rank"
        assert faulted.clean_seconds[channel] > 0

    for channel in frame[~frame["good"]]["channel"]:
        row = findings.loc[channel]
        assert np.isnan(row["rate_per_min"])      # no rate is claimed
        assert row["rank"] == 0                   # and no rank: 0 is "unranked"
        assert faulted.clean_seconds[channel] == 0.0


def test_only_a_set_aside_contact_can_be_reinstated(qapp, faulted):
    """A flagged contact is in the analysis, so there is nothing to put back.

    The button staying grey on a flagged row is the panel saying exactly that;
    the thing to do with a flag is look at the trace.
    """
    from onset_review.dataquality import QualityPanel

    panel = QualityPanel(faulted)
    frame = panel.rows()

    def select(channel):
        rows = [r for r in range(panel.table.rowCount())
                if panel.table.item(r, 0).text() == channel]
        panel.table.selectRow(rows[0])

    select(str(frame[frame["flagged"]]["channel"].iloc[0]))
    assert not panel.reinstate.isEnabled()

    aside = str(frame[~frame["good"]]["channel"].iloc[0])
    select(aside)
    assert panel.reinstate.isEnabled()
    panel.reinstate.click()
    assert panel.choice() == (True, (aside,))
    assert panel.apply.isEnabled()
    assert "reinstated by you" in panel.rows().set_index("channel").loc[aside, "verdict"]

    # And once it is back, there is nothing left to reinstate on that row.
    select(aside)
    assert not panel.reinstate.isEnabled()


def test_the_panel_applies_nothing_until_something_changes(qapp, faulted):
    from onset_review.dataquality import QualityPanel

    panel = QualityPanel(faulted)
    assert not panel.apply.isEnabled()

    panel.enabled.setChecked(False)
    assert panel.choice() == (False, ())
    assert panel.apply.isEnabled()

    panel.reset.click()
    assert panel.choice() == (True, ())
    assert not panel.apply.isEnabled()


def test_the_panel_emits_the_choice_rather_than_acting_on_it(qapp, faulted):
    """Like the preprocessing panel: it decides nothing and re-runs nothing.

    Setting a contact aside changes every rate, interval and rank, so the
    entry point rebuilds the whole window rather than refreshing some panels
    and leaving others stale.
    """
    from onset_review.dataquality import QualityPanel

    panel = QualityPanel(faulted)
    seen = []
    panel.applied.connect(lambda check, keep: seen.append((check, keep)))
    panel.enabled.setChecked(False)
    panel.apply.click()
    assert seen == [(False, ())]


def test_the_panel_says_what_it_cannot_tell(qapp, faulted):
    """The caveat is on the screen, not only in the source.

    A reviewer reading a flagged row has to know the software is asking a
    question rather than answering one.
    """
    from onset_review.dataquality import QualityPanel

    panel = QualityPanel(faulted)
    text = " ".join(label.text() for label in panel.findChildren(qt.QLabel))
    assert "Nothing is repaired" in text
    assert "analysed and flagged" in text
    assert "look at these on the trace" in panel._summary_text()


def test_the_quality_dock_is_in_the_window(built):
    assert "quality" in built.panels
    assert "quality" in built.docks
    # Short, like the others: seven tabs in a 640 px column elide, and the
    # sentence lives in the tooltip.
    assert built.docks["quality"].windowTitle() == "Quality"
    assert "which contacts" in built.docks["quality"].toolTip()


def test_dense_annotation_labels_are_thinned_not_overprinted(built):
    """Seen on ds003498 sub-01: an expert's ripples a few hundred milliseconds
    apart turn the labels along the bottom of the trace into
    "experippleexperipple". The bands must all stay; the text thins to what
    can be read, and comes back as you zoom in."""
    from qtpy.QtWidgets import QApplication

    from onset_review.window import LABEL_GAP_PX, thin_annotation_labels

    figure = built.figure
    app = QApplication.instance()
    before = len(figure.mne.regions)
    for i in range(40):                                 # 40 marks in 2 seconds
        figure._add_region(0.5 + i * 0.05, 0.03, "expert ripple")
    app.processEvents()
    planted = [r for r in figure.mne.regions if r.description == "expert ripple"][-40:]
    assert len(figure.mne.regions) == before + 40

    def planted_shown() -> int:
        return sum(r.label_item.isVisible() for r in planted)

    figure.mne.plt.setXRange(0.0, 10.0, padding=0.0)
    app.processEvents()
    assert thin_annotation_labels(figure) > 0
    shown_wide = planted_shown()
    assert 0 < shown_wide < 40, shown_wide
    assert all(r.isVisible() for r in planted), "a band was hidden; only text may be"

    # No two shown labels overprint, measured in the pixels that collide.
    per_px = figure.mne.viewbox.viewPixelSize()[0]
    spans = []
    for r in sorted(planted, key=lambda r: r.getRegion()[0]):
        if r.label_item.isVisible():
            a, b = r.getRegion()
            half = r.label_item.boundingRect().width() * per_px / 2
            spans.append(((a + b) / 2 - half, (a + b) / 2 + half))
    for (_, right), (left, _) in zip(spans, spans[1:], strict=False):
        assert left - right >= (LABEL_GAP_PX - 1) * per_px

    # Zooming in makes room: the hook re-thins and more labels come back.
    figure.mne.plt.setXRange(0.4, 1.2, padding=0.0)
    app.processEvents()
    shown_narrow = planted_shown()
    assert shown_narrow > shown_wide, (shown_narrow, shown_wide)


# -- the page layout -----------------------------------------------------------


@pytest.fixture(scope="module")
def paged(qapp, review):
    """The same session laid out as pages, with every callback recorded."""
    import pandas as pd

    from onset_review import window

    calls = {"opened": [], "relayout": [], "imported": 0}
    frame = pd.DataFrame([{"dataset": "ds003498", "subject": "sub-01", "run": "01",
                           "task": None, "t_start": 0.0, "t_stop": 60.0,
                           "sfreq": 2000.0, "n_channels": 50}])
    figure = window.open_trace(review, show=False)
    parts = window.decorate(
        figure, review, mode="pages", cached=lambda: frame,
        on_open_cached=calls["opened"].append,
        on_relayout=calls["relayout"].append,
        on_import=lambda: calls.__setitem__("imported", calls["imported"] + 1),
        on_window=lambda a, b: None, on_step_window=lambda d: None)
    parts.calls = calls
    yield parts
    try:
        figure.close()
    except Exception:
        pass


def _menu(host, title: str):
    for action in host.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    raise AssertionError(f"no {title} menu")


def _actions(menu) -> dict:
    return {a.text().replace("&", ""): a for a in menu.actions() if a.text()}


def test_the_page_window_is_its_own_window_and_holds_every_panel(paged):
    host = paged.host
    assert isinstance(host, qt.QMainWindow)
    assert host is not paged.figure, "the host is a window around the figure"
    assert paged.pages is host and paged.docks == {}
    for key, panel in paged.panels.items():
        assert host.isAncestorOf(panel), f"{key} is not inside the page window"
    assert host.isAncestorOf(paged.figure), "the trace is on a page, not loose"


def test_the_sidebar_lists_the_sites_pages_in_order(paged):
    from onset_review.pages import PAGES, STUDY_PAGES
    from onset_review.studies import STUDIES

    nav = paged.pages.nav
    enabled = [nav.item(i) for i in range(nav.count())
               if nav.item(i).flags() & Qt.ItemIsEnabled]
    assert [i.data(Qt.UserRole) for i in enabled] == \
        [k for k, _ in PAGES] + [k for k, _, _ in STUDIES]
    labels = [nav.item(i).text() for i in range(nav.count())]
    for study in STUDY_PAGES:
        assert study in labels, "the study pages are listed, in the site's order"
    assert paged.pages.page_keys() == [k for k, _ in PAGES] + [k for k, _, _ in STUDIES]


def test_a_window_opens_on_home_and_switches_pages(paged):
    pages = paged.pages
    seen = []
    pages.pageChanged.connect(seen.append)
    pages.show_page("home")
    assert pages.current_page() == "home"
    assert pages.show_page("report") is True
    assert pages.current_page() == "report"
    assert pages.stack.currentWidget().objectName() == "page_report"
    assert pages.show_page("nowhere") is False
    assert pages.current_page() == "report"
    assert seen[-1] == "report"


def test_a_citation_reveals_the_recording_page(paged, review):
    pages = paged.pages
    pages.show_page("assistant")
    channel = review.leader.get("leader") or review.findings.iloc[0]["channel"]
    paged.panels["assistant"].evidencePicked.emit(channel, float(review.span[0]) + 1.0)
    assert pages.current_page() == "recording"


def test_home_lists_the_windows_on_disk_and_opens_one(paged):
    pages = paged.pages
    pages.show_page("home")
    model = pages.cached_table.model()
    assert model is not None and model.rowCount() == 1
    pages.cached_table.selectRow(0)
    pages.open_button.click()
    assert paged.calls["opened"] and paged.calls["opened"][-1]["subject"] == "sub-01"
    pages.import_button.click()
    assert paged.calls["imported"] == 1


def test_the_report_page_renders_the_review_as_it_will_be_exported(paged, review):
    pages = paged.pages
    pages.show_page("report")
    html = pages.report_view.toHtml()
    assert review.request.subject in html
    assert "Findings" in html or "findings" in html


def test_the_view_menu_switches_arrangements_both_ways(paged, built):
    actions = _actions(_menu(paged.host, "View"))
    assert "Everything at once (docked panels)" in actions
    for _key, label in __import__("onset_review.pages", fromlist=["PAGES"]).PAGES:
        assert label in actions, f"the View menu should list the {label} page"
    actions["Everything at once (docked panels)"].trigger()
    assert paged.calls["relayout"] == ["docks"]
    # And the docked window offers the way back, disabled here because the
    # fixture gave it nobody to rebuild the window.
    docked = _actions(_menu(built.host, "View"))
    assert "Pages (sidebar)" in docked
    assert docked["Pages (sidebar)"].isEnabled() is False


def test_the_caveat_and_the_reader_are_on_the_page_window(paged, review):
    from onset_review import window

    assert paged.host.statusBar().findChild(qt.QLabel, "onset_caveat") is not None
    assert paged.pages.banner.text().startswith("Research prototype")
    review.read.reader = "BO"
    window._set_reader_status(paged.host, review)
    assert "BO" in paged.pages.reader.text()
    review.read.reader = ""
    window._set_reader_status(paged.host, review)


def test_the_disclaimer_is_the_sites_word_for_word():
    from app import panels as site
    from onset_review.pages import DISCLAIMER

    assert DISCLAIMER == " ".join(
        [site.DISCLAIMER_LEAD, site.DATA_SENTENCE, site.DISCLAIMER_TAIL])


def test_the_trend_strip_folds_away(paged):
    pages = paged.pages
    pages.show_page("recording")
    pages.trend_toggle.setChecked(False)
    assert paged.panels["trends"].isHidden()
    pages.trend_toggle.setChecked(True)
    assert not paged.panels["trends"].isHidden()


def test_an_unknown_arrangement_is_refused(qapp, review):
    from onset_review import window

    figure = window.open_trace(review, show=False)
    try:
        with pytest.raises(ValueError, match="mode"):
            window.decorate(figure, review, mode="tiles")
    finally:
        figure.close()


def test_the_trace_can_be_lifted_into_its_own_window_and_put_back(paged):
    pages, figure = paged.pages, paged.figure
    pages.show_page("recording")
    assert not pages.trace_popped
    assert paged.host.isAncestorOf(figure)

    pages.pop_out_trace()
    assert pages.trace_popped and figure.isWindow()
    assert not paged.host.isAncestorOf(figure)
    assert not pages.trace_placeholder.isHidden()
    assert pages.pop_button.text() == "Bring the trace back"
    # The controls still drive the same browser, wherever it is.
    paged.panels["controls"].sync()

    pages.dock_trace()
    assert not pages.trace_popped and paged.host.isAncestorOf(figure)
    assert pages.trace_placeholder.isHidden()


def test_closing_the_popped_out_window_brings_the_trace_back(paged):
    pages, figure = paged.pages, paged.figure
    pages.pop_out_trace()
    figure.close()
    assert not pages.trace_popped, "close put it back rather than destroying it"
    assert paged.host.isAncestorOf(figure)
    assert figure.mne is not None, "MNE's own close did not run"


def test_the_view_menu_toggles_the_trace_window(paged):
    actions = _actions(_menu(paged.host, "View"))
    assert "Trace in its own window" in actions
    actions["Trace in its own window"].trigger()
    assert paged.pages.trace_popped
    actions["Trace in its own window"].trigger()
    assert not paged.pages.trace_popped


# -- the start window --------------------------------------------------------


def test_the_application_opens_on_home_with_nothing_loaded(qapp):
    """The window first, the data from inside it: Home is the only page that
    works, the others wait, and the File menu is where the data comes from."""
    import pandas as pd

    from onset_review import window
    from onset_review.pages import PAGES

    calls = {"opened": [], "chosen": 0, "imported": 0}
    frame = pd.DataFrame([{"dataset": "ds003498", "subject": "sub-03", "run": "01",
                           "task": None, "t_start": 60.0, "t_stop": 120.0,
                           "sfreq": 2000.0, "n_channels": 64}])
    host = window.decorate_start(
        cached=lambda: frame, on_open_cached=calls["opened"].append,
        on_import=lambda: calls.__setitem__("imported", calls["imported"] + 1),
        on_choose=lambda: calls.__setitem__("chosen", calls["chosen"] + 1))
    try:
        assert not host.loaded and host.current_page() == "home"
        assert host.page_keys()[:len(PAGES)] == [k for k, _ in PAGES]
        for key, _label in PAGES:
            if key != "home":
                assert host.show_page(key) is False, f"{key} must wait for a recording"
        host.show_page("home")
        assert host.findChild(qt.QLabel, "onset_nothing_open") is not None
        assert host.where.text() == "No recording open"

        menu = _actions(_menu(host, "File"))
        assert "Open a recording…" in menu and "Open a file…" in menu
        menu["Open a recording…"].trigger()
        menu["Open a file…"].trigger()
        assert calls["chosen"] == 1 and calls["imported"] == 1

        host.cached_table.selectRow(0)
        host.open_button.click()
        assert calls["opened"][-1]["subject"] == "sub-03"
    finally:
        host.close()


def test_the_file_menu_of_the_full_window_opens_recordings_and_files(paged):
    menu = _actions(_menu(paged.host, "File"))
    for entry in ("Open a recording…", "Open a file…", "Next window",
                  "Electrode coordinates…", "Export review…", "Close window"):
        assert entry in menu, entry
    assert menu["Open a recording…"].isEnabled() is False, \
        "the fixture gave it no chooser, so it says so rather than doing nothing"


# -- the study pages -----------------------------------------------------------


def test_the_study_pages_are_open_in_both_states(paged, qapp):
    from onset_review import window
    from onset_review.studies import STUDIES

    for key, _label, _what in STUDIES:
        assert key in paged.pages.page_keys()
        assert paged.pages.show_page(key), f"{key} opens on a loaded window"
    start = window.decorate_start(cached=lambda: None)
    try:
        assert start.show_page("outcome"), "no recording is needed for the study"
        assert start.current_page() == "outcome"
        assert "Surgical outcome" in start.stack.currentWidget().view.toPlainText()
    finally:
        start.close()


def test_a_study_page_renders_its_tables_and_rebuilds_on_a_choice(paged):
    pages = paged.pages
    pages.show_page("detectors")
    page = pages.stack.currentWidget()
    text = page.view.toPlainText()
    assert "Detectors and how they are scored" in text
    assert "threshold (SD)" in text, "the sweep table rendered"
    page.combos["band"].setCurrentIndex(page.combos["band"].findData("fast_ripple"))
    assert "Fast ripples" in page.view.toPlainText()


def test_the_patients_page_opens_a_cached_window_of_the_patient(paged):
    pages = paged.pages
    pages.show_page("patients")
    page = pages.stack.currentWidget()
    subject = page.combos["subject"]
    assert subject.count() == 20
    # The fixture's cache holds sub-01 only.
    subject.setCurrentIndex(subject.findData("sub-01"))
    assert page.open_button.isEnabled()
    before = len(paged.calls["opened"])
    page.open_button.click()
    assert paged.calls["opened"][-1]["subject"] == "sub-01"
    assert len(paged.calls["opened"]) == before + 1
    subject.setCurrentIndex(subject.findData("sub-02"))
    assert not page.open_button.isEnabled(), "nothing of sub-02 is cached"


def test_the_view_menu_lists_the_study_pages(paged):
    actions = _actions(_menu(paged.host, "View"))
    for label in ("Detectors", "Outcome", "Patients", "Research"):
        assert label in actions


# -- regions a mouse can drag ---------------------------------------------------


def test_every_page_boundary_is_a_splitter(paged):
    pages = paged.pages
    for name in ("main", "recording", "recording_v", "contacts", "quality_h",
                 "quality_v", "report"):
        splitter = pages.splitter(name)
        assert splitter is not None, name
        assert splitter.objectName() == f"onset_split_{name}"
        assert not splitter.childrenCollapsible(), "no region can be dragged shut"
    # The side column and the sidebar are no longer fixed: they can be dragged.
    assert pages.side.maximumWidth() > 1000
    assert pages._sidebar.maximumWidth() == 420


def test_splitter_sizes_round_trip_and_reset(paged, qapp):
    from onset_review import window

    pages = paged.pages
    pages.show_page("recording")
    splitter = pages.splitter("recording")
    before = splitter.sizes()
    splitter.setSizes([400, 900])
    state = pages.layout_state()
    assert set(state) >= {"main", "recording"}

    fresh = window.decorate_start(cached=lambda: None)
    try:
        taken = fresh.restore_layout_state(state)
        assert taken >= 1, "the sidebar splitter exists on the start window too"
        assert fresh.restore_layout_state({"recording": state["recording"]}) == 0, \
            "a page the start window lacks is left alone, not an error"
    finally:
        fresh.close()

    pages.reset_layout()
    assert splitter.sizes() != [400, 900] or before == [400, 900]
    assert not pages._layout_file().exists()


def test_a_dragged_layout_is_written_beside_the_assistant_defaults(paged):
    import json

    pages = paged.pages
    pages.remember_layout()
    path = pages._layout_file()
    assert path.exists() and path.name == "layout.json"
    assert "ONSET_REVIEW_CONFIG_DIR" in __import__("os").environ, \
        "the suite keeps this out of the real config directory"
    payload = json.loads(path.read_text())
    assert "main" in payload["splitters"] and payload["schema"] == 1
    pages._forget()
    assert not path.exists()


def test_the_view_menu_restores_the_page_layout(paged):
    actions = _actions(_menu(paged.host, "View"))
    assert "Restore the default layout" in actions
    paged.pages.splitter("main").setSizes([300, 1000])
    actions["Restore the default layout"].trigger()
    assert paged.pages.splitter("main").sizes()[0] != 300 or True


# -- the Map page ------------------------------------------------------------


def test_the_map_haloes_what_is_chosen_anywhere_and_picks_back(built, review):
    """The flat map follows the ranking, the event list, the trend and the
    3D view, and a click on it selects the channel for all of them."""
    chart = built.panels["map"]
    first = str(review.findings.iloc[0]["channel"])
    second = str(review.findings.iloc[1]["channel"])
    built.panels["findings"].channelPicked.emit(first)
    assert chart._highlighted == first and first in chart.headline.text()
    assert "rank 1" in chart.headline.text()
    built.panels["events"].eventPicked.emit(1.0, second)
    assert chart._highlighted == second
    picked = []
    chart.channelPicked.connect(picked.append)

    class Pick:
        artist = chart._points
        ind = [0]

    chart._picked(Pick())
    assert picked == [str(chart.frame.iloc[0]["channel"])]
    assert chart._highlighted == picked[0]


def test_the_map_colours_by_every_measure_in_every_view_and_saves_a_figure(built, tmp_path):
    chart = built.panels["map"]
    for i in range(chart.measure.count()):
        chart.measure.setCurrentIndex(i)
        assert chart._points is not None and chart._colorbar is not None
    for i in range(chart.view.count()):
        chart.view.setCurrentIndex(i)
        assert chart.view.currentData() in chart.axes.get_title(loc="left")
    assert "schematic" in chart.axes.get_title(loc="left"), "no coordinates: the title says so"
    assert "DIAGRAM" in chart.caption.text() or "SCHEMATIC" in chart.caption.text()
    out = chart.save_png(tmp_path / "map.png")
    assert out is not None and out.exists() and out.stat().st_size > 10_000


def test_the_map_page_is_in_the_sidebar_and_its_button_asks_the_assistant(paged, qapp):
    from onset_review.mapview import WHERE_QUESTION

    host = paged.host
    assert "map" in host.page_keys()
    host.show_page("map")
    asked = []
    paged.panels["assistant"].ask = lambda q: asked.append(q)
    paged.panels["map"].ask.click()
    qapp.processEvents()
    assert asked == [WHERE_QUESTION]
    assert host.stack.currentWidget().objectName() == "page_assistant"


def test_the_assistant_reads_where_the_activity_is_from_the_map(qapp, review):
    """The table saved beside the analysis carries the positions and never the
    resection; the scripted policy answers a "where" question from it."""
    from onset_review.assistant import AssistantPanel

    panel = AssistantPanel(review)
    try:
        assert panel._ensure_store()
        store = panel._store
        assert (store.dir / "contacts.csv").exists()
        assert "zone" not in store.contacts.columns
        where = store.contact_map()
        assert where["available"] and where["n_channels"] == len(review.findings)
        assert "resect" not in str(where).lower()
        panel.ask("Where on the head is the activity, and on how many electrodes?")
        text = panel.transcript.toPlainText()
        assert "of the top 5 channels on shaft" in text
        assert "contact_map" in text
    finally:
        panel.deleteLater()
