"""One window: a short sidebar of places, the case held in it, Analysis mode.

The redesign's first step. What has to hold:

* the sidebar is the places, not the pages: Patient, Review, Assistant and
  Report, the Library at its foot, and Analysis only in Analysis mode; the
  pages inside a place are a segmented control above the page, and every
  page is still reached by its key;
* Analysis mode is off until asked for: no Analysis place, no Workspace or
  Files beside the pages; asking for the Analysis page turns it on, turning
  it off takes the panes away, and the choice is remembered;
* a patient's case opens on Patient → Case in this window rather than as a
  window of its own, survives the window being rebuilt by a re-analysis, and
  a recording opened from it goes to Review with the case still held.
"""

from __future__ import annotations

import os

import pytest

from onset_review.session import ReviewRequest, session_from_recording

pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets

    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ONSET_REVIEW_READS", str(tmp_path / "reads"))
    return tmp_path


@pytest.fixture
def start(qapp, config):
    from onset_review import window

    host = window.decorate_start(cached=lambda: None)
    host.resize(1280, 820)
    yield host
    host.close()


def _settle(app, n=10):
    from qtpy.QtCore import QThread

    for _ in range(n):
        app.processEvents()
        QThread.msleep(15)


def _case(tmp_path, name="P017"):
    from onset_hfo.case.model import Case

    return Case.create(tmp_path / f"case-{name}", name)


# -- the sidebar ------------------------------------------------------------------------------
def test_the_sidebar_is_places_and_a_place_s_pages_are_its_segments(start):
    from qtpy.QtCore import Qt

    places = [start.nav.item(i).data(Qt.UserRole) for i in range(start.nav.count())
              if start.nav.item(i).data(Qt.UserRole)]
    assert places == ["patient", "review", "ask", "report", "library", "code"]
    assert start.current_page() == "home" and start.title.text() == "Patient"
    assert list(start.segment_buttons) == ["home", "case"]

    assert start.show_page("quality")
    assert start.nav.currentItem().data(Qt.UserRole) == "review"
    assert start.title.text() == "Review"
    assert list(start.segment_buttons) == ["recording", "quality", "contacts", "map"]
    assert start.segment_buttons["quality"].isChecked()
    assert not start.segment_buttons["recording"].isChecked()
    start.segment_buttons["contacts"].click()
    assert start.current_page() == "contacts"

    assert start.show_page("home")
    from qtpy.QtWidgets import QPushButton

    shown = [b.text() for b in start.segment_bar.findChildren(QPushButton)]
    assert shown == ["Overview", "Case"], "nothing left over from the place before"
    assert start.show_page("report") and start.segment_bar.isHidden(), \
        "a place with one page has no segments"
    assert start.show_page("chat") and start.title.text() == "Assistant"
    assert list(start.segment_buttons) == ["assistant", "chat"]


def test_places_come_back_where_they_were_left_and_step_by_keyboard(start):
    start.show_page("map")
    start.show_page("home")
    assert start.show_section("review") and start.current_page() == "map"
    assert start.step_segment(+1) and start.current_page() == "recording", "wraps round"
    assert start.step_segment(-1) and start.current_page() == "map"
    start.show_page("report")
    assert not start.step_segment(+1), "nothing to step to in a one-page place"
    assert start.show_section("patient") and start.current_page() == "home"
    assert not start.show_section("nowhere")


def test_every_page_has_a_name_the_command_search_finds(start):
    from onset_review import guide

    entries = {e["key"]: e for e in start.page_entries()}
    assert set(entries) == set(start.page_keys())
    assert entries["quality"]["label"] == "Review › Signal"
    assert entries["report"]["label"] == "Report"
    assert entries["outcome"]["label"] == "Library › Outcome"
    from qtpy.QtCore import Qt

    palette = guide.open_palette(start, show=False)
    assert palette.filter("review signal") >= 1
    found = [palette.list.item(i).data(Qt.UserRole) for i in range(palette.list.count())]
    signal = next(e for e in found if e["text"] == "Review › Signal")
    signal["run"]()
    assert start.current_page() == "quality"


# -- Analysis mode ----------------------------------------------------------------------------
def test_analysis_mode_is_off_until_asked_for_and_remembered(qapp, start):
    from onset_review.pages import PageWindow

    assert not start.analysis_mode and start._items["code"].isHidden()
    assert not any(dock.isVisible() for dock in start.docks.values()), \
        "no Workspace or Files beside the pages until asked for"
    start.show()
    _settle(qapp)
    assert start.show_page("analysis"), "asking for the page is asking for the mode"
    assert start.analysis_mode and not start._items["code"].isHidden()
    assert start.current_page() == "analysis" and start.title.text() == "Analysis"

    again = PageWindow(cached=lambda: None)
    try:
        assert again.analysis_mode, "remembered"
    finally:
        again.close()

    start.set_analysis_mode(False)
    assert start.current_page() == "home", "leaving the mode leaves its page"
    assert start._items["code"].isHidden()
    assert not any(dock.isVisible() for dock in start.docks.values())
    start.set_analysis_mode(True)
    assert start.docks["workspace"].isVisible() and start.docks["files"].isVisible()
    start.reset_layout()
    assert not start.analysis_mode, "the default layout is without it"


def test_the_view_menu_turns_analysis_mode_on_and_off(start):
    from onset_review.pages import ANALYSIS_MODE_SHORTCUT

    view = next(a.menu() for a in start.menuBar().actions()
                if a.text().replace("&", "") == "View")
    mode = next(a for a in view.actions() if a.text().replace("&", "") == "Analysis mode")
    assert mode.isCheckable() and not mode.isChecked()
    assert mode.shortcut().toString() == ANALYSIS_MODE_SHORTCUT
    mode.trigger()
    assert start.analysis_mode
    mode.trigger()
    assert not start.analysis_mode


# -- the case, held in the window -------------------------------------------------------------
def test_a_case_is_held_on_patient_case_not_in_a_window_of_its_own(qapp, start, tmp_path):
    from onset_review.casewindow import CaseWindow

    start.show()
    case = CaseWindow(_case(tmp_path), reader="dr test")
    start.hold_case(case)
    _settle(qapp)
    assert start.current_page() == "case" and start.held_case is case
    assert not case.isWindow() and case.window() is start
    assert case.isVisible() and start.case_page.views.currentWidget() is start.case_page.held
    assert case.show_step("annotate") and case.current_step() == "annotate", \
        "its steps work as they did in a window of their own"

    other = CaseWindow(_case(tmp_path, "P018"), reader="dr test")
    start.hold_case(other)
    assert start.held_case is other, "one case at a time; the one before is closed"

    start.case_page.close_button.click()
    assert start.held_case is None
    assert start.case_page.views.currentWidget() is start.case_page.front


def test_a_case_moves_into_the_next_window_with_nothing_lost(qapp, start, tmp_path):
    from onset_review import window
    from onset_review.casewindow import CaseWindow

    case = CaseWindow(_case(tmp_path), reader="dr test")
    start.hold_case(case)
    case.show_step("segments")
    after = window.decorate_start(cached=lambda: None)
    try:
        after.hold_case(start.held_case, show=False)
        assert start.held_case is None, "the window it left no longer holds it"
        start.close()
        _settle(qapp)
        assert after.held_case is case and case.window() is after
        assert case.current_step() == "segments"
    finally:
        after.close()


def test_the_app_holds_a_new_case_and_carries_it_through_a_recording(qapp, recording,
                                                                      config, tmp_path,
                                                                      monkeypatch):
    from onset_review import launcher
    from onset_review.app import _Review, build_parser

    def load(request, cache_dir=None, parent=None):
        return session_from_recording(recording, request)

    monkeypatch.setattr(launcher, "load_with_progress", load)
    review = _Review(qapp, None, False, build_parser().parse_args([]))
    review.start()
    _settle(qapp)
    held = review.new_case(_case(tmp_path))
    try:
        assert review.start_window.held_case is held
        assert review.start_window.current_page() == "case"
        assert not getattr(review, "case_windows", []), "no window of its own"

        request = ReviewRequest(dataset="", subject="synthetic", t_start=0.0,
                                t_stop=float(recording.duration), detectors=("rms",))
        held.openRequested.emit(request)
        _settle(qapp)
        pages = review.parts.pages
        assert pages.held_case is held, "carried into the window the recording opened in"
        assert pages.current_page() == "recording", "opened from the case, shown in Review"

        pages.show_page("case")
        review._rerun(threshold_sd=5.0)
        _settle(qapp)
        assert review.parts.pages.held_case is held, "and through a re-analysis"
        assert review.parts.pages.current_page() == "case"
    finally:
        review.parts.panels["console"].clear_user_variables()
        review.parts.host.close()
        _settle(qapp)
