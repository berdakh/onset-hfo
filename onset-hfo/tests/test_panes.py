"""Pane layouts and the View menu entries that keep the screen uncluttered.

Asked for: options on the menu so the page is not too busy with windows.
The View menu toggles each pane, applies a layout in one step (Page only,
Spyder, MATLAB), closes every variable and figure window the panes
opened, and hides the page sidebar; the choices are kept for next time.
"""

from __future__ import annotations

import os

import pytest

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")

from qtpy.QtCore import Qt, QThread  # noqa: E402

from onset_review.session import ReviewRequest, session_from_recording  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield qt.QApplication.instance() or qt.QApplication([])


def _settle(app, n=8):
    for _ in range(n):
        app.processEvents()
        QThread.msleep(15)


@pytest.fixture
def window(qapp, recording, tmp_path, monkeypatch):
    import pandas as pd

    from onset_review import window as window_module

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))
    figure = window_module.open_trace(session, show=False)
    parts = window_module.decorate(figure, session, mode="pages",
                                   cached=lambda: pd.DataFrame())
    # A wide screen, so that a layout docks its panes rather than floating
    # them; the narrow case has a test of its own.
    monkeypatch.setattr(parts.pages, "_screen_width", lambda: 2560)
    parts.pages.show()
    _settle(qapp)
    yield parts
    parts.panels["workspace"].close_windows()
    parts.pages.close()
    try:
        figure.close()
    except Exception:
        pass


def _view(host) -> dict:
    for action in host.menuBar().actions():
        if action.text().replace("&", "") == "View":
            menu = action.menu()
            found = {a.text().replace("&", ""): a for a in menu.actions()}
            layouts = found.get("Pane layout")
            if layouts is not None and layouts.menu() is not None:
                found.update({a.text(): a for a in layouts.menu().actions()})
            return found
    raise AssertionError("no View menu")


def test_page_only_closes_every_pane_and_a_pane_shown_again_returns_to_its_place(
        qapp, window):
    host = window.pages
    shown = host.apply_pane_layout("Spyder")
    assert shown == ["workspace", "files", "console"]
    assert shown_now(host) == {"workspace", "files", "console"}
    assert host.apply_pane_layout("Page only") == []
    assert shown_now(host) == set()
    host.pane("console").show()
    assert host.dockWidgetArea(host.pane("console")) == Qt.RightDockWidgetArea


def shown_now(host) -> set:
    return {key for key in ("workspace", "files", "console") if not host.pane(key).isHidden()}


def test_the_spyder_layout_tabs_workspace_and_files_with_the_console_under_them(
        qapp, window):
    host = window.pages
    host.apply_pane_layout("Spyder")
    _settle(qapp)
    workspace, files, console = (host.pane(k) for k in ("workspace", "files", "console"))
    assert host.tabifiedDockWidgets(workspace) == [files]
    assert console not in host.tabifiedDockWidgets(workspace)
    assert all(host.dockWidgetArea(d) == Qt.RightDockWidgetArea
               for d in (workspace, files, console))
    assert console.geometry().top() > workspace.geometry().top(), "under, not beside"


def test_the_matlab_layout_puts_files_on_the_left(qapp, window):
    host = window.pages
    host.apply_pane_layout("MATLAB")
    _settle(qapp)
    assert host.dockWidgetArea(host.pane("files")) == Qt.LeftDockWidgetArea
    assert host.dockWidgetArea(host.pane("workspace")) == Qt.RightDockWidgetArea
    assert host.dockWidgetArea(host.pane("console")) == Qt.RightDockWidgetArea
    assert host.tabifiedDockWidgets(host.pane("workspace")) == []


def test_the_layout_is_remembered_for_the_next_window(qapp, window, recording):
    import pandas as pd

    from onset_review import window as window_module

    window.pages.apply_pane_layout("MATLAB")
    second_session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))
    figure = window_module.open_trace(second_session, show=False)
    again = window_module.decorate(figure, second_session, mode="pages",
                                   cached=lambda: pd.DataFrame())
    again.pages.show()
    try:
        assert again.pages.dockWidgetArea(again.pages.pane("files")) == Qt.LeftDockWidgetArea
    finally:
        again.pages.close()
        figure.close()


def test_the_view_menu_has_the_toggles_the_layouts_and_the_clean_ups(window):
    found = _view(window.pages)
    for name, shortcut in (("Workspace", "Ctrl+Shift+W"), ("Files", "Ctrl+Shift+F"),
                           ("Console", "Ctrl+Shift+I"), ("Page only", "Ctrl+Shift+P"),
                           ("Page sidebar", "Ctrl+Shift+B")):
        assert name in found, sorted(found)
        assert found[name].shortcut().toString() == shortcut
    for name in ("Spyder", "MATLAB", "Close variable and figure windows", "Full screen"):
        assert name in found, sorted(found)


def test_the_page_only_shortcut_entry_clears_the_screen(qapp, window):
    host = window.pages
    host.apply_pane_layout("Spyder")
    _view(host)["Page only"].trigger()
    assert shown_now(host) == set()


def test_the_sidebar_hides_comes_back_and_is_remembered(qapp, window, recording):
    import pandas as pd

    from onset_review import window as window_module

    host = window.pages
    side = _view(host)["Page sidebar"]
    assert side.isChecked() and host.sidebar_shown
    side.trigger()
    assert not host.sidebar_shown
    assert host.show_page("report"), "pages are still reachable without it"
    figure = window_module.open_trace(window.session, show=False)
    again = window_module.decorate(figure, window.session, mode="pages",
                                   cached=lambda: pd.DataFrame())
    try:
        assert not again.pages.sidebar_shown, "kept for the next window"
        again.pages.reset_layout()
        assert again.pages.sidebar_shown, "and Restore the default layout brings it back"
    finally:
        again.pages.close()
        figure.close()
    host.set_sidebar(True)


def test_one_entry_closes_every_variable_and_figure_window(qapp, window):
    host, workspace, console = window.pages, window.panels["workspace"], window.panels["console"]
    workspace.open("findings")
    workspace.open("request")
    console.run("fig, ax = plt.subplots()\nax.plot([1, 2, 3])")
    _settle(qapp)
    try:
        assert len(workspace.windows) == 2 and len(console.figure_windows) == 1
        _view(host)["Close variable and figure windows"].trigger()
        assert workspace.windows == [] and console.figure_windows == []
    finally:
        console.clear_user_variables()
        console.shutdown()


def test_on_a_narrow_screen_a_layout_opens_its_panes_as_windows(qapp, window, monkeypatch):
    host = window.pages
    monkeypatch.setattr(host, "_screen_width", lambda: 1366)
    shown = host.apply_pane_layout("Spyder")
    floated = [k for k in shown if host.pane(k).isFloating()]
    assert floated, "beside the page they would push the window past the screen"
    assert "too narrow" in host.statusBar().currentMessage()
    host.apply_pane_layout("Page only")
    assert not any(host.pane(k).isFloating() for k in ("workspace", "files", "console"))


def test_the_start_window_and_the_docked_window_have_their_entries(qapp, recording,
                                                                  tmp_path, monkeypatch):
    import pandas as pd

    from onset_review import window as window_module

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path))
    start = window_module.decorate_start(cached=lambda: pd.DataFrame())
    try:
        found = _view(start)
        for name in ("Workspace", "Files", "Console", "Page only", "Spyder", "MATLAB",
                     "Close variable and figure windows", "Page sidebar"):
            assert name in found, sorted(found)
    finally:
        start.close()
    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))
    figure = window_module.open_trace(session, show=False)
    docked = window_module.decorate(figure, session, mode="docks")
    try:
        found = _view(docked.host)
        assert "Close variable and figure windows" in found
        assert "Page only" not in found, "the docked window's task layouts already decide"
    finally:
        figure.close()


def test_a_layout_change_leaves_the_window_its_size(qapp, window, monkeypatch):
    """Arranging shows every pane for a moment; that once left a 1366 px
    window 1502 px wide, past the screen."""
    host = window.pages
    monkeypatch.setattr(host, "_screen_width", lambda: 1366)    # the laptop
    host.resize(1366, 768)
    _settle(qapp)
    width = host.width()
    host.apply_pane_layout("Page only")
    _settle(qapp)
    assert host.width() == width
    host.reset_layout()
    _settle(qapp)
    assert host.width() == width
