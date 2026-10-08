"""The Console pane: Python in the window's memory, as Spyder's.

What has to hold: the names are the window's own objects, not copies; what
the console makes shows in the Workspace; the working directory and the
Files pane follow each other; a figure opens in a window; every command is
kept with the session and printed in the report; and the console, with what
was made in it, survives a re-analysis.
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


def _settle(app, n=10):
    for _ in range(n):
        app.processEvents()
        QThread.msleep(20)


@pytest.fixture
def review(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


@pytest.fixture
def window(qapp, review, tmp_path, monkeypatch):
    import pandas as pd

    from onset_review import window as window_module

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    here = os.getcwd()
    figure = window_module.open_trace(review, show=False)
    parts = window_module.decorate(figure, review, mode="pages",
                                   cached=lambda: pd.DataFrame())
    yield parts
    # One kernel per process: what one test made must not be in the next.
    parts.panels["console"].clear_user_variables()
    parts.panels["console"].shutdown()
    try:
        figure.close()
    except Exception:
        pass
    os.chdir(here)


def test_the_console_starts_on_use_and_names_the_window_s_own_objects(qapp, window):
    console = window.panels["console"]
    assert not console.started, "nothing is paid for until the pane is used"
    console.run("n_rows = len(findings)")
    _settle(qapp)
    assert console.started and console.kind == "ipython"
    ns = console.namespace
    assert ns["session"] is window.session
    assert ns["raw"] is window.session.raw, "the window's object, not a copy"
    assert ns["findings"] is window.session.findings
    assert ns["n_rows"] == len(window.session.findings)
    for module in ("np", "pd", "mne", "plt", "onset_hfo"):
        assert module in ns
    assert "Named here" in console.text()


def test_what_the_console_makes_is_listed_in_the_workspace(qapp, window):
    console, workspace = window.panels["console"], window.panels["workspace"]
    console.run("rates = findings.set_index('channel')['rate_per_min']\n"
                "busiest = rates.idxmax()")
    _settle(qapp)
    assert {"rates", "busiest"} <= set(workspace.names())
    console_group = [workspace.tree.topLevelItem(g)
                     for g in range(workspace.tree.topLevelItemCount())
                     if workspace.tree.topLevelItem(g).text(0) == "Console"][0]
    listed = {console_group.child(c).text(0) for c in range(console_group.childCount())}
    assert listed == {"rates", "busiest"}
    assert not listed & {"np", "pd", "raw", "findings", "session", "open", "In", "Out"}, \
        "modules, IPython's names and the window's own stay out"
    assert workspace.variable("rates").kind == "Series"
    window_ = workspace.open("busiest")
    assert window_ is not None
    workspace.close_windows()


def test_the_working_directory_and_the_files_pane_follow_each_other(
        qapp, window, tmp_path):
    console, files = window.panels["console"], window.panels["files"]
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    console.start()
    files.set_folder(first)
    assert os.getcwd() == str(first.resolve())
    console.run(f"import os\nos.chdir({str(second)!r})")
    _settle(qapp)
    assert files.folder == second.resolve(), "Files follows a %cd or os.chdir"


def test_a_figure_drawn_in_the_console_opens_in_a_window_of_its_own(qapp, window):
    console = window.panels["console"]
    console.run("fig, ax = plt.subplots()\nax.plot(raw.times[:200], raw.get_data()[0, :200])"
                "\nplt.show()")
    _settle(qapp)
    assert len(console.figure_windows) == 1
    shown = console.figure_windows[0]
    assert shown.isWindow() and "Console" in shown.windowTitle()
    import matplotlib.pyplot as plt

    assert shown.figure.number not in plt.get_fignums(), "handed over, not left in pyplot"
    console.close_figures()


def test_every_command_is_kept_with_the_session_and_printed_in_the_report(
        qapp, window):
    from onset_review.report import review_markdown

    console = window.panels["console"]
    console.run("x = 1")
    console.run("findings.loc[0, 'rate_per_min']")
    console.run("   ")
    _settle(qapp)
    assert window.session.console_log == ["x = 1", "findings.loc[0, 'rate_per_min']"]
    text = review_markdown(window.session)
    assert "## Python console" in text and "2 commands were run" in text
    assert "```python\nx = 1\nfindings.loc[0, 'rate_per_min']\n```" in text


def test_a_report_without_console_use_says_nothing_about_it(review):
    from onset_review.report import review_markdown

    assert "Python console" not in review_markdown(review)


def test_the_console_and_what_was_made_in_it_survive_a_reanalysis(
        qapp, window, recording):
    """A re-analysis rebuilds the window; the console moves into the new one,
    its names point at the new session, and the user's names stay."""
    import pandas as pd

    from onset_review import window as window_module

    console = window.panels["console"]
    console.run("mine = 42")
    _settle(qapp)
    second = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration) / 2))
    figure = window_module.open_trace(second, show=False)
    rebuilt = window_module.decorate(figure, second, mode="pages",
                                     cached=lambda: pd.DataFrame(), console=console)
    try:
        assert rebuilt.panels["console"] is console
        assert rebuilt.pages.isAncestorOf(console)
        assert console.namespace["session"] is second
        assert console.namespace["raw"] is second.raw
        assert console.namespace["mine"] == 42
        console.run("mine += 1")
        _settle(qapp)
        console.clear_user_variables()
        assert second.console_log == ["mine += 1"], "the new session's own log"
        assert "mine" in rebuilt.panels["workspace"].names()
    finally:
        figure.close()


def test_the_console_sits_under_the_workspace_closed_until_asked_for(window):
    """Spyder's arrangement: the console in the right column under the
    variables. Along the bottom it took the Recording page's height."""
    host = window.pages
    host.show()      # a window laid out on screen, as a reviewer's always is
    dock = host.pane("console")
    assert dock is not None and dock.isHidden()
    host.reset_layout()
    assert host.dockWidgetArea(dock) == Qt.RightDockWidgetArea and dock.isHidden()
    workspace, files = host.pane("workspace"), host.pane("files")
    for pane in (workspace, dock):
        pane.show()
    assert dock not in host.tabifiedDockWidgets(workspace), \
        "under it, not tabbed behind it: Qt did that when it was split after tabbing"
    files.show()
    assert host.tabifiedDockWidgets(workspace) == [files]
    workspace.hide()
    files.hide()
    workspace.show()
    assert dock not in host.tabifiedDockWidgets(workspace), "and stays there"
    from tests.test_review_app import _menu

    actions = {a.text().replace("&", ""): a for a in _menu(host, "View").actions()}
    assert actions["Console"].shortcut().toString() == "Ctrl+Shift+I"


def test_laying_out_the_panes_does_not_start_the_console(qapp, window):
    """Seen in the full suite: a layout reset shows every pane for a moment,
    and that moment started IPython and moved the working directory."""
    host, console = window.pages, window.panels["console"]
    here = os.getcwd()
    host.show()
    host.reset_layout()
    _settle(qapp)
    assert not console.started and os.getcwd() == here
    host.pane("console").show()
    _settle(qapp)
    assert console.started, "opening the pane does start it"


def test_without_qtconsole_the_pane_is_a_plain_console_that_says_what_to_install(
        qapp, monkeypatch):
    from onset_review import console as console_module

    def missing(*_args, **_kwargs):
        raise ImportError("No module named 'qtconsole'")

    monkeypatch.setattr(console_module, "_RichConsole", missing)
    console = console_module.ConsolePanel()
    console.run("z = 3 * 7")
    console.run("z")
    assert console.kind == "basic" and console.missing == ("qtconsole",)
    assert "pip install qtconsole" in console.text()
    assert console.namespace["z"] == 21 and "21" in console.text()
    console.run("for i in range(2):\n    print('line', i)")
    assert "line 1" in console.text()
    console.run("1/0")
    assert "ZeroDivisionError" in console.text()


def test_the_namespace_and_the_user_s_names_are_qt_free_functions(review):
    from onset_review.console import namespace, user_variables

    ns = namespace(review)
    injected = dict(ns)
    assert ns["findings"] is review.findings and ns["channels"] == list(review.raw.ch_names)
    ns.update({"mine": 1, "_hidden": 2, "In": [], "np": ns["np"],
               "helper": lambda: None, "Thing": type("Thing", (), {}), "maximum": max})
    ns["findings"] = "reassigned"
    assert user_variables(ns, injected) == {"mine": 1, "findings": "reassigned"}, \
        "functions and classes left out, as Spyder does"
    assert namespace(None)["session"] is None
