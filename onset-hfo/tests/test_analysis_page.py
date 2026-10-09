"""The Analysis page: Spyder's editor, workspace and console on one page.

Asked for: a page of its own, after Recording, where the console reaches the
data that is loaded; scripts that can be kept and run again; the workspace
saved and loaded; and the local model writing Python. What has to hold:

* every name the Workspace lists is the same object in the console, and
  `signal` is a read-only view of the samples, not a copy;
* the editor runs cells, selections and files in that namespace, and the
  report's console log keeps a file's text, not just its name;
* a notebook opens as cells and saves back as a notebook;
* the panes come onto the page and go back to their docks, as they were,
  and the page's arrangement is never saved as the docks';
* a model's draft opens in a tab of its own, says who wrote it and what
  could be checked, and nothing runs it;
* a saved workspace loads back, and never replaces the window's own names.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from onset_review import cells, codewriter, variables

# -- Qt-free --------------------------------------------------------------------


def test_cells_are_split_at_percent_markers_and_found_by_line():
    text = "import x\n# %% first\na = 1\n\n# %% [markdown] notes\n# words\n# %%\nb = 2\n"
    found = cells.split_cells(text)
    assert [(c.start, c.stop, c.title, c.markdown) for c in found] == [
        (0, 1, "", False), (1, 4, "first", False), (4, 6, "notes", True), (6, 9, "", False)]
    assert cells.cell_at(text, 2).title == "first"
    assert found[1].source(text.split("\n")) == "a = 1"
    assert cells.split_cells("x = 1\n")[0].source(["x = 1", ""]) == "x = 1"


def test_a_notebook_survives_the_round_trip_through_cells(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n", "\n", "Words."]},
            {"cell_type": "code", "metadata": {}, "execution_count": 3,
             "outputs": [{"output_type": "stream", "text": "old"}],
             "source": ["x = 1\n", "print(x)"]},
        ],
        "metadata": {"colab": {"name": "kept"}}, "nbformat": 4, "nbformat_minor": 5,
    }
    script = cells.notebook_to_script(notebook)
    assert script.startswith("# %% [markdown]\n# # Title\n#\n# Words.")
    path = cells.write_notebook(script, tmp_path / "out.ipynb", notebook)
    back = json.loads(path.read_text())
    assert [c["cell_type"] for c in back["cells"]] == ["markdown", "code"]
    assert "".join(back["cells"][0]["source"]) == "# Title\n\nWords."
    assert "".join(back["cells"][1]["source"]) == "x = 1\nprint(x)"
    assert back["cells"][1]["outputs"] == [], "outputs are the console's now, not the file's"
    assert back["metadata"]["colab"] == {"name": "kept"}


def test_the_project_s_notebooks_open_and_save_back_with_the_same_cells(tmp_path):
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "notebooks"
    for path in sorted(folder.glob("*.ipynb")):
        notebook = cells.read_notebook(path)
        again = cells.script_to_notebook(cells.notebook_to_script(notebook), notebook)
        before = [(c["cell_type"], "".join(c["source"]).strip())
                  for c in notebook["cells"] if c["cell_type"] in ("code", "markdown")]
        after = [(c["cell_type"], "".join(c["source"]).strip()) for c in again["cells"]]
        assert after == before, path.name


def test_the_model_is_told_names_types_and_columns_never_values():
    names = {"findings": pd.DataFrame({"channel": ["SECRET-A"], "rate": [123.456]}),
             "signal": np.full((2, 3), 7.25), "leader": {"leader": "SECRET-B"}, "sfreq": 2000.0,
             "np": np, "helper": lambda: None}
    listing = codewriter.inventory(names)
    assert "findings: DataFrame, 1 × 2; columns: channel" in listing
    assert "signal: ndarray float64, 2 × 3" in listing
    assert "leader: dict, 1; keys: 'leader'" in listing
    for value in ("SECRET-A", "SECRET-B", "123.456", "7.25", "2000"):
        assert value not in listing
    assert "helper" not in listing and "np:" not in listing
    sent = codewriter.messages("plot it", names, script="x = 1", history=[("a", "b")])
    assert sent[0]["content"] == codewriter.CODE_PROMPT
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "x = 1" in sent[-1]["content"] and sent[-1]["content"].endswith("REQUEST\nplot it")


def test_a_draft_is_taken_from_its_fence_and_checked_without_running():
    reply = "Sure.\n```python\nimport os\nimport scipy.signal as ss\nf, p = ss.welch(signal)\n" \
            "print(p, mystery)\nos.remove('x')\n```\nThat is all."
    code = codewriter.extract_code(reply)
    assert code.startswith("import os") and "That is all" not in code
    draft = codewriter.check_code(code, {"signal": 1})
    assert draft.parses and draft.unknown == ["mystery"]
    assert draft.risky == ["line 5 calls os.remove() — deletes a file"]
    assert codewriter.check_code("for x in y:\nprint(x)").parses is False
    assert codewriter.check_code("def f(a, *b, c=1, **d):\n    return a, b, c, d\n"
                                 "g = lambda q: q\n[i for i in range(3)]\n").unknown == []
    assert codewriter.extract_code("no code here") == ""
    assert codewriter.extract_code("x = 1") == "x = 1\n"
    assert "Not checked" in codewriter.header("qwen", "a request")


def test_the_console_s_names_are_the_session_s_own_objects(recording):
    from onset_review.session import ReviewRequest, session_from_recording

    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))
    names = variables.session_names(session)
    listed = {v.name for v in variables.variables(session)}
    assert set(names) == listed, "every Workspace name is a console name"
    for name in ("raw", "events", "findings", "quality", "segments", "read", "request"):
        assert names[name] is getattr(session, name), name
    assert np.shares_memory(names["signal"], session.raw._data), "a view, not a copy"
    assert not names["signal"].flags.writeable
    with pytest.raises(ValueError):
        names["signal"][0, 0] = 1.0
    np.testing.assert_allclose(names["times"][[0, -1]], session.raw.times[[0, -1]])


def test_a_workspace_is_saved_and_loaded_in_each_format(tmp_path):
    names = {"rates": np.arange(4.0), "n": 3, "label": "SA1", "table": pd.DataFrame(
        {"ch": ["A", "B"], "r": [1.0, 2.0]}), "f": lambda: 1, "chs": ["A", "B"]}
    path, saved, skipped = variables.save_workspace(names, tmp_path / "w.pkl")
    assert saved == ["rates", "n", "label", "table", "chs"] and set(skipped) == {"f"}
    back = variables.load_workspace(path)
    assert back["label"] == "SA1" and back["table"].equals(names["table"])

    _, saved, skipped = variables.save_workspace(names, tmp_path / "w.npz")
    assert saved == ["rates", "n"] and "save as .pkl" in skipped["table"]
    back = variables.load_workspace(tmp_path / "w.npz")
    assert back["n"] == 3 and np.array_equal(back["rates"], names["rates"])

    _, saved, skipped = variables.save_workspace(names, tmp_path / "w.mat")
    assert set(saved) == {"rates", "n", "label", "table", "chs"} and set(skipped) == {"f"}
    back = variables.load_workspace(tmp_path / "w.mat")
    assert back["label"] == "SA1" and list(back["table"]["r"]) == [1.0, 2.0]

    _, saved, _ = variables.save_workspace({"f": lambda: 1}, tmp_path / "none.npz")
    assert saved == [] and not (tmp_path / "none.npz").exists()
    with pytest.raises(ValueError):
        variables.save_workspace(names, tmp_path / "w.txt")


# -- in the window ---------------------------------------------------------------
# Skipped here, not at the top of the file, so the Qt-free tests above still
# run on the job that installs no `review` extra.


@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def _settle(app, n=10):
    from qtpy.QtCore import QThread

    for _ in range(n):
        app.processEvents()
        QThread.msleep(20)


@pytest.fixture
def window(qapp, recording, tmp_path, monkeypatch):
    from onset_review import window as window_module
    from onset_review.session import ReviewRequest, session_from_recording

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))
    figure = window_module.open_trace(session, show=False)
    parts = window_module.decorate(figure, session, mode="pages",
                                   cached=lambda: pd.DataFrame())
    monkeypatch.setattr(parts.pages, "_screen_width", lambda: 2560)
    parts.pages.resize(1680, 980)
    parts.pages.show()
    _settle(qapp)
    yield parts
    console = parts.panels["console"]
    console.clear_user_variables()
    parts.pages.close()
    console.shutdown()
    try:
        figure.close()
    except Exception:
        pass


def test_the_page_comes_after_recording_and_previews_without_one(qapp, window, tmp_path,
                                                              monkeypatch):
    from onset_review import window as window_module

    host = window.pages
    assert host.page_keys()[:3] == ["home", "recording", "analysis"]
    assert host.show_page("analysis") and host.current_page() == "analysis"
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "start"))
    start = window_module.decorate_start(cached=lambda: pd.DataFrame())
    try:
        from onset_review.guide import PreviewPage

        assert start.show_page("analysis"), "never greyed out"
        assert isinstance(start._pages["analysis"], PreviewPage), \
            "no recording: what the page is for, and how to open one"
        assert not start.panes_lent, "a preview borrows no panes"
    finally:
        start.close()


def test_the_panes_come_onto_the_page_and_go_back_as_they_were(qapp, window):
    host, docks, panels = window.pages, window.pages.docks, window.panels
    host.apply_pane_layout("Spyder")
    _settle(qapp)
    before = {k: d.toggleViewAction().isChecked() for k, d in docks.items()}
    assert before == {"workspace": True, "files": True, "console": True}
    host.show_page("analysis")
    _settle(qapp)
    assert host.panes_lent
    for key in ("workspace", "files", "console"):
        assert panels[key].parent() is host._slots[key], key
        assert not docks[key].isVisible()
    assert panels["workspace"].isVisible() and panels["console"].isVisible()
    assert panels["console"].started, "the page shows the console, so it starts"
    saved = json.loads(host._layout_file().read_text()) if host._layout_file().exists() else {}
    host.remember_layout()
    after = json.loads(host._layout_file().read_text())
    assert after.get("panes") == saved.get("panes"), "the lent arrangement is never saved"
    host.show_page("recording")
    _settle(qapp)
    assert not host.panes_lent
    assert {k: d.toggleViewAction().isChecked() for k, d in docks.items()} == before
    for key in ("workspace", "files", "console"):
        assert docks[key].widget() is panels[key], key


def test_a_layout_chosen_on_the_page_applies_to_the_docks_when_it_is_left(qapp, window):
    host, docks = window.pages, window.pages.docks
    host.apply_pane_layout("Page only")
    host.show_page("analysis")
    _settle(qapp)
    host.apply_pane_layout("MATLAB")
    _settle(qapp)
    assert host.panes_lent and host.current_page() == "analysis"
    host.show_page("home")
    _settle(qapp)
    assert all(d.toggleViewAction().isChecked() for d in docks.values())


def test_a_cell_runs_in_the_shared_namespace_and_shows_in_the_workspace(qapp, window):
    host, panels, session = window.pages, window.panels, window.session
    editor, console = panels["editor"], panels["console"]
    host.show_page("analysis")
    _settle(qapp)
    tab = editor.new_file("# %% one\npeak = float(np.max(np.abs(signal)))\n"
                          "# %% two\nn_events = len(events)\n")
    tab.go_to_line(1)
    assert editor.run_cell(advance=True)
    _settle(qapp)
    assert editor.run_cell()
    _settle(qapp)
    ns = console.namespace
    assert ns["peak"] == pytest.approx(float(np.max(np.abs(session.raw.get_data()))))
    assert ns["n_events"] == len(session.events)
    assert {"peak", "n_events"} <= set(panels["workspace"].names())
    assert session.console_log[-2:] == ["peak = float(np.max(np.abs(signal)))",
                                        "n_events = len(events)"]
    tab.setPlainText("    doubled = n_events * 2\n")
    tab.selectAll()
    assert editor.run_selection(), "an indented selection runs on its own"
    _settle(qapp)
    assert ns["doubled"] == 2 * len(session.events)


def test_a_file_runs_from_disk_and_the_report_keeps_its_text(qapp, window, tmp_path):
    from onset_review.report import review_markdown

    editor, console, session = window.panels["editor"], window.panels["console"], window.session
    window.pages.show_page("analysis")
    _settle(qapp)
    script = tmp_path / "rates.py"
    script.write_text("top_rate = float(findings['rate_per_min'].max())\n")
    tab = editor.open_file(script)
    tab.appendPlainText("print('top', top_rate)")
    assert tab.modified()
    assert editor.run_file()
    _settle(qapp)
    assert not tab.modified() and "print('top'" in script.read_text(), "saved before running"
    assert console.namespace["top_rate"] == pytest.approx(session.findings["rate_per_min"].max())
    assert session.console_log[-1].startswith(f"# Ran the file {script.resolve()}\n")
    assert "top_rate = float(" in session.console_log[-1]
    assert "top_rate = float(" in review_markdown(session)


def test_a_notebook_opens_as_cells_and_saves_back_as_a_notebook(qapp, window, tmp_path):
    editor = window.panels["editor"]
    source = tmp_path / "study.ipynb"
    source.write_text(json.dumps(cells.script_to_notebook("# %% [markdown]\n# Hi\n# %%\nx = 1\n")))
    tab = editor.open_file(source)
    assert tab.is_notebook and tab.toPlainText().startswith("# %% [markdown]\n# Hi")
    tab.appendPlainText("y = 2")
    assert editor.save() == source.resolve()
    saved = json.loads(source.read_text())
    assert "".join(saved["cells"][-1]["source"]).endswith("y = 2")


def test_a_draft_opens_in_its_own_tab_with_its_checks_and_nothing_runs_it(qapp, window):
    editor, console, session = window.panels["editor"], window.panels["console"], window.session
    asked = []

    def model(messages):
        asked.append(messages)
        return "```python\nspectrum = np.abs(np.fft.rfft(signal[0]))\nprint(nyquist)\n```"

    assistant = editor.assistant
    assistant.ask_model = model
    assistant.refresh_model()
    before = list(session.console_log)
    tabs = editor.tabs.count()
    tab = assistant.write("spectrum of the first channel")
    assert editor.tabs.count() == tabs + 1 and editor.current() is tab
    assert tab.title.startswith("Draft") and tab.path is None
    text = tab.toPlainText()
    assert text.startswith("# Drafted by the local model (test model) from: spectrum of the "
                           "first channel\n# Not checked")
    assert "# Check: Uses names that do not exist yet: nyquist" in text
    assert "nyquist" in assistant.notes.text()
    assert session.console_log == before and "spectrum" not in console.namespace, \
        "a draft is written, never run"
    sent = asked[-1][-1]["content"]
    assert "signal: ndarray" in sent and "findings: DataFrame" in sent

    console.run("1 / 0")
    _settle(qapp)
    assert "ZeroDivisionError" in console.last_error
    assistant.sync()
    assert assistant.fix_button.isEnabled()
    assistant.fix_last_error()
    assert "ZeroDivisionError" in asked[-1][-1]["content"]
    console.run("ok = 1")
    _settle(qapp)
    assert console.last_error == ""


def test_no_model_no_drafting(qapp, window):
    assistant = window.panels["editor"].assistant
    assistant.ask_model = None
    assistant.refresh_model()
    assert not assistant.write_button.isEnabled()
    assert "Assistant page" in assistant.model_line.text()
    assert assistant.write("anything") is None


def test_a_script_double_clicked_in_files_opens_on_the_page(qapp, window, tmp_path):
    host, files, editor = window.pages, window.panels["files"], window.panels["editor"]
    script = tmp_path / "mine.py"
    script.write_text("a = 1\n")
    files.set_folder(tmp_path)
    assert files.open_path(script)
    _settle(qapp)
    assert host.current_page() == "analysis" and editor.current().path == script.resolve()


def test_open_tabs_and_unsaved_text_are_kept_for_next_time(qapp, window, tmp_path):
    from onset_review.editor import EditorPanel

    editor = window.panels["editor"]
    script = tmp_path / "kept.py"
    script.write_text("kept = 1\n")
    editor.open_file(script)
    draft = editor.new_file("unsaved = 2\n", title="Draft 9")
    editor.remember()
    again = EditorPanel(restore=True)
    try:
        by_name = {e.name(): e for e in again.editors()}
        assert by_name["kept.py"].path == script.resolve()
        assert by_name["Draft 9"].toPlainText() == draft.toPlainText()
        assert by_name["Draft 9"].modified()
    finally:
        again.deleteLater()


def test_the_workspace_saves_and_loads_the_console_s_variables(qapp, window, tmp_path):
    workspace, console, session = (window.panels["workspace"], window.panels["console"],
                                   window.session)
    console.run("mine = np.arange(5)\nraw_copy_note = 'x'")
    _settle(qapp)
    saved, skipped = workspace.save_variables(tmp_path / "ws.npz")
    assert saved == ["mine"] and "raw_copy_note" in skipped
    assert "Saved 1 variable to ws.npz" in workspace.count.text()
    variables.save_workspace({"raw": 1, "loaded_one": 2.5}, tmp_path / "other.pkl")
    console.clear_user_variables()
    loaded = workspace.load_variables(tmp_path / "other.pkl")
    assert loaded == ["loaded_one", "raw_loaded"]
    assert console.namespace["raw"] is session.raw, "the window's raw is never replaced"
    assert console.namespace["raw_loaded"] == 1
    assert "loaded_one" in workspace.names()
    assert session.console_log[-1].startswith("# Loaded loaded_one, raw_loaded from ")


# -- Plots and History ------------------------------------------------------------------
def test_a_figure_drawn_on_the_page_goes_to_the_plots_pane_not_a_window(qapp, window,
                                                                        tmp_path):
    host, console = window.pages, window.panels["console"]
    host.show_page("analysis")
    _settle(qapp)
    tabs = host.analysis_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Workspace", "Files", "Plots", "History"]
    console.run("fig, ax = plt.subplots()\nax.plot(times[:100], signal[0, :100])")
    console.run("plt.figure()\nplt.plot([1, 2, 3])")
    _settle(qapp)
    plots = console.plots
    assert console.figure_windows == [], "on the page, no window of its own"
    assert tabs.currentWidget() is plots and plots.isVisible()
    assert len(plots.entries) == 2 and plots.current_index == 0
    assert "plt.figure()" in plots.entries[0]["command"]     # newest first
    assert not plots.view.pixmap().isNull()
    plots.step(1)
    assert plots.current_index == 1 and "2 of 2" in plots.where.text()
    saved = plots.save(tmp_path / "first.pdf")
    assert saved.read_bytes().startswith(b"%PDF")
    plots.open_current()
    assert len(console.figure_windows) == 1 and console.figure_windows[0].isVisible()
    console.close_figures()
    plots.remove_current()
    assert len(plots.entries) == 1 and plots.current_index == 0
    plots.clear()
    assert plots.entries == [] and plots.empty.isVisible()
    # Off the page, a figure still opens in a window, and is kept in the pane.
    host.show_page("recording")
    _settle(qapp)
    console.run("plt.figure()\nplt.plot([3, 2, 1])")
    _settle(qapp)
    assert len(console.figure_windows) == 1 and len(plots.entries) == 1
    console.close_figures()


def test_history_keeps_commands_across_sessions_and_sends_them_to_the_editor(
        qapp, window, tmp_path):
    from qtpy.QtGui import QTextCursor

    from onset_review.consolepanes import HistoryPanel, history_path, load_history

    host, console, editor = window.pages, window.panels["console"], window.panels["editor"]
    host.show_page("analysis")
    history = console.history
    history.clear(ask=False)
    console.run("alpha = 1")
    console.run("alpha = 1")                   # a repeat of the last: kept once
    console.run("   ")
    console.run("beta = alpha + 1\nprint(beta)")
    sources = [e["source"] for e in history.entries]
    assert sources == ["alpha = 1", "beta = alpha + 1\nprint(beta)"]
    assert [e["source"] for e in load_history(history_path())] == sources
    assert "(+1 lines)" in history.list.item(1).text()
    # A new pane (the next session) reads the same history back.
    assert [e["source"] for e in HistoryPanel().entries] == sources
    history.search.setText("beta")
    assert history.list.count() == 1
    history.search.setText("")
    history.list.selectAll()
    code = editor.new_file("# mine\n")
    cursor = code.textCursor()
    cursor.movePosition(QTextCursor.End)
    code.setTextCursor(cursor)
    assert history.to_editor()
    assert code.toPlainText() == "# mine\nalpha = 1\nbeta = alpha + 1\nprint(beta)\n"
    history.list.clearSelection()
    history.list.setCurrentRow(0)
    history.list.item(0).setSelected(True)
    console.namespace["alpha"] = 5
    assert history.run_again() and console.namespace["alpha"] == 1
    assert history.clear(ask=False) and load_history(history_path()) == []
