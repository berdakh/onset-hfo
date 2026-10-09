"""The Analysis page's template library: a script for each analysis the window does.

What has to hold: every template has the header the library reads and a known
group; every one runs to the end against an analysed session's console
names; the reproduction recomputes the window's ranking from the recording
and gets the same rates, order, leader and tied set; the anatomy templates
use electrode positions and a resected zone when the session has them, and
say so when it does not; and the editor offers every template in its menu
and its browser, opening each in a new unsaved tab.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from onset_review import codetemplates  # noqa: E402
from onset_review.console import namespace  # noqa: E402
from onset_review.session import ReviewRequest, session_from_recording  # noqa: E402

TEMPLATES = codetemplates.templates()


@pytest.fixture(scope="module")
def session(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


def _run(template, session, tmp_path, monkeypatch) -> tuple[dict, str]:
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    names = namespace(session)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(template.text, str(template.path), "exec"), names)
    plt.close("all")
    return names, out.getvalue()


def test_every_template_has_its_header_and_a_known_group():
    assert len(TEMPLATES) >= 12
    for template in TEMPLATES:
        assert template.group in codetemplates.GROUPS, template.key
        assert template.title and template.about and template.mirrors, template.key
        assert "# %%" in template.text, f"{template.key} runs a cell at a time"
    titles = [t.title for t in TEMPLATES]
    assert len(set(titles)) == len(titles)
    assert codetemplates.find("b10_reproduce_ranking").title == \
        "Reproduce the ranking, step by step"
    assert [g for g, _ in codetemplates.grouped()][0] == "Start here"


@pytest.mark.parametrize("key", [t.key for t in TEMPLATES])
def test_every_template_runs_on_an_analysed_session(key, session, tmp_path, monkeypatch):
    _names, out = _run(codetemplates.find(key), session, tmp_path, monkeypatch)
    assert out.strip(), f"{key} says what it found"


def test_the_reproduction_gets_the_windows_numbers(session, tmp_path, monkeypatch):
    names, out = _run(codetemplates.find("b10_reproduce_ranking"), session, tmp_path,
                      monkeypatch)
    assert names["same_rates"] and names["same_order"]
    assert names["my_leader"]["leader"] == session.leader["leader"]
    assert list(names["my_tied"]) == list(session.candidates)
    assert "rates match the window: True" in out


def test_quality_and_the_interval_by_hand_match_the_window(session, tmp_path, monkeypatch):
    _names, out = _run(codetemplates.find("c20_signal_quality"), session, tmp_path, monkeypatch)
    assert "channel verdicts match: True" in out
    names, _ = _run(codetemplates.find("b11_rates_and_intervals"), session, tmp_path, monkeypatch)
    row = names["row"]
    assert names["low"] == pytest.approx(row["rate_ci_low"])
    assert names["high"] == pytest.approx(row["rate_ci_high"])


def test_export_writes_the_tables(session, tmp_path, monkeypatch):
    names, _ = _run(codetemplates.find("g60_export"), session, tmp_path, monkeypatch)
    written = {p.name for p in names["OUT"].iterdir()}
    assert {"ranking.csv", "events.csv", "steps.txt"} <= written
    assert len(pd.read_csv(names["OUT"] / "ranking.csv")) == len(session.findings)


def test_the_anatomy_templates_use_positions_and_a_resection_when_there_are_some(
        session, tmp_path, monkeypatch):
    from onset_hfo.clinical import Resection

    contacts = sorted({c for ch in session.raw.ch_names for c in ch.split("-")})
    rng = np.random.default_rng(0)
    electrodes = pd.DataFrame({"name": contacts, "x": rng.uniform(-60, 60, len(contacts)),
                               "y": rng.uniform(-60, 60, len(contacts)),
                               "z": rng.uniform(-40, 60, len(contacts))})
    busiest = session.findings["channel"].iloc[0].split("-")
    known = dataclasses.replace(
        session, electrodes=electrodes,
        resection=Resection(subject="sub-01", resected=tuple(busiest), eloquent=()))
    names, out = _run(codetemplates.find("f50_contacts"), known, tmp_path, monkeypatch)
    assert "no electrode positions" not in out and len(names["where"]) == len(session.findings)
    names, out = _run(codetemplates.find("f51_resection"), known, tmp_path, monkeypatch)
    assert "busiest channel" in out and "is resected" in out
    assert names["share"]["resected"] > 0
    _names, out = _run(codetemplates.find("f51_resection"), session, tmp_path, monkeypatch)
    assert "No resected zone is known" in out


# -- the editor ------------------------------------------------------------------------------
@pytest.fixture
def editor(tmp_path, monkeypatch):
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    widgets.QApplication.instance() or widgets.QApplication([])
    from onset_review.editor import EditorPanel

    panel = EditorPanel()
    yield panel
    panel.deleteLater()


def test_the_editor_offers_every_template_and_opens_it_unsaved(editor):
    menu = editor.buttons["templates"].menu()
    offered = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert offered[0] == "Browse all templates…"
    for template in TEMPLATES:
        assert template.title in offered, template.title
    action = next(a for a in menu.actions() if a.text() == TEMPLATES[1].title)
    action.trigger()
    opened = editor.current()
    assert opened.toPlainText() == TEMPLATES[1].text
    assert opened.path is None and not opened.modified()
    assert editor.tabs.tabText(editor.tabs.currentIndex()).startswith(TEMPLATES[1].title)
    assert "a template" in editor.where.text()


def test_the_template_browser_shows_and_opens(editor):
    from qtpy.QtCore import Qt

    dialog = editor.browse_templates(show=False)
    keys = [dialog.list.item(i).data(Qt.UserRole) for i in range(dialog.list.count())]
    assert [k for k in keys if k] == [t.key for t in TEMPLATES]
    assert dialog.preview.toPlainText() == TEMPLATES[0].text
    target = keys.index("e40_ictal_index")
    dialog.list.setCurrentRow(target)
    assert "Epileptogenicity" in dialog.about.text()
    dialog.open_current()
    assert editor.current().toPlainText() == codetemplates.find("e40_ictal_index").text
    assert editor.tabs.count() == 1, "the untouched starter tab makes way"
    editor.open_template("b10_reproduce_ranking")
    assert editor.tabs.count() == 2, "a second template opens beside the first"


def test_the_ictal_template_explains_a_window_too_short_for_its_baseline(
        session, tmp_path, monkeypatch):
    names, out = _run(codetemplates.find("e40_ictal_index"), session, tmp_path, monkeypatch)
    if session.raw.times[-1] < 60:
        assert names["table"] is None and "needs 30 s of baseline" in out
    else:
        assert names["table"] is not None
