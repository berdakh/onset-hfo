"""Finding your way: nothing greyed out, a quick start guide whose links do
what they say, and a search over every command.

What has to hold: before anything is open every page in the sidebar can be
reached, a page that needs a recording says what it shows and offers the
ways to open one; the guide's links open pages and run the File menu's
handlers; its list of menu entries is read from the menus, so a new entry
appears without anyone editing the guide; the command search finds an entry
by any word in it or in its tooltip and runs it, and will not run what is
disabled; F1 and Ctrl+K are each bound once; the patient case page opens the
case dialogs; the interface size is remembered and applied before Qt starts
unless the environment says otherwise; and muted text keeps WCAG AA contrast.
"""

from __future__ import annotations

import os

import pytest

widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import Qt  # noqa: E402

from onset_review import guide  # noqa: E402


@pytest.fixture
def qapp():
    yield widgets.QApplication.instance() or widgets.QApplication([])


@pytest.fixture
def start(qapp, tmp_path, monkeypatch):
    from onset_review import window

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    calls = {"chosen": 0, "imported": 0, "new_case": 0, "open_case": 0, "batch": 0}
    host = window.decorate_start(
        cached=lambda: None,
        on_import=lambda: calls.__setitem__("imported", calls["imported"] + 1),
        on_choose=lambda: calls.__setitem__("chosen", calls["chosen"] + 1))
    host.on_new_case = lambda: calls.__setitem__("new_case", calls["new_case"] + 1)
    host.on_open_case = lambda: calls.__setitem__("open_case", calls["open_case"] + 1)
    host.on_batch = lambda: calls.__setitem__("batch", calls["batch"] + 1)
    host.calls = calls
    yield host
    host.close()


def test_every_page_can_be_looked_at_before_anything_is_open(start):
    from onset_review.pages import GUIDE_PAGES, PAGES

    for key, _label in PAGES + GUIDE_PAGES:
        assert start.show_page(key), key
        assert start.current_page() == key
    start.show_page("map")
    page = start._pages["map"]
    assert "fills in once a recording is open" in page.findChild(
        widgets.QLabel, "onset_preview_needs").text()
    page.buttons["do:open-recording"].click()
    page.buttons["do:open-file"].click()
    assert start.calls["chosen"] == 1 and start.calls["imported"] == 1
    page.buttons["page:quickstart"].click()
    assert start.current_page() == "quickstart"
    heading_colours = {start.nav.item(i).foreground().color().name()
                       for i in range(start.nav.count())
                       if not start.nav.item(i).data(Qt.UserRole)}
    from onset_review import theme

    assert heading_colours == {theme.current().text_muted.lower()}


def test_the_guides_links_do_what_they_say(start):
    assert start.show_page("quickstart")
    page = start.quickstart
    text = page.browser.toPlainText()
    for words in ("Look at a recording", "Work up a patient", "Read the evidence",
                  "Ask the local model", "Your own analysis", "Every menu entry",
                  "Ictal onset", "Open a file…"):
        assert words in text, words
    assert page.run("page:detectors") and start.current_page() == "detectors"
    for target, counter in (("do:open-recording", "chosen"), ("do:open-file", "imported"),
                            ("do:new-case", "new_case"), ("do:open-case", "open_case"),
                            ("do:batch", "batch")):
        assert page.run(target)
        assert start.calls[counter] == 1, target


def test_the_menu_list_is_read_from_the_menus(start):
    help_menu = next(a.menu() for a in start.menuBar().actions()
                     if a.text().replace("&", "") == "Help")
    help_menu.addAction("A brand-new entry")
    start.show_page("home")
    start.show_page("quickstart")
    assert "A brand-new entry" in start.quickstart.browser.toPlainText()


def test_find_a_command_by_any_word_and_run_it(start):
    palette = guide.open_palette(start, show=False)
    assert palette.filter("case new") >= 1
    entry = palette.list.currentItem().data(Qt.UserRole)
    assert entry["text"] == "New case…" and entry["where"] == "File"
    assert palette.run_current() and start.calls["new_case"] == 1
    palette = guide.open_palette(start, show=False)
    assert palette.filter("removal cured") >= 1, "a page found by the words of its tooltip"
    assert palette.list.currentItem().data(Qt.UserRole)["text"] == "Outcome"
    palette.run_current()
    assert start.current_page() == "outcome"
    assert guide.open_palette(start, show=False).filter("zzz nothing") == 0


def test_a_disabled_command_is_listed_but_not_run(qapp, tmp_path, monkeypatch):
    from onset_review import window

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "c"))
    host = window.decorate_start(cached=lambda: None)          # no chooser given
    try:
        palette = guide.open_palette(host, show=False)
        palette.filter("open a recording")
        item = palette.list.currentItem()
        assert "not available here" in item.text()
        assert palette.run_current() is False
    finally:
        host.close()


def test_f1_and_ctrl_k_are_bound_once(start):
    bound: dict[str, int] = {}
    for entry in guide.commands(start):
        if entry["shortcut"]:
            bound[entry["shortcut"]] = bound.get(entry["shortcut"], 0) + 1
    assert bound.get("F1") == 1 and bound.get("Ctrl+K") == 1
    duplicated = {key: n for key, n in bound.items() if n > 1}
    assert not duplicated, f"ambiguous shortcuts: {duplicated}"


def test_the_case_page_opens_the_case_dialogs(start):
    start.show_page("case")
    page = start.case_page
    assert "Ictal onset" in page.findChild(widgets.QLabel, "onset_case_steps").text()
    page.new_button.click()
    page.open_button.click()
    assert start.calls["new_case"] == 1 and start.calls["open_case"] == 1


def test_the_interface_size_is_remembered_and_applied_before_qt(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path))
    assert guide.interface_scale() == 1.0
    env: dict[str, str] = {}
    assert guide.apply_interface_scale(env) == 1.0 and "QT_SCALE_FACTOR" not in env
    guide.set_interface_scale(1.5)
    assert guide.interface_scale() == 1.5
    assert guide.apply_interface_scale(env) == 1.5 and env["QT_SCALE_FACTOR"] == "1.5"
    assert guide.apply_interface_scale({"QT_SCALE_FACTOR": "2"}) == 2.0, \
        "the environment wins"


def _contrast(foreground: str, background: str) -> float:
    def luminance(colour: str) -> float:
        channels = [int(colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("name", ["LIGHT", "DARK"])
def test_text_keeps_wcag_aa_contrast_on_every_background(name):
    from onset_review import theme

    palette = getattr(theme, name)
    for background in (palette.window, palette.surface, palette.sidebar):
        for foreground in (palette.text, palette.text_muted):
            assert _contrast(foreground, background) >= 4.5, (name, foreground, background)
