"""Finding your way: the quick start guide, a search over every command, and
what a page is for before there is anything on it.

A new reader should be able to see everything the program does from the
first screen, without loading anything and without knowing which menu hides
what. So:

* **Quick start** (sidebar, Help → Quick start guide, F1) lays the program out
  as the jobs people come with: look at a recording, work up a patient, read
  the evidence, ask the model, write your own analysis. Each step is a link
  that does the thing. Under that is every menu entry with its shortcut,
  read from the menus themselves, so the list cannot fall behind them.
* **Find a command** (Ctrl+K, Help → Find a command…) searches the menus and
  the pages by any word and runs what is chosen, from the keyboard.
* **No page is greyed out.** Before a recording is open, a page that needs one
  says what it shows, with a picture of it, and offers the ways to open one.
  A greyed entry tells a reader only that something is not for them.
"""

from __future__ import annotations

import html
from pathlib import Path

from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["add_help_entries", "add_size_menu", "interface_scale", "set_interface_scale",
           "SCALES", "QuickStartPage", "CommandPalette", "PreviewPage", "CasePage", "commands",
           "quickstart_html", "CASE_STEPS", "PREVIEW_IMAGES", "FIND_SHORTCUT"]

FIND_SHORTCUT = "Ctrl+K"

#: What each step of a case does, in one line: the Patient case page and the guide.
CASE_STEPS = {
    "import": "convert the clinical system's files (EDF, Nihon Kohden, Micromed, …) into "
              "BIDS-iEEG, leaving identifying header fields behind",
    "channels": "say which channels are intracranial, which are bad, and where the "
                "contacts are",
    "annotate": "mark seizures, sleep and artefacts over hours, in MNE's browser",
    "segments": "choose the interictal stretches to analyse, by rule",
    "preprocess": "filters, reference and the detectors' settings",
    "interictal": "HFO and spike rates per channel, pooled over the segments with intervals",
    "ictal": "the Epileptogenicity Index for every marked seizure, and which channels lead "
             "consistently",
    "review": "read the events and the ranking, and record your verdicts",
    "map": "contacts on a template brain or the patient's own MRI, with the onset zone "
           "you mark",
    "report": "a numbered, signed PDF once the checks pass",
}

#: A picture of each page, from docs/images, shown before a recording is open.
PREVIEW_IMAGES = {
    "recording": "onset-review.png", "analysis": "onset-review-analysis.png",
    "contacts": "onset-review-3d.png", "map": "onset-review-map.png",
    "quality": "onset-review-quality.png", "report": "onset-review-report.png",
    "assistant": "onset-review-assistant-answer.png", "case": "onset-review-case.png",
}


def _image(name: str) -> Path | None:
    try:
        from onset_agent.knowledge import docs_dir

        folder = docs_dir()
    except Exception:       # noqa: BLE001 - a picture is never worth a failure
        folder = None
    path = folder / "images" / name if folder is not None else None
    return path if path is not None and path.exists() else None


# -- every command ----------------------------------------------------------------------------
def commands(window) -> list[dict]:
    """Every menu entry of `window` and every page in its sidebar: where it
    is, what it says it does, its shortcut, whether it can run now, and the
    callable that runs it."""
    found: list[dict] = []

    def walk(menu, path: list[str]) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = action.text().replace("&", "")
            if action.menu() is not None:
                walk(action.menu(), path + [text])
                continue
            if not text:
                continue
            found.append({"where": " → ".join(path), "text": text,
                          "about": action.toolTip() if action.toolTip() != text else "",
                          "shortcut": action.shortcut().toString(QKeySequence.NativeText),
                          "enabled": action.isEnabled(), "run": action.trigger,
                          "kind": "menu"})

    bar = window.menuBar() if hasattr(window, "menuBar") else None
    if bar is not None:
        for action in bar.actions():
            if action.menu() is not None:
                walk(action.menu(), [action.text().replace("&", "")])
    labels = getattr(window, "_labels", {})
    items = getattr(window, "_items", {})
    for key, item in items.items():
        found.append({"where": "Pages", "text": labels.get(key, key),
                      "about": item.toolTip(), "shortcut": "", "enabled": True,
                      "run": (lambda key=key: window.show_page(key)), "kind": "page"})
    return found


def _matches(entry: dict, words: list[str]) -> bool:
    text = f"{entry['where']} {entry['text']} {entry['about']}".lower()
    return all(word in text for word in words)


class CommandPalette(QDialog):
    """Type any word; Enter runs the highlighted command."""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.setWindowTitle("Find a command")
        self.setObjectName("onset_palette")
        self.resize(620, 460)
        self.window_ = window
        self.entries = commands(window)
        self.search = QLineEdit()
        self.search.setObjectName("onset_palette_search")
        self.search.setPlaceholderText("Type what you want to do: open, export, map, "
                                       "case, outcome, shortcut…")
        self.search.setAccessibleName("Search the commands")
        self.search.setClearButtonEnabled(True)
        self.list = QListWidget()
        self.list.setObjectName("onset_palette_list")
        self.list.setAccessibleName("Matching commands")
        self.list.itemActivated.connect(lambda _item: self.run_current())
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self.run_current)
        box = QVBoxLayout(self)
        box.addWidget(self.search)
        box.addWidget(self.list, 1)
        box.addWidget(theme.muted("Enter runs the highlighted command; ↑ and ↓ move. "
                                  "Every menu entry and every page is here.", size=9))
        self.search.installEventFilter(self)
        self.filter("")

    def eventFilter(self, watched, event):      # noqa: N802 - Qt's name
        from qtpy.QtCore import QEvent

        if watched is self.search and event.type() == QEvent.KeyPress \
                and event.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self.list.currentRow() + (1 if event.key() == Qt.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
            return True
        return super().eventFilter(watched, event)

    def filter(self, text: str) -> int:
        words = [w for w in text.lower().split() if w]
        self.list.clear()
        for entry in self.entries:
            if words and not _matches(entry, words):
                continue
            label = f"{entry['text']}    ·  {entry['where']}"
            if entry["shortcut"]:
                label += f"  ·  {entry['shortcut']}"
            if not entry["enabled"]:
                label += "  ·  not available here"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, entry)
            if entry["about"]:
                item.setToolTip(entry["about"])
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        return self.list.count()

    def run_current(self) -> bool:
        item = self.list.currentItem()
        if item is None:
            return False
        entry = item.data(Qt.UserRole)
        if not entry["enabled"]:
            return False
        self.accept()
        entry["run"]()
        return True


# -- the quick start guide --------------------------------------------------------------------
def _link(target: str, text: str) -> str:
    return f"<a href='onset:{target}'>{html.escape(text)}</a>"


def quickstart_html(window) -> str:
    from onset_hfo.case.model import STEPS

    tokens = theme.current()
    muted = tokens.text_muted
    steps = "".join(f"<li><b>{html.escape(title)}</b> — {html.escape(CASE_STEPS.get(key, ''))}"
                    "</li>" for key, title, _phase in STEPS)
    parts = [f"""
<p>Onset Review is a research tool for intracranial EEG: high-frequency
oscillations, interictal discharges and seizure onset, in public recordings or
your own. It is <b>not a medical device</b>. Every page that needs a
recording works the same way: open one, and it fills in.</p>
<p style='color:{muted}'>Every underlined line below does what it says.
{_link('do:find', f'Find any command ({FIND_SHORTCUT})')} searches all of them by
any word.</p>

<h3>1 · Look at a recording <span style='color:{muted}'>(five minutes)</span></h3>
<ol>
<li>{_link('page:home', 'Home')} lists the windows already on this machine;
select one and press <b>Open</b>. Or {_link('do:open-recording', 'open a recording')}
to choose the band and the detectors as well, or
{_link('do:open-file', 'open a file of your own')} (EDF, BrainVision, FIF, …).</li>
<li>{_link('page:recording', 'Recording')} ranks the channels by event rate, both
detectors side by side. Click a channel for its events; click an event for its
raw, filtered and time-frequency views.</li>
<li>Judge events with the keys shown on the page; your verdicts go into the
{_link('page:report', 'Report')}.</li>
<li>{_link('page:quality', 'Signal')} shows what was analysed and how;
{_link('page:contacts', 'Contacts')} and {_link('page:map', 'Map')} show where.</li>
</ol>

<h3>2 · Work up a patient, step by step</h3>
<p>A <b>case</b> holds one patient's recordings and everything done with them, in
its own window: {_link('do:new-case', 'start a new case')} or
{_link('do:open-case', 'open one')}. The steps, in order:</p>
<ol>{steps}</ol>
<p>{_link('page:case', 'Patient case')} in the sidebar says the same, with the buttons.</p>

<h3>3 · Read the evidence behind the numbers</h3>
<p>The study's pages need no recording:
{_link('page:detectors', 'Detectors')} (against expert markings),
{_link('page:outcome', 'Outcome')} (against surgery),
{_link('page:patients', 'Patients')}, {_link('page:ictal', 'Ictal onset')},
{_link('page:template', 'Template map')} and {_link('page:data', 'Data')}.
Each opens with its main result and says what it cannot support.</p>

<h3>4 · Ask the local model</h3>
<p>{_link('page:assistant', 'Assistant')} answers about the open recording with
its tools and checks every number it quotes; {_link('page:chat', 'Chat')} is the
model on its own. Both run on this machine; nothing leaves it.</p>

<h3>5 · Your own analysis</h3>
<p>{_link('page:analysis', 'Analysis')} is Python on the open recording, as in
Spyder: an editor, a console that shares the Workspace, and plots.
{_link('do:batch', 'Analyse many recordings')} runs the same settings over a
list into one table.</p>

<h3>Seeing and reading</h3>
<ul>
<li><b>View → Interface size</b> makes everything larger (applies on restart);
the sidebar can be hidden from <b>View</b>.</li>
<li>Every page is reachable from the keyboard: <b>Alt+1</b> … <b>Alt+9</b> for the
first nine, {FIND_SHORTCUT} for anything, <b>F1</b> for this page.</li>
<li>Hover over any page in the sidebar for what it is for.</li>
</ul>
"""]
    entries = [e for e in commands(window) if e["kind"] == "menu"]
    if entries:
        rows = "".join(
            f"<tr><td style='color:{muted};padding-right:12px'>{html.escape(e['where'])}</td>"
            f"<td style='padding-right:12px'>{html.escape(e['text'])}</td>"
            f"<td style='color:{muted};padding-right:12px'>{html.escape(e['shortcut'])}</td>"
            f"<td>{html.escape(e['about'])}</td></tr>" for e in entries)
        parts.append(f"<h3>Every menu entry</h3><table cellspacing='0' cellpadding='3'>"
                     f"{rows}</table>")
    return "".join(parts)


class QuickStartPage(QWidget):
    """The guide, in a reader whose links do what they say."""

    linkRun = Signal(str)

    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window_ = window
        self.browser = QTextBrowser()
        self.browser.setObjectName("onset_quickstart")
        self.browser.setAccessibleName("Quick start guide")
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.anchorClicked.connect(self._clicked)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.browser)
        self.refresh()

    def refresh(self) -> None:
        self.browser.setHtml(quickstart_html(self.window_))

    def showEvent(self, event):     # noqa: N802 - Qt's name
        self.refresh()          # the menus may have changed since it was built
        super().showEvent(event)

    def _clicked(self, url: QUrl) -> None:
        self.run(url.toString().removeprefix("onset:"))

    def run(self, target: str) -> bool:
        self.linkRun.emit(target)
        return run_target(self.window_, target)


def run_target(window, target: str) -> bool:
    """Do what a guide link names: ``page:KEY`` or ``do:ACTION``."""
    kind, _, what = target.partition(":")
    if kind == "page":
        return bool(window.show_page(what))
    handlers = {
        "find": lambda: open_palette(window),
        "open-recording": getattr(window, "on_choose", None),
        "open-file": getattr(window, "_on_import", None),
        "new-case": getattr(window, "on_new_case", None),
        "open-case": getattr(window, "on_open_case", None),
        "batch": getattr(window, "on_batch", None),
    }
    handler = handlers.get(what)
    if callable(handler):
        handler()
        return True
    from qtpy.QtWidgets import QMessageBox

    QMessageBox.information(window, "Not here", "That is not available in this window. "
                            f"{FIND_SHORTCUT} lists what is.")
    return False


def open_palette(window, show: bool = True) -> CommandPalette:
    palette = CommandPalette(window)
    if show:
        palette.show()
        palette.search.setFocus()
    return palette


# -- a page before there is anything on it ------------------------------------------------------
class PreviewPage(QWidget):
    """What a page shows, a picture of it, and the ways to fill it."""

    def __init__(self, window, key: str, label: str, about: str, parent=None):
        super().__init__(parent)
        self.key = key
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        lead = QLabel(html.escape(about[:1].upper() + about[1:]) + "." if about else "")
        lead.setWordWrap(True)
        lead.setObjectName("onset_preview_about")
        box.addWidget(lead)
        need = QLabel("<b>This page fills in once a recording is open.</b> Nothing is "
                      "greyed out here: open one and come back, or look around first.")
        need.setWordWrap(True)
        need.setStyleSheet(theme.card("info"))
        need.setObjectName("onset_preview_needs")
        box.addWidget(need)
        row = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for target, text in (("page:home", "Pick one on Home"),
                             ("do:open-recording", "Open a recording…"),
                             ("do:open-file", "Open a file of your own…"),
                             ("page:quickstart", "Quick start guide")):
            button = QPushButton(text)
            button.setObjectName(f"onset_preview_{target.split(':')[1]}")
            button.clicked.connect(lambda _=False, t=target: run_target(window, t))
            row.addWidget(button)
            self.buttons[target] = button
        self.buttons["page:home"].setProperty("primary", True)
        row.addStretch(1)
        box.addLayout(row)
        picture = _image(PREVIEW_IMAGES.get(key, ""))
        self.picture = QLabel()
        self.picture.setObjectName("onset_preview_picture")
        if picture is not None:
            from qtpy.QtGui import QPixmap

            pixmap = QPixmap(str(picture))
            if not pixmap.isNull():
                self.picture.setPixmap(pixmap.scaledToWidth(min(900, pixmap.width()),
                                                            Qt.SmoothTransformation))
                self.picture.setAccessibleName(f"A picture of the {label} page with a "
                                               "recording open")
                box.addWidget(theme.muted("With a recording open, it looks like this:",
                                          size=9))
        box.addWidget(self.picture, 1, Qt.AlignTop | Qt.AlignLeft)
        box.addStretch(0)


class CasePage(QWidget):
    """A patient case from the main window: what it is, its steps, and the doors in."""

    def __init__(self, window, parent=None):
        super().__init__(parent)
        from onset_hfo.case.model import STEPS

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        lead = QLabel(
            "A case is one patient's monitoring, end to end: their recordings converted "
            "from the clinical system, and every step done with them, in its own window "
            "with an audit log. Use it for your own patients' data; the rest of this "
            "window is for looking at one recording at a time.")
        lead.setWordWrap(True)
        box.addWidget(lead)
        row = QHBoxLayout()
        self.new_button = QPushButton("New case…")
        self.new_button.setObjectName("onset_case_new")
        self.new_button.setProperty("primary", True)
        self.new_button.clicked.connect(lambda _=False: run_target(window, "do:new-case"))
        self.open_button = QPushButton("Open a case…")
        self.open_button.setObjectName("onset_case_open")
        self.open_button.clicked.connect(lambda _=False: run_target(window, "do:open-case"))
        row.addWidget(self.new_button)
        row.addWidget(self.open_button)
        row.addStretch(1)
        box.addLayout(row)
        steps = "".join(f"<li><b>{html.escape(title)}</b> — "
                        f"{html.escape(CASE_STEPS.get(key, ''))}</li>"
                        for key, title, _phase in STEPS)
        listing = QLabel(f"<p><b>The steps, in order</b></p><ol>{steps}</ol>")
        listing.setWordWrap(True)
        listing.setObjectName("onset_case_steps")
        listing.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(listing)
        picture = _image(PREVIEW_IMAGES["case"])
        if picture is not None:
            from qtpy.QtGui import QPixmap

            pixmap = QPixmap(str(picture))
            if not pixmap.isNull():
                image = QLabel()
                image.setPixmap(pixmap.scaledToWidth(min(820, pixmap.width()),
                                                     Qt.SmoothTransformation))
                image.setAccessibleName("A picture of a case window")
                box.addWidget(image, 0, Qt.AlignLeft)
        box.addStretch(1)


# -- the menus' share ---------------------------------------------------------------------------
def add_help_entries(help_menu, host) -> None:
    """Quick start guide (F1) and Find a command (Ctrl+K), first in Help."""
    first = help_menu.actions()[0] if help_menu.actions() else None
    guide = help_menu.addAction("Quick start guide")
    guide.setShortcut("F1")
    guide.setToolTip("Everything the program does, and every menu entry")
    guide.triggered.connect(lambda _=False: show_quickstart(host))
    find = help_menu.addAction("Find a command…")
    find.setShortcut(FIND_SHORTCUT)
    find.setToolTip("Search every menu entry and page by any word, and run it")
    find.triggered.connect(lambda _=False: open_palette(host))
    if first is not None:
        for action in (guide, find):
            help_menu.removeAction(action)
            help_menu.insertAction(first, action)
        help_menu.insertSeparator(first)


def show_quickstart(host):
    """The guide's page in a window with pages; the guide in a dialog otherwise."""
    if hasattr(host, "show_page") and host.show_page("quickstart"):
        return host
    dialog = QDialog(host)
    dialog.setWindowTitle("Quick start guide")
    dialog.resize(820, 680)
    box = QVBoxLayout(dialog)
    box.addWidget(QuickStartPage(host))
    dialog.show()
    return dialog


#: Interface sizes offered under View; applied through Qt's scale factor at
#: the next start, so text, icons and spacing grow together.
SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)


def _scale_file():
    from onset_review.assistant_config import config_path

    return config_path().with_name("interface.json")


def interface_scale() -> float:
    import json

    try:
        value = float(json.loads(_scale_file().read_text(encoding="utf-8"))["scale"])
        return value if 0.5 <= value <= 3.0 else 1.0
    except (OSError, ValueError, TypeError, KeyError):
        return 1.0


def set_interface_scale(value: float) -> None:
    import json

    path = _scale_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scale": float(value)}) + "\n", encoding="utf-8")


def apply_interface_scale(env=None) -> float:
    """Before the application starts: the remembered size, unless the
    environment already sets one. Returns the factor in force."""
    import os

    env = os.environ if env is None else env
    if env.get("QT_SCALE_FACTOR"):
        return float(env["QT_SCALE_FACTOR"])
    scale = interface_scale()
    if scale != 1.0:
        env["QT_SCALE_FACTOR"] = f"{scale:g}"
    return scale


def add_size_menu(view_menu, host) -> None:
    from qtpy.QtWidgets import QActionGroup, QMessageBox

    menu = view_menu.addMenu("Interface &size")
    menu.setToolTip("Larger text, icons and spacing everywhere; applies when the "
                    "program next starts")
    group = QActionGroup(menu)
    current = interface_scale()
    for scale in SCALES:
        action = menu.addAction(f"{scale:.0%}")
        action.setCheckable(True)
        action.setChecked(abs(scale - current) < 1e-6)
        group.addAction(action)

        def chosen(_=False, scale=scale):
            set_interface_scale(scale)
            QMessageBox.information(host, "Interface size",
                                    f"The interface will be {scale:.0%} of its usual size "
                                    "when Onset Review next starts.")
        action.triggered.connect(chosen)
