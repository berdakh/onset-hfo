"""The review window as pages: a sidebar on the left, one view at a time.

This is the shape of the project's Streamlit site (``app/``), carried onto the
desktop. The site has a page per question -- Recording, Report, Assistant, and
the study pages behind them -- and every page carries the same disclaimer
line. The window here has the same sidebar, the same names in the same order,
and the same line; what it adds is what a web page cannot hold: the live
trace, the 3D contacts, and the panels that re-run the analysis.

Nothing in here is a panel. The panels are the ones `window.build_panels`
makes for the docked window too, and the whole point of this module is that
the same widgets, wired the same way, can be laid out either as docks or as
pages. A reviewer who prefers the docks gets them back from the View menu.

The first group needs a recording and is disabled until one is open; three of
its pages (Contacts, Map, Signal) have no counterpart on the site because they
need a running analysis. The six study pages of the site follow, read-only,
built from the committed tables by `onset_review.studies`; they need no
recording, so they are open even before one is loaded. Each sidebar entry is
one word, and its tooltip is the line Home prints about the page.
"""

from __future__ import annotations

import contextlib
import html
import importlib.util

from qtpy.QtCore import QByteArray, QEvent, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["PageWindow", "PAGES", "STUDY_PAGES", "DISCLAIMER"]

#: (key, sidebar label). The order is the site's.
PAGES = (
    ("home", "Home"),
    ("recording", "Recording"),
    ("analysis", "Analysis"),
    ("contacts", "Contacts"),
    ("map", "Map"),
    ("quality", "Signal"),
    ("report", "Report"),
    ("assistant", "Assistant"),
)

#: Pages for finding your way, open in every state: the quick start guide
#: and a patient case's front door (`onset_review.guide`).
GUIDE_PAGES = (("quickstart", "Quick start"), ("case", "Patient case"))

#: The site's cohort pages, read-only, built from the committed tables by
#: `onset_review.studies` and shown by `onset_review.studypages`. They need
#: no recording, so they are open in the start state too.
STUDY_PAGES = ("Detectors", "Outcome", "Patients", "Data", "Architecture",
               "Research")

#: The site's disclaimer, word for word (`app/panels.py`). A test pins the
#: two copies to each other; it is duplicated rather than imported because the
#: wheel does not ship the Streamlit app.
DISCLAIMER = ("Research tool — not a medical device. "
              "Methods tested on real recordings with expert markings and "
              "surgical outcomes — and nothing here is validated for clinical use. "
              "Every number cites the window it came from. There is no "
              "recommendation anywhere in this product; the clinician decides.")

#: The one line of it that is always showing; the rest opens on a click.
DISCLAIMER_LINE = ("Research tool — not a medical device. Nothing here is "
                   "validated for clinical use; the clinician decides.")

#: The site's front door, trimmed to what applies on the desktop.
WHAT_THIS_IS = (
    "<b>What this is.</b> An AI-assisted research tool for intracranial EEG, "
    "for your own recordings as well as public ones. It detects high-frequency "
    "oscillations (ripples 80–250 Hz, fast ripples 250–500 Hz) and interictal "
    "discharges, finds where seizures start, places contacts on a template brain, "
    "and carries a patient's recordings from the clinical system's files to a "
    "signed report. Its methods are tested on public archives against expert "
    "markings, clinicians' onset zones and surgical outcomes. An assistant on a "
    "local open-weight model reads the results, must cite them, and is refused by "
    "code when it strays.")
WHAT_IT_IS_NOT = (
    "<b>What it is not.</b> Not a diagnostic device, not validated on patients, "
    "not a seizure-onset-zone finder. A high event rate is a measurement; "
    "physiological ripples occur in healthy tissue. This software organises "
    "evidence. The clinician decides.")

#: How to read the pages, each entry a link to its page.
HOW_TO_READ = (
    ("recording", "channel ranking with both detectors side by side, "
                  "disagreement highlighted, and the signal behind any event"),
    ("analysis", "your own Python on this recording, as in Spyder: scripts and notebooks "
                 "run a cell at a time in a console that shares the Workspace, and a "
                 "local model that drafts code for you to read before you run it"),
    ("contacts", "where the contacts are, ranked, relative to the resection"),
    ("map", "the same contacts flat, coloured by rate with a colour scale: the "
            "figure a paper prints, and the one the assistant can describe"),
    ("quality", "the signal before any detector sees it: which contacts and "
                "seconds were analysed, and what was done to it first"),
    ("report", "the structured, cited report: findings, your read, data "
               "quality, limitations"),
    ("assistant", "ask about a channel or the evidence; try asking what to "
                  "resect"),
    ("chat", "the same local model on its own: not connected to this recording, "
             "nothing checked, and the page says so"),
)

#: The page that needs no recording and belongs to no study: the model alone.
CHAT_PAGE = ("chat", "Chat")

SIDEBAR_WIDTH = 190
COLUMN_WIDTH = 320


def _cell(row: dict, key: str, fmt: str = "{}") -> str:
    """A cached-window field as the tree shows it; blank where there is none."""
    value = row.get(key)
    if value is None or str(value) in ("", "nan", "None"):
        return ""
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _recent_file():
    from onset_review.assistant_config import config_path

    return config_path().with_name("recent.json")


def remember_recent(request) -> None:
    """Note the window just opened, so the next launch can offer to continue
    with it. Subject and seconds only; nothing from the recording."""
    import json

    try:
        path = _recent_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"subject": str(request.subject),
                                    "t_start": float(request.t_start),
                                    "t_stop": float(request.t_stop)}) + "\n",
                        encoding="utf-8")
    except (OSError, TypeError, ValueError, AttributeError):
        pass


def recent_window() -> dict | None:
    """The window noted by `remember_recent`, or None."""
    import json

    try:
        payload = json.loads(_recent_file().read_text(encoding="utf-8"))
        return {"subject": str(payload["subject"]), "t_start": float(payload["t_start"]),
                "t_stop": float(payload["t_stop"])}
    except (OSError, ValueError, TypeError, KeyError):
        return None


class _CardPage(QWidget):
    """One widget on one card, answering for that widget's attributes."""

    def __init__(self, body: QWidget, parent=None):
        super().__init__(parent)
        self.body = body
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(theme.card_frame(body))

    def __getattr__(self, name: str):
        body = self.__dict__.get("body")
        if body is None or name.startswith("_"):
            raise AttributeError(name)
        return getattr(body, name)


class PageWindow(QMainWindow):
    """A sidebar of pages around the same panels the docked window uses."""

    #: The key of the page just shown.
    pageChanged = Signal(str)

    def __init__(self, figure=None, panels: dict | None = None, session=None, *,
                 cached=None, on_open_cached=None, on_import=None,
                 on_place_contacts=None, on_export=None, on_open_path=None,
                 parent=None):
        """With a session, the whole window. Without one, the start state:
        the same sidebar and Home page, the other pages disabled until a
        recording is opened from Home or from the File menu. The application
        opens on this and loads the data from inside it."""
        super().__init__(parent)
        self.figure = figure
        self.panels = panels or {}
        self.session = session
        self._cached = cached
        self._on_open_cached = on_open_cached
        self._on_import = on_import
        self._on_place_contacts = on_place_contacts
        self._on_export = on_export
        self._on_open_path = on_open_path
        if self.loaded:
            remember_recent(self.session.request)
        self._pages: dict[str, QWidget] = {}
        self._items: dict[str, QListWidgetItem] = {}
        #: name -> (splitter, default sizes). Every region boundary a mouse
        #: can drag, so a page can be arranged and the arrangement kept.
        self._splitters: dict[str, tuple[QSplitter, list[int]]] = {}

        self.nav = QListWidget()
        self.nav.setObjectName("onset_pages")
        self.nav.setFrameShape(QFrame.NoFrame)
        self.nav.setSpacing(1)
        self.stack = QStackedWidget()
        self.stack.setObjectName("onset_stack")

        # One quiet line rather than a box: the site's words, the first of
        # them always showing and the rest a click away. A banner that takes
        # two lines of every page is read once and then looked past, which is
        # the opposite of what a disclaimer is for.
        self.banner = QLabel()
        self.banner.setObjectName("onset_banner")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            f"color:{theme.current().warn};font-size:9pt;padding:2px 0 0 0;")
        self.banner.setOpenExternalLinks(False)
        self.banner.linkActivated.connect(lambda _link: self.toggle_disclaimer())
        self._disclaimer_open = False
        self._say_disclaimer()

        #: Per-page actions shown at the toolbar's right while that page is
        #: up: the trace's window button, the report's export.
        self._page_actions: dict[str, list[QWidget]] = {}
        self._build_sidebar()
        self._build_toolbar()
        self._build_pages()

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._toolbar)
        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(theme.PAGE_MARGIN, theme.SPACING, theme.PAGE_MARGIN,
                                 theme.CARD_GAP)
        inner.setSpacing(theme.SPACING)
        inner.addWidget(self.banner)
        inner.addWidget(self.stack, 1)
        column.addWidget(content, 1)

        body = self._split("main", Qt.Horizontal, [self._sidebar, right],
                           [SIDEBAR_WIDTH, 1400], stretch=(0, 1))
        self.setCentralWidget(body)
        # Placing the panes moves them, and a move saves the arrangement:
        # without this the default would be written over the remembered one
        # before it is read.
        self._building = True
        self._build_panes()

        self.nav.currentItemChanged.connect(self._nav_changed)
        self.restore_layout_state(self._remembered())
        self._restore_panes()
        if self._remembered_sidebar() is False:
            self._sidebar.setVisible(False)
        self._building = False
        if self._remembered_compact() is False:
            for key in self.COMPACT_PANELS:
                panel = self.panels.get(key)
                if panel is not None and hasattr(panel, "set_compact"):
                    panel.set_compact(False)
        from onset_review.studies import STUDIES

        keyed = [key for key, _label in PAGES] + [key for key, _, _ in STUDIES]
        for index, key in enumerate(keyed[:9], start=1):
            shortcut = QShortcut(QKeySequence(f"Alt+{index}"), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda key=key: self.show_page(key))
        # F1 and Ctrl+K belong to Help's entries (`guide.add_help_entries`):
        # a second shortcut on the same keys would make both ambiguous.
        self.nav.setAccessibleName("Pages")
        self.nav.setAccessibleDescription("Every page of the program; Alt+1 to Alt+9 for the "
                                          "first nine, Ctrl+K to search everything")
        self.show_page("home")

    # -- the sidebar ------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _build_toolbar(self) -> None:
        """The strip above the page: the page's name on the left, the
        recording in the middle, the page's actions and the reader on the
        right. Where a Mac window puts the document's name and its buttons;
        the sidebar is left to navigation."""
        tokens = theme.current()
        self.title = QLabel("")
        self.title.setObjectName("onset_page_title")
        self.title.setStyleSheet(f"color:{tokens.text};font-size:13pt;font-weight:600;"
                                 "background:transparent;border:none;")
        if self.loaded:
            request = self.session.request
            self.where = QLabel(f"<b>{request.subject}</b> · "
                                f"{request.t_start:g}–{request.t_stop:g} s · "
                                f"{request.band_label()}")
        else:
            self.where = QLabel("No recording open")
        self.where.setObjectName("onset_where")
        self.where.setAlignment(Qt.AlignCenter)
        self.where.setStyleSheet(f"color:{tokens.text};background:transparent;border:none;")
        self.reader = QLabel("No reader named")
        self.reader.setObjectName("onset_sidebar_reader")
        self.reader.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;"
                                  "background:transparent;border:none;")

        self._toolbar = QWidget()
        self._toolbar.setObjectName("onset_toolbar")
        self._toolbar.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(self._toolbar)
        row.setContentsMargins(theme.PAGE_MARGIN, theme.SPACING, theme.PAGE_MARGIN,
                               theme.SPACING)
        row.setSpacing(theme.SPACING)
        row.addWidget(self.title, 1)
        row.addWidget(self.where, 2)
        tail = QHBoxLayout()
        tail.setContentsMargins(0, 0, 0, 0)
        tail.setSpacing(theme.SPACING)
        tail.addStretch(1)
        self._actions_box = tail
        tail.addWidget(self.reader)
        row.addLayout(tail, 1)

    def _register_action(self, key: str, widget: QWidget) -> None:
        """Put `widget` in the toolbar, shown while page `key` is up."""
        widget.setVisible(False)
        index = self._actions_box.count() - 1      # before the reader
        self._actions_box.insertWidget(index, widget)
        self._page_actions.setdefault(key, []).append(widget)

    def _build_sidebar(self) -> None:
        from onset_review import glyphs

        tokens = theme.current()
        self._labels: dict[str, str] = {}
        self.nav.setIconSize(QSize(16, 16))

        def heading(text: str, fold: bool = False) -> QListWidgetItem:
            item = QListWidgetItem(text.upper())
            # A heading that folds is clickable, never selectable: a click
            # opens or closes its group and does not change the page.
            item.setFlags(Qt.ItemIsEnabled if fold else Qt.NoItemFlags)
            font = item.font()
            font.setPointSize(max(7, font.pointSize() - 2))
            font.setBold(True)
            item.setFont(font)
            # The theme's muted text, not Qt's grey: 2.2:1 on the light
            # sidebar, where WCAG asks 4.5:1 of text this small.
            item.setForeground(QColor(tokens.text_muted))
            self.nav.addItem(item)
            return item

        # What each page is for, as its tooltip: the sidebar is the table of
        # contents, and a reader who hovers gets the line Home prints.
        what = dict(HOW_TO_READ)
        guide_about = {
            "quickstart": "Everything this program does, as the jobs people come with, "
                          "and every menu entry (F1)",
            "case": "One patient's recordings worked up step by step, in a case window"}

        def entry(key: str, label: str, about: str) -> None:
            item = QListWidgetItem(glyphs.icon(key, tokens), label)
            item.setData(Qt.UserRole, key)
            self._labels[key] = label
            tip = about[0].upper() + about[1:] if about else ""
            if key in dict(PAGES) and key != "home" and not self.loaded:
                tip += (" — open a recording to fill it in" if tip
                        else "Open a recording to fill it in")
            item.setToolTip(tip)
            item.setData(Qt.AccessibleDescriptionRole, tip)
            self.nav.addItem(item)
            self._items[key] = item

        # Nothing is greyed out: a page that needs a recording shows what it
        # is for and how to open one, so the sidebar is the whole program
        # from the first screen.
        heading("Start")
        entry("home", dict(PAGES)["home"], what.get("home", ""))
        for key, label in GUIDE_PAGES:
            entry(key, label, guide_about[key])
        heading("This recording")
        for key, label in PAGES:
            if key != "home":
                entry(key, label, what.get(key, ""))
        self._study_heading = heading("The study", fold=True)
        self._study_heading.setToolTip("Click to fold or unfold the study's pages")
        from onset_review.studies import STUDIES

        for key, label, what in STUDIES:
            item = QListWidgetItem(glyphs.icon(key, tokens), label)
            item.setData(Qt.UserRole, key)
            item.setToolTip(what)
            self._labels[key] = label
            self.nav.addItem(item)
            self._items[key] = item
        self.nav.itemClicked.connect(self._heading_clicked)
        self.set_study_folded(bool(self._remembered_value("study_folded")), remember=False)
        heading("The model")
        item = QListWidgetItem(glyphs.icon(CHAT_PAGE[0], tokens), CHAT_PAGE[1])
        item.setData(Qt.UserRole, CHAT_PAGE[0])
        self._labels[CHAT_PAGE[0]] = CHAT_PAGE[1]
        item.setToolTip("The local model on its own: not connected to this recording, "
                        "nothing checked")
        self.nav.addItem(item)
        self._items[CHAT_PAGE[0]] = item

        sidebar = QWidget()
        sidebar.setObjectName("onset_sidebar")
        sidebar.setAttribute(Qt.WA_StyledBackground, True)
        sidebar.setMinimumWidth(140)
        sidebar.setMaximumWidth(420)
        box = QVBoxLayout(sidebar)
        box.setContentsMargins(theme.SPACING, theme.SPACING + 4, 0, theme.SPACING)
        box.setSpacing(4)
        box.addWidget(self.nav, 1)
        self._sidebar = sidebar

    # -- the study's pages, folded away ---------------------------------------
    @property
    def study_folded(self) -> bool:
        from onset_review.studies import STUDIES

        return all(self._items[key].isHidden() for key, _, _ in STUDIES
                   if key in self._items)

    def set_study_folded(self, folded: bool, remember: bool = True) -> None:
        """Fold the six study pages under their heading, or show them. The
        page that is showing stays, and Alt+number still reaches every page."""
        from onset_review.studies import STUDIES

        for key, _label, _what in STUDIES:
            item = self._items.get(key)
            if item is not None:
                item.setHidden(bool(folded))
        self._study_heading.setText(("▸ " if folded else "▾ ") + "THE STUDY")
        if remember:
            self.remember_layout()

    def _heading_clicked(self, item) -> None:
        if item is self._study_heading:
            self.set_study_folded(not self.study_folded)

    def _remembered_value(self, name: str):
        import json

        try:
            payload = json.loads(self._layout_file().read_text(encoding="utf-8"))
            return payload.get(name)
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _nav_changed(self, current, _previous) -> None:
        key = current.data(Qt.UserRole) if current is not None else None
        if key and key in self._pages:
            if key != "analysis":
                self._return_panes()
            self.stack.setCurrentWidget(self._pages[key])
            if key == "analysis":
                self._lend_panes()
            self.title.setText(self._labels.get(key, ""))
            for page, widgets in self._page_actions.items():
                for widget in widgets:
                    widget.setVisible(page == key)
            refresh = getattr(self._pages[key], "refresh", None)
            if callable(refresh):
                refresh()
            self.pageChanged.emit(key)

    # -- the disclaimer -----------------------------------------------------
    def toggle_disclaimer(self) -> None:
        self._disclaimer_open = not self._disclaimer_open
        self._say_disclaimer()

    def _say_disclaimer(self) -> None:
        if self._disclaimer_open:
            self.banner.setText(f"{DISCLAIMER} <a href='less' style='color:"
                                f"{theme.current().accent};'>Less</a>")
        else:
            self.banner.setText(f"{DISCLAIMER_LINE} <a href='more' style='color:"
                                f"{theme.current().accent};'>Read more</a>")

    # -- the pages --------------------------------------------------------
    def _build_pages(self) -> None:
        builders = {
            "home": self._home_page, "recording": self._recording_page,
            "analysis": self._analysis_page, "contacts": self._contacts_page, "map": self._map_page,
            "quality": self._quality_page,
            "report": self._report_page, "assistant": self._assistant_page,
        }
        from onset_review import guide

        what = dict(HOW_TO_READ)
        for key, label in PAGES:
            if key != "home" and not self.loaded:
                page = guide.PreviewPage(self, key, label, what.get(key, ""))
            else:
                page = builders[key]()
            page.setObjectName(f"page_{key}")
            self._pages[key] = page
            self.stack.addWidget(page)
        self.quickstart = guide.QuickStartPage(self)
        self.case_page = guide.CasePage(self)
        for (key, _label), page in zip(GUIDE_PAGES, (self.quickstart, self.case_page),
                                       strict=True):
            page.setObjectName(f"page_{key}")
            self._pages[key] = page
            self.stack.addWidget(page)
        from onset_review.studies import STUDIES
        from onset_review.studypages import StudyPage

        for key, _label, _what in STUDIES:
            study = StudyPage(key, cached=self._cached, session=self.session)
            if self._on_open_cached is not None:
                study.openRequested.connect(self._on_open_cached)
            study.notebookRequested.connect(self._open_notebook)
            page = self._on_card(study)
            page.setObjectName(f"page_{key}")
            self._pages[key] = page
            self.stack.addWidget(page)
        # The chat needs no recording: the window's panel when there is one,
        # its own otherwise, so the page is there before anything is opened.
        chat = self.panels.get("chat")
        if chat is None:
            from onset_review.chatview import ChatPanel

            chat = ChatPanel()
        page = self._on_card(chat)
        page.setObjectName(f"page_{CHAT_PAGE[0]}")
        self._pages[CHAT_PAGE[0]] = page
        self.stack.addWidget(page)

    def _home_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        title = QLabel("<h2>Onset Review</h2>")
        box.addWidget(title)
        from onset_review import credit

        byline = theme.muted(credit(), size=9)
        byline.setObjectName("onset_credit")
        box.addWidget(byline)
        lead = QLabel(WHAT_THIS_IS)
        lead.setWordWrap(True)
        box.addWidget(lead)
        caveat = theme.muted(WHAT_IT_IS_NOT, size=10)
        caveat.setTextFormat(Qt.RichText)
        box.addWidget(caveat)

        self.continue_button = None
        if self.loaded:
            summary = self.session.summary()
            numbers = QHBoxLayout()
            numbers.setSpacing(theme.SPACING)
            for value, caption in (
                (summary.get("n_channels", len(self.session.findings)),
                 "channels analysed"),
                (len(self.session.accepted), "events accepted"),
                (len(self.session.findings), "channels ranked"),
                (f"{self.session.span_duration:g} s", "analysed span"),
            ):
                tile = QLabel(f"<div style='font-size:18pt;font-weight:700;'>{value}</div>"
                              f"<div style='font-size:9pt;'>{caption}</div>")
                tile.setStyleSheet(theme.card("plain"))
                tile.setObjectName("onset_tile")
                numbers.addWidget(tile)
            numbers.addStretch(1)
            box.addLayout(numbers)
        else:
            opener = QLabel("<b>Nothing is open yet.</b> Pick a window below and "
                            "press Open, or use <b>File → Open a recording…</b> to "
                            "choose the band and the detectors as well, or "
                            "<b>File → Open a file…</b> for a recording of your own. "
                            "<b>New here?</b> <a href='quickstart'>The quick start guide</a> "
                            "(F1) shows everything the program does; every page in the "
                            "sidebar can be looked at before anything is open.")
            opener.setTextFormat(Qt.RichText)
            opener.linkActivated.connect(lambda _link: self.show_page("quickstart"))
            opener.setObjectName("onset_nothing_open")
            opener.setWordWrap(True)
            opener.setStyleSheet(theme.card("info"))
            box.addWidget(opener)
            # The first moves, as buttons: what a new reader is most likely
            # to want, without opening a menu to find it.
            from onset_review import guide

            starts = QHBoxLayout()
            starts.setSpacing(theme.SPACING)
            self.start_buttons: dict[str, QPushButton] = {}
            for target, text, tip in (
                    ("do:open-file", "Open a file of your own…",
                     "EDF, BrainVision, FIF, Nihon Kohden and more"),
                    ("do:open-recording", "Open a public recording…",
                     "Choose a patient, a window, the band and the detectors"),
                    ("do:new-case", "Start a patient case…",
                     "One patient's recordings, step by step to a signed report"),
                    ("page:quickstart", "Quick start guide",
                     "Everything the program does (F1)")):
                button = QPushButton(text)
                button.setObjectName(f"onset_start_{target.split(':')[1]}")
                button.setToolTip(tip)
                button.clicked.connect(lambda _=False, t=target: guide.run_target(self, t))
                starts.addWidget(button)
                self.start_buttons[target] = button
            self.start_buttons["do:open-file"].setProperty("primary", True)
            starts.addStretch(1)
            box.addLayout(starts)
            box.addWidget(self._continue_card())

        shelf = QWidget()
        shelf_box = QVBoxLayout(shelf)
        shelf_box.setContentsMargins(0, 0, 0, 0)
        shelf_box.setSpacing(theme.SPACING)
        # The windows on disk, grouped by patient: a reader looks for a
        # patient first and a minute of them second.
        self.cached_tree = QTreeWidget()
        self.cached_tree.setObjectName("onset_cached")
        self.cached_tree.setHeaderLabels(["Patient · window", "Dataset", "Run", "Sampling",
                                          "Channels"])
        self.cached_tree.setRootIsDecorated(True)
        self.cached_tree.setAlternatingRowColors(False)
        self.cached_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.cached_tree.setUniformRowHeights(True)
        self.cached_tree.setFrameShape(QFrame.NoFrame)
        self.cached_tree.itemDoubleClicked.connect(lambda _item, _col: self._open_selected())
        self.cached_tree.itemSelectionChanged.connect(self._selection_changed)
        shelf_box.addWidget(self.cached_tree, 1)
        self.cached_empty = theme.muted(
            "No windows on this machine yet. Open a file of your own, open a public "
            "recording (it is fetched once and kept), or start a patient case.", size=10)
        self.cached_empty.setObjectName("onset_cached_empty")
        self.cached_empty.setWordWrap(True)
        shelf_box.addWidget(self.cached_empty, 0, Qt.AlignTop)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open the selected window")
        self.open_button.setObjectName("onset_open_cached")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_selected)
        buttons.addWidget(self.open_button)
        self.import_button = QPushButton("Open a file…")
        self.import_button.setEnabled(self._on_import is not None)
        if self._on_import is not None:
            self.import_button.clicked.connect(lambda _=False: self._on_import())
        buttons.addWidget(self.import_button)
        buttons.addStretch(1)
        shelf_box.addLayout(buttons)
        shelf_box.addWidget(theme.muted(
            "Windows are fetched once and kept. To add another minute of a "
            "patient, from a terminal:  onset-hfo fetch --subject sub-02 "
            "--t-start 0 --t-stop 60"))
        box.addWidget(theme.card_frame(shelf, "Windows on this machine"), 1)
        page.refresh = self._refresh_cached      # type: ignore[attr-defined]
        self._refresh_cached()
        return page

    def _continue_card(self) -> QWidget:
        """The window opened last time, one click away; an empty widget when
        there is none or it is no longer on disk."""
        holder = QWidget()
        holder.setObjectName("onset_continue_holder")
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        recent = recent_window()
        row = self._cached_row(recent) if recent else None
        if row is None:
            holder.setVisible(False)
            return holder
        body = QWidget()
        inner = QHBoxLayout(body)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(theme.SPACING)
        what = QLabel(f"<b>{html.escape(str(row['subject']))}</b> · "
                      f"{float(row['t_start']):g}–{float(row['t_stop']):g} s · "
                      f"{html.escape(str(row.get('dataset', '')))}")
        what.setObjectName("onset_continue_what")
        inner.addWidget(what, 1)
        self.continue_button = QPushButton("Continue")
        self.continue_button.setObjectName("onset_continue")
        self.continue_button.setProperty("primary", True)
        self.continue_button.setEnabled(self._on_open_cached is not None)
        self.continue_button.clicked.connect(
            lambda _=False, row=row: self._on_open_cached and self._on_open_cached(row))
        inner.addWidget(self.continue_button)
        box.addWidget(theme.card_frame(body, "Continue where you left off"))
        return holder

    def _cached_row(self, wanted: dict) -> dict | None:
        frame = self._cached_frame if hasattr(self, "_cached_frame") else None
        if frame is None:
            frame = self._load_cached()
        if frame is None or len(frame) == 0:
            return None
        for row in frame.to_dict("records"):
            if (str(row.get("subject")) == wanted["subject"]
                    and abs(float(row.get("t_start", -1)) - wanted["t_start"]) < 1e-6
                    and abs(float(row.get("t_stop", -1)) - wanted["t_stop"]) < 1e-6):
                return row
        return None

    def _load_cached(self):
        import pandas as pd

        frame = None
        if self._cached is not None:
            try:
                frame = self._cached()
            except Exception:       # noqa: BLE001 - a listing must not kill a window
                frame = None
        if frame is None or len(frame) == 0:
            frame = pd.DataFrame(columns=["subject", "t_start", "t_stop", "dataset"])
        self._cached_frame = frame
        return frame

    def _refresh_cached(self) -> None:
        frame = self._load_cached()
        self.cached_tree.clear()
        self._window_items: list[QTreeWidgetItem] = []
        groups: dict[str, QTreeWidgetItem] = {}
        rows = frame.sort_values([c for c in ("subject", "t_start") if c in frame.columns]) \
            if len(frame) else frame
        for row in rows.to_dict("records"):
            subject = str(row.get("subject", ""))
            group = groups.get(subject)
            if group is None:
                group = QTreeWidgetItem([subject])
                group.setFlags(group.flags() & ~Qt.ItemIsSelectable)
                font = group.font(0)
                font.setBold(True)
                group.setFont(0, font)
                self.cached_tree.addTopLevelItem(group)
                groups[subject] = group

            item = QTreeWidgetItem([
                f"{float(row.get('t_start', 0)):g}–{float(row.get('t_stop', 0)):g} s",
                _cell(row, "dataset"), _cell(row, "run"),
                _cell(row, "sfreq", "{:g} Hz"), _cell(row, "n_channels", "{:g}")])
            item.setData(0, Qt.UserRole, row)
            group.addChild(item)
            self._window_items.append(item)
        for group in groups.values():
            group.setText(0, f"{group.text(0)}  ·  {group.childCount()} "
                             f"window{'s' if group.childCount() != 1 else ''}")
        self.cached_tree.expandAll()
        if hasattr(self, "cached_empty"):
            # Empty, the shelf says so in a line instead of a blank table, and
            # its buttons give way to the ones above that do the same.
            self.cached_empty.setVisible(not groups)
            for widget in (self.cached_tree, self.open_button, self.import_button):
                widget.setVisible(bool(groups))
        for column in range(self.cached_tree.columnCount()):
            self.cached_tree.resizeColumnToContents(column)
        self._selection_changed()

    def windows_listed(self) -> int:
        """How many windows the tree lists, across every patient."""
        return len(getattr(self, "_window_items", []))

    def select_window(self, index: int) -> None:
        """Select the `index`-th window in the tree, in patient order."""
        items = getattr(self, "_window_items", [])
        if 0 <= index < len(items):
            self.cached_tree.setCurrentItem(items[index])
            items[index].setSelected(True)

    def _selected_row(self) -> dict | None:
        for item in self.cached_tree.selectedItems():
            row = item.data(0, Qt.UserRole)
            if isinstance(row, dict):
                return row
        return None

    def _selection_changed(self) -> None:
        self.open_button.setEnabled(self._on_open_cached is not None
                                    and self._selected_row() is not None)

    def _open_selected(self) -> None:
        row = self._selected_row()
        if self._on_open_cached is None or row is None:
            return
        self._on_open_cached(row)

    def _recording_page(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.SPACING)

        left = QWidget()
        box = QVBoxLayout(left)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        # The site's first question. Answered in plain text when there is a
        # clear answer; the tinted box is kept for the case that needs care,
        # a leader that cannot be told from its neighbours.
        self.leader = QLabel(self.session.caveat())
        self.leader.setObjectName("onset_leader")
        self.leader.setWordWrap(True)
        self.leader.setStyleSheet(
            f"color:{theme.current().text};background:transparent;border:none;"
            if self.session.leader.get("distinguishable") else theme.card("bad"))
        head = QWidget()
        head_box = QVBoxLayout(head)
        head_box.setContentsMargins(0, 0, 0, 0)
        head_box.setSpacing(4)
        head_box.addWidget(theme.section_label("Does any channel actually stand out?"))
        head_box.addWidget(self.leader)

        self.trend_toggle = QToolButton()
        self.trend_toggle.setText("Trend — rate per channel over time")
        self.trend_toggle.setCheckable(True)
        self.trend_toggle.setChecked(True)
        self.trend_toggle.setArrowType(Qt.DownArrow)
        self.trend_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.trend_toggle.setAutoRaise(True)
        self.trend_toggle.toggled.connect(self._toggle_trend)
        self.pop_button = QPushButton()
        self.pop_button.setObjectName("onset_pop_trace")
        self.pop_button.setText("Open the trace in a new window")
        self.pop_button.setToolTip("Give MNE's browser a window of its own — on a "
                                   "second monitor, or just bigger. Everything "
                                   "here keeps driving it. Ctrl+Shift+T.")
        self.pop_button.clicked.connect(lambda _=False: self.toggle_trace_window())
        self._register_action("recording", self.pop_button)
        head_box.addWidget(self.trend_toggle)
        head_box.addWidget(self.panels["trends"], 1)
        self.panels["trends"].setMinimumHeight(90)
        upper = theme.card_frame(head)
        lower_body = QWidget()
        under = QVBoxLayout(lower_body)
        under.setContentsMargins(0, 0, 0, 0)
        under.setSpacing(4)
        under.addWidget(self.panels["controls"])
        # MNE's browser is a QMainWindow of its own; given a parent it becomes
        # an ordinary child widget, toolbar and all. It sits in a slot of its
        # own so that it can be lifted out into a window and put back.
        self._trace_slot = QWidget()
        self._trace_slot.setObjectName("onset_trace_slot")
        slot = QVBoxLayout(self._trace_slot)
        slot.setContentsMargins(0, 0, 0, 0)
        slot.setSpacing(0)
        self.trace_placeholder = QLabel(
            "The trace is open in its own window. Close that window, or press "
            "<b>Bring the trace back</b>, to put it back here.")
        self.trace_placeholder.setObjectName("onset_trace_placeholder")
        self.trace_placeholder.setWordWrap(True)
        self.trace_placeholder.setAlignment(Qt.AlignCenter)
        self.trace_placeholder.setStyleSheet(theme.card("info"))
        self.trace_placeholder.setVisible(False)
        slot.addWidget(self.trace_placeholder, 1)
        self.figure.setParent(self._trace_slot)
        slot.addWidget(self.figure, 1)
        under.addWidget(self._trace_slot, 1)
        lower = theme.card_frame(lower_body)
        box.addWidget(self._split("recording_v", Qt.Vertical,
                                  [upper, lower], [250, 700],
                                  stretch=(0, 1)), 1)

        self.side = QTabWidget()
        self.side.setObjectName("onset_side")
        self.side.addTab(self.panels["findings"], "Ranking")
        self.side.addTab(self.panels["events"], "Events")
        self.side.addTab(self.panels["detail"], "Event")
        self.side.addTab(self.panels["spectrum"], "Spectrum")
        self.side.addTab(self.panels["average"], "Average")
        self.side.addTab(self.panels["sensitivity"], "Threshold")
        # Wide enough for every tab to show: six tabs in a 260 px column
        # overflowed into scroll arrows, and a tab a reviewer cannot see is a
        # view they do not have. The splitter honours this minimum.
        self.side.tabBar().setExpanding(False)
        self.side.setMinimumWidth(max(260, self.side.tabBar().sizeHint().width() + 12))
        self.side.setTabToolTip(0, "Channels ranked by rate — evidence, not a "
                                   "recommendation. Tinted rows are tied with "
                                   "the busiest.")
        self.side.setTabToolTip(2, "The selected event wideband, filtered and in "
                                   "time-frequency — oscillation or filter ringing?")
        self.side.setTabToolTip(3, "Each channel's power spectrum, the chosen one in "
                                   "front — noisy, busy, or mains?")
        self.side.setTabToolTip(4, "A channel's events aligned and averaged — do they "
                                   "average to an oscillation or to a transient?")
        self.side.setTabToolTip(5, "The leading channels re-tested at stricter thresholds "
                                   "— does the ranking survive?")
        side_card = theme.card_frame(self.side)
        side_card.setMinimumWidth(self.side.minimumWidth() + 2 * theme.SPACING + 2)
        outer.addWidget(self._split("recording", Qt.Horizontal, [left, side_card],
                                    [1000, max(COLUMN_WIDTH + 150, side_card.minimumWidth())],
                                    stretch=(1, 0)), 1)
        return page

    # -- the trace in a window of its own ----------------------------------
    @property
    def trace_popped(self) -> bool:
        return self.figure is not None and bool(self.figure.isWindow())

    def toggle_trace_window(self) -> None:
        if self.trace_popped:
            self.dock_trace()
        else:
            self.pop_out_trace()

    def pop_out_trace(self) -> None:
        """Lift MNE's browser out of the page into a top-level window.

        The same widget, reparented: the trace controls, the event list, the
        trend and every citation keep driving it, because they hold the
        figure and not its place on the page. A placeholder stays where it
        was so the page does not read as broken.
        """
        if self.trace_popped:
            return
        size = self.figure.size()
        self.figure.setParent(None)
        self.figure.setWindowFlags(Qt.Window)
        # In a window of its own the browser's toolbar is its only bar.
        self.panels["controls"].show_browser_toolbar(True)
        self.figure.setWindowTitle(
            f"Onset Review — trace — {self.session.request.label()}")
        if size.width() > 200 and size.height() > 200:
            self.figure.resize(size)
        # Closing that window must not close the browser: MNE's own close
        # tears the figure down, and the page would be left holding a corpse.
        self.figure.installEventFilter(self)
        self.figure.show()
        self.trace_placeholder.setVisible(True)
        self.pop_button.setText("Bring the trace back")
        self.figure.raise_()
        self.figure.activateWindow()

    def dock_trace(self) -> None:
        """Put the browser back in its slot on the Recording page."""
        if not self.trace_popped:
            return
        self.figure.removeEventFilter(self)
        self.figure.setWindowFlags(Qt.Widget)
        self.figure.setParent(self._trace_slot)
        self.panels["controls"].show_browser_toolbar(False)
        self._trace_slot.layout().addWidget(self.figure, 1)
        self.figure.show()
        self.trace_placeholder.setVisible(False)
        self.pop_button.setText("Open the trace in a new window")

    def eventFilter(self, watched, event):      # noqa: N802  (Qt's spelling)
        if watched is self.figure and event.type() == QEvent.Close and self.trace_popped:
            event.ignore()
            self.dock_trace()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):      # noqa: N802  (Qt's spelling)
        # A popped-out trace is a second top-level window; closing this one
        # must take it along rather than leave a browser nobody can reach.
        if self.trace_popped:
            self.dock_trace()
        self._return_panes()
        self.remember_layout()
        editor = self.panels.get("editor")
        if editor is not None:
            editor.remember()
        workspace = self.panels.get("workspace")
        if workspace is not None:
            workspace.close_windows()
        for page in self._pages.values():
            closer = getattr(getattr(page, "body", page), "close_windows", None)
            if callable(closer) and page is not workspace:
                closer()
        console = self.panels.get("console")
        if console is not None and console.window() is self:
            # Only a console this window still holds: one carried into the
            # next window by a re-analysis has already left.
            console.close_figures()
        super().closeEvent(event)

    def _toggle_trend(self, on: bool) -> None:
        self.panels["trends"].setVisible(bool(on))
        self.trend_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        splitter = self.splitter("recording_v")
        if splitter is not None and not on:
            upper = splitter.widget(0)
            splitter.setSizes([upper.sizeHint().height(), max(300, sum(splitter.sizes()))])

    def _analysis_page(self) -> QWidget:
        """Spyder on one page: the Editor on the left; the Workspace and
        Files tabbed on the right, the Console under them. The three panes
        are the window's own -- the same widgets as the docks -- brought
        here while the page is up and given back when it is left, so a name
        made here is in the Workspace on every page."""
        from onset_review.editor import EditorPanel

        editor = self.panels.get("editor")
        if editor is None:
            editor = EditorPanel(restore=True)
            self.panels["editor"] = editor
        self._slots: dict[str, QWidget] = {}
        for key in ("workspace", "files", "console"):
            slot = QWidget()
            slot.setObjectName(f"onset_analysis_{key}")
            box = QVBoxLayout(slot)
            box.setContentsMargins(0, 0, 0, 0)
            self._slots[key] = slot
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setObjectName("onset_analysis_side")
        self.analysis_tabs.setDocumentMode(True)
        self.analysis_tabs.addTab(self._slots["workspace"], "Workspace")
        self.analysis_tabs.addTab(self._slots["files"], "Files")
        # Plots and History are the console's, put here once the console is
        # built (`_attach_console_panes`): it may be carried in from the last
        # window, or made with the panes after the pages.
        console = theme.card_frame(self._slots["console"], "Console", padding=4)
        side = self._split("analysis_side", Qt.Vertical, [self.analysis_tabs, console],
                           [360, 420])
        code = theme.card_frame(editor, "Editor", padding=4)
        body = self._split("analysis", Qt.Horizontal, [code, side], [780, 520],
                           stretch=(3, 2))
        #: key -> whether its dock was showing when the page took the pane.
        self._lent: dict[str, bool] | None = None
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(body, 1)
        return page

    def _attach_console_panes(self) -> None:
        """The console's Plots and History as tabs beside the Workspace and
        Files; a figure drawn while the page is up shows there rather than in
        a window of its own."""
        console = self.panels.get("console")
        tabs = getattr(self, "analysis_tabs", None)
        if console is None or tabs is None:
            return
        for pane, title, tip in (
                (console.plots, "Plots", "Every figure the console or the editor drew"),
                (console.history, "History", "Every command run in the console, "
                                             "searchable; send any to the editor")):
            if tabs.indexOf(pane) < 0:
                index = tabs.addTab(pane, title)
                tabs.setTabToolTip(index, tip)
        console.plots_shown = self._show_plots

    def _show_plots(self) -> bool:
        console = self.panels.get("console")
        if (console is None or self.current_page() != "analysis"
                or console.plots.window() is not self):
            return False
        self.analysis_tabs.setCurrentWidget(console.plots)
        return True

    # -- lending the panes to the Analysis page ------------------------------
    @property
    def panes_lent(self) -> bool:
        return getattr(self, "_lent", None) is not None

    def _lend_panes(self) -> None:
        """Bring the Workspace, Files and Console onto the Analysis page; the
        docks stand hidden, a note in each, until the page is left."""
        if self.panes_lent or not getattr(self, "docks", None) \
                or not hasattr(self, "_slots"):
            return
        lent = {}
        building, self._building = getattr(self, "_building", False), True
        try:
            for key, slot in self._slots.items():
                dock, panel = self.docks[key], self.panels[key]
                lent[key] = dock.toggleViewAction().isChecked()
                note = QLabel(f"The {dock.windowTitle()} is on the Analysis page while "
                              "that page is open.")
                note.setWordWrap(True)
                note.setAlignment(Qt.AlignCenter)
                note.setStyleSheet(f"color:{theme.current().text_muted};")
                dock.setWidget(note)
                slot.layout().addWidget(panel)
                panel.show()
                dock.setVisible(False)
        finally:
            self._building = building
        self._lent = lent

    def _return_panes(self) -> None:
        """Give the panes back to their docks, each shown as it was."""
        if not self.panes_lent:
            return
        lent, self._lent = self._lent, None
        building, self._building = getattr(self, "_building", False), True
        try:
            for key, shown in lent.items():
                dock, panel = self.docks[key], self.panels[key]
                if panel.window() is not self:
                    continue        # carried into the next window already
                note = dock.widget()
                dock.setWidget(panel)
                panel.show()
                if note is not None and note is not panel:
                    note.hide()
                    note.deleteLater()
                dock.setVisible(shown)
        finally:
            self._building = building

    @contextlib.contextmanager
    def _panes_home(self):
        """The panes in their docks for the length of the block (a layout
        being applied, the arrangement being saved), and back on the
        Analysis page after, if that is where they were."""
        lent = self.panes_lent
        if lent:
            self._return_panes()
        try:
            yield
        finally:
            if lent and self.current_page() == "analysis":
                self._lend_panes()

    def _call_handler(self, name: str, *args) -> None:
        handler = getattr(self, name, None)
        if callable(handler):
            handler(*args)

    def _open_notebook(self, path: str) -> None:
        """A study page written as a notebook: on the Analysis page when there
        is one; otherwise where it was written is said, for Jupyter or later."""
        if "analysis" in self._pages and self.panels.get("editor") is not None:
            self._open_script(path)
        elif self.statusBar() is not None:
            self.statusBar().showMessage(
                f"Notebook written to {path}. Open a recording to run it on the Analysis "
                "page, or open it in Jupyter.", 10000)

    def _open_script(self, path: str) -> None:
        """A script or notebook double-clicked in the Files pane."""
        editor = self.panels.get("editor")
        if editor is None or "analysis" not in self._pages:
            if self.statusBar() is not None:
                self.statusBar().showMessage(
                    "Scripts open on the Analysis page, once a recording is open.", 6000)
            return
        if editor.open_file(path) is not None:
            self.show_page("analysis")

    def _contacts_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING)
        column = QWidget()
        column.setMinimumWidth(220)
        box = QVBoxLayout(column)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        box.addWidget(theme.section_label("Patient"))
        box.addWidget(self.panels["patient"], 1)
        box.addWidget(theme.section_label("Coordinates"))
        self.place_button = QPushButton("Place the contacts from a file…")
        self.place_button.setObjectName("onset_place_contacts")
        self.place_button.setToolTip(
            "A BIDS electrodes.tsv, or a CSV from a surgical planning system. "
            "The 3D view stops being a montage diagram and becomes this "
            "patient's head. Coordinates do not affect a rate.")
        self.place_button.setEnabled(self._on_place_contacts is not None)
        if self._on_place_contacts is not None:
            self.place_button.clicked.connect(lambda _=False: self._on_place_contacts())
        box.addWidget(self.place_button)
        box.addWidget(theme.muted(
            "Without a coordinate file the layout is schematic: shafts in "
            "name order, contacts in number order. Enough to see which shafts "
            "are active; not enough for anything metric."))
        box.addWidget(theme.section_label("Template brain"))
        self.fetch_button = QPushButton("Fetch the template brain (MNE fsaverage)…")
        self.fetch_button.setObjectName("onset_fetch_template")
        self.fetch_button.setToolTip(
            "Downloads MNE's fsaverage bundle once, a few hundred megabytes, into "
            "your MNE data folder. Then 'Template brain' on the 3D view draws the "
            "average cortex under measured contacts in MNI or fsaverage space.")
        self.fetch_button.clicked.connect(lambda _=False: self.fetch_template())
        self.fetch_status = theme.muted("")
        self.fetch_status.setObjectName("onset_fetch_status")
        box.addWidget(self.fetch_button)
        box.addWidget(self.fetch_status)
        box.addWidget(theme.muted(
            "The template is an average brain, not this patient's. It is only "
            "drawn under measured coordinates, and only meaningfully for "
            "coordinates in a template space; the caption says so each time."))
        self._say_template_state()
        box.addStretch(1)
        row.addWidget(self._split("contacts", Qt.Horizontal,
                                  [theme.card_frame(self.panels["brain"]),
                                   theme.card_frame(column)],
                                  [1100, COLUMN_WIDTH], stretch=(1, 0)), 1)
        return page

    def _say_template_state(self) -> None:
        from onset_review.anatomy import template_dir

        root = template_dir()
        if root is not None:
            self.fetch_status.setText(f"On this machine: {root}")
            self.fetch_button.setText("Template brain is fetched")
            self.fetch_button.setEnabled(False)
        elif importlib.util.find_spec("nilearn") is not None:
            self.fetch_status.setText(
                "Drawn from nilearn's fsaverage5, installed with the app (a coarser mesh "
                "of the same template). Fetch MNE's for the full resolution.")
        else:
            self.fetch_status.setText("Not on this machine yet.")

    def fetch_template(self, fetch=None) -> None:
        """Fetch the template on a worker, saying what is happening. `fetch`
        is the function to run, for tests; the real one is MNE's."""
        from qtpy.QtCore import QThread

        from onset_review.anatomy import fetch_template

        run = fetch or fetch_template
        self.fetch_button.setEnabled(False)
        self.fetch_status.setText("Fetching the template brain… a few hundred megabytes, once.")

        class Worker(QThread):
            def __init__(worker, parent):
                super().__init__(parent)
                worker.error = ""
                worker.path = None

            def run(worker):
                try:
                    worker.path = run()
                except Exception as error:      # noqa: BLE001 - reported, not raised
                    worker.error = str(error)

        self._fetcher = Worker(self)

        def done():
            if self._fetcher.error:
                self.fetch_status.setText(f"The fetch failed: {self._fetcher.error}")
                self.fetch_button.setEnabled(True)
                return
            self._say_template_state()
            brain = self.panels.get("brain")
            if brain is not None:
                brain._surface, brain._surface_error = None, ""
                brain._update_template_gate()

        self._fetcher.finished.connect(done)
        from onset_review.workers import track

        track(self._fetcher).start()

    def _map_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING)
        column = QWidget()
        column.setMinimumWidth(220)
        box = QVBoxLayout(column)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        box.addWidget(theme.section_label("How to read it"))
        box.addWidget(theme.muted(
            "Each dot is one channel at its contacts' midpoint, coloured by the "
            "measure chosen above; the biggest, brightest dots lead. Numbers are "
            "ranks; a star marks a channel in the statistically tied set. Rings "
            "say whether the surgeon removed the contact, when that is known. "
            "The channel chosen anywhere in the window — the ranking, the event "
            "list, the trace, the 3D view — is haloed here."))
        box.addWidget(theme.section_label("And the assistant"))
        box.addWidget(theme.muted(
            "\u201cAsk the assistant where\u201d sends it the counts behind this "
            "map: which shafts and which side the leading channels are on, and "
            "whether the positions are measured or schematic. It is never told "
            "what was resected."))
        box.addWidget(theme.section_label("Measured or schematic"))
        box.addWidget(theme.muted(
            "With no coordinate file the layout is schematic: shafts in name "
            "order, contacts in number order, placed where the structure the "
            "name claims would be. Enough to see which shafts are active and "
            "their order; not anatomy. Place the contacts from a file on the "
            "Contacts page and this map becomes this patient's head."))
        box.addStretch(1)
        row.addWidget(self._split("map", Qt.Horizontal,
                                  [theme.card_frame(self.panels["map"]),
                                   theme.card_frame(column)],
                                  [1100, COLUMN_WIDTH], stretch=(1, 0)), 1)
        self.panels["map"].askRequested.connect(self._ask_about_map)
        return page

    def _ask_about_map(self, question: str) -> None:
        """The map's button: go to the assistant and ask, so the answer lands
        where the person can read it."""
        self.show_page("assistant")
        self.panels["assistant"].ask(question)

    def _quality_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        split = self._split(
            "quality_h", Qt.Horizontal,
            [self._titled("Data quality", self.panels["quality"]),
             self._titled("Preprocessing", self.panels["preprocess"])],
            [700, 700])
        under = QTabWidget()
        under.setObjectName("onset_quality_under")
        under.addTab(self.panels["provenance"], "Provenance")
        under.addTab(self.panels["components"], "Components (ICA, experimental)")
        under.setTabToolTip(0, "How this was produced, step by step")
        under.setTabToolTip(1, "What ICA found, scored, for you to choose from; "
                               "nothing is removed until you choose")
        box.addWidget(self._split("quality_v", Qt.Vertical,
                                  [split, theme.card_frame(under)], [540, 360]), 1)
        return page

    def _report_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING)
        self.report_view = QTextBrowser()
        self.report_view.setObjectName("onset_report")
        self.report_view.setOpenExternalLinks(False)
        self.report_view.setFrameShape(QFrame.NoFrame)
        preview = self._titled("The review as it will be exported", self.report_view)

        column = QWidget()
        column.setMinimumWidth(220)
        box = QVBoxLayout(column)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        box.addWidget(theme.section_label("Findings paragraph"))
        self.findings_edit = QPlainTextEdit()
        self.findings_edit.setObjectName("onset_findings_draft")
        self.findings_edit.setPlaceholderText(
            "Write the findings in your own words, or let the assistant draft them "
            "from the analysis; the report says which.")
        self.findings_edit.setMaximumHeight(170)
        self.findings_edit.setPlainText(getattr(self.session.read, "draft", "") or "")
        self.findings_edit.textChanged.connect(self._findings_edited)
        box.addWidget(self.findings_edit)
        self.findings_status = theme.muted("")
        self.findings_status.setObjectName("onset_findings_status")
        box.addWidget(self.findings_status)
        self.draft_button = QPushButton("Draft it with the assistant")
        self.draft_button.setObjectName("onset_draft_findings")
        self.draft_button.setToolTip(
            "Asks the assistant for the findings paragraph from the evidence it is "
            "briefed with; every number is checked against a query, and the report "
            "marks the paragraph as the assistant's until you edit it.")
        self.draft_button.setEnabled("assistant" in self.panels)
        self.draft_button.clicked.connect(lambda _=False: self.draft_findings())
        box.addWidget(self.draft_button)
        self._say_findings_state()
        box.addWidget(theme.section_label("Agreement with the annotators"))
        box.addWidget(self.panels["agreement"], 1)
        self.appendix = QCheckBox("Show the appendix (every measurement)")
        self.appendix.setObjectName("onset_report_appendix")
        self.appendix.setChecked(False)
        self.appendix.setToolTip("The exported file always carries the appendix; "
                                 "this only shortens the preview")
        self.appendix.toggled.connect(lambda _on: self.refresh_report())
        box.addWidget(self.appendix)
        self.export_button = QPushButton("Export review…")
        self.export_button.setObjectName("onset_export")
        self.export_button.setEnabled(self._on_export is not None)
        if self._on_export is not None:
            self.export_button.clicked.connect(lambda _=False: self._on_export())
        self.export_button.setToolTip("Markdown or a web page. Your verdicts and "
                                      "notes go in under your name; name yourself "
                                      "under Read first.")
        self._register_action("report", self.export_button)
        box.addStretch(1)
        row.addWidget(self._split("report", Qt.Horizontal,
                                  [preview, theme.card_frame(column)],
                                  [1100, COLUMN_WIDTH], stretch=(1, 0)), 1)
        page.refresh = self.refresh_report      # type: ignore[attr-defined]
        assistant = self.panels.get("assistant")
        if assistant is not None and hasattr(assistant, "drafted"):
            assistant.drafted.connect(self.take_draft)
        return page

    # -- the findings paragraph ---------------------------------------------
    def draft_findings(self) -> bool:
        """Have the assistant draft the paragraph; it lands in the box and the
        report marks it as the assistant's."""
        assistant = self.panels.get("assistant")
        if assistant is None:
            return False
        self.draft_button.setEnabled(False)
        self.findings_status.setText("Asking the assistant…")
        try:
            drafted = assistant.draft_findings()
        finally:
            self.draft_button.setEnabled(True)
        if not drafted:
            self.findings_status.setText(
                "No draft: the assistant refused or returned nothing. Its page says why.")
        return drafted

    def take_draft(self, text: str, by: str) -> None:
        """A paragraph from the assistant: shown in the box, filed as its."""
        read = self.session.read
        read.set_draft(text, by)
        self._keep_read()
        self.findings_edit.blockSignals(True)
        self.findings_edit.setPlainText(read.draft)
        self.findings_edit.blockSignals(False)
        self._say_findings_state()
        self.refresh_report()

    def _findings_edited(self) -> None:
        """The reader typed: the paragraph is theirs now, and says so."""
        read = self.session.read
        text = self.findings_edit.toPlainText()
        if text.strip() == read.draft.strip():
            return
        who = (getattr(read, "reader", "") or "").strip() or "the reader"
        read.set_draft(text, who)
        self._keep_read()
        self._say_findings_state()
        self.refresh_report()

    def _keep_read(self) -> None:
        from onset_review import adjudication

        try:
            adjudication.save(self.session.request, self.session.read)
        except OSError as error:
            self.findings_status.setText(f"Could not save the read: {error}")

    def _say_findings_state(self) -> None:
        read = self.session.read
        if not read.draft:
            self.findings_status.setText("Nothing written yet. The report then carries the "
                                         "detector's numbers alone.")
        elif read.draft_is_assistants:
            self.findings_status.setText(
                f"Drafted by the {read.draft_by}; the report marks it as unedited by you "
                "until you change it.")
        else:
            self.findings_status.setText(f"Written by {read.draft_by}.")

    def refresh_report(self) -> None:
        """Rebuild the preview from the session as it is now, verdicts and all."""
        from onset_review import report
        from onset_review.studypages import set_markdown

        reader = getattr(getattr(self.session, "read", None), "reader", "") or None
        appendix = bool(getattr(self, "appendix", None) and self.appendix.isChecked())
        try:
            set_markdown(self.report_view,
                         report.review_markdown(self.session, reviewer=reader,
                                                appendix=appendix))
        except Exception as error:      # noqa: BLE001 - a preview must not kill a page
            self.report_view.setPlainText(f"The report could not be rendered: {error}")

    def _assistant_page(self) -> QWidget:
        return self._on_card(self.panels["assistant"])

    # -- panes: Spyder's dockable panels ----------------------------------
    #: (key, title, what it is). The order they are tabbed in.
    PANES = (
        ("workspace", "Workspace",
         "Every variable this window holds, with its type and size, as MATLAB's "
         "Workspace or Spyder's Variable Explorer lists them. Double-click one "
         "to open it in a window of its own."),
        ("files", "Files",
         "The current folder. Double-click a recording to open it, a folder to "
         "go into it."),
        ("console", "Console",
         "Python in this window's memory, as Spyder's IPython console: session, "
         "raw, events and findings are the window's own objects. Commands run here "
         "are listed in the exported report."),
    )
    #: Below this screen width the panes start hidden on a loaded window: at
    #: 1366 px the Recording page needs the width. View → Workspace / Files
    #: shows them, and whatever the reviewer arranges is remembered.
    PANES_MIN_WIDTH = 1600

    def _build_panes(self) -> None:
        """Workspace and Files as dock widgets: dragged to any edge, floated
        as windows of their own, tabbed together or apart, closed and
        reopened from View. The pages stay in the middle, as Spyder's editor
        does."""
        from qtpy.QtWidgets import QDockWidget

        from onset_review.console import ConsolePanel, connect_panes
        from onset_review.files import FilesPanel
        from onset_review.workspace import WorkspacePanel

        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks
                            | QMainWindow.AllowNestedDocks)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        if self.panels.get("workspace") is None:
            self.panels["workspace"] = WorkspacePanel(self.session)
        if self.panels.get("files") is None:
            self.panels["files"] = FilesPanel()
        if self.panels.get("console") is None:
            self.panels["console"] = ConsolePanel(self.session)
        files = self.panels["files"]
        files.openRequested.connect(self._open_path)
        self.panels["workspace"].folder = lambda: files.folder
        connect_panes(self.panels["workspace"], files, self.panels["console"],
                      self.panels.get("editor"))
        files.scriptRequested.connect(self._open_script)
        files.projectRequested.connect(lambda path: self._call_handler("on_open_project", path))
        files.batchRequested.connect(lambda paths: self._call_handler("on_batch", paths))
        self.docks: dict = {}
        for key, title, what in self.PANES:
            dock = QDockWidget(title, self)
            dock.setObjectName(f"pane_{key}")
            dock.setToolTip(what)
            dock.setWidget(self.panels[key])
            dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
                             | QDockWidget.DockWidgetClosable)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock.setMinimumWidth(240)
            dock.dockLocationChanged.connect(lambda *_: self.remember_layout())
            dock.topLevelChanged.connect(lambda *_: self.remember_layout())
            dock.toggleViewAction().triggered.connect(lambda *_: self.remember_layout())
            dock.toggleViewAction().triggered.connect(
                lambda on, dock=dock: on and self._fit_pane(dock))
            self.docks[key] = dock
        self.docks["console"].visibilityChanged.connect(self._size_console)
        self._attach_console_panes()
        self._place_panes()

    def _place_panes(self) -> None:
        """The opening arrangement, Spyder's (see `onset_review.panes`): the
        Workspace and Files tabbed on the right, the Console under them;
        Files in front before a recording is open, the Workspace after; the
        Console closed until asked for, since it is the one pane that runs
        code and starting it costs a second of IPython."""
        from onset_review.panes import arrange

        arrange(self, self.docks, "Spyder")
        workspace, files = self.docks["workspace"], self.docks["files"]
        (workspace if self.loaded else files).raise_()
        self.resizeDocks([workspace], [380], Qt.Horizontal)
        shown = (not self.loaded) or self._wide_screen()
        for dock in (workspace, files):
            dock.setVisible(shown)
        self.docks["console"].setVisible(False)

    def apply_pane_layout(self, name: str) -> list[str]:
        """View → Pane layout: arrange the panes as `name` has them, keeping
        this window's sizes and its narrow-screen rule, and remember it.
        Returns the panes left showing."""
        from onset_review.panes import arrange

        self._building = True
        try:
            with self._panes_home(), self._keeping_size():
                shown = arrange(self, self.docks, name)
                if shown:
                    self.resizeDocks([self.docks["workspace"]], [380], Qt.Horizontal)
        finally:
            self._building = False
        if "console" in shown:
            self._size_console(True)
        floated = [key for key in shown if self._fit_pane(self.docks[key])]
        if floated and self.statusBar() is not None:
            self.statusBar().showMessage(
                "The screen is too narrow to dock "
                + ", ".join(self.docks[k].windowTitle() for k in floated)
                + " beside the page; opened as windows instead.", 8000)
        self.remember_layout()
        return shown

    @contextlib.contextmanager
    def _keeping_size(self):
        """Arranging shows every pane for a moment, which raises the window's
        minimum width, and Qt widens the window to match and never narrows it
        back: a 1366 px laptop window came out 1502 px wide. The size the
        window had is put back, unless what is left showing needs more."""
        maximised, size = self.isMaximized(), self.size()

        def restore() -> None:
            if not self.isVisible():
                return
            if maximised:
                self.showMaximized()
            else:
                self.resize(size.expandedTo(self.minimumSizeHint()))

        try:
            yield
        finally:
            # Now, and again once the event loop has turned: Qt applies the
            # widening when it next lays the window out, after this returns.
            restore()
            QTimer.singleShot(0, restore)

    # -- the sidebar ----------------------------------------------------------
    @property
    def sidebar_shown(self) -> bool:
        return not self._sidebar.isHidden()

    def set_sidebar(self, on: bool) -> None:
        """View → Page sidebar: the list of pages, hidden to give the page its
        width. Alt+1… and the View menu still move between pages."""
        if bool(on) == self.sidebar_shown:
            return
        self._sidebar.setVisible(bool(on))
        self.remember_layout()

    def _remembered_sidebar(self):
        import json

        try:
            payload = json.loads(self._layout_file().read_text(encoding="utf-8"))
            value = payload.get("sidebar")
            return None if value is None else bool(value)
        except (OSError, ValueError, TypeError):
            return None

    def _size_console(self, visible: bool) -> None:
        """A console opened under the Workspace gets half the column; Qt
        would give it the least it can, a few lines."""
        console, workspace = self.docks["console"], self.docks["workspace"]
        if (not visible or getattr(self, "_building", False) or console.isFloating()
                or workspace.isHidden() or console.height() >= 220):
            return
        column = workspace.height() + console.height()
        self.resizeDocks([workspace, console], [column // 2, column - column // 2],
                         Qt.Vertical)

    def _fit_pane(self, dock) -> bool:
        """A pane shown beside the pages on a screen too narrow for both
        opens as a window of its own instead: a window wider than its screen
        loses its maximise button on X11, which is worse than a floating
        pane. True if it was floated."""
        width = self._screen_width()
        if width is None or dock.isFloating():
            return False
        if self.minimumSizeHint().width() <= width:
            return False
        dock.setFloating(True)
        dock.resize(420, 560)
        return True

    def _screen_width(self) -> int | None:
        """The usable width of the screen this window opens on, or None."""
        from qtpy.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
        return None if screen is None else int(screen.availableGeometry().width())

    def _wide_screen(self) -> bool:
        width = self._screen_width()
        return width is not None and width >= self.PANES_MIN_WIDTH

    @property
    def _panes_key(self) -> str:
        return "panes" if self.loaded else "panes_start"

    def _restore_panes(self) -> bool:
        import base64
        import json

        try:
            payload = json.loads(self._layout_file().read_text(encoding="utf-8"))
            blob = payload.get(self._panes_key)
            return bool(blob) and self.restoreState(QByteArray(base64.b64decode(blob)))
        except (OSError, ValueError, TypeError):
            return False

    def _open_path(self, path: str) -> None:
        if self._on_open_path is not None:
            self._on_open_path(path)

    def pane(self, key: str):
        return getattr(self, "docks", {}).get(key)

    # -- regions a mouse can drag ------------------------------------------
    def _split(self, name: str, orientation, widgets: list, sizes: list[int],
               stretch: tuple[int, ...] | None = None) -> QSplitter:
        """A splitter between regions, registered so its sizes can be kept.

        Nothing collapses to zero: a region dragged shut is a region a
        reviewer cannot find again, and every one of these is worth having.
        """
        splitter = QSplitter(orientation)
        splitter.setObjectName(f"onset_split_{name}")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        for index, widget in enumerate(widgets):
            splitter.addWidget(widget)
            if stretch is not None:
                splitter.setStretchFactor(index, stretch[index])
        splitter.setSizes(sizes)
        splitter.splitterMoved.connect(lambda *_: self.remember_layout())
        self._splitters[name] = (splitter, list(sizes))
        return splitter

    def splitter(self, name: str) -> QSplitter | None:
        entry = self._splitters.get(name)
        return entry[0] if entry else None

    def layout_state(self) -> dict[str, bytes]:
        """Every splitter's sizes, to carry across a rebuild or a launch."""
        return {name: bytes(splitter.saveState().data())
                for name, (splitter, _default) in self._splitters.items()}

    # -- compact tables, everywhere at once ----------------------------------
    COMPACT_PANELS = ("findings", "events", "agreement", "quality")

    def set_compact(self, on: bool) -> None:
        """Every table to its compact form, or every column; remembered."""
        for key in self.COMPACT_PANELS:
            panel = self.panels.get(key)
            if panel is not None and hasattr(panel, "set_compact"):
                panel.set_compact(bool(on))
        self.remember_layout()

    def compact(self) -> bool:
        panel = self.panels.get("findings")
        return bool(getattr(panel, "compact", True))

    def restore_layout_state(self, state: dict | None) -> int:
        """Apply saved sizes to the splitters this window has. Returns how
        many took: a page not built in this state is simply left alone."""
        taken = 0
        for name, blob in (state or {}).items():
            entry = self._splitters.get(name)
            if entry is None or not blob:
                continue
            if entry[0].restoreState(QByteArray(bytes(blob))):
                taken += 1
        return taken

    def reset_layout(self) -> None:
        """Every region back to its opening size, and nothing remembered."""
        for splitter, default in self._splitters.values():
            splitter.setSizes(default)
        if getattr(self, "docks", None):
            self._building = True
            try:
                with self._panes_home(), self._keeping_size():
                    self._place_panes()
            finally:
                self._building = False
        self._sidebar.setVisible(True)
        self._forget()

    def _layout_file(self):
        from onset_review.assistant_config import config_path

        return config_path().with_name("layout.json")

    def _remembered(self) -> dict:
        import base64
        import json

        try:
            payload = json.loads(self._layout_file().read_text(encoding="utf-8"))
            return {k: base64.b64decode(v) for k, v in payload.get("splitters", {}).items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _remembered_compact(self):
        import json

        try:
            payload = json.loads(self._layout_file().read_text(encoding="utf-8"))
            value = payload.get("compact")
            return None if value is None else bool(value)
        except (OSError, ValueError, TypeError):
            return None

    def remember_layout(self) -> None:
        """Write the splitter sizes beside the assistant's defaults, so the
        next launch opens the way this one was left."""
        import base64
        import json

        if getattr(self, "_building", False):
            return
        try:
            path = self._layout_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            except (OSError, ValueError):
                payload = {}
            # Merged, not replaced: the start window has fewer regions than a
            # loaded one, and its save must not forget the others.
            splitters = dict(payload.get("splitters") or {})
            splitters.update({k: base64.b64encode(v).decode("ascii")
                              for k, v in self.layout_state().items()})
            payload.update({"schema": 1, "compact": self.compact(), "splitters": splitters,
                            "sidebar": self.sidebar_shown,
                            "study_folded": self.study_folded})
            if getattr(self, "docks", None) and not self.panes_lent:
                # While the Analysis page holds the panes their docks stand
                # hidden; that is not an arrangement to remember.
                payload[self._panes_key] = base64.b64encode(
                    bytes(self.saveState().data())).decode("ascii")
            path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        except OSError:
            pass        # a layout that cannot be saved is still a layout

    def _forget(self) -> None:
        try:
            self._layout_file().unlink()
        except OSError:
            pass

    @staticmethod
    def _titled(title: str, widget: QWidget) -> QWidget:
        return theme.card_frame(widget, title)

    @staticmethod
    def _on_card(widget: QWidget) -> QWidget:
        """A page that is one card: the card, with the page's margins. The
        page answers for its body (`page.view`, `page.refresh`), so nothing
        that held the body has to know it was put on a card."""
        return _CardPage(widget)

    # -- the public surface ----------------------------------------------
    def page_keys(self) -> list[str]:
        from onset_review.studies import STUDIES

        return ([key for key, _label in PAGES] + [key for key, _, _ in STUDIES]
                + [CHAT_PAGE[0]] + [key for key, _label in GUIDE_PAGES])

    def current_page(self) -> str:
        widget = self.stack.currentWidget()
        for key, page in self._pages.items():
            if page is widget:
                return key
        return ""

    def show_page(self, key: str) -> bool:
        item = self._items.get(key)
        if item is None:
            return False
        self.nav.setCurrentItem(item)
        return True

    def set_reader(self, text: str) -> None:
        self.reader.setText(text or "No reader named")
