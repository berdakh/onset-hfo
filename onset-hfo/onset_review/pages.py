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

Two desktop-only pages sit in the first group (Contacts, Quality) because they
need a running analysis. The six study pages are listed and disabled: they
are phase 2, and a sidebar that showed only half the site's pages would
misdescribe what the software is.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["PageWindow", "PAGES", "STUDY_PAGES", "DISCLAIMER"]

#: (key, sidebar label). The order is the site's.
PAGES = (
    ("home", "Home"),
    ("recording", "Recording"),
    ("contacts", "Contacts"),
    ("quality", "Quality"),
    ("report", "Report"),
    ("assistant", "Assistant"),
)

#: The site's cohort pages, listed so the sidebar describes the whole
#: product, and disabled until they are built (phase 2).
STUDY_PAGES = ("Detectors", "Outcome", "Patients", "Data", "Architecture",
               "Research")

#: Desktop-only pages, marked as such in the sidebar.
DESKTOP_ONLY = {"contacts", "quality"}

#: The site's disclaimer, word for word (`app/panels.py`). A test pins the
#: two copies to each other; it is duplicated rather than imported because the
#: wheel does not ship the Streamlit app.
DISCLAIMER = ("Research prototype — not a medical device. "
              "Real public recordings, real expert markings, real surgical "
              "outcomes — and nothing here is validated for clinical use. "
              "Every number cites the window it came from. There is no "
              "recommendation anywhere in this product; the clinician decides.")

#: The site's front door, trimmed to what applies on the desktop.
WHAT_THIS_IS = (
    "<b>What this is.</b> A working prototype built on two public archives of "
    "real patients. It detects high-frequency oscillations (ripples 80–250 Hz, "
    "fast ripples 250–500 Hz) and interictal discharges, scores itself against "
    "expert HFO markings on 20 patients, and tests its map against what "
    "happened to those patients after surgery. An assistant on an open-weight "
    "model reads the results, must cite them, and is refused by code when it "
    "strays.")
WHAT_IT_IS_NOT = (
    "<b>What it is not.</b> Not a diagnostic device, not validated on patients, "
    "not a seizure-onset-zone finder. A high event rate is a measurement; "
    "physiological ripples occur in healthy tissue. This software organises "
    "evidence. The clinician decides.")

#: How to read the pages, each entry a link to its page.
HOW_TO_READ = (
    ("recording", "channel ranking with both detectors side by side, "
                  "disagreement highlighted, and the signal behind any event"),
    ("contacts", "where the contacts are, ranked, relative to the resection"),
    ("quality", "which contacts and seconds were analysed, and what was done "
                "to the signal first"),
    ("report", "the structured, cited report: findings, your read, data "
               "quality, limitations"),
    ("assistant", "ask about a channel or the evidence; try asking what to "
                  "resect"),
)

SIDEBAR_WIDTH = 190
COLUMN_WIDTH = 320


class PageWindow(QMainWindow):
    """A sidebar of pages around the same panels the docked window uses."""

    #: The key of the page just shown.
    pageChanged = Signal(str)

    def __init__(self, figure, panels: dict, session, *, cached=None,
                 on_open_cached=None, on_import=None, on_place_contacts=None,
                 on_export=None, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.panels = panels
        self.session = session
        self._cached = cached
        self._on_open_cached = on_open_cached
        self._on_import = on_import
        self._on_place_contacts = on_place_contacts
        self._on_export = on_export
        self._pages: dict[str, QWidget] = {}
        self._items: dict[str, QListWidgetItem] = {}

        self.nav = QListWidget()
        self.nav.setObjectName("onset_pages")
        self.nav.setFixedWidth(SIDEBAR_WIDTH)
        self.nav.setFrameShape(QFrame.NoFrame)
        self.nav.setSpacing(1)
        self.stack = QStackedWidget()
        self.stack.setObjectName("onset_stack")

        self.banner = QLabel(DISCLAIMER)
        self.banner.setObjectName("onset_banner")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(theme.card("warn"))

        self._build_sidebar()
        self._build_pages()

        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(theme.SPACING, theme.SPACING, theme.SPACING, 0)
        column.setSpacing(theme.SPACING)
        column.addWidget(self.banner)
        column.addWidget(self.stack, 1)

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._sidebar)
        row.addWidget(right, 1)
        self.setCentralWidget(body)

        self.nav.currentItemChanged.connect(self._nav_changed)
        for index, (key, _label) in enumerate(PAGES, start=1):
            shortcut = QShortcut(QKeySequence(f"Alt+{index}"), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda key=key: self.show_page(key))
        self.show_page("home")

    # -- the sidebar ------------------------------------------------------
    def _build_sidebar(self) -> None:
        request = self.session.request
        self.where = QLabel(f"<b>{request.subject}</b><br>"
                            f"{request.t_start:g}–{request.t_stop:g} s · "
                            f"{request.band_label()}")
        self.where.setObjectName("onset_where")
        self.where.setWordWrap(True)
        self.reader = QLabel("No reader named")
        self.reader.setObjectName("onset_sidebar_reader")
        self.reader.setWordWrap(True)
        self.reader.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")

        def heading(text: str) -> None:
            item = QListWidgetItem(text.upper())
            item.setFlags(Qt.NoItemFlags)
            font = item.font()
            font.setPointSize(max(7, font.pointSize() - 2))
            font.setBold(True)
            item.setFont(font)
            item.setForeground(Qt.gray)
            self.nav.addItem(item)

        heading("This recording")
        for key, label in PAGES:
            item = QListWidgetItem(label + ("   desktop" if key in DESKTOP_ONLY else ""))
            item.setData(Qt.UserRole, key)
            item.setToolTip("Only on the desktop: needs a running analysis"
                            if key in DESKTOP_ONLY else "")
            self.nav.addItem(item)
            self._items[key] = item
        heading("The study")
        for label in STUDY_PAGES:
            item = QListWidgetItem(label)
            item.setFlags(Qt.NoItemFlags)
            item.setToolTip("The site's study pages come to the desktop in "
                            "phase 2; until then they are on the results site.")
            self.nav.addItem(item)

        sidebar = QWidget()
        sidebar.setObjectName("onset_sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        box = QVBoxLayout(sidebar)
        box.setContentsMargins(theme.SPACING, theme.SPACING, 0, theme.SPACING)
        box.setSpacing(4)
        box.addWidget(theme.section_label("Recording"))
        box.addWidget(self.where)
        box.addWidget(theme.section_label("Reader"))
        box.addWidget(self.reader)
        box.addWidget(self.nav, 1)
        self._sidebar = sidebar

    def _nav_changed(self, current, _previous) -> None:
        key = current.data(Qt.UserRole) if current is not None else None
        if key and key in self._pages:
            self.stack.setCurrentWidget(self._pages[key])
            refresh = getattr(self._pages[key], "refresh", None)
            if callable(refresh):
                refresh()
            self.pageChanged.emit(key)

    # -- the pages --------------------------------------------------------
    def _build_pages(self) -> None:
        builders = {
            "home": self._home_page, "recording": self._recording_page,
            "contacts": self._contacts_page, "quality": self._quality_page,
            "report": self._report_page, "assistant": self._assistant_page,
        }
        for key, _label in PAGES:
            page = builders[key]()
            page.setObjectName(f"page_{key}")
            self._pages[key] = page
            self.stack.addWidget(page)

    def _home_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setSpacing(theme.SPACING)
        title = QLabel("<h2>Onset Review</h2>")
        box.addWidget(title)
        for text in (WHAT_THIS_IS, WHAT_IT_IS_NOT):
            label = QLabel(text)
            label.setWordWrap(True)
            box.addWidget(label)

        summary = self.session.summary()
        numbers = QHBoxLayout()
        numbers.setSpacing(theme.SPACING)
        for value, caption in (
            (summary.get("n_channels", len(self.session.findings)), "channels analysed"),
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

        box.addWidget(theme.section_label("How to read the pages"))
        guide = QLabel("<ol>" + "".join(
            f"<li><a href='page:{key}'><b>{label}</b></a> — {what}</li>"
            for (key, label), (_key, what) in zip(PAGES[1:], HOW_TO_READ, strict=True))
            + "</ol>")
        guide.setWordWrap(True)
        guide.setOpenExternalLinks(False)
        guide.linkActivated.connect(
            lambda link: self.show_page(link.split(":", 1)[1]))
        box.addWidget(guide)

        box.addWidget(theme.section_label("Windows on this machine"))
        self.cached_table = QTableView()
        self.cached_table.setObjectName("onset_cached")
        self.cached_table.setSelectionBehavior(QTableView.SelectRows)
        self.cached_table.setSelectionMode(QTableView.SingleSelection)
        self.cached_table.verticalHeader().setVisible(False)
        self.cached_table.doubleClicked.connect(lambda _index: self._open_selected())
        box.addWidget(self.cached_table, 1)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open the selected window")
        self.open_button.setObjectName("onset_open_cached")
        self.open_button.setEnabled(self._on_open_cached is not None)
        self.open_button.clicked.connect(self._open_selected)
        buttons.addWidget(self.open_button)
        self.import_button = QPushButton("Open a file…")
        self.import_button.setEnabled(self._on_import is not None)
        if self._on_import is not None:
            self.import_button.clicked.connect(lambda _=False: self._on_import())
        buttons.addWidget(self.import_button)
        buttons.addStretch(1)
        box.addLayout(buttons)
        box.addWidget(theme.muted(
            "Windows are fetched once and kept. To add another minute of a "
            "patient, from a terminal:  onset-hfo fetch --subject sub-02 "
            "--t-start 0 --t-stop 60"))
        page.refresh = self._refresh_cached      # type: ignore[attr-defined]
        self._refresh_cached()
        return page

    def _refresh_cached(self) -> None:
        import pandas as pd

        from onset_review.panels import DataFrameModel

        frame = None
        if self._cached is not None:
            try:
                frame = self._cached()
            except Exception:       # noqa: BLE001 - a listing must not kill a window
                frame = None
        if frame is None or len(frame) == 0:
            frame = pd.DataFrame(columns=["subject", "t_start", "t_stop", "dataset"])
        columns = [c for c in ("subject", "t_start", "t_stop", "dataset", "run",
                               "sfreq", "n_channels") if c in frame.columns]
        self._cached_frame = frame
        self.cached_table.setModel(DataFrameModel(frame[columns].reset_index(drop=True)))
        self.cached_table.resizeColumnsToContents()

    def _open_selected(self) -> None:
        if self._on_open_cached is None:
            return
        rows = self.cached_table.selectionModel().selectedRows() \
            if self.cached_table.selectionModel() else []
        if not rows:
            return
        row = self._cached_frame.iloc[rows[0].row()].to_dict()
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
        # The site's first question, answered in the site's colours.
        self.leader = QLabel(self.session.caveat())
        self.leader.setObjectName("onset_leader")
        self.leader.setWordWrap(True)
        self.leader.setStyleSheet(theme.card(
            "info" if self.session.leader.get("distinguishable") else "bad"))
        box.addWidget(QLabel("<b>Does any channel actually stand out?</b>"))
        box.addWidget(self.leader)

        self.trend_toggle = QToolButton()
        self.trend_toggle.setText("Trend — rate per channel over time")
        self.trend_toggle.setCheckable(True)
        self.trend_toggle.setChecked(True)
        self.trend_toggle.setArrowType(Qt.DownArrow)
        self.trend_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.trend_toggle.setAutoRaise(True)
        self.trend_toggle.toggled.connect(self._toggle_trend)
        box.addWidget(self.trend_toggle)
        self.panels["trends"].setMinimumHeight(100)
        self.panels["trends"].setMaximumHeight(220)
        box.addWidget(self.panels["trends"])
        box.addWidget(self.panels["controls"])
        # MNE's browser is a QMainWindow of its own; given a parent it becomes
        # an ordinary child widget, toolbar and all.
        self.figure.setParent(left)
        box.addWidget(self.figure, 1)
        outer.addWidget(left, 1)

        self.side = QTabWidget()
        self.side.setObjectName("onset_side")
        self.side.setFixedWidth(COLUMN_WIDTH + 40)
        self.side.addTab(self.panels["findings"], "Ranking")
        self.side.addTab(self.panels["events"], "Events")
        self.side.addTab(self.panels["detail"], "This event")
        self.side.setTabToolTip(0, "Channels ranked by rate — evidence, not a "
                                   "recommendation. Tinted rows are tied with "
                                   "the busiest.")
        self.side.setTabToolTip(2, "The selected event wideband, filtered and in "
                                   "time-frequency — oscillation or filter ringing?")
        outer.addWidget(self.side)
        return page

    def _toggle_trend(self, on: bool) -> None:
        self.panels["trends"].setVisible(bool(on))
        self.trend_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def _contacts_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING)
        row.addWidget(self.panels["brain"], 1)

        column = QWidget()
        column.setFixedWidth(COLUMN_WIDTH)
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
        box.addStretch(1)
        row.addWidget(column)
        return page

    def _quality_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        split = QSplitter(Qt.Horizontal)
        split.setObjectName("onset_quality_split")
        split.addWidget(self._titled("Data quality — which contacts and which "
                                     "seconds were analysed",
                                     self.panels["quality"]))
        split.addWidget(self._titled("Preprocessing — what is done to the signal "
                                     "before any detector sees it",
                                     self.panels["preprocess"]))
        split.setSizes([1, 1])
        box.addWidget(split, 3)
        box.addWidget(self._titled("Provenance — how this was produced, step by step",
                                   self.panels["provenance"]), 2)
        return page

    def _report_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SPACING)
        self.report_view = QTextBrowser()
        self.report_view.setObjectName("onset_report")
        self.report_view.setOpenExternalLinks(False)
        row.addWidget(self._titled("The review as it will be exported",
                                   self.report_view), 1)

        column = QWidget()
        column.setFixedWidth(COLUMN_WIDTH)
        box = QVBoxLayout(column)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        box.addWidget(theme.section_label("Agreement with the archive's annotators"))
        box.addWidget(self.panels["agreement"], 1)
        self.export_button = QPushButton("Export review…")
        self.export_button.setObjectName("onset_export")
        self.export_button.setEnabled(self._on_export is not None)
        if self._on_export is not None:
            self.export_button.clicked.connect(lambda _=False: self._on_export())
        box.addWidget(self.export_button)
        box.addWidget(theme.muted("Markdown or a web page. Your verdicts and notes "
                                  "go in under your name; name yourself under "
                                  "Read first."))
        row.addWidget(column)
        page.refresh = self.refresh_report      # type: ignore[attr-defined]
        return page

    def refresh_report(self) -> None:
        """Rebuild the preview from the session as it is now, verdicts and all."""
        from onset_review import report

        reader = getattr(getattr(self.session, "read", None), "reader", "") or None
        try:
            text = report.review_markdown(self.session, reviewer=reader)
            self.report_view.setHtml(report._as_html(text, self.session))
        except Exception as error:      # noqa: BLE001 - a preview must not kill a page
            self.report_view.setPlainText(f"The report could not be rendered: {error}")

    def _assistant_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.panels["assistant"])
        return page

    @staticmethod
    def _titled(title: str, widget: QWidget) -> QWidget:
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(theme.section_label(title))
        box.addWidget(widget, 1)
        return holder

    # -- the public surface ----------------------------------------------
    def page_keys(self) -> list[str]:
        return [key for key, _label in PAGES]

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
