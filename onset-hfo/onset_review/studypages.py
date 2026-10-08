"""The study pages as Qt widgets: the key message, the choices, the text.

Each page is :func:`onset_review.studies.build` rendered by a `QTextBrowser`,
which reads GitHub-flavoured Markdown natively -- tables, images, code -- so
no HTML is written by hand and no extra package is needed. The controls the
site offers (band, metric, patient) are combo boxes above the text, and a
change rebuilds the document from the committed tables.

Three things are done for someone who is new to the field and reads the page
once:

* **The key message first.** A card at the top says in two or three plain
  sentences what the page shows, with up to four numbers as tiles. It is
  computed from the same tables the page renders (`studies.key_message`),
  so it cannot drift from the page.
* **A reading layout.** The text sits in a column no wider than a book's
  page, in a slightly larger type, with tables stripped of their rules and
  the long ones folded until asked for. A glossary term on the page gets a
  dotted underline and its definition as a tooltip.
* **Ask about this page.** A box under the text puts a question to the
  local model with this page's sections as the only material; every number
  in the answer must be on the page. With no model loaded the box still
  answers from the page itself.

The Patients page carries the one link the site cannot make: a patient picked
there can be opened on the Recording page, if a window of theirs is cached.

And the pages answer to the analysis that is open, without a published
number changing (`onset_review.thisrecording`, `onset_review.yourstudy`):

* **This recording** under each title: the open window set against the study,
  measured the same way, and said to be one window of one patient.
* **Explore the chart** opens the page's figure live (`onset_review.chartview`):
  hover for values, click a patient to open them, drag the threshold.
* **Re-run with your settings** runs the Detectors or the Outcome study again
  with the reader's band, detectors, thresholds and window, in the
  background, and shows the result dashed beside the published lines.
* **Add this recording to your cohort** measures the open window as the
  Outcome study measures a patient, with the resection and outcome the
  reader enters.
* **Open as notebook** writes the page as a notebook that rebuilds it
  (`onset_review.studynotebooks`) and opens it on the Analysis page.
"""

from __future__ import annotations

import html
import re

from qtpy.QtCore import QRegularExpression, Qt, QThread, QUrl, Signal
from qtpy.QtGui import QFont, QImage, QTextCharFormat, QTextDocument
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_agent import pageask
from onset_agent.backends import make_backend
from onset_review import studies, theme
from onset_review.assistant import ANSWER_TIMEOUT, SMALL_PT, _run
from onset_review.assistant_config import load_defaults

__all__ = ["StudyPage", "ReadingView", "set_markdown", "annotate_terms", "style_tables",
           "READING_WIDTH"]


_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

#: Widest an embedded figure is drawn, in pixels. The site's figures are
#: rendered at print width and would otherwise push the page sideways.
IMAGE_MAX_WIDTH = 1100

#: The reading column, in pixels: about 85 characters of 11 pt type. Past
#: that the eye loses the line on the way back.
READING_WIDTH = 880

#: Glossary definitions are looked up once per process.
_GLOSSARY: dict[str, str] | None = None


def _glossary() -> dict[str, str]:
    global _GLOSSARY
    if _GLOSSARY is None:
        try:
            from onset_agent import knowledge

            _GLOSSARY = knowledge.glossary()
        except Exception:       # noqa: BLE001 - no documents, no tooltips
            _GLOSSARY = {}
    return _GLOSSARY


def document_stylesheet(palette=None) -> str:
    """The reading style: type that carries the hierarchy, tables without
    rules, quiet captions. Qt's rich text honours this subset of CSS."""
    p = palette or theme.current()
    return (
        f"h1 {{ font-size: 19pt; font-weight: 600; margin-bottom: 6px; }} "
        f"h2 {{ font-size: 14pt; font-weight: 600; margin-top: 18px; margin-bottom: 4px; }} "
        f"h3 {{ font-size: 12pt; font-weight: 600; margin-top: 14px; margin-bottom: 2px; }} "
        f"p, li {{ margin-top: 3px; margin-bottom: 7px; }} "
        f"table {{ border-collapse: collapse; margin-top: 6px; margin-bottom: 10px; }} "
        f"th {{ padding: 4px 10px; border-bottom: 2px solid {p.separator}; "
        f"color: {p.text_muted}; font-weight: 600; text-align: left; }} "
        f"td {{ padding: 4px 10px; border-bottom: 1px solid {p.surface_alt}; }} "
        f"blockquote {{ color: {p.text_muted}; }} "
        f"a {{ color: {p.accent}; }} "
        f"code {{ font-size: 10pt; }}")


def set_markdown(browser: QTextBrowser, text: str) -> None:
    """GitHub-flavoured Markdown into a browser, with the dialect spelled out:
    the default dialect has no tables, and a page of tables that renders as
    pipes and dashes would look like a bug rather than a setting.

    Images are loaded here and scaled to the view, because a rich-text
    document draws a figure at its stored size and the study figures are
    wider than any screen.
    """
    document = browser.document()
    width = max(320, min(IMAGE_MAX_WIDTH, browser.viewport().width() - 40))
    images: list[tuple[str, QImage]] = []

    def place(match):
        alt, path = match.group(1), match.group(2)
        image = QImage(path)
        if image.isNull():
            return match.group(0)
        if image.width() > width:
            image = image.scaledToWidth(width, Qt.SmoothTransformation)
        name = f"onset-figure-{abs(hash(path))}"
        images.append((name, image))
        return f"![{alt}]({name})"

    text = _IMAGE.sub(place, text)
    try:
        document.setMarkdown(text, QTextDocument.MarkdownDialectGitHub)
    except (AttributeError, TypeError):      # an older binding
        browser.setMarkdown(text)
    # The resources go in after the text: setting the document's text clears
    # what it cached, so a figure added before it was a broken-image icon.
    for name, image in images:
        document.addResource(QTextDocument.ImageResource, QUrl(name), image)
    if images:
        document.markContentsDirty(0, document.characterCount())
    style_tables(document)


def style_tables(document: QTextDocument, palette=None) -> int:
    """Tables without a grid: a rule under the header, a hairline under each
    row, the header in the muted colour. Rich text ignores per-side CSS
    borders, so this walks the document's tables and sets the formats
    directly. Returns how many tables were styled."""
    from qtpy.QtGui import QBrush, QColor, QTextFrameFormat, QTextTableCellFormat

    p = palette or theme.current()
    count = 0

    def walk(frame):
        nonlocal count
        for child in frame.childFrames():
            table = child if hasattr(child, "cellAt") else None
            if table is not None:
                form = table.format()
                form.setBorder(0)
                form.setBorderStyle(QTextFrameFormat.BorderStyle_None)
                form.setCellSpacing(0)
                form.setCellPadding(5)
                form.setTopMargin(4)
                form.setBottomMargin(8)
                table.setFormat(form)
                for row in range(table.rows()):
                    for column in range(table.columns()):
                        cell = table.cellAt(row, column)
                        form = QTextTableCellFormat(cell.format().toTableCellFormat()
                                                    if hasattr(cell.format(), "toTableCellFormat")
                                                    else cell.format())
                        form.setBottomBorder(2 if row == 0 else 1)
                        form.setBottomBorderStyle(QTextFrameFormat.BorderStyle_Solid)
                        form.setBottomBorderBrush(QBrush(QColor(
                            p.separator if row == 0 else p.surface_alt)))
                        if row == 0:
                            form.setForeground(QBrush(QColor(p.text_muted)))
                        cell.setFormat(form)
                count += 1
            walk(child)

    walk(document.rootFrame())
    return count


def annotate_terms(document: QTextDocument, glossary: dict[str, str],
                   palette=None, limit: int = 40) -> int:
    """Give every glossary term on the page a dotted underline and its
    definition as a tooltip. Returns how many occurrences were marked.

    Whole words, case-insensitive, longest terms first so "fast ripple" is
    marked before "ripple" inside it. Headings are left alone: a definition
    hanging off a title reads as a correction.
    """
    p = palette or theme.current()
    marked = 0
    for term in sorted(glossary, key=len, reverse=True):
        pattern = QRegularExpression(r"\b" + re.escape(term) + r"\b")
        pattern.setPatternOptions(QRegularExpression.CaseInsensitiveOption)
        cursor = document.find(pattern)
        seen = 0
        while not cursor.isNull() and seen < limit:
            block = cursor.block()
            if block.blockFormat().headingLevel() == 0 and \
                    not cursor.charFormat().toolTip():
                form = QTextCharFormat()
                form.setToolTip(glossary[term])
                form.setUnderlineStyle(QTextCharFormat.DotLine)
                form.setUnderlineColor(p.text_muted)
                cursor.mergeCharFormat(form)
                marked += 1
            seen += 1
            cursor = document.find(pattern, cursor)
    return marked


class ReadingView(QTextBrowser):
    """A text browser that keeps its text in a reading column.

    The column is `READING_WIDTH` at most and centred: the viewport margins
    grow with the window so a 1,600-pixel page does not become 1,600-pixel
    lines. A figure wider than the column is scaled by `set_markdown`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QTextBrowser.NoFrame)
        self.setOpenExternalLinks(True)
        self.setOpenLinks(True)
        font = QFont(self.font())
        font.setPointSizeF(11.0)
        self.document().setDefaultFont(font)
        self.document().setDefaultStyleSheet(document_stylesheet())
        self.document().setDocumentMargin(theme.SPACING + 4)

    def resizeEvent(self, event):      # noqa: N802  (Qt's spelling)
        margin = max(0, (self.width() - READING_WIDTH) // 2)
        self.setViewportMargins(margin, 0, margin, 0)
        super().resizeEvent(event)


class _AskWorker(QThread):
    """One question about one page, off the GUI thread."""

    def __init__(self, question: str, title: str, text: str, lead: str, parent=None):
        super().__init__(parent)
        self._question, self._title, self._text, self._lead = question, title, text, lead
        self._backend = None
        self.answer = None
        self.error: str | None = None

    def stop(self) -> None:
        backend = self._backend
        if backend is not None:
            backend.abort()

    def run(self) -> None:
        try:
            defaults = load_defaults()
            kind = str(getattr(defaults, "kind", "scripted") or "scripted")
            self._backend = make_backend(
                kind, model=str(getattr(defaults, "model", "") or "") or None,
                base_url=str(getattr(defaults, "base_url", "") or "") or None,
                max_tokens=400, timeout=ANSWER_TIMEOUT)
            self.answer = pageask.answer(self._backend, self._question, self._title,
                                         self._text, self._lead)
        except Exception as error:      # noqa: BLE001 - reported in the box
            self.error = str(error)


class PageAsk(QWidget):
    """The box under a study page: a question, an answer from the page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        tokens = theme.current()
        self._title, self._text, self._lead = "", "", ""
        self._worker = None
        self.question = QLineEdit()
        self.question.setObjectName("onset_page_question")
        self.question.setPlaceholderText(
            "Ask about this page — summarise it, or ask what a term or a number means")
        self.question.returnPressed.connect(lambda: self.ask())
        self.ask_button = QPushButton("Ask")
        self.ask_button.setObjectName("onset_page_ask")
        self.ask_button.clicked.connect(lambda _=False: self.ask())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self.stop())
        self.answer = QLabel("")
        self.answer.setObjectName("onset_page_answer")
        self.answer.setWordWrap(True)
        self.answer.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.answer.setStyleSheet(theme.card("info"))
        self.answer.setVisible(False)
        self.verdict = QLabel("")
        self.verdict.setObjectName("onset_page_verdict")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet(f"color:{tokens.text_muted};font-size:{SMALL_PT}pt;"
                                   "background:transparent;border:none;")
        self.verdict.setVisible(False)

        chips = QHBoxLayout()
        chips.setSpacing(4)
        chips.addWidget(theme.section_label("Ask about this page"))
        chips.addStretch(1)
        for text in pageask.SUGGESTIONS:
            chip = QPushButton(text.rstrip(".").split(",")[0])
            chip.setToolTip(text)
            chip.setProperty("suggested", text)
            chip.clicked.connect(lambda _=False, q=text: self.ask(q))
            chips.addWidget(chip)
        row = QHBoxLayout()
        row.setSpacing(theme.SPACING // 2)
        row.addWidget(self.question, 1)
        row.addWidget(self.ask_button)
        row.addWidget(self.stop_button)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING // 2)
        box.addLayout(chips)
        box.addLayout(row)
        box.addWidget(self.answer)
        box.addWidget(self.verdict)
        self.ask_button.setToolTip(
            "Answered from this page only: every number in the answer must be on it, "
            "or the answer is marked as refused. With no model loaded the box answers "
            "with the page's own words.")

    def set_page(self, title: str, text: str, lead: str) -> None:
        self._title, self._text, self._lead = title, text, lead

    def ask(self, question: str | None = None):
        text = (question if isinstance(question, str) and question
                else self.question.text()).strip()
        if not text or not self._text:
            return None
        self.question.setText(text)
        self._worker = _AskWorker(text, self._title, self._text, self._lead, self)
        self._busy(True)
        try:
            _run(self._worker)
        finally:
            self._busy(False)
        if self._worker.error:
            self._show_text(f"The model could not answer: {self._worker.error}",
                            "not answered", "bad")
            return None
        answer = self._worker.answer
        if answer is None:
            return None
        if answer.refused:
            self._show_text(answer.text, f"Refused — {answer.reason}.", "bad")
        else:
            sources = ", ".join(label for label, _ in answer.sources[:3])
            self._show_text(answer.text, f"Checked against this page. From: {sources}.",
                            "info")
        return answer

    def stop(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()

    def _show_text(self, text: str, verdict: str, kind: str) -> None:
        self.answer.setText(html.escape(text))
        self.answer.setStyleSheet(theme.card(kind))
        self.answer.setVisible(True)
        self.verdict.setText(verdict)
        self.verdict.setVisible(True)

    def _busy(self, on: bool) -> None:
        self.question.setEnabled(not on)
        self.ask_button.setEnabled(not on)
        self.stop_button.setEnabled(on)
        if on:
            self._show_text("Thinking…", "", "plain")


class StudyPage(QWidget):
    """One study page: the key message, the choices, the text, the ask box."""

    #: A cached window row the reviewer asked to open, from the Patients page.
    openRequested = Signal(dict)
    #: A notebook written by Open as notebook, to open in the editor.
    notebookRequested = Signal(str)

    def __init__(self, key: str, *, cached=None, session=None, parent=None):
        super().__init__(parent)
        if key not in {k for k, _, _ in studies.STUDIES}:
            raise KeyError(f"no study page {key!r}")
        self.key = key
        self._cached = cached
        #: The recording open in the window, or None: the page's *This
        #: recording* layer is drawn from it.
        self.session = session
        self.chart_windows: list = []
        #: Replaces the study's own runner (tests): called with the settings.
        self.study_runner = None
        self._built = False
        self._text = ""
        self.combos: dict[str, QComboBox] = {}
        tokens = theme.current()

        # -- the key message -------------------------------------------------
        self.lead = QLabel("")
        self.lead.setObjectName(f"onset_key_{key}")
        self.lead.setWordWrap(True)
        self.lead.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lead.setStyleSheet(f"color:{tokens.text};font-size:11pt;"
                                "background:transparent;border:none;")
        self.tiles = QHBoxLayout()
        self.tiles.setSpacing(theme.SPACING)
        self._tile_widgets: list[QLabel] = []
        key_body = QWidget()
        key_box = QVBoxLayout(key_body)
        key_box.setContentsMargins(0, 0, 0, 0)
        key_box.setSpacing(theme.SPACING)
        key_box.addWidget(self.lead)
        key_box.addLayout(self.tiles)
        self.key_card = theme.card_frame(key_body, "What this page shows")
        self.key_card.setObjectName("onset_key_card")

        # -- the choices -------------------------------------------------------
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(theme.SPACING)
        for name, (label, options, default) in studies.controls(key).items():
            combo = QComboBox()
            combo.setObjectName(f"onset_study_{key}_{name}")
            for value, text in options.items():
                combo.addItem(text, value)
            index = combo.findData(default)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(lambda _i: self.rebuild())
            top.addWidget(QLabel(label))
            top.addWidget(combo)
            self.combos[name] = combo
        self.open_button = None
        if key == "patients":
            self.open_button = QPushButton("Open this patient's window")
            self.open_button.setObjectName("onset_study_open_patient")
            self.open_button.setToolTip("Open the earliest cached window of the "
                                        "patient picked above on the Recording page")
            self.open_button.clicked.connect(self._open_patient)
            top.addWidget(self.open_button)
        # What a reader can do with the page, beyond reading it.
        self.actions: dict[str, QPushButton] = {}

        def action(name: str, text: str, tip: str, slot) -> None:
            button = QPushButton(text)
            button.setObjectName(f"onset_study_{key}_{name}")
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False: slot())
            self.actions[name] = button

        if key in ("detectors", "outcome"):
            action("explore", "Explore the chart", "The page's figure in a window that "
                   "answers the pointer: hover for values, click a patient to open them"
                   + (", drag the threshold" if key == "detectors" else ""),
                   self.explore_chart)
            action("rerun", "Re-run with your settings…", "Run this study again with your "
                   "band, detectors, thresholds and window; the result is shown beside the "
                   "published one, labelled yours", self.rerun_dialog)
        if key == "outcome":
            action("add", "Add this recording to your cohort…", "Measure the open window "
                   "the way the study measures a patient, with the resection and outcome "
                   "you enter", self.add_to_cohort_dialog)
            action("remove", "Remove…", "Take a patient out of your cohort",
                   self.remove_from_cohort_dialog)
        action("notebook", "Open as notebook", "Write this page as a notebook that "
               "rebuilds it from the same tables, and open it on the Analysis page",
               self.open_notebook)
        top.addStretch(1)
        self.show_tables = QCheckBox("Show every table")
        self.show_tables.setObjectName(f"onset_study_{key}_tables")
        self.show_tables.setToolTip(f"Tables longer than {studies.FOLD_ROWS} rows are folded "
                                    "until asked for; the figures and the short tables "
                                    "always show")
        self.show_tables.toggled.connect(lambda _on: self.rebuild())
        top.addWidget(self.show_tables)

        # -- the text ----------------------------------------------------------
        self.view = ReadingView()
        self.view.setObjectName(f"onset_study_{key}")

        # -- the ask box -------------------------------------------------------
        self.ask = PageAsk()

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        box.addWidget(self.key_card)
        box.addLayout(top)
        if self.actions:
            row = theme.flow_layout()
            for button in self.actions.values():
                row.addWidget(button)
            holder = QWidget()
            holder.setLayout(row)
            self.note = QLabel("")
            self.note.setObjectName(f"onset_study_{key}_note")
            self.note.setWordWrap(True)
            self.note.setStyleSheet(f"color:{tokens.text_muted};font-size:9pt;")
            self.note.hide()
            box.addWidget(holder)
            box.addWidget(self.note)
        box.addWidget(self.view, 1)
        box.addWidget(self.ask)

    # -- choices -----------------------------------------------------------
    def choices(self) -> dict:
        return {name: combo.currentData() for name, combo in self.combos.items()}

    def _fill_subjects(self) -> None:
        combo = self.combos.get("subject")
        if combo is None or combo.count():
            return
        rows = studies.patient_rows(self.choices().get("band", "fast_ripple"),
                                    self.choices().get("scope", "reviewed"))
        if rows is None or len(rows) == 0:
            return
        combo.blockSignals(True)
        for subject in rows["subject"]:
            combo.addItem(str(subject), str(subject))
        from onset_review import thisrecording

        if thisrecording.in_cohort(self.session):
            index = combo.findData(self.session.request.subject)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    # -- building ----------------------------------------------------------
    def refresh(self) -> None:
        """Build on first show only: the tables are read once, and a page a
        reviewer never opens costs nothing."""
        if not self._built:
            self.rebuild()

    def rebuild(self) -> None:
        self._fill_subjects()
        choices = self.choices()
        try:
            text = studies.build(self.key, session=self.session, **choices)
        except Exception as error:      # noqa: BLE001 - a study page must not kill the window
            text = (f"# {self.key.title()}\n\n> **This page could not be built:** "
                    f"{error}\n\nThe same page is on the results site: {studies.SITE}")
        self._text = text
        message = studies.key_message(self.key, **choices)
        self._say_key_message(message)
        shown = text if self.show_tables.isChecked() else studies.fold_tables(text)[0]
        set_markdown(self.view, shown)
        annotate_terms(self.view.document(), _glossary())
        title = next((label for k, label, _ in studies.STUDIES if k == self.key), self.key)
        self.ask.set_page(title, text, message.get("lead", ""))
        self._built = True
        self._enable_actions()
        if self.open_button is not None:
            self.open_button.setEnabled(self._window_for_current() is not None)
            self.open_button.setToolTip(
                "Open the earliest cached window of this patient on the Recording page"
                if self.open_button.isEnabled() else
                "No window of this patient is cached on this machine; fetch one "
                "with onset-hfo fetch")

    @property
    def text(self) -> str:
        """The page's Markdown as built, whole, folded tables and all."""
        return self._text

    def _say_key_message(self, message: dict) -> None:
        self.lead.setText(message.get("lead", ""))
        for widget in self._tile_widgets:
            self.tiles.removeWidget(widget)
            widget.deleteLater()
        self._tile_widgets = []
        while self.tiles.count():
            item = self.tiles.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        tokens = theme.current()
        for value, caption in message.get("tiles", []):
            tile = QLabel(f"<div style='font-size:17pt;font-weight:700;'>{html.escape(str(value))}"
                          f"</div><div style='font-size:9pt;color:{tokens.text_muted};'>"
                          f"{html.escape(str(caption))}</div>")
            tile.setObjectName("onset_tile")
            tile.setStyleSheet(f"background:{tokens.surface_alt};border-radius:{theme.RADIUS}px;"
                               f"padding:{theme.SPACING}px {theme.SPACING + 4}px;")
            self.tiles.addWidget(tile)
            self._tile_widgets.append(tile)
        self.tiles.addStretch(1)

    # -- the link the site cannot make ---------------------------------------
    def _window_for_current(self) -> dict | None:
        subject = self.choices().get("subject") or ""
        if not subject or self._cached is None:
            return None
        try:
            return studies.cached_window_for(subject, self._cached())
        except Exception:       # noqa: BLE001
            return None

    def _open_patient(self) -> None:
        row = self._window_for_current()
        if row is not None:
            self.openRequested.emit(row)

    # -- doing things with the page ------------------------------------------------
    def _enable_actions(self) -> None:
        from onset_review import yourstudy

        if "add" in self.actions:
            self.actions["add"].setEnabled(self.session is not None)
            if self.session is None:
                self.actions["add"].setToolTip("Open a recording first")
        if "remove" in self.actions:
            self.actions["remove"].setEnabled(bool(yourstudy.cohort_entries()))

    def say(self, text: str) -> None:
        if hasattr(self, "note"):
            self.note.setText(text)
            self.note.setVisible(bool(text))

    def explore_chart(self):
        """The page's figure, live. Returns the window, or None."""
        from onset_review.chartview import ChartWindow

        data = studies.chart_data(self.key, session=self.session, **self.choices())
        if data is None:
            self.say("This page's tables are not on this machine, so there is no chart.")
            return None
        title = next((label for k, label, _ in studies.STUDIES if k == self.key), self.key)
        window = ChartWindow(data["kind"], data, title=f"{title} — chart")
        window.subjectClicked.connect(self._open_subject)
        self.chart_windows = [w for w in self.chart_windows if not w.isHidden()] + [window]
        window.show()
        return window

    def _open_subject(self, subject: str) -> None:
        row = None
        if self._cached is not None:
            try:
                row = studies.cached_window_for(subject, self._cached())
            except Exception:       # noqa: BLE001
                row = None
        if row is None:
            self.say(f"No window of {subject} is cached on this machine; fetch one with "
                     "onset-hfo fetch.")
            return
        self.openRequested.emit(row)

    def rerun_dialog(self) -> dict | None:
        dialog = RerunDialog(self.key, self.session, self)
        if dialog.exec() != QDialog.Accepted:
            return None
        return self.run_study(dialog.settings())

    def run_study(self, settings: dict) -> dict | None:
        """Run the study with `settings` (as `RerunDialog.settings` gives
        them) behind a progress dialog with a Stop button, then rebuild the
        page with the result beside the published one."""
        from onset_review import yourstudy

        runner = self.study_runner
        if runner is None:
            function = (yourstudy.rerun_detectors if self.key == "detectors"
                        else yourstudy.rerun_outcome)

            def runner(settings, progress=None, should_stop=None):
                arguments = {k: v for k, v in settings.items() if k != "offline"}
                return function(**arguments, progress=progress, should_stop=should_stop)

        progress = QProgressDialog("Starting…", "Stop", 0, max(1, len(settings["subjects"])),
                                   self)
        progress.setWindowTitle("Re-running the study with your settings")
        progress.setMinimumDuration(0)
        worker = _StudyWorker(runner, settings, self)
        worker.progressed.connect(lambda i, n, subject: (
            progress.setValue(i), progress.setLabelText(f"{subject} ({i + 1} of {n})")))
        progress.canceled.connect(worker.stop)
        from onset_review.workers import run

        run(worker)
        progress.close()
        if worker.error:
            self.say(f"The re-run failed: {worker.error}")
            return None
        result = worker.result or {}
        skipped = result.get("skipped") or {}
        ran = len(settings["subjects"]) - len(skipped)
        self.say(f"Re-run on {ran} of {len(settings['subjects'])} patients"
                 + (f"; skipped {len(skipped)} (named on the page)" if skipped else "")
                 + ". It is under *Your re-run* below, and dashed on the chart.")
        self.rebuild()
        return result

    def add_to_cohort_dialog(self) -> dict | None:
        if self.session is None:
            return None
        dialog = CohortDialog(self.session, self)
        if dialog.exec() != QDialog.Accepted:
            return None
        return self.add_to_cohort(*dialog.values())

    def add_to_cohort(self, outcome: str, resected, label: str = "") -> dict | None:
        from onset_review import yourstudy

        try:
            entry = yourstudy.add_to_cohort(self.session, outcome, resected, label)
        except ValueError as error:
            self.say(str(error))
            return None
        self.say(f"{entry['label']} is in your cohort: {entry['n_candidates']} tied channel(s), "
                 f"{entry['candidates_resected']:.0%} of them inside the resection you "
                 "entered.")
        self.rebuild()
        return entry

    def remove_from_cohort_dialog(self) -> bool:
        from onset_review import yourstudy

        entries = yourstudy.cohort_entries()
        if not entries:
            return False
        labels = [f"{e.get('label')} — {e.get('recording')} "
                  f"{e['window_s'][0]:g}–{e['window_s'][1]:g} s" for e in entries]
        chosen, ok = QInputDialog.getItem(self, "Remove from your cohort",
                                          "Patient to remove:", labels, 0, False)
        if not ok:
            return False
        removed = yourstudy.remove_from_cohort(entries[labels.index(chosen)]["id"])
        self.rebuild()
        return removed

    def open_notebook(self, folder=None):
        """Write the page as a notebook in the Files pane's folder and ask the
        window to open it. Returns its path."""
        from onset_review.files import current_folder
        from onset_review.studynotebooks import write_study_notebook

        try:
            path = write_study_notebook(self.key, folder or current_folder(), self._text,
                                        **self.choices())
        except OSError as error:
            self.say(f"Could not write the notebook: {error}")
            return None
        self.say(f"Wrote {path}.")
        self.notebookRequested.emit(str(path))
        return path

    def close_windows(self) -> None:
        for window in self.chart_windows:
            window.close()
        self.chart_windows = []

    def keyPressEvent(self, event):      # noqa: N802  (Qt's spelling)
        if event.key() == Qt.Key_F5:
            self.rebuild()
            return
        super().keyPressEvent(event)


class _StudyWorker(QThread):
    """A re-run of a study, off the GUI thread, one patient at a time."""

    progressed = Signal(int, int, str)

    def __init__(self, runner, settings: dict, parent=None):
        super().__init__(parent)
        self._runner, self._settings = runner, settings
        self._stop = False
        self.result: dict | None = None
        self.error: str | None = None

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import os

        from onset_hfo.datasets import OFFLINE_ENV

        previous = os.environ.get(OFFLINE_ENV)
        if self._settings.get("offline"):
            # Only what is on this machine: the fetch layer refuses the rest.
            os.environ[OFFLINE_ENV] = "1"
        try:
            self.result = self._runner(
                self._settings, progress=lambda i, n, s: self.progressed.emit(i, n, s),
                should_stop=lambda: self._stop)
        except Exception as error:      # noqa: BLE001 - reported on the page
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if self._settings.get("offline"):
                if previous is None:
                    os.environ.pop(OFFLINE_ENV, None)
                else:
                    os.environ[OFFLINE_ENV] = previous


def _cohort_subjects() -> list[str]:
    rows = studies.patient_rows()
    if rows is None or not len(rows):
        return [f"sub-{i:02d}" for i in range(1, 21)]
    return [str(s) for s in rows["subject"]]


class RerunDialog(QDialog):
    """The settings of a re-run: for the Detectors study, band, detectors,
    thresholds and window; for the Outcome study, detector, band, threshold
    and window. Opens on the open recording's choices where it has them."""

    DETECTORS = ("rms", "line_length", "hilbert", "energy")

    def __init__(self, key: str, session=None, parent=None):
        super().__init__(parent)
        from onset_review.studycharts import DETECTOR_LABELS

        self.key = key
        self.setWindowTitle("Re-run the Detectors study" if key == "detectors"
                            else "Re-run the Outcome study")
        request = session.request if session is not None else None
        form = QFormLayout()
        self.band = QComboBox()
        for value, text in studies.BANDS.items():
            self.band.addItem(text, value)
        band = request.band if request is not None else (
            "ripple" if key == "detectors" else "fast_ripple")
        self.band.setCurrentIndex(max(0, self.band.findData(band)))
        form.addRow("Band", self.band)
        chosen = set(request.detectors) if request is not None else {"rms", "line_length"}
        self.detector_boxes: dict[str, QCheckBox] = {}
        self.detector = QComboBox()
        if key == "detectors":
            row = QHBoxLayout()
            for name in self.DETECTORS:
                box = QCheckBox(DETECTOR_LABELS.get(name, name))
                box.setChecked(name in chosen)
                self.detector_boxes[name] = box
                row.addWidget(box)
            form.addRow("Detectors", row)
            self.thresholds = QLineEdit("2, 2.5, 3, 3.5, 4, 5")
            self.thresholds.setToolTip("Thresholds in robust SD, separated by commas")
            form.addRow("Thresholds (SD)", self.thresholds)
        else:
            for name in self.DETECTORS:
                self.detector.addItem(DETECTOR_LABELS.get(name, name), name)
            first = request.detectors[0] if request is not None else "rms"
            self.detector.setCurrentIndex(max(0, self.detector.findData(first)))
            form.addRow("Detector", self.detector)
            self.thresholds = QLineEdit("" if request is None or request.threshold_sd is None
                                        else f"{request.threshold_sd:g}")
            self.thresholds.setPlaceholderText("the detector's measured default")
            form.addRow("Threshold (SD)", self.thresholds)
        self.t_start = QDoubleSpinBox()
        self.t_stop = QDoubleSpinBox()
        for spin in (self.t_start, self.t_stop):
            spin.setRange(0.0, 3600.0)
            spin.setDecimals(0)
            spin.setSuffix(" s")
        self.t_stop.setValue(60.0 if key == "detectors" else 300.0)
        window = QHBoxLayout()
        window.addWidget(self.t_start)
        window.addWidget(QLabel("to"))
        window.addWidget(self.t_stop)
        form.addRow("Window of each run-01", window)
        self.subjects = QLineEdit(", ".join(_cohort_subjects()))
        self.subjects.setToolTip("The patients to include, separated by commas")
        form.addRow("Patients", self.subjects)
        self.offline = QCheckBox("Only recordings already on this machine (no downloads)")
        self.offline.setChecked(True)
        self.offline.setToolTip("Unticked, a patient not yet fetched is downloaded from "
                                "OpenNeuro first: a few tens of MB each")
        form.addRow("", self.offline)
        self.problem = QLabel("")
        self.problem.setStyleSheet(f"color:{theme.current().bad};")
        self.problem.setWordWrap(True)
        published = ("The published study used RMS and line length on the first 60 s"
                     if key == "detectors" else
                     "The published study used RMS in the fast-ripple band on the whole "
                     "300 s run")
        lead = QLabel(f"{published}. Your re-run is kept beside it, labelled yours; the "
                      "published tables are never changed. It runs in the background and "
                      "can be stopped between patients.")
        lead.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(lead)
        box.addLayout(form)
        box.addWidget(self.problem)
        box.addWidget(buttons)

    def _numbers(self) -> list[float]:
        text = self.thresholds.text().replace(";", ",")
        return [float(part) for part in text.split(",") if part.strip()]

    def settings(self) -> dict:
        subjects = [s.strip() for s in self.subjects.text().split(",") if s.strip()]
        window = {"t_start": float(self.t_start.value()), "t_stop": float(self.t_stop.value())}
        if self.key == "detectors":
            return {"subjects": subjects, "thresholds": self._numbers(),
                    "detectors": [n for n, b in self.detector_boxes.items() if b.isChecked()],
                    "bands": [self.band.currentData()], **window,
                    "offline": self.offline.isChecked()}
        numbers = self._numbers()
        return {"subjects": subjects, "detector": self.detector.currentData(),
                "band": self.band.currentData(),
                "threshold_sd": numbers[0] if numbers else None, **window,
                "offline": self.offline.isChecked()}

    def _accept(self) -> None:
        try:
            settings = self.settings()
        except ValueError:
            self.problem.setText("Thresholds are numbers separated by commas.")
            return
        if not settings["subjects"]:
            self.problem.setText("Name at least one patient.")
        elif self.key == "detectors" and not settings["detectors"]:
            self.problem.setText("Tick at least one detector.")
        elif self.key == "detectors" and not settings["thresholds"]:
            self.problem.setText("Give at least one threshold.")
        elif settings["t_stop"] <= settings["t_start"]:
            self.problem.setText("The window must end after it starts.")
        else:
            self.accept()


class CohortDialog(QDialog):
    """The open recording into your cohort: a name, the outcome, and the
    contacts that were resected -- ticked from the recording's own contacts,
    the archive's resection already ticked where there is one."""

    def __init__(self, session, parent=None, *, subject: str | None = None,
                 channels=None, known=None):
        super().__init__(parent)
        from onset_review.yourstudy import contacts_of

        if session is not None:
            subject = session.request.subject
            channels = list(session.raw.ch_names)
            known = getattr(session.resection, "resected", ()) or ()
        self.setWindowTitle("Add this recording to your cohort")
        self.label = QLineEdit(subject or "")
        self.free = QRadioButton("Seizure-free")
        self.recurrence = QRadioButton("Recurrence")
        group = QButtonGroup(self)
        group.addButton(self.free)
        group.addButton(self.recurrence)
        outcome = QHBoxLayout()
        outcome.addWidget(self.free)
        outcome.addWidget(self.recurrence)
        outcome.addStretch(1)
        known = {c.upper() for c in (known or ())}
        self.contacts = QListWidget()
        self.contacts.setObjectName("onset_cohort_contacts")
        for contact in contacts_of(channels or ()):
            item = QListWidgetItem(contact)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if contact in known else Qt.Unchecked)
            self.contacts.addItem(item)
        self.problem = QLabel("")
        self.problem.setStyleSheet(f"color:{theme.current().bad};")
        lead = QLabel("Measured the way the Outcome study measures a patient: the channels "
                      "tied with the busiest in this window, and the share of them inside "
                      "the contacts you tick. Your entry, kept on this machine, labelled "
                      "yours wherever it shows.")
        lead.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Patient", self.label)
        form.addRow("Outcome", outcome)
        form.addRow("Resected contacts", self.contacts)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(lead)
        box.addLayout(form)
        box.addWidget(self.problem)
        box.addWidget(buttons)

    @classmethod
    def for_channels(cls, subject: str, channels, parent=None) -> CohortDialog:
        """The dialog for a recording that is not open: a batch row."""
        return cls(None, parent, subject=subject, channels=channels)

    def values(self) -> tuple[str, list[str], str]:
        outcome = "seizure-free" if self.free.isChecked() else (
            "recurrence" if self.recurrence.isChecked() else "")
        ticked = [self.contacts.item(i).text() for i in range(self.contacts.count())
                  if self.contacts.item(i).checkState() == Qt.Checked]
        return outcome, ticked, self.label.text().strip()

    def _accept(self) -> None:
        outcome, ticked, _label = self.values()
        if not outcome:
            self.problem.setText("Choose the outcome.")
        elif not ticked:
            self.problem.setText("Tick the contacts that were resected.")
        else:
            self.accept()
