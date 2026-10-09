"""The Editor: scripts and notebooks, run in the console a cell at a time.

Spyder's editor, for the Analysis page. A script is kept as a file and run
again tomorrow, which the console alone cannot give: what was typed there
is gone when the window closes. Each tab is a file -- a ``.py`` script or a
Jupyter notebook opened as ``# %%`` cells (`onset_review.cells`) -- with
line numbers, Python colouring and the cell under the cursor shaded.

Running is the console's: F5 runs the file (saved first), Ctrl+Enter the
cell the cursor is in, Shift+Enter the cell and then moves to the next, F9
the selection or the line. Everything runs in the same namespace as the
console, so a script reads `raw`, `findings` and the rest of the Workspace
by name, and what it makes appears in the Workspace after it runs. Every
run is in the session's console log, and so in the exported report, with
the file's text rather than only its name.

Under the tabs, *Write code*: say what you want in words and the local
model the Assistant page chose drafts it (`onset_review.codewriter`). The
draft opens in a tab of its own with a comment saying who wrote it; what
could be checked without running it -- that it parses, that every name it
reads exists, which calls touch files or the network -- is listed under the
request. Nothing runs it but you.

Open tabs, including unsaved text, are kept between launches
(``editor.json`` beside the assistant's settings), so closing the window
never loses a script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qtpy.QtCore import QRect, QSize, Qt, QThread, QTimer, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
    QTextFormat,
)
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from onset_review import cells, codewriter, theme

__all__ = ["EditorPanel", "CodeEdit", "PythonHighlighter", "CodeAssistant", "SCRIPT_FILTER",
           "TemplateDialog"]

SCRIPT_FILTER = "Python scripts and notebooks (*.py *.ipynb);;Python scripts (*.py);;" \
                "Jupyter notebooks (*.ipynb);;All files (*)"

#: What a new tab starts with: the names there are, so a first script has
#: somewhere to begin.
STARTER = '''# %% A script for this recording
# raw, signal, times, channels, sfreq, events, findings and the rest of the
# Workspace are already here. Ctrl+Enter runs the cell the cursor is in,
# Shift+Enter runs it and moves on, F5 runs the whole file.

print(raw)
print(findings.head())

# %% A plot: the busiest channel's first two seconds
channel = findings.iloc[0]["channel"] if "channel" in findings else channels[0]
index = channels.index(channel)
first_two = times < times[0] + 2
plt.figure(figsize=(9, 3))
plt.plot(times[first_two], signal[index, first_two] * 1e6)
plt.xlabel("time (s)")
plt.ylabel("µV")
plt.title(channel)
'''

KEYWORDS = ("False None True and as assert async await break class continue def del elif "
            "else except finally for from global if import in is lambda nonlocal not or "
            "pass raise return try while with yield match case").split()
BUILTINS = ("print len range enumerate zip dict list tuple set int float str bool abs min "
            "max sum sorted round open isinstance type any all map filter super").split()


class PythonHighlighter(QSyntaxHighlighter):
    """Keywords, builtins, numbers, strings, comments and cell markers, in
    the theme's colours; triple-quoted strings across lines."""

    def __init__(self, document):
        super().__init__(document)
        p = theme.current()

        def style(colour: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            out = QTextCharFormat()
            out.setForeground(QColor(colour))
            if bold:
                out.setFontWeight(QFont.Bold)
            out.setFontItalic(italic)
            return out

        self.rules = [
            (re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b"), style(p.accent, bold=True)),
            (re.compile(r"\b(" + "|".join(BUILTINS) + r")\b"), style(p.accent)),
            (re.compile(r"\b(self|cls)\b"), style(p.text_muted, italic=True)),
            (re.compile(r"\b[0-9][0-9_]*\.?[0-9_]*([eE][-+]?[0-9]+)?j?\b"), style(p.warn)),
            (re.compile(r"@\w+(\.\w+)*"), style(p.warn)),
            (re.compile(r"\bdef\s+(\w+)|\bclass\s+(\w+)"), style(p.text, bold=True)),
        ]
        self.string = style(p.good)
        self.comment = style(p.text_muted, italic=True)
        self.cell = style(p.accent, bold=True)
        self._strings = re.compile(r"""(?:[rbfuRBFU]{0,2})("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")
        self._triple = (re.compile('"""'), re.compile("'''"))

    def highlightBlock(self, text: str) -> None:      # noqa: N802  (Qt's spelling)
        if cells.MARKER.match(text):
            self.setFormat(0, len(text), self.cell)
            self.setCurrentBlockState(0)
            return
        for pattern, fmt in self.rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)
        strings = []
        for match in self._strings.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self.string)
            strings.append((match.start(), match.end()))
        hash_at = -1
        for index, char in enumerate(text):
            if char == "#" and not any(a <= index < b for a, b in strings):
                hash_at = index
                break
        if hash_at >= 0:
            self.setFormat(hash_at, len(text) - hash_at, self.comment)
        self._triple_quotes(text)

    def _triple_quotes(self, text: str) -> None:
        self.setCurrentBlockState(0)
        start = 0
        state = self.previousBlockState()
        if state in (1, 2):
            pattern = self._triple[state - 1]
            match = pattern.search(text)
            if match is None:
                self.setCurrentBlockState(state)
                self.setFormat(0, len(text), self.string)
                return
            self.setFormat(0, match.end(), self.string)
            start = match.end()
        while True:
            found = [(m.start(), i) for i, p in enumerate(self._triple)
                     if (m := p.search(text, start)) is not None]
            if not found:
                return
            begin, which = min(found)
            end = self._triple[which].search(text, begin + 3)
            if end is None:
                self.setFormat(begin, len(text) - begin, self.string)
                self.setCurrentBlockState(which + 1)
                return
            self.setFormat(begin, end.end() - begin, self.string)
            start = end.end()


class _Gutter(QWidget):
    def __init__(self, editor: CodeEdit):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:      # noqa: N802
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, event) -> None:      # noqa: N802
        self.editor.paint_gutter(event)


class CodeEdit(QPlainTextEdit):
    """One file: line numbers, colouring, the current cell shaded, Python's
    indentation on Tab and Enter, Ctrl+/ to comment lines in or out."""

    def __init__(self, text: str = "", path: Path | None = None, parent=None):
        super().__init__(parent)
        from onset_review.console import _monospace

        self.path: Path | None = Path(path) if path is not None else None
        #: The notebook this tab was opened from, for its metadata on save.
        self.notebook: dict | None = None
        #: Shown on the tab when there is no file: "Untitled 2", "Draft 1".
        self.title = ""
        self.setFont(_monospace())
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.highlighter = PythonHighlighter(self.document())
        self.gutter = _Gutter(self)
        self.blockCountChanged.connect(self._fit_gutter)
        self.updateRequest.connect(self._scroll_gutter)
        self.cursorPositionChanged.connect(self._shade_cell)
        self._fit_gutter()
        self.setPlainText(text)
        self.document().setModified(False)
        self._shade_cell()

    # -- what it is ----------------------------------------------------------
    @property
    def is_notebook(self) -> bool:
        return self.path is not None and self.path.suffix.lower() == ".ipynb"

    def name(self) -> str:
        return self.path.name if self.path is not None else (self.title or "Untitled")

    def modified(self) -> bool:
        return self.document().isModified()

    # -- the gutter --------------------------------------------------------------
    def gutter_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _fit_gutter(self, *_args) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _scroll_gutter(self, rect, dy: int) -> None:
        if dy:
            self.gutter.scroll(0, dy)
        else:
            self.gutter.update(0, rect.y(), self.gutter.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._fit_gutter()

    def resizeEvent(self, event) -> None:      # noqa: N802
        super().resizeEvent(event)
        area = self.contentsRect()
        self.gutter.setGeometry(QRect(area.left(), area.top(), self.gutter_width(),
                                      area.height()))

    def paint_gutter(self, event) -> None:
        p = theme.current()
        painter = QPainter(self.gutter)
        painter.fillRect(event.rect(), QColor(p.surface_alt))
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        height = self.fontMetrics().height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(p.text if number == current else p.text_muted))
                painter.drawText(0, top, self.gutter.width() - 6, height, Qt.AlignRight,
                                 str(number + 1))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1
        painter.end()

    def paintEvent(self, event) -> None:      # noqa: N802
        super().paintEvent(event)
        # A rule above each cell marker, as Spyder draws one.
        painter = QPainter(self.viewport())
        painter.setPen(QColor(theme.current().separator))
        block = self.firstVisibleBlock()
        offset = self.contentOffset()
        while block.isValid():
            geometry = self.blockBoundingGeometry(block).translated(offset)
            if geometry.top() > event.rect().bottom():
                break
            if block.blockNumber() > 0 and cells.MARKER.match(block.text()):
                y = int(geometry.top())
                painter.drawLine(0, y, self.viewport().width(), y)
            block = block.next()
        painter.end()

    # -- cells ---------------------------------------------------------------
    def current_cell(self) -> cells.Cell:
        return cells.cell_at(self.toPlainText(), self.textCursor().blockNumber())

    def cell_source(self, cell: cells.Cell) -> str:
        return cell.source(self.toPlainText().split("\n"))

    def _shade_cell(self) -> None:
        text = self.toPlainText()
        if not cells.MARKER.search(text) and "%%" not in text:
            self.setExtraSelections([])
            return
        cell = self.current_cell()
        shade = QTextEdit.ExtraSelection()
        colour = QColor(theme.current().accent)
        colour.setAlpha(18)
        shade.format.setBackground(colour)
        shade.format.setProperty(QTextFormat.FullWidthSelection, True)
        selections = []
        document = self.document()
        for line in range(cell.start, cell.stop):
            block = document.findBlockByNumber(line)
            if not block.isValid():
                break
            one = QTextEdit.ExtraSelection()
            one.format = shade.format
            one.cursor = QTextCursor(block)
            selections.append(one)
        self.setExtraSelections(selections)

    def go_to_line(self, line: int) -> None:
        block = self.document().findBlockByNumber(max(0, line))
        if block.isValid():
            self.setTextCursor(QTextCursor(block))
            self.centerCursor()

    # -- typing -------------------------------------------------------------
    def keyPressEvent(self, event) -> None:      # noqa: N802
        key, mods = event.key(), event.modifiers()
        cursor = self.textCursor()
        if key == Qt.Key_Tab and not mods and not cursor.hasSelection():
            cursor.insertText("    ")
            return
        if key == Qt.Key_Backtab or (key == Qt.Key_Tab and cursor.hasSelection()):
            self._shift_lines(-1 if key == Qt.Key_Backtab else 1)
            return
        if key in (Qt.Key_Return, Qt.Key_Enter) and not (mods & (Qt.ControlModifier
                                                                 | Qt.ShiftModifier)):
            line = cursor.block().text()[:cursor.positionInBlock()]
            indent = line[:len(line) - len(line.lstrip(" "))]
            if line.rstrip().endswith(":"):
                indent += "    "
            cursor.insertText("\n" + indent)
            self.ensureCursorVisible()
            return
        if key == Qt.Key_Slash and mods & Qt.ControlModifier:
            self.toggle_comment()
            return
        super().keyPressEvent(event)

    def _selected_blocks(self):
        cursor = self.textCursor()
        document = self.document()
        first = document.findBlock(cursor.selectionStart()).blockNumber()
        end = cursor.selectionEnd()
        last_block = document.findBlock(end)
        last = last_block.blockNumber()
        if cursor.hasSelection() and end == last_block.position() and last > first:
            last -= 1
        return [document.findBlockByNumber(n) for n in range(first, last + 1)]

    def _shift_lines(self, step: int) -> None:
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for block in self._selected_blocks():
            edit = QTextCursor(block)
            if step > 0:
                edit.insertText("    ")
            else:
                text = block.text()
                remove = min(4, len(text) - len(text.lstrip(" ")))
                for _ in range(remove):
                    edit.deleteChar()
        cursor.endEditBlock()

    def toggle_comment(self) -> None:
        blocks = [b for b in self._selected_blocks() if b.text().strip()]
        if not blocks:
            return
        commented = all(b.text().lstrip().startswith("#") for b in blocks)
        cursor = self.textCursor()
        cursor.beginEditBlock()
        for block in blocks:
            text = block.text()
            at = len(text) - len(text.lstrip())
            edit = QTextCursor(block)
            edit.setPosition(block.position() + at)
            if commented:
                width = 2 if text[at:at + 2] == "# " else 1
                edit.setPosition(block.position() + at + width, QTextCursor.KeepAnchor)
                edit.removeSelectedText()
            else:
                edit.insertText("# ")
        cursor.endEditBlock()


class _DraftWorker(QThread):
    """One draft from the local model, off the GUI thread."""

    def __init__(self, messages: list[dict], parent=None):
        super().__init__(parent)
        self._messages = messages
        self._backend = None
        self.reply = ""
        self.model = ""
        self.error: str | None = None

    def stop(self) -> None:
        backend = self._backend
        if backend is not None:
            backend.abort()

    def run(self) -> None:
        try:
            from onset_agent.backends import make_backend
            from onset_review.assistant import ANSWER_TIMEOUT
            from onset_review.assistant_config import load_defaults

            defaults = load_defaults()
            kind = str(getattr(defaults, "kind", "scripted") or "scripted")
            self.model = str(getattr(defaults, "model", "") or "") or kind
            self._backend = make_backend(
                kind, model=str(getattr(defaults, "model", "") or "") or None,
                base_url=str(getattr(defaults, "base_url", "") or "") or None,
                max_tokens=codewriter.CODE_TOKENS, timeout=ANSWER_TIMEOUT)
            if not getattr(self._backend, "is_language_model", True):
                self.error = "no model is loaded"
                return
            message = self._backend.chat(self._messages, [])
            self.reply = (message.content or "").strip()
        except Exception as error:      # noqa: BLE001 - reported under the box
            self.error = f"{type(error).__name__}: {error}"


class CodeAssistant(QWidget):
    """*Write code*: a request in words, a draft from the local model in a
    tab of its own, and what could be checked about it."""

    def __init__(self, editor: EditorPanel, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.history: list[tuple[str, str]] = []
        self.last: codewriter.Draft | None = None
        self._worker = None
        self._kind, self._model = "scripted", ""
        #: Replaces the model in tests: called with the messages, returns the reply.
        self.ask_model = None
        p = theme.current()
        self.request = QLineEdit()
        self.request.setObjectName("onset_code_request")
        self.request.setPlaceholderText(
            "Describe the analysis, e.g. “power spectrum of the top 3 channels”")
        self.request.returnPressed.connect(lambda: self.write())
        self.write_button = QPushButton("Write code")
        self.write_button.setObjectName("onset_code_write")
        self.write_button.setToolTip("The local model drafts Python into a new tab. "
                                     "Nothing runs until you run it.")
        self.write_button.clicked.connect(lambda _=False: self.write())
        self.fix_button = QPushButton("Fix the last error")
        self.fix_button.setObjectName("onset_code_fix")
        self.fix_button.setToolTip("Send the console's last error and this script to the "
                                   "model, for a corrected draft")
        self.fix_button.clicked.connect(lambda _=False: self.fix_last_error())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self.stop())
        self.model_line = QLabel("")
        self.model_line.setObjectName("onset_code_model")
        self.model_line.setStyleSheet(f"color:{p.text_muted};font-size:9pt;")
        self.model_line.setWordWrap(True)
        self.notes = QLabel("")
        self.notes.setObjectName("onset_code_notes")
        self.notes.setWordWrap(True)
        self.notes.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.notes.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.notes.hide()
        row = QHBoxLayout()
        row.setSpacing(theme.SPACING // 2)
        row.addWidget(self.request, 1)
        row.addWidget(self.write_button)
        row.addWidget(self.fix_button)
        row.addWidget(self.stop_button)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, theme.SPACING // 2, 0, 0)
        box.setSpacing(2)
        box.addLayout(row)
        box.addWidget(self.model_line)
        box.addWidget(self.notes)
        self.refresh_model()

    # -- which model ----------------------------------------------------------
    def refresh_model(self) -> None:
        from onset_review.assistant_config import load_defaults

        defaults = load_defaults()
        self._kind = str(getattr(defaults, "kind", "scripted") or "scripted")
        self._model = str(getattr(defaults, "model", "") or "")
        if self.available:
            self.model_line.setText(
                f"Write code: {self._model or self._kind} (the Assistant page's model) drafts; "
                "it is told the Workspace's names, types and columns, never the data. "
                "Read a draft before you run it.")
        else:
            self.model_line.setText("Write code needs a local model: choose one on the "
                                    "Assistant page.")
        self._enable(busy=False)

    @property
    def available(self) -> bool:
        return self.ask_model is not None or self._kind != "scripted"

    def _enable(self, busy: bool) -> None:
        on = self.available and not busy
        self.request.setEnabled(on)
        self.write_button.setEnabled(on)
        console = self.editor.console
        self.fix_button.setEnabled(on and bool(getattr(console, "last_error", "")))
        self.stop_button.setEnabled(busy)

    def sync(self) -> None:
        """After a console command: Fix the last error is offered only while
        there is one."""
        self._enable(busy=bool(self._worker is not None and self._worker.isRunning()))

    def showEvent(self, event) -> None:      # noqa: N802
        self.refresh_model()
        super().showEvent(event)

    # -- drafting --------------------------------------------------------------
    def names(self) -> dict:
        console = self.editor.console
        if console is not None and hasattr(console, "visible_names"):
            return console.visible_names()
        return {}

    def write(self, request: str | None = None) -> CodeEdit | None:
        """Ask for a draft of `request` (the box's text by default). Returns
        the tab the draft opened in, or None."""
        text = (request if isinstance(request, str) and request else self.request.text()).strip()
        if not text or not self.available:
            return None
        names = self.names()
        current = self.editor.current()
        script = current.toPlainText() if current is not None else ""
        if script.strip() == STARTER.strip():
            script = ""
        conversation = codewriter.messages(text, names, script, self.history)
        reply, model, error = self._ask(conversation)
        if error:
            import logging

            logging.getLogger("onset_review.editor").warning("Write code failed: %s", error)
            self._say([f"The model could not draft: {error}"], bad=True)
            return None
        code = codewriter.extract_code(reply)
        draft = codewriter.check_code(code, names)
        import logging

        logging.getLogger("onset_review.editor").info(
            "Write code via %s: %d lines, parses %s, %d unknown names, %d flagged calls",
            model, len(code.splitlines()), draft.parses, len(draft.unknown), len(draft.risky))
        self.last = draft
        self.history.append((conversation[-1]["content"], reply))
        self.history = self.history[-4:]
        if not code.strip():
            self._say(["The model gave no code. Its reply: " + " ".join(reply.split())[:300]],
                      bad=True)
            return None
        notes = draft.notes
        top = codewriter.header(model, text)
        if notes:
            top += "".join(f"# Check: {note}\n" for note in notes)
        tab = self.editor.new_file(top + "\n" + code, title=self.editor.next_title("Draft"))
        self.request.clear()
        self._say(notes or ["Parses, and every name it reads exists. That says nothing "
                            "about whether it computes what you meant: read it."],
                  bad=bool(notes))
        return tab

    def fix_last_error(self) -> CodeEdit | None:
        error = getattr(self.editor.console, "last_error", "")
        if not error:
            return None
        return self.write("Running this script raised the error below. Correct the script "
                          "and return the whole corrected script.\n\n" + error)

    def _ask(self, conversation: list[dict]) -> tuple[str, str, str | None]:
        if self.ask_model is not None:
            try:
                return str(self.ask_model(conversation)), "test model", None
            except Exception as error:      # noqa: BLE001 - reported under the box
                return "", "", str(error)
        from onset_review.assistant import _run

        self._worker = _DraftWorker(conversation, self)
        self._enable(busy=True)
        self.model_line.setText("Drafting…")
        try:
            _run(self._worker)
        finally:
            self.refresh_model()
        worker = self._worker
        return worker.reply, worker.model, worker.error

    def stop(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()

    def _say(self, lines: list[str], bad: bool = False) -> None:
        p = theme.current()
        self.notes.setStyleSheet(f"color:{p.warn if bad else p.text_muted};font-size:9pt;")
        self.notes.setText("\n".join(lines))
        self.notes.show()


class EditorPanel(QWidget):
    """Tabs of scripts and notebooks, a row of run buttons, and Write code."""

    #: A tab was saved, opened or closed: its path, or "".
    filesChanged = Signal()

    def __init__(self, console=None, parent=None, restore: bool = False):
        super().__init__(parent)
        #: The Console pane runs what this sends; set by whoever holds both.
        self.console = console
        #: A callable giving the Files pane's folder, for the file dialogs.
        self.folder = None
        self._counters: dict[str, int] = {}

        def button(text: str, tip: str, slot, name: str = "") -> QToolButton:
            out = QToolButton()
            out.setText(text)
            out.setToolTip(tip)
            out.setAutoRaise(True)
            out.setToolButtonStyle(Qt.ToolButtonTextOnly)
            if name:
                out.setObjectName(name)
            out.clicked.connect(lambda _=False: slot())
            return out

        self.buttons = {
            "new": button("New", "A new script (Ctrl+N)", lambda: self.new_file()),
            "open": button("Open…", "Open a script or a notebook", lambda: self.open_dialog()),
            "save": button("Save", "Save this tab (Ctrl+S)", lambda: self.save()),
            "save_as": button("Save as…", "Save this tab under another name, as .py or "
                              ".ipynb", lambda: self.save_as()),
            "run_file": button("▶ Run file", "Run the whole file in the console (F5)",
                               lambda: self.run_file(), "onset_editor_run_file"),
            "run_cell": button("Run cell", "Run the cell the cursor is in (Ctrl+Enter); "
                               "cells start at lines beginning # %%", lambda: self.run_cell()),
            "run_next": button("Run cell, advance", "Run the cell and move to the next "
                               "(Shift+Enter)", lambda: self.run_cell(advance=True)),
            "run_selection": button("Run selection", "Run the selected lines, or the line "
                                    "the cursor is on (F9)", lambda: self.run_selection()),
        }
        # The template library: a script for each analysis this window does.
        templates = button("Templates", "A ready-made script for each analysis this window "
                           "does, on the open recording", lambda: None, "onset_editor_templates")
        templates.setPopupMode(QToolButton.InstantPopup)
        templates.setToolButtonStyle(Qt.ToolButtonTextOnly)
        templates.setMenu(self._template_menu())
        self.buttons["templates"] = templates
        bar = theme.flow_layout()
        for key in ("new", "open", "templates", "save", "save_as"):
            bar.addWidget(self.buttons[key])
        bar.addWidget(theme.cluster(*(self.buttons[k] for k in
                                      ("run_file", "run_cell", "run_next", "run_selection"))))
        bar_host = QWidget()
        bar_host.setLayout(bar)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("onset_editor_tabs")
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(lambda index: self.close_tab(index))
        self.tabs.currentChanged.connect(lambda _index: self._say_where())
        self.where = QLabel("")
        self.where.setObjectName("onset_editor_where")
        self.where.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
        self.where.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.assistant = CodeAssistant(self)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(bar_host)
        box.addWidget(self.tabs, 1)
        box.addWidget(self.where)
        box.addWidget(self.assistant)

        for keys, slot in (("F5", self.run_file), ("Ctrl+Return", self.run_cell),
                           ("Ctrl+Enter", self.run_cell),
                           ("Shift+Return", lambda: self.run_cell(advance=True)),
                           ("Shift+Enter", lambda: self.run_cell(advance=True)),
                           ("F9", self.run_selection), ("Ctrl+S", self.save),
                           ("Ctrl+N", self.new_file)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda slot=slot: slot())

        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(2000)
        self._autosave.timeout.connect(self.remember)
        if not (restore and self.restore()):
            self.new_file(STARTER, title=self.next_title("Untitled"), modified=False)

    # -- the template library ---------------------------------------------------------
    def _template_menu(self) -> QMenu:
        from onset_review import codetemplates

        menu = QMenu(self)
        menu.setObjectName("onset_template_menu")
        menu.setToolTipsVisible(True)
        browse = menu.addAction("Browse all templates…")
        browse.setToolTip("Every template with what it does and its code, before opening it")
        browse.triggered.connect(lambda _=False: self.browse_templates())
        for group, members in codetemplates.grouped():
            menu.addSection(group)
            for template in members:
                action = menu.addAction(template.title)
                action.setToolTip(f"{template.about}\n(Mirrors {template.mirrors}.)")
                action.triggered.connect(lambda _=False, key=template.key: self.open_template(key))
        return menu

    def open_template(self, key: str) -> CodeEdit | None:
        """A template in a new tab, unsaved: running it changes nothing on
        disk, and Save as… keeps a copy of your own."""
        from onset_review import codetemplates

        template = codetemplates.find(key)
        if template is None:
            return None
        self._drop_untouched_starter()
        editor = self.new_file(template.text, title=template.title, modified=False)
        self._tell(f"{template.title}: a template — run it a cell at a time with Ctrl+Enter; "
                   "Save as… keeps your own copy")
        return editor

    def browse_templates(self, show: bool = True) -> TemplateDialog:
        dialog = TemplateDialog(self)
        if show:
            dialog.show()
        return dialog

    # -- tabs ---------------------------------------------------------------------
    def editors(self) -> list[CodeEdit]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def current(self) -> CodeEdit | None:
        return self.tabs.currentWidget()

    def next_title(self, stem: str) -> str:
        taken = {e.title for e in self.editors()}
        number = self._counters.get(stem, 0)
        while True:
            number += 1
            title = f"{stem} {number}"
            if title not in taken:
                self._counters[stem] = number
                return title

    def _add(self, editor: CodeEdit) -> CodeEdit:
        index = self.tabs.addTab(editor, editor.name())
        editor.document().modificationChanged.connect(lambda _on, e=editor: self._retitle(e))
        editor.textChanged.connect(self._autosave.start)
        self.tabs.setCurrentIndex(index)
        self._retitle(editor)
        editor.setFocus()
        return editor

    def _retitle(self, editor: CodeEdit) -> None:
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.setTabText(index, editor.name() + (" •" if editor.modified() else ""))
            self.tabs.setTabToolTip(index, str(editor.path) if editor.path else
                                    "Not saved to a file yet")
        self._say_where()

    def _say_where(self) -> None:
        editor = self.current()
        if editor is None:
            self.where.setText("")
            return
        if editor.path is None:
            what = "not saved yet · Run file runs the text as it stands"
        elif editor.is_notebook:
            what = f"{editor.path} · a notebook as # %% cells; Save writes it back without outputs"
        else:
            what = str(editor.path)
        self.where.setText(what)

    def new_file(self, text: str = "", title: str | None = None,
                 modified: bool = True) -> CodeEdit:
        editor = CodeEdit(text)
        editor.title = title or self.next_title("Untitled")
        editor.document().setModified(bool(modified and text))
        return self._add(editor)

    def open_file(self, path: str | Path) -> CodeEdit | None:
        """Open a script or a notebook, or go to its tab if it is open."""
        path = Path(path).expanduser().resolve()
        for index, editor in enumerate(self.editors()):
            if editor.path == path:
                self.tabs.setCurrentIndex(index)
                return editor
        try:
            if path.suffix.lower() == ".ipynb":
                notebook = cells.read_notebook(path)
                editor = CodeEdit(cells.notebook_to_script(notebook), path)
                editor.notebook = notebook
            else:
                editor = CodeEdit(path.read_text(encoding="utf-8"), path)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            self._tell(f"Could not open {path.name}: {error}")
            return None
        self._drop_untouched_starter()
        self._add(editor)
        self.filesChanged.emit()
        self.remember()
        return editor

    def _drop_untouched_starter(self) -> None:
        """The starter tab, unchanged, goes when a real file opens."""
        editors = self.editors()
        if (len(editors) == 1 and editors[0].path is None and not editors[0].modified()
                and editors[0].toPlainText().strip() == STARTER.strip()):
            self.tabs.removeTab(0)
            editors[0].deleteLater()

    def open_dialog(self) -> CodeEdit | None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Open a script or a notebook",
                                                str(self._folder()), SCRIPT_FILTER)
        return self.open_file(chosen) if chosen else None

    def close_tab(self, index: int, ask: bool = True) -> bool:
        editor = self.tabs.widget(index)
        if editor is None:
            return False
        if ask and editor.modified():
            answer = QMessageBox.question(
                self, "Close without saving?",
                f"{editor.name()} has changes that are not saved.",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                return False
            if answer == QMessageBox.Save and self.save(editor) is None:
                return False
        self.tabs.removeTab(index)
        editor.deleteLater()
        if self.tabs.count() == 0:
            self.new_file("", title=self.next_title("Untitled"), modified=False)
        self.filesChanged.emit()
        self.remember()
        return True

    # -- saving --------------------------------------------------------------
    def _folder(self) -> Path:
        try:
            folder = self.folder() if callable(self.folder) else self.folder
        except Exception:      # noqa: BLE001 - a suggestion, never a failure
            folder = None
        return Path(folder) if folder else Path.cwd()

    def save(self, editor: CodeEdit | None = None, path: str | Path | None = None) -> Path | None:
        """Save `editor` (the current tab) to its file, or to `path`; a tab
        with no file asks where. A .ipynb path writes a notebook."""
        editor = editor if isinstance(editor, CodeEdit) else self.current()
        if editor is None:
            return None
        if path is None and editor.path is None:
            return self.save_as(editor)
        target = Path(path) if path is not None else editor.path
        try:
            if target.suffix.lower() == ".ipynb":
                cells.write_notebook(editor.toPlainText(), target, editor.notebook)
            else:
                target.write_text(editor.toPlainText(), encoding="utf-8")
        except OSError as error:
            self._tell(f"Could not save {target.name}: {error}")
            return None
        editor.path = target.resolve()
        editor.document().setModified(False)
        self._retitle(editor)
        self.filesChanged.emit()
        self.remember()
        return editor.path

    def save_as(self, editor: CodeEdit | None = None) -> Path | None:
        editor = editor if isinstance(editor, CodeEdit) else self.current()
        if editor is None:
            return None
        suggested = editor.path or (self._folder() / (re.sub(r"\W+", "_", editor.name()).strip(
            "_").lower() + ".py"))
        chosen, _ = QFileDialog.getSaveFileName(self, "Save the script", str(suggested),
                                                SCRIPT_FILTER)
        if not chosen:
            return None
        path = Path(chosen)
        if path.suffix.lower() not in (".py", ".ipynb"):
            path = path.with_suffix(".py")
        return self.save(editor, path)

    # -- running -----------------------------------------------------------------
    def _console(self):
        if self.console is None:
            self._tell("There is no console to run this in.")
        return self.console

    def run_file(self) -> bool:
        """F5: a saved script runs from its file (saved first); a notebook or
        an unsaved tab runs as its text stands."""
        editor, console = self.current(), self._console()
        if editor is None or console is None:
            return False
        if editor.path is not None and not editor.is_notebook:
            if editor.modified() and self.save(editor) is None:
                return False
            console.run_file(editor.path)
            return True
        text = editor.toPlainText()
        if not text.strip():
            return False
        console.run(text)
        return True

    def run_cell(self, advance: bool = False) -> bool:
        editor, console = self.current(), self._console()
        if editor is None or console is None:
            return False
        cell = editor.current_cell()
        source = editor.cell_source(cell)
        if advance:
            editor.go_to_line(min(cell.stop, editor.blockCount() - 1))
        if cell.markdown or not source.strip():
            return False
        console.run(source)
        return True

    def run_selection(self) -> bool:
        editor, console = self.current(), self._console()
        if editor is None or console is None:
            return False
        cursor = editor.textCursor()
        if cursor.hasSelection():
            source = cursor.selection().toPlainText()
        else:
            source = cursor.block().text()
            editor.go_to_line(cursor.blockNumber() + 1)
        source = _dedent(source)
        if not source.strip():
            return False
        console.run(source)
        return True

    # -- kept between launches ------------------------------------------------------
    @staticmethod
    def _state_file() -> Path:
        from onset_review.assistant_config import config_path

        return config_path().with_name("editor.json")

    def remember(self) -> None:
        """The open tabs, with the text of any not saved, to `editor.json`."""
        tabs = []
        for editor in self.editors():
            entry: dict = {"title": editor.title}
            if editor.path is not None:
                entry["path"] = str(editor.path)
            if editor.path is None or editor.modified():
                text = editor.toPlainText()
                if editor.path is None and not text.strip():
                    continue
                entry["text"] = text
                entry["modified"] = editor.modified()
            tabs.append(entry)
        try:
            path = self._state_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"tabs": tabs, "current": self.tabs.currentIndex()},
                                       indent=1) + "\n", encoding="utf-8")
        except OSError:
            pass        # tabs that cannot be remembered are still open

    def restore(self) -> bool:
        try:
            payload = json.loads(self._state_file().read_text(encoding="utf-8"))
            tabs = list(payload.get("tabs") or [])
        except (OSError, ValueError, TypeError, AttributeError):
            return False
        opened = 0
        for entry in tabs:
            if not isinstance(entry, dict):
                continue
            path = Path(entry["path"]) if entry.get("path") else None
            if "text" in entry:
                editor = CodeEdit(str(entry["text"]), path)
                editor.title = str(entry.get("title") or "")
                if path is not None and path.suffix.lower() == ".ipynb" and path.exists():
                    try:
                        editor.notebook = cells.read_notebook(path)
                    except (OSError, ValueError):
                        pass
                self._add(editor)
                editor.document().setModified(bool(entry.get("modified", True)))
                self._retitle(editor)
                opened += 1
            elif path is not None and path.exists():
                if self.open_file(path) is not None:
                    opened += 1
        if opened:
            current = payload.get("current")
            if isinstance(current, int) and 0 <= current < self.tabs.count():
                self.tabs.setCurrentIndex(current)
        return opened > 0

    def _tell(self, text: str) -> None:
        self.where.setText(text)


def _dedent(source: str) -> str:
    """Selected lines from inside a block, made runnable on their own."""
    import textwrap

    return textwrap.dedent(source)


class TemplateDialog(QDialog):
    """Every template: grouped on the left, what it does and its code on the right."""

    def __init__(self, editor_panel: EditorPanel, parent=None):
        super().__init__(parent or editor_panel)
        from onset_review import codetemplates

        self.setWindowTitle("Analysis templates")
        self.setObjectName("onset_template_dialog")
        self.resize(980, 620)
        self.panel = editor_panel
        self.list = QListWidget()
        self.list.setObjectName("onset_template_list")
        self.list.setAccessibleName("Templates")
        for group, members in codetemplates.grouped():
            heading = QListWidgetItem(group.upper())
            heading.setFlags(Qt.NoItemFlags)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            heading.setForeground(QColor(theme.current().text_muted))
            self.list.addItem(heading)
            for template in members:
                item = QListWidgetItem(template.title)
                item.setData(Qt.UserRole, template.key)
                item.setToolTip(template.about)
                self.list.addItem(item)
        self.about = QLabel("")
        self.about.setObjectName("onset_template_about")
        self.about.setWordWrap(True)
        self.preview = QPlainTextEdit()
        self.preview.setObjectName("onset_template_preview")
        self.preview.setReadOnly(True)
        self.preview.setFont(CodeEdit().font())
        PythonHighlighter(self.preview.document())
        self.open_button = QPushButton("Open in the editor")
        self.open_button.setObjectName("onset_template_open")
        self.open_button.setProperty("primary", True)
        self.open_button.clicked.connect(lambda _=False: self.open_current())
        self.list.itemDoubleClicked.connect(lambda _item: self.open_current())
        self.list.currentItemChanged.connect(lambda item, _old: self._show(item))
        right = QWidget()
        column = QVBoxLayout(right)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.about)
        column.addWidget(self.preview, 1)
        row = QHBoxLayout()
        row.addWidget(theme.muted("Each runs on the open recording; the names it uses are "
                                  "the Workspace's.", size=9), 1)
        row.addWidget(self.open_button)
        column.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.list)
        split.addWidget(right)
        split.setSizes([300, 680])
        box = QVBoxLayout(self)
        box.addWidget(split)
        first = next((self.list.item(i) for i in range(self.list.count())
                      if self.list.item(i).data(Qt.UserRole)), None)
        if first is not None:
            self.list.setCurrentItem(first)

    def _show(self, item) -> None:
        from onset_review import codetemplates

        key = item.data(Qt.UserRole) if item is not None else None
        template = codetemplates.find(key) if key else None
        self.open_button.setEnabled(template is not None)
        if template is None:
            return
        self.about.setText(f"<b>{template.title}</b> — mirrors {template.mirrors}.<br>"
                           f"{template.about}")
        self.preview.setPlainText(template.text)

    def open_current(self) -> CodeEdit | None:
        item = self.list.currentItem()
        key = item.data(Qt.UserRole) if item is not None else None
        if not key:
            return None
        editor = self.panel.open_template(key)
        self.accept()
        return editor
