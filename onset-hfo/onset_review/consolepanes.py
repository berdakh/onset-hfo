"""Plots and History: the console's two other panes, as Spyder has them.

**Plots** collects every figure the console or the editor draws, newest
first in a strip of thumbnails beside the one being looked at, instead of a
window per figure. A figure can still be opened in a window of its own (to
zoom and pan), saved as PNG, PDF or SVG, or removed.

**History** lists every command run in the console, from this session and
earlier ones (kept in ``console_history.json`` beside the assistant's
settings), searchable. A command can be run again, copied, or sent to the
editor -- several at once, in the order they ran -- which is how something
worked out at the prompt becomes a script.

The history is the reader's own aid, not the record: the commands that
touched a session are kept with that session and printed in its report, as
before, whatever is cleared here.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QFontDatabase, QIcon, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["PlotsPanel", "HistoryPanel", "history_path", "load_history", "save_history",
           "MAX_FIGURES", "MAX_HISTORY"]

#: Figures kept in the Plots pane; the oldest goes when one more arrives.
MAX_FIGURES = 60
#: Commands kept in the history file.
MAX_HISTORY = 1000


# -- the history file (Qt-free) ----------------------------------------------------------
def history_path() -> Path:
    from onset_review.assistant_config import config_path

    return config_path().with_name("console_history.json")


def load_history(path: Path | None = None) -> list[dict]:
    """The saved commands, oldest first; an unreadable file is an empty one."""
    path = path or history_path()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("source"), str)]


def save_history(entries: list[dict], path: Path | None = None) -> None:
    path = path or history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries[-MAX_HISTORY:], indent=0), encoding="utf-8")
    except OSError:
        pass        # the history is an aid; a read-only home must not stop the console


def _button(text: str, tip: str, slot, name: str = "") -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setToolTip(tip)
    button.setAutoRaise(True)
    button.clicked.connect(lambda _=False: slot())
    if name:
        button.setObjectName(name)
    return button


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet(f"color:{theme.current().text_muted};")
    return label


# -- Plots ----------------------------------------------------------------------------------
class _FigureView(QLabel):
    """The figure being looked at, scaled to fit whenever its room changes."""

    resized = Signal()
    doubleClicked = Signal()

    def resizeEvent(self, event) -> None:      # noqa: N802  (Qt's spelling)
        super().resizeEvent(event)
        self.resized.emit()

    def mouseDoubleClickEvent(self, event) -> None:      # noqa: N802
        self.doubleClicked.emit()


class PlotsPanel(QWidget):
    """The figures the console drew, one large and the rest as thumbnails."""

    #: A figure arrived.
    added = Signal()

    def __init__(self, open_window=None, folder=None, parent=None):
        super().__init__(parent)
        self.setObjectName("onset_plots")
        #: Called with (figure, title) to open a figure in a window of its own.
        self.open_window = open_window
        #: Where Save starts: a callable returning a folder, or None.
        self.folder = folder
        self.entries: list[dict] = []
        self.total = 0
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(2)
        self.previous_button = _button("◀", "The figure before (newer)", lambda: self.step(-1))
        self.next_button = _button("▶", "The figure after (older)", lambda: self.step(1))
        self.where = QLabel("")
        self.where.setStyleSheet(f"color:{theme.current().text_muted};")
        bar.addWidget(self.previous_button)
        bar.addWidget(self.next_button)
        bar.addWidget(self.where, 1)
        self.window_button = _button("Open in a window", "Zoom and pan this figure in a "
                                     "window of its own", self.open_current,
                                     "onset_plots_window")
        self.save_button = _button("Save…", "Save this figure as PNG, PDF or SVG",
                                   self.save_dialog, "onset_plots_save")
        self.remove_button = _button("Remove", "Take this figure out of the pane",
                                     self.remove_current, "onset_plots_remove")
        self.clear_button = _button("Remove all", "Empty the pane", self.clear,
                                    "onset_plots_clear")
        for button in (self.window_button, self.save_button, self.remove_button,
                       self.clear_button):
            bar.addWidget(button)
        box.addLayout(bar)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.view = _FigureView()
        self.view.setObjectName("onset_plots_view")
        self.view.resized.connect(self._scale)
        self.view.doubleClicked.connect(self.open_current)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setMinimumSize(160, 120)
        self.view.setToolTip("Double-click to open the figure in a window of its own")
        self.empty = _muted("Figures the console or the editor draws collect here — "
                            "plt.plot(…), raw.plot_psd(), any matplotlib figure.")
        self.strip = QListWidget()
        self.strip.setObjectName("onset_plots_strip")
        self.strip.setViewMode(QListWidget.IconMode)
        self.strip.setFlow(QListWidget.TopToBottom)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListWidget.Static)
        self.strip.setIconSize(QSize(96, 72))
        self.strip.setFixedWidth(124)
        self.strip.setSpacing(4)
        self.strip.currentRowChanged.connect(self._show_row)
        self.strip.itemDoubleClicked.connect(lambda _item: self.open_current())
        row.addWidget(self.view, 1)
        row.addWidget(self.empty, 1)
        row.addWidget(self.strip)
        box.addLayout(row, 1)
        self._refresh()

    # -- what it holds ---------------------------------------------------------
    @property
    def current_index(self) -> int:
        return self.strip.currentRow()

    def current(self) -> dict | None:
        row = self.current_index
        return self.entries[row] if 0 <= row < len(self.entries) else None

    def add(self, figure, title: str = "", command: str = "") -> dict:
        """Take a figure; it becomes the one shown. Newest first."""
        self.total += 1
        title = title or f"Figure {self.total}"
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=figure.dpi, bbox_inches="tight")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        entry = {"figure": figure, "title": title, "command": command.strip(),
                 "pixmap": pixmap}
        self.entries.insert(0, entry)
        item = QListWidgetItem(QIcon(pixmap), title)
        tip = title + (f"\n{entry['command'].splitlines()[0][:120]}" if entry["command"]
                       else "")
        item.setToolTip(tip)
        self.strip.insertItem(0, item)
        while len(self.entries) > MAX_FIGURES:
            self.entries.pop()
            self.strip.takeItem(self.strip.count() - 1)
        self.strip.setCurrentRow(0)
        self._refresh()
        self.added.emit()
        return entry

    def step(self, by: int) -> None:
        if self.entries:
            self.strip.setCurrentRow(max(0, min(len(self.entries) - 1,
                                                self.current_index + by)))

    def remove_current(self) -> None:
        row = self.current_index
        if 0 <= row < len(self.entries):
            self.entries.pop(row)
            self.strip.takeItem(row)
            if self.entries:
                self.strip.setCurrentRow(min(row, len(self.entries) - 1))
        self._refresh()

    def clear(self) -> None:
        self.entries = []
        self.strip.clear()
        self._refresh()

    def open_current(self) -> None:
        entry = self.current()
        if entry is not None and self.open_window is not None:
            self.open_window(entry["figure"], entry["title"])

    def save(self, path: str | Path, index: int | None = None) -> Path:
        """Write a figure (the one shown by default) to `path`; the suffix
        picks the format."""
        entry = self.entries[self.current_index if index is None else index]
        path = Path(path)
        entry["figure"].savefig(path, dpi=max(150, int(entry["figure"].dpi)),
                                bbox_inches="tight")
        return path

    def save_dialog(self) -> Path | None:
        entry = self.current()
        if entry is None:
            return None
        start = self.folder() if callable(self.folder) else Path.cwd()
        name = entry["title"].lower().replace(" ", "-") + ".png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the figure", str(Path(start) / name),
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg)")
        return self.save(path) if path else None

    # -- showing -----------------------------------------------------------------
    def _show_row(self, _row: int) -> None:
        self._refresh()

    def _refresh(self) -> None:
        entry = self.current()
        have = bool(self.entries)
        self.view.setVisible(have)
        self.empty.setVisible(not have)
        self.strip.setVisible(have)
        for button in (self.window_button, self.save_button, self.remove_button,
                       self.clear_button, self.previous_button, self.next_button):
            button.setEnabled(have)
        if entry is None:
            self.where.setText("")
            self.view.clear()
            return
        self.where.setText(f"{entry['title']} — {self.current_index + 1} of "
                           f"{len(self.entries)}")
        self.view.setToolTip((entry["command"].splitlines()[0][:120] + "\n"
                              if entry["command"] else "")
                             + "Double-click to open the figure in a window of its own")
        self._scale()

    def _scale(self) -> None:
        entry = self.current()
        if entry is None:
            return
        size = self.view.size()
        if size.width() < 10 or size.height() < 10:
            size = QSize(480, 360)
        self.view.setPixmap(entry["pixmap"].scaled(size, Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))



# -- History --------------------------------------------------------------------------------
def _summary(source: str) -> str:
    lines = [line for line in source.splitlines() if line.strip()] or [""]
    first = lines[0].strip()
    if len(first) > 90:
        first = first[:89] + "…"
    return first + (f"   (+{len(lines) - 1} lines)" if len(lines) > 1 else "")


class HistoryPanel(QWidget):
    """Every command run in the console, newest at the bottom, searchable."""

    def __init__(self, console=None, path: Path | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("onset_history")
        self.console = console
        #: The Editor panel commands are sent to; set when the panes are linked.
        self.editor = None
        self.path = path
        self.entries: list[dict] = load_history(self._path())
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        self.search = QLineEdit()
        self.search.setObjectName("onset_history_search")
        self.search.setPlaceholderText("Search past commands")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _text: self._fill())
        box.addWidget(self.search)
        self.list = QListWidget()
        self.list.setObjectName("onset_history_list")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.list.itemDoubleClicked.connect(lambda _item: self.to_editor())
        self.list.itemSelectionChanged.connect(self._enable)
        box.addWidget(self.list, 1)
        self.empty = _muted("Commands run in the console are listed here, from this "
                            "session and earlier ones.")
        box.addWidget(self.empty)
        bar = QHBoxLayout()
        bar.setSpacing(2)
        self.editor_button = _button("To the editor", "Put the selected commands at the "
                                     "editor's cursor, in the order they ran (double-click "
                                     "does the same)", self.to_editor, "onset_history_editor")
        self.run_button = _button("Run again", "Run the selected commands in the console",
                                  self.run_again, "onset_history_run")
        self.copy_button = _button("Copy", "Copy the selected commands", self.copy,
                                   "onset_history_copy")
        self.clear_button = _button("Clear…", "Forget the history (the report's record of "
                                    "each session's commands is kept)", self.clear,
                                    "onset_history_clear")
        for button in (self.editor_button, self.run_button, self.copy_button):
            bar.addWidget(button)
        bar.addStretch(1)
        bar.addWidget(self.clear_button)
        box.addLayout(bar)
        self._fill()

    def _path(self) -> Path:
        return self.path or history_path()

    # -- what it holds ---------------------------------------------------------
    def add(self, source: str) -> None:
        """A command just run. Blank ones and a repeat of the last are not kept."""
        text = source.rstrip()
        if not text.strip() or (self.entries and self.entries[-1]["source"] == text):
            return
        self.entries.append({"source": text, "when": time.strftime("%Y-%m-%d %H:%M")})
        del self.entries[:-MAX_HISTORY]
        save_history(self.entries, self._path())
        self._fill(scroll=True)

    def shown(self) -> list[dict]:
        """The entries the search leaves, oldest first."""
        words = self.search.text().lower().split()
        return [e for e in self.entries
                if all(word in e["source"].lower() for word in words)]

    def selected(self) -> list[str]:
        """The selected commands in the order they ran."""
        rows = sorted(self.list.row(item) for item in self.list.selectedItems())
        if not rows and self.list.currentRow() >= 0:
            rows = [self.list.currentRow()]
        return [self.list.item(row).data(Qt.UserRole) for row in rows]

    # -- doing -----------------------------------------------------------------
    def to_editor(self) -> bool:
        sources = self.selected()
        if not sources or self.editor is None:
            return False
        text = "\n".join(sources) + "\n"
        code = self.editor.current()
        if code is None:
            self.editor.new_file(text)
        else:
            cursor = code.textCursor()
            if cursor.positionInBlock() > 0:
                text = "\n" + text
            cursor.insertText(text)
            code.setTextCursor(cursor)
            code.setFocus()
        return True

    def run_again(self) -> bool:
        sources = self.selected()
        if not sources or self.console is None:
            return False
        self.console.run("\n".join(sources))
        return True

    def copy(self) -> None:
        sources = self.selected()
        if sources:
            QApplication.clipboard().setText("\n".join(sources))

    def clear(self, ask: bool = True) -> bool:
        if ask and self.entries and QMessageBox.question(
                self, "Clear the history",
                f"Forget all {len(self.entries)} commands? Each session's report keeps "
                "its own record of the commands run on it.") != QMessageBox.Yes:
            return False
        self.entries = []
        save_history(self.entries, self._path())
        self._fill()
        return True

    # -- showing -----------------------------------------------------------------
    def _fill(self, scroll: bool = False) -> None:
        self.list.clear()
        for entry in self.shown():
            item = QListWidgetItem(_summary(entry["source"]))
            item.setData(Qt.UserRole, entry["source"])
            item.setToolTip(f"{entry.get('when', '')}\n{entry['source'][:2000]}".strip())
            self.list.addItem(item)
        self.list.setVisible(bool(self.entries))
        self.empty.setVisible(not self.entries)
        if scroll or self.list.count():
            self.list.scrollToBottom()
        self._enable()

    def _enable(self) -> None:
        have = self.list.count() > 0
        for button in (self.editor_button, self.run_button, self.copy_button):
            button.setEnabled(have)
        self.clear_button.setEnabled(bool(self.entries))
