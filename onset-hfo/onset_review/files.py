"""The Files pane: a current folder, as Spyder and MATLAB keep one.

A path bar with Up, Home and Browse, and the folder's contents as a tree.
Recordings the importer can read are listed in full colour; anything else
is greyed but still shown, because a folder is easier to recognise whole.
Double-click a folder to go into it, a recording to open it -- through the
same import dialog as File → Open a file, so the channel types are still
confirmed before anything is analysed.

The current folder is remembered between launches (`files.json` beside the
assistant's settings) and is where the Workspace's Export suggests writing,
so what is exported shows up here.
"""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import QDir, QModelIndex, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["FilesPanel", "current_folder", "set_current_folder", "is_recording"]


def _state_file() -> Path:
    from onset_review.assistant_config import config_path

    return config_path().with_name("files.json")


def current_folder() -> Path:
    """The folder last made current, if it still exists; else the home folder."""
    try:
        folder = Path(json.loads(_state_file().read_text(encoding="utf-8"))["folder"])
        if folder.is_dir():
            return folder
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return Path.home()


def set_current_folder(folder: str | Path) -> Path:
    folder = Path(folder).expanduser().resolve()
    try:
        path = _state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"folder": str(folder)}) + "\n", encoding="utf-8")
    except OSError:
        pass        # a folder that cannot be remembered is still the folder
    return folder


def is_recording(path: str | Path) -> bool:
    """Whether the importer has a reader for this file."""
    from onset_hfo.io import detect_format

    path = Path(path)
    return path.is_file() and detect_format(path) is not None


class FilesPanel(QWidget):
    """The current folder and what is in it."""

    #: A recording the reviewer asked to open (absolute path).
    openRequested = Signal(str)
    #: The folder just made current.
    folderChanged = Signal(str)

    def __init__(self, folder: str | Path | None = None, parent=None):
        super().__init__(parent)
        from onset_hfo.io import supported_suffixes

        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("onset_files_path")
        self.path_edit.setToolTip("The current folder. Type a path and press Enter.")
        self.path_edit.returnPressed.connect(lambda: self.set_folder(self.path_edit.text()))

        def tool(text: str, tip: str, slot) -> QToolButton:
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(lambda _=False: slot())
            return button

        self.up_button = tool("↑", "Up one folder", self.go_up)
        self.home_button = tool("⌂", "Your home folder", lambda: self.set_folder(Path.home()))
        self.browse_button = tool("…", "Choose a folder", self.browse)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.model.setNameFilters([f"*{s}" for s in supported_suffixes()])
        self.model.setNameFilterDisables(True)       # others greyed, not hidden
        self.tree = QTreeView()
        self.tree.setObjectName("onset_files")
        self.tree.setModel(self.model)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.tree.setColumnHidden(2, True)           # "Kind": the suffix says it
        self.tree.doubleClicked.connect(self._activated)
        self.tree.activated.connect(self._activated)

        self.hint = QLabel("Double-click a recording to open it; a folder to go into it.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")

        bar = QHBoxLayout()
        bar.setSpacing(2)
        bar.addWidget(self.up_button)
        bar.addWidget(self.home_button)
        bar.addWidget(self.path_edit, 1)
        bar.addWidget(self.browse_button)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING // 2)
        box.addLayout(bar)
        box.addWidget(self.tree, 1)
        box.addWidget(self.hint)
        self.folder = Path()
        self.set_folder(folder or current_folder(), remember=False)

    # -- where we are -----------------------------------------------------
    def set_folder(self, folder: str | Path, remember: bool = True) -> bool:
        folder = Path(str(folder)).expanduser()
        if not folder.is_dir():
            self.path_edit.setText(str(self.folder))
            self.hint.setText(f"No such folder: {folder}")
            return False
        folder = folder.resolve()
        self.folder = folder
        root = self.model.setRootPath(str(folder))
        self.tree.setRootIndex(root)
        self.path_edit.setText(str(folder))
        self.up_button.setEnabled(folder.parent != folder)
        self.hint.setText("Double-click a recording to open it; a folder to go into it.")
        if remember:
            set_current_folder(folder)
        self.folderChanged.emit(str(folder))
        return True

    def go_up(self) -> None:
        if self.folder.parent != self.folder:
            self.set_folder(self.folder.parent)

    def browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose the current folder",
                                                  str(self.folder))
        if chosen:
            self.set_folder(chosen)

    # -- opening ----------------------------------------------------------
    def _activated(self, index: QModelIndex) -> None:
        path = Path(self.model.filePath(index))
        self.open_path(path)

    def open_path(self, path: str | Path) -> bool:
        """Go into a folder, or ask to open a recording. False for anything
        else, with the reason in the hint line."""
        path = Path(path)
        if path.is_dir():
            return self.set_folder(path)
        if is_recording(path):
            self.openRequested.emit(str(path.resolve()))
            return True
        self.hint.setText(f"{path.name} is not a recording this software reads; "
                          "File → Open a file lists the formats.")
        return False

    def listed(self) -> list[str]:
        """Names in the current folder as the tree has loaded them."""
        root = self.tree.rootIndex()
        return [self.model.fileName(self.model.index(row, 0, root))
                for row in range(self.model.rowCount(root))]
