"""The Workspace: the session's variables as a list, each one a window away.

What MATLAB's Workspace and Spyder's Variable Explorer do: a table of what
exists -- name, type, size, a glimpse of the value -- and a double-click
that opens the thing in a window of its own. The windows are free: a table
of events next to the array of samples next to the request, each dragged
where it is wanted, each closed when it is done with. The list itself is
`onset_review.variables`, Qt-free; this module only draws it.

Nothing here is a way of changing the analysis. The windows are read-only,
and Export writes a copy. The point is that nothing about a window is
hidden: a reviewer who wants to see the numbers behind a rate can open the
array the rate was computed from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QPoint, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme, variables
from onset_review.panels import DataFrameModel

__all__ = ["WorkspacePanel", "VariableWindow", "ArrayModel"]


class ArrayModel(QAbstractTableModel):
    """A read-only table over a 1-D or 2-D array. Qt asks only for the
    cells on screen, so a channels × samples array of five million values
    is as cheap to show as a small one. Higher-rank arrays are shown with
    their leading dimensions folded into rows, and the window says so."""

    def __init__(self, array: np.ndarray, row_labels=None, column_labels=None,
                 parent=None):
        super().__init__(parent)
        self.folded = array.ndim > 2
        if array.ndim == 0:
            array = array.reshape(1, 1)
        elif array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim > 2:
            array = array.reshape(-1, array.shape[-1])
        self._array = array
        self._rows = list(row_labels) if row_labels is not None else None
        self._columns = column_labels

    def rowCount(self, parent=QModelIndex()) -> int:      # noqa: N802
        return 0 if parent.isValid() else int(self._array.shape[0])

    def columnCount(self, parent=QModelIndex()) -> int:   # noqa: N802
        return 0 if parent.isValid() else int(self._array.shape[1])

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._array[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4g}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            if self._rows is not None and section < len(self._rows):
                return str(self._rows[section])
            return str(section)
        if callable(self._columns):
            return str(self._columns(section))
        if self._columns is not None and section < len(self._columns):
            return str(self._columns[section])
        return str(section)


def _label(text: str, muted: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if muted:
        label.setStyleSheet(f"color:{theme.current().text_muted};font-size:9pt;")
    return label


class VariableWindow(QWidget):
    """One variable in a window of its own: a table, an array, a tree or
    text, by what the object is. Free-floating, so several can be open
    side by side and dragged about."""

    def __init__(self, variable: variables.Variable, session=None, parent=None,
                 folder=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.variable = variable
        self.session = session
        #: Where Export suggests writing: the Files pane's current folder.
        self.folder = folder
        self.setObjectName(f"onset_variable_{variable.name}")
        value = variable.get()
        kind, size, summary = variables.describe(value)
        self.setWindowTitle(f"{variable.name} — {kind}" + (f" · {size}" if size else ""))
        self.view_kind = ""

        box = QVBoxLayout(self)
        box.setContentsMargins(theme.SPACING, theme.SPACING, theme.SPACING, theme.SPACING)
        box.setSpacing(theme.SPACING // 2)
        head = QHBoxLayout()
        head.addWidget(_label(f"<b>{variable.name}</b> &nbsp; {kind}"
                              + (f" &nbsp;·&nbsp; {size}" if size else "")), 1)
        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("Write a copy: .csv for a table, .npy for an array, "
                                      ".json otherwise")
        self.export_button.clicked.connect(self.export)
        head.addWidget(self.export_button)
        box.addLayout(head)
        if variable.about:
            box.addWidget(_label(variable.about, muted=True))

        self.body = self._body(value)
        box.addWidget(self.body, 1)
        box.addWidget(_label(summary, muted=True))
        self.resize(760, 520)

    def _body(self, value) -> QWidget:
        table = variables.as_table(value)
        array = variables.as_array(value)
        if table is not None and not isinstance(value, np.ndarray):
            self.view_kind = "table"
            view = QTableView()
            view.setObjectName("onset_variable_table")
            view.setModel(DataFrameModel(table))
            view.setSortingEnabled(True)
            view.setAlternatingRowColors(True)
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setEditTriggers(QAbstractItemView.NoEditTriggers)
            view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            view.horizontalHeader().setResizeContentsPrecision(100)
            return view
        if array is not None:
            self.view_kind = "array"
            rows, columns = None, None
            raw = self.session.raw if self.session is not None else None
            if raw is not None and array.ndim == 2 and array.shape[0] == len(raw.ch_names):
                rows = list(raw.ch_names)
                sfreq = float(raw.info["sfreq"])
                offset = float(getattr(self.session, "t_offset", 0.0))

                digits = max(3, int(np.ceil(np.log10(max(sfreq, 1.0)))))

                def columns(i, offset=offset, sfreq=sfreq, digits=digits):
                    return f"{offset + i / sfreq:.{digits}f} s"
            view = QTableView()
            view.setObjectName("onset_variable_array")
            model = ArrayModel(array, rows, columns)
            view.setModel(model)
            view.setEditTriggers(QAbstractItemView.NoEditTriggers)
            view.horizontalHeader().setDefaultSectionSize(100)
            view.verticalHeader().setDefaultSectionSize(20)
            if model.folded:
                self.setToolTip("Leading dimensions folded into rows")
            return view
        if variables.as_tree(value):
            self.view_kind = "tree"
            tree = QTreeWidget()
            tree.setObjectName("onset_variable_tree")
            tree.setHeaderLabels(["Field", "Value"])
            tree.setAlternatingRowColors(True)
            tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            tree.header().setStretchLastSection(True)
            _fill(tree.invisibleRootItem(), variables.as_tree(value))
            tree.expandToDepth(0)
            return tree
        self.view_kind = "text"
        text = QPlainTextEdit()
        text.setObjectName("onset_variable_text")
        text.setReadOnly(True)
        text.setPlainText("" if value is None else str(value))
        return text

    def export(self, path: str | Path | None = None) -> Path | None:
        """Write a copy where the reviewer says; `path` skips the dialog."""
        value = self.variable.get()
        suffix = variables.export_suffix(value)
        if path is None:
            label = ""
            if self.session is not None:
                label = str(getattr(self.session.request, "subject", "") or "")
            suggested = f"{label + '_' if label else ''}{self.variable.name}{suffix}"
            if self.folder is not None:
                suggested = str(Path(self.folder) / suggested)
            path, _filter = QFileDialog.getSaveFileName(
                self, f"Export {self.variable.name}", suggested, f"*{suffix}")
            if not path:
                return None
        return variables.export(value, path)


def _fill(parent: QTreeWidgetItem, rows) -> None:
    for key, text, children in rows:
        item = QTreeWidgetItem(parent, [str(key), text])
        item.setToolTip(1, text)
        if children:
            _fill(item, children)


class WorkspacePanel(QWidget):
    """The list: group by group, name, type, size and a glimpse. Double-click
    a row, or select it and press Open, to get it in a window of its own."""

    #: The name just opened in a window.
    opened = Signal(str)

    COLUMNS = ("Name", "Type", "Size", "Value")

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        self.session = session
        self.windows: list[VariableWindow] = []
        self._by_name: dict[str, variables.Variable] = {}
        #: A callable giving the current folder, set by whoever holds the
        #: Files pane; Export suggests writing there.
        self.folder = None
        #: The Console pane, when there is one: its names are listed too.
        self.console = None

        self.search = QLineEdit()
        self.search.setObjectName("onset_workspace_search")
        self.search.setPlaceholderText("Filter by name…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.open_button = QPushButton("Open in a window")
        self.open_button.setObjectName("onset_workspace_open")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda _=False: self.open_selected())
        self.export_button = QPushButton("Export…")
        self.export_button.setObjectName("onset_workspace_export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(lambda _=False: self.export_selected())
        self.copy_button = QPushButton("Copy name")
        self.copy_button.setEnabled(False)
        self.copy_button.setToolTip("Put the variable's name on the clipboard")
        self.copy_button.clicked.connect(lambda _=False: self.copy_name())

        self.tree = QTreeWidget()
        self.tree.setObjectName("onset_workspace")
        self.tree.setHeaderLabels(list(self.COLUMNS))
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        self.tree.setMinimumWidth(160)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda item, _col: self._open_item(item))
        self.tree.itemActivated.connect(lambda item, _col: self._open_item(item))

        self.count = _label("", muted=True)
        # Narrow on purpose: a pane's minimum width is added to the window's,
        # and a wide button row here made the window wider than the screen.
        self.search.setMinimumWidth(80)
        for button, text in ((self.open_button, "Open"), (self.export_button, "Export…"),
                             (self.copy_button, "Copy name")):
            button.setText(text)
        self.open_button.setToolTip("Open the selected variable in a window of its own "
                                    "(or double-click it)")
        bar = QHBoxLayout()
        bar.setSpacing(theme.SPACING // 2)
        bar.addWidget(self.search, 1)
        bar.addWidget(theme.help_button(
            "Everything this window holds, as MATLAB's Workspace or Spyder's "
            "Variable Explorer would list it. Double-click a row to open it in a "
            "window of its own; open as many as you like and drag them where you "
            "want them. Export writes a copy: .csv for a table, .npy for an array, "
            ".json otherwise. Nothing here changes the analysis."))
        actions = QHBoxLayout()
        actions.setSpacing(theme.SPACING // 2)
        for button in (self.open_button, self.export_button, self.copy_button):
            actions.addWidget(button)
        actions.addStretch(1)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING // 2)
        box.addLayout(bar)
        box.addWidget(self.tree, 1)
        box.addLayout(actions)
        box.addWidget(self.count)
        self.refresh()

    # -- the list -------------------------------------------------------------
    def refresh(self, session=None) -> None:
        """Rebuild the list from the session (a new one if given)."""
        if session is not None:
            self.session = session
        self.tree.clear()
        self._by_name = {}
        console = self.console.user_variables() if self.console is not None else {}
        if self.session is None and not console:
            self.count.setText("No recording open.")
            return
        p = theme.current()
        groups: dict[str, QTreeWidgetItem] = {}
        for name, about in variables.GROUPS:
            item = QTreeWidgetItem(self.tree, [name, "", "", ""])
            item.setToolTip(0, about)
            item.setFlags(Qt.ItemIsEnabled)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, Qt.gray)
            groups[name] = item
        listed = variables.variables(self.session, console)
        for variable in listed:
            self._by_name[variable.name] = variable
            row = QTreeWidgetItem(groups[variable.group],
                                  [variable.name, variable.kind, variable.size,
                                   variable.summary])
            row.setData(0, Qt.UserRole, variable.name)
            row.setToolTip(0, variable.about)
            row.setToolTip(3, variable.summary)
            if variable.missing:
                for column in range(4):
                    row.setForeground(column, QColor(p.text_muted))
        for group in groups.values():
            group.setHidden(group.childCount() == 0)
        self.tree.expandAll()
        self.count.setText(f"{len(listed)} variables · double-click one to open it "
                           "in a window")
        self._filter(self.search.text())

    def names(self) -> list[str]:
        return list(self._by_name)

    def variable(self, name: str) -> variables.Variable | None:
        return self._by_name.get(name)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for g in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(g)
            shown = 0
            for c in range(group.childCount()):
                child = group.child(c)
                hit = needle in child.text(0).lower() or needle in child.text(1).lower()
                child.setHidden(bool(needle) and not hit)
                shown += 0 if child.isHidden() else 1
            group.setHidden(shown == 0)

    def _selected_name(self) -> str | None:
        items = self.tree.selectedItems()
        if not items:
            return None
        name = items[0].data(0, Qt.UserRole)
        return str(name) if name else None

    def _selection_changed(self) -> None:
        on = self._selected_name() is not None
        for button in (self.open_button, self.export_button, self.copy_button):
            button.setEnabled(on)

    def _open_item(self, item: QTreeWidgetItem) -> None:
        name = item.data(0, Qt.UserRole)
        if name:
            self.open(str(name))

    # -- the windows --------------------------------------------------------
    def open(self, name: str) -> VariableWindow | None:
        """Open `name` in a window of its own, or raise the one already open."""
        variable = self._by_name.get(name)
        if variable is None:
            return None
        for window in self.windows:
            if window.variable.name == name and not window.isHidden():
                window.raise_()
                window.activateWindow()
                return window
        window = VariableWindow(variable, self.session, folder=self._folder())
        self.windows = [w for w in self.windows if not w.isHidden()] + [window]
        offset = 24 * (len(self.windows) - 1)
        window.move(self.mapToGlobal(self.rect().topLeft()) + QPoint(60 + offset, 60 + offset))
        window.show()
        self.opened.emit(name)
        return window

    def open_selected(self) -> VariableWindow | None:
        name = self._selected_name()
        return self.open(name) if name else None

    def export_selected(self, path: str | Path | None = None) -> Path | None:
        name = self._selected_name()
        if not name:
            return None
        window = VariableWindow(self._by_name[name], self.session, folder=self._folder())
        try:
            return window.export(path)
        finally:
            window.deleteLater()

    def _folder(self):
        try:
            return self.folder() if callable(self.folder) else self.folder
        except Exception:        # noqa: BLE001 - a suggestion, never a failure
            return None

    def copy_name(self) -> None:
        name = self._selected_name()
        if name:
            QApplication.clipboard().setText(name)

    def close_windows(self) -> None:
        for window in self.windows:
            window.close()
        self.windows = []

    def closeEvent(self, event) -> None:      # noqa: N802
        self.close_windows()
        super().closeEvent(event)
