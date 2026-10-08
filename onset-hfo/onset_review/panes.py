"""Pane layouts: the Workspace, Files and Console arranged in one step.

Three panes and any number of variable and figure windows make a busy
screen. Spyder answers that with *Window layouts* and a way to close every
plot at once; this is the same for this window. `arrange` puts the three
panes where a named layout wants them, and `add_pane_menu` gives the View
menu the pane toggles, the layouts, a way to close every window the panes
opened, and (on the page window) the sidebar.

The placement follows the rule Qt imposes: every pane is shown while it is
placed, stacked first and tabbed second, then the ones the layout does not
want are hidden. A hidden pane keeps the place it was given, so showing it
later from View puts it back where the layout said.
"""

from __future__ import annotations

from qtpy.QtCore import Qt

__all__ = ["PANE_KEYS", "PANE_LAYOUTS", "arrange", "add_pane_menu", "close_pane_windows"]

PANE_KEYS = ("workspace", "files", "console")

#: (name, what it is, shortcut). Page only first: it is the one a crowded
#: screen wants.
PANE_LAYOUTS = (
    ("Page only", "Close every pane: the page and nothing else beside it",
     "Ctrl+Shift+P"),
    ("Spyder", "Workspace and Files tabbed on the right, the Console under them", ""),
    ("MATLAB", "Files on the left; the Workspace on the right, the Console under it", ""),
)

#: What each pane toggle's shortcut is.
SHORTCUTS = {"workspace": "Ctrl+Shift+W", "files": "Ctrl+Shift+F",
             "console": "Ctrl+Shift+I"}


def arrange(host, docks: dict, name: str) -> list[str]:
    """Place the three panes of `docks` on `host` (a QMainWindow) as layout
    `name` has them. Returns the keys left showing. Unknown names raise."""
    if name not in {n for n, _, _ in PANE_LAYOUTS}:
        raise ValueError(f"no pane layout called {name!r}")
    workspace, files, console = (docks[key] for key in PANE_KEYS)
    panes = (workspace, files, console)
    for dock in panes:
        dock.setFloating(False)
        dock.setVisible(True)
        host.removeDockWidget(dock)
    if name == "MATLAB":
        host.addDockWidget(Qt.LeftDockWidgetArea, files)
        host.addDockWidget(Qt.RightDockWidgetArea, workspace, Qt.Vertical)
        host.addDockWidget(Qt.RightDockWidgetArea, console, Qt.Vertical)
    else:
        # Spyder's placement; Page only keeps it, hidden, so a pane shown
        # again from View comes back to Spyder's place.
        host.addDockWidget(Qt.RightDockWidgetArea, workspace, Qt.Vertical)
        host.addDockWidget(Qt.RightDockWidgetArea, console, Qt.Vertical)
        host.tabifyDockWidget(workspace, files)
    for dock in panes:
        dock.setVisible(True)
    shown = () if name == "Page only" else PANE_KEYS
    for key, dock in zip(PANE_KEYS, panes, strict=True):
        dock.setVisible(key in shown)
    if shown:
        workspace.raise_()
    return list(shown)


def close_pane_windows(panels: dict) -> int:
    """Close every window the Workspace and the Console opened: variable
    windows and figures. Returns how many were closed."""
    closed = 0
    workspace, console = panels.get("workspace"), panels.get("console")
    if workspace is not None:
        closed += sum(1 for w in workspace.windows if not w.isHidden())
        workspace.close_windows()
    if console is not None:
        closed += sum(1 for w in console.figure_windows if not w.isHidden())
        console.close_figures()
    return closed


def add_pane_menu(menu, host, docks: dict, panels: dict, *, toggles: bool = True,
                  layouts: bool = True, apply=None, sidebar=None) -> dict:
    """The View menu's pane entries. `toggles` adds each pane's own show/hide
    (the docked window lists those with its other docks already). `apply` is
    called with a layout name instead of `arrange` when the host has its own
    way (the page window keeps its sizes and its narrow-screen rule).
    `sidebar` is the page window, for the sidebar toggle. Returns the actions
    by name, for tests and for anything that wants to trigger one."""
    actions: dict = {}
    if not all(key in docks for key in PANE_KEYS):
        return actions
    if toggles:
        for key in PANE_KEYS:
            action = docks[key].toggleViewAction()
            action.setShortcut(SHORTCUTS[key])
            action.setToolTip(docks[key].toolTip())
            menu.addAction(action)
            actions[key] = action
    if layouts:
        submenu = menu.addMenu("Pane &layout")
        submenu.setToolTipsVisible(True)
        for name, what, shortcut in PANE_LAYOUTS:
            action = submenu.addAction(name)
            action.setToolTip(what)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(
                lambda _=False, name=name: (apply(name) if apply is not None
                                            else arrange(host, docks, name)))
            actions[name] = action
        actions["layouts"] = submenu
    closer = menu.addAction("Close variable and figure &windows")
    closer.setToolTip("Every window the Workspace and the Console opened")
    closer.triggered.connect(lambda _=False: close_pane_windows(panels))
    actions["close"] = closer
    if sidebar is not None:
        side = menu.addAction("Page &sidebar")
        side.setShortcut("Ctrl+Shift+B")
        side.setCheckable(True)
        side.setToolTip("The list of pages on the left; hide it to give the page "
                        "its width")
        side.setChecked(sidebar.sidebar_shown)
        side.toggled.connect(sidebar.set_sidebar)
        menu.aboutToShow.connect(lambda: side.setChecked(sidebar.sidebar_shown))
        actions["sidebar"] = side
    return actions
