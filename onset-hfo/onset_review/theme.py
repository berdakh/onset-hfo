"""The look: typography, colour, spacing, and the widgets built from them.

Qt's defaults are a 1990s desktop toolkit, and a clinical application that
looks like one gets read as a lab tool rather than something to rely on. The
reference points here are Apple's: a single accent colour used sparingly,
semantic colour reserved for meaning, generous whitespace instead of lines,
flat surfaces, a strict spacing grid, and type that does the work of
hierarchy so borders do not have to.

Three constraints make this more than decoration.

**Legibility beats elegance.** Every number on these screens is a measurement
someone may act on. Tabular figures, right-aligned, at a size that survives a
clinic monitor at arm's length. Where Apple would use grey-on-grey, this uses
a contrast ratio that passes.

**A warning must never be made quiet by the design.** The status-bar caveat,
the schematic banner on the 3D view and the preprocessing panel's red text are
the places this project refuses to be polite. The palette gives them their own
colours and the stylesheet never softens them.

**It must not fight MNE.** The trace browser is someone else's widget tree
inside our window. The stylesheet is written to style our panels and leave its
toolbar and its pyqtgraph canvas alone, which is why the selectors are narrow
and why there is no blanket `QWidget` rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtGui import QFont, QFontDatabase

__all__ = ["Palette", "LIGHT", "DARK", "apply_theme", "current", "qt_palette",
           "section_label", "plain_buttons", "scrolled", "help_button",
           "card", "muted", "SPACING", "RADIUS", "FONT_STACK"]

#: The spacing grid, in pixels. Everything is a multiple of four; most things
#: are a multiple of eight. A layout that picks its margins ad hoc reads as
#: untidy even when nobody can say why.
SPACING = 8
#: Corner radius for cards, inputs and buttons.
RADIUS = 8

#: Preferred UI faces, best first. Inter is the closest freely-available face
#: to the one this is modelled on and is packaged on most distributions; the
#: rest are what Ubuntu, GNOME and a minimal container actually ship. Resolved
#: at runtime against installed fonts rather than named blindly, because a
#: missing family silently falls back to something with the wrong metrics.
FONT_STACK = ("Inter", "SF Pro Text", "Helvetica Neue", "Cantarell", "Ubuntu",
              "Noto Sans", "DejaVu Sans")


@dataclass(frozen=True)
class Palette:
    """Named colours. Nothing in the interface writes a hex value directly."""

    name: str
    window: str          #: the application background
    surface: str         #: cards and panels that sit on it
    surface_alt: str     #: table headers, hovered rows
    text: str
    text_muted: str
    separator: str
    accent: str          #: the one colour that means "this is interactive"
    accent_text: str
    #: Semantic, and used for nothing else. A green that also means "primary
    #: button" stops meaning "inside the resection".
    good: str
    warn: str
    bad: str
    warn_surface: str
    bad_surface: str
    info_surface: str
    highlight: str       #: the tied-set tint in the findings table

    @property
    def dark(self) -> bool:
        return self.name == "dark"


LIGHT = Palette(
    name="light",
    window="#F5F5F7", surface="#FFFFFF", surface_alt="#F0F0F3",
    text="#1D1D1F", text_muted="#6E6E73", separator="#D8D8DE",
    accent="#0B6BCB", accent_text="#FFFFFF",
    good="#1D7A33", warn="#8A5300", bad="#B3261E",
    warn_surface="#FFF6E5", bad_surface="#FDECEA", info_surface="#EDF4FD",
    highlight="#FFF3CD",
)

DARK = Palette(
    name="dark",
    window="#1C1C1E", surface="#2C2C2E", surface_alt="#3A3A3C",
    text="#F2F2F7", text_muted="#A1A1A6", separator="#48484A",
    accent="#4C9AFF", accent_text="#0B1220",
    good="#4ED16B", warn="#FFB84D", bad="#FF6B5E",
    warn_surface="#3A2F16", bad_surface="#3A1F1C", info_surface="#16304A",
    highlight="#4A401C",
)


#: The palette in force. Set once by `apply_theme`; read by every panel as it
#: builds. A module-level value rather than an argument threaded through six
#: constructors -- and emphatically not a palette each panel imports by name,
#: which is what the first version did and why the dark theme rendered tinted
#: rows as light text on a light tint.
_CURRENT: Palette = LIGHT


def current() -> Palette:
    """The palette in force. Panels call this; they do not import a palette."""
    return _CURRENT


def set_current(palette: Palette) -> Palette:
    """Used by `apply_theme`, and by tests that render one theme directly."""
    global _CURRENT
    _CURRENT = palette
    return _CURRENT


def best_font() -> str:
    """The first face in `FONT_STACK` that is actually installed."""
    try:
        families = set(QFontDatabase.families())
    except (AttributeError, TypeError):      # Qt 5 spelling
        families = set(QFontDatabase().families())
    for family in FONT_STACK:
        if family in families:
            return family
    return ""


def apply_theme(app, palette: Palette | None = None) -> Palette:
    """Set the application font and stylesheet. Returns the palette used.

    Called once, on the `QApplication`. Panels never style themselves from
    scratch; they ask for a token here, which is what keeps one warning red the
    same red as the next.
    """
    chosen = set_current(palette or _detect(app))
    family = best_font()
    font = QFont(family) if family else QFont()
    font.setPointSizeF(10.0)
    font.setHintingPreference(QFont.PreferFullHinting)
    app.setFont(font)
    app.setPalette(qt_palette(chosen))
    app.setStyleSheet(stylesheet(chosen))
    app.setProperty("onset_palette", chosen.name)
    return chosen


def qt_palette(p: Palette):
    """The same colours as a `QPalette`, for everything the stylesheet misses.

    A stylesheet does not reach controls the platform style draws itself --
    check indicators, spin-box arrows, the text cursor -- so forcing a dark
    theme with CSS alone leaves those drawn for a light one. It is also what
    `mne-qt-browser` reads to decide whether its trace is light or dark, which
    is why the embedded trace follows without being told.
    """
    from qtpy.QtGui import QColor, QPalette

    palette = QPalette()
    window, surface = QColor(p.window), QColor(p.surface)
    text, muted_text = QColor(p.text), QColor(p.text_muted)
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        palette.setColor(group, QPalette.Window, window)
        palette.setColor(group, QPalette.Base, surface)
        palette.setColor(group, QPalette.AlternateBase, QColor(p.surface_alt))
        palette.setColor(group, QPalette.Button, surface)
        palette.setColor(group, QPalette.ToolTipBase, surface)
        palette.setColor(group, QPalette.Highlight, QColor(p.accent))
        palette.setColor(group, QPalette.HighlightedText, QColor(p.accent_text))
        palette.setColor(group, QPalette.Link, QColor(p.accent))
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText,
                     QPalette.ToolTipText):
            palette.setColor(group, role,
                             muted_text if group == QPalette.Disabled else text)
    return palette


def _detect(app) -> Palette:
    """Follow the desktop's own light/dark setting when it says one.

    Read off the window colour rather than a platform API: Qt reports that
    consistently on X11, Wayland and offscreen, and the three disagree about
    everything else.
    """
    try:
        window = app.palette().window().color()
    except Exception:
        return LIGHT
    return DARK if window.lightness() < 128 else LIGHT


def stylesheet(p: Palette) -> str:
    """The whole stylesheet, built from one palette.

    Deliberately not a blanket `QWidget` rule: MNE's browser and pyqtgraph live
    inside this window, and restyling every widget in the tree is how a
    third-party toolbar ends up unreadable. Selectors name the widgets this
    project creates.
    """
    return f"""
    QMainWindow, QDialog {{ background: {p.window}; }}
    QToolTip {{
        background: {p.surface}; color: {p.text};
        border: 1px solid {p.separator}; border-radius: {RADIUS // 2}px;
        padding: {SPACING // 2}px {SPACING}px;
    }}

    /* Docks: a title that reads as a label, not a title bar. */
    QDockWidget {{
        color: {p.text_muted};
        titlebar-close-icon: none; titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background: {p.window}; padding: {SPACING // 2}px {SPACING}px;
        border: none; text-align: left;
    }}
    QDockWidget > QWidget {{ background: {p.surface}; }}

    QTabBar::tab {{
        background: transparent; color: {p.text_muted};
        padding: {SPACING // 2}px {SPACING + 4}px; margin-right: 2px;
        border: none; border-radius: {RADIUS - 2}px;
    }}
    QTabBar::tab:selected {{ background: {p.surface}; color: {p.text}; }}
    QTabBar::tab:hover:!selected {{ color: {p.text}; }}

    /* The page sidebar: rows a finger can hit, the current one in the accent. */
    QListWidget#onset_pages {{ background: transparent; border: none; outline: 0; }}
    QListWidget#onset_pages::item {{
        padding: 3px {SPACING}px; border-radius: {RADIUS - 3}px;
        margin-right: {SPACING}px;
    }}
    QListWidget#onset_pages::item:selected {{
        background: {p.accent}; color: {p.accent_text};
    }}
    QListWidget#onset_pages::item:hover:!selected {{ background: {p.surface_alt}; }}

    /* Tables: rules replaced by space. A grid is the most common way to make
       a clinical table hard to scan. */
    QTableView {{
        background: {p.surface}; alternate-background-color: {p.surface};
        color: {p.text}; gridline-color: transparent;
        border: none; selection-background-color: {p.accent};
        selection-color: {p.accent_text};
    }}
    QHeaderView::section {{
        background: {p.surface}; color: {p.text_muted};
        padding: {SPACING // 2}px {SPACING}px; border: none;
        border-bottom: 1px solid {p.separator}; font-weight: 600;
    }}
    QTableView::item {{ padding: 2px {SPACING // 2}px; }}

    QGroupBox {{
        border: none; margin-top: {SPACING + 6}px;
        font-weight: 600; color: {p.text};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 0px; padding: 0 0 {SPACING // 2}px 0;
        color: {p.text_muted}; font-size: 9pt; font-weight: 600;
    }}

    QPushButton {{
        background: {p.surface}; color: {p.text};
        border: 1px solid {p.separator}; border-radius: {RADIUS}px;
        padding: {SPACING // 2 + 1}px {SPACING + 4}px; min-height: 18px;
    }}
    QPushButton:hover {{ background: {p.surface_alt}; }}
    QPushButton:pressed {{ background: {p.separator}; }}
    QPushButton:disabled {{ color: {p.text_muted}; border-color: {p.surface_alt}; }}
    QPushButton[primary="true"] {{
        background: {p.accent}; color: {p.accent_text}; border: none;
        font-weight: 600;
    }}
    QPushButton[primary="true"]:disabled {{
        background: {p.surface_alt}; color: {p.text_muted};
    }}

    QToolButton {{
        background: transparent; color: {p.text};
        border: 1px solid transparent; border-radius: {RADIUS - 2}px;
        padding: 2px {SPACING // 2}px;
    }}
    QToolButton:hover {{ background: {p.surface_alt}; }}

    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox {{
        background: {p.surface}; color: {p.text};
        border: 1px solid {p.separator}; border-radius: {RADIUS - 2}px;
        padding: 3px {SPACING}px; min-height: 18px;
        selection-background-color: {p.accent};
    }}
    QComboBox:focus, QLineEdit:focus, QAbstractSpinBox:focus {{
        border-color: {p.accent};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}

    QCheckBox, QRadioButton {{ color: {p.text}; spacing: {SPACING // 2 + 2}px; }}

    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent; border: none;
    }}
    QScrollBar:vertical {{ width: 10px; }}
    QScrollBar:horizontal {{ height: 10px; }}
    QScrollBar::handle {{
        background: {p.separator}; border-radius: 5px; min-height: 28px;
    }}
    QScrollBar::handle:hover {{ background: {p.text_muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QStatusBar {{ background: {p.window}; border-top: 1px solid {p.separator}; }}
    QStatusBar::item {{ border: none; }}

    QMenuBar {{ background: {p.window}; color: {p.text}; border: none; }}
    QMenuBar::item {{ padding: {SPACING // 2}px {SPACING + 2}px; background: transparent; }}
    QMenuBar::item:selected {{ background: {p.surface_alt}; border-radius: {RADIUS - 3}px; }}
    QMenu {{
        background: {p.surface}; color: {p.text};
        border: 1px solid {p.separator}; border-radius: {RADIUS}px;
        padding: {SPACING // 2}px;
    }}
    QMenu::item {{ padding: {SPACING // 2}px {SPACING + 8}px; border-radius: {RADIUS - 3}px; }}
    QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
    """


# -- the few building blocks panels share ---------------------------------

def section_label(text: str, palette: Palette | None = None):
    """A small, quiet heading. Type carries the hierarchy; no rule under it."""
    from qtpy.QtWidgets import QLabel

    p = palette or current()
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color:{p.text_muted};font-size:8pt;font-weight:700;"
        f"letter-spacing:0.08em;padding:{SPACING}px 0 2px 0;")
    return label


def muted(text: str, palette: Palette | None = None, size: int = 9):
    """Secondary text: captions, units, the sentence under a control."""
    from qtpy.QtWidgets import QLabel

    p = palette or current()
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color:{p.text_muted};font-size:{size}pt;")
    return label


def help_button(text: str, palette: Palette | None = None):
    """A small "?" whose tooltip is `text`: the hint that used to be a grey
    sentence beside a control, kept for whoever wants it and out of the way of
    whoever has read it. One per panel at most; a row of them is a manual."""
    from qtpy.QtWidgets import QToolButton

    p = palette or current()
    button = QToolButton()
    button.setText("?")
    button.setAutoRaise(True)
    button.setToolTip(text)
    button.setCursor(__import__("qtpy.QtCore", fromlist=["Qt"]).Qt.WhatsThisCursor)
    button.setFixedSize(20, 20)
    button.setStyleSheet(
        f"QToolButton{{color:{p.text_muted};border:1px solid {p.separator};"
        f"border-radius:10px;font-size:9pt;font-weight:700;padding:0;}}"
        f"QToolButton:hover{{color:{p.text};background:{p.surface_alt};}}")
    return button


def card(kind: str = "info", palette: Palette | None = None) -> str:
    """Stylesheet for a callout box: `info`, `warn`, `bad` or `plain`.

    One function so that every warning in the application is the same warning
    colour. Before this there were four hand-written hex values and two of them
    disagreed.
    """
    p = palette or current()
    surface, colour = {
        "info": (p.info_surface, p.text),
        "warn": (p.warn_surface, p.warn),
        "bad": (p.bad_surface, p.bad),
        "plain": (p.surface, p.text),
    }.get(kind, (p.info_surface, p.text))
    return (f"background:{surface};color:{colour};border-radius:{RADIUS}px;"
            f"padding:{SPACING}px {SPACING + 2}px;font-size:9pt;")


def plain_buttons(box) -> None:
    """Strip the platform's icons from a `QDialogButtonBox`'s buttons.

    Some Qt styles put a folder on Open and a cross on Cancel. Nothing else in
    this application has an icon on a button, and the two conventions side by
    side read as two applications. A stylesheet cannot reach an icon, so it is
    cleared here -- in one place, so the dialogs cannot disagree.
    """
    from qtpy.QtGui import QIcon

    for button in box.buttons():
        button.setIcon(QIcon())


def scrolled(body):
    """Put `body` in a frameless scroll area that fills the space it is given.

    Used by the docked panels whose content does not shrink on its own -- the
    ones built from stacked rows of labels and controls rather than from a
    table -- and for a reason that is about the window rather than the panel:
    Qt builds a main window's minimum size by summing its docks' minimums down
    each column, so a panel that insists on 360 px makes the whole window
    insist on it too. A few such panels and the window no longer fits a laptop
    -- at which point most window managers also withhold the maximise button,
    because a window that cannot shrink to the work area cannot be maximised
    into it.

    A scroll area breaks that chain: the panel keeps its own preferred size,
    the window stops inheriting it as a floor, and the one screen small enough
    to matter gets a scrollbar instead of a window with its status bar off the
    bottom. `setWidgetResizable` is what keeps it invisible on a large screen:
    the body is given the whole viewport whenever the viewport is big enough,
    so nothing scrolls until something has to.
    """
    from qtpy.QtWidgets import QFrame, QScrollArea

    area = QScrollArea()
    area.setWidget(body)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    return area
