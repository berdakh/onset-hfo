"""The study pages as Qt widgets: a row of choices, then the Markdown.

Each page is :func:`onset_review.studies.build` rendered by a `QTextBrowser`,
which reads GitHub-flavoured Markdown natively -- tables, images, code -- so
no HTML is written by hand and no extra package is needed. The controls the
site offers (band, metric, patient) are combo boxes above the text, and a
change rebuilds the document from the committed tables.

The Patients page carries the one link the site cannot make: a patient picked
there can be opened on the Recording page, if a window of theirs is cached.
"""

from __future__ import annotations

import re

from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QImage, QTextDocument
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_review import studies, theme

__all__ = ["StudyPage", "set_markdown"]


_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

#: Widest an embedded figure is drawn, in pixels. The site's figures are
#: rendered at print width and would otherwise push the page sideways.
IMAGE_MAX_WIDTH = 1100


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

    def place(match):
        alt, path = match.group(1), match.group(2)
        image = QImage(path)
        if image.isNull():
            return match.group(0)
        if image.width() > width:
            image = image.scaledToWidth(width, Qt.SmoothTransformation)
        name = f"onset-figure-{abs(hash(path))}"
        document.addResource(QTextDocument.ImageResource, QUrl(name), image)
        return f"![{alt}]({name})"

    text = _IMAGE.sub(place, text)
    try:
        document.setMarkdown(text, QTextDocument.MarkdownDialectGitHub)
    except (AttributeError, TypeError):      # an older binding
        browser.setMarkdown(text)


class StudyPage(QWidget):
    """One study page: choices above, the built Markdown below."""

    #: A cached window row the reviewer asked to open, from the Patients page.
    openRequested = Signal(dict)

    def __init__(self, key: str, *, cached=None, parent=None):
        super().__init__(parent)
        if key not in {k for k, _, _ in studies.STUDIES}:
            raise KeyError(f"no study page {key!r}")
        self.key = key
        self._cached = cached
        self._built = False
        self.combos: dict[str, QComboBox] = {}

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
        top.addStretch(1)

        self.view = QTextBrowser()
        self.view.setObjectName(f"onset_study_{key}")
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(True)
        self.view.document().setDefaultStyleSheet(
            f"table {{ border-collapse: collapse; }} "
            f"th, td {{ padding: 2px 8px; border: 1px solid {theme.current().surface_alt}; }} "
            f"th {{ background: {theme.current().surface_alt}; }} "
            f"blockquote {{ color: {theme.current().text_muted}; }}")

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.SPACING)
        if top.count() > 1:
            box.addLayout(top)
        box.addWidget(self.view, 1)

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
        combo.blockSignals(False)

    # -- building ----------------------------------------------------------
    def refresh(self) -> None:
        """Build on first show only: the tables are read once, and a page a
        reviewer never opens costs nothing."""
        if not self._built:
            self.rebuild()

    def rebuild(self) -> None:
        self._fill_subjects()
        try:
            text = studies.build(self.key, **self.choices())
        except Exception as error:      # noqa: BLE001 - a study page must not kill the window
            text = (f"# {self.key.title()}\n\n> **This page could not be built:** "
                    f"{error}\n\nThe same page is on the results site: {studies.SITE}")
        set_markdown(self.view, text)
        self._built = True
        if self.open_button is not None:
            self.open_button.setEnabled(self._window_for_current() is not None)
            self.open_button.setToolTip(
                "Open the earliest cached window of this patient on the Recording page"
                if self.open_button.isEnabled() else
                "No window of this patient is cached on this machine; fetch one "
                "with onset-hfo fetch")

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

    def keyPressEvent(self, event):      # noqa: N802  (Qt's spelling)
        if event.key() == Qt.Key_F5:
            self.rebuild()
            return
        super().keyPressEvent(event)
