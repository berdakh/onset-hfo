"""The assistant's transcript as cards.

One card per exchange: the question, the answer, and a chip that says what
the answer is -- *Checked* against this analysis, *From the documents*,
*Not checked*, or *Refused* -- with the trace of how it got there (what the
model was given, what it asked for, what the checks made of it) folded under
the answer. The trace shows while the question runs, so the wait is a
visible process, and folds when the answer lands, so the transcript reads as
questions and answers rather than as a log.

A column of widgets in a scroll area rather than one rich-text document:
a document cannot fold a region, and a card that can be opened is the whole
point. `toPlainText` and `toHtml` are kept so that whatever read the old
transcript -- tests, mostly -- still can.
"""

from __future__ import annotations

import html
import re

from qtpy.QtCore import Qt, QTimer, QUrl, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["Transcript", "ExchangeCard", "CHIPS"]

#: Chip kinds -> (label, which palette colours). The label is what a reader
#: scans; the colour is spent on the two that need care.
CHIPS = {
    "checked": ("Checked", "accent"),
    "documents": ("From the documents", "neutral"),
    "unchecked": ("Not checked", "warn"),
    "refused": ("Refused", "bad"),
    "unverified": ("Unverified", "warn"),
    "note": ("No model needed", "neutral"),
    "thinking": ("Thinking…", "neutral"),
}

_TAGS = re.compile(r"<[^>]+>")


def _plain(markup: str) -> str:
    text = _TAGS.sub("", markup.replace("<br>", "\n").replace("</div>", "\n")
                     .replace("</li>", "\n").replace("</p>", "\n"))
    return html.unescape(re.sub(r"[ \t]+", " ", text)).strip()


def _chip_style(kind: str) -> str:
    p = theme.current()
    back, fore = {
        "accent": (p.accent, p.accent_text),
        "warn": (p.warn_surface, p.warn),
        "bad": (p.bad_surface, p.bad),
    }.get(kind, (p.surface_alt, p.text))
    return (f"background:{back};color:{fore};border-radius:9px;padding:1px 8px;"
            f"font-size:9pt;font-weight:600;")


class ExchangeCard(QFrame):
    """One question and what came of it."""

    #: A link in the answer: an evidence id, a document section.
    linkActivated = Signal(str)

    def __init__(self, question: str, parent=None):
        super().__init__(parent)
        p = theme.current()
        self.setObjectName("onset_exchange")
        self.setStyleSheet(
            f"QFrame#onset_exchange{{background:{p.surface};border:1px solid {p.separator};"
            f"border-radius:{theme.RADIUS}px;}}")
        self.question = question
        self._steps: list[str] = []
        self._chip_kind = "thinking"
        self._answer_html = ""

        self.asked = QLabel(html.escape(question))
        self.asked.setWordWrap(True)
        self.asked.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.asked.setStyleSheet(f"font-weight:600;color:{p.text};background:transparent;"
                                 "border:none;")
        self.chip = QLabel(CHIPS["thinking"][0])
        self.chip.setObjectName("onset_chip")
        self.chip.setStyleSheet(_chip_style("neutral"))
        self.meta = QLabel("")
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet(f"color:{p.text_muted};font-size:9pt;background:transparent;"
                                "border:none;")
        self.answer = QLabel("Thinking…")
        self.answer.setObjectName("onset_answer")
        self.answer.setWordWrap(True)
        self.answer.setOpenExternalLinks(False)
        self.answer.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.answer.linkActivated.connect(self.linkActivated.emit)
        self.answer.setStyleSheet(f"color:{p.text_muted};background:transparent;border:none;")

        self.toggle = QToolButton()
        self.toggle.setObjectName("onset_trace_toggle")
        self.toggle.setText("How it got there")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setAutoRaise(True)
        self.toggle.setStyleSheet(f"color:{p.text_muted};font-size:9pt;")
        self.toggle.toggled.connect(self._toggle)
        self.steps = QLabel("")
        self.steps.setObjectName("onset_steps")
        self.steps.setWordWrap(True)
        self.steps.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.steps.setStyleSheet(f"color:{p.text_muted};font-size:9pt;background:transparent;"
                                 f"border:none;padding-left:{theme.SPACING}px;")
        self.toggle.setVisible(False)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(theme.SPACING)
        head.addWidget(self.chip)
        head.addWidget(self.meta, 1)
        box = QVBoxLayout(self)
        box.setContentsMargins(theme.SPACING + 2, theme.SPACING, theme.SPACING + 2, theme.SPACING)
        box.setSpacing(theme.SPACING // 2)
        box.addWidget(self.asked)
        box.addLayout(head)
        box.addWidget(self.answer)
        box.addWidget(self.toggle)
        box.addWidget(self.steps)

    # -- while it runs ---------------------------------------------------------
    def add_step(self, markup: str) -> None:
        self._steps.append(markup)
        self.steps.setText("<br>".join(f"· {step}" for step in self._steps))
        self.steps.setVisible(True)
        self.toggle.setVisible(True)
        self.toggle.setText(f"How it got there ({len(self._steps)} step"
                            f"{'s' if len(self._steps) != 1 else ''})")

    # -- when it lands -----------------------------------------------------------
    def finish(self, kind: str, body: str, meta: str = "") -> None:
        """`kind` is a key of `CHIPS`; `body` and `meta` are HTML."""
        p = theme.current()
        label, colour = CHIPS.get(kind, CHIPS["note"])
        self._chip_kind = kind
        self.chip.setText(label)
        self.chip.setStyleSheet(_chip_style(colour))
        self.meta.setText(meta)
        self._answer_html = body
        self.answer.setText(body)
        self.answer.setStyleSheet(f"color:{p.text};background:transparent;border:none;")
        if kind in ("refused", "unchecked"):
            surface = p.bad_surface if kind == "refused" else p.warn_surface
            edge = p.bad if kind == "refused" else p.warn
            self.answer.setStyleSheet(
                f"color:{p.text};background:{surface};border:none;border-left:3px solid {edge};"
                f"border-radius:{theme.RADIUS - 3}px;padding:{theme.SPACING // 2}px {theme.SPACING}px;")
        # The trace folds once there is an answer to read instead.
        if self._steps:
            self.toggle.setChecked(False)
        else:
            self.toggle.setVisible(False)
            self.steps.setVisible(False)

    def _toggle(self, on: bool) -> None:
        self.steps.setVisible(bool(on))
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    @property
    def kind(self) -> str:
        return self._chip_kind

    # -- as text -----------------------------------------------------------------
    def toPlainText(self) -> str:      # noqa: N802  (the widget it replaces)
        parts = [self.question, self.chip.text(), _plain(self.answer.text())]
        if self._steps:
            parts.append(_plain("\n".join(self._steps)))
        if self.meta.text():
            parts.append(_plain(self.meta.text()))
        return "\n".join(part for part in parts if part)

    def toHtml(self) -> str:      # noqa: N802
        return (f"<div class='exchange'><b>{html.escape(self.question)}</b> "
                f"<span class='chip'>{html.escape(self.chip.text())}</span>"
                f"<div>{self.answer.text()}</div><div class='steps'>"
                + "<br>".join(self._steps) + f"</div><div class='meta'>{self.meta.text()}</div></div>")


class Transcript(QScrollArea):
    """The column of cards and notes, scrolled, newest at the bottom."""

    #: A link clicked in any card, as a URL.
    anchorClicked = Signal(QUrl)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setObjectName("onset_transcript")
        self._column = QWidget()
        self._column.setObjectName("onset_transcript_column")
        self._box = QVBoxLayout(self._column)
        self._box.setContentsMargins(0, 0, theme.SPACING // 2, 0)
        self._box.setSpacing(theme.SPACING)
        self._box.addStretch(1)
        self.setWidget(self._column)
        self._entries: list = []
        self._welcome: QLabel | None = None
        self.current: ExchangeCard | None = None

    # -- adding --------------------------------------------------------------
    def _add(self, widget) -> None:
        if self._welcome is not None:
            self._box.removeWidget(self._welcome)
            self._welcome.deleteLater()
            self._welcome = None
        self._box.insertWidget(self._box.count() - 1, widget)
        self._entries.append(widget)
        QTimer.singleShot(0, self._scroll_to_end)

    def _scroll_to_end(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def welcome(self, markup: str) -> None:
        """What an empty transcript says; gone with the first exchange."""
        p = theme.current()
        label = QLabel(markup)
        label.setObjectName("onset_welcome")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setStyleSheet(f"color:{p.text_muted};background:transparent;border:none;"
                            f"padding:{theme.SPACING}px;")
        self._box.insertWidget(0, label)
        self._welcome = label

    def add_note(self, markup: str) -> QLabel:
        """A line between cards: a system message, an error, a new conversation."""
        p = theme.current()
        label = QLabel(markup)
        label.setObjectName("onset_note")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        label.setStyleSheet(f"color:{p.text_muted};font-size:9pt;background:transparent;"
                            f"border:none;padding:0 {theme.SPACING // 2}px;")
        self._add(label)
        return label

    def start_exchange(self, question: str) -> ExchangeCard:
        card = ExchangeCard(question)
        card.linkActivated.connect(lambda link: self.anchorClicked.emit(QUrl(link)))
        self._add(card)
        self.current = card
        return card

    def cards(self) -> list[ExchangeCard]:
        return [w for w in self._entries if isinstance(w, ExchangeCard)]

    def clear(self) -> None:
        for widget in self._entries:
            self._box.removeWidget(widget)
            widget.deleteLater()
        self._entries = []
        self.current = None

    # -- as text -----------------------------------------------------------------
    def toPlainText(self) -> str:      # noqa: N802
        parts = []
        if self._welcome is not None:
            parts.append(_plain(self._welcome.text()))
        for widget in self._entries:
            parts.append(widget.toPlainText() if hasattr(widget, "toPlainText")
                         else _plain(widget.text()))
        return "\n\n".join(parts)

    def toHtml(self) -> str:      # noqa: N802
        parts = []
        if self._welcome is not None:
            parts.append(self._welcome.text())
        for widget in self._entries:
            parts.append(widget.toHtml() if hasattr(widget, "toHtml") else widget.text())
        return "\n".join(parts)
