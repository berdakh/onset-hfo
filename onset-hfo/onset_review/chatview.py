"""The Chat page: the local model on its own, unconnected to the analysis.

The evidence assistant answers about this window and checks every number;
the Chat page is where a reader asks the same local model anything else,
knowing exactly what they get. Three things keep it honest beside the rest
of the window, none of them a guard on the reader:

* a permanent banner says the page is not connected to this recording and
  nothing on it is checked, including about medicine;
* no patient data reaches it: the page gets no session, no store, no
  channel names, no window, and nothing is added to what the reader types;
* nothing from it enters the report or the saved read.

It uses the model chosen on the Assistant page, read from the saved
settings each time the page is shown, so one model serves both.
"""

from __future__ import annotations

import html

from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_agent.backends import make_backend
from onset_agent.prompts import CHAT_PROMPT
from onset_review import theme
from onset_review.assistant import ANSWER_TIMEOUT, SMALL_PT, TEXT_PT, _paragraphs, _run
from onset_review.assistant_config import load_defaults
from onset_review.theme import muted

__all__ = ["ChatPanel", "BANNER", "CHAT_TOKENS"]

BANNER = ("<b>General chat with the local model.</b> Not connected to this recording: "
          "it sees nothing of the patient unless you type it. Nothing here is checked; it "
          "can be wrong, including about medicine. For this window, use Assistant › This recording.")
#: Longer than the evidence assistant's answers: nothing here has to be
#: checked, and a general question may want a paragraph.
CHAT_TOKENS = 700
#: Turns kept in the conversation.
TURNS = 12
#: What an empty page offers: a click on one puts it in the box.
EXAMPLES = ("What is the difference between a ripple and a fast ripple?",
            "Explain a bipolar montage to a medical student.",
            "Summarise the evidence for HFOs as a biomarker, with the caveats.")


class _ChatWorker(QThread):
    """One exchange, off the GUI thread; a local model can take a while."""

    def __init__(self, kind: str, model: str, base_url: str, messages: list[dict], parent=None):
        super().__init__(parent)
        self._kind, self._model, self._base_url = kind, model, base_url
        self._messages = messages
        self._backend = None
        self._stop = False
        self.reply: str = ""
        self.error: str | None = None

    def stop(self) -> None:
        self._stop = True
        backend = self._backend
        if backend is not None:
            backend.abort()

    def run(self) -> None:
        try:
            self._backend = make_backend(self._kind, model=self._model or None,
                                         base_url=self._base_url or None,
                                         max_tokens=CHAT_TOKENS, timeout=ANSWER_TIMEOUT)
            if self._stop:
                self._backend.abort()
            if not getattr(self._backend, "is_language_model", True):
                self.error = "no model is loaded"
                return
            message = self._backend.chat(self._messages, [])
            self.reply = (message.content or "").strip()
        except Exception as error:      # noqa: BLE001 - reported in the transcript
            self.error = f"{type(error).__name__}: {error}"


class ChatPanel(QWidget):
    """Ask the local model anything. Unchecked, unconnected, and labelled so."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[tuple[str, str]] = []
        self._worker = None
        self._kind, self._model, self._base_url = "scripted", "", ""
        tokens = theme.current()

        self.banner = QLabel(BANNER)
        self.banner.setObjectName("onset_chat_banner")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(
            f"color:{tokens.warn};font-size:9pt;padding:2px 0 0 0;")
        self.banner.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.model_line = muted("")
        self.model_line.setObjectName("onset_chat_model")
        self.model_line.setWordWrap(True)

        self.transcript = QTextBrowser()
        self.transcript.setObjectName("onset_chat_transcript")
        self.transcript.setOpenLinks(False)
        self.transcript.setStyleSheet(f"font-size:{TEXT_PT}pt;")
        self.transcript.anchorClicked.connect(self._example_picked)
        self._fresh = True
        self._note = ""

        self.question = QLineEdit()
        self.question.setObjectName("onset_chat_question")
        self.question.setPlaceholderText("Ask the model anything. It cannot see this recording.")
        self.question.returnPressed.connect(lambda: self.ask())
        self.ask_button = QPushButton("Send")
        self.ask_button.setObjectName("onset_chat_send")
        self.ask_button.clicked.connect(lambda _=False: self.ask())
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("onset_chat_stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda _=False: self.stop())
        self.new_button = QPushButton("New conversation")
        self.new_button.setObjectName("onset_chat_new")
        self.new_button.clicked.connect(lambda _=False: self.new_conversation())
        self.clock = QLabel("")
        self.clock.setStyleSheet(f"color:{tokens.text_muted};font-size:{SMALL_PT}pt;")

        top = QHBoxLayout()
        top.addWidget(self.model_line, 1)
        top.addWidget(self.new_button)
        row = QHBoxLayout()
        row.addWidget(self.question, 1)
        row.addWidget(self.ask_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.clock)

        box = QVBoxLayout(self)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(6)
        box.addWidget(self.banner)
        box.addLayout(top)
        box.addWidget(self.transcript, 1)
        box.addLayout(row)
        self.refresh_model()
        self._welcome()

    # -- which model ----------------------------------------------------------
    def refresh_model(self) -> None:
        """The Assistant page's choice, from the saved settings."""
        defaults = load_defaults()
        self._kind = str(getattr(defaults, "kind", "scripted") or "scripted")
        self._model = str(getattr(defaults, "model", "") or "")
        self._base_url = str(getattr(defaults, "base_url", "") or "")
        if self._kind == "scripted":
            self.model_line.setText("No model is loaded — choose one under Assistant › This recording.")
            self.question.setEnabled(False)
            self.ask_button.setEnabled(False)
        else:
            self.model_line.setText(
                f"{self._model or 'the default on the server'} via {self._kind} · "
                "the Assistant page's choice · answers are the model's own")
            self.question.setEnabled(True)
            self.ask_button.setEnabled(True)
        if getattr(self, "_fresh", False):
            self._welcome()

    @property
    def available(self) -> bool:
        return self._kind != "scripted"

    def showEvent(self, event):      # noqa: N802  (Qt's spelling)
        self.refresh_model()
        super().showEvent(event)

    # -- the conversation ---------------------------------------------------
    def messages(self, question: str) -> list[dict]:
        out = [{"role": "system", "content": CHAT_PROMPT}]
        for asked, answered in self._history[-TURNS:]:
            out.append({"role": "user", "content": asked})
            out.append({"role": "assistant", "content": answered})
        out.append({"role": "user", "content": question})
        return out

    def ask(self, question: str | None = None) -> None:
        text = (question if isinstance(question, str) and question
                else self.question.text()).strip()
        if not text:
            return
        if not self.available:
            self._say_system("No model is loaded; choose one under Assistant › This recording.")
            return
        self.question.clear()
        self._append(f"<div style='margin:6px 0 2px;'><b>You:</b> {html.escape(text)}</div>")
        self._worker = _ChatWorker(self._kind, self._model, self._base_url,
                                   self.messages(text), self)
        self._set_busy(True)
        try:
            _run(self._worker)
        finally:
            self._set_busy(False)
            self.question.setFocus()
        if self._worker.error:
            self._say_system(f"The model could not answer: {html.escape(self._worker.error)}")
            return
        reply = self._worker.reply or "(the model returned nothing)"
        self._append(
            f"<div style='margin:2px 0 6px;padding:6px;background:{theme.current().surface};"
            f"border-left:3px solid {theme.current().text_muted};'>"
            f"{_paragraphs(reply)}"
            f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
            f"margin-top:3px;'>{html.escape(self._model or self._kind)} · unchecked</div></div>")
        self._history.append((text, reply))
        self._history = self._history[-TURNS:]

    def stop(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            self._say_system("Stopping…")

    def new_conversation(self) -> None:
        self._history = []
        self._note = "New conversation: earlier messages are forgotten."
        self._welcome()

    # -- the empty page ---------------------------------------------------------
    def _welcome(self) -> None:
        """What an empty transcript shows: what the page is, and three things
        to try. Gone with the first message."""
        tokens = theme.current()
        if self.available:
            lead = "Ask the model anything."
            tries = "".join(
                f"<div style='margin:4px 0;'><a href='example:{i}' style='color:{tokens.accent};"
                f"text-decoration:none;'>{html.escape(text)}</a></div>"
                for i, text in enumerate(EXAMPLES))
        else:
            lead = "No model is loaded."
            tries = ("<div style='margin:4px 0;'>Choose one under <b>Assistant › This recording</b> "
                     "— the box above its transcript — and this page talks to it.</div>")
        note = (f"<div style='font-size:{SMALL_PT}pt;margin-bottom:12px;'>"
                f"{html.escape(self._note)}</div>" if self._note else "")
        self.transcript.setHtml(
            f"<div style='margin:48px 24px;color:{tokens.text_muted};'>{note}"
            f"<div style='font-size:{TEXT_PT + 2}pt;color:{tokens.text};'>{lead}</div>"
            f"<div style='margin:4px 0 12px;'>It cannot see this recording, and nothing "
            f"it says is checked.</div>{tries}</div>")
        self._fresh = True

    def _example_picked(self, url) -> None:
        link = url.toString() if hasattr(url, "toString") else str(url)
        if link.startswith("example:"):
            index = int(link.split(":", 1)[1])
            if 0 <= index < len(EXAMPLES):
                self.question.setText(EXAMPLES[index])
                self.question.setFocus()

    # -- plumbing --------------------------------------------------------------
    def _set_busy(self, on: bool) -> None:
        self.question.setEnabled(not on and self.available)
        self.ask_button.setEnabled(not on and self.available)
        self.new_button.setEnabled(not on)
        self.stop_button.setEnabled(on)
        self.clock.setText("thinking…" if on else "")

    def _append(self, body: str) -> None:
        if self._fresh:
            self.transcript.clear()
            self._fresh = False
            if self._note:
                note, self._note = self._note, ""
                self._say_system(html.escape(note))
        self.transcript.append(body)
        self.transcript.verticalScrollBar().setValue(
            self.transcript.verticalScrollBar().maximum())

    def _say_system(self, body: str) -> None:
        self._append(f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
                     f"margin:2px 0;'>{body}</div>")

    def closeEvent(self, event):      # noqa: N802  (Qt's spelling)
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(5000)
        super().closeEvent(event)
