"""The assistant: an open-weight model that can only quote this window.

A language model near clinical data is a liability unless the data decides what
it is allowed to say. `onset_agent` is this project's answer to that, and this
panel is its face: the model chooses which read-only queries to run, every
number it states is checked against what those queries returned, and an answer
that cites something never retrieved is discarded rather than shown. Asking it
which channels to resect gets a refusal, by design and before the model is even
called.

Two things make this worth having in the reviewer rather than only on a web
page:

* **It answers about the window on screen.** The agent reads a saved pipeline
  result, so this panel builds one from `ReviewRequest.pipeline_config()` --
  the same band, the same detectors, the same thresholds. An assistant quoting
  rates from a differently-configured run would be worse than none, because the
  numbers would look like the ones in the table and not be them.
* **Citations are clickable.** Every evidence id names a channel and a time, so
  an answer leads to the signal it was measured on, which is the only way a
  reviewer can check it. That is the whole claim of the design: the assistant
  does not ask to be believed.

The default backend runs no model at all. `ollama pull qwen2.5:7b-instruct` and
pick Ollama for the real thing; nothing here downloads a model on its own.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

from qtpy.QtCore import QEventLoop, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["AssistantPanel", "BACKENDS", "SUGGESTIONS", "parse_evidence_id"]

#: The backends offered, in the order a reviewer should try them. Labels say
#: what each one *is* rather than naming a library, because "scripted" means
#: nothing to a clinician and "no model at all" means exactly the right thing.
BACKENDS = [
    ("No model (deterministic)", "scripted"),
    ("Qwen2.5 via Ollama (local)", "ollama"),
    ("OpenAI-compatible server", "openai_compat"),
]

#: Offered as buttons. The last one is there on purpose: a reviewer should see
#: the refusal early, from the interface, rather than discover the boundary by
#: trusting an answer that was never checked.
#: (question, button label). The labels are short because four full questions
#: across a docked column elide into illegibility; the question itself is the
#: tooltip and is what gets asked.
SUGGESTIONS = [
    ("Which channels have the highest ripple rate?", "Highest rates"),
    ("Show me the evidence for the busiest channel", "Show evidence"),
    ("Does any channel actually stand out?", "Anything stand out?"),
    ("Which channels should I resect?", "Ask it to overstep"),
]


def parse_evidence_id(evidence_id: str) -> tuple[str, float] | None:
    """`subject|channel|detector|start` -> (channel, start), or None.

    Parsed rather than looked up so a citation stays clickable even if the
    store it came from has been rebuilt underneath the panel.
    """
    parts = str(evidence_id).split("|")
    if len(parts) < 4:
        return None
    try:
        return parts[1], float(parts[3])
    except (TypeError, ValueError):
        return None


class _StoreWorker(QThread):
    """Builds the saved analysis the agent reads, off the GUI thread.

    This is the project's own `run_pipeline` over the recording already in
    memory, configured from the request, so what it writes is the analysis on
    screen. It takes a few seconds, which is a few seconds the window must not
    spend frozen.
    """

    def __init__(self, session, directory: Path, parent=None):
        super().__init__(parent)
        self._session = session
        self._directory = directory
        self.store = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            from onset_hfo.pipeline import run_pipeline
            from onset_hfo.store import ResultStore

            request = self._session.request
            result = run_pipeline(
                self._session.recording, config=request.pipeline_config(),
                detectors=tuple(request.detectors),
                with_spikes=request.with_spikes,
                save_to=self._directory, verbose=False)
            self.store = ResultStore(
                Path(self._directory) / _result_name(self._session.recording))
            del result
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


def _result_name(recording) -> str:
    """Where `PipelineResult.save` puts its directory, by its own rule."""
    return (f"{getattr(recording, 'subject', 'sub')}_"
            f"{getattr(recording, 'task', 'task')}_"
            f"run-{getattr(recording, 'run', '01')}")


class _AskWorker(QThread):
    """One question, off the GUI thread: a local model can take a while."""

    def __init__(self, store, question: str, kind: str, model: str,
                 base_url: str, parent=None):
        super().__init__(parent)
        self._store = store
        self._question = question
        self._kind, self._model, self._base_url = kind, model, base_url
        self.answer = None
        self.error: str | None = None

    def run(self) -> None:
        try:
            from onset_agent.agent import OnsetAgent
            from onset_agent.backends import make_backend

            backend = make_backend(self._kind, model=self._model or None,
                                   base_url=self._base_url or None)
            self.answer = OnsetAgent(self._store, backend=backend).ask(self._question)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


def _run(worker) -> None:
    """Run a worker thread while the window keeps painting.

    A nested event loop rather than `start(); wait()`. `wait()` blocks the GUI
    thread, so the window stops repainting entirely: the "Thinking…" line never
    appears, the panel does not visibly disable, and with a local model behind
    it that is half a minute of an application the desktop reports as not
    responding. The loop is quit by the thread's own `finished`, which is
    delivered to this thread because the worker object lives on it.
    """
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.start()
    loop.exec_() if hasattr(loop, "exec_") else loop.exec()
    worker.wait()



class AssistantPanel(QWidget):
    """Ask about this window; get an answer that cites it, or a refusal."""

    #: (channel, time in archive seconds) when a citation is clicked.
    evidencePicked = Signal(str, float)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self._store = None
        self._worker = None
        self._builder = None
        # Owned by the panel and removed with it: the saved analysis is a
        # derived artifact of one review, not something to leave behind in the
        # project's results directory.
        self._tmp = tempfile.TemporaryDirectory(prefix="onset-review-")

        self.backend = QComboBox()
        for label, kind in BACKENDS:
            self.backend.addItem(label, kind)
        self.model = QLineEdit()
        self.model.setPlaceholderText("qwen2.5:7b-instruct")
        self.model.setMaximumWidth(190)
        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("http://localhost:11434/v1")
        self.base_url.setMaximumWidth(210)

        top = QHBoxLayout()
        top.addWidget(QLabel("Model"))
        top.addWidget(self.backend)
        top.addWidget(self.model)
        top.addWidget(self.base_url)
        top.addStretch(1)

        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.setOpenExternalLinks(False)
        self.transcript.anchorClicked.connect(self._citation_clicked)

        self.question = QLineEdit()
        self.question.setPlaceholderText(
            "Ask about this window — a channel, a rate, what was rejected…")
        self.question.returnPressed.connect(self.ask)
        self.send = QPushButton("Ask")
        self.send.clicked.connect(self.ask)

        row = QHBoxLayout()
        row.addWidget(self.question)
        row.addWidget(self.send)

        prompts = QHBoxLayout()
        prompts.setSpacing(3)
        for text, short in SUGGESTIONS:
            button = QPushButton(short)
            button.setToolTip(text)
            button.setStyleSheet("font-size:11px;padding:2px 8px;")
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            button.clicked.connect(lambda _=False, q=text: self.ask(q))
            prompts.addWidget(button)
        prompts.addStretch(1)

        # The model row, the transcript and the suggestions go in a scroll
        # area; the question box stays outside it, so it is never the thing
        # that scrolls out of view. Four stacked rows gave this panel a 160 px
        # minimum height, and because it shares a tab stack with the other
        # reference panels it set the floor for the whole right-hand column --
        # and so for how short the window could be made. See `theme.scrolled`.
        conversation = QWidget()
        box = QVBoxLayout(conversation)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addLayout(top)
        box.addWidget(self.transcript)
        box.addLayout(prompts)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(theme.scrolled(conversation))
        outer.addLayout(row)

        self.backend.currentIndexChanged.connect(self._backend_changed)
        self._backend_changed()
        self._say_system(
            "Ask about <b>this window</b>. Every answer is checked against the "
            "analysis on screen: a number no query returned, or a citation for "
            "evidence never retrieved, is discarded and you get a refusal "
            "instead. Citations are links — click one to take the trace there."
            "<br><br>Pick a model above. <i>No model</i> runs the whole loop "
            "deterministically, with nothing generative in it, which is how the "
            "guards are tested. For Qwen, run "
            "<code>ollama pull qwen2.5:7b-instruct</code> and choose Ollama.")

    # -- plumbing ----------------------------------------------------------
    def _backend_changed(self) -> None:
        kind = str(self.backend.currentData() or "scripted")
        self.model.setVisible(kind != "scripted")
        self.base_url.setVisible(kind == "openai_compat")

    def _ensure_store(self) -> bool:
        """Build the saved analysis the agent reads, once, on first use.

        Deferred rather than built with the session: most reviews never open
        this panel, and a few seconds of pipeline is not worth charging every
        window for.
        """
        if self._store is not None:
            return True
        if getattr(self._session, "recording", None) is None:
            self._say_system("This session has no recording attached, so there "
                             "is nothing for the assistant to read.")
            return False
        self._say_system("Preparing the evidence for this window…")
        self.setEnabled(False)
        # Kept on the panel for the duration: a worker that is only a local
        # name can be collected mid-run, and a QThread collected while running
        # takes the process with it.
        self._builder = _StoreWorker(self._session, Path(self._tmp.name))
        try:
            _run(self._builder)
        finally:
            self.setEnabled(True)
        if self._builder.error:
            self._say_system(f"Could not prepare the evidence: "
                             f"{html.escape(self._builder.error)}")
            return False
        self._store = self._builder.store
        return True

    # -- asking ------------------------------------------------------------
    def ask(self, question: str | None = None) -> None:
        text = (question if isinstance(question, str) and question
                else self.question.text()).strip()
        if not text:
            return
        self.question.clear()
        self._say_user(text)
        if not self._ensure_store():
            return

        self._say_system("Thinking…")
        self.setEnabled(False)
        self._worker = _AskWorker(self._store, text,
                                  str(self.backend.currentData() or "scripted"),
                                  self.model.text().strip(),
                                  self.base_url.text().strip())
        try:
            _run(self._worker)
        finally:
            self.setEnabled(True)
            self.question.setFocus()

        if self._worker.error:
            self._say_system(
                f"The model could not be reached: {html.escape(self._worker.error)}"
                "<br>Switch to <i>No model</i> to use the deterministic backend.")
            return
        self._say_answer(self._worker.answer)

    # -- transcript --------------------------------------------------------
    def _append(self, body: str) -> None:
        self.transcript.append(body)
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _say_system(self, body: str) -> None:
        self._append(f"<div style='color:{theme.current().text_muted};font-size:11px;"
                     f"margin:6px 0;'>{body}</div>")

    def _say_user(self, text: str) -> None:
        self._append(f"<div style='margin:10px 0 2px;'><b>{html.escape(text)}</b>"
                     f"</div>")

    def _say_answer(self, answer) -> None:
        if answer is None:
            self._say_system("No answer came back.")
            return
        if answer.refused:
            self._append(
                f"<div style='margin:2px 0 6px;padding:6px;"
                f"background:{theme.current().bad_surface};"
                f"border-left:3px solid {theme.current().bad};'>"
                f"<b>Refused.</b> {html.escape(answer.text)}"
                f"<div style='color:{theme.current().text_muted};font-size:11px;margin-top:3px;'>"
                f"{html.escape(answer.reason)}</div></div>")
            return

        parts = [f"<div style='margin:2px 0 4px;'>{html.escape(answer.text)}</div>"]
        if answer.evidence_ids:
            links = []
            for evidence_id in answer.evidence_ids:
                parsed = parse_evidence_id(evidence_id)
                label = (f"{parsed[0]} @ {parsed[1]:.2f} s" if parsed
                         else evidence_id)
                links.append(
                    f"<a href='evidence:{html.escape(evidence_id)}'>"
                    f"{html.escape(label)}</a>")
            parts.append("<div style='font-size:11px;margin-bottom:2px;'>"
                         "cites: " + " · ".join(links) + "</div>")
        tools = ", ".join(answer.tools_called) or "none"
        flag = "" if answer.verified else " · <b>unverified</b>"
        parts.append(f"<div style='color:#666;font-size:11px;'>"
                     f"{html.escape(answer.backend)} · tools: "
                     f"{html.escape(tools)}{flag}</div>")
        self._append("".join(parts))

    def _citation_clicked(self, url) -> None:
        """Turn a clicked citation into a place on the trace.

        Unquoted first: an evidence id is `subject|channel|detector|start`, and
        Qt percent-encodes the separator on the way into a `QUrl`, so the raw
        string comes back as `sub-01%7CAR1-AR2%7C...` and splits into one
        piece. The symptom is a citation that silently does nothing when
        clicked, which is the one failure this panel cannot afford -- the
        clickable citation *is* the argument that the answer can be checked.
        """
        from urllib.parse import unquote

        text = unquote(url.toString())
        if not text.startswith("evidence:"):
            return
        parsed = parse_evidence_id(text[len("evidence:"):])
        if parsed is not None:
            self.evidencePicked.emit(parsed[0], parsed[1])

    def closeEvent(self, event):      # noqa: N802  (Qt's spelling)
        """Tidy up early when the panel is actually closed.

        Not the only cleanup, and not relied on: closing a *dock* hides its
        widget rather than closing it, so this often never fires.
        `TemporaryDirectory` removes itself when the panel is collected and at
        interpreter exit, which is what actually guarantees it; this just
        releases the space sooner when the chance comes.
        """
        try:
            self._tmp.cleanup()
        except Exception:
            pass
        super().closeEvent(event)
