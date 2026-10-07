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

The backend the panel opens on is decided by `assistant_config.load_defaults`:
what the installer recorded, failing that a live Ollama on this machine with a
catalogue model pulled, failing that no model at all. Nothing here downloads a
model or starts a server; `packaging/install-ubuntu.sh --with-assistant` does.
"""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

from qtpy.QtCore import QEventLoop, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
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

__all__ = ["AssistantPanel", "BACKENDS", "SUGGESTIONS", "parse_evidence_id",
           "explain_refusal", "describe_step"]

#: The transcript's type sizes, in points. The application font is 10 pt; a
#: transcript is read, not scanned, and was asked to be larger.
TEXT_PT = 11
SMALL_PT = 10

#: The most a served model may write per call. An answer is two or three
#: sentences; a tool call is a line. At the five tokens a second a CPU
#: manages, 512 was nearly two minutes of a model that had lost the thread.
ANSWER_TOKENS = 320
#: How long one model call may take before the panel gives up on it. The
#: library's 180 s was hit by a 7B on a CPU reading its prompt; the person
#: has a Stop button and a clock, so the limit is for a server that has
#: really gone away, not for a slow one.
ANSWER_TIMEOUT = 900.0

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
    ("What can you do?", "What can you do?"),
    ("What is an HFO?", "What is an HFO?"),
    ("Which channels have the highest ripple rate?", "Highest rates"),
    ("Show me the evidence for the busiest channel", "Show evidence"),
    ("Does any channel actually stand out?", "Anything stand out?"),
    ("Where on the head is the activity, and on how many electrodes?", "Where is it?"),
    ("Why does the selected event read as real, or not?", "Explain this event"),
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
            saved = Path(self._directory) / _result_name(self._session.recording)
            # Where each channel sits, for the map query. Written beside the
            # analysis so the assistant reads it like any other table; no
            # resection in it (see onset_review.contactmap).
            try:
                from onset_review.contactmap import contacts_table

                contacts = contacts_table(
                    self._session, resection=getattr(self._session, "resection", None),
                    electrodes=getattr(self._session, "electrodes", None))
                if not contacts.empty:
                    contacts.to_csv(saved / "contacts.csv", index=False)
            except Exception:       # noqa: BLE001 - a map is a convenience, not the analysis
                pass
            self.store = ResultStore(saved)
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

    #: Every trace entry as the agent makes it: a query and what it returned,
    #: what the model wrote, a check. The panel shows them while it waits.
    progress = Signal(dict)

    def __init__(self, store, question: str, kind: str, model: str,
                 base_url: str, parent=None, extra_tools=None, history=None,
                 extra_briefing=None):
        super().__init__(parent)
        self._store = store
        self._question = question
        self._kind, self._model, self._base_url = kind, model, base_url
        self._extra_tools = dict(extra_tools or {})
        self._history = list(history or [])
        self._extra_briefing = list(extra_briefing or [])
        self._backend = None
        self._stop = False
        self.answer = None
        self.error: str | None = None

    def stop(self) -> None:
        """Abandon the question: the loop ends at its next step, and a
        served model's connection is closed so it stops generating."""
        self._stop = True
        backend = self._backend
        if backend is not None:
            backend.abort()

    def run(self) -> None:
        try:
            from onset_agent.agent import OnsetAgent
            from onset_agent.backends import make_backend

            served = self._kind in ("ollama", "openai_compat")
            self._backend = make_backend(self._kind, model=self._model or None,
                                         base_url=self._base_url or None,
                                         **({"max_tokens": ANSWER_TOKENS,
                                             "timeout": ANSWER_TIMEOUT} if served else {}))
            if self._stop:
                self._backend.abort()
            self.answer = OnsetAgent(self._store, backend=self._backend,
                                     extra_tools=self._extra_tools).ask(
                self._question, should_stop=lambda: self._stop,
                on_event=self.progress.emit, history=self._history,
                extra_briefing=self._extra_briefing)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"


def describe_step(entry: dict) -> str | None:
    """One trace entry as a line a reviewer can read while the loop runs: which
    data the model was given, what it asked for, what it wrote, and what the
    checks made of it. HTML. None for entries that say nothing on their own."""
    kind = entry.get("type")
    if kind == "tool_call":
        call = html.escape(str(entry.get("tool") or ""))
        args = entry.get("arguments") or {}
        if args:
            call += "(" + html.escape(", ".join(f"{k}={v}" for k, v in args.items())) + ")"
        if not entry.get("ok", True):
            return f"A query failed: <code>{call}</code> — {html.escape(str(entry.get('error') or ''))}"
        who = "Retrieved for the model" if entry.get("briefing") else "The model asked for"
        if entry.get("analysis"):
            who = "<b>Ran an analysis</b> for the model" if not entry.get("briefing") \
                else "<b>Ran</b> for the model"
        took = f" ({entry['seconds']:.1f} s)" if entry.get("analysis") and entry.get("seconds") else ""
        return f"{who} <code>{call}</code>{took}: {html.escape(str(entry.get('digest') or ''))}"
    if kind == "model":
        # A request for tools is said by the tool lines that follow; a
        # well-formed answer is shown as the answer. A failed one is quoted
        # by the refusal, with the check it failed.
        return None
    if kind == "prose":
        return ("The model answered in a sentence rather than the JSON form; "
                "checking the sentence as written.")
    if kind == "format_error":
        return "The model's reply was empty; asked again."
    if kind == "citations":
        parts = []
        if entry.get("attached"):
            parts.append(f"Attached {len(entry['attached'])} citation(s) from the retrieved "
                         "evidence for the channels the answer names.")
        if entry.get("dropped"):
            dropped = ", ".join(html.escape(str(i)) for i in entry["dropped"][:3])
            parts.append(f"Dropped {len(entry['dropped'])} citation(s) the model made up: "
                         f"<i>{dropped}</i>.")
        return " ".join(parts) or None
    if kind == "answer":
        if entry.get("verified"):
            n = len(entry.get("evidence_ids") or [])
            return ("Checks passed: every number traced to a query result"
                    + (f"; {n} citation(s) resolved." if n else "."))
        problems = "; ".join(html.escape(str(p)) for p in entry.get("problems") or [])
        return f"Check failed: {problems or 'the checks did not pass'}. Asking the model again."
    if kind == "model_refusal":
        return "The model declined to answer."
    if kind == "scope_check":
        return "Refused before any model ran: out of scope."
    if kind == "stopped":
        return "Stopped."
    if kind == "gave_up":
        return (f"Gave up after {entry.get('steps')} step(s) and "
                f"{entry.get('retries')} retry/retries.")
    return None


def explain_refusal(answer) -> list[str]:
    """What the model did and why it was refused, from the trace, as HTML lines.

    A refusal that says only "verification failed" leaves a reviewer unable to
    tell a model that invented a number from one that copied it with a
    different rounding. The trace knows; this reads it out.
    """
    lines: list[str] = []
    for entry in getattr(answer, "trace", None) or []:
        kind = entry.get("type")
        if kind == "answer" and not entry.get("verified", True):
            said = html.escape(str(entry.get("text") or "")[:300])
            problems = "; ".join(html.escape(str(p)) for p in entry.get("problems") or [])
            lines.append(f"The model wrote: <i>{said}</i> — refused because {problems or 'the checks failed'}.")
        elif kind == "format_error":
            said = html.escape(str(entry.get("content") or "")[:200])
            lines.append(f"The model did not reply in the required JSON shape; it wrote: <i>{said}</i>")
        elif kind == "tool_call" and not entry.get("ok", True):
            lines.append(f"A tool call failed: {html.escape(str(entry.get('error') or ''))}")
        elif kind == "citations" and entry.get("dropped"):
            dropped = ", ".join(html.escape(str(i)) for i in entry["dropped"][:3])
            lines.append(f"Dropped {len(entry['dropped'])} citation(s) the model made up: "
                         f"<i>{dropped}</i>")
        elif kind == "model_refusal":
            lines.append("The model itself declined to answer"
                         + (f": <i>{html.escape(str(entry.get('text') or '')[:300])}</i>"
                            if entry.get("text") else "."))
        elif kind == "stopped":
            lines.append("Stopped by you.")
        elif kind == "gave_up":
            lines.append(f"Gave up after {entry.get('steps')} steps and "
                         f"{entry.get('retries')} retries.")
    return lines


def _paragraphs(text: str) -> str:
    """Plain text with line breaks, as HTML. The capabilities answer is a
    list; the rest is a sentence or three."""
    return html.escape(text).replace("\n", "<br>")


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



#: What the assistant is asked when the Report page asks for a draft. The
#: briefing already carries the leaders, their evidence and the detectors'
#: disagreements, so the question names the shape of the paragraph and
#: nothing the model could not have been given.
DRAFT_QUESTION = ("Draft the findings paragraph of this window's report in three to six "
                  "sentences: which channels led and at what rate, whether they are tied "
                  "with others, whether the two detectors agreed, and what limits the "
                  "finding. State only numbers from the evidence, as measurements.")


class AssistantPanel(QWidget):
    """Ask about this window; get an answer that cites it, or a refusal."""

    #: A findings paragraph the model drafted, with who drafted it, for the
    #: Report page to take up. Emitted only for a verified, unrefused answer.
    drafted = Signal(str, str)

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
        # Analyses by consent: each run costs seconds on a CPU and is announced
        # with its cost in the lines under the question. Off until ticked.
        self.analyses = QCheckBox("Let it run analyses on this window")
        self.analyses.setObjectName("onset_assistant_analyses")
        self.analyses.setToolTip(
            "Re-run a detector at another threshold, take a contact's spectral "
            "power, compare the detectors, time which channel leads. Each run is "
            "announced with what it costs, works on a copy of the recording, and "
            "changes nothing on screen.")
        self.new_chat = QPushButton("New conversation")
        self.new_chat.setObjectName("onset_assistant_new")
        self.new_chat.setToolTip("Forget the questions so far. Until then, a question "
                                 "can refer back: \u201cand the second one?\u201d")
        self.new_chat.clicked.connect(self.new_conversation)
        self._history: list[tuple[str, str]] = []
        self._drafting = False
        self._sources: list = []
        self._event_key: str = ""

        top = QHBoxLayout()
        top.addWidget(QLabel("Model"))
        top.addWidget(self.backend)
        top.addWidget(self.model)
        top.addWidget(self.base_url)
        top.addWidget(self.analyses)
        top.addStretch(1)
        top.addWidget(self.new_chat)

        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.setOpenExternalLinks(False)
        self.transcript.anchorClicked.connect(self._citation_clicked)
        font = self.transcript.font()
        font.setPointSizeF(TEXT_PT)
        self.transcript.setFont(font)
        self.transcript.document().setDefaultFont(font)

        self.question = QLineEdit()
        self.question.setPlaceholderText(
            "Ask about this window — a channel, a rate, what was rejected…")
        self.question.returnPressed.connect(self.ask)
        self.send = QPushButton("Ask")
        self.send.clicked.connect(self.ask)
        # A model on a CPU can take a minute per step, and a loop is up to
        # six of them. The reviewer must be able to end that, and to see it
        # is still alive meanwhile.
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("onset_assistant_stop")
        self.stop_button.setToolTip("Abandon this question: the model's request is "
                                    "closed and the loop ends at its next step")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.busy = QLabel("")
        self.busy.setObjectName("onset_assistant_busy")
        self.busy.setStyleSheet(f"color:{theme.current().text_muted};font-size:{SMALL_PT}pt;")
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self._started = 0.0

        row = QHBoxLayout()
        row.addWidget(self.question)
        row.addWidget(self.send)
        row.addWidget(self.stop_button)
        row.addWidget(self.busy)

        prompts = QHBoxLayout()
        prompts.setSpacing(3)
        for text, short in SUGGESTIONS:
            button = QPushButton(short)
            button.setToolTip(text)
            button.setStyleSheet(f"font-size:{SMALL_PT}pt;padding:2px 8px;")
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
        # Getting a model onto this machine, from here rather than from a
        # terminal. The probe it runs is deferred to the first time the panel
        # is shown: most windows never open this panel, and a torch import
        # on a worker thread is still work nobody asked for.
        from onset_review.modelsetup import ModelSetupBox

        self.setup = ModelSetupBox(parent=self)
        self.setup.modelReady.connect(self._use_served)
        self._looked = False
        box.addWidget(self.setup)
        box.addWidget(self.transcript)
        box.addLayout(prompts)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        outer.addWidget(theme.scrolled(conversation))
        outer.addLayout(row)

        self.backend.currentIndexChanged.connect(self._backend_changed)
        self._apply_defaults()
        self._backend_changed()
        self._say_system(
            "Ask about <b>this window</b>. Every answer is checked against the "
            "analysis on screen: a number no query returned is refused rather "
            "than shown. Citations are links — click one to take the trace "
            "there. While a question runs, the lines below it say which data "
            "the model was given, what it asked for, what it wrote, and what "
            "the checks made of it. Ask <i>What can you do?</i> to start."
            "<br><br>Pick a model above. <i>No model</i> runs the whole loop "
            "deterministically, with nothing generative in it, which is how the "
            "guards are tested. For a Qwen on this machine, use the box above: "
            "it lists the sizes that fit, opens on the one that answers in "
            "reasonable time on this hardware, and downloads it.")

    # -- plumbing ----------------------------------------------------------
    def showEvent(self, event):      # noqa: N802  (Qt's spelling)
        super().showEvent(event)
        if self._looked:
            return
        self._looked = True
        if os.environ.get("ONSET_ASSISTANT_NO_PROBE"):
            self.setup.idle()
        else:
            self.setup.look()

    def _use_served(self, tag: str, base_url: str) -> None:
        """Switch this panel to the model the setup box just made ready."""
        index = self.backend.findData("ollama")
        if index >= 0:
            self.backend.setCurrentIndex(index)
        self.model.setText(tag)
        self._say_system(f"Using <b>{html.escape(tag)}</b> on Ollama at "
                         f"{html.escape(base_url)}.")

    def _apply_defaults(self) -> None:
        """Open on whatever `assistant_config` decided, and say where it came
        from -- a preselected model with no explanation looks like a guess."""
        from onset_review.assistant_config import load_defaults

        chosen = load_defaults()
        index = self.backend.findData(chosen.kind)
        if index >= 0:
            self.backend.setCurrentIndex(index)
        if chosen.model:
            self.model.setText(chosen.model)
        if chosen.base_url and chosen.kind == "openai_compat":
            self.base_url.setText(chosen.base_url)
        self._defaults = chosen

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
    def draft_findings(self) -> bool:
        """Ask for the findings paragraph and hand it on as a draft.

        The same loop, the same guards: a draft with a number no query
        returned is refused like any answer, and the refusal is shown here.
        True when a draft was produced.
        """
        self._drafting = True
        try:
            self.ask(DRAFT_QUESTION)
        finally:
            self._drafting = False
        answer = getattr(self._worker, "answer", None) if self._worker is not None else None
        if answer is None or answer.refused or not str(answer.text).strip():
            return False
        self.drafted.emit(str(answer.text).strip(), f"assistant ({answer.backend})")
        return True

    def ask(self, question: str | None = None) -> None:
        text = (question if isinstance(question, str) and question
                else self.question.text()).strip()
        if not text:
            return
        self.question.clear()
        self._say_user(text)
        if self._about(text):
            return
        if not self._ensure_store():
            return

        self._say_system("Thinking…")
        self._worker = _AskWorker(self._store, text,
                                  str(self.backend.currentData() or "scripted"),
                                  self.model.text().strip(),
                                  self.base_url.text().strip(),
                                  extra_tools=self.extra_tools(),
                                  history=list(self._history),
                                  extra_briefing=self.extra_briefing(text))
        self._worker.progress.connect(self._progress)
        self._set_busy(True)
        try:
            _run(self._worker)
        finally:
            self._set_busy(False)
            self.question.setFocus()

        if self._worker.error:
            self._say_system(
                f"The model could not be reached: {html.escape(self._worker.error)}"
                "<br>Switch to <i>No model</i> to use the deterministic backend.")
            return
        self._say_answer(self._worker.answer)
        answer = self._worker.answer
        if answer is not None and not answer.refused and answer.verified:
            self._history.append((text, str(answer.text)))
            self._history = self._history[-6:]

    # -- what the window adds to the tool set --------------------------------
    def extra_tools(self) -> dict:
        from onset_review.assistant_tools import (
            analysis_tools,
            explain_tools,
            sensitivity_tools,
            window_tools,
        )

        tools = explain_tools(self._session)
        tools.update(sensitivity_tools(self._session,
                                       allow_run=self.analyses.isChecked()))
        tools.update(window_tools(self._session, allow_run=self.analyses.isChecked()))
        if self.analyses.isChecked():
            tools.update(analysis_tools(self._session))
        return tools

    _EVENT_WORDS = ("this event", "selected event", "explain", "real", "ringing",
                    "artifact", "artefact", "why was", "why is")

    _THRESHOLD_WORDS = ("threshold", "stricter", "survive", "robust", "how sure")
    _WINDOW_WORDS = ("other window", "windows", "minute", "earlier", "later", "between",
                     "another", "the first", "the second", "same in", "change between")

    def extra_briefing(self, question: str) -> list:
        """What the window already knows that the question is about: the
        selected event's reading, the threshold re-test when it has run."""
        lowered = question.lower()
        out: list = []
        if (getattr(self._session, "sensitivity", None) is not None
                and any(word in lowered for word in self._THRESHOLD_WORDS)):
            out.append(("threshold_sensitivity", {}))
        if any(word in lowered for word in self._WINDOW_WORDS):
            from onset_review import windows

            if windows.other_windows(self._session):
                out.append(("other_windows", {}))
        if self._event_key and any(word in lowered for word in self._EVENT_WORDS):
            parts = str(self._event_key).split("|")
            if len(parts) >= 2:
                try:
                    out.append(("explain_event",
                                {"channel": parts[0], "start": float(parts[1])}))
                except ValueError:
                    pass
        return out

    def note_event(self, key: str) -> None:
        """The event list's selection, so "this event" means something."""
        self._event_key = str(key or "")

    def new_conversation(self) -> None:
        self._history = []
        self._say_system("New conversation: earlier questions are forgotten.")

    def _about(self, text: str) -> bool:
        """"What can you do?" is answered here, at once, from a fixed text: it
        needs no evidence built and no model, and a reviewer asking it is
        often waiting to find out whether the thing works at all."""
        from onset_agent import guard
        from onset_agent.agent import AgentAnswer
        from onset_agent.prompts import what_i_can_do

        if not guard.about_the_assistant(text):
            return False
        recording = getattr(self._session, "recording", None)
        subject = getattr(recording, "subject", None) or "this window"
        self._say_answer(AgentAnswer(question=text, text=what_i_can_do(str(subject)),
                                     backend="no model needed",
                                     trace=[{"type": "about"}]))
        return True

    def _progress(self, entry: dict) -> None:
        """A trace entry, as it happens: shown under the question in small
        type, so the wait is a visible process rather than a spinner."""
        line = describe_step(entry)
        if line:
            self._say_step(line)

    def stop(self) -> None:
        """The Stop button: end the question in flight, if there is one."""
        worker = self._worker
        if worker is None or not worker.isRunning():
            return
        worker.stop()
        self.stop_button.setEnabled(False)
        self.busy.setText("stopping…")

    def _set_busy(self, on: bool) -> None:
        """While a question runs: only Stop works, and the clock shows."""
        import time

        for widget in (self.question, self.send, self.backend, self.model,
                       self.base_url, self.setup, self.analyses, self.new_chat):
            widget.setEnabled(not on)
        self.stop_button.setEnabled(on)
        if on:
            self._started = time.monotonic()
            self.busy.setText("thinking… 0 s")
            self._clock.start()
        else:
            self._clock.stop()
            self.busy.setText("")

    def _tick(self) -> None:
        import time

        if self.stop_button.isEnabled():
            self.busy.setText(f"thinking… {time.monotonic() - self._started:.0f} s")

    # -- transcript --------------------------------------------------------
    def _append(self, body: str) -> None:
        self.transcript.append(body)
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _say_system(self, body: str) -> None:
        self._append(f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
                     f"margin:6px 0;'>{body}</div>")

    def _say_step(self, body: str) -> None:
        self._append(f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
                     f"margin:1px 0 1px 14px;'>· {body}</div>")

    def _say_user(self, text: str) -> None:
        self._append(f"<div style='margin:10px 0 2px;'><b>{html.escape(text)}</b>"
                     f"</div>")

    def _say_answer(self, answer) -> None:
        if answer is None:
            self._say_system("No answer came back.")
            return
        if answer.refused:
            why = "".join(f"<li>{line}</li>" for line in explain_refusal(answer))
            self._append(
                f"<div style='margin:2px 0 6px;padding:6px;"
                f"background:{theme.current().bad_surface};"
                f"border-left:3px solid {theme.current().bad};'>"
                f"<b>Refused.</b> {_paragraphs(answer.text)}"
                f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
                f"margin-top:3px;'>{html.escape(answer.reason)}</div>"
                + (f"<ul style='font-size:{SMALL_PT}pt;margin:4px 0 0 0;'>{why}</ul>" if why else "")
                + "</div>")
            return

        mode = getattr(answer, "mode", "data")
        if mode == "general":
            self._append(
                f"<div style='margin:2px 0 6px;padding:6px;"
                f"background:{theme.current().warn_surface};"
                f"border-left:3px solid {theme.current().warn};'>"
                f"<b>Not checked.</b> This is the model's own knowledge, not this analysis "
                f"and not the project's documents; nothing here vouches for it."
                f"<div style='margin-top:4px;'>{_paragraphs(answer.text)}</div>"
                f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;"
                f"margin-top:3px;'>{html.escape(answer.backend)} · unchecked</div></div>")
            return
        parts = [f"<div style='margin:2px 0 4px;'>{_paragraphs(answer.text)}</div>"]
        if mode == "background":
            self._sources = list(getattr(answer, "sources", []) or [])
            links = [f"<a href='doc:{i}'>{html.escape(label)}</a>"
                     for i, (label, _text) in enumerate(self._sources)]
            parts.append(f"<div style='font-size:{SMALL_PT}pt;margin-bottom:2px;'>"
                         "from the documents: " + " · ".join(links) + "</div>")
            parts.append(f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;'>"
                         f"{html.escape(answer.backend)} · numbers checked against the cited "
                         "sections</div>")
            self._append("".join(parts))
            return
        if answer.evidence_ids:
            links = []
            for evidence_id in answer.evidence_ids:
                parsed = parse_evidence_id(evidence_id)
                label = (f"{parsed[0]} @ {parsed[1]:.2f} s" if parsed
                         else evidence_id)
                links.append(
                    f"<a href='evidence:{html.escape(evidence_id)}'>"
                    f"{html.escape(label)}</a>")
            attached = getattr(answer, "attached_citations", None) or []
            how = (" (attached from the retrieved evidence for the channels named)"
                   if attached and set(attached) >= set(answer.evidence_ids) else "")
            parts.append(f"<div style='font-size:{SMALL_PT}pt;margin-bottom:2px;'>"
                         "cites: " + " · ".join(links) + how + "</div>")
        looked = len(getattr(answer, "looked_at", None) or [])
        tools = ", ".join(answer.tools_called) or "none"
        flag = "" if answer.verified else " · <b>unverified</b>"
        parts.append(f"<div style='color:{theme.current().text_muted};font-size:{SMALL_PT}pt;'>"
                     f"{html.escape(answer.backend)} · {looked} quer{'y' if looked == 1 else 'ies'} "
                     f"run · the model asked for: {html.escape(tools)}{flag}</div>")
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
        if text.startswith("doc:"):
            self._show_source(text[len("doc:"):])
            return
        if not text.startswith("evidence:"):
            return
        parsed = parse_evidence_id(text[len("evidence:"):])
        if parsed is not None:
            self.evidencePicked.emit(parsed[0], parsed[1])

    def _show_source(self, index: str) -> None:
        """The cited document section, in a window of its own."""
        from qtpy.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

        from onset_review.studypages import set_markdown

        try:
            label, body = self._sources[int(index)]
        except (ValueError, IndexError):
            return
        dialog = QDialog(self)
        dialog.setObjectName("onset_source_dialog")
        dialog.setWindowTitle(label)
        dialog.resize(640, 420)
        box = QVBoxLayout(dialog)
        view = QTextBrowser()
        set_markdown(view, f"### {label}\n\n{body}")
        box.addWidget(view)
        self._source_dialog = dialog
        dialog.show()

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
