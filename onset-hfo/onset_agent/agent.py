"""The agent loop.

    question
      -> about the assistant itself?            answered from a fixed text, no model
      -> scope check (no model involved)        guard.check_question
      -> briefing: the standard queries run     tools.dispatch, before the model
      -> model answers, or asks for more        backend.chat
      -> tool runs against the saved results    tools.dispatch
      -> ... up to `max_steps` times ...
      -> model returns JSON                     {"answer", "evidence_ids"} or {"refusal"}
         (or prose, which is read as the answer)
      -> citations: kept, attached or dropped   attach_citations
      -> number check                           guard.verify_answer
      -> answer, or a refusal explaining why

What makes this an *agent* rather than a chatbot with a database: the model
decides which questions to ask of the data and when it has enough, and the
loop lets it act on what it learned. What keeps it safe is that everything it
can do is a read-only query, and everything it says is checked against what
those queries returned.

**The briefing.** A 7B model on a CPU takes half a minute per call, and the
first two or three calls of every conversation were the same: what was
analysed, which channels lead, the evidence behind them. So those queries are
run before the model is asked anything and handed to it as tool results it
can read at once. The model may still call tools for what the briefing did
not cover; on most questions it does not need to, and one call answers.

**The prompt is kept small, and in a fixed order.** On a CPU the model reads
every token of the prompt before it writes one, at a rate that made a
3,000-token prompt a three-minute wait. So the first call carries no tool
schemas (884 tokens the model does not need to read the briefing), the
briefing fetches two leading channels rather than three, and the part of the
conversation that is the same for every question -- the system prompt and
the base briefing -- comes *before* the question, so a served model's prompt
cache covers it on the second question and only the question itself is new.
Question-specific retrievals (a channel named in it, the seizure comparison,
a report section) follow the question. Tool schemas are offered from the
second step on, when the model has said it needs more.

**Prose is an answer.** Small models write the answer as a sentence instead
of the JSON object they were asked for. The sentence is what a reader wanted;
it is checked exactly as a JSON answer would be, number by number. What is
never relaxed is the number check: a rate no tool returned is refused however
it was formatted.

**Citations are attached, never invented.** A model that names a channel
without citing a window, or cites a placeholder, gets the real evidence ids
that were retrieved for that channel attached to its answer, and the trace
says so. An id the model made up is dropped, never shown.

Failure is a first-class outcome. If the model states a number no tool
produced, the answer is discarded and the agent says it cannot answer. That is
a deliberate design choice: a prototype that occasionally says "I can't" is
usable; one that occasionally invents a rate is not.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from onset_agent import guard
from onset_agent.backends import Backend, ScriptedBackend, extract_json_object
from onset_agent.prompts import system_prompt, what_i_can_do
from onset_agent.tools import TOOLS, ToolError, dispatch, tool_schemas
from onset_hfo.store import ResultStore

__all__ = ["OnsetAgent", "AgentAnswer", "briefing", "briefing_base", "briefing_extras",
           "attach_citations", "digest"]

MAX_CALLS_PER_TURN = 3
#: How many leading channels the briefing fetches evidence windows for. Two:
#: each costs about 190 tokens of prompt, and the questions that need a third
#: name it, which fetches it.
BRIEFING_CHANNELS = 2


@dataclass
class AgentAnswer:
    """What the agent returns: text, citations, and the full trace."""

    question: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    trace: list[dict] = field(default_factory=list)
    backend: str = ""
    verified: bool = True

    @property
    def tools_called(self) -> list[str]:
        """The tools the model asked for, in order. The briefing is not the
        model's doing and is listed separately."""
        return [step["tool"] for step in self.trace
                if step.get("type") == "tool_call" and not step.get("briefing")]

    @property
    def looked_at(self) -> list[str]:
        """Every query that ran, the briefing included."""
        return [step["tool"] for step in self.trace if step.get("type") == "tool_call"]

    @property
    def attached_citations(self) -> list[str]:
        """Evidence ids the agent attached for the channels the model named,
        as opposed to ids the model cited itself."""
        out: list[str] = []
        for step in self.trace:
            if step.get("type") == "citations":
                out += step.get("attached") or []
        return out

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.text, "refused": self.refused,
                "reason": self.reason, "evidence_ids": self.evidence_ids,
                "tools_called": self.tools_called, "looked_at": self.looked_at,
                "backend": self.backend, "verified": self.verified, "trace": self.trace}

    def __str__(self) -> str:
        head = "REFUSED: " if self.refused else ""
        cites = ("\n  cites: " + ", ".join(self.evidence_ids)) if self.evidence_ids else ""
        tools = ("\n  tools: " + " -> ".join(self.tools_called)) if self.tools_called else ""
        return f"{head}{self.text}{cites}{tools}"


# --------------------------------------------------------------------------
# The briefing: what is fetched before the model is asked
# --------------------------------------------------------------------------

_SEIZURE = re.compile(r"seizure|ictal|before|during|change over time", re.IGNORECASE)
_SECTIONS = (
    ("limitations", re.compile(r"limitation|caveat|weakness|trust|reliab", re.IGNORECASE)),
    ("methods", re.compile(r"\bmethod|how (?:did|do|does) (?:you|it|the)|algorithm|threshold",
                           re.IGNORECASE)),
    ("data_quality", re.compile(r"quality|artifact|rejected|excluded|bad channel|noisy",
                                re.IGNORECASE)),
)


def briefing_base(store: ResultStore) -> list[tuple[str, dict]]:
    """The queries run for every question, in a fixed order: what was
    analysed, the leading channels with their intervals, the evidence windows
    of the first two, and where the detectors disagree. The same for every
    question, so a served model's prompt cache covers it."""
    calls: list[tuple[str, dict]] = [("get_recording_metadata", {}),
                                     ("top_channels", {"k": 5})]
    try:
        leading = [str(r["channel"]) for r in store.top_channels(None, BRIEFING_CHANNELS)]
    except Exception:       # noqa: BLE001 - a store with no rates still gets a briefing
        leading = []
    for channel in leading:
        calls.append(("get_evidence", {"channel": channel, "k": 2}))
    calls.append(("detector_disagreements", {}))
    return calls


def briefing_extras(store: ResultStore, question: str,
                    already: list[tuple[str, dict]] = ()) -> list[tuple[str, dict]]:
    """What the question's wording asks for beyond the base: a channel it
    names, the seizure comparison, a report section."""
    calls: list[tuple[str, dict]] = []
    known = {name.upper(): name for name in _channels(store)}
    named: list[str] = []
    for token in guard.CHANNEL_TOKEN.findall(question):
        channel = known.get(token.upper())
        if channel and channel not in named:
            named.append(channel)
    for channel in named[:2]:
        evidence = ("get_evidence", {"channel": channel, "k": 2})
        if evidence not in already:
            calls.append(evidence)
        calls.append(("channel_summary", {"channel": channel}))
    if _SEIZURE.search(question):
        calls.append(("rate_change", {}))
    for section, pattern in _SECTIONS:
        if pattern.search(question):
            calls.append(("report_section", {"section": section}))
    return calls


def briefing(store: ResultStore, question: str) -> list[tuple[str, dict]]:
    """Every query run for the model before it is asked anything: the base,
    then what the question's wording asks for."""
    base = briefing_base(store)
    return base + briefing_extras(store, question, base)


def _channels(store: ResultStore) -> list[str]:
    try:
        return list(store.channels())
    except Exception:       # noqa: BLE001
        return []


# --------------------------------------------------------------------------
# Reading the model's answer
# --------------------------------------------------------------------------

#: Leftovers a small model appends to a prose answer: a tool-call tag it did
#: not close, or a citation list written in its own words (sometimes in its
#: own language). Neither is part of the answer.
_NOISE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)|"
                    r"(?:证据|evidence)_?ids?\s*[:：].*$",
                    re.IGNORECASE | re.DOTALL)


def _as_prose(content: str) -> str:
    return _NOISE.sub("", content or "").strip()


def attach_citations(text: str, cited: list[str],
                     retrieved: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Decide the citations an answer goes out with.

    Returns ``(final, attached, dropped)``. Ids the model cited that a tool
    really returned are kept as they are. Ids it made up are dropped. When
    nothing real was cited, the retrieved evidence windows of every channel the
    answer names are attached instead, two per channel, in the order they
    were retrieved -- so a sentence about AR1-AR2 links to AR1-AR2's windows,
    and the reader can click through to the signal exactly as if the model
    had cited them.
    """
    kept = [i for i in cited if i in retrieved]
    dropped = [i for i in cited if i not in retrieved]
    if kept:
        return kept, [], dropped
    by_channel: dict[str, list[str]] = {}
    for evidence_id in retrieved:
        parts = evidence_id.split("|")
        if len(parts) >= 4:
            by_channel.setdefault(parts[1].upper(), []).append(evidence_id)
    attached: list[str] = []
    for token in guard.CHANNEL_TOKEN.findall(text):
        for evidence_id in by_channel.get(token.upper(), [])[:2]:
            if evidence_id not in attached:
                attached.append(evidence_id)
    return attached, attached, dropped


def digest(tool: str, payload) -> str:
    """One line saying what a tool result contained, for a person watching
    the loop. Every number in it is copied from the payload."""
    if not isinstance(payload, dict):
        return str(payload)[:160]
    if payload.get("error"):
        return f"error: {payload['error']}"
    try:
        if tool in ("top_channels", "list_channels"):
            rows = payload.get("channels") or []
            head = ", ".join(f"{r['channel']} {r['rate_per_min']}/min" for r in rows[:5])
            more = f" … {len(rows)} channels" if len(rows) > 5 else ""
            return f"{payload.get('detector', '')}: {head or 'no channels'}{more}"
        if tool == "get_evidence":
            rows = payload.get("evidence") or []
            if not rows:
                return f"{payload.get('channel')}: no accepted events"
            windows = "; ".join(f"{r['start']}–{r['stop']} s"
                                + (f" ({r['peak_frequency_hz']} Hz)"
                                   if r.get("peak_frequency_hz") is not None else "")
                                for r in rows[:3])
            return f"{payload.get('channel')}: {len(rows)} window(s), {windows}"
        if tool == "channel_summary":
            detectors = payload.get("detectors") or {}
            return f"{payload.get('channel')}: " + ", ".join(
                f"{name} {d.get('rate_per_min')}/min (rank {d.get('rank')})"
                for name, d in detectors.items())
        if tool == "detector_disagreements":
            rows = payload.get("disagreements") or []
            names = ", ".join(str(r.get("channel")) for r in rows[:4])
            return (f"{len(rows)} channel(s) the detectors rank very differently"
                    + (f": {names}" if names else ""))
        if tool == "get_recording_metadata":
            return (f"{payload.get('subject')} ({payload.get('source')}), "
                    f"{payload.get('duration_s')} s at {payload.get('sampling_rate_hz')} Hz, "
                    f"{payload.get('channels_analysed')} channels analysed, "
                    f"band {payload.get('band_hz')} Hz")
        if tool == "rate_change":
            if not payload.get("available"):
                return "no clinician-marked seizure in the analysed window"
            return f"{len(payload.get('rows') or [])} channel(s) before versus during"
        if tool == "report_section":
            content = payload.get("content")
            text = content if isinstance(content, str) else json.dumps(content, default=str)
            return f"{payload.get('section')}: {text[:160]}"
    except (KeyError, TypeError, AttributeError):
        pass
    return json.dumps(payload, default=str)[:160]


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class OnsetAgent:
    """An evidence-only assistant over one saved analysis."""

    def __init__(self, store: ResultStore, backend: Backend | None = None,
                 max_steps: int = 6, max_retries: int = 1, verbose: bool = False,
                 brief: bool | None = None):
        self.store = store
        self.backend = backend or ScriptedBackend()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        # The briefing saves a language model its first round-trips. The
        # scripted policy has no round-trips to save and routes by question,
        # so it keeps calling its own tools.
        self.brief = self.backend.is_language_model if brief is None else brief
        meta = store.metadata()
        self._subject, self._source = str(meta.get("subject")), str(meta.get("source"))
        self.system = self._system(tools_offered=not self.brief)

    def _system(self, tools_offered: bool) -> str:
        return system_prompt(subject=self._subject, source=self._source,
                             tool_names=list(TOOLS), briefed=self.brief,
                             tools_offered=tools_offered)

    # -- public API -------------------------------------------------------
    def ask(self, question: str, should_stop: Callable[[], bool] | None = None,
            on_event: Callable[[dict], None] | None = None) -> AgentAnswer:
        """Answer one question, or refuse and say why.

        `should_stop` is polled between steps; when it answers True, or when
        the backend reports its request was aborted, the loop ends with a
        refusal that says so rather than an answer nobody asked for.

        `on_event` receives every trace entry as it is made -- each query and
        what it returned, each thing the model wrote, each check -- so a
        window can show the work while it happens rather than after.
        """
        from onset_agent.backends import Interrupted

        trace: list[dict] = []

        def note(entry: dict) -> None:
            trace.append(entry)
            if on_event is not None:
                try:
                    on_event(dict(entry))
                except Exception:       # noqa: BLE001 - a listener must not end the loop
                    pass

        def stopped(step: int) -> AgentAnswer:
            note({"type": "stopped", "step": step})
            return AgentAnswer(question=question,
                               text="Stopped before an answer was produced.",
                               refused=True, reason="stopped by the reviewer",
                               trace=trace, backend=self.backend.name, verified=False)

        if guard.about_the_assistant(question):
            note({"type": "about", "result": "answered without the model"})
            return AgentAnswer(question=question,
                               text=what_i_can_do(self.store.subject, list(TOOLS)),
                               reason="about the assistant itself (answered without the model)",
                               trace=trace, backend=self.backend.name, verified=True)
        scope = guard.check_question(question, self.store.subject)
        if not scope.ok:
            note({"type": "scope_check", "result": "refused"})
            return AgentAnswer(question=question, text=scope.reason, refused=True,
                               reason="out of scope (checked before the model ran)",
                               trace=trace, backend=self.backend.name)

        messages = [{"role": "system", "content": self.system}]
        retrieved: list[str] = []
        numbers_seen: set[float] = set()
        retries = 0

        def run_calls(calls, *, briefed: bool) -> None:
            messages.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name,
                                             "arguments": json.dumps(c.arguments)}}
                               for c in calls]})
            for call in calls:
                payload, entry = self._run_tool(call)
                if briefed:
                    entry["briefing"] = True
                entry["digest"] = digest(call.name, payload)
                note(entry)
                if entry["ok"]:
                    for evidence_id in _collect_ids(payload):
                        if evidence_id not in retrieved:
                            retrieved.append(evidence_id)
                    numbers_seen.update(guard.collect_numbers(payload))
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": json.dumps(payload, default=str)})
                if self.verbose:
                    print(f"  [tool] {call.name}({call.arguments}) -> "
                          f"{'ok' if entry['ok'] else entry['error']}")

        if self.brief:
            from onset_agent.backends import ToolCall

            base = briefing_base(self.store)
            run_calls([ToolCall(name, args, id=f"brief-{i + 1}")
                       for i, (name, args) in enumerate(base)], briefed=True)
            messages.append({"role": "user", "content": question})
            extras = briefing_extras(self.store, question, base)
            if extras:
                run_calls([ToolCall(name, args, id=f"brief-{len(base) + i + 1}")
                           for i, (name, args) in enumerate(extras)], briefed=True)
        else:
            messages.append({"role": "user", "content": question})

        for step in range(self.max_steps):
            if should_stop is not None and should_stop():
                return stopped(step)
            # The briefed first call carries no tool schemas: they are most of
            # a thousand tokens the model does not need to read the briefing.
            # From the second step on they are offered, and the system prompt
            # says so.
            offer_tools = not self.brief or step > 0
            if self.brief and step == 1:
                messages[0] = {"role": "system", "content": self._system(tools_offered=True)}
            try:
                message = self.backend.chat(messages, tool_schemas() if offer_tools else [])
            except Interrupted:
                return stopped(step)
            if message.tool_calls:
                calls = message.tool_calls[:MAX_CALLS_PER_TURN]
                note({"type": "model", "step": step,
                      "asked_for": [f"{c.name}({_args(c.arguments)})" for c in calls],
                      "content": (message.content or "")[:400]})
                run_calls(calls, briefed=False)
                continue

            content = message.content or ""
            parsed = extract_json_object(content)
            if parsed is None:
                prose = _as_prose(content)
                if not prose:
                    note({"type": "format_error", "step": step, "content": content[:400]})
                    if retries >= self.max_retries:
                        break
                    retries += 1
                    messages.append({"role": "user", "content":
                                     "That was empty. Reply with exactly "
                                     '{"answer": "...", "evidence_ids": [...]} or '
                                     '{"refusal": "..."} and nothing else.'})
                    continue
                note({"type": "prose", "step": step, "content": prose[:400]})
                parsed = {"answer": prose, "evidence_ids": []}
            else:
                note({"type": "model", "step": step, "asked_for": [],
                      "content": content[:400]})

            if "refusal" in parsed:
                note({"type": "model_refusal", "step": step, "text": str(parsed["refusal"])[:400]})
                return AgentAnswer(question=question, text=str(parsed["refusal"]), refused=True,
                                   reason="the model declined", trace=trace,
                                   backend=self.backend.name)

            text = str(parsed.get("answer", "")).strip()
            cited = [str(i) for i in (parsed.get("evidence_ids") or []) if str(i).strip()]
            ids, attached, dropped = attach_citations(text, cited, retrieved)
            if attached or dropped:
                note({"type": "citations", "step": step, "kept": [i for i in cited if i in ids],
                      "attached": attached, "dropped": dropped})
            # An answer whose only citations were made up, with no retrieved
            # window for any channel it names, is checked with the made-up
            # ids and refused for them: a citation to nothing is not nothing.
            check = guard.verify_answer(text, ids or dropped, set(retrieved), numbers_seen,
                                        self.store)
            note({"type": "answer", "step": step, "verified": check.ok,
                  "problems": check.problems, "text": text[:600], "evidence_ids": ids})
            if check.ok:
                return AgentAnswer(question=question, text=text, evidence_ids=ids, trace=trace,
                                   backend=self.backend.name, verified=True)
            if retries >= self.max_retries:
                break
            retries += 1
            messages.append({"role": "user", "content":
                             "Your answer failed verification: " + "; ".join(check.problems) +
                             ". Every number must be copied from a tool result. Answer "
                             "again using only numbers that appear in the results above."})

        note({"type": "gave_up", "steps": self.max_steps, "retries": retries})
        return AgentAnswer(
            question=question,
            text=("I could not produce an answer I can stand behind: the checks on citations "
                  "and numbers did not pass. The saved report and tables are authoritative; "
                  "see report.md."),
            refused=True, reason="verification failed", trace=trace,
            backend=self.backend.name, verified=False)

    def ask_many(self, questions: list[str]) -> list[AgentAnswer]:
        return [self.ask(q) for q in questions]

    # -- internals --------------------------------------------------------
    def _run_tool(self, call) -> tuple[dict, dict]:
        entry = {"type": "tool_call", "tool": call.name, "arguments": call.arguments, "ok": True}
        try:
            payload = dispatch(self.store, call.name, call.arguments)
        except ToolError as exc:
            entry.update(ok=False, error=str(exc))
            return {"error": str(exc)}, entry
        except Exception as exc:  # a bug in a tool must not crash the session
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            return {"error": "the tool failed"}, entry
        return payload if isinstance(payload, dict) else {"result": payload}, entry


def _args(arguments) -> str:
    if not arguments:
        return ""
    return ", ".join(f"{k}={v}" for k, v in arguments.items())


def _collect_ids(payload) -> list[str]:
    """Every evidence_id a tool returned, in order -- the set a citation must
    come from."""
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "evidence_id" and isinstance(value, str):
                found.append(value)
            else:
                found += _collect_ids(value)
    elif isinstance(payload, list):
        for item in payload:
            found += _collect_ids(item)
    return found
