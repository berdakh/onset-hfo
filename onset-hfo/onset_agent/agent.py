"""The agent loop.

    question
      -> scope check (no model involved)        guard.check_question
      -> model chooses a tool                   backend.chat
      -> tool runs against the saved results    tools.dispatch
      -> ... up to `max_steps` times ...
      -> model returns JSON                     {"answer", "evidence_ids"} or {"refusal"}
      -> citation and number check              guard.verify_answer
      -> answer, or a refusal explaining why

What makes this an *agent* rather than a chatbot with a database: the model
decides which questions to ask of the data and when it has enough, and the
loop lets it act on what it learned. What keeps it safe is that everything it
can do is a read-only query, and everything it says is checked against what
those queries returned.

Failure is a first-class outcome. If the model cites something it never
retrieved, or states a number no tool produced, the answer is discarded and
the agent says it cannot answer. That is a deliberate design choice: a
prototype that occasionally says "I can't" is usable; one that occasionally
invents a rate is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from onset_agent import guard
from onset_agent.backends import Backend, ScriptedBackend, extract_json_object
from onset_agent.prompts import system_prompt
from onset_agent.tools import TOOLS, ToolError, dispatch, tool_schemas
from onset_hfo.store import ResultStore

__all__ = ["OnsetAgent", "AgentAnswer"]

MAX_CALLS_PER_TURN = 3


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
        return [step["tool"] for step in self.trace if step.get("type") == "tool_call"]

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.text, "refused": self.refused,
                "reason": self.reason, "evidence_ids": self.evidence_ids,
                "tools_called": self.tools_called, "backend": self.backend,
                "verified": self.verified, "trace": self.trace}

    def __str__(self) -> str:
        head = "REFUSED: " if self.refused else ""
        cites = ("\n  cites: " + ", ".join(self.evidence_ids)) if self.evidence_ids else ""
        tools = ("\n  tools: " + " -> ".join(self.tools_called)) if self.tools_called else ""
        return f"{head}{self.text}{cites}{tools}"


class OnsetAgent:
    """An evidence-only assistant over one saved analysis."""

    def __init__(self, store: ResultStore, backend: Backend | None = None,
                 max_steps: int = 6, max_retries: int = 2, verbose: bool = False):
        self.store = store
        self.backend = backend or ScriptedBackend()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose
        meta = store.metadata()
        self.system = system_prompt(subject=str(meta.get("subject")),
                                    source=str(meta.get("source")),
                                    tool_names=list(TOOLS))

    # -- public API -------------------------------------------------------
    def ask(self, question: str, should_stop=None) -> AgentAnswer:
        """Answer one question, or refuse and say why.

        `should_stop` is polled between steps; when it answers True, or when
        the backend reports its request was aborted, the loop ends with a
        refusal that says so rather than an answer nobody asked for.
        """
        from onset_agent.backends import Interrupted

        trace: list[dict] = []

        def stopped(step: int) -> AgentAnswer:
            trace.append({"type": "stopped", "step": step})
            return AgentAnswer(question=question,
                               text="Stopped before an answer was produced.",
                               refused=True, reason="stopped by the reviewer",
                               trace=trace, backend=self.backend.name, verified=False)
        scope = guard.check_question(question, self.store.subject)
        if not scope.ok:
            trace.append({"type": "scope_check", "result": "refused"})
            return AgentAnswer(question=question, text=scope.reason, refused=True,
                               reason="out of scope (checked before the model ran)",
                               trace=trace, backend=self.backend.name)

        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": question}]
        retrieved_ids: set[str] = set()
        numbers_seen: set[float] = set()
        retries = 0

        for step in range(self.max_steps):
            if should_stop is not None and should_stop():
                return stopped(step)
            try:
                message = self.backend.chat(messages, tool_schemas())
            except Interrupted:
                return stopped(step)
            if message.tool_calls:
                calls = message.tool_calls[:MAX_CALLS_PER_TURN]
                messages.append({
                    "role": "assistant", "content": message.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name,
                                                 "arguments": json.dumps(c.arguments)}}
                                   for c in calls]})
                for call in calls:
                    payload, entry = self._run_tool(call)
                    trace.append(entry)
                    if entry["ok"]:
                        retrieved_ids.update(_collect_ids(payload))
                        numbers_seen |= guard.collect_numbers(payload)
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": json.dumps(payload, default=str)})
                    if self.verbose:
                        print(f"  [tool] {call.name}({call.arguments}) -> "
                              f"{'ok' if entry['ok'] else entry['error']}")
                continue

            parsed = extract_json_object(message.content or "")
            if parsed is None:
                trace.append({"type": "format_error", "step": step,
                              "content": (message.content or "")[:400]})
                if retries >= self.max_retries:
                    break
                retries += 1
                messages.append({"role": "user", "content":
                                 "That was not a single JSON object. Reply with exactly "
                                 '{"answer": "...", "evidence_ids": [...]} or '
                                 '{"refusal": "..."} and nothing else.'})
                continue

            if "refusal" in parsed:
                trace.append({"type": "model_refusal", "step": step})
                return AgentAnswer(question=question, text=str(parsed["refusal"]), refused=True,
                                   reason="the model declined", trace=trace,
                                   backend=self.backend.name)

            text = str(parsed.get("answer", "")).strip()
            ids = [str(i) for i in (parsed.get("evidence_ids") or []) if str(i).strip()]
            check = guard.verify_answer(text, ids, retrieved_ids, numbers_seen, self.store)
            trace.append({"type": "answer", "step": step, "verified": check.ok,
                          "problems": check.problems, "text": text[:600],
                          "evidence_ids": ids})
            if check.ok:
                return AgentAnswer(question=question, text=text, evidence_ids=ids, trace=trace,
                                   backend=self.backend.name, verified=True)
            if retries >= self.max_retries:
                break
            retries += 1
            messages.append({"role": "user", "content":
                             "Your answer failed verification: " + "; ".join(check.problems) +
                             ". Every number must be copied from a tool result and every "
                             "evidence_id must come from get_evidence. Call the tools you "
                             "need and answer again."})

        trace.append({"type": "gave_up", "steps": self.max_steps, "retries": retries})
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


def _collect_ids(payload) -> set[str]:
    """Every evidence_id a tool returned -- the set a citation must come from."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "evidence_id" and isinstance(value, str):
                found.add(value)
            else:
                found |= _collect_ids(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _collect_ids(item)
    return found
