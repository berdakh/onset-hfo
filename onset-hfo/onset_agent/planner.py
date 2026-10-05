"""The planner: four rungs of increasing model control over the analysis.

The claim being tested is not "an agent is good". It is narrower and testable:

    An agent that selects analyses, sets their parameters, inspects
    intermediate results and re-plans accordingly can characterise a recording
    better -- and with a more defensible evidence trail -- than a fixed
    pipeline **using the same analyzers**.

"Using the same analyzers" is what makes the comparison clean. All four rungs
call the identical tool registry (:mod:`onset_agent.analysis`) against the
identical recording. The only thing that changes is who decides what to run:

===== ===================================================================
 S0   Fixed pipeline. A hard-coded sequence of tool calls, no model at all.
      This is the prior-work baseline, and it is the number to beat.
 S1   Single-shot tool calling. The model chooses the analyses once, they
      all run, it writes the report. No revision.
 S2   Planner with re-planning. The model sees each result before choosing
      the next call, and stops when a stopping rule fires.
 S3   Planner plus verifier. As S2, then every claim in the draft must
      resolve to a run id or be struck.
===== ===================================================================

Where the difference can actually show up
-----------------------------------------
A ladder whose rungs cannot produce different answers measures nothing. The
mechanism here is :func:`rank_channels`. A channel's score is its measured
rate at the survey threshold, multiplied by how well that rate survived any
*stricter* threshold the planner chose to re-test it at. S0 and S1 never
re-test, so their robustness factor is always 1. S2 and S3 can demote a
channel whose rate collapses under scrutiny -- which is exactly the behaviour
the thesis is about, and it is measurable against the SOZ labels in
:mod:`onset_hfo.cohort`.

The model never produces a number. It chooses which measurements to make and
how to phrase them; the ranking is assembled from the evidence store by code,
and the report is checked against it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from onset_agent.analysis import AnalysisSession, build_registry
from onset_agent.backends import Backend, extract_json_object
from onset_agent.contract import ToolRegistry
from onset_agent.evidence import EvidenceStore
from onset_agent.prompts import planner_prompt, report_prompt

__all__ = ["Rung", "PlannerResult", "StopRule", "FixedBudget", "ModelJudged",
           "TiedSetWidth", "ConformalWidth", "rank_channels", "run_rung",
           "detected_nothing", "ScriptedPlanner", "FIXED_PLAN"]


class Rung(str, Enum):
    """How much control the model has. See the module docstring."""

    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"

    @property
    def uses_model(self) -> bool:
        return self is not Rung.S0

    @property
    def replans(self) -> bool:
        return self in (Rung.S2, Rung.S3)

    @property
    def verifies(self) -> bool:
        return self is Rung.S3


# --------------------------------------------------------------------------
# S0: the fixed pipeline
# --------------------------------------------------------------------------

#: The prior-work baseline: the analyses a fixed pipeline runs on every
#: patient, in the same order, whatever the recording turns out to look like.
#: It surveys and reports; it never challenges its own result, because there
#: is nobody in the loop to decide to.
FIXED_PLAN: list[tuple[str, dict]] = [
    ("get_recording_metadata", {}),
    ("channel_qc", {"max_channels": 20}),
    ("detect_hfo", {"detector": "rms", "k": 40}),
    ("compare_detectors", {}),
    ("rate_change", {"k": 10}),
]


# --------------------------------------------------------------------------
# Stopping rules
# --------------------------------------------------------------------------


class StopRule:
    """Decides when the planner has gathered enough evidence.

    Comparing stopping rules is a result in its own right: accuracy against
    cost, for each rule, on the same recordings.
    """

    name = "stop_rule"

    def should_stop(self, store: EvidenceStore, model_said_done: bool) -> tuple[bool, str]:
        raise NotImplementedError


@dataclass
class FixedBudget(StopRule):
    """Stop after ``k`` tool calls. The cheapest rule and the honest floor."""

    k: int = 8
    name: str = "fixed_budget"

    def should_stop(self, store: EvidenceStore, model_said_done: bool) -> tuple[bool, str]:
        if len(store) >= self.k:
            return True, f"budget of {self.k} tool calls exhausted"
        return False, ""


@dataclass
class ModelJudged(StopRule):
    """Stop when the model says the evidence is sufficient.

    Cheap, and exactly as trustworthy as the model's self-assessment -- which
    is the point of measuring it against the other two rules rather than
    assuming it.
    """

    max_calls: int = 12
    name: str = "model_judged"

    def should_stop(self, store: EvidenceStore, model_said_done: bool) -> tuple[bool, str]:
        if model_said_done:
            return True, "the model judged the evidence sufficient"
        if len(store) >= self.max_calls:
            return True, f"hard ceiling of {self.max_calls} tool calls reached"
        return False, ""


@dataclass
class ConformalWidth(StopRule):
    """Stop when the learned model's candidate set is narrow enough to act on.

    This is the uncertainty-driven stopping rule, and the one place the two
    halves of the system actually meet. The planner keeps gathering evidence
    while the set of channels that cannot be ruled out at ``1 - alpha`` is
    wider than ``max_width``, and stops when it is not.

    Unlike :class:`TiedSetWidth`, this one rests on a real split-conformal
    threshold, so ``guarantees_coverage`` is True *in the sense the theorem
    means it*: coverage holds when calibration and test data are
    exchangeable. On a new patient from a new centre they are not, and the
    measured degradation is reported by
    :func:`onset_hfo.uncertainty.exchangeability_stress_test` rather than
    assumed away. A rule that stops early because a broken guarantee said the
    set was narrow is the failure mode to watch for, and it is why the
    stopping rules are compared rather than one of them chosen.

    Requires a model on the session; without one the width is unavailable and
    the rule falls back to its call ceiling, so a missing model degrades the
    planner rather than crashing it.
    """

    max_width: int = 5
    max_calls: int = 12
    name: str = "conformal_width"
    guarantees_coverage: bool = True

    @staticmethod
    def width(store: EvidenceStore) -> int | None:
        runs = store.runs("estimate_soz_probability", ok_only=True)
        if not runs:
            return None
        return (runs[-1].output or {}).get("candidate_set_size")

    def should_stop(self, store: EvidenceStore, model_said_done: bool) -> tuple[bool, str]:
        if len(store) >= self.max_calls:
            return True, f"hard ceiling of {self.max_calls} tool calls reached"
        current = self.width(store)
        if current is None:
            return False, ""
        if current == 0:
            # An empty candidate set is not a narrow one. It means the model
            # found neither label plausible for any channel, which is what
            # happens when it is applied outside the distribution it was
            # fitted on -- a cohort model on a different montage, a different
            # sampling rate, a simulation. Reporting that as "narrowed enough
            # to act on" would turn the most obvious failure mode into the
            # success condition, so it stops and says what happened.
            return True, ("the conformal candidate set is EMPTY: the model ruled out every "
                          "channel, which means it is being applied outside the "
                          "distribution it was calibrated on, not that the recording was "
                          "localized")
        if current <= self.max_width:
            return True, (f"the conformal candidate set is down to {current} channel(s), "
                          f"at or below the {self.max_width} needed to act on")
        return False, ""


@dataclass
class TiedSetWidth(StopRule):
    """Stop when the set of channels that cannot be told apart is small enough.

    Two channels whose Poisson rate intervals overlap are tied, not ranked, so
    the honest answer to "which channel leads?" is a *set*. This rule keeps
    gathering evidence while that set is wider than ``max_width``.

    This is the interface the calibrated-classifier half of the project plugs
    into: replace :meth:`width` with the size of a conformal prediction set at
    ``1 - alpha`` and the rule becomes the uncertainty-driven stopping
    criterion, with everything around it unchanged. What is implemented here
    is a tied-rank set from intervals the pipeline already computes -- it is
    not a conformal set and carries no coverage guarantee, and the attribute
    ``guarantees_coverage`` says so to anyone who asks.
    """

    max_width: int = 3
    max_calls: int = 12
    name: str = "tied_set_width"
    guarantees_coverage: bool = False

    @staticmethod
    def width(store: EvidenceStore) -> int | None:
        """How many channels are statistically tied with the leader."""
        best: dict | None = None
        for run in store.runs("detect_hfo", ok_only=True):
            channels = (run.output or {}).get("channels") or {}
            if len(channels) > len(((best or {}).get("channels")) or {}):
                best = run.output
        channels = (best or {}).get("channels") or {}
        if len(channels) < 2:
            return None
        rows = sorted(channels.values(), key=lambda r: -float(r.get("rate_per_min", 0.0)))
        leader_low = float(rows[0].get("rate_ci", [0, 0])[0])
        return sum(1 for row in rows if float(row.get("rate_ci", [0, 0])[1]) >= leader_low)

    def should_stop(self, store: EvidenceStore, model_said_done: bool) -> tuple[bool, str]:
        if len(store) >= self.max_calls:
            return True, f"hard ceiling of {self.max_calls} tool calls reached"
        current = self.width(store)
        if current is None:
            return False, ""
        if current <= self.max_width:
            return True, f"only {current} channel(s) remain statistically tied for the lead"
        return False, ""


# --------------------------------------------------------------------------
# Turning an evidence store into a ranking
# --------------------------------------------------------------------------


@dataclass
class RankedChannel:
    channel: str
    survey_rate_per_min: float
    survey_threshold_sd: float
    robustness: float
    score: float
    run_ids: list[str] = field(default_factory=list)
    retested_at: list[float] = field(default_factory=list)
    #: Stricter thresholds that found nothing *anywhere* and so were not
    #: allowed to lower this channel's robustness. Carried separately from
    #: ``retested_at`` because they are not re-tests: see `rank_channels`.
    unmeasured_at: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {"channel": self.channel,
               "survey_rate_per_min": round(self.survey_rate_per_min, 3),
               "survey_threshold_sd": self.survey_threshold_sd,
               "robustness": round(self.robustness, 3),
               "score": round(self.score, 3),
               "retested_at": self.retested_at, "run_ids": self.run_ids}
        if self.unmeasured_at:
            out["unmeasured_at"] = self.unmeasured_at
            out["unmeasured_note"] = (
                "found no events on any channel, so it could not distinguish "
                "this channel from any other and did not lower its robustness")
        return out


def detected_nothing(output: dict) -> bool:
    """Did this ``detect_hfo`` run find no events at all, on any channel?

    Asked of the run rather than of a channel on purpose. A channel that went
    quiet while its neighbours kept firing has been *refuted*; a threshold at
    which the whole band went quiet has measured nothing, and the difference
    decides whether it may lower anybody's robustness.

    ``n_accepted`` is the run-level count over every channel analysed, which is
    what the question needs: ``channels`` holds only the top *k*, so polling it
    cannot tell silence from truncation. It is still the fallback for a run
    recorded before that field existed -- and a sound one, because the rows are
    ordered by rate, so a leading rate of zero means every rate is zero. An
    output carrying neither is *not* reported silent: absence of evidence here
    would wrongly suppress a real re-test.
    """
    accepted = output.get("n_accepted")
    if accepted is not None:
        try:
            return int(accepted) <= 0
        except (TypeError, ValueError):
            pass
    rows = (output.get("channels") or {}).values()
    rates = []
    for row in rows:
        try:
            rates.append(float(row.get("rate_per_min")))
        except (TypeError, ValueError, AttributeError):
            return False
    return bool(rates) and max(rates) <= 0.0


def rank_channels(store: EvidenceStore) -> list[RankedChannel]:
    """Assemble a channel ranking from the measurements in the evidence store.

    The rule, stated plainly because it decides every number downstream:

    * a channel's **survey rate** is its rate at the *lowest* threshold it was
      measured at -- the broad look;
    * its **robustness** is the worst ratio ``rate_at_stricter / survey_rate``
      over every stricter threshold it was re-tested at, capped at 1.0;
    * its **score** is ``survey_rate x robustness``.

    Capping at 1.0 is deliberate: a re-test can lower confidence in a channel,
    never raise it. A channel nobody re-tested keeps robustness 1.0, which
    means *unchallenged*, not *verified* -- and that distinction is the whole
    difference between the fixed rungs and the re-planning ones.

    **A re-test that found nothing anywhere is not a re-test.** If a stricter
    threshold detected no events on any channel it analysed, then the band went
    silent rather than the channels being refuted: a robustness of
    ``0 / survey_rate`` there means *unmeasured*, and it is the mirror image of
    the *unchallenged, not verified* case above. Firing on it sets every
    channel's robustness to 0, so every score becomes 0 and the ranking
    degenerates to alphabetical order -- a confident-looking ordering carrying
    no information. Measured over the 20-subject cohort this hit 84 of 600
    multiplied windows and annihilated five patients' rankings outright at a
    2.0x re-test, taking fast-ripple AUC from 0.753 to 0.528. Such a threshold
    is skipped and recorded in ``unmeasured_at``, never silently dropped.

    A *partially* silent re-test is left alone: when some channels keep events
    and others go quiet, the zeros discriminate, which is the rule working.

    Only channels some tool actually measured are ranked. A ranking over
    channels nobody looked at would be a guess.
    """
    measured: dict[str, dict[float, tuple[float, str, bool]]] = {}
    for run in store.runs("detect_hfo", ok_only=True):
        output = run.output or {}
        threshold = float(output.get("threshold_sd", 0.0))
        silent = detected_nothing(output)
        for channel, row in (output.get("channels") or {}).items():
            try:
                rate = float(row.get("rate_per_min"))
            except (TypeError, ValueError, AttributeError):
                continue
            measured.setdefault(channel, {})[threshold] = (rate, run.run_id, silent)

    ranked: list[RankedChannel] = []
    for channel, by_threshold in measured.items():
        base_threshold = min(by_threshold)
        base_rate, base_run, _ = by_threshold[base_threshold]
        robustness, retested, runs, unmeasured = 1.0, [], [base_run], []
        for threshold, (rate, run_id, silent) in sorted(by_threshold.items()):
            if threshold <= base_threshold:
                continue
            runs.append(run_id)
            if silent:
                # Unmeasured, not refuted. See the docstring.
                unmeasured.append(threshold)
                continue
            retested.append(threshold)
            ratio = 1.0 if base_rate <= 0 else min(1.0, rate / base_rate)
            robustness = min(robustness, ratio)
        ranked.append(RankedChannel(channel=channel, survey_rate_per_min=base_rate,
                                    survey_threshold_sd=base_threshold, robustness=robustness,
                                    score=base_rate * robustness, run_ids=runs,
                                    retested_at=retested, unmeasured_at=unmeasured))
    ranked.sort(key=lambda r: (-r.score, r.channel))
    return ranked


# --------------------------------------------------------------------------
# The result of one run
# --------------------------------------------------------------------------


@dataclass
class PlannerResult:
    """Everything one rung produced, including how it got there."""

    rung: Rung
    subject: str
    ranking: list[RankedChannel] = field(default_factory=list)
    report: str = ""
    cited_run_ids: list[str] = field(default_factory=list)
    struck: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    steps: list[dict] = field(default_factory=list)
    store: EvidenceStore | None = None
    backend: str = ""
    verification: dict = field(default_factory=dict)

    @property
    def top_channels(self) -> list[str]:
        return [r.channel for r in self.ranking]

    def top_k(self, k: int = 5) -> list[str]:
        return self.top_channels[:k]

    @property
    def cost(self) -> dict:
        return self.store.cost() if self.store else {}

    @property
    def n_retested(self) -> int:
        """How many channels the planner actually challenged. Zero for S0 and S1."""
        return sum(1 for r in self.ranking if r.retested_at)

    def as_dict(self) -> dict:
        return {"rung": self.rung.value, "subject": self.subject, "backend": self.backend,
                "stop_reason": self.stop_reason, "cost": self.cost,
                "n_channels_retested": self.n_retested,
                "ranking": [r.as_dict() for r in self.ranking[:20]],
                "report": self.report, "cited_run_ids": self.cited_run_ids,
                "struck": self.struck, "verification": self.verification,
                "steps": self.steps}


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_rung(session: AnalysisSession, rung: Rung | str = Rung.S2,
             backend: Backend | None = None, stop_rule: StopRule | None = None,
             max_steps: int = 12, verifier: Callable | None = None,
             verbose: bool = False) -> PlannerResult:
    """Run one rung of the ladder end to end on one recording.

    Parameters
    ----------
    session:
        The cached recording. All rungs share the same analyzers through it.
    rung:
        ``S0``-``S3``; see the module docstring.
    backend:
        The language-model backend. Ignored for ``S0``. Defaults to
        :class:`ScriptedPlanner`, which is deterministic and needs no model,
        so the ladder runs in CI.
    stop_rule:
        Only consulted for the re-planning rungs. Defaults to
        :class:`ModelJudged`.
    verifier:
        ``f(report, store) -> VerificationResult``; only used for ``S3``.
        Defaults to the deterministic verifier.
    """
    rung = Rung(rung) if not isinstance(rung, Rung) else rung
    registry = build_registry(session)
    store = EvidenceStore(subject=session.recording.subject, source=session.recording.source)
    backend = backend or ScriptedPlanner()
    steps: list[dict] = []

    if rung is Rung.S0:
        stop_reason = _run_fixed_plan(registry, store, steps, verbose)
        result = PlannerResult(rung=rung, subject=store.subject, store=store, steps=steps,
                               stop_reason=stop_reason, backend="none (fixed pipeline)")
        result.ranking = rank_channels(store)
        result.report = _template_report(result, store)
        result.cited_run_ids = sorted({rid for r in result.ranking for rid in r.run_ids})
        return result

    stop_rule = stop_rule or (ModelJudged() if rung.replans else FixedBudget(k=max_steps))
    stop_reason = _run_planned(registry, store, steps, backend, session, rung, stop_rule,
                               max_steps, verbose)

    result = PlannerResult(rung=rung, subject=store.subject, store=store, steps=steps,
                           stop_reason=stop_reason, backend=backend.name)
    result.ranking = rank_channels(store)
    result.report, result.cited_run_ids = _draft_report(backend, session, store, steps, verbose)

    if rung.verifies:
        from onset_agent.verifier import DeterministicVerifier
        check = (verifier or DeterministicVerifier())(result.report, store)
        result.report = check.kept_text
        result.struck = check.struck
        result.verification = check.as_dict()
        steps.append({"type": "verify", **check.as_dict()})
    return result


def _run_fixed_plan(registry: ToolRegistry, store: EvidenceStore, steps: list[dict],
                    verbose: bool) -> str:
    for tool, args in FIXED_PLAN:
        run = registry.run(tool, args)
        store.append(run)
        steps.append({"type": "tool", "run_id": run.run_id, "tool": tool,
                      "arguments": args, "ok": run.ok, "chosen_by": "fixed plan"})
        if verbose:
            print(f"  [{run.run_id}] {tool}({args}) {'ok' if run.ok else run.error}")
    return "the fixed plan finished; it has no stopping decision to make"


def _run_planned(registry: ToolRegistry, store: EvidenceStore, steps: list[dict],
                 backend: Backend, session: AnalysisSession, rung: Rung,
                 stop_rule: StopRule, max_steps: int, verbose: bool) -> str:
    """S1/S2/S3: the model chooses the calls. S1 gets no feedback between them."""
    system = planner_prompt(subject=session.recording.subject,
                            source=session.recording.source,
                            tool_names=registry.names())
    messages = [{"role": "system", "content": system},
                {"role": "user", "content":
                 "Plan the analysis of this recording. Begin."}]
    model_said_done = False
    idle_turns = 0

    for step in range(max_steps):
        message = backend.chat(messages, registry.schemas())
        calls = _parse_plan(message)
        if not calls:
            steps.append({"type": "format_error", "step": step,
                          "content": (message.content or "")[:300]})
            messages.append({"role": "user", "content":
                             "That was not a valid plan. Reply with exactly one JSON object: "
                             '{"thought": "...", "tool": "...", "arguments": {...}} or '
                             '{"thought": "...", "done": true}.'})
            if step >= max_steps - 1:
                break
            continue

        done, tool, args, thought = calls
        if done:
            model_said_done = True
            idle_turns += 1
            # The model has nothing more it wants to run, but the stopping
            # rule is not satisfied. Ask once for something that would narrow
            # the answer; if it still has nothing, stop rather than spin
            # through the step limit producing no evidence.
            if idle_turns >= 2:
                steps.append({"type": "stop", "rule": stop_rule.name, "step": step,
                              "reason": "the model had nothing further to run"})
                return ("the model had nothing further to run, although "
                        f"the {stop_rule.name} rule was not satisfied")
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({"role": "user", "content":
                             "The stopping rule is not satisfied yet. Is there an analysis "
                             "that would narrow which channels you would name? Run it, or "
                             "say done again."})
        else:
            idle_turns = 0
            run = registry.run(tool, args)
            store.append(run)
            steps.append({"type": "tool", "run_id": run.run_id, "tool": tool,
                          "arguments": args, "ok": run.ok, "chosen_by": "model",
                          "thought": thought})
            if verbose:
                print(f"  [{run.run_id}] {tool}({args}) "
                      f"{'ok' if run.ok else 'FAILED: ' + run.error}")
            messages.append({"role": "assistant", "content": message.content or ""})
            # S1 is single-shot by construction: the model is not shown what
            # its calls returned, so it cannot revise. S2/S3 see everything.
            # S1 is single-shot by design: it learns that its call happened
            # (a model knows what it asked for) but never what came back, so
            # it has nothing to revise against.
            feedback = (f"[{run.run_id}] returned: "
                        f"{json.dumps(run.output, default=str)[:1200]}"
                        if rung.replans else
                        f"Call recorded as [{run.run_id}]. You will not see its result.")
            messages.append({"role": "user", "content": feedback + "\nWhat next?"})

        should_stop, reason = stop_rule.should_stop(store, model_said_done)
        if should_stop:
            steps.append({"type": "stop", "rule": stop_rule.name, "reason": reason,
                          "step": step})
            return reason
    return f"reached the step limit of {max_steps}"


def _draft_report(backend: Backend, session: AnalysisSession, store: EvidenceStore,
                  steps: list[dict], verbose: bool) -> tuple[str, list[str]]:
    # A larger per-run budget than the planner gets: the planner needs a small
    # context to choose well, the writer needs the numbers it is quoting.
    prompt = report_prompt(subject=session.recording.subject,
                           evidence=store.digest(per_run_chars=900))
    message = backend.chat([{"role": "user", "content": prompt}], [])
    parsed = extract_json_object(message.content or "") or {}
    text = str(parsed.get("report", "")).strip()
    run_ids = [str(i) for i in (parsed.get("run_ids") or []) if str(i).strip()]
    steps.append({"type": "draft", "n_chars": len(text), "cited": run_ids})
    if verbose:
        print(f"  [draft] {len(text)} characters, cites {', '.join(run_ids) or 'nothing'}")
    return text, run_ids


def _template_report(result: PlannerResult, store: EvidenceStore) -> str:
    """S0's report: assembled by code, because S0 has no model in it."""
    if not result.ranking:
        return "The fixed pipeline produced no channel measurements."
    lines = []
    for ranked in result.ranking[:3]:
        lines.append(f"{ranked.channel} had {ranked.survey_rate_per_min:.1f} ripples/min at "
                     f"{ranked.survey_threshold_sd:.0f} robust SD "
                     f"[{ranked.run_ids[0]}].")
    agreement = store.runs("compare_detectors", ok_only=True)
    if agreement:
        jaccard = (agreement[-1].output or {}).get("agreement", {}).get("jaccard")
        if jaccard is not None:
            lines.append(f"The two detectors' event-by-event agreement was "
                         f"{jaccard} [{agreement[-1].run_id}].")
    lines.append("No channel was re-tested at a stricter threshold: the fixed pipeline runs "
                 "the same analyses on every recording and has no mechanism for challenging "
                 "its own result.")
    return " ".join(lines)


def _parse_plan(message) -> tuple[bool, str, dict, str] | None:
    """Read one planning decision, from a native tool call or from JSON text."""
    if getattr(message, "tool_calls", None):
        call = message.tool_calls[0]
        return False, call.name, dict(call.arguments or {}), ""
    parsed = extract_json_object(message.content or "")
    if not isinstance(parsed, dict):
        return None
    thought = str(parsed.get("thought", ""))
    if parsed.get("done"):
        return True, "", {}, thought
    tool = parsed.get("tool")
    if not isinstance(tool, str) or not tool:
        return None
    args = parsed.get("arguments")
    return False, tool, dict(args) if isinstance(args, dict) else {}, thought


# --------------------------------------------------------------------------
# A deterministic planner, so the ladder runs without a model
# --------------------------------------------------------------------------


class ScriptedPlanner(Backend):
    """A planner with no language model in it, for tests and for S1's floor.

    It follows the strategy the prompt describes -- survey, then challenge the
    leaders at a stricter threshold -- using the evidence it has been shown.
    That makes it a useful control in two ways: the ladder runs end to end in
    CI with no weights, and any improvement a real model shows has to beat a
    competent script rather than a straw man.

    It reads only what a model would read: the text of the messages it is
    given. When it is run as S1 it is told nothing about its results, so, like
    a model in that rung, it cannot challenge anything.
    """

    name = "scripted-planner"

    def __init__(self, retest_threshold: float = 7.0, max_retests: int = 2):
        self.retest_threshold = retest_threshold
        self.max_retests = max_retests

    def describe(self) -> str:
        return "a deterministic planner (no language model)"

    def chat(self, messages: list[dict], tools: list[dict]):
        from onset_agent.backends import AssistantMessage

        last = messages[-1]["content"] if messages else ""
        if "Reply with ONE JSON object:" in str(last) and '"report"' in str(last):
            return AssistantMessage(content=json.dumps(self._report(str(last))))

        seen = " ".join(str(m.get("content", "")) for m in messages)
        plan = self._next_step(seen)
        return AssistantMessage(content=json.dumps(plan))

    # -- the strategy -----------------------------------------------------
    def _next_step(self, transcript: str) -> dict:
        if "meta_001" not in transcript:
            return {"thought": "find out what is being analysed",
                    "tool": "get_recording_metadata", "arguments": {}}
        if "qc_001" not in transcript:
            return {"thought": "check which channels are analysable",
                    "tool": "channel_qc", "arguments": {"max_channels": 20}}
        if "hfo_001" not in transcript:
            return {"thought": "survey every channel once before looking closely at any",
                    "tool": "detect_hfo", "arguments": {"detector": "rms", "k": 12}}

        leaders = self._leaders(transcript)
        # Which leaders have already been challenged. Asking the transcript
        # for the channel by name is exact; counting occurrences of the
        # threshold is not, because the transcript echoes each call back.
        challenged = [c for c in leaders if f'"channels": ["{c}"]' in transcript]
        todo = [c for c in leaders if c not in challenged]
        if todo and len(challenged) < self.max_retests:
            channel = todo[0]
            return {"thought": f"test whether {channel}'s rate survives a stricter threshold",
                    "tool": "detect_hfo",
                    "arguments": {"channels": [channel], "detector": "rms",
                                  "threshold_sd": self.retest_threshold, "k": 5}}
        if "compare_001" not in transcript:
            return {"thought": "check whether the two detectors agree about the leaders",
                    "tool": "compare_detectors", "arguments": {}}
        if "soz_001" not in transcript and "no learned SOZ model" not in transcript:
            return {"thought": "ask the learned model how far the candidate set has narrowed",
                    "tool": "estimate_soz_probability", "arguments": {"k": 10}}
        if "evidence_001" not in transcript and leaders:
            return {"thought": "retrieve citable events for the leading channel",
                    "tool": "get_event_evidence",
                    "arguments": {"channel": leaders[0], "k": 3}}
        return {"thought": "the leaders have been surveyed, challenged and cross-checked",
                "done": True}

    #: A channel entry in any detect_hfo output, whatever order its keys are in.
    _RATE_ENTRY = re.compile(r'"([A-Za-z0-9\'\-]+)":\s*\{[^{}]*?"rate_per_min":\s*([\d.]+)')

    @classmethod
    def _leaders(cls, transcript: str) -> list[str]:
        """The channels the survey put on top, read back out of the transcript.

        Order matters and is preserved: the tool returns channels highest rate
        first, so the first matches are the leaders. Duplicates are dropped so
        that a channel re-tested at a stricter threshold does not push the
        others out of the list.
        """
        seen: list[str] = []
        for channel, _rate in cls._RATE_ENTRY.findall(transcript):
            if channel not in seen:
                seen.append(channel)
        return seen[:3]

    def _report(self, prompt: str) -> dict:
        """Write the findings straight out of the evidence digest it was handed."""
        rows = re.findall(
            r'\[(\w+_\d+)\] detect_hfo\((.*?)\) -> (.*?)(?=\n\[|\Z)', prompt, re.DOTALL)
        sentences, cited = [], []
        for run_id, args, body in rows[:3]:
            rate = self._RATE_ENTRY.search(body)
            threshold = re.search(r'threshold_sd=([\d.]+)', args)
            if not rate:
                continue
            level = f" at {threshold.group(1)} robust SD" if threshold else ""
            sentences.append(f"{rate.group(1)} had {rate.group(2)} ripples/min"
                             f"{level} [{run_id}].")
            cited.append(run_id)
        if not sentences:
            sentences.append("No channel rates were measured in this session.")
        sentences.append("Rates come from one short window and physiological ripples occur "
                         "in healthy tissue, so this is a measurement and not a localisation.")
        return {"report": " ".join(sentences), "run_ids": cited}
