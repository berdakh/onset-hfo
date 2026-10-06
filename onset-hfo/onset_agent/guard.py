"""Checks around the model: what it may be asked, and what it may say.

Two gates:

* **Before the model runs** -- :func:`check_question` refuses questions that
  are out of scope no matter what the evidence says (treatment, diagnosis,
  another patient). Refusing before the model is asked is cheaper, faster and
  not subject to persuasion.
* **After the model answers** -- :func:`verify_answer` checks that every
  citation resolves to a window the agent actually retrieved, and that every
  number carrying a unit appears in the tool results. A claim that fails is
  not corrected or softened: the answer is dropped and the agent says it
  cannot answer.

This is where "the language model never produces a number" stops being a
slogan. The model chooses which tool to call and how to phrase the result; the
numbers come from the pipeline, and this module proves it for each answer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

__all__ = ["check_question", "verify_answer", "collect_numbers", "GuardResult",
           "about_the_assistant"]

DIRECTIVE = re.compile(
    r"\b(resect\w*|ablat\w*|remov\w*|operat\w*|surg\w*|implant\w*|"
    r"treat\w*|therap\w*|medicat\w*|drug|dose|dosage|"
    r"should (?:we|i|they|he|she)|recommend\w*|advise|advice|what would you do)\b",
    re.IGNORECASE)

DIAGNOSIS = re.compile(
    r"\b(diagnos\w*|prognos\w*|outcome|cure[d]?|seizure[- ]?free|"
    r"does (?:this |the )?patient have|is (?:this|the) patient|"
    r"where (?:do|does) (?:the )?seizures? (?:start|begin|originate)|"
    r"seizure onset zone|soz|epileptogenic zone)\b",
    re.IGNORECASE)

OTHER_SUBJECT = re.compile(r"\b(sub-[a-z]{2,4}\d{1,3}|patient\s+\d{1,3}|pt\s?\d{1,3})\b",
                           re.IGNORECASE)

#: A number followed by one of these units must be traceable to a tool result.
UNIT = r"(?:/\s*min|per\s+minute|hz|hertz|ms|milliseconds?|seconds?|\bs\b|db|uv|µv|%|x\b|times)"
#: "71 events/min", "8 ripples per minute": a noun may sit between the two.
NOUN = r"(?:\s+(?:events?|detections?|ripples?|discharges?|spikes?|oscillations?|windows?))?"
# The lookbehind stops "45.898-45.93 s" from being read as the negative
# number -45.93: inside a range, the dash is a separator, not a sign.
NUMBER_WITH_UNIT = re.compile(rf"(?<![\d.])(-?\d+(?:\.\d+)?){NOUN}\s*({UNIT})", re.IGNORECASE)
#: Any other number in the text. Small integers are allowed through (ranks,
#: "the top 3 channels", "two detectors"); everything else must be traceable.
BARE_NUMBER = re.compile(r"(?<![\d.\w])(-?\d+(?:\.\d+)?)(?![\d.]*\w)")
MAX_FREE_INTEGER = 20
#: Channel names and evidence ids contain digits that are not measurements.
CHANNEL_TOKEN = re.compile(r"\b[A-Za-z]{1,6}'?\d{1,3}(?:-[A-Za-z]{1,6}'?\d{1,3})?\b")
EVIDENCE_TOKEN = re.compile(r"\[[^\]]*\]|\b\S+\|\S+\|\S+\|\S+\b")


#: Questions about the assistant itself rather than the data. Answered from
#: a fixed text, before any model runs: "what can you do" deserves a reply in
#: under a second, and a model has no way of knowing the answer anyway. The
#: list is deliberately tight -- "what can you tell me about AR1-AR2" and
#: "what do you know about the top channel" are questions about the data.
ABOUT = re.compile(
    r"^\s*(?:hi|hello|hey)?[\s,!.]*(?:"
    r"what can (?:you|i|this(?: assistant| tool)?) (?:do|ask(?: you)?|answer)|"
    r"what (?:do|can) you do|what are you|who are you|how do you work|"
    r"what (?:kinds? of |sort of )?questions? can i ask|"
    r"what are your (?:capabilit\w*|limits|rules)|"
    r"what (?:can't|cannot|won't) you (?:do|answer)|"
    r"help|\?)[\s?!.]*$",
    re.IGNORECASE)


def about_the_assistant(question: str) -> bool:
    """True for a question about what the assistant is and does."""
    return bool(ABOUT.match(question or ""))


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    problems: list[str] = field(default_factory=list)


def check_question(question: str, subject: str) -> GuardResult:
    """Refuse out-of-scope questions before the model sees them."""
    if DIRECTIVE.search(question):
        return GuardResult(False,
                           "I do not answer questions about treatment or what should be done. "
                           "I can show which channels carry the most detected activity and the "
                           "signal windows behind that; the clinical decision is the team's.")
    if DIAGNOSIS.search(question):
        return GuardResult(False,
                           "I cannot diagnose, give a prognosis, or say where seizures start. "
                           "A high event rate is a measurement, not a seizure-onset zone. I can "
                           "tell you what the detectors measured and where they disagree.")
    for match in OTHER_SUBJECT.finditer(question):
        token = match.group(0).lower().replace(" ", "")
        if token.replace("patient", "sub-pt").replace("pt", "pt") not in subject.lower() \
                and token not in subject.lower():
            return GuardResult(False,
                               f"I can only see the saved analysis of {subject}. Open that "
                               "patient's results to ask about their channels.")
    return GuardResult(True)


def collect_numbers(value, out: set[float] | None = None) -> set[float]:
    """Every number that appeared anywhere in the tool results (recursively)."""
    out = set() if out is None else out
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.add(round(float(value), 4))
        return out
    if isinstance(value, str):
        for token in re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?", value):
            out.add(round(float(token), 4))
        return out
    if isinstance(value, dict):
        for key, v in value.items():
            # A number in a key is in the result too: "rate_ci_95" is how a
            # tool says 95 %, and an answer that says "95 % interval" is
            # copying it, not inventing it.
            collect_numbers(str(key), out)
            collect_numbers(v, out)
        return out
    if isinstance(value, (list, tuple, set)):
        for v in value:
            collect_numbers(v, out)
    return out


def _is_traceable(number: float, seen: set[float]) -> bool:
    """A number is traceable if a tool reported it, allowing for rounding."""
    for value in seen:
        if abs(value - number) <= max(0.05, 0.01 * abs(value)):
            return True
        # Models legitimately round 46.0 -> 46 and 12.345 -> 12.3
        if abs(round(value, 1) - number) <= 0.051 or abs(round(value) - number) < 1e-9:
            return True
    return False


def verify_answer(text: str, evidence_ids: list[str], retrieved: set[str],
                  numbers_seen: set[float], store=None) -> GuardResult:
    """Check citations and numbers in a finished answer.

    Parameters
    ----------
    text:
        The answer the model produced.
    evidence_ids:
        The ids it cited.
    retrieved:
        Ids that tool calls actually returned during this conversation.
    numbers_seen:
        Every number any tool returned during this conversation.
    store:
        Optional store, used to confirm an id exists at all (a stronger check
        than "the model echoed something we sent it").
    """
    problems: list[str] = []
    for eid in evidence_ids:
        if eid not in retrieved:
            problems.append(f"citation {eid!r} was never returned by a tool")
        elif store is not None and store.resolve(eid) is None:
            problems.append(f"citation {eid!r} does not exist in the saved results")

    stripped = EVIDENCE_TOKEN.sub(" ", text)
    stripped = CHANNEL_TOKEN.sub(" ", stripped)
    flagged: set[str] = set()
    for raw, unit in NUMBER_WITH_UNIT.findall(stripped):
        number = round(float(raw), 4)
        if not _is_traceable(number, numbers_seen):
            flagged.add(raw)
            problems.append(f"the value '{raw} {unit.strip()}' does not appear in any tool result")
    for raw in BARE_NUMBER.findall(stripped):
        number = round(float(raw), 4)
        if raw in flagged or _is_traceable(number, numbers_seen):
            continue
        # A small whole number with no unit is counting something the model
        # can legitimately count ("the top 3 channels", "two detectors").
        if float(raw).is_integer() and 0 <= float(raw) <= MAX_FREE_INTEGER:
            continue
        problems.append(f"the value '{raw}' does not appear in any tool result")
    return GuardResult(not problems, "", problems)
