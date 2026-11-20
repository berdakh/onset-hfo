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
           "about_the_assistant", "numbers_in", "traced_to"]

#: The same request in the words a surgeon uses for it: "cut out", "come
#: out", "take out", "excise" all went past a list that knew only "resect"
#: and "remove".
DIRECTIVE = re.compile(
    r"\b(resect\w*|ablat\w*|remov\w*|excis\w*|operat\w*|surg\w*|implant\w*|"
    r"cut(?:ting)?\s+(?:out|away)|(?:come|comes|coming|take|taken|taking)\s+out|"
    r"treat\w*|therap\w*|medicat\w*|drug|dose|dosage|"
    r"should (?:we|i|they|he|she|come|go|be)|would you (?:cut|take|resect|remove)|"
    r"recommend\w*|advise|advice|what would you do)\b",
    re.IGNORECASE)

DIAGNOSIS = re.compile(
    r"\b(diagnos\w*|prognos\w*|outcome|cure[d]?|seizure[- ]?free|"
    r"does (?:this |the )?patient have|is (?:this|the) patient|"
    r"where (?:do|does) (?:the )?seizures? (?:start|begin|originate|come from)|"
    r"seizure onset zone|onset zone|soz|epileptogenic(?: zone)?|"
    r"(?:epilep\w+|seizure) (?:focus|foci|generator|source)|"
    r"(?:source|origin|generator|focus) of (?:the |this patient's |his |her )?"
    r"(?:seizures?|epilepsy))\b",
    re.IGNORECASE)

#: What an *answer* must not say, whatever the question was. The gate on the
#: question is the first line and can be talked around; this one reads the
#: sentence that would be shown. Narrower than the lists above on purpose: an
#: answer may say "this is a rate, not a seizure-onset zone", and may not say
#: that a channel is one or that anything should be done to it.
_DONE_TO = r"(?:resect\w*|remov\w*|ablat\w*|excis\w*|cut out|taken out|operated on|spared|left in)"
_PLACE = (r"(?:seizure[- ]onset zone|onset zone|soz|epileptogenic zone|"
          r"epilep\w+ (?:focus|zone|generator)|seizure (?:focus|generator|source)|"
          r"(?:source|origin|focus) of (?:the |this patient's )?(?:seizures?|epilepsy))")
ANSWER_DIRECTIVE = re.compile(
    rf"\b(?:(?:should|must|ought to|needs? to|has to|have to)(?: not| also| probably| likely)? "
    rf"(?:be |get )?{_DONE_TO}|"
    rf"(?:i |we )?(?:recommend|advise|suggest)\w*\b[^.]{{0,60}}?\b{_DONE_TO}|"
    rf"(?:candidates?|targets?) for (?:resection|removal|ablation|surgery|excision))\b",
    re.IGNORECASE)
ANSWER_DIAGNOSIS = re.compile(
    rf"\b(?:is|are|as|marks?|indicates?|identifies|locali[sz]es|points? to|lies in|"
    rf"corresponds? to)\s+(?:most likely |likely |probably |clearly |therefore |thus )?"
    rf"(?:the |a |an |this patient's )?{_PLACE}\b",
    re.IGNORECASE)

#: Another patient named in a question: a BIDS subject label (letters and
#: digits, at least one digit, so "sub-band" is not a patient), "patient 3"
#: or "pt3". It once required letters after "sub-", so the numeric labels of
#: ds003498 ("sub-02") went straight past it to the model.
OTHER_SUBJECT = re.compile(r"\b(sub-(?=[a-z0-9]*\d)[a-z0-9]{1,16}|patient\s+\d{1,3}|pt\s?\d{1,3})\b",
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


def _same_subject(named: str, subject: str) -> bool:
    """Whether a subject named in a question is the one whose analysis is open.
    A full label must match exactly ("sub-01" is not "sub-012"); "patient 1"
    and "pt1" are compared by number, so they match sub-pt01 and sub-01."""
    token = named.lower().replace(" ", "")
    if token.startswith("sub-"):
        return token == subject.lower()
    digits = re.findall(r"\d+", subject)
    return bool(digits) and int(re.findall(r"\d+", token)[0]) == int(digits[-1])


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
        if not _same_subject(match.group(0), subject):
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


def numbers_in(text: str) -> list[tuple[str, str]]:
    """The numbers in a text that the check looks at, as (number, unit) pairs
    in order, with "" for a bare number. Evidence ids and channel names are
    removed first, as in :func:`verify_answer`, because their digits are not
    measurements."""
    stripped = CHANNEL_TOKEN.sub(" ", EVIDENCE_TOKEN.sub(" ", text or ""))
    found: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for match in NUMBER_WITH_UNIT.finditer(stripped):
        found.append((match.start(), match.group(1), match.group(2).strip()))
        taken.append(match.span(1))
    for match in BARE_NUMBER.finditer(stripped):
        if any(start <= match.start() < stop for start, stop in taken):
            continue
        found.append((match.start(), match.group(1), ""))
    return [(raw, unit) for _, raw, unit in sorted(found)]


def traced_to(number: float, sources: dict) -> list[str]:
    """The labels of the tool calls a number can be traced to, by the same
    rounding rule the check uses; empty when none."""
    for value, labels in sources.items():
        if _is_traceable(float(number), {float(value)}):
            return list(labels)
    return []


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

    # What the answer says should be done, or says a channel *is*. No tool
    # returns either, so no amount of evidence makes one traceable.
    said = ANSWER_DIRECTIVE.search(text or "")
    if said:
        problems.append(f"the answer says what should be done ({said.group(0)!r}); "
                        "this software reports measurements and recommends nothing")
    said = ANSWER_DIAGNOSIS.search(text or "")
    if said:
        problems.append(f"the answer names a seizure-onset zone or focus ({said.group(0)!r}); "
                        "a rate is a measurement, not a localisation")

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
