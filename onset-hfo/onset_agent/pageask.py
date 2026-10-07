"""Questions about one page of text, answered from that page and checked.

A study page in the window is a document built from the committed tables:
prose, figures and tables about the detectors, the outcome study, the
patients. A reader new to the field wants to ask it things -- "summarise
this", "what does channel-rank ρ mean here?" -- and the local model can
answer, under the same discipline the evidence assistant keeps for the
project's documents:

* the model sees the page's own sections, retrieved for the question, and
  the page's key message; nothing else, so it cannot answer from memory
  about a patient it was never shown;
* every number in its answer must appear on the page, allowing for
  rounding, or the answer is refused and shown as refused;
* with no model loaded the page still answers: the key message for a
  summary, the best-matching section for anything else, verbatim.

Nothing here reaches the analysis, the store or the saved read; a page is
text, and this module reads text.
"""

from __future__ import annotations

import re

from onset_agent import guard, knowledge
from onset_agent.agent import AgentAnswer
from onset_agent.backends import Interrupted, extract_json_object
from onset_agent.prompts import BACKGROUND_PROMPT

__all__ = ["answer", "is_summary_question", "SUGGESTIONS"]

#: What the box under a page offers to ask.
SUGGESTIONS = ("Summarise this page in three sentences.",
               "What is the main result, and what limits it?",
               "Which numbers here should I be careful with?")

_SUMMARY = re.compile(
    r"\b(summar\w*|main (message|result|point|finding)|what does this page|in short|"
    r"tl;?dr|explain this page|key (message|point|result)|gist)", re.IGNORECASE)

#: Asking the page for clinical advice, as opposed to asking what it says
#: about patients: "patient" alone is the subject of every study page.
_ADVICE = re.compile(
    r"\b(should (i|we|my patient|he|she|they)|treat(ment)? (for|of) (my|me|him|her)|"
    r"dos(e|age)|prescri\w*|diagnos\w* (my|me|him|her)|is it safe|"
    r"my (patient|child|son|daughter|mother|father|wife|husband|partner))\b",
    re.IGNORECASE)

#: Passages handed to the model at most, by text length.
MAX_CHARS = 3000


def is_summary_question(question: str) -> bool:
    return bool(_SUMMARY.search(question or ""))


def _sections(title: str, text: str) -> list[knowledge.Section]:
    return knowledge.split_document(title, text)


def answer(backend, question: str, title: str, text: str,
           key_message: str = "") -> AgentAnswer:
    """Answer `question` about the page `title` whose Markdown is `text`.

    `key_message` is the page's own plain-language lead, which is always the
    first passage the model sees and the whole answer when no model is loaded
    and the question asks for a summary.
    """
    question = (question or "").strip()
    name = getattr(backend, "name", "") or ""
    if not question:
        return AgentAnswer(question=question, text="Ask something about this page.",
                           refused=True, reason="empty question", mode="background",
                           backend=name, verified=False)
    if _ADVICE.search(question):
        return AgentAnswer(question=question, refused=True, mode="background",
                           text="That asks for clinical advice, and this page cannot vouch for "
                                "any. Ask the clinical team.",
                           reason="a request for clinical advice", backend=name, verified=False)

    sections = _sections(title, text)
    summary = is_summary_question(question)
    passages = knowledge.retrieve(question, sections) if sections else []
    if summary or not passages:
        # The lead and the first sections of the page: what a summary is of.
        passages = [s for s in sections if s.text.strip()][:3]
    head = [(f"{title}, key message", key_message)] if key_message else []
    sources = head + [(p.label, p.text) for p in passages]

    if not getattr(backend, "is_language_model", True):
        if summary and key_message:
            return AgentAnswer(question=question, mode="background", sources=sources,
                               text=key_message, verified=True, backend=name,
                               reason="the page's key message; no model is loaded")
        if passages:
            top = passages[0]
            return AgentAnswer(question=question, mode="background", sources=sources,
                               text=f"From {top.label}: {knowledge.plain(top.text)}",
                               verified=True, backend=name,
                               reason="the closest section of the page; no model is loaded")
        return AgentAnswer(question=question, mode="background", refused=True,
                           text="Nothing on this page covers that, and no model is loaded.",
                           reason="not covered; no model", backend=name, verified=False)

    body, used = [], 0
    for label, passage in sources:
        if used + len(passage) > MAX_CHARS and body:
            break
        body.append(f"[{label}]\n{passage}")
        used += len(passage)
    messages = [{"role": "system", "content": BACKGROUND_PROMPT},
                {"role": "user", "content": "Passages:\n\n" + "\n\n".join(body)
                                            + f"\n\nQuestion: {question}"}]
    try:
        message = backend.chat(messages, [])
    except Interrupted:
        return AgentAnswer(question=question, text="Stopped before an answer was produced.",
                           refused=True, reason="stopped by the reader", mode="background",
                           sources=sources, backend=name, verified=False)
    content = (message.content or "").strip()
    parsed = extract_json_object(content)
    if isinstance(parsed, dict):
        content = str(parsed.get("refusal") or parsed.get("answer") or content)
    if not content:
        return AgentAnswer(question=question, text="The model returned nothing.",
                           refused=True, reason="empty answer", mode="background",
                           sources=sources, backend=name, verified=False)
    stated = knowledge.numbers_in(" ".join(passage for _label, passage in sources))
    loose = sorted(n for n in knowledge.numbers_in(content)
                   if not guard._is_traceable(n, stated))
    if loose:
        return AgentAnswer(question=question, refused=True, mode="background",
                           sources=sources, text=content, backend=name, verified=False,
                           reason="the answer states " + ", ".join(f"{n:g}" for n in loose)
                                  + ", which this page does not")
    return AgentAnswer(question=question, mode="background", sources=sources,
                       text=content, verified=True, backend=name,
                       reason="answered from this page; every number traced to it")
