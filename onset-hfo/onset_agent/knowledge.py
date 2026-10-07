"""Background questions, answered from the project's own documents.

"What is an HFO?" is not a question about this window, and the guards that
make the assistant trustworthy -- every number traced to a query over the
analysis -- have nothing to check it against. Letting a small model answer
from memory would put unverifiable sentences on the same screen as verified
ones. So a background question is answered from the documents this project
ships and vouches for: the glossary, the methods, the clinician's guide, the
limitations and the evaluation. The few sections that match the question are
retrieved, handed to the model with the instruction to answer from them
alone, cited back by name, and the numbers in the answer are checked against
the passages the way data answers are checked against queries.

Qt-free and dependency-free: the retrieval is term overlap weighted by how
rare a term is across sections, which over a few hundred sections of prose
finds the glossary entry for "ripple" and the methods section for "notch"
without an embedding model.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Section", "docs_dir", "load_sections", "split_document", "retrieve",
           "title_match", "kind_of", "looks_medical", "numbers_in", "plain", "glossary",
           "DOCUMENTS"]

#: The documents searched, in the order they are listed as sources. Not the
#: install page or the agent's own page: a question about installing is not
#: a background question about the field.
DOCUMENTS = ("GLOSSARY", "METHODS", "CLINICAL_GUIDE", "LIMITATIONS", "EVALUATION")
#: Passages handed to the model: a few, and short, because a CPU reads them.
MAX_PASSAGES = 3
MAX_CHARS = 2500
#: A section longer than this is split at paragraph breaks.
CHUNK_CHARS = 1200


@dataclass
class Section:
    doc: str
    title: str
    text: str
    terms: dict[str, int] = field(default_factory=dict)
    #: Terms the section is also about: the words it sets in bold or italics
    #: ("**Ripple**: 80-250 Hz" inside the HFO entry), so a question about a
    #: ripple finds the entry that defines it.
    aliases: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.doc}, {self.title}"

    @property
    def title_terms(self) -> dict[str, int]:
        return {**_terms(self.title), **self.aliases}


# --------------------------------------------------------------------------
# Where the documents are
# --------------------------------------------------------------------------

def docs_dir() -> Path | None:
    """The docs folder: ``ONSET_DOCS_DIR``, the checkout, or the installed copy."""
    named = os.environ.get("ONSET_DOCS_DIR")
    candidates = [Path(named).expanduser()] if named else []
    try:
        from onset_hfo.config import PROJECT_ROOT

        candidates.append(Path(PROJECT_ROOT) / "docs")
    except Exception:       # noqa: BLE001 - no checkout, no harm
        pass
    candidates.append(Path.home() / ".local" / "share" / "onset-review" / "docs")
    for root in candidates:
        if (root / "GLOSSARY.md").exists():
            return root
    return None


# --------------------------------------------------------------------------
# Splitting the documents into sections
# --------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*$")
_BOLD_LEAD = re.compile(r"^\*\*([^*]+)\*\*")
_STOP = set("""a an and are as at be been by can do does for from has have how in into is it its
of on or that the this to was were what when which who why will with you your not no than then
there these those their they them we our us i me my""".split())
_WORD = re.compile(r"[a-z][a-z0-9'-]+|\d+(?:\.\d+)?")


def _terms(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in _WORD.findall(text.lower()):
        word = word.strip("'-")
        if len(word) < 2 or word in _STOP:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]                      # ripples -> ripple, a plain plural
        counts[word] = counts.get(word, 0) + 1
    return counts


def _strip_markup(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links -> their text
    text = re.sub(r"^\s*\|.*\|\s*$", lambda m: m.group(0).replace("|", " "), text, flags=re.M)
    text = re.sub(r"^\s*[-:| ]+\s*$", "", text, flags=re.M)      # table rules
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def _chunks(body: str) -> list[tuple[str | None, str]]:
    """Paragraph-bounded pieces of at most CHUNK_CHARS, each with the bold
    term it opens with when it has one (the glossary's entries)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[tuple[str | None, str]] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        # A paragraph that opens with a bold term is an entry of its own (the
        # glossary), so "ripple" finds the entry titled Ripple and not the
        # entry three above it.
        if current and (size + len(paragraph) > CHUNK_CHARS or _lead(paragraph)):
            out.append((_lead(current[0]), "\n\n".join(current)))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        out.append((_lead(current[0]), "\n\n".join(current)))
    return out


def _lead(paragraph: str) -> str | None:
    match = _BOLD_LEAD.match(paragraph)
    return match.group(1).strip() if match else None


def load_sections(root: Path | None = None, documents: tuple = DOCUMENTS) -> list[Section]:
    """Every document split at its headings, long sections at paragraphs."""
    root = root or docs_dir()
    if root is None:
        return []
    sections: list[Section] = []
    for name in documents:
        path = Path(root) / f"{name}.md"
        if not path.exists():
            continue
        sections.extend(_split_document(name, _strip_markup(path.read_text(encoding="utf-8"))))
    return sections


def split_document(name: str, text: str) -> list[Section]:
    """A document's sections, for a text that is not one of the shipped
    documents: a study page built in the window, say."""
    return _split_document(name, text)


#: Glossary entries that are code names rather than terms a page uses.
_NOT_TERMS = {"glossary", "recording", "prepared", "event", "evidence_id", "resultstore",
              "scripted backend", "hot contact", "channel", "contact"}


def glossary(sections: list[Section] | None = None) -> dict[str, str]:
    """{term: one-sentence definition} from the shipped glossary, the term
    being the entry's name before any bracketed expansion, for a tooltip on
    the term where it appears on a page. "Channel" and "Contact" are left
    out: on these pages they are not jargon, they are the furniture."""
    out: dict[str, str] = {}
    for section in sections if sections is not None else load_sections():
        if section.doc != "GLOSSARY":
            continue
        title = section.title.strip()
        term = re.split(r"\s*[(/]", title, maxsplit=1)[0].strip().strip('"')
        if len(term) < 3 or term.lower() in _NOT_TERMS:
            continue
        definition = plain(section.text).strip()
        if definition.lower().startswith(title.lower()):
            definition = definition[len(title):].lstrip(" —-:")
        out[term] = definition[:320]
        inside = re.search(r"\(([^)]+)\)", title)
        if inside and len(inside.group(1)) >= 3 and " " not in inside.group(1).strip():
            out.setdefault(inside.group(1).strip(), out[term])
    return out


def _split_document(name: str, text: str) -> list[Section]:
    sections: list[Section] = []
    heading = name.replace("_", " ").title()
    body: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.extend(_sections_of(name, heading, body))
            heading = match.group(2).strip()
            body = []
            continue
        body.append(line)
    sections.extend(_sections_of(name, heading, body))
    return sections


_SKIP_TITLES = re.compile(r"^(references|bibliography|citation|further reading|see also)\b",
                          re.IGNORECASE)


def _sections_of(name: str, title: str, body: list[str]) -> list[Section]:
    joined = "\n".join(body).strip()
    if not joined or _SKIP_TITLES.match(title):      # a bibliography answers nothing
        return []
    return [Section(name, lead or title, chunk, _terms(f"{lead or title} {chunk}"),
                    _terms(" ".join(a or b for a, b in _EMPHASIS.findall(chunk))))
            for lead, chunk in _chunks(joined)]


_EMPHASIS = re.compile(r"\*\*([^*\n]{2,40})\*\*|(?<!\*)\*([^*\n]{2,30})\*(?!\*)")


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve(question: str, sections: list[Section], k: int = MAX_PASSAGES,
             max_chars: int = MAX_CHARS) -> list[Section]:
    """The sections that match the question best, by BM25 over their terms,
    with a section whose title is the thing asked about ranked first and
    the glossary preferred for a definition. Empty when the best match only
    shares common words with the question."""
    if not sections:
        return []
    asked = _terms(question)
    if not asked:
        return []
    n = len(sections)
    frequency: dict[str, int] = {}
    for section in sections:
        for term in section.terms:
            frequency[term] = frequency.get(term, 0) + 1
    average = sum(sum(s.terms.values()) for s in sections) / n
    definition = bool(_DEFINITION.match(question))
    scored = []
    for section in sections:
        length = sum(section.terms.values()) or 1
        title_terms = section.title_terms
        score = 0.0
        for term in asked:
            tf = section.terms.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * (tf * (_K1 + 1.0)) / (tf + _K1 * (1.0 - _B + _B * length / average))
            if term in title_terms:
                score += idf
        if score <= 0:
            continue
        # The section *about* the thing asked: every content term of the
        # question is in its title. A definition goes to the glossary first.
        if all(term in title_terms for term in asked):
            score *= 2.0
        if definition and section.doc == "GLOSSARY":
            score *= 2.0
        scored.append((score, section))
    scored.sort(key=lambda pair: -pair[0])
    if not scored or not _covers(asked, scored[0][1], frequency, n):
        return []
    chosen, used = [], 0
    for _score, section in scored[: max(1, int(k))]:
        if used and used + len(section.text) > max_chars:
            break
        chosen.append(section)
        used += len(section.text)
    return chosen


_K1, _B = 1.2, 0.75
_DEFINITION = re.compile(r"^\s*(what (is|are|does .* mean)|define|explain what|meaning of)",
                         re.IGNORECASE)


def title_match(question: str, section: Section) -> bool:
    """Whether the section's title names something the question asks about."""
    asked = _terms(question)
    return any(t in section.title_terms for t in asked)


def _covers(asked: dict[str, int], section: Section, frequency: dict[str, int], n: int) -> bool:
    """Whether the best match is about the question rather than merely
    sharing common words with it: a question term in its title, or a
    matched term rare enough across the documents to mean something."""
    matched = [t for t in asked if t in section.terms]
    if not matched:
        return False
    if any(t in section.title_terms for t in matched):
        return True
    return any(frequency.get(t, n) <= max(1, n // 8) for t in matched)


# --------------------------------------------------------------------------
# What kind of question this is
# --------------------------------------------------------------------------

_DATA = re.compile(
    r"\b(this|the|that) (window|event|recording|patient|analysis|channel|contact|minute|"
    r"ranking|report|table|detector)s?\b|selected|busiest|\bleader|\bleading|\brank|"
    r"\brates?\b|events?/min|per min|threshold|stricter|surviv|\bdetectors?\b|\brms\b|"
    r"line.length|hilbert|evidence|\bcite|\btied\b|resect|other window|another window|"
    r"\bearlier\b|\blater\b|between the|how many|which channel|which contact|stand out|"
    r"agree|disagree|analys[ei]d|analyzed|sampling rate|notch(ed)? at|montage (used|here)|"
    r"\b(was|were) (analys|analyz|used|detected|rejected|excluded|found|removed|dropped|"
    r"left|set|marked|flagged|the)|\bdid (you|it|the|any|they)\b|you (find|detect|measure|"
    r"analy)|rejected|excluded|\bchannels?\b|\bcontacts?\b|\bevents?\b|\bwindows?\b|"
    r"\brecording\b|\bsubject\b|"
    r"limitation|caveat|how did you|\bmethods?\b|quality check|set aside|flagged|"
    r"\bseizure\b.*\b(window|here)\b|\bhere\b",
    re.IGNORECASE)
_BACKGROUND = re.compile(
    r"^\s*(what|why|how|when|where|who|explain|define|describe|tell me about|is|are|does|do|"
    r"can|could|should)\b", re.IGNORECASE)
_MEDICAL = re.compile(
    r"\b(patient|treat|therap|surg|medic|drug|dose|dosage|prognos|diagnos|symptom|"
    r"should (i|we|one)|my (son|daughter|mother|father|wife|husband|child)|cure|remission|"
    r"prescri)", re.IGNORECASE)


def kind_of(question: str) -> str:
    """``"data"`` (about this analysis: the guarded path) or ``"background"``
    (a question, with nothing of the data in it, that the documents may
    answer; the agent calls it general when they do not). Anything else is
    data, an imperative included: that path's failure is a refusal, which
    is the safe one."""
    from onset_agent.guard import CHANNEL_TOKEN

    text = question or ""
    if CHANNEL_TOKEN.search(text) or _DATA.search(text):
        return "data"
    if _BACKGROUND.search(text) and _terms(text):
        return "background"
    return "data"


def looks_medical(question: str) -> bool:
    """A general question that is really a medical one: refused, not answered
    from a model's memory."""
    return bool(_MEDICAL.search(question or ""))


_NUMBER = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w.])")


def plain(text: str) -> str:
    """The passage without its markdown emphasis, for showing verbatim."""
    return re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text or "")


def numbers_in(text: str) -> set[float]:
    out: set[float] = set()
    for token in _NUMBER.findall(text or ""):
        try:
            out.add(float(token))
        except ValueError:
            continue
    return out
