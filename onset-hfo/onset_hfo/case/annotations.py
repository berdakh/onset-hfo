"""What can be marked on a recording, and how a clinical system's marks map
onto it. Qt-free.

A clinical system's event list is free text typed by whoever was on shift:
"Sz", "sz onset ?", "Pt pushed button", "lights off", "artifact - chewing".
The analysis needs a small fixed vocabulary -- a seizure must be a seizure
whatever it was called -- so each mark gets one of `KINDS`, and its
original text is kept beside it (`value`) so nothing typed is lost and the
mapping can be checked. Anything not recognised is a `note`.

Free text can hold a name ("called Mrs ... at 14:00"): a bridge keeps the
text, and the conversion report says how many free-text marks there are so
someone looks at them before the case leaves the building.
"""

from __future__ import annotations

import re
import time

import pandas as pd

__all__ = ["KINDS", "SLEEP_STAGES", "kind_of", "map_marks", "new_mark", "label_of",
           "SEIZURE_KINDS"]

#: kind -> what it means. Durations: a seizure spans onset to offset; a
#: sleep stage spans its epoch; the onset markers are instants.
KINDS = {
    "seizure": "a seizure, electrographic onset to offset",
    "seizure-onset": "the moment a seizure starts, electrographically",
    "seizure-clinical-onset": "the first clinical sign of a seizure",
    "seizure-offset": "the moment a seizure ends",
    "artefact": "signal not to analyse: movement, chewing, disconnection",
    "sleep-W": "awake",
    "sleep-N1": "non-REM sleep, stage 1",
    "sleep-N2": "non-REM sleep, stage 2",
    "sleep-N3": "non-REM sleep, stage 3 (slow-wave)",
    "sleep-REM": "REM sleep",
    "stimulation": "electrical stimulation in progress",
    "button": "the patient's or a nurse's event button",
    "medication": "a medication given or changed",
    "note": "anything else; the original text is kept",
}
SLEEP_STAGES = ("sleep-W", "sleep-N1", "sleep-N2", "sleep-N3", "sleep-REM")
SEIZURE_KINDS = ("seizure", "seizure-onset", "seizure-clinical-onset", "seizure-offset")

_RULES = (
    # (pattern, kind), first match wins; matched on lower-cased text
    (r"\b(clin(ical)?\.?\s*onset|first clin)", "seizure-clinical-onset"),
    (r"\b(eo|eeg onset|electrographic onset|sz onset|seizure onset|onset)\b",
     "seizure-onset"),
    (r"\b(sz end|seizure end|offset|sz off|end of (sz|seizure))\b", "seizure-offset"),
    (r"\b(sz|seizure|ictal|crisis|crise|anfall)\b", "seizure"),
    (r"\b(artifact|artefact|movement|chewing|electrode pop|disconnect|noise)\b", "artefact"),
    (r"\b(rem)\b", "sleep-REM"),
    (r"\b(n3|sws|stage 3|slow wave)\b", "sleep-N3"),
    (r"\b(n2|stage 2)\b", "sleep-N2"),
    (r"\b(n1|stage 1)\b", "sleep-N1"),
    (r"\b(awake|wake|lights on)\b", "sleep-W"),
    (r"\b(stim|stimulation|spes|ccep)\b", "stimulation"),
    (r"\b(button|push|pushed|event)\b", "button"),
    (r"\b(medication|lorazepam|midazolam|diazepam|levetiracetam|dose|mg)\b", "medication"),
)


def kind_of(text: str) -> str:
    """The kind a clinical system's mark most plausibly is; `note` when unsure.
    A kind's own name maps to itself."""
    raw = str(text or "").strip()
    if raw in KINDS:
        return raw
    lower = raw.lower()
    for pattern, kind in _RULES:
        if re.search(pattern, lower):
            return kind
    return "note"


def label_of(kind: str) -> str:
    return KINDS.get(kind, kind)


def new_mark(onset: float, kind: str, duration: float = 0.0, channels=(), value: str = "",
             source: str = "reader", by: str = "") -> dict:
    """One row of events.tsv."""
    if kind not in KINDS:
        raise ValueError(f"not a kind of mark: {kind!r}")
    if not (onset >= 0) or not (duration >= 0):
        raise ValueError("a mark starts at or after the recording's start and lasts 0 s or more")
    return {"onset": float(onset), "duration": float(duration), "trial_type": kind,
            "channels": ",".join(channels) if channels else "n/a",
            "value": value or "n/a", "source": source, "by": by or "n/a",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def map_marks(annotations, first_time: float = 0.0) -> pd.DataFrame:
    """A clinical file's annotations (MNE `Annotations`) as events.tsv rows:
    onset from the recording's first sample, the kind they map to, the
    original text kept, source "file". BrainVision's own segment markers
    and empty texts are left out."""
    rows = []
    for item in annotations:
        text = str(item["description"]).strip()
        if not text or text.lower() in ("new segment", "bad_acq_skip", "edge"):
            continue
        onset = float(item["onset"]) - float(first_time)
        if onset < 0:
            continue
        channels = tuple(item.get("ch_names", ()) or ())
        rows.append(new_mark(onset, kind_of(text), float(item["duration"]), channels,
                             value=text, source="file", by="clinical system"))
    from onset_hfo.case.bids import EVENT_COLUMNS

    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
