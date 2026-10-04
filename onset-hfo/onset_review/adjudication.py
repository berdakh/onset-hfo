"""What the reader decided, kept separately from what the detector decided.

Every other module here reports an algorithm's opinion. This one records a
person's, and the distinction is the whole point: a review in which the reader
cannot say *I looked at this and I disagree* is not a review, it is a reading.
Three things follow from taking that seriously, and they are the reasons this
file is shaped the way it is.

**A judgement is attributed and timed.** Each verdict carries the name of the
person who gave it and the moment they gave it. An anonymous clinical opinion
is worse than none, because it cannot be questioned later by the person who
holds it.

**A judgement outlives the analysis that prompted it.** Change the notch
filter and every event object in memory is replaced -- but a reader's opinion
that there is a real oscillation on AD1-AD2 at 12.34 s is about the *signal*,
not about the detector run. So verdicts are stored against the recording and
the window, never against the detector settings, and :func:`reconcile` matches
them back onto whatever the detector produces next.

**A judgement is never silently discarded.** When re-analysis leaves a verdict
with no event to attach to, it becomes an *orphan*: kept in the file, counted
in the summary, and named in the report. Deleting a person's recorded opinion
because the software's opinion changed is the one behaviour this module must
not have.

The verdicts are deliberately coarse. Three states, not a confidence slider:
a reader looking at four hundred events will use three buttons and will not
use a slider, and a number nobody means is worse than a category they do.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = [
    "EVENT_VERDICTS", "CHANNEL_VERDICTS", "EVENT_LABELS", "CHANNEL_LABELS",
    "MATCH_TOLERANCE_S", "Judgement", "Adjudication",
    "event_key", "window_id", "reader_name", "store_dir", "path_for",
    "load", "save", "reconcile",
]

#: What a reader may say about one detected event. "unsure" is not a failure
#: to decide -- it is the honest state for an event that is genuinely
#: ambiguous, and recording it is what keeps those events out of both the
#: confirmed and the rejected count.
EVENT_VERDICTS = ("agree", "disagree", "unsure")

#: What a reader may say about a whole contact. Separate from the event
#: verdicts because it answers a different question: not "is this one event
#: real" but "is this contact's count worth anything at all".
CHANNEL_VERDICTS = ("accept", "ignore", "unsure")

#: Wording for the interface and the report. The verdicts are stored as the
#: short keys above so that a file written today still reads in a year.
EVENT_LABELS = {
    "agree": "A real event",
    "disagree": "Not a real event",
    "unsure": "Cannot tell",
}
CHANNEL_LABELS = {
    "accept": "Count this contact",
    "ignore": "Ignore this contact",
    "unsure": "Cannot tell",
}

#: How far an event may move before a stored verdict stops following it, in
#: seconds. Re-running with a different threshold shifts an event's onset by a
#: few milliseconds, because the onset is where the feature crossed the
#: threshold; it does not move it to a different oscillation. Ripples run
#: 20-100 ms, so 50 ms is inside "the same event" and well short of the next
#: one. A match made by proximity rather than exactly is marked as such, so a
#: report can say how many verdicts were carried rather than re-given.
MATCH_TOLERANCE_S = 0.05

#: Schema version, written into every file. If the vocabulary above ever
#: changes, this is what tells a reader of an old file which vocabulary it is
#: in.
SCHEMA = 1


@dataclass(frozen=True)
class Judgement:
    """One recorded opinion: what was said, by whom, when, and about what."""

    verdict: str
    reader: str
    at: str                      #: ISO-8601 UTC, to the second
    note: str = ""
    #: "exact" or "near" once :func:`reconcile` has matched this to an event,
    #: "orphan" when no event in the current analysis corresponds to it, and
    #: "" in the file itself, where matching has not happened yet.
    match: str = ""

    def to_json(self) -> dict:
        out = {"verdict": self.verdict, "reader": self.reader, "at": self.at}
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_json(cls, data: dict) -> "Judgement":
        return cls(verdict=str(data.get("verdict", "unsure")),
                   reader=str(data.get("reader", "")),
                   at=str(data.get("at", "")),
                   note=str(data.get("note", "")))


def now() -> str:
    """The timestamp written onto a verdict. UTC, because a review may be read
    in a different time zone than the one it was given in."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def event_key(channel: str, start: float, detector: str = "") -> str:
    """The stable name of one event: channel, onset in file seconds, detector.

    File seconds rather than window seconds, so that a verdict survives
    reviewing the same event from a window that starts somewhere else -- the
    same reason :class:`onset_hfo.detectors.base.Event` quotes file seconds.
    Three decimals is a millisecond, which is finer than any detector's onset
    is meaningful to.
    """
    return f"{channel}|{float(start):.3f}|{detector}"


def _split_key(key: str) -> tuple[str, float, str]:
    channel, _, rest = key.partition("|")
    start, _, detector = rest.partition("|")
    try:
        return channel, float(start), detector
    except ValueError:
        return channel, float("nan"), detector


def reader_name(default: str | None = None) -> str:
    """A sensible starting value for the reader's name, never a final one.

    The operating-system user name is a reasonable *prefill* for the box that
    asks who is reviewing. It is not an identity: `root` in a container and
    the clinician at the keyboard are not the same person, and a report that
    quietly attributes a judgement to a login name is worse than one with a
    blank where the name should be. So this only suggests, and the caller is
    responsible for having a person confirm it.
    """
    if default:
        return default
    for name in ("ONSET_REVIEWER", "USER", "USERNAME"):
        value = os.environ.get(name, "").strip()
        if value and value not in {"root", "nobody"}:
            return value
    return ""


# --------------------------------------------------------------------------
# Where a read is kept
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "unknown"


def window_id(request) -> str:
    """The identity a read is filed under: the signal, not the analysis.

    Dataset, subject, run and the window's bounds -- and deliberately *not*
    the band, the detector, the threshold or the preprocessing. A reader who
    says "that is a real ripple at 12.34 s on AD1-AD2" has said something
    about the recording. Re-running with a 4 SD threshold does not make it
    untrue, so it must not file it somewhere else.

    An imported local file has no accession, so it is identified by its name
    and its full path, hashed: two files called `export.edf` in different
    folders are different recordings and must not share a read.
    """
    if getattr(request, "path", None) is not None:
        import hashlib

        path = Path(request.path)
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        stem = f"file-{_slug(path.stem)}-{digest}"
    else:
        stem = (f"{_slug(request.dataset)}-{_slug(request.subject)}"
                f"-run{_slug(request.run)}")
    # The *span*, not the trace window. A verdict is about an event in the
    # analysed stretch; scrolling the trace to a different minute of that
    # stretch must not file it somewhere else, or a reviewer working a ten
    # minute span would leave ten separate reads behind.
    span = request.span() if hasattr(request, "span") else (request.t_start,
                                                            request.t_stop)
    return f"{stem}-{float(span[0]):g}-{float(span[1]):g}s"


def store_dir() -> Path:
    """Where reads live. Beside the project's other artifacts, overridable.

    `ONSET_HFO_HOME` already relocates the data cache and the results, and a
    read belongs with them: it is as much a product of this software as a
    detection table, and someone moving the one will expect the other to
    follow.
    """
    from onset_hfo.config import HOME

    return Path(os.environ.get("ONSET_REVIEW_READS", HOME / "reads"))


def path_for(request) -> Path:
    return store_dir() / f"{window_id(request)}.json"


# --------------------------------------------------------------------------
# The read itself
# --------------------------------------------------------------------------


@dataclass
class Adjudication:
    """One reader's verdicts on one window, and the bookkeeping around them."""

    window: str = ""
    reader: str = ""
    events: dict = field(default_factory=dict)      #: event key -> Judgement
    channels: dict = field(default_factory=dict)    #: channel  -> Judgement
    #: Verdicts with no event in the current analysis to attach to, set by
    #: `reconcile`. Held apart from `events` so that they are not counted as
    #: work done on a channel -- and saved with them, because deleting a
    #: person's recorded opinion because the software's opinion changed is the
    #: one behaviour this module must not have.
    orphaned: dict = field(default_factory=dict)
    #: Free text about the window as a whole, which is the thing a reader
    #: actually wants to write at the end of a read.
    note: str = ""

    # -- recording a verdict ----------------------------------------------
    def judge_event(self, key: str, verdict: str, reader: str = "",
                    note: str = "") -> "Judgement":
        if verdict not in EVENT_VERDICTS:
            raise ValueError(f"not an event verdict: {verdict!r}")
        return self._record(self.events, key, verdict, reader, note)

    def judge_channel(self, channel: str, verdict: str, reader: str = "",
                      note: str = "") -> "Judgement":
        if verdict not in CHANNEL_VERDICTS:
            raise ValueError(f"not a channel verdict: {verdict!r}")
        return self._record(self.channels, channel, verdict, reader, note)

    def _record(self, into: dict, key: str, verdict: str, reader: str,
                note: str) -> "Judgement":
        who = (reader or self.reader or "").strip()
        # A verdict keeps whatever note it already had unless a new one is
        # given: changing your mind about an event should not silently erase
        # the sentence saying why you thought the other thing.
        previous = into.get(key)
        if not note and previous is not None:
            note = previous.note
        judged = Judgement(verdict=verdict, reader=who, at=now(), note=note,
                           match=previous.match if previous else "")
        into[key] = judged
        if who and not self.reader:
            self.reader = who
        return judged

    def clear_event(self, key: str) -> None:
        """Take a verdict back. The *reader* may do this; software may not."""
        self.events.pop(key, None)

    def clear_channel(self, channel: str) -> None:
        self.channels.pop(channel, None)

    # -- reading it back ---------------------------------------------------
    def verdict_of(self, key: str) -> str:
        judged = self.events.get(key)
        return judged.verdict if judged else ""

    def channel_verdict(self, channel: str) -> str:
        judged = self.channels.get(channel)
        return judged.verdict if judged else ""

    @property
    def empty(self) -> bool:
        return (not self.events and not self.channels and not self.orphaned
                and not self.note.strip())

    @property
    def readers(self) -> list:
        """Everyone who has given a verdict in this file, in the order met.

        More than one is not an error -- a second opinion is a normal thing to
        want -- but it is something a report has to say out loud rather than
        collapsing into a single "Reviewer" line.
        """
        seen = []
        for judged in (list(self.events.values()) + list(self.channels.values())
                       + list(self.orphaned.values())):
            if judged.reader and judged.reader not in seen:
                seen.append(judged.reader)
        return seen

    def counts(self) -> dict:
        """How many of each verdict, over the whole window."""
        out = {verdict: 0 for verdict in EVENT_VERDICTS}
        for judged in self.events.values():
            if judged.verdict in out:
                out[judged.verdict] += 1
        out["judged"] = len(self.events)
        out["orphans"] = len(self.orphaned)
        return out

    def channel_counts(self, channel: str) -> dict:
        """The same, for one contact. `total` is filled in by `progress`."""
        out = {verdict: 0 for verdict in EVENT_VERDICTS}
        for key, judged in self.events.items():
            if _split_key(key)[0] == channel and judged.verdict in out:
                out[judged.verdict] += 1
        out["judged"] = sum(out[v] for v in EVENT_VERDICTS)
        return out

    def progress(self, events) -> dict:
        """Channel -> how much of it has been judged, and what was said.

        The `complete` flag is the one that matters downstream. A rate built
        from a partly-judged channel is not a confirmed rate, it is a confirmed
        count divided by the whole window, and reporting it as a rate would
        understate every channel the reader has not finished. So the confirmed
        rate is offered only where `complete` is true, and the interface says
        "12 of 51" everywhere else.
        """
        totals: dict = {}
        for event in events:
            totals[event.channel] = totals.get(event.channel, 0) + 1
        out = {}
        for channel, total in sorted(totals.items()):
            counts = self.channel_counts(channel)
            counts["total"] = int(total)
            counts["complete"] = bool(total and counts["judged"] >= total)
            counts["verdict"] = self.channel_verdict(channel)
            out[channel] = counts
        return out

    # -- files -------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "window": self.window,
            "reader": self.reader,
            "note": self.note,
            # Orphans are written back among the events they came from: under
            # their own original keys, so that re-running the settings they
            # were given under finds them again.
            "events": {key: judged.to_json()
                       for key, judged in sorted({**self.orphaned,
                                                  **self.events}.items())},
            "channels": {key: judged.to_json()
                         for key, judged in sorted(self.channels.items())},
        }

    @classmethod
    def from_json(cls, data: dict) -> "Adjudication":
        return cls(
            window=str(data.get("window", "")),
            reader=str(data.get("reader", "")),
            note=str(data.get("note", "")),
            events={str(key): Judgement.from_json(value)
                    for key, value in (data.get("events") or {}).items()},
            channels={str(key): Judgement.from_json(value)
                      for key, value in (data.get("channels") or {}).items()},
        )


def load(request) -> Adjudication:
    """The read for this window, or an empty one. Never raises on a bad file.

    A corrupt or unreadable file returns an empty read rather than stopping
    the window from opening: losing the reviewer's previous verdicts is bad,
    but refusing to show them the recording at all is worse, and the file is
    left on disk untouched for them to look at.
    """
    path = path_for(request)
    blank = Adjudication(window=window_id(request))
    if not path.exists():
        return blank
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return blank
    if not isinstance(data, dict):
        return blank
    read = Adjudication.from_json(data)
    read.window = read.window or blank.window
    return read


def save(request, read: Adjudication) -> Path:
    """Write the read, atomically. Called after every single verdict.

    There is no save button on purpose. A reader who has worked through three
    hundred events and lost them to a crash will not use the software again,
    and a confirmation dialog on every click is not an interface. The write is
    small -- a few kilobytes of JSON -- so doing it per verdict costs nothing
    and removes the question.

    Atomic via a temporary file in the same directory, because the failure
    being guarded against is a crash *during* the write, which would otherwise
    leave a truncated file where a read used to be.
    """
    path = path_for(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    read.window = read.window or window_id(request)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(read.to_json(), indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def reconcile(read: Adjudication, events,
              tolerance_s: float = MATCH_TOLERANCE_S) -> Adjudication:
    """Re-attach stored verdicts to the events of the current analysis.

    Returns a new :class:`Adjudication` whose keys are the current events'
    keys, so the interface can look a verdict up by the event it is drawing.
    Each carried verdict is marked `exact` or `near`, and anything with no
    event to attach to is listed in `orphans` -- and kept, under its original
    key, so that re-running the previous settings finds it again.

    Matching is nearest-first within the tolerance and one event per verdict:
    without that, two verdicts on neighbouring events could both claim the
    same event and one reader's opinion would quietly overwrite another's.
    """
    carried: dict = {}
    orphans: list = []
    # Current events, grouped by channel and sorted, so a verdict only ever
    # looks at the handful of events it could plausibly be about.
    by_channel: dict = {}
    for event in events:
        key = event_key(event.channel, event.start, event.detector)
        by_channel.setdefault(event.channel, []).append((float(event.start), key))
    for items in by_channel.values():
        items.sort()
    taken: set = set()

    for key, judged in read.events.items():
        channel, start, _ = _split_key(key)
        candidates = by_channel.get(channel, [])
        best, best_gap, best_exact = None, float("inf"), False
        for when, candidate_key in candidates:
            if candidate_key in taken:
                continue
            if candidate_key == key:
                best, best_gap, best_exact = candidate_key, 0.0, True
                break
            gap = abs(when - start)
            if gap <= tolerance_s and gap < best_gap:
                best, best_gap = candidate_key, gap
        if best is None:
            orphans.append(key)
            continue
        taken.add(best)
        carried[best] = replace(judged, match="exact" if best_exact else "near")

    return Adjudication(
        window=read.window, reader=read.reader, note=read.note,
        events=carried, channels=dict(read.channels),
        orphaned={key: replace(read.events[key], match="orphan")
                  for key in sorted(orphans)})
