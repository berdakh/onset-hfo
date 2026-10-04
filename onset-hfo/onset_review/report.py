"""The record a reviewer saves when they are done looking.

A review that leaves nothing behind is a review nobody can check, so the
product exports one file per session. It is Markdown rather than PDF for a
deliberate reason: a reviewer can read it in a terminal, diff two of them, and
grep a directory of them, none of which is true of a rendered page. The
rendered page is a `--html` away for anyone who wants to print it.

Three rules shape what goes in:

1. **Provenance before findings.** The dataset, subject, window, montage,
   detector and threshold come first, because a rate without them is not a
   result. Every preprocessing step the pipeline logged is reproduced verbatim.
2. **The caveat is not a footnote.** When no channel stands out, that sentence
   sits directly under the heading, above the table it disqualifies.
3. **Nothing is computed here.** Every number is read off the `ReviewSession`.
   If a figure in the report disagrees with the screen, that is a bug in one
   place rather than a discrepancy between two implementations.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.quality import REASONS, quality_summary
from onset_review import adjudication, trends
from onset_review.session import DETECTOR_LABELS, ReviewSession

__all__ = ["review_markdown", "write_review", "DISCLAIMER"]

#: Carried on every export. The software is a research prototype that has never
#: been through a clinical trial or a regulatory submission, and a document that
#: leaves a hospital without saying so is the one failure mode of this whole
#: project that would matter to a patient.
DISCLAIMER = (
    "**Research prototype — not a medical device.** This software is not "
    "CE-marked, not FDA-cleared, and has not been validated for clinical use. "
    "Nothing in this document is a diagnosis or a surgical recommendation. The "
    "cohort evidence behind the method, including what it fails to show, is in "
    "`docs/EVALUATION.md` and `docs/LIMITATIONS.md`.")


def _cell(value) -> str:
    """One table cell, formatted so a column of numbers lines up when read.

    Written out rather than delegated to `DataFrame.to_markdown`, which needs
    `tabulate`. An export is the last step of a review and must not fail for
    want of an optional dependency on a machine someone else installed.
    """
    # NumPy's bool is checked explicitly because it is *not* a subclass of
    # Python's, so a column of `True`/`False` coming out of pandas would miss
    # the branch below and be printed as "True". It has to come before the
    # integer branch for the same reason the integer branch exists.
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """A Markdown table of the columns that exist, in the order asked for."""
    present = [c for c in columns if c in frame.columns]
    if frame.empty or not present:
        return "_nothing to show._\n"
    view = frame[present]
    if limit is not None:
        view = view.head(limit)
    lines = ["| " + " | ".join(c.replace("_", " ") for c in present) + " |",
             "|" + "|".join(["---"] * len(present)) + "|"]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _header(session: ReviewSession, reviewer: str | None) -> list[str]:
    request = session.request
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    detectors = ", ".join(DETECTOR_LABELS.get(d, d) for d in request.detectors)
    threshold = ("each detector's measured default"
                 if request.threshold_sd is None
                 else f"{request.threshold_sd:g} robust SD (set by the reviewer)")
    # An imported window has no accession, and an empty cell in a document
    # that will be read beside a cohort result reads as a missing value rather
    # than an absent one.
    source = (f"local file `{request.path.name}` (not a public archive "
              f"recording; nothing here has checked its provenance)"
              if request.imported else f"`{request.dataset}`")
    span_start, span_stop = session.span
    rows = [
        ("Dataset", source),
        ("Subject", f"`{request.subject}`, run `{request.run}`"),
        ("Window", f"{request.t_start:g}–{request.t_stop:g} s of the original "
                   f"recording ({request.duration:g} s)"),
        *([("Analysed", f"{span_start:g}–{span_stop:g} s "
                        f"({session.span_duration / 60:.1f} min), streamed in "
                        f"chunks — every rate below is over this, not over the "
                        f"window above, which is what the trace held")]
          if session.streamed else []),
        ("Band", request.band_label()),
        ("Detector(s)", f"{detectors} — ranking from "
                        f"{DETECTOR_LABELS.get(request.primary, request.primary)}"),
        ("Threshold", threshold),
        ("Montage", f"{session.montage}, {len(session.findings)} channels"),
        ("Sampling rate", f"{session.sfreq:g} Hz"),
        ("Reviewed on", now),
    ]
    named = reviewer or ", ".join(session.read.readers) or session.read.reader
    if named:
        rows.append(("Reviewer", named))
    return ["| field | value |", "|---|---|",
            *[f"| {name} | {value} |" for name, value in rows], ""]


def review_markdown(session: ReviewSession, reviewer: str | None = None,
                    notes: str = "", max_events: int = 50) -> str:
    """The whole review as one Markdown document."""
    summary = session.summary()
    out: list[str] = [
        f"# iEEG review — {session.request.subject}, "
        f"{session.request.t_start:g}–{session.request.t_stop:g} s", "",
        DISCLAIMER, "",
        "## What was reviewed", "",
    ]
    out += _header(session, reviewer)

    out += ["## Does anything stand out?", "", session.caveat(), ""]
    if session.leader.get("available"):
        out += [
            f"- Busiest channel: **{summary['leader']}** at "
            f"{session.leader['leader_rate_per_min']:g} events/min "
            f"(95% Poisson interval {session.leader['leader_ci'][0]:g}–"
            f"{session.leader['leader_ci'][1]:g}).",
            f"- Median channel: {session.leader['median_rate_per_min']:g}/min "
            f"(upper bound {session.leader['median_ci_high']:g}).",
            f"- Channels statistically tied with the busiest: "
            f"**{', '.join(session.candidates) or 'none'}**. This set, not the "
            f"single busiest channel, is what the data supports.", "",
        ]

    out += ["## Per-channel findings", "",
            f"{summary['accepted']} of {summary['detected']} candidate events "
            f"survived artifact rejection; {summary['spikes']} interictal "
            f"discharges were detected alongside them.", ""]
    out += [_table(session.findings,
                   ["rank", "channel", "n_events", "rate_per_min", "rate_ci_low",
                    "rate_ci_high", "mean_amplitude_uv", "mean_frequency_hz",
                    "mean_duration_ms", "mean_prominence_db", "n_with_spike",
                    "reviewed", "expert_n", "expert_rate_per_min"])]

    accord = trends.agreement_summary(session)
    out += ["## Against the archive's annotators", ""]
    if accord.get("available"):
        out += [accord["statement"], "",
                _table(trends.agreement(session),
                       ["channel", "n_expert", "n_detector", "matched",
                        "detector_only", "expert_only", "sensitivity",
                        "precision"])]
    else:
        out += [f"_{accord['reason']}._", ""]

    table = trends.event_table(session)
    out += ["## Events", "",
            f"{len(table)} accepted events, earliest first"
            + (f"; the first {max_events} are listed." if len(table) > max_events
               else "."), "",
            _table(table, ["t_file", "t_local", "channel", "kind",
                           "duration_ms", "amplitude_uv", "frequency_hz",
                           "prominence_db", "n_peaks", "with_spike"],
                   limit=max_events)]

    out += _read_section(session)

    out += _quality_section(session)

    out += ["## How the signal was prepared", ""]
    out += [f"{index}. {step}" for index, step in enumerate(session.steps, 1)]
    out += [""]
    if session.notes:
        out += ["### Dataset notes", ""]
        out += [f"- {note}" for note in session.notes] + [""]

    if notes.strip():
        out += ["## Reviewer notes", "", notes.strip(), ""]

    if session.citation:
        out += ["## Source", "", session.citation, ""]
    out += ["---", "",
            "Produced by `onset-review` "
            "(https://github.com/berdakh/onset-hfo). Re-running the same "
            "subject, window, band, detector and threshold reproduces this "
            "document.", ""]
    return "\n".join(out)


def write_review(session: ReviewSession, path: str | Path,
                 reviewer: str | None = None, notes: str = "") -> Path:
    """Write the review beside whatever name the reviewer chose.

    A `.html` suffix renders through `markdown` when it is installed and falls
    back to a readable `<pre>` block when it is not, because an export must not
    fail for want of an optional dependency on a clinical machine.
    """
    path = Path(path)
    text = review_markdown(session, reviewer=reviewer, notes=notes)
    if path.suffix.lower() in {".html", ".htm"}:
        path.write_text(_as_html(text, session), encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _as_html(text: str, session: ReviewSession) -> str:
    title = f"iEEG review — {session.request.subject}"
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables"])
    except Exception:
        body = "<pre>" + text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>"
        "body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;"
        "max-width:54rem;margin:3rem auto;padding:0 1.5rem;color:#1a1a1a}"
        "table{border-collapse:collapse;margin:1rem 0;font-size:13px}"
        "th,td{border:1px solid #d0d0d0;padding:.3rem .55rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "th{background:#f3f3f3}code{background:#f3f3f3;padding:.1rem .3rem}"
        "h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;"
        "border-bottom:1px solid #e0e0e0;padding-bottom:.2rem}"
        "</style></head><body>\n" + body + "\n</body></html>\n")


def _read_section(session: ReviewSession) -> list[str]:
    """What the *reader* decided, kept visibly apart from what the detector did.

    Its own section, and placed before the quality stage rather than tucked
    after it, because this is the only part of the document a person is
    accountable for. Everything above it is an algorithm's output, which is
    reproducible and therefore not anybody's opinion; this is the opposite.

    Three things are stated whether or not they are flattering: how much of
    the window was actually judged, which events the reader disagreed with,
    and how many verdicts no longer match an event. A report that quietly
    omitted any of them would read as a complete review when it was a partial
    one.
    """
    read = session.read
    if read is None or read.empty:
        return ["## The reader's own read", "",
                "_Nobody has recorded a verdict on this window. Every number "
                "above is the detector's, unreviewed._", ""]

    counts = read.counts()
    accepted = [e for e in session.events if e.accepted]
    # Per contact, counted over the events the ranking is built from, so the
    # `judged` column is comparable with `n_events` in the table above.
    primary = session.request.primary
    progress = read.progress([e for e in accepted if e.detector == primary])
    judged, total = counts["judged"], len(accepted)
    out = ["## The reader's own read", ""]
    out += ["| field | value |", "|---|---|",
            f"| Read by | {', '.join(read.readers) or read.reader or '—'} |",
            f"| Events judged | {judged} of {total} accepted events |",
            f"| Real | {counts['agree']} |",
            f"| Not real | {counts['disagree']} |",
            f"| Cannot tell | {counts['unsure']} |", ""]

    if judged < total:
        out += [f"**This is a partial read.** {total - judged} of {total} "
                f"accepted events carry no verdict. The counts above are not "
                f"a confirmed rate and must not be read as one.", ""]

    if counts["orphans"]:
        out += [f"{counts['orphans']} earlier verdict(s) no longer match any "
                f"event in this analysis — the detector settings changed after "
                f"they were given. They are kept in the stored read, not "
                f"deleted, and reappear if the earlier settings are used "
                f"again.", ""]

    confirmed = _confirmed_rates(session)
    rows = []
    for channel, entry in progress.items():
        if not (entry["judged"] or entry["verdict"]):
            continue
        rate = confirmed.get(channel, "")
        rows.append(
            f"| {channel} | "
            f"{adjudication.CHANNEL_LABELS.get(entry['verdict'], '—')} | "
            f"{entry['judged']} of {entry['total']} | {entry['agree']} | "
            f"{entry['disagree']} | {entry['unsure']} | {rate or '—'} |")
    if rows:
        out += ["### Per contact", "",
                "| channel | my read | judged | real | not real | cannot tell "
                "| confirmed /min |", "|---|---|---|---|---|---|---|", *rows, "",
                "_`judged` counts the events the ranking above is built "
                "from, so it is comparable with `n_events` there; a verdict "
                "on a discharge is recorded but not counted here. "
                "`confirmed /min` is the reader's own rate: it counts "
                "only those same events — accepted "
                f"`{DETECTOR_LABELS.get(session.request.primary, session.request.primary)}` "
                "events — over the same analysed seconds, and it is left blank "
                "unless every one of them carries a verdict. A rate from a "
                "half-judged contact would be a confirmed count divided by the "
                "whole window, which understates it._", ""]

    disagreed = [(key, judged_) for key, judged_ in sorted(read.events.items())
                 if judged_.verdict == "disagree"]
    if disagreed:
        out += ["### Events the reader rejected", "",
                "| channel | file time s | detector | why | at | by |",
                "|---|---|---|---|---|---|"]
        for key, judged_ in disagreed[:50]:
            channel, _, rest = key.partition("|")
            when, _, detector = rest.partition("|")
            out.append(f"| {channel} | {when} | {detector} | "
                       f"{judged_.note or '—'} | {judged_.at} | "
                       f"{judged_.reader or '—'} |")
        if len(disagreed) > 50:
            out.append(f"| … | | | {len(disagreed) - 50} more | | |")
        out.append("")

    notes = [(key, j) for key, j in sorted(read.channels.items()) if j.note]
    if notes:
        out += ["### Notes on contacts", ""]
        out += [f"- **{channel}** — {j.note} ({j.reader or '—'}, {j.at})"
                for channel, j in notes] + [""]

    if read.note.strip():
        out += ["### The reader's conclusion", "", read.note.strip(), ""]
    return out


def _confirmed_rates(session: ReviewSession) -> dict:
    """Channel -> the reader's own rate, as text, where they finished the channel.

    Built to be comparable with `rate_per_min` in the findings table and with
    nothing else: the same events (accepted, from the detector the ranking
    comes from) over the same denominator (that channel's analysed seconds,
    not the window length). Any other pairing would produce a number that
    looks like the one above it in the document and is not.
    """
    read = session.read
    primary = session.request.primary
    basis: dict = {}
    for event in session.events:
        if not event.accepted or event.detector != primary:
            continue
        basis.setdefault(event.channel, []).append(event)

    out = {}
    for channel, events in basis.items():
        verdicts = [read.verdict_of(adjudication.event_key(
            e.channel, e.start, e.detector)) for e in events]
        if not all(verdicts):
            continue
        seconds = float(session.clean_seconds.get(
            channel, session.request.duration) or 0.0)
        if seconds <= 0:
            continue
        out[channel] = f"{verdicts.count('agree') / (seconds / 60.0):.2f}"
    return out


def _quality_section(session: ReviewSession) -> list[str]:
    """Which contacts and seconds were analysed, and which were only flagged.

    In the document rather than only on screen, because the rate table above
    it is meaningless without it: a blank rate is a contact that was set
    aside, and a flagged contact's rank is a number the software is explicitly
    not vouching for.
    """
    quality = getattr(session, "quality", None)
    out = ["## Which contacts and seconds were analysed", ""]
    if quality is None or not len(quality):
        return out + ["_The data-quality checks did not run on this window, so "
                      "every contact was analysed and every rate was divided "
                      "by the whole window._", ""]

    out += [quality_summary(quality, session.segments,
                            kept=tuple(session.request.keep_channels)), ""]
    interesting = quality[(~quality["good"]) | quality["flagged"]]
    if not len(interesting):
        return out
    shown = interesting.assign(
        verdict=["set aside" if not row.good else "analysed, flagged"
                 for row in interesting.itertuples()],
        why=[REASONS.get(row.reason, row.reason)
             for row in interesting.itertuples()],
        analysed_s=[session.clean_seconds.get(str(c), float("nan"))
                    for c in interesting["channel"]])
    out += [_table(shown, ["channel", "verdict", "why", "amplitude_uv",
                           "hf_ratio_sd", "line_fraction", "analysed_s"]), ""]
    if bool(interesting["flagged"].any()):
        out += ["A flagged contact was analysed and ranked like any other. The "
                "check that flags it cannot tell a noisy amplifier from a "
                "contact carrying a great deal of real activity -- both raise "
                "the share of power in the band -- so it measures, says so, "
                "and leaves the judgement to whoever can look at the trace.",
                ""]
    return out
