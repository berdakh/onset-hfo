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

from onset_review import trends
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
    rows = [
        ("Dataset", source),
        ("Subject", f"`{request.subject}`, run `{request.run}`"),
        ("Window", f"{request.t_start:g}–{request.t_stop:g} s of the original "
                   f"recording ({request.duration:g} s)"),
        ("Band", request.band_label()),
        ("Detector(s)", f"{detectors} — ranking from "
                        f"{DETECTOR_LABELS.get(request.primary, request.primary)}"),
        ("Threshold", threshold),
        ("Montage", f"{session.montage}, {len(session.findings)} channels"),
        ("Sampling rate", f"{session.sfreq:g} Hz"),
        ("Reviewed on", now),
    ]
    if reviewer:
        rows.append(("Reviewer", reviewer))
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
