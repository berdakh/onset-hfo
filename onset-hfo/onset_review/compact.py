"""Compact views of the tables: what a reader needs first, the rest behind a click.

Every table in the window was written for the person who built the pipeline,
with every measurement in a column of its own. A clinician reading a window
wants four things from the ranking -- which channel, how busy, how sure, what
the annotators said -- and the other eleven columns behind a toggle. This
module decides what "compact" means for each table, and formats the cells
that put two numbers in one ("52 (39–67)"), so that the panels and the
exported report say the same thing in the same words.

Qt-free: the panels call these, the report calls these, and the tests read
them without a display.
"""

from __future__ import annotations

import math

import pandas as pd

__all__ = ["COMPACT_COLUMNS", "FULL_COLUMNS", "rate_cell", "compact_findings",
           "compact_events", "agreement_bars", "quality_chips", "QUALITY_KINDS"]

#: The columns each table shows in its compact form, in order. The label is
#: what the header says; a name not in the frame is dropped by the model.
COMPACT_COLUMNS = {
    "findings": ["rank", "channel", "rate", "annotators", "my_read", "judged"],
    # `key` is hidden in the view; it is how a selected row becomes the event
    # it stands for, so it travels with every column set.
    "events": ["verdict", "t_local", "channel", "kind", "frequency_hz", "duration_ms", "key"],
    "agreement": ["channel", "n_expert", "n_detector", "matched"],
}

#: The full sets, kept here so a toggle has one place to read both from.
FULL_COLUMNS = {
    "findings": ["rank", "channel", "my_read", "judged", "n_events",
                 "rate_per_min", "rate_ci_low", "rate_ci_high",
                 "mean_frequency_hz", "mean_duration_ms", "mean_prominence_db",
                 "n_with_spike", "reviewed", "expert_n", "expert_rate_per_min"],
    "events": ["verdict", "t_local", "channel", "kind", "duration_ms",
               "amplitude_uv", "frequency_hz", "prominence_db", "n_peaks",
               "with_spike", "reject_reason", "key"],
    "agreement": ["channel", "n_expert", "n_detector", "matched", "detector_only",
                  "expert_only", "sensitivity", "precision"],
}


def rate_cell(rate, low=None, high=None) -> str:
    """"52 (39–67)": the rate and its interval in one cell, whole numbers,
    because a reader comparing channels does not need the second decimal
    and a column of them hides the one digit that matters."""
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return "—"
    text = f"{float(rate):.0f}"
    if low is not None and high is not None and not (
            _nan(low) or _nan(high)):
        text += f" ({float(low):.0f}–{float(high):.0f})"
    return text


def _nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def compact_findings(frame: pd.DataFrame) -> pd.DataFrame:
    """The findings table with the two cells a reader scans added: the rate
    with its interval, and what the annotators marked on that channel (blank
    where they never looked, which is a different thing from zero)."""
    out = frame.copy()
    if out.empty:
        out["rate"] = pd.Series(dtype=str)
        out["annotators"] = pd.Series(dtype=str)
        return out
    out["rate"] = [rate_cell(r.get("rate_per_min"), r.get("rate_ci_low"), r.get("rate_ci_high"))
                   for r in out.to_dict("records")]
    if "expert_n" in out.columns:
        reviewed = out["reviewed"] if "reviewed" in out.columns else pd.Series(True, index=out.index)
        out["annotators"] = [
            (f"{int(n)} marked" if bool(seen) and not _nan(n) else "not reviewed")
            for n, seen in zip(out["expert_n"], reviewed, strict=True)]
    else:
        out["annotators"] = "—"
    return out


def compact_events(frame: pd.DataFrame) -> pd.DataFrame:
    """The event list is already one row per event; compact is a column
    choice, so the frame comes back as it is and the columns do the work."""
    return frame


def agreement_bars(frame: pd.DataFrame, top: int = 20) -> list[dict]:
    """Per reviewed channel, the two counts a reader compares: what the
    annotators marked and what the detector marked, plus the overlap. Sorted
    by the annotators' count, the way the table is, and capped so the bars
    stay readable."""
    if frame is None or frame.empty:
        return []
    rows = frame.sort_values("n_expert", ascending=False).head(int(top))
    return [{"channel": str(r.channel), "expert": int(r.n_expert),
             "detector": int(r.n_detector), "matched": int(getattr(r, "matched", 0))}
            for r in rows.itertuples()]


#: Quality verdict kinds, in the order the strip shows them, with the
#: palette token each is drawn in.
QUALITY_KINDS = (
    ("set_aside", "set aside", "bad"),
    ("flagged", "analysed, flagged", "warn"),
    ("reinstated", "reinstated by you", "accent"),
    ("analysed", "analysed", "good"),
)


def quality_chips(quality: pd.DataFrame, kept=()) -> list[dict]:
    """One chip per contact: its name, which kind of verdict, and the reason,
    for a strip a reader can take in at a glance where a table of nine
    columns says the same thing in a minute. Set-aside first, then flagged,
    so the contacts worth a look are at the front."""
    if quality is None or quality.empty:
        return []
    kept = set(kept or ())
    chips = []
    for row in quality.itertuples():
        reason = str(getattr(row, "reason", "") or "")
        good, flagged = bool(row.good), bool(getattr(row, "flagged", False))
        name = str(row.channel)
        if not good and name in kept:
            kind = "reinstated"
        elif not good:
            kind = "set_aside"
        elif flagged:
            kind = "flagged"
        else:
            kind = "analysed"
        chips.append({"channel": name, "kind": kind, "reason": reason})
    order = {kind: i for i, (kind, _l, _t) in enumerate(QUALITY_KINDS)}
    chips.sort(key=lambda c: (order[c["kind"]], c["channel"]))
    return chips


def counts_by_kind(chips: list[dict]) -> dict[str, int]:
    counts = {kind: 0 for kind, _l, _t in QUALITY_KINDS}
    for chip in chips:
        counts[chip["kind"]] = counts.get(chip["kind"], 0) + 1
    return counts


