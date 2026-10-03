"""Overview quantities: where in the window, on which channel, and versus whom.

Persyst's reviewers do not read a 20-minute recording trace-first. They read a
*trend* -- one row per channel, one column per time bin, colour for activity --
pick the bright patch, and only then open the signal there. That is the right
order for HFO review too, and more honestly so: a window of HFO rates is a
sparse count, and a trend makes the sparsity visible in a way a ranked table
hides.

Three things live here, all pure functions of a loaded `ReviewSession`:

* `rate_matrix` -- the trend itself: counts per channel per time bin.
* `event_table` -- one row per event, which is what an arrow-key walk through
  the recording steps along.
* `agreement` -- the detector against the archive's own annotators, per
  channel, for the window on screen.

No Qt, no plotting. The panels in `onset_review.panels` render what these
return and compute nothing of their own, so a test can assert on the numbers a
clinician sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from onset_hfo.detectors.base import Event
from onset_hfo.metrics import match_events
from onset_review.session import ReviewSession

__all__ = ["rate_matrix", "rate_curve", "event_table", "agreement",
           "agreement_summary", "DEFAULT_BIN_S"]

#: Seconds per trend column. Five is a compromise found by looking: at 1 s most
#: bins hold zero or one event and the trend is salt-and-pepper, and at 10 s a
#: 60 s window is six columns wide and there is nothing to see. It is a display
#: choice, exposed so a reviewer can change it, and no measurement depends on
#: it.
DEFAULT_BIN_S = 5.0


def _bins(session: ReviewSession, bin_s: float) -> np.ndarray:
    """Bin edges in window-local seconds, always covering the whole window.

    The last bin is allowed to be short rather than dropped: a reviewer who
    sees 60 s of trace must see 60 s of trend above it, and silently trimming
    the tail is how an event goes missing from an overview that claims to be
    complete.
    """
    duration = max(float(session.request.duration), float(bin_s))
    n = max(1, int(np.ceil(duration / float(bin_s))))
    edges = np.arange(n + 1, dtype=float) * float(bin_s)
    edges[-1] = max(edges[-1], duration)
    return edges


def _local(event: Event, session: ReviewSession) -> float:
    """An event's start in window-local seconds (the trace's own time base)."""
    return float(event.start) - float(session.t_offset)


def rate_matrix(session: ReviewSession, bin_s: float = DEFAULT_BIN_S,
                detector: str | None = None, source: str = "detector") -> pd.DataFrame:
    """Counts per channel per time bin: the trend panel's whole content.

    Rows are channels in findings order (busiest first), so the trend and the
    findings table can be read across. Columns are the left edge of each bin in
    window-local seconds.

    `source="expert"` returns the same shape built from the archive's markings,
    which is what lets the panel show the two trends stacked -- the comparison
    that teaches a new reviewer what the detector is doing.
    """
    if session.findings.empty:
        return pd.DataFrame()
    channels = list(session.findings["channel"])
    edges = _bins(session, bin_s)
    counts = pd.DataFrame(0, index=pd.Index(channels, name="channel"),
                          columns=pd.Index(edges[:-1], name="t_start"),
                          dtype=int)

    if source == "expert":
        events = list(session.expert)
    else:
        name = detector or session.request.primary
        events = session.events_of(name)

    index = {channel: position for position, channel in enumerate(channels)}
    for event in events:
        row = index.get(event.channel)
        if row is None:
            continue
        column = int(np.searchsorted(edges, _local(event, session), side="right") - 1)
        column = min(max(column, 0), counts.shape[1] - 1)
        counts.iat[row, column] += 1
    return counts


def rate_curve(session: ReviewSession, bin_s: float = DEFAULT_BIN_S,
               channel: str | None = None) -> pd.DataFrame:
    """Events per minute over time, for the whole window or one channel.

    Reported as a rate rather than a count so the curve does not change shape
    when the bin width does, and with the bin's own duration in a column
    because the final bin can be short.
    """
    edges = _bins(session, bin_s)
    matrix = rate_matrix(session, bin_s)
    if matrix.empty:
        return pd.DataFrame(columns=["t_start", "t_stop", "n_events", "rate_per_min"])
    counts = (matrix.loc[channel] if channel is not None and channel in matrix.index
              else matrix.sum(axis=0))
    widths = np.diff(edges)
    return pd.DataFrame({
        "t_start": edges[:-1],
        "t_stop": edges[1:],
        "n_events": np.asarray(counts, dtype=int),
        "rate_per_min": np.asarray(counts, dtype=float) / (widths / 60.0),
    })


def event_table(session: ReviewSession, include_rejected: bool = False) -> pd.DataFrame:
    """One row per event, in time order: the list a reviewer walks.

    Both time bases are present on purpose. `t_local` is where to scroll the
    trace, `t_file` is what a report quotes and what someone can find again in
    the archive, and keeping only one of them is how a reviewer ends up unable
    to cite what they just looked at.

    Rejected events are available but off by default: they are the filter's
    working, useful when asking why the detector did *not* mark something, and
    noise in a list meant for reading.
    """
    rows = []
    for event in sorted(session.events, key=lambda e: e.start):
        if not include_rejected and not event.accepted:
            continue
        rows.append({
            "t_local": round(_local(event, session), 4),
            "t_file": round(float(event.start), 4),
            "channel": event.channel,
            "kind": "spike" if event.detector == "spike" else (
                "fast ripple" if event.band and event.band[0] >= 200 else "ripple"),
            "detector": event.detector,
            "duration_ms": round(float(event.duration_ms), 1),
            "amplitude_uv": round(float(event.peak_amplitude_uv), 1),
            "frequency_hz": round(float(event.peak_frequency_hz), 1),
            "prominence_db": round(float(event.spectral_prominence_db), 1),
            "n_peaks": int(event.n_peaks),
            "with_spike": bool(event.co_occurs_with_spike),
            "accepted": bool(event.accepted),
            "reject_reason": event.reject_reason or "",
        })
    columns = ["t_local", "t_file", "channel", "kind", "detector", "duration_ms",
               "amplitude_uv", "frequency_hz", "prominence_db", "n_peaks",
               "with_spike", "accepted", "reject_reason"]
    return pd.DataFrame(rows, columns=columns)


def agreement(session: ReviewSession, tolerance: float = 0.02) -> pd.DataFrame:
    """Detector against the archive's annotators, per channel, on this window.

    Restricted to the channels the annotators actually reviewed, because
    `ds003498` marked only a subset: a detection on an unreviewed channel is
    unjudged, not a false positive, and counting it as one is the single
    easiest way to make a detector look worse than it is.

    `matched` uses the project's own `match_events`, so a pair here is a pair
    by the same rule the evaluation chapter uses -- a reviewer comparing the
    screen against `docs/EVALUATION.md` should find the same numbers.
    """
    if not session.expert or not session.reviewed_channels:
        return pd.DataFrame(columns=["channel", "n_detector", "n_expert",
                                     "matched", "detector_only", "expert_only",
                                     "sensitivity", "precision"])
    reviewed = set(session.reviewed_channels)
    mine = [e for e in session.hfo_events if e.channel in reviewed]
    theirs = [e for e in session.expert if e.channel in reviewed]

    rows = []
    for channel in sorted(reviewed):
        a = [e for e in mine if e.channel == channel]
        b = [e for e in theirs if e.channel == channel]
        pairs, only_a, only_b = match_events(a, b, tolerance=tolerance)
        rows.append({
            "channel": channel,
            "n_detector": len(a),
            "n_expert": len(b),
            "matched": len(pairs),
            "detector_only": len(only_a),
            "expert_only": len(only_b),
            "sensitivity": len(pairs) / len(b) if b else float("nan"),
            "precision": len(pairs) / len(a) if a else float("nan"),
        })
    return (pd.DataFrame(rows)
            .sort_values("n_expert", ascending=False)
            .reset_index(drop=True))


def agreement_summary(session: ReviewSession, tolerance: float = 0.02) -> dict:
    """The pooled version of `agreement`, plus the sentence that goes with it.

    Pooled over channels rather than averaged over them: a per-channel mean
    gives a channel with two expert marks the same weight as one with sixty,
    which is how a sensitivity gets quoted that no patient's recording has.
    """
    table = agreement(session, tolerance)
    if table.empty:
        return {"available": False,
                "reason": ("this recording carries no expert markings, so there "
                           "is nothing to agree with")}
    matched = int(table["matched"].sum())
    mine = int(table["n_detector"].sum())
    theirs = int(table["n_expert"].sum())
    sensitivity = matched / theirs if theirs else float("nan")
    precision = matched / mine if mine else float("nan")
    return {
        "available": True,
        "n_channels": int(len(table)),
        "n_detector": mine,
        "n_expert": theirs,
        "matched": matched,
        "sensitivity": sensitivity,
        "precision": precision,
        "tolerance_s": float(tolerance),
        "statement": (
            f"On the {len(table)} reviewed channels in this window the detector "
            f"marked {mine} events and the annotators marked {theirs}; "
            f"{matched} overlap within {tolerance * 1000:.0f} ms "
            f"(sensitivity {sensitivity:.0%}, precision {precision:.0%}). "
            f"Agreement on a single window is not the detector's accuracy -- "
            f"see docs/EVALUATION.md for the cohort figures."),
    }
