"""Turning a list of events into the numbers a clinician or reviewer reads.

The unit of clinical interest is not the individual event: it is the **rate**
of events on a channel (per minute), and how channels rank against each other.
This module computes those, plus the two comparisons that keep the prototype
honest:

* **agreement between detectors** -- event-by-event and rank-by-rank;
* **rate change over time** -- e.g. before versus during a marked seizure.

Nothing here decides anything. It produces tables; the report states them and
the disagreements between them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from onset_hfo.detectors.base import Event

__all__ = [
    "channel_rates",
    "leader_separation",
    "rank_channels",
    "match_events",
    "detector_agreement",
    "agreement_matrix",
    "consensus_ranking",
    "compare_rankings",
    "rate_timecourse",
    "rate_change",
    "poisson_ci",
]


def poisson_ci(count: int, duration_min: float, alpha: float = 0.05) -> tuple[float, float]:
    """A confidence interval for a rate, treating events as a Poisson process.

    Why bother: "12 ripples/min on AD1-AD2 versus 9 on AD2-AD3" sounds like a
    difference until you notice that in a 60-second window those are 12 and 9
    events, whose intervals overlap almost completely. The interval is printed
    next to every rate in the report so that nobody over-reads a small count.

    Uses the Wilson-style normal approximation on sqrt(count), which is
    adequate here and has no SciPy dependency.
    """
    if duration_min <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963985 if abs(alpha - 0.05) < 1e-9 else 1.959963985
    root = math.sqrt(max(count, 0) + 0.25)
    lo = max(0.0, (root - z / 2) ** 2 - 0.25)
    hi = (root + z / 2) ** 2 - 0.25
    return (lo / duration_min, hi / duration_min)


def leader_separation(rates, top_k: int = 5) -> dict:
    """Does any channel actually stand out, or are they all tied?

    Every ranking function will sort noise. Ask one to rank a recording with
    nothing in it and it returns a leader, a runner-up and a confident-looking
    order -- and nothing anywhere in the output says "there is no signal
    here". This function is the missing null hypothesis, and it is built from
    a quantity the pipeline already computes.

    A channel is **tied with the leader** when its Poisson interval overlaps
    the leader's. If the leader is tied with the median channel, the recording
    has no leader: the apparent ordering is counting noise. That is a
    statement about intervals, not a tuned threshold, so it tightens by itself
    as the analysed window grows rather than needing a new constant.

    Returns ``distinguishable=False`` when the leader cannot be told apart
    from the middle of the pack. ``report.py`` and the ``detect_hfo`` tool
    both surface it, so the planner and the reader see the same caveat.
    """
    if rates is None or not len(rates) or "rate_ci_low" not in rates.columns:
        return {"available": False,
                "reason": "no rate table, or no confidence intervals in it"}
    ordered = rates.sort_values("rate_per_min", ascending=False).reset_index(drop=True)
    leader = ordered.iloc[0]
    leader_low = float(leader["rate_ci_low"])
    tied = ordered[ordered["rate_ci_high"].astype(float) >= leader_low]
    median_row = ordered.iloc[len(ordered) // 2]
    distinguishable = float(median_row["rate_ci_high"]) < leader_low
    return {
        "available": True,
        "leader": str(leader["channel"]),
        "leader_rate_per_min": round(float(leader["rate_per_min"]), 2),
        "leader_ci": [round(leader_low, 2), round(float(leader["rate_ci_high"]), 2)],
        "median_rate_per_min": round(float(median_row["rate_per_min"]), 2),
        "median_ci_high": round(float(median_row["rate_ci_high"]), 2),
        "n_tied_with_leader": int(len(tied)),
        "n_channels": int(len(ordered)),
        "top_k": int(top_k),
        "distinguishable": bool(distinguishable),
        "statement": (f"{leader['channel']} stands out: its interval does not overlap the "
                      f"median channel's" if distinguishable else
                      "NO CHANNEL STANDS OUT: the leader's confidence interval overlaps "
                      "the median channel's, so the ordering is counting noise and should "
                      "not be read as a ranking"),
    }


def channel_rates(events: list[Event],
                  duration_s: float | Mapping[str, float],
                  channels: list[str] | None = None,
                  accepted_only: bool = True) -> pd.DataFrame:
    """Per-channel event counts, rates and summary statistics.

    Channels with zero events are kept (with rate 0) when ``channels`` is
    given: "we looked and found nothing" is information, and dropping those
    rows is how a ranking silently becomes a list of only the noisy channels.

    ``duration_s`` may be one number for the whole window, or a mapping of
    channel to the seconds that channel was actually analysed over --
    :func:`onset_hfo.quality.clean_seconds` produces the latter. The mapping
    form exists because once any time can be rejected, one shared denominator
    is wrong: a channel that lost four seconds to an artifact would have its
    events divided by a minute that was not analysed, and the Poisson interval
    would inherit the error. Rejecting the artifact would then make the table
    *less* accurate than leaving it in.

    A channel with no clean time left gets a rate of NaN rather than zero.
    Zero is a measurement -- "we looked and found none" -- and this is the
    opposite of one.
    """
    rows = [e for e in events if (e.accepted or not accepted_only)]
    per_channel = isinstance(duration_s, Mapping)
    frame = pd.DataFrame([{
        "channel": e.channel,
        "amplitude_uv": e.peak_amplitude_uv,
        "frequency_hz": e.peak_frequency_hz,
        "duration_ms": e.duration_ms,
        "prominence_db": e.spectral_prominence_db,
        "with_spike": bool(e.co_occurs_with_spike),
    } for e in rows])

    index = pd.Index(channels if channels is not None
                     else sorted({e.channel for e in rows}), name="channel")
    if frame.empty:
        out = pd.DataFrame(index=index).reset_index()
        out["n_events"] = 0
        out["rate_per_min"] = 0.0
    else:
        grouped = frame.groupby("channel")
        out = pd.DataFrame({
            "n_events": grouped.size(),
            "mean_amplitude_uv": grouped["amplitude_uv"].mean(),
            "mean_frequency_hz": grouped["frequency_hz"].mean(),
            "mean_duration_ms": grouped["duration_ms"].mean(),
            "mean_prominence_db": grouped["prominence_db"].mean(),
            "n_with_spike": grouped["with_spike"].sum(),
        }).reindex(index).fillna({"n_events": 0, "n_with_spike": 0}).reset_index()
        out["n_events"] = out["n_events"].astype(int)

    seconds = ([float(duration_s.get(str(c), 0.0)) for c in out["channel"]]
               if per_channel else [float(duration_s)] * len(out))
    minutes = [s / 60.0 for s in seconds]
    out["duration_s"] = seconds
    out["rate_per_min"] = [
        (n / m if m > 0 else float("nan"))
        for n, m in zip(out["n_events"], minutes, strict=True)]

    ci = [poisson_ci(int(n), m) for n, m in zip(out["n_events"], minutes, strict=True)]
    out["rate_ci_low"] = [c[0] for c in ci]
    out["rate_ci_high"] = [c[1] for c in ci]
    return out.sort_values("rate_per_min", ascending=False).reset_index(drop=True)


def rank_channels(rates: pd.DataFrame, by: str = "rate_per_min") -> pd.DataFrame:
    """Add a 1-based ``rank`` column (1 = highest rate). Ties share a rank.

    A channel with no rate at all -- one whose clean time the quality stage
    reduced to nothing -- is kept, sorted to the bottom, and given **rank 0**,
    which is outside the 1-based space and means "not ranked". It is not given
    the last rank: last place is a measurement, and this channel has none. Its
    row stays so that "we looked and could not use it" is visible rather than
    being a gap in the montage.
    """
    out = rates.sort_values(by, ascending=False, na_position="last")
    out = out.reset_index(drop=True).copy()
    ranked = out[by].rank(ascending=False, method="min")
    out["rank"] = ranked.fillna(0).astype(int)
    return out


def match_events(a: list[Event], b: list[Event], tolerance: float = 0.02,
                 require_same_channel: bool = True) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy one-to-one matching of two event lists by time overlap.

    Returns ``(pairs, unmatched_a, unmatched_b)`` as index lists. Two events
    match when they overlap in time (within ``tolerance`` seconds) and, by
    default, sit on the same channel.
    """
    order_a = sorted(range(len(a)), key=lambda i: a[i].start)
    order_b = sorted(range(len(b)), key=lambda i: b[i].start)
    used_b: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i in order_a:
        best, best_gap = None, None
        for j in order_b:
            if j in used_b:
                continue
            if require_same_channel and a[i].channel != b[j].channel:
                continue
            if b[j].stop < a[i].start - tolerance:
                continue
            if b[j].start > a[i].stop + tolerance:
                if not require_same_channel:
                    continue
                break
            gap = abs(a[i].mid - b[j].mid)
            if best_gap is None or gap < best_gap:
                best, best_gap = j, gap
        if best is not None:
            used_b.add(best)
            pairs.append((i, best))
    matched_a = {i for i, _ in pairs}
    return pairs, [i for i in range(len(a)) if i not in matched_a], \
        [j for j in range(len(b)) if j not in used_b]


def detector_agreement(a: list[Event], b: list[Event], name_a: str = "a", name_b: str = "b",
                       tolerance: float = 0.02) -> dict:
    """How much two detectors agree, event by event.

    ``jaccard`` is matched events divided by all distinct events. It is not a
    kappa: there is no meaningful count of "events both detectors correctly did
    not detect", so a chance-corrected statistic would be invented, not
    measured.
    """
    a = [e for e in a if e.accepted]
    b = [e for e in b if e.accepted]
    pairs, only_a, only_b = match_events(a, b, tolerance)
    union = len(pairs) + len(only_a) + len(only_b)
    return {
        "detector_a": name_a,
        "detector_b": name_b,
        f"n_{name_a}": len(a),
        f"n_{name_b}": len(b),
        "n_matched": len(pairs),
        f"n_only_{name_a}": len(only_a),
        f"n_only_{name_b}": len(only_b),
        "jaccard": (len(pairs) / union) if union else float("nan"),
        "tolerance_s": tolerance,
    }


def agreement_matrix(events_by_detector: dict[str, list[Event]],
                     tolerance: float = 0.02) -> pd.DataFrame:
    """Pairwise Jaccard agreement between every pair of detectors.

    Two detectors matching about half their events is a finding; three or four
    show whether that disagreement is *structural* or an idiosyncrasy of one
    pair. The diagonal is 1.0 by construction and is filled in so the table
    reads as a matrix rather than as a triangle with holes.

    The matrix is also the only place the redundancy question gets answered.
    ``short_time_energy`` is ``rms`` under a monotone transform, so a high cell
    between those two is the expected result and a *low* one would be the
    surprise worth chasing. An ensemble of four detectors, two of which agree
    with each other on nearly every event, holds three opinions.

    Returns a square frame indexed and columned by detector name. Off-diagonal
    cells are ``nan`` only when neither detector accepted a single event, since
    a Jaccard index over two empty sets is undefined rather than perfect.
    """
    names = list(events_by_detector)
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        matrix.loc[a, a] = 1.0
        for b in names[i + 1:]:
            jaccard = detector_agreement(events_by_detector[a], events_by_detector[b],
                                         a, b, tolerance)["jaccard"]
            matrix.loc[a, b] = matrix.loc[b, a] = jaccard
    return matrix


def consensus_ranking(events_by_detector: dict[str, list[Event]], duration_s: float,
                      channels: list[str] | None = None, top_k: int = 5) -> pd.DataFrame:
    """Rank channels by how many *detectors* place them in their own top group.

    The motivation is that a rate is a fragile statistic -- it is an argmax
    over channels whose Poisson intervals overlap (see
    :func:`leader_separation`) -- while "three of four detectors put this
    channel in their top five" is a vote, and a vote over independent-ish
    features degrades more gracefully than an average of four rates would.

    Averaging the rates instead would be worse for two reasons. The features
    are on different scales -- short-time energy is in microvolts squared
    times samples -- so an average would be dominated by whichever detector
    has the largest units. And a detector that is nearly a copy of another
    would get counted twice in the mean; here it still gets one vote, which is
    the right amount of influence for a fourth opinion that is really a second
    copy of the first. Read this column next to
    :func:`agreement_matrix`: a unanimous vote among detectors that agree with
    each other on every event is not unanimity, it is one detector.

    ``n_detectors`` is the vote. ``mean_rank`` breaks ties, and the per-
    detector rate columns are kept so a reader can see what the vote was built
    from rather than taking it on trust.
    """
    if not events_by_detector:
        return pd.DataFrame(columns=["channel", "n_detectors", "mean_rank"])

    per_detector = {name: rank_channels(channel_rates(events, duration_s, channels))
                    for name, events in events_by_detector.items()}

    frame: pd.DataFrame | None = None
    for name, ranked in per_detector.items():
        part = ranked[["channel", "rank", "rate_per_min"]].rename(
            columns={"rank": f"rank_{name}", "rate_per_min": f"rate_{name}"})
        frame = part if frame is None else frame.merge(part, on="channel", how="outer")

    rank_cols = [c for c in frame.columns if c.startswith("rank_")]
    n_channels = len(frame)
    for col in rank_cols:
        # A channel a detector never saw sits below everything it did see.
        frame[col] = frame[col].fillna(n_channels + 1).astype(int)
    for col in [c for c in frame.columns if c.startswith("rate_")]:
        frame[col] = frame[col].fillna(0.0)

    frame["n_detectors"] = (frame[rank_cols] <= top_k).sum(axis=1)
    frame["mean_rank"] = frame[rank_cols].mean(axis=1)
    frame["top_k"] = top_k
    ordered = frame.sort_values(["n_detectors", "mean_rank"], ascending=[False, True])
    return ordered.reset_index(drop=True)


def compare_rankings(rates_a: pd.DataFrame, rates_b: pd.DataFrame,
                     name_a: str = "rms", name_b: str = "line_length",
                     disagreement_ranks: int = 5, top_k: int = 5) -> pd.DataFrame:
    """Merge two per-channel rankings and mark where they disagree.

    A channel is flagged ``disagrees`` when the two detectors' ranks differ by
    at least ``disagreement_ranks`` *and* at least one of them puts it in the
    top ``top_k``. A disagreement deep in the list is not interesting; a
    disagreement about a leading channel is exactly what a reader must see.
    """
    a = rank_channels(rates_a)[["channel", "rank", "rate_per_min", "n_events"]]
    b = rank_channels(rates_b)[["channel", "rank", "rate_per_min", "n_events"]]
    merged = a.merge(b, on="channel", how="outer", suffixes=(f"_{name_a}", f"_{name_b}"))
    max_rank = int(np.nanmax([merged[f"rank_{name_a}"].max(), merged[f"rank_{name_b}"].max(), 1]))
    for col in (f"rank_{name_a}", f"rank_{name_b}"):
        merged[col] = merged[col].fillna(max_rank + 1).astype(int)
    for col in merged.columns:
        if col.startswith("rate_") or col.startswith("n_events"):
            merged[col] = merged[col].fillna(0.0)
    merged["rank_gap"] = (merged[f"rank_{name_a}"] - merged[f"rank_{name_b}"]).abs()
    merged["best_rank"] = merged[[f"rank_{name_a}", f"rank_{name_b}"]].min(axis=1)
    merged["disagrees"] = (merged["rank_gap"] >= disagreement_ranks) & (merged["best_rank"] <= top_k)
    return merged.sort_values(["best_rank", "rank_gap"]).reset_index(drop=True)


def rate_timecourse(events: list[Event], t_start: float, t_stop: float, bin_s: float = 5.0,
                    channels: list[str] | None = None, accepted_only: bool = True) -> pd.DataFrame:
    """Events per minute in consecutive time bins (optionally per channel)."""
    rows = [e for e in events if (e.accepted or not accepted_only)
            and (channels is None or e.channel in channels)]
    edges = np.arange(t_start, t_stop + bin_s, bin_s)
    counts, _ = np.histogram([e.mid for e in rows], bins=edges)
    return pd.DataFrame({
        "t_start": edges[:-1],
        "t_stop": edges[1:],
        "n_events": counts,
        "rate_per_min": counts / (bin_s / 60.0),
    })


def rate_change(events: list[Event], window_a: tuple[float, float], window_b: tuple[float, float],
                channels: list[str] | None = None, label_a: str = "before",
                label_b: str = "during") -> pd.DataFrame:
    """Per-channel rate in two time windows and the ratio between them.

    Used for the pre-ictal versus ictal comparison. The ratio is reported with
    both counts, because a ratio computed from three events and one event is a
    number, not a finding.
    """
    def _rates(window):
        sel = [e for e in events if e.accepted and window[0] <= e.mid < window[1]]
        return channel_rates(sel, window[1] - window[0], channels)

    a = _rates(window_a).rename(columns={"rate_per_min": f"rate_{label_a}",
                                         "n_events": f"n_{label_a}"})
    b = _rates(window_b).rename(columns={"rate_per_min": f"rate_{label_b}",
                                         "n_events": f"n_{label_b}"})
    merged = a[["channel", f"n_{label_a}", f"rate_{label_a}"]].merge(
        b[["channel", f"n_{label_b}", f"rate_{label_b}"]], on="channel", how="outer").fillna(0)
    merged["ratio"] = merged[f"rate_{label_b}"] / merged[f"rate_{label_a}"].replace(0, np.nan)
    return merged.sort_values(f"rate_{label_b}", ascending=False).reset_index(drop=True)
