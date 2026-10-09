"""Rates pooled over a case's segments. Qt-free.

Each segment is analysed as the review window analyses a window -- same
preprocessing, same detector, same quality stage -- and gives, per channel,
the accepted events and the seconds that were clean enough to analyse. The
pooled rate is the sum of the events over the sum of the clean minutes, its
interval the same Poisson interval the window prints, and a channel is tied
with the busiest when its interval overlaps the busiest's -- the window's own
rule. So one segment pooled alone gives exactly the window's numbers for
that segment (a test holds this), and more segments narrow the intervals
rather than change what they mean.

Beside the pooled rate, how consistently each channel led: in how many
segments it was tied with the busiest. A channel tied in one segment of six
is a different finding from one tied in all six, even at the same pooled
rate.
"""

from __future__ import annotations

import pandas as pd

from onset_hfo.metrics import leader_separation, poisson_ci, rank_channels

__all__ = ["pool", "POOLED_COLUMNS"]

POOLED_COLUMNS = ("channel", "n_events", "minutes", "rate_per_min", "rate_ci_low",
                  "rate_ci_high", "rank", "tied", "segments_tied", "segments_analysed")


def pool(segments: list[dict]) -> tuple[pd.DataFrame, dict]:
    """`segments`: one dict per analysed segment, {segment, counts: {channel:
    accepted events}, seconds: {channel: clean seconds}, tied: [channels]}.
    Returns the pooled table and the busiest-channel summary
    (`onset_hfo.metrics.leader_separation`'s, over the pooled table)."""
    channels: list[str] = []
    for seg in segments:
        for name in list(seg["seconds"]) + list(seg["counts"]):
            if name not in channels:
                channels.append(name)
    rows = []
    for name in channels:
        n = int(sum(int(seg["counts"].get(name, 0)) for seg in segments))
        seconds = float(sum(float(seg["seconds"].get(name, 0.0)) for seg in segments))
        minutes = seconds / 60.0
        low, high = poisson_ci(n, minutes)
        rows.append({"channel": name, "n_events": n, "minutes": round(minutes, 4),
                     "rate_per_min": (n / minutes) if minutes > 0 else float("nan"),
                     "rate_ci_low": low, "rate_ci_high": high,
                     "segments_tied": int(sum(name in set(seg.get("tied", ()))
                                              for seg in segments)),
                     "segments_analysed": int(sum(float(seg["seconds"].get(name, 0.0)) > 0
                                                  for seg in segments))})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=list(POOLED_COLUMNS)), {"available": False}
    usable = table.dropna(subset=["rate_per_min"])
    ranked = rank_channels(usable)
    leader = leader_separation(ranked)
    tied = set()
    if leader.get("available"):
        top = ranked.sort_values("rate_per_min", ascending=False).iloc[0]
        low = float(top["rate_ci_low"])
        tied = set(ranked.loc[ranked["rate_ci_high"].astype(float) >= low, "channel"])
    table = table.merge(ranked[["channel", "rank"]], on="channel", how="left")
    table["tied"] = table["channel"].isin(tied)
    table = table.sort_values(["rank", "channel"], na_position="last").reset_index(drop=True)
    return table[list(POOLED_COLUMNS)], leader
