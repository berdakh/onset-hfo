"""Threshold sensitivity: does the ranking survive a stricter detector?

A rate is a count above a threshold, and the threshold is a choice. The
research ladder tests every leader the same way -- re-run the detector at a
higher threshold and see whether the channel still leads -- and the window
could not. This is that test on the window in front of the reviewer: the
ranking detector re-run at the threshold it used and at 1.25, 1.5 and 2 times
it, on the same prepared signal, and the leading channels' rates at each.

Qt-free. The detector is MNE-free NumPy and runs in a few seconds for a
minute of forty channels; the panel in :mod:`onset_review.sensitivityview`
runs it on a worker and draws. Nothing here changes the window's analysis:
the re-runs go through :class:`onset_agent.analysis.AnalysisSession` on the
recording the window holds, and the window's own numbers stay the saved
table's.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

__all__ = ["Sensitivity", "THRESHOLD_FACTORS", "TOP", "compute", "describe", "as_dict"]

#: Multiples of the threshold the window used. The ladder's own re-test is 2x.
THRESHOLD_FACTORS = (1.0, 1.25, 1.5, 2.0)
#: How many of the window's leading channels are followed.
TOP = 5


@dataclass
class Sensitivity:
    detector: str = ""
    base_threshold: float = float("nan")
    thresholds: list[float] = field(default_factory=list)
    #: The window's leading channels at its own threshold, in rank order.
    channels: list[str] = field(default_factory=list)
    #: Rates per minute, one row per analysed channel, one column per threshold.
    rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per threshold: who leads among the analysed channels, who is tied with
    #: them, and whether the leader can be told from the median channel.
    leaders: list[str | None] = field(default_factory=list)
    tied: list[list[str]] = field(default_factory=list)
    stands_out: list[bool | None] = field(default_factory=list)
    runtime_s: float = 0.0
    reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.thresholds) and not self.rates.empty

    def survives_to(self) -> float | None:
        """The largest factor at which the window's leader still leads, or
        None when it does not lead even at its own threshold."""
        if not self.available or not self.channels:
            return None
        leader = self.channels[0]
        best = None
        for threshold, who in zip(self.thresholds, self.leaders, strict=True):
            if who == leader:
                best = threshold / self.base_threshold
            else:
                break
        return best


def compute(session, factors: tuple = THRESHOLD_FACTORS, top: int = TOP,
            analysis=None) -> Sensitivity:
    """Re-run the ranking detector at each multiple of its threshold.

    `analysis` is an :class:`AnalysisSession` to reuse, for tests and for the
    assistant, which may already hold one; built from the window otherwise.
    """
    out = Sensitivity()
    recording = getattr(session, "recording", None)
    findings = getattr(session, "findings", None)
    if recording is None:
        out.reason = "this session has no recording attached, so the detector cannot be re-run"
        return out
    if findings is None or findings.empty:
        out.reason = "no channel was analysed in this window"
        return out
    from onset_agent.analysis import AnalysisSession
    from onset_hfo.metrics import leader_separation

    request = session.request
    detector = str(request.primary)
    config = request.pipeline_config()
    base = float(getattr(config, detector).threshold_sd)
    started = time.time()
    analysis = analysis or AnalysisSession(recording, config, verbose=False)
    all_names = list(analysis.prepared.ch_names)
    # Only the channels the window analysed: a contact the quality stage set
    # aside stays set aside, or the re-test would rank a noisy amplifier the
    # window had already excluded.
    analysed = [str(c) for c in findings["channel"] if str(c) in all_names]
    ranked = findings.sort_values("rank")
    leading = [str(c) for c in ranked["channel"] if str(c) in analysed][:max(1, int(top))]
    thresholds = [round(base * float(f), 3) for f in factors]
    rates: dict[float, pd.Series] = {}
    counts: dict[float, pd.Series] = {}
    for threshold in thresholds:
        _events, table, _prep, _window, _cached = analysis.run_hfo(
            detector, all_names, None, threshold, None)
        table = table[table["channel"].astype(str).isin(analysed)].copy()
        table["channel"] = table["channel"].astype(str)
        sep = leader_separation(table, top_k=int(getattr(config, "top_k", 5)))
        by_channel = table.set_index("channel")
        rates[threshold] = by_channel["rate_per_min"].astype(float)
        counts[threshold] = by_channel["n_events"].astype(int)
        if sep.get("available") and int(by_channel["n_events"].sum()) > 0:
            ordered = table.sort_values("rate_per_min", ascending=False)
            leader_low = float(ordered.iloc[0]["rate_ci_low"])
            tied = [str(c) for c in ordered[ordered["rate_ci_high"].astype(float)
                                              >= leader_low]["channel"]]
            out.leaders.append(str(sep["leader"]))
            out.tied.append(tied)
            out.stands_out.append(bool(sep["distinguishable"]))
        else:
            out.leaders.append(None)
            out.tied.append([])
            out.stands_out.append(None)
    out.detector, out.base_threshold, out.thresholds = detector, base, thresholds
    out.channels = leading
    out.rates = pd.DataFrame(rates).reindex(analysed)
    out.counts = pd.DataFrame(counts).reindex(analysed).fillna(0).astype(int)
    out.runtime_s = round(time.time() - started, 2)
    return out


def describe(sens: Sensitivity) -> str:
    """What happened to the ranking as the threshold rose, in sentences."""
    if not sens.available:
        return f"Not re-tested: {sens.reason}." if sens.reason else ""
    leader = sens.channels[0] if sens.channels else None
    parts = []
    for threshold, who, tied, out in zip(sens.thresholds, sens.leaders, sens.tied,
                                         sens.stands_out, strict=True):
        factor = threshold / sens.base_threshold
        label = (f"At {threshold:g} SD" + (" (the window's own)" if abs(factor - 1.0) < 1e-9
                                            else f" ({factor:g}×)"))
        if who is None:
            parts.append(f"{label}: nothing detected on any channel.")
            continue
        rate = float(sens.rates.loc[who, threshold]) if who in sens.rates.index else float("nan")
        others = [c for c in tied if c != who]
        sentence = f"{label}: <b>{who}</b> leads at {rate:.0f}/min"
        if others:
            sentence += f", tied with {', '.join(others[:4])}" + (" and others" if len(others) > 4 else "")
        if out is False:
            sentence += "; no channel stands out from the median"
        parts.append(sentence + ".")
    survives = sens.survives_to()
    if leader is None:
        verdict = ""
    elif survives is None:
        verdict = f"{leader} does not lead even at the window's own threshold here."
    elif survives >= max(sens.thresholds) / sens.base_threshold - 1e-9:
        verdict = f"<b>{leader} leads at every threshold tested</b>, up to {survives:g}× the window's."
    else:
        verdict = (f"<b>{leader} leads up to {survives:g}× the threshold and not beyond</b>: "
                   "the ranking depends on the threshold, which is a property of this "
                   "recording's signal-to-noise, not of the channel.")
    return " ".join(parts + ([verdict] if verdict else []))


def as_dict(sens: Sensitivity) -> dict:
    """What the assistant is handed: the same numbers the panel shows."""
    if not sens.available:
        return {"available": False, "reason": sens.reason}
    rows = {}
    for channel in sens.channels:
        rows[channel] = {f"{t:g}": (round(float(sens.rates.loc[channel, t]), 1)
                                    if channel in sens.rates.index else None)
                         for t in sens.thresholds}
    return {
        "available": True,
        "detector": sens.detector,
        "base_threshold_sd": sens.base_threshold,
        "thresholds_sd": [float(t) for t in sens.thresholds],
        "leading_channels_rate_per_min": rows,
        "leader_at_each_threshold": sens.leaders,
        "leader_stands_out_at_each_threshold": sens.stands_out,
        "window_leader_survives_to_factor": sens.survives_to(),
        "note": ("Rates are over the whole window, so the first column can differ a little "
                 "from the ranking. A leader that holds to 2x is robust to the threshold."),
    }
