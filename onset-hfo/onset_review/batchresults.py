"""A finished batch read across its recordings: Qt-free.

The batch table is a row per recording. This reads down the columns instead,
for what a row cannot show:

* **The busiest channel in each recording**, its rate with its interval and
  the median channel's rate beside it, so it is plain which recordings have
  a channel that stands out and which only have a busiest one.
* **The same busiest channel again**: for a subject with several windows in
  the batch, how often the busiest channel was the same one.
* **Agreement with the experts, by detector**: for the recordings that carry
  expert markings, each detector's F1 in every one of them, so the spread is
  seen, not only a mean.
* **Recordings that differ from the rest**: on the event rate, the share of
  contacts set aside, the busiest channel's rate and the agreement with the
  experts, a recording more than 3.5 robust z from the batch's median
  (Iglewicz & Hoaglin's modified z, from the median absolute deviation). It
  needs five analysed recordings; with fewer, nothing is called an outlier.

None of it is a new analysis: every number is read from ``summary.csv`` and
``scores.csv``, which the batch wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Across", "across", "robust_z", "draw_leaders", "draw_agreement", "short_labels",
           "OUTLIER_Z", "MIN_FOR_OUTLIERS", "MEASURES"]

#: |modified z| above this is called different from the rest (Iglewicz &
#: Hoaglin's recommendation).
OUTLIER_Z = 3.5
#: Fewer analysed recordings than this and nothing is called an outlier.
MIN_FOR_OUTLIERS = 5

#: (column, what it is, how it is shown). Each is read per recording.
MEASURES = (
    ("events_per_min", "accepted events per minute", "{:.1f}"),
    ("set_aside_share", "share of contacts set aside", "{:.0%}"),
    ("leader_rate_per_min", "the busiest channel's rate (/min)", "{:.1f}"),
    ("expert_f1", "F1 against the experts (first detector)", "{:.2f}"),
)


def robust_z(values) -> np.ndarray:
    """Modified z: 0.6745 (x − median) / MAD. When over half the values are
    equal (MAD 0) the mean absolute deviation stands in (× 1.2533); when
    every value is equal, every z is 0."""
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    out = np.full(x.shape, np.nan)
    if not finite.size:
        return out
    centre = np.median(finite)
    mad = np.median(np.abs(finite - centre))
    if mad > 0:
        out = 0.6745 * (x - centre) / mad
    else:
        mean_ad = np.mean(np.abs(finite - centre))
        out = (x - centre) / (1.2533 * mean_ad) if mean_ad > 0 else np.where(
            np.isfinite(x), 0.0, np.nan)
    return out


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _yes(value) -> bool:
    return value is True or str(value).strip().lower() in ("true", "1", "yes")


@dataclass
class Across:
    """What a batch says read across its recordings."""

    #: One row per analysed recording: recording, subject, leader, rate,
    #: ci_low, ci_high, median, stands_out, n_tied, and the measures.
    leaders: pd.DataFrame
    #: Expert scores, one row per recording and detector (empty without markings).
    scores: pd.DataFrame
    #: Per detector: n, median F1 and its quartiles, median ρ.
    agreement: pd.DataFrame
    #: recording, measure, value, batch median, z: the ones beyond OUTLIER_Z.
    outliers: pd.DataFrame
    #: subject, n windows, the channel most often busiest, how often.
    recurring: pd.DataFrame
    n_rows: int = 0
    n_failed: int = 0
    sentences: list[str] = field(default_factory=list)

    @property
    def can_flag(self) -> bool:
        return len(self.leaders) >= MIN_FOR_OUTLIERS


def across(rows: pd.DataFrame, scores: pd.DataFrame | None = None) -> Across:
    """Read a batch's table (and its expert scores) across its recordings."""
    rows = rows.copy()
    error = rows["error"].astype(str).str.strip() if "error" in rows else pd.Series(
        "", index=rows.index)
    error = error.where(~error.isin(["nan", "None"]), "")
    done = rows[error == ""].copy()
    leaders = pd.DataFrame({
        "recording": done.get("recording", pd.Series(dtype=str)).astype(str),
        "subject": done.get("subject", pd.Series(dtype=str)).astype(str),
        "leader": done.get("leader", pd.Series(dtype=str)).astype(str),
        "rate": [_number(v) for v in done.get("leader_rate_per_min", [])],
        "ci_low": [_number(v) for v in done.get("leader_ci_low", [np.nan] * len(done))],
        "ci_high": [_number(v) for v in done.get("leader_ci_high", [np.nan] * len(done))],
        "median": [_number(v) for v in done.get("median_rate_per_min",
                                                 [np.nan] * len(done))],
        "stands_out": [_yes(v) for v in done.get("stands_out", [False] * len(done))],
        "n_tied": [_number(v) for v in done.get("n_tied", [np.nan] * len(done))],
    })
    minutes = np.array([_number(v) for v in done.get("analysed_seconds",
                                                      [np.nan] * len(done))]) / 60.0
    events = np.array([_number(v) for v in done.get("events_total", [np.nan] * len(done))])
    channels = np.array([_number(v) for v in done.get("channels", [np.nan] * len(done))])
    set_aside = np.array([_number(v) for v in done.get("quality_set_aside",
                                                        [np.nan] * len(done))])
    with np.errstate(divide="ignore", invalid="ignore"):
        leaders["events_per_min"] = np.where(minutes > 0, events / minutes, np.nan)
        leaders["set_aside_share"] = np.where(channels > 0, set_aside / channels, np.nan)
    leaders["leader_rate_per_min"] = leaders["rate"]
    leaders["expert_f1"] = [_number(v) for v in done.get("expert_f1", [np.nan] * len(done))]
    leaders = leaders.reset_index(drop=True)

    outliers = []
    if len(leaders) >= MIN_FOR_OUTLIERS:
        for column, what, fmt in MEASURES:
            values = leaders[column].to_numpy(dtype=float)
            if np.isfinite(values).sum() < MIN_FOR_OUTLIERS:
                continue
            z = robust_z(values)
            centre = float(np.nanmedian(values))
            for i in np.flatnonzero(np.abs(np.nan_to_num(z)) > OUTLIER_Z):
                outliers.append({"recording": leaders.at[i, "recording"], "measure": what,
                                 "value": fmt.format(values[i]), "median": fmt.format(centre),
                                 "z": round(float(z[i]), 1),
                                 "direction": "higher" if z[i] > 0 else "lower"})
    outliers = pd.DataFrame(outliers, columns=["recording", "measure", "value", "median",
                                               "z", "direction"])

    recurring = []
    for subject, part in leaders.groupby("subject", sort=True):
        named = part[part["leader"].str.strip() != ""]
        if len(named) < 2:
            continue
        counts = named["leader"].value_counts()
        recurring.append({"subject": subject, "windows": len(named),
                          "leader": counts.index[0], "times": int(counts.iloc[0])})
    recurring = pd.DataFrame(recurring, columns=["subject", "windows", "leader", "times"])

    scores = scores.copy() if scores is not None and len(scores) else pd.DataFrame(
        columns=["recording", "detector", "f1", "rank_rho"])
    agreement = []
    for detector, part in scores.groupby("detector", sort=False):
        f1 = part["f1"].astype(float).dropna()
        rho = part["rank_rho"].astype(float).dropna() if "rank_rho" in part else pd.Series()
        if not len(f1):
            continue
        agreement.append({"detector": detector, "n": len(f1),
                          "median_f1": float(f1.median()), "q1": float(f1.quantile(0.25)),
                          "q3": float(f1.quantile(0.75)), "min": float(f1.min()),
                          "max": float(f1.max()),
                          "median_rho": float(rho.median()) if len(rho) else np.nan})
    agreement = pd.DataFrame(agreement, columns=["detector", "n", "median_f1", "q1", "q3",
                                                 "min", "max", "median_rho"])
    result = Across(leaders=leaders, scores=scores, agreement=agreement, outliers=outliers,
                    recurring=recurring, n_rows=len(rows), n_failed=int((error != "").sum()))
    result.sentences = _sentences(result)
    return result


def _sentences(a: Across) -> list[str]:
    from onset_review.studycharts import DETECTOR_LABELS

    n = len(a.leaders)
    if not n:
        return [f"None of the {a.n_rows} recordings was analysed."]
    out = [f"{n} of {a.n_rows} recordings analysed"
           + (f"; {a.n_failed} not (their reasons are in the table)." if a.n_failed else ".")]
    standing = int(a.leaders["stands_out"].sum())
    out.append(f"In all {n} the busiest channel stands out from the rest." if standing == n
               else
               f"In {standing} of {n} the busiest channel stands out from the rest; in the "
               f"other {n - standing} it is tied with others, and is only the busiest."
               if standing else
               f"In none of the {n} does the busiest channel stand out from the rest: each "
               "is tied with others, so no ranking should be read from them.")
    for row in a.recurring.itertuples():
        out.append(f"{row.subject}: the busiest channel was {row.leader} in {row.times} of "
                   f"{row.windows} windows.")
    if len(a.agreement):
        best = a.agreement.sort_values("median_f1", ascending=False).iloc[0]
        label = DETECTOR_LABELS.get(best["detector"], best["detector"])
        out.append(f"{int(a.agreement['n'].max())} recording(s) carry expert markings; "
                   f"{label} agreed best with them (median F1 {best['median_f1']:.2f}, "
                   f"{best['min']:.2f}–{best['max']:.2f}).")
    else:
        out.append("None of these recordings carries expert markings, so agreement with "
                   "the experts cannot be shown.")
    if not a.can_flag:
        out.append(f"With fewer than {MIN_FOR_OUTLIERS} recordings analysed, none is called "
                   "different from the rest.")
    elif len(a.outliers):
        names = sorted(set(a.outliers["recording"]))
        out.append(f"{len(names)} recording(s) differ from the rest on at least one measure "
                   f"(more than {OUTLIER_Z:g} robust z from the batch median): "
                   + "; ".join(names) + ".")
    else:
        out.append(f"No recording differs from the rest by more than {OUTLIER_Z:g} robust z "
                   "on any measure.")
    return out


# -- figures (matplotlib, no Qt) -------------------------------------------------------------
def _short(label: str, width: int = 38) -> str:
    return label if len(label) <= width else label[:width - 1] + "…"


def short_labels(names) -> list[str]:
    """Recording labels without the parts every one of them shares at the
    front ("ds003498 · "), so the part that differs fits beside a chart."""
    parts = [str(n).split(" · ") for n in names]
    drop = 0
    while parts and all(len(p) > drop + 1 for p in parts) \
            and len({p[drop] for p in parts}) == 1:
        drop += 1
    return [_short(" · ".join(p[drop:])) for p in parts]


def draw_leaders(figure, axis, leaders: pd.DataFrame, highlight: str | None = None) -> dict:
    """Each recording's busiest channel: its rate as a dot with its interval,
    filled when it stands out, open when tied; the median channel's rate as
    a short grey tick. Returns the dots by recording, for a window that
    answers the pointer."""
    from onset_review.studycharts import GRID, MARK, MUTED, SERIES, _quiet

    _quiet(axis)
    axis.yaxis.grid(False)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    frame = leaders.reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    points = {}
    for yi, row in zip(y, frame.itertuples(), strict=True):
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            axis.plot([row.ci_low, row.ci_high], [yi, yi], color=MUTED, linewidth=1.2,
                      solid_capstyle="butt")
        if np.isfinite(row.median):
            axis.plot([row.median, row.median], [yi - 0.28, yi + 0.28], color=MUTED,
                      linewidth=2, alpha=0.55)
        colour = SERIES[0]
        edge = MARK if row.recording == highlight else colour
        (dot,) = axis.plot([row.rate], [yi], marker="o", markersize=7,
                           markerfacecolor=colour if row.stands_out else "white",
                           markeredgecolor=edge, markeredgewidth=1.6, linestyle="none")
        points[row.recording] = dot
        right = row.ci_high if np.isfinite(row.ci_high) else row.rate
        axis.annotate(row.leader, (right, yi), xytext=(5, 0), textcoords="offset points",
                      va="center", fontsize=8, color=MUTED)
    axis.set_yticks(y)
    axis.set_yticklabels(short_labels(frame["recording"]), fontsize=8)
    axis.set_ylim(-0.7, len(frame) - 0.3)
    axis.set_xlabel("busiest channel's rate, events per minute (line: 95% interval; "
                    "grey tick: the median channel)", fontsize=8, color=MUTED)
    lo, hi = axis.get_xlim()
    axis.set_xlim(left=min(0, lo), right=hi * 1.12 if hi > 0 else 1)
    figure.tight_layout()
    return points


def draw_agreement(figure, axis, scores: pd.DataFrame, metric: str = "f1") -> dict:
    """Each detector's agreement with the experts in every recording with
    markings, a dot per recording and a bar at the median."""
    from onset_review.studycharts import DETECTOR_LABELS, MARK, MUTED, SERIES, _quiet

    _quiet(axis)
    order = [d for d in ("rms", "line_length", "hilbert", "energy")
             if d in set(scores["detector"])]
    order += [d for d in scores["detector"].unique() if d not in order]
    rng = np.random.default_rng(0)        # fixed jitter: the same figure every time
    points = {}
    for x, detector in enumerate(order):
        values = scores.loc[scores["detector"] == detector, metric].astype(float).dropna()
        if not len(values):
            continue
        jitter = rng.uniform(-0.12, 0.12, len(values))
        dots = axis.scatter(x + jitter, values, s=28, color=SERIES[0], alpha=0.85,
                            edgecolors="white", linewidths=0.6, zorder=3)
        points[detector] = dots
        axis.plot([x - 0.25, x + 0.25], [values.median()] * 2, color=MARK, linewidth=2,
                  zorder=4)
    counts = [int(scores.loc[scores["detector"] == d, metric].notna().sum()) for d in order]
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([f"{DETECTOR_LABELS.get(d, d)}\n{n} recordings"
                          for d, n in zip(order, counts, strict=True)], fontsize=8)
    axis.set_xlim(-0.6, len(order) - 0.4)
    axis.set_ylim(-0.08, 1.02)
    axis.set_ylabel("F1 against the experts" if metric == "f1" else metric, fontsize=8,
                    color=MUTED)
    figure.tight_layout()
    return points
