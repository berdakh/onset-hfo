"""The two figures the study pages draw from their own tables. No Qt.

A newcomer reads a figure before a table. The Detectors page gets the
threshold sweep as lines, one per detector, with the chosen operating point
marked; the Outcome page gets every patient as a dot, by outcome, so the AUC
in the text is something a reader can see rather than take on trust.

Both are drawn with matplotlib's Agg backend into a PNG the page embeds, from
the same frames the tables beside them render, so the figure and the table
cannot disagree. Two hues only, validated for colour-vision deficiency and
for contrast on the page's white: the window's accent blue for the first
series and an orange for the second. Everything else is ink: text in the
text colours, grid and axes recessive, no title inside the figure (the page
heading above it is the title).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["sweep_chart", "outcome_chart", "SERIES", "DETECTOR_LABELS"]

#: Categorical hues, in fixed order: the accent, then an orange. Checked with
#: the palette validator: adjacent-pair CVD separation 26.6, contrast >= 3:1.
SERIES = ("#0B6BCB", "#C2410C")
TEXT = "#1D1D1F"
MUTED = "#6E6E73"
GRID = "#E5E5EA"

DETECTOR_LABELS = {"rms": "RMS energy", "line_length": "Line length",
                   "hilbert": "Hilbert envelope", "energy": "Short-time energy"}
METRIC_LABELS = {"rank_rho": "channel-rank ρ (agreement with the experts)",
                 "f1": "F1", "precision": "precision", "recall": "recall",
                 "detections": "detections per 60 s", "top5_overlap": "top-5 channels shared"}


def _figure(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), dpi=150)
    return plt, figure, axes


def _quiet(axis) -> None:
    """Recessive axes: no top or right spine, grey left and bottom, a light
    horizontal grid, muted tick labels."""
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9, length=3)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def sweep_chart(frame, band: str, metric: str, best, path: Path) -> Path:
    """The threshold sweep: `metric` against threshold (SD), one line per
    detector, the operating point chosen by the page's criterion marked and
    named. `frame` is the committed sweep, `best` the operating-point rows."""
    part = frame[frame["band"] == band].sort_values("threshold_sd")
    detectors = [d for d in ("rms", "line_length", "hilbert", "energy")
                 if d in set(part["detector"])]
    plt, figure, axis = _figure(8.4, 3.4)
    _quiet(axis)
    for index, detector in enumerate(detectors):
        rows = part[part["detector"] == detector].dropna(subset=[metric])
        colour = SERIES[index % len(SERIES)]
        axis.plot(rows["threshold_sd"], rows[metric], color=colour, linewidth=2,
                  marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.5,
                  solid_capstyle="round", label=DETECTOR_LABELS.get(detector, detector))
        if len(rows):
            last = rows.iloc[-1]
            axis.annotate(DETECTOR_LABELS.get(detector, detector),
                          (last["threshold_sd"], last[metric]), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=9, color=TEXT)
        chosen = best[(best["band"] == band) & (best["detector"] == detector)] \
            if best is not None else None
        if chosen is not None and len(chosen):
            at = float(chosen["threshold_sd"].iloc[0])
            hit = rows[rows["threshold_sd"] == at]
            if len(hit):
                value = float(hit[metric].iloc[0])
                axis.plot([at], [value], marker="o", markersize=9, color=colour,
                          markeredgecolor="white", markeredgewidth=2, zorder=5)
                axis.annotate(f"chosen: {at:g} SD, {value:.2f}", (at, value),
                              xytext=(0, 11), textcoords="offset points", ha="center",
                              fontsize=9, color=TEXT)
    axis.set_xlabel("threshold (standard deviations above the baseline)", color=MUTED,
                    fontsize=9)
    axis.set_ylabel(METRIC_LABELS.get(metric, metric), color=MUTED, fontsize=9)
    axis.set_xlim(left=max(0.0, float(part["threshold_sd"].min()) - 0.3))
    axis.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=TEXT)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return path


def outcome_chart(cohort, groups, path: Path) -> Path:
    """Every patient as a dot: the share of the tied busiest set that the
    surgeon removed, by outcome, for the expert markings and for our
    detector side by side, with each group's mean as a bar and the AUC
    the page reports written in the corner."""
    import numpy as np

    plt, figure, axes = _figure(8.4, 3.2)
    axes.remove()
    panels = figure.subplots(1, 2, sharex=True, sharey=True)
    free = cohort["seizure_free"].astype(bool)
    # Seizure-free on top, as the text reads it; recurrence below.
    rows = [("recurrence", ~free, SERIES[1]), ("seizure-free", free, SERIES[0])]
    rng = np.random.default_rng(7)
    for axis, (source, title) in zip(panels, (("expert", "Expert markings"),
                                              ("rms", "Our RMS detector")), strict=True):
        _quiet(axis)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.yaxis.grid(False)
        column = f"candidates_resected_{source}"
        for level, (_label, mask, colour) in enumerate(rows):
            values = cohort.loc[mask, column].astype(float).to_numpy()
            y = level + rng.uniform(-0.26, 0.26, size=len(values))
            axis.scatter(values, y, s=46, color=colour, edgecolor="white", linewidth=1.5,
                         alpha=0.9, zorder=4)
            mean = float(values.mean()) if len(values) else float("nan")
            axis.plot([mean, mean], [level - 0.32, level + 0.32], color=colour, linewidth=2.5,
                      zorder=5)
            axis.annotate(f"mean {mean:.2f}", (mean, level + 0.36), ha="center", va="bottom",
                          fontsize=8.5, color=TEXT)
        axis.set_yticks([0, 1])
        axis.set_yticklabels([f"{label} ({int(mask.sum())})" for label, mask, _ in rows],
                             color=TEXT, fontsize=9)
        axis.set_ylim(-0.6, 1.75)
        axis.set_xlim(-0.05, 1.05)
        heading = title
        if groups is not None and len(groups):
            match = groups[(groups["source"] == source)
                           & (groups["metric"] == "candidates_resected")]
            if len(match):
                row = match.iloc[0]
                heading = (f"{title}\nAUC {row['auc']:.2f} "
                           f"({row['auc_lo']:.2f}–{row['auc_hi']:.2f}), p = {row['p_permutation']:.2f}")
        axis.set_title(heading, loc="left", fontsize=9.5, color=TEXT, linespacing=1.4)
        axis.set_xlabel("share of the tied busiest channels inside the resection",
                        color=MUTED, fontsize=9)
    figure.tight_layout(w_pad=2.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return path
