"""Scoring the detectors against expert HFO markings, across a cohort.

This module exists because of one sentence that used to be in
``docs/EVALUATION.md``: *"no HFO ground truth exists for this public
recording, so precision and recall are not reported here"*. That was true of
the ictal dataset. It is not true of ``ds003498`` -- the Zurich interictal
sleep recordings, where the authors marked HFOs channel by channel -- so the
honest thing is to measure.

Three numbers come out, and they answer different questions:

**Event agreement** (precision, recall, F1). Did we mark the same events?
Strict, and strictness is the point, but read it knowing what the reference
is: the published events are what the original study's Morphology detector
found *and* a human validated. They are not an exhaustive census of every
oscillation in the recording. So "recall" here means "recall of that
detector's validated events", and an event we find that it did not is
counted against us even when it may be real.

**Channel-rate agreement** (Spearman rho over reviewed channels, and top-k
overlap). Do we rank the same channels as active? This is closer to the
clinical question -- nobody operates on an event, they operate on tissue --
and it tolerates the event-matching quibbles above.

**Both as a function of threshold.** A detector is a family of detectors, one
per operating point, and quoting a single F1 hides the choice that matters
most. Every function here sweeps.

Only channels the annotators reviewed are scored, on either measure. In this
dataset that is a subset (the study kept the three most mesial bipolar
channels in temporal-lobe cases), and counting a detection on an unreviewed
channel as a false positive would punish us for reading more of the recording
than the reviewer did.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.config import BANDS, PIPELINE_VERSION, RESULTS_DIR, DetectorConfig
from onset_hfo.datasets import fetch_slice, list_subjects
from onset_hfo.detectors import HFO_DETECTORS
from onset_hfo.evaluate import evaluate_detections
from onset_hfo.preprocess import prepare
from onset_hfo.validate import validate_events

__all__ = ["DEFAULT_THRESHOLDS", "benchmark_subject", "benchmark_cohort", "BenchmarkResult"]

#: Operating points swept by default. 5.0 is the literature default the
#: pipeline ships with; the rest are there to show what it costs.
DEFAULT_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)

#: The shared registry. This file used to keep its own two-entry copy,
#: so `--detectors hilbert` failed with a KeyError rather than sweeping.
DETECTORS = HFO_DETECTORS


@dataclass
class BenchmarkResult:
    """Everything one benchmark run produced."""

    scores: pd.DataFrame          #: one row per subject x detector x band x threshold
    ranking: pd.DataFrame         #: channel-rate agreement, same keys
    subjects: pd.DataFrame        #: what was analysed per subject
    dataset: str = "ds003498"
    window_s: tuple[float, float] = (0.0, 60.0)
    pipeline_version: str = PIPELINE_VERSION

    def summary(self, band: str = "ripple") -> pd.DataFrame:
        """Cohort means per detector and threshold -- the table for the docs."""
        scores = self.scores[self.scores["band"] == band]
        ranks = self.ranking[self.ranking["band"] == band]
        agg = scores.groupby(["detector", "threshold_sd"]).agg(
            precision=("precision", "mean"), recall=("recall", "mean"),
            f1=("f1", "mean"), detections=("n_detections", "mean")).round(3)
        rho = ranks.groupby(["detector", "threshold_sd"]).agg(
            rank_rho=("spearman_rho", "mean"),
            top5_overlap=("top5_overlap", "mean")).round(3)
        return agg.join(rho).reset_index()

    def best_threshold(self, band: str = "ripple", by: str = "f1") -> dict:
        """The operating point the data prefers, by F1 or by rank agreement."""
        table = self.summary(band)
        column = {"f1": "f1", "rank": "rank_rho"}[by]
        row = table.loc[table[column].idxmax()]
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in row.items()}

    def save(self, directory: str | Path | None = None) -> Path:
        out = Path(directory or RESULTS_DIR) / f"benchmark_{self.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        self.scores.to_csv(out / "scores.csv", index=False)
        self.ranking.to_csv(out / "ranking.csv", index=False)
        self.subjects.to_csv(out / "subjects.csv", index=False)
        for band in sorted(self.scores["band"].unique()):
            self.summary(band).to_csv(out / f"summary_{band}.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "dataset": self.dataset, "window_s": list(self.window_s),
            "pipeline_version": self.pipeline_version,
            "n_subjects": int(self.subjects["subject"].nunique()),
            "thresholds": sorted(self.scores["threshold_sd"].unique().tolist()),
        }, indent=2))
        print(f"[onset-hfo] benchmark written to {out}")
        return out


@dataclass
class _SubjectRun:
    rows: list[dict] = field(default_factory=list)
    ranks: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def benchmark_subject(subject: str, run: str = "01", t_start: float = 0.0, t_stop: float = 60.0,
                      dataset: str = "ds003498", thresholds=DEFAULT_THRESHOLDS,
                      detectors=("rms", "line_length"), bands=("ripple",),
                      top_k: int = 5, verbose: bool = True) -> _SubjectRun:
    """Score one subject's recording against its expert markings."""
    recording = fetch_slice(dataset=dataset, subject=subject, run=run,
                            t_start=t_start, t_stop=t_stop, verbose=verbose)
    if recording.ground_truth is None or not len(recording.ground_truth):
        raise ValueError(f"{subject} run-{run} carries no expert markings")
    prep = prepare(recording, verbose=False)
    truth = recording.ground_truth
    # Only channels that were reviewed AND survived preprocessing.
    reviewed = [c for c in recording.reviewed_channels if c in prep.ch_names]

    out = _SubjectRun(meta={
        "subject": subject, "run": run, "duration_s": prep.duration,
        "sfreq_hz": prep.sfreq, "n_channels": prep.n_channels,
        "n_reviewed_channels": len(reviewed),
        "n_expert_events": int(len(truth)),
        "expert_ripples": int((truth["kind"] == "ripple").sum()),
        "expert_fast_ripples": int((truth["kind"] == "fast_ripple").sum()),
    })
    if not reviewed:
        return out

    for band_name in bands:
        band = getattr(BANDS, band_name)
        if not BANDS.usable(prep.sfreq, band):
            continue
        expert_rate = truth[truth["kind"] == band_name].groupby("channel").size() \
            .reindex(reviewed, fill_value=0)
        for detector in detectors:
            for threshold in thresholds:
                cfg = DetectorConfig(band=band, threshold_sd=float(threshold))
                events = DETECTORS[detector](prep, cfg, channels=reviewed)
                validate_events(events, prep)
                score = evaluate_detections(events, truth, kind=band_name, channels=reviewed)
                row = {**score.as_dict(), "subject": subject, "run": run, "band": band_name,
                       "detector": detector, "threshold_sd": float(threshold)}
                row.pop("false_positive_causes", None)
                out.rows.append(row)

                ours = pd.Series({c: sum(1 for e in events if e.accepted and e.channel == c)
                                  for c in reviewed})
                out.ranks.append({"subject": subject, "run": run, "band": band_name,
                                  "detector": detector, "threshold_sd": float(threshold),
                                  **_rank_agreement(expert_rate, ours, top_k)})
    if verbose:
        best = max((r for r in out.rows if r["band"] == bands[0]), key=lambda r: r["f1"], default=None)
        if best:
            print(f"[onset-hfo]   best F1 {best['f1']:.2f} at {best['threshold_sd']:g} SD "
                  f"({best['detector']}), {out.meta['n_expert_events']} expert events, "
                  f"{len(reviewed)} reviewed channels")
    return out


def _rank_agreement(expert: pd.Series, ours: pd.Series, top_k: int) -> dict:
    """Do we call the same channels active? Spearman plus a top-k overlap.

    Spearman rather than Pearson because only the ordering is claimed, and a
    top-k overlap alongside it because a correlation over 20-odd channels can
    look respectable while disagreeing about every channel anyone would look
    at.
    """
    from scipy.stats import spearmanr

    common = expert.index.intersection(ours.index)
    a, b = expert.loc[common].to_numpy(float), ours.loc[common].to_numpy(float)
    if len(common) < 3 or np.all(b == b[0]) or np.all(a == a[0]):
        rho, pvalue = float("nan"), float("nan")
    else:
        result = spearmanr(a, b)
        rho, pvalue = float(result.statistic), float(result.pvalue)
    k = min(top_k, len(common))
    top_expert = set(expert.loc[common].sort_values(ascending=False).head(k).index)
    top_ours = set(ours.loc[common].sort_values(ascending=False).head(k).index)
    return {"n_channels": int(len(common)), "spearman_rho": rho, "p_value": pvalue,
            f"top{top_k}_overlap": len(top_expert & top_ours), "top_k": k,
            "expert_top": ", ".join(sorted(top_expert)), "ours_top": ", ".join(sorted(top_ours))}


def benchmark_cohort(subjects: list[str] | None = None, n_subjects: int | None = None,
                     dataset: str = "ds003498", run: str = "01",
                     t_start: float = 0.0, t_stop: float = 60.0,
                     thresholds=DEFAULT_THRESHOLDS, detectors=("rms", "line_length"),
                     bands=("ripple",), verbose: bool = True) -> BenchmarkResult:
    """Run :func:`benchmark_subject` across a cohort and collect the tables.

    Subjects that fail (a missing run, a download that dies) are skipped with
    a message rather than aborting the cohort: a benchmark that needs twenty
    downloads to succeed in a row is a benchmark nobody reruns.
    """
    if subjects is None:
        subjects = list_subjects(dataset)
        if n_subjects:
            subjects = subjects[:n_subjects]
    rows, ranks, meta = [], [], []
    for i, subject in enumerate(subjects, 1):
        if verbose:
            print(f"\n[onset-hfo] ({i}/{len(subjects)}) {subject}")
        try:
            result = benchmark_subject(subject, run=run, t_start=t_start, t_stop=t_stop,
                                       dataset=dataset, thresholds=thresholds,
                                       detectors=detectors, bands=bands, verbose=verbose)
        except Exception as exc:
            print(f"[onset-hfo]   skipped {subject}: {type(exc).__name__}: {exc}")
            continue
        rows += result.rows
        ranks += result.ranks
        meta.append(result.meta)
    return BenchmarkResult(scores=pd.DataFrame(rows), ranking=pd.DataFrame(ranks),
                           subjects=pd.DataFrame(meta), dataset=dataset,
                           window_s=(t_start, t_stop))
