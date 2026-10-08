"""Does the HFO map point at the tissue whose removal cured the patient?

This is the only question in the package whose answer a surgeon would act
on, and the only one with a reference standard that is not another
algorithm: **what happened to the patient after surgery**.

The design, and why each piece is there
---------------------------------------

For every subject in ``ds003498`` we know three things: where the HFOs are
(expert markings, and our detector), which contacts were removed (the
clinical sheet -- see :mod:`onset_hfo.clinical`), and whether the patient
became seizure-free (``participants.tsv``). So for each subject we compute
one number -- **the share of that subject's HFO activity that sat in tissue
the surgeon removed** -- and ask whether it is higher in the patients who
became seizure-free. That is the claim of Fedele et al. 2017, the study this
dataset comes from, and it is the claim the HFO Trial (Jacobs et al., Lancet
Neurology 2022) failed to reproduce prospectively.

Four design choices are deliberate, and each of them can only lower the
result:

**An expert positive control.** Every number is computed twice: once from
the published expert markings and once from our detector, on the same
channels, with the same resection labels. This is what makes a null
interpretable. If our detector separates the groups but the expert markings
do not, something is wrong with the analysis. If neither separates them, the
limit is the 60-second window or the cohort size, not the detector. Only
"experts separate, we do not" is evidence against *us*.

**A fixed operating point, chosen before this analysis.** The detector runs
at ``THRESHOLDS["interictal-agreement"]`` (2.0 SD), which was selected on
*channel-ranking agreement with the experts* in
:mod:`onset_hfo.benchmark` -- a different question, on data that says
nothing about outcome. Tuning a threshold against outcome and then reporting
the outcome result would be circular, and with 20 subjects it would be
trivially easy to do by accident.

**Margin channels are not claimed.** A bipolar channel with one contact
inside the resection is ``partial`` (see
:func:`onset_hfo.clinical.classify_channels`). It counts in the denominator
and not the numerator, so a detector that fires along the resection margin
gets no credit for it.

**Both bands, because the source study's claim rests on the faster one.**
Fedele et al. built their prediction on fast ripples (and on ripples and
fast ripples co-occurring), not on ripples alone, and the wider literature
agrees that ripples are the less specific marker. These recordings are
sampled at 2 kHz, so the 250-500 Hz band is honestly analysable -- unlike
the 1 kHz ictal dataset -- and reporting only the ripple arm would test a
weaker version of the published claim.

**Two channel scopes, both reported.** ``reviewed`` uses only the channels
the original study annotated -- the fair head-to-head with the experts, but
a set that already reflects clinical judgement (in temporal-lobe cases the
study kept the three most mesial bipolar channels). ``all`` uses every
channel that survives preprocessing, which is what a deployed tool would
face and is the harder test. Reporting only the flattering one would be a
choice made after seeing both.

What this cannot be
-------------------

Thirteen seizure-free patients and seven recurrences is a very small study.
:func:`min_detectable_auc` says in one number how small: with these group
sizes, only a *large* separation could reach significance at all, so a
p-value above 0.05 here means "underpowered", not "no effect". Every result
this module prints carries an effect size and a confidence interval for
exactly that reason, and :meth:`OutcomeResult.summary` prints the power
floor next to the p-values rather than in a footnote.

Twelve to sixteen comparisons come out of one run (metric x source x scope
x band). Nothing here is corrected for that, because nothing here is being
claimed as a finding -- but it means a single p just under 0.05 in that
table is what a table that size produces by chance, and should be read as
a hypothesis for a larger study rather than a result.

One 60-second slice of one night is also not what the source study used: it
scored whole interictal sleep recordings over several nights. A negative
result here is first of all a statement about 60 seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.clinical import classify_channels, fetch_participants, resection_map
from onset_hfo.config import BANDS, PIPELINE_VERSION, RESULTS_DIR, THRESHOLDS, DetectorConfig
from onset_hfo.datasets import fetch_slice, list_subjects
from onset_hfo.detectors import HFO_DETECTORS
from onset_hfo.preprocess import prepare

__all__ = [
    "NotACount",
    "OutcomeResult",
    "candidate_channels",
    "outcome_subject",
    "outcome_study",
    "compare_groups",
    "min_detectable_auc",
    "rank_comparison",
]

class NotACount(ValueError):
    """A Poisson-interval metric was handed something that is not an event count.

    Its own error, rather than a bare ``ValueError``, because the thing a
    caller usually wants to do about it is not "fix the number" but "do not
    report this metric for this rule" -- and that distinction is worth being
    able to catch.
    """


DETECTORS = HFO_DETECTORS

#: Length of one ``ds003498`` run, in seconds. Every run in the archive is
#: exactly this long, so this is not a slice -- it is the whole recording and
#: every expert marking in it.
#:
#: It is the default because the first version of this analysis used 60 s and
#: got a *different answer*: the expert arm reached AUC 0.82 (p = 0.007) on the
#: first minute and 0.71 (p = 0.12) on the whole run. Two patients' busiest
#: fast-ripple channel moved in or out of the resection when the other four
#: minutes were included. A window short enough to change the conclusion is not
#: a defensible default, however much faster it is.
FULL_RUN_S = 300.0

#: Operating point used for the detector arm, **per band**. Both values come
#: from :mod:`onset_hfo.benchmark`, where they were chosen on channel-ranking
#: agreement with the expert markings -- a question that says nothing about
#: outcome, so using them here is not circular.
#:
#: They differ by a factor of two and a half, which is the point of having
#: two of them. Ripples sit on a noisier background and a 5 SD cut finds
#: almost nothing; fast ripples sit on a quieter one, and a 2 SD cut there
#: buries the real events under an order of magnitude of background. Applying
#: the ripple operating point to the fast-ripple band -- which is what the
#: first version of this analysis did -- destroys exactly the channel ranking
#: the outcome question depends on.
BAND_THRESHOLD_SD: dict[str, float] = {
    "ripple": THRESHOLDS["interictal-agreement"],   # 2.0 SD, rank rho 0.655
    "fast_ripple": THRESHOLDS["literature"],        # 5.0 SD, rank rho 0.610
}

#: Fallback for a band with no measured operating point.
DEFAULT_THRESHOLD_SD = THRESHOLDS["interictal-agreement"]


def _threshold_for(band: str, threshold_sd) -> float:
    """Resolve the operating point for one band.

    ``threshold_sd`` may be a number (the same cut in every band, for
    sensitivity analyses) or a mapping from band name to number (the default,
    and the only defensible choice when bands are compared to each other).
    """
    if threshold_sd is None:
        threshold_sd = BAND_THRESHOLD_SD
    if isinstance(threshold_sd, dict):
        return float(threshold_sd.get(band, DEFAULT_THRESHOLD_SD))
    return float(threshold_sd)

#: Seizure-free vs recurrence, as ``participants.tsv`` spells them.
_FREE, _RECUR = "S", "F"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def rank_comparison(free: np.ndarray, recurrence: np.ndarray,
                    n_boot: int = 10000, seed: int = 0) -> dict:
    """Compare two small samples without assuming a distribution.

    Returns the common-language effect size (**AUC**: the probability that a
    randomly chosen seizure-free patient has a higher value than a randomly
    chosen recurrence patient, ties counted as half), a bootstrap confidence
    interval for it, the rank-biserial correlation, and an **exact
    permutation p-value** -- exact because with groups this small the normal
    approximation behind the usual Mann-Whitney p is not trustworthy, and
    enumerating every relabelling is cheap.

    AUC rather than a difference in means because the quantity being compared
    is a proportion on twenty patients: its mean is unstable and its rank is
    not.
    """
    free = np.asarray([v for v in free if np.isfinite(v)], dtype=float)
    recurrence = np.asarray([v for v in recurrence if np.isfinite(v)], dtype=float)
    n1, n2 = len(free), len(recurrence)
    out = {
        "n_seizure_free": n1, "n_recurrence": n2,
        "median_seizure_free": float(np.median(free)) if n1 else float("nan"),
        "median_recurrence": float(np.median(recurrence)) if n2 else float("nan"),
        # Means as well as medians: ``top_channel_resected`` is 0/1, and the
        # median of a binary variable is 0 in both groups however far apart
        # the proportions are.
        "mean_seizure_free": float(np.mean(free)) if n1 else float("nan"),
        "mean_recurrence": float(np.mean(recurrence)) if n2 else float("nan"),
        "iqr_seizure_free": _iqr(free), "iqr_recurrence": _iqr(recurrence),
        "auc": float("nan"), "auc_lo": float("nan"), "auc_hi": float("nan"),
        "rank_biserial": float("nan"), "p_permutation": float("nan"),
    }
    if n1 < 2 or n2 < 2:
        return out

    auc = _auc(free, recurrence)
    out["auc"] = auc
    out["rank_biserial"] = 2.0 * auc - 1.0
    out["p_permutation"] = _permutation_p(free, recurrence, auc, seed=seed)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = _auc(rng.choice(free, n1, replace=True),
                        rng.choice(recurrence, n2, replace=True))
    out["auc_lo"], out["auc_hi"] = (float(np.percentile(boots, 2.5)),
                                    float(np.percentile(boots, 97.5)))
    return out


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a == b) -- Mann-Whitney U scaled to [0, 1]."""
    if not len(a) or not len(b):
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(a) * len(b)))


def _iqr(values: np.ndarray) -> str:
    if not len(values):
        return ""
    lo, hi = np.percentile(values, [25, 75])
    return f"{lo:.3f}-{hi:.3f}"


def _permutation_p(a: np.ndarray, b: np.ndarray, auc: float, seed: int = 0,
                   exact_limit: int = 500000, n_draws: int = 200000) -> float:
    """Two-sided permutation p for the AUC, exact where that is affordable.

    Done on **midranks** rather than on the values. The AUC is a function of
    the ranks alone -- ``AUC = (sum of group-A ranks - n1(n1+1)/2) / (n1 n2)``,
    with average ranks giving ties their half-credit exactly as
    :func:`_auc` does -- so a relabelling is a sum over a row of an index
    matrix, and the whole null distribution is one vectorised numpy
    expression instead of a hundred thousand passes of a Python loop. With 13
    versus 7 patients that is the difference between minutes and milliseconds,
    which is why the exact path is affordable at all.
    """
    from itertools import chain, combinations
    from math import comb

    from scipy.stats import rankdata

    pooled = np.concatenate([a, b])
    n, n1, n2 = len(pooled), len(a), len(b)
    ranks = rankdata(pooled)
    offset = n1 * (n1 + 1) / 2.0
    observed = abs(auc - 0.5)
    total = comb(n, n1)

    if total <= exact_limit:
        index = np.fromiter(chain.from_iterable(combinations(range(n), n1)),
                            dtype=np.int64, count=total * n1).reshape(total, n1)
        sums = ranks[index].sum(axis=1)
    else:
        rng = np.random.default_rng(seed)
        draws = rng.random((n_draws, n)).argsort(axis=1)[:, :n1]
        sums = ranks[draws].sum(axis=1)
    aucs = (sums - offset) / (n1 * n2)
    hits = int((np.abs(aucs - 0.5) >= observed - 1e-12).sum())
    return float(hits / len(aucs)) if total <= exact_limit else float((hits + 1) / (len(aucs) + 1))


def min_detectable_auc(n1: int, n2: int, power: float = 0.80, alpha: float = 0.05,
                       n_sims: int = 2000, seed: int = 0) -> float:
    """The smallest effect these group sizes could detect, as an AUC.

    Simulated rather than derived: draw two normal samples separated so that
    the true AUC is a candidate value, run the same rank test, and count how
    often it reaches ``alpha``. The returned number is the candidate AUC at
    which that fraction first reaches ``power``.

    Read it as the honest ceiling on what a null result here means. If the
    floor is 0.85 and the observed AUC is 0.70, the study could not have
    detected 0.70 whether or not it is real.
    """
    from scipy.stats import mannwhitneyu, norm

    rng = np.random.default_rng(seed)
    for target in np.arange(0.55, 1.00, 0.01):
        delta = norm.ppf(target) * np.sqrt(2.0)
        # Every simulation at once. One row is one simulation's two groups,
        # drawn in the order a loop over simulations would draw them (the
        # first n1 values of the row, then n2), so the stream -- and so the
        # result -- is the same as one test at a time, about 150 times faster.
        draws = rng.standard_normal((n_sims, n1 + n2))
        pvalues = mannwhitneyu(draws[:, :n1] + delta, draws[:, n1:],
                               alternative="two-sided", axis=1).pvalue
        if float(np.mean(pvalues < alpha)) >= power:
            return float(round(target, 2))
    return float("nan")


# --------------------------------------------------------------------------
# Per-subject measurement
# --------------------------------------------------------------------------


def _share_in_resection(rates: pd.Series, zones: pd.Series) -> dict:
    """Share of HFO activity on resected channels, plus the raw counts.

    ``share`` is strict: ``partial`` (margin) channels sit in the denominator
    only. ``share_incl_partial`` credits them, and the gap between the two is
    how much of a subject's result rests on the margin.
    """
    total = float(rates.sum())
    inside = float(rates[zones == "resected"].sum())
    margin = float(rates[zones == "partial"].sum())
    return {
        "n_events": total,
        "n_events_resected": inside,
        "n_events_partial": margin,
        "n_events_spared": total - inside - margin,
        "share_in_rz": inside / total if total else float("nan"),
        "share_in_rz_incl_partial": (inside + margin) / total if total else float("nan"),
    }


def candidate_channels(counts: pd.Series, duration_min: float,
                       alpha: float = 0.05) -> list[str]:
    """The channels that cannot be told apart from the busiest one.

    A channel joins the set when its Poisson rate interval overlaps the
    leader's -- the same rule :func:`onset_hfo.metrics.leader_separation`
    already uses to decide whether a recording has a leader at all. It is a
    statement about intervals rather than a tuned constant, so it tightens by
    itself as the analysed window grows.

    The set exists because of a measurement, not a preference: across five
    disjoint minutes of the same ``ds003498`` recordings, the experts' busiest
    fast-ripple channel is the same channel in only 7 of 20 patients (see
    ``docs/OUTCOME.md``). An argmax over channels whose intervals overlap
    reports one of them as though the data had chosen it. Naming the whole
    tied set instead is what the data actually supports, and it is what a
    surgeon should be shown.

    Returned in descending rate order, ties broken by channel name so the
    same data always yields the same set.

    **Counts, not scores.** A Poisson interval is a statement about a number of
    events, so this refuses anything else rather than truncating it. The caller
    this protects against is the planner's own ranking: ``rate x robustness``
    is not a count and has no such interval, so the project's tie-aware metric
    is *undefined* for the multiplied rule -- which is a finding about the rule
    rather than a gap to paper over with ``int()``. Until now that was a
    convention held up by one comment in one script; a float arriving here was
    silently floored, and 8.37 "ripples" became an interval around 8.

    Raises ``NotACount`` when a value is not a whole number. Integral floats
    are fine: pandas holds counts as ``float64`` all over this codebase.
    """
    from onset_hfo.metrics import poisson_ci

    counts = counts[counts.notna()]
    if not len(counts) or not counts.sum() or duration_min <= 0:
        return []
    fractional = {str(ch): float(n) for ch, n in counts.items()
                  if not float(n).is_integer()}
    if fractional:
        shown = ", ".join(f"{ch}={n:g}" for ch, n in list(fractional.items())[:3])
        raise NotACount(
            f"candidate_channels needs integer event counts, got {len(fractional)} "
            f"fractional value(s): {shown}. A Poisson interval is a statement "
            "about a number of events; a score such as rate x robustness is not "
            "a count and has no interval, so the tied set is undefined for it. "
            "Pass the survey counts, or do not report a tie-aware metric.")
    ordered = counts.sort_index(kind="mergesort").sort_values(
        ascending=False, kind="mergesort")
    intervals = {ch: poisson_ci(int(n), duration_min, alpha=alpha)
                 for ch, n in ordered.items()}
    leader_low = intervals[ordered.index[0]][0]
    return [ch for ch in ordered.index if intervals[ch][1] >= leader_low]


def _candidate_metrics(rates: pd.Series, zones: pd.Series, duration_min: float) -> dict:
    """The tie-aware version of "did the surgeon remove what the map pointed at".

    ``candidates_resected`` is the share of the tied set that was removed. It
    reduces to :func:`_top_channel_resected` exactly when the set has one
    member, so it is a strict generalisation rather than a different question.

    ``leader_alone`` records how often the data picked a single channel at
    all. On this cohort it is the number worth knowing: a metric that looks
    decisive while resting on a set of eleven tied channels is not measuring
    what its name says.
    """
    candidates = candidate_channels(rates, duration_min)
    if not candidates:
        return {"n_candidates": 0, "candidates_resected": float("nan"),
                "candidates_all_resected": float("nan"), "leader_alone": float("nan")}
    inside = [c for c in candidates if zones.loc[c] == "resected"]
    return {
        "n_candidates": len(candidates),
        "candidates_resected": len(inside) / len(candidates),
        "candidates_all_resected": float(len(inside) == len(candidates)),
        "leader_alone": float(len(candidates) == 1),
    }


def _top_k_resected(rates: pd.Series, zones: pd.Series, k: int = 3) -> float:
    """Share of the ``k`` most active channels that were removed.

    The closest threshold-free stand-in for the source study's "HFO area":
    Fedele et al. defined an area from the channels with the highest rates
    and asked whether the surgeon took it. Using a fixed small ``k`` rather
    than a rate cut-off avoids inventing a per-patient threshold, which is
    the step in that analysis that is hardest to reproduce.
    """
    if not rates.sum():
        return float("nan")
    k = min(k, len(rates))
    top = rates.sort_values(ascending=False, kind="mergesort").head(k).index
    return float((zones.loc[top] == "resected").mean())


def _top_channel_resected(rates: pd.Series, zones: pd.Series) -> float:
    """1.0 if the single most active channel was removed, 0.0 if not, NaN if silent.

    The crudest version of the clinical question -- "would following this map
    have pointed the surgeon at tissue they actually took?" -- and the one
    that does not depend on how activity is distributed across the rest of
    the grid.
    """
    if not rates.sum():
        return float("nan")
    top = rates.idxmax()
    return 1.0 if zones.loc[top] == "resected" else 0.0


def outcome_subject(subject: str, resection, run: str = "01",
                    t_start: float = 0.0, t_stop: float = FULL_RUN_S,
                    dataset: str = "ds003498", detector: str = "rms",
                    threshold_sd: float | dict[str, float] | None = None,
                    bands: tuple[str, ...] = ("ripple", "fast_ripple"),
                    drop_eloquent: bool = True,
                    verbose: bool = True) -> tuple[list[dict], list[dict], dict]:
    """Measure one subject: expert share and detector share, both scopes.

    Returns ``(rows, channel_rows, meta)``. ``rows`` has one entry per
    (source, scope, band); ``channel_rows`` is the per-channel detail that
    makes any row auditable.
    """
    recording = fetch_slice(dataset=dataset, subject=subject, run=run,
                            t_start=t_start, t_stop=t_stop, verbose=verbose)
    prep = prepare(recording, verbose=False)
    truth = recording.ground_truth
    if truth is None:
        truth = pd.DataFrame(columns=["kind", "channel"])

    labels = classify_channels(prep.ch_names, resection).set_index("channel")
    scopes = {
        "all": list(prep.ch_names),
        "reviewed": [c for c in recording.reviewed_channels if c in prep.ch_names],
    }
    if drop_eloquent:
        for name, channels in scopes.items():
            scopes[name] = [c for c in channels if not bool(labels.loc[c, "eloquent"])]

    coverage = resection.coverage(prep.ch_names)
    meta = {
        "subject": subject, "run": run, "duration_s": prep.duration, "sfreq_hz": prep.sfreq,
        "n_channels": len(prep.ch_names), "n_reviewed": len(scopes["reviewed"]),
        "n_resected_channels": int((labels["zone"] == "resected").sum()),
        "n_partial_channels": int((labels["zone"] == "partial").sum()),
        "n_eloquent_channels": int(labels["eloquent"].sum()),
        "n_expert_events": int(len(truth)),
        **{k: v for k, v in coverage.items() if k != "subject"},
    }

    rows: list[dict] = []
    channel_rows: list[dict] = []
    for band_name in bands:
        band = getattr(BANDS, band_name)
        if not BANDS.usable(prep.sfreq, band):
            continue
        cut = _threshold_for(band_name, threshold_sd)
        cfg = DetectorConfig(band=band, threshold_sd=cut)
        events = DETECTORS[detector](prep, cfg, channels=scopes["all"])
        ours_all = pd.Series({c: 0 for c in prep.ch_names}, dtype=float)
        for event in events:
            if event.accepted:
                ours_all[event.channel] += 1.0
        expert_all = pd.Series({c: 0 for c in prep.ch_names}, dtype=float)
        if len(truth):
            counts = truth[truth["kind"] == band_name].groupby("channel").size()
            for channel, count in counts.items():
                if channel in expert_all.index:
                    expert_all[channel] += float(count)

        for scope, channels in scopes.items():
            if not channels:
                continue
            zones = labels.loc[channels, "zone"]
            for source, rates in (("expert", expert_all[channels]), (detector, ours_all[channels])):
                rows.append({
                    "subject": subject, "band": band_name, "scope": scope, "source": source,
                    "threshold_sd": cut if source != "expert" else float("nan"),
                    "n_channels": len(channels),
                    "n_resected_channels": int((zones == "resected").sum()),
                    **_share_in_resection(rates, zones),
                    "top_channel_resected": _top_channel_resected(rates, zones),
                    "top3_resected": _top_k_resected(rates, zones, k=3),
                    **_candidate_metrics(rates, zones, prep.duration / 60.0),
                })
        for channel in prep.ch_names:
            channel_rows.append({
                "subject": subject, "band": band_name, "channel": channel,
                "zone": labels.loc[channel, "zone"],
                "eloquent": bool(labels.loc[channel, "eloquent"]),
                "reviewed": channel in scopes["reviewed"],
                "expert_events": float(expert_all[channel]),
                f"{detector}_events": float(ours_all[channel]),
            })
    if verbose:
        shown = [r for r in rows if r["scope"] == "reviewed" and r["band"] == bands[0]]
        for row in shown:
            print(f"[onset-hfo]   {row['source']:>11}: {row['share_in_rz']:.2f} of "
                  f"{row['n_events']:.0f} events inside the resection "
                  f"({row['n_resected_channels']}/{row['n_channels']} reviewed channels resected)")
    return rows, channel_rows, meta


# --------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------


@dataclass
class OutcomeResult:
    """Everything one outcome study produced, and how to read it."""

    subjects: pd.DataFrame        #: one row per subject x source x scope x band
    channels: pd.DataFrame        #: per-channel detail behind every row above
    meta: pd.DataFrame            #: what was analysed per subject, incl. RZ coverage
    groups: pd.DataFrame          #: the group comparison, one row per arm
    participants: pd.DataFrame    #: outcome, ILAE, follow-up as published
    dataset: str = "ds003498"
    window_s: tuple[float, float] = (0.0, FULL_RUN_S)
    detector: str = "rms"
    threshold_sd: dict = field(default_factory=lambda: dict(BAND_THRESHOLD_SD))
    power_floor: float = float("nan")
    pipeline_version: str = PIPELINE_VERSION

    def summary(self, metric: str = "share_in_rz") -> pd.DataFrame:
        """The group table for one metric, ready to print."""
        table = self.groups[self.groups["metric"] == metric].copy()
        columns = ["source", "scope", "band", "n_seizure_free", "n_recurrence",
                   "median_seizure_free", "median_recurrence", "mean_seizure_free",
                   "mean_recurrence", "auc", "auc_lo", "auc_hi",
                   "rank_biserial", "p_permutation", "p_bonferroni"]
        return table[columns].round(3).reset_index(drop=True)

    #: The comparison the source publication predicts, and the one this
    #: analysis was designed around: the most fast-ripple-active channel,
    #: inside the resection or not. Named here so that "the headline" is a
    #: fixed choice in the code rather than whichever row came out best.
    PRIMARY = ("top_channel_resected", "fast_ripple")

    def verdicts(self) -> dict[str, str]:
        """Every metric x band, so no arm can be quoted without the others."""
        out: dict[str, str] = {}
        for metric in sorted(self.groups["metric"].unique()):
            for band in sorted(self.groups["band"].unique()):
                key = f"{metric}/{band}"
                out[key] = self.verdict(metric=metric, band=band)
        return out

    def verdict(self, metric: str | None = None, scope: str = "reviewed",
                band: str | None = None) -> str:
        """One paragraph saying what the numbers do and do not support.

        Written here rather than in a notebook so that the interpretation
        travels with the result and cannot drift away from it.
        """
        metric = metric or self.PRIMARY[0]
        band = band or self.PRIMARY[1]
        table = self.groups.query(
            "metric == @metric and scope == @scope and band == @band")
        if table.empty:
            return "No comparison was possible for this metric."
        lines = []
        for _, row in table.iterrows():
            sig = "separates" if row["p_permutation"] < 0.05 else "does not separate"
            lines.append(
                f"{row['source']}: AUC {row['auc']:.2f} "
                f"(95% CI {row['auc_lo']:.2f}-{row['auc_hi']:.2f}), "
                f"p = {row['p_permutation']:.3f} "
                f"({int(row['n_comparisons'])}-comparison Bonferroni "
                f"{row['p_bonferroni']:.2f}) -- {sig} the groups.")
        floor = (f"With {int(table.iloc[0]['n_seizure_free'])} seizure-free and "
                 f"{int(table.iloc[0]['n_recurrence'])} recurrence patients, the smallest "
                 f"effect reaching 80% power is AUC {self.power_floor:.2f}; anything below "
                 "that could not have been detected here whether or not it is real.")
        expert = table[table["source"] == "expert"]
        control = ""
        if len(expert):
            if expert.iloc[0]["p_permutation"] >= 0.05:
                control = ("The expert markings do not separate the groups either, so this "
                           "is a statement about a 60-second window and 20 patients, not "
                           "about the detector.")
            else:
                control = ("The expert markings do separate the groups on the same channels, "
                           "so any failure of the detector arm is the detector's.")
        return " ".join(lines + [floor, control]).strip()

    def save(self, directory: str | Path | None = None) -> Path:
        out = Path(directory or RESULTS_DIR) / f"outcome_{self.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        self.subjects.to_csv(out / "subjects.csv", index=False)
        self.channels.to_csv(out / "channels.csv", index=False)
        self.meta.to_csv(out / "recordings.csv", index=False)
        self.groups.to_csv(out / "groups.csv", index=False)
        self.participants.to_csv(out / "participants.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "dataset": self.dataset, "window_s": list(self.window_s),
            "detector": self.detector, "threshold_sd": self.threshold_sd,
            "pipeline_version": self.pipeline_version,
            "power_floor_auc": self.power_floor,
            "n_subjects": int(self.meta["subject"].nunique()) if len(self.meta) else 0,
            "primary_comparison": "/".join(self.PRIMARY),
            "verdict": self.verdict(),
            "verdicts": self.verdicts(),
        }, indent=2))
        print(f"[onset-hfo] outcome study written to {out}")
        return out


def compare_groups(subjects: pd.DataFrame, participants: pd.DataFrame,
                   metrics: tuple[str, ...] = ("share_in_rz", "share_in_rz_incl_partial",
                                               "top_channel_resected", "top3_resected",
                                               "candidates_resected",
                                               "candidates_all_resected"),
                   seed: int = 0) -> pd.DataFrame:
    """Seizure-free vs recurrence, for every (source, scope, band, metric)."""
    merged = subjects.merge(participants[["subject", "outcome"]], on="subject", how="left")
    rows = []
    keys = ["source", "scope", "band"]
    for metric in metrics:
        for key, group in merged.groupby(keys, dropna=False):
            free = group.loc[group["outcome"] == _FREE, metric].to_numpy(float)
            recur = group.loc[group["outcome"] == _RECUR, metric].to_numpy(float)
            rows.append({**dict(zip(keys, key, strict=True)), "metric": metric,
                         **rank_comparison(free, recur, seed=seed)})
    table = pd.DataFrame(rows)
    # One run produces a few dozen comparisons. Carrying the corrected p in
    # the table -- rather than mentioning correction in prose -- makes it
    # impossible to quote the raw p without seeing what a table this size
    # does to it. The expert arm duplicates across scopes (the annotators
    # only marked reviewed channels), so those duplicates are not counted
    # twice.
    unique = table.drop_duplicates(
        subset=["source", "band", "metric", "auc", "p_permutation"])
    n = int(len(unique))
    table["n_comparisons"] = n
    table["p_bonferroni"] = (table["p_permutation"] * n).clip(upper=1.0)
    return table


def outcome_study(subjects: list[str] | None = None, n_subjects: int | None = None,
                  dataset: str = "ds003498", run: str = "01",
                  t_start: float = 0.0, t_stop: float = FULL_RUN_S,
                  detector: str = "rms", threshold_sd: float | dict[str, float] | None = None,
                  bands: tuple[str, ...] = ("ripple", "fast_ripple"),
                  drop_eloquent: bool = True,
                  verbose: bool = True) -> OutcomeResult:
    """Run the whole study: per-subject shares, then the group comparison.

    Subjects that fail are skipped with a message rather than aborting the
    cohort, and the skip is visible in ``meta`` -- a study that silently
    analyses 14 of 20 patients and reports "the cohort" is worse than one
    that crashes.
    """
    resections = resection_map(dataset, verbose=verbose)
    participants = fetch_participants(dataset)
    if subjects is None:
        subjects = list_subjects(dataset)
        if n_subjects:
            subjects = subjects[:n_subjects]

    rows, channel_rows, meta_rows = [], [], []
    for i, subject in enumerate(subjects, 1):
        if verbose:
            outcome = participants.loc[participants["subject"] == subject, "outcome"]
            tag = outcome.iloc[0] if len(outcome) else "?"
            print(f"\n[onset-hfo] ({i}/{len(subjects)}) {subject} [outcome {tag}]")
        resection = resections.get(subject)
        if resection is None or not len(resection):
            print(f"[onset-hfo]   skipped {subject}: no resected zone in the clinical sheet")
            continue
        try:
            subject_rows, subject_channels, meta = outcome_subject(
                subject, resection, run=run, t_start=t_start, t_stop=t_stop,
                dataset=dataset, detector=detector, threshold_sd=threshold_sd,
                bands=bands, drop_eloquent=drop_eloquent, verbose=verbose)
        except Exception as exc:
            print(f"[onset-hfo]   skipped {subject}: {type(exc).__name__}: {exc}")
            continue
        rows += subject_rows
        channel_rows += subject_channels
        meta_rows.append(meta)

    subjects_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta_rows)
    groups = (compare_groups(subjects_df, participants) if len(subjects_df)
              else pd.DataFrame())
    analysed = participants[participants["subject"].isin(meta_df.get("subject", []))]
    floor = min_detectable_auc(int((analysed["outcome"] == _FREE).sum()),
                               int((analysed["outcome"] == _RECUR).sum())) \
        if len(analysed) else float("nan")
    return OutcomeResult(subjects=subjects_df, channels=pd.DataFrame(channel_rows),
                         meta=meta_df, groups=groups, participants=participants,
                         dataset=dataset, window_s=(t_start, t_stop), detector=detector,
                         threshold_sd={b: _threshold_for(b, threshold_sd) for b in bands},
                         power_floor=floor)
