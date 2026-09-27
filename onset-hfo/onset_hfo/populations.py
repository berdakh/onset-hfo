"""Ripples are not one population, and this module refuses to report them as one.

Roadmap item 6, and the biggest scientific gap in the project. A high ripple
rate in healthy occipital cortex is not a finding: physiological ripples occur
in tissue nobody should resect, and nothing here can tell them from epileptic
ones. Every rate this pipeline has ever printed silently merges the two.

**What this module does not do is classify them.** There is no label in any
public archive saying *this ripple was physiological*, so a classifier would
be trained on a proxy and read as though it were trained on the thing. What it
does instead is the honest half: split the events by a property that is
actually observed -- whether the ripple rides on an interictal discharge --
report the sub-populations side by side, and *measure whether the split
separates anything at all*.

**Measured on the shipped 60-second recording, mostly it does not**, and the
two detectors disagree about the one place it does. See
``docs/EVALUATION.md`` §4. That result is the reason this module reports two
rates rather than labelling events: a split that cannot be seen in the
waveform is a split you must carry forward, not resolve.

Why discharge co-occurrence is the property to split on: it is the one
distinction in the literature that this pipeline already measures per event
(``Event.co_occurs_with_spike``), it needs no coordinates, no sleep staging
and no second annotator, and ripples riding on epileptiform discharges are
the sub-population most consistently reported as pathological. It is a proxy.
It is not a label, and nothing here pretends otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from onset_hfo.detectors.base import Event

__all__ = [
    "SPIKE_COUPLED",
    "INDEPENDENT",
    "MORPHOLOGY",
    "split_by_discharge",
    "population_rates",
    "compare_populations",
    "population_notes",
]

#: Rides on an interictal epileptiform discharge on the same channel.
SPIKE_COUPLED = "spike_coupled"

#: Does not. The larger population on every recording measured so far, and the
#: one most likely to contain whatever physiological ripples are present.
INDEPENDENT = "independent"

#: The per-event measurements the detectors already produce. Every one is
#: reported in the HFO literature as differing between the two populations;
#: none of them is a label.
MORPHOLOGY = ("duration_ms", "n_cycles", "peak_frequency_hz",
              "spectral_prominence_db", "peak_amplitude_uv")


def split_by_discharge(events: list[Event],
                       accepted_only: bool = True) -> dict[str, list[Event]]:
    """Partition events into the two sub-populations. Both keys always exist.

    An empty sub-population is reported as empty rather than dropped: "no
    ripple on this channel rode a discharge" is a finding, and a missing key
    reads as "not looked at".
    """
    keep = [e for e in events if (e.accepted or not accepted_only)]
    return {
        SPIKE_COUPLED: [e for e in keep if e.co_occurs_with_spike],
        INDEPENDENT: [e for e in keep if not e.co_occurs_with_spike],
    }


def population_rates(events: list[Event], duration_s: float,
                     channels: list[str] | None = None,
                     accepted_only: bool = True) -> pd.DataFrame:
    """Per-channel rates, one column per sub-population and one for the total.

    The total is kept because every earlier result in this project is a total
    and a reader must be able to reconcile them. It is deliberately the last
    column rather than the first.
    """
    from onset_hfo.metrics import channel_rates

    parts = split_by_discharge(events, accepted_only=accepted_only)
    names = channels if channels is not None else sorted(
        {e.channel for e in events})
    frame = pd.DataFrame({"channel": names})
    for population, subset in parts.items():
        rates = channel_rates(subset, duration_s, names, accepted_only=accepted_only)
        column = rates.set_index("channel")["rate_per_min"] if len(rates) else pd.Series(dtype=float)
        frame[f"rate_{population}"] = frame["channel"].map(column).fillna(0.0)
        counts = rates.set_index("channel")["n_events"] if len(rates) else pd.Series(dtype=float)
        frame[f"n_{population}"] = frame["channel"].map(counts).fillna(0).astype(int)
    frame["rate_total"] = frame[f"rate_{SPIKE_COUPLED}"] + frame[f"rate_{INDEPENDENT}"]
    frame["n_total"] = frame[f"n_{SPIKE_COUPLED}"] + frame[f"n_{INDEPENDENT}"]
    # The share is the interesting column: a channel whose ripples nearly all
    # ride discharges is a different object from one whose ripples never do,
    # even at the same total rate.
    frame["share_spike_coupled"] = np.where(
        frame["n_total"] > 0, frame[f"n_{SPIKE_COUPLED}"] / frame["n_total"].replace(0, np.nan),
        np.nan)
    return frame.sort_values("rate_total", ascending=False).reset_index(drop=True)


def compare_populations(events: list[Event], features: tuple[str, ...] = MORPHOLOGY,
                        accepted_only: bool = True, n_boot: int = 2000,
                        seed: int = 0, with_power: bool = True) -> pd.DataFrame:
    """Does the split separate the events in any measurable way?

    One row per feature: the two medians, the common-language effect size
    (AUC: the chance a randomly chosen spike-coupled event scores higher than
    a randomly chosen independent one), a bootstrap interval, an exact-where-
    affordable permutation p, and **the Bonferroni correction across the
    features tested**, because testing five features and quoting the smallest
    p is how a null result becomes a finding.

    ``min_detectable_auc`` is carried on every row so a null can be read
    properly: an AUC of 0.55 that the sample could never have resolved is not
    evidence of no difference. It is simulated and costs about ten seconds;
    ``with_power=False`` skips it, which is the right choice inside a test and
    the wrong one inside a claim.

    **This is slow on purpose** -- an exact-where-affordable permutation over a
    few thousand events takes tens of seconds per feature. It is an analysis
    you run deliberately, not part of ``run_pipeline``.
    """
    from onset_hfo.outcome import min_detectable_auc, rank_comparison

    parts = split_by_discharge(events, accepted_only=accepted_only)
    coupled, alone = parts[SPIKE_COUPLED], parts[INDEPENDENT]
    rows = []
    floor = float("nan")
    if with_power and len(coupled) >= 2 and len(alone) >= 2:
        floor = min_detectable_auc(len(coupled), len(alone))
    for feature in features:
        a = np.array([getattr(e, feature, np.nan) for e in coupled], dtype=float)
        b = np.array([getattr(e, feature, np.nan) for e in alone], dtype=float)
        stats = rank_comparison(a, b, n_boot=n_boot, seed=seed)
        rows.append({
            "feature": feature,
            "n_spike_coupled": stats["n_seizure_free"],
            "n_independent": stats["n_recurrence"],
            "median_spike_coupled": stats["median_seizure_free"],
            "median_independent": stats["median_recurrence"],
            "auc": stats["auc"], "auc_lo": stats["auc_lo"], "auc_hi": stats["auc_hi"],
            "p_permutation": stats["p_permutation"],
            "min_detectable_auc": floor,
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["p_bonferroni"] = (frame["p_permutation"] * len(frame)).clip(upper=1.0)
        frame["separates"] = frame["p_bonferroni"] < 0.05
    return frame


def population_notes(events_by_detector: dict[str, list[Event]],
                     duration_s: float) -> list[str]:
    """Plain sentences for the report, one per detector plus a standing caveat.

    These go into the report's existing ``data_quality`` section rather than a
    new one: the agent's tool contract is frozen and enumerates the sections
    it may read, and a scientific caveat is worth less than a stable contract
    is worth.
    """
    notes: list[str] = []
    for detector, events in events_by_detector.items():
        parts = split_by_discharge(events)
        n_coupled, n_alone = len(parts[SPIKE_COUPLED]), len(parts[INDEPENDENT])
        total = n_coupled + n_alone
        if not total:
            continue
        minutes = duration_s / 60.0
        notes.append(
            f"{detector}: {n_coupled} of {total} accepted events "
            f"({n_coupled / total:.0%}) coincide with an interictal discharge "
            f"on the same channel — {n_coupled / minutes:.1f}/min against "
            f"{n_alone / minutes:.1f}/min for the rest, summed over all "
            f"channels. The two are reported "
            f"separately because nothing here can tell a physiological ripple "
            f"from an epileptic one, and merging them hides that.")
    if notes:
        notes.append(
            "Discharge co-occurrence is a proxy for the physiological/epileptic "
            "distinction, not a label for it: no public archive marks which "
            "ripples were physiological, so neither sub-population should be "
            "read as one or the other.")
    return notes
