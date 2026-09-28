#!/usr/bin/env python
"""Does splitting ripples by discharge coupling localise the resection better?

`onset_hfo/populations.py` splits detected events by whether they ride an
interictal discharge, and `EVALUATION.md` §3b measures whether the two
sub-populations *differ* in waveform. This script answers the question that
would make the split clinically interesting and that §3b could not: whether
either sub-population predicts the resected zone better than the merged rate
does.

It reuses the outcome study's own metric functions, so the `merged` arm is
directly comparable to the published per-subject numbers.

    # both bands, from the slices already in artifacts/data/ (~40 min)
    ONSET_HFO_OFFLINE=1 python scripts/run_subpopulation_outcome.py

    # re-print the analysis from the committed extract, no re-analysis
    python scripts/run_subpopulation_outcome.py --from-csv data/outcome/subpopulation_screen.csv

Why five 60 s windows rather than the whole run
-----------------------------------------------
The outcome study proper analyses the whole 300 s of each `run-01`. That run is
not cached here and fetching it needs the archive, so this screen uses the five
cached 60 s windows that tile the same 300 s and pools them by taking each
subject's mean -- the same pooling `data/stability/` uses. Naming the subjects
explicitly is what lets the whole thing run with `ONSET_HFO_OFFLINE=1`: only
the subject *listing* touches the network.

Why both bands, when the pre-specified arm is fast ripples
----------------------------------------------------------
Because in the fast ripple band the spike-coupled population is too small to
rank channels with -- a median of **one** event per 60 s window, and under five
in 56 of the 100 subject-windows. An argmax over that is a coin flip, and it
duly produced an apparent AUC of 0.806 that the tie-aware
`candidates_resected` metric contradicted at 0.479. The ripple band carries
~30x the events, which converts "cannot tell" into an answer. The cost is that
a ripple-band result is not comparable to the published fast-ripple AUC; this
is a secondary analysis and is reported as one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    # So the script runs from a bare checkout, not only an installed one.
    sys.path.insert(0, str(REPO))

# Everything from onset_hfo is imported inside the functions, after that path
# insert has taken effect -- so the two population names are spelled out here
# and checked against the package by `_check_population_names` below.
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 21)]
METRICS = ["top_channel_resected", "candidates_resected", "share_in_rz"]
SPIKE_COUPLED = "spike_coupled"
INDEPENDENT = "independent"
POPULATIONS = ["merged", SPIKE_COUPLED, INDEPENDENT]


def _check_population_names() -> None:
    """Fail loudly if `onset_hfo.populations` renames what this script hardcodes.

    Without this the script would quietly produce a table whose `population`
    column no longer matches the split it came from.
    """
    from onset_hfo import populations

    assert (SPIKE_COUPLED, INDEPENDENT) == (populations.SPIKE_COUPLED,
                                            populations.INDEPENDENT), (
        f"onset_hfo.populations now calls them "
        f"{populations.SPIKE_COUPLED!r}/{populations.INDEPENDENT!r}; "
        f"this script still says {SPIKE_COUPLED!r}/{INDEPENDENT!r}")


def screen(bands, subjects, n_windows, run="01"):
    """One row per subject x window x population, with the outcome metrics."""
    import pandas as pd

    from onset_hfo.clinical import classify_channels, resection_map
    from onset_hfo.config import BANDS, DetectorConfig, PipelineConfig
    from onset_hfo.datasets import fetch_slice
    from onset_hfo.detectors import detect_rms, detect_spikes
    from onset_hfo.outcome import (
        _candidate_metrics,
        _share_in_resection,
        _threshold_for,
        _top_channel_resected,
    )
    from onset_hfo.populations import split_by_discharge
    from onset_hfo.preprocess import prepare
    from onset_hfo.validate import flag_spike_cooccurrence, validate_events

    cfg = PipelineConfig()
    rmap = resection_map("ds003498")
    windows = [(w * 60.0, (w + 1) * 60.0) for w in range(n_windows)]
    rows = []

    for band_name in bands:
        band = getattr(BANDS, band_name)
        for subject in subjects:
            for index, (t_start, t_stop) in enumerate(windows):
                record = fetch_slice(dataset="ds003498", subject=subject, run=run,
                                     t_start=t_start, t_stop=t_stop, verbose=False)
                prep = prepare(record, verbose=False)
                if not BANDS.usable(prep.sfreq, band):
                    continue

                labels = classify_channels(prep.ch_names,
                                           rmap[subject]).set_index("channel")
                # Eloquent cortex is never resected, so including it would let a
                # detector look right for the wrong reason.
                reviewed = [c for c in record.reviewed_channels
                            if c in prep.ch_names
                            and not bool(labels.loc[c, "eloquent"])]
                if not reviewed:
                    continue

                det_cfg = DetectorConfig(band=band,
                                         threshold_sd=_threshold_for(band_name, None))
                events = detect_rms(prep, det_cfg, channels=reviewed,
                                    describe_spectrum=False)
                validate_events(events, prep, cfg.validation)
                flag_spike_cooccurrence(events, detect_spikes(prep, cfg.spikes))

                parts = split_by_discharge(events)
                parts["merged"] = [e for e in events if e.accepted]

                zones = labels.loc[reviewed, "zone"]
                for population, subset in parts.items():
                    rates = pd.Series({c: 0.0 for c in reviewed}, dtype=float)
                    for event in subset:
                        rates[event.channel] += 1.0
                    rows.append({
                        "band": band_name, "subject": subject, "window": index,
                        "population": population, "n_events": float(rates.sum()),
                        **_share_in_resection(rates, zones),
                        "top_channel_resected": _top_channel_resected(rates, zones),
                        **_candidate_metrics(rates, zones, prep.duration / 60.0),
                    })
                print(f"{band_name} {subject} w{index}: "
                      + " ".join(f"{k}={len(v)}" for k, v in parts.items()),
                      flush=True)

    return pd.DataFrame(rows)


def analyse(rows):
    """Pool windows by subject mean, then rank seizure-free against recurrence."""
    import pandas as pd

    from onset_hfo.outcome import rank_comparison

    participants = pd.read_csv(REPO / "data" / "outcome" / "participants.csv")
    seizure_free = set(participants[participants.outcome == "S"].subject)
    rows = rows.assign(seizure_free=rows.subject.isin(seizure_free))

    out = []
    for band in rows.band.unique():
        for metric in METRICS:
            for population in POPULATIONS:
                part = rows[(rows.band == band) & (rows.population == population)]
                # Each subject contributes once, as the stability study pools.
                per_subject = (part.groupby(["subject", "seizure_free"])[metric]
                               .mean().reset_index())
                free = per_subject[per_subject.seizure_free][metric].to_numpy()
                recurrence = per_subject[~per_subject.seizure_free][metric].to_numpy()
                stat = rank_comparison(free, recurrence, n_boot=3000)
                out.append({
                    "band": band, "metric": metric, "population": population,
                    # `rank_comparison` drops non-finite values, so these are
                    # the patients the arm actually compared -- not always 13
                    # and 7. A population with no events in any window has no
                    # share and no leader, so that patient falls out of that
                    # arm and only that arm.
                    "n_SF": stat["n_seizure_free"], "n_rec": stat["n_recurrence"],
                    "mean_SF": stat["mean_seizure_free"],
                    "mean_rec": stat["mean_recurrence"],
                    "auc": stat["auc"], "lo": stat["auc_lo"], "hi": stat["auc_hi"],
                    "p": stat["p_permutation"],
                })

    result = pd.DataFrame(out)
    # Corrected within each band: nine rows per band, one family per band.
    result["p_bonferroni"] = (result.p * len(METRICS) * len(POPULATIONS)).clip(upper=1.0)
    return result


def report(rows, result) -> None:
    import pandas as pd

    from onset_hfo.outcome import min_detectable_auc

    pd.set_option("display.width", 200)
    for band in rows.band.unique():
        part = rows[rows.band == band]
        coupled = part[part.population == SPIKE_COUPLED]
        print(f"\n=== {band}: {len(part)} rows · {part.subject.nunique()} subjects "
              f"× {part.window.nunique()} windows")
        print("  events per window, median by population: "
              + "  ".join(f"{k}={v:.1f}" for k, v in
                          part.groupby('population').n_events.median().items()))
        print(f"  spike-coupled windows with <5 events: "
              f"{int((coupled.n_events < 5).sum())}/{len(coupled)}"
              f"   subjects whose median is <5: "
              f"{int((coupled.groupby('subject').n_events.median() < 5).sum())}"
              f"/{coupled.subject.nunique()}")
        part_result = result[result.band == band]
        print(part_result.drop(columns="band").round(3).to_string(index=False))
        best = part_result.set_index(["metric", "population"])["auc"]
        for metric in METRICS:
            gain = (max(best[(metric, SPIKE_COUPLED)], best[(metric, INDEPENDENT)])
                    - best[(metric, "merged")])
            print(f"  {metric:22s} best sub-population gain over merged: {gain:+.3f}")

    # Not every arm compares all 20 patients, so the floor is per arm.
    print("\npower floor per arm, min_detectable_auc(n_SF, n_rec):")
    floors = {(int(r.n_SF), int(r.n_rec)) for r in result.itertuples()}
    for n_free, n_recurrence in sorted(floors, reverse=True):
        arms = result[(result.n_SF == n_free) & (result.n_rec == n_recurrence)]
        print(f"  n = {n_free}/{n_recurrence}: floor "
              f"{min_detectable_auc(n_free, n_recurrence)}, "
              f"best observed AUC {arms.auc.max():.3f}  ({len(arms)} arms)")
    clears = [r for r in result.itertuples()
              if r.auc >= min_detectable_auc(int(r.n_SF), int(r.n_rec))]
    print(f"  arms clearing their own floor: {len(clears)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_subpopulation_outcome", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", nargs="+", default=["ripple", "fast_ripple"])
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--windows", type=int, default=5,
                        help="how many cached 60 s windows per subject")
    parser.add_argument("--from-csv", default=None,
                        help="skip the analysis and report this extract instead")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "results"
                                             / "subpopulation_screen.csv"))
    args = parser.parse_args(argv)

    import pandas as pd

    _check_population_names()
    if args.from_csv:
        rows = pd.read_csv(args.from_csv)
    else:
        rows = screen(args.bands, args.subjects, args.windows)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        print(f"\nwrote {len(rows)} rows to {out}")

    report(rows, analyse(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
