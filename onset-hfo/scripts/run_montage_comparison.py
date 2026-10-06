#!/usr/bin/env python
"""Does the bipolar montage help or hurt for HFO *rate*? Measured on ds003498.

`ROADMAP.md` has carried this as an open question since the pipeline was
built: the bipolar montage is standard for HFO work because a common
reference shares its noise with every channel, but nothing here had measured
what the montage does to the one thing this project reports -- the ranking of
channels by HFO rate. That is the experiment the roadmap called "easy and
currently unmeasured": run the whole pipeline under each montage and compare
the rankings.

    # all three montages, both bands, from the cached slices
    ONSET_HFO_OFFLINE=1 python scripts/run_montage_comparison.py

    # re-print the published tables from the committed extract
    python scripts/run_montage_comparison.py \
        --from-csv data/outcome/montage_screen.csv

Three arms, everything else held fixed
--------------------------------------
Same cached 60 s windows, same detector, same per-band thresholds, same
validation, same metric functions. Only `PreprocessConfig`'s re-referencing
changes:

    bipolar       neighbouring contacts subtracted (the shipped default)
    referential   the archive's own reference, left alone
    average       common average across every recorded contact

Two yardsticks, because they can disagree
-----------------------------------------
**Surgical outcome.** Each arm's ranking is scored against the resection with
the plain rule (score = rate) exactly as `run_robustness_ablation.py` does,
and seizure-free patients are compared with recurrences by AUC. A montage that
ranks the resected contacts higher in the patients whose surgery worked is
doing the job the ranking is for.

**The expert markings.** The annotators marked HFOs on *bipolar pairs*, and
the archive stores *referential contacts*. So the bipolar arm is scored on the
pairs the experts reviewed, and the two referential arms on the contacts
behind those pairs -- the same physical electrodes, in the unit each arm
actually ranks. Rank agreement with the expert is Spearman over that unit; for
a contact, the expert count is the sum over the reviewed pairs it belongs to.
Event-level precision and recall need one unit on both sides, so for the
referential arms each detection is projected onto the pairs its contact
belongs to (overlapping detections on a pair's two contacts merge into one),
and the projected events are scored against the pair markings. That is an
approximation and is labelled as one in the output: it answers "would a reader
of pair A-B have been shown this event", not "did the detector fire on A-B".

What the two yardsticks cannot share
------------------------------------
The resection labels are not symmetric across arms. A bipolar pair is
``resected`` only when both its contacts were removed and ``partial`` when
one was; a contact is simply in or out. The referential arms therefore have
no partial channels, which changes what ``share_in_rz`` can count. That is a
property of the question, not a bug to normalise away, and the group table
reports every arm with its own channel count so the reader can see it.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    # So the script runs from a bare checkout, not only an installed one.
    sys.path.insert(0, str(REPO))

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 21)]
BANDS = ["ripple", "fast_ripple"]
MONTAGES = ("bipolar", "referential", "average")
#: Scored against the resection and compared between outcome groups.
OUTCOME_METRICS = ["top_channel_resected", "top3_resected", "share_in_rz",
                   "candidates_resected"]
#: Scored against the expert markings and reported as cohort means.
AGREEMENT_METRICS = ["spearman_rho", "top5_overlap", "precision", "recall", "f1",
                     "rho_vs_bipolar"]
TOP_K = 5


# -- pure helpers, tested on their own ------------------------------------


def montage_config(montage: str):
    """The `PreprocessConfig` for one arm; everything else at its default."""
    from onset_hfo.config import PreprocessConfig

    if montage == "bipolar":
        return PreprocessConfig()
    if montage == "referential":
        return PreprocessConfig(bipolar=False, average_reference=False)
    if montage == "average":
        return PreprocessConfig(bipolar=False, average_reference=True)
    raise ValueError(f"unknown montage {montage!r}; expected one of {MONTAGES}")


def pair_contacts(pair: str) -> tuple[str, str]:
    a, _, b = pair.partition("-")
    return a, b


def contacts_behind(pairs: list[str]) -> list[str]:
    """The physical contacts the reviewed pairs are built from, in order, once."""
    out: list[str] = []
    for pair in pairs:
        for contact in pair_contacts(pair):
            if contact and contact not in out:
                out.append(contact)
    return out


def expert_on_contacts(expert_pairs, contacts: list[str]):
    """Project per-pair expert counts onto contacts: each contact is credited
    with every marking on a reviewed pair it belongs to."""
    import pandas as pd

    out = pd.Series(0.0, index=list(contacts), dtype=float)
    for pair, count in expert_pairs.items():
        for contact in pair_contacts(str(pair)):
            if contact in out.index:
                out[contact] += float(count)
    return out


def contacts_on_pairs(contact_counts, pairs: list[str]):
    """Project per-contact counts onto pairs (mean of the two contacts), for
    comparing a referential ranking with a bipolar one on the same unit.
    Pairs with a missing contact are left out rather than half-counted."""
    import pandas as pd

    out = {}
    for pair in pairs:
        a, b = pair_contacts(pair)
        if a in contact_counts.index and b in contact_counts.index:
            out[pair] = 0.5 * (float(contact_counts[a]) + float(contact_counts[b]))
    return pd.Series(out, dtype=float)


def events_as_pairs(events, pairs: list[str]):
    """Re-label referential detections onto the pairs their contact belongs to.

    For each pair, the accepted events on either of its contacts are pooled
    and overlapping ones merged, so a burst seen on both contacts is one
    event on the pair rather than two. A contact shared by two pairs
    contributes to both, which is what a reader of each pair would see.
    """
    out = []
    for pair in pairs:
        a, b = pair_contacts(pair)
        mine = sorted((e for e in events if e.accepted and e.channel in (a, b)),
                      key=lambda e: (float(e.start), float(e.stop)))
        merged = []
        for event in mine:
            if merged and float(event.start) <= float(merged[-1].stop):
                last = merged[-1]
                merged[-1] = replace(last, stop=max(float(last.stop), float(event.stop)),
                                     score=max(float(last.score), float(event.score)))
            else:
                # A re-labelled copy: the caller's events are not modified.
                merged.append(replace(event, channel=pair, contacts=[a, b]))
        out.extend(merged)
    return out


def spearman(a, b) -> float:
    """Spearman over the common index; NaN when either side cannot rank."""
    import numpy as np
    from scipy.stats import spearmanr

    common = a.index.intersection(b.index)
    if len(common) < 3:
        return float("nan")
    x, y = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


# -- the screen ------------------------------------------------------------


def _counts(prep, band, threshold, channels, validation):
    """Accepted events per channel at one threshold, as integer-valued floats."""
    import pandas as pd

    from onset_hfo.config import DetectorConfig
    from onset_hfo.detectors import detect_rms
    from onset_hfo.validate import validate_events

    events = detect_rms(prep, DetectorConfig(band=band, threshold_sd=threshold),
                        channels=channels, describe_spectrum=False)
    validate_events(events, prep, validation)
    series = pd.Series({c: 0.0 for c in channels}, dtype=float)
    for event in events:
        if event.accepted:
            series[event.channel] += 1.0
    return series, events


def _outcome_metrics(counts, zones, duration_min) -> dict:
    from onset_hfo.outcome import (
        _candidate_metrics,
        _share_in_resection,
        _top_channel_resected,
        _top_k_resected,
    )

    row = {
        **_share_in_resection(counts, zones),
        "top_channel_resected": _top_channel_resected(counts, zones),
        "top3_resected": _top_k_resected(counts, zones, k=3),
        **_candidate_metrics(counts, zones, duration_min),
    }
    return row


def screen(bands, subjects, n_windows, run="01", partial_dir: Path | None = None):
    """One row per band x subject x window x montage."""
    import pandas as pd

    from onset_hfo.benchmark import _rank_agreement
    from onset_hfo.clinical import classify_channels, resection_map
    from onset_hfo.config import BANDS as BAND_DEFS
    from onset_hfo.config import PipelineConfig
    from onset_hfo.datasets import fetch_slice
    from onset_hfo.evaluate import evaluate_detections
    from onset_hfo.outcome import _threshold_for
    from onset_hfo.preprocess import prepare

    validation = PipelineConfig().validation
    rmap = resection_map("ds003498")
    windows = [(w * 60.0, (w + 1) * 60.0) for w in range(n_windows)]
    frames = []

    for subject in subjects:
        done = partial_dir / f"{subject}.csv" if partial_dir else None
        if done is not None and done.exists():
            frames.append(pd.read_csv(done))
            print(f"{subject}: resumed from {done}", flush=True)
            continue
        rows = []
        for index, (t_start, t_stop) in enumerate(windows):
            record = fetch_slice(dataset="ds003498", subject=subject, run=run,
                                 t_start=t_start, t_stop=t_stop, verbose=False)
            truth = record.ground_truth
            if truth is None:
                truth = pd.DataFrame(columns=["kind", "channel"])
            prepared = {m: prepare(record, montage_config(m), verbose=False)
                        for m in MONTAGES}

            # The unit each arm ranks: reviewed pairs, or the contacts behind
            # them. Eloquent channels are dropped in every arm, as elsewhere.
            pairs_all = [c for c in record.reviewed_channels
                         if c in prepared["bipolar"].ch_names]
            units = {"bipolar": pairs_all}
            behind = contacts_behind(pairs_all)
            for m in ("referential", "average"):
                units[m] = [c for c in behind if c in prepared[m].ch_names]
            labels, channels = {}, {}
            for m in MONTAGES:
                table = classify_channels(prepared[m].ch_names, rmap[subject]).set_index("channel")
                labels[m] = table
                channels[m] = [c for c in units[m] if not bool(table.loc[c, "eloquent"])]
            pairs = channels["bipolar"]
            if not pairs:
                continue

            for band_name in bands:
                band = getattr(BAND_DEFS, band_name)
                threshold = _threshold_for(band_name, None)
                expert_pairs = (truth[truth["kind"] == band_name]
                                .groupby("channel").size()
                                .reindex(pairs, fill_value=0).astype(float))
                bipolar_counts = None
                for m in MONTAGES:
                    prep = prepared[m]
                    chans = channels[m]
                    if not chans or not BAND_DEFS.usable(prep.sfreq, band):
                        continue
                    counts, events = _counts(prep, band, threshold, chans, validation)
                    zones = labels[m].loc[chans, "zone"]
                    duration_min = prep.duration / 60.0

                    if m == "bipolar":
                        bipolar_counts = counts
                        expert = expert_pairs
                        scored = events
                        on_pairs = counts
                    else:
                        expert = expert_on_contacts(expert_pairs, chans)
                        scored = events_as_pairs(events, pairs)
                        on_pairs = contacts_on_pairs(counts, pairs)
                    agreement = _rank_agreement(expert, counts, TOP_K)
                    score = evaluate_detections(scored, truth, kind=band_name,
                                                channels=pairs).as_dict()
                    row = {
                        "band": band_name, "subject": subject, "window": index,
                        "montage": m, "threshold_sd": threshold,
                        "n_channels": len(chans),
                        "n_resected_channels": int((zones == "resected").sum()),
                        "n_partial_channels": int((zones == "partial").sum()),
                        "n_expert_events": float(expert_pairs.sum()),
                        **_outcome_metrics(counts, zones, duration_min),
                        "spearman_rho": agreement["spearman_rho"],
                        "top5_overlap": agreement[f"top{TOP_K}_overlap"],
                        "precision": score.get("precision", float("nan")),
                        "recall": score.get("recall", float("nan")),
                        "f1": score.get("f1", float("nan")),
                        "event_scoring": "direct" if m == "bipolar" else "projected",
                        "rho_vs_bipolar": (1.0 if m == "bipolar" else
                                           spearman(bipolar_counts, on_pairs)),
                    }
                    rows.append(row)
                print(f"{band_name} {subject} w{index}: "
                      + ", ".join(f"{r['montage']} {r['n_events']:.0f}"
                                  for r in rows
                                  if r["band"] == band_name and r["window"] == index),
                      flush=True)
        frame = pd.DataFrame(rows)
        if done is not None:
            done.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(done, index=False)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -- the analysis ----------------------------------------------------------


def analyse(rows):
    """Pool windows by subject mean, then rank seizure-free against recurrence
    for the outcome metrics; cohort means for the agreement metrics."""
    import pandas as pd

    from onset_hfo.outcome import rank_comparison

    participants = pd.read_csv(REPO / "data" / "outcome" / "participants.csv")
    seizure_free = set(participants[participants.outcome == "S"].subject)
    rows = rows.assign(seizure_free=rows.subject.isin(seizure_free))

    out = []
    for band in rows.band.unique():
        for montage in MONTAGES:
            arm = rows[(rows.band == band) & (rows.montage == montage)]
            if arm.empty:
                continue
            per_subject = arm.groupby(["subject", "seizure_free"]).mean(numeric_only=True)
            per_subject = per_subject.reset_index()
            for metric in OUTCOME_METRICS:
                free = per_subject[per_subject.seizure_free][metric].dropna().to_numpy()
                rec = per_subject[~per_subject.seizure_free][metric].dropna().to_numpy()
                stat = rank_comparison(free, rec, n_boot=3000)
                out.append({
                    "band": band, "montage": montage, "kind": "outcome",
                    "metric": metric,
                    "n_SF": stat["n_seizure_free"], "n_rec": stat["n_recurrence"],
                    "value": float("nan"),
                    "auc": stat["auc"], "lo": stat["auc_lo"], "hi": stat["auc_hi"],
                    "p": stat["p_permutation"],
                })
            for metric in AGREEMENT_METRICS:
                values = per_subject[metric].dropna()
                out.append({
                    "band": band, "montage": montage, "kind": "agreement",
                    "metric": metric,
                    "n_SF": int(per_subject[per_subject.seizure_free][metric].notna().sum()),
                    "n_rec": int(per_subject[~per_subject.seizure_free][metric].notna().sum()),
                    "value": float(values.mean()) if len(values) else float("nan"),
                    "auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                    "p": float("nan"),
                })
    return pd.DataFrame(out)


def report(rows, result) -> None:
    import pandas as pd

    pd.set_option("display.width", 220)
    for band in rows.band.unique():
        part = rows[rows.band == band]
        print(f"\n=== {band}: {part.subject.nunique()} subjects × "
              f"{part.window.nunique()} windows at {part.threshold_sd.iloc[0]:g} SD")
        print("  per window (median over windows):")
        for montage in MONTAGES:
            arm = part[part.montage == montage]
            if arm.empty:
                continue
            print(f"    {montage:12s} {arm.n_channels.median():4.0f} channels, "
                  f"{arm.n_events.median():6.0f} events, "
                  f"{(arm.n_events == 0).mean() * 100:3.0f}% of windows empty")
        table = result[result.band == band]
        print("\n  against surgical outcome (AUC seizure-free vs recurrence):")
        outcome = table[table.kind == "outcome"].pivot(index="metric", columns="montage",
                                                       values="auc")
        print(outcome.reindex(OUTCOME_METRICS)[list(MONTAGES)].round(3).to_string())
        print("\n  patients retained (seizure-free / recurrence):")
        kept = table[table.kind == "outcome"].assign(
            kept=lambda t: t.n_SF.astype(str) + "/" + t.n_rec.astype(str)
        ).pivot(index="metric", columns="montage", values="kept")
        print(kept.reindex(OUTCOME_METRICS)[list(MONTAGES)].to_string())
        print("\n  against the expert markings (cohort means of per-patient means):")
        agree = table[table.kind == "agreement"].pivot(index="metric", columns="montage",
                                                       values="value")
        print(agree.reindex(AGREEMENT_METRICS)[list(MONTAGES)].round(3).to_string())
    print("\nprecision/recall/f1 for the referential and average arms are on events")
    print("projected onto the reviewed pairs; see the module docstring.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_montage_comparison", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", nargs="+", default=BANDS)
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--windows", type=int, default=5,
                        help="how many cached 60 s windows per subject")
    parser.add_argument("--from-csv", default=None,
                        help="skip the analysis and report this extract instead")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "results"
                                             / "montage_screen.csv"))
    parser.add_argument("--partial-dir", default=str(REPO / "artifacts" / "results"
                                                     / "montage_partial"),
                        help="per-subject extracts, so an interrupted run resumes")
    args = parser.parse_args(argv)

    import pandas as pd

    if args.from_csv:
        rows = pd.read_csv(args.from_csv)
    else:
        rows = screen(args.bands, args.subjects, args.windows,
                      partial_dir=Path(args.partial_dir))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        print(f"\nwrote {len(rows)} rows to {out}")

    report(rows, analyse(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
