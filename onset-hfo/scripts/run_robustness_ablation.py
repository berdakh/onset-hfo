#!/usr/bin/env python
"""Is the robustness multiplier the right rule? Ablated against surgical outcome.

`onset_agent.planner.rank_channels` scores a channel as

    score = survey_rate x robustness,
    robustness = min over stricter thresholds of min(1, rate_stricter / rate_survey)

and `ROADMAP.md` has carried it as an open question since the ladder was built:
*"That is a design choice, not a law, and it is the mechanism by which the
re-planning rungs can differ from the fixed ones at all."* On `sub-pt01` the
leading channels are tied closely enough that a 5% penalty reshuffles them,
which means the rule is doing more than the evidence supports.

This measures it. Not through the ladder -- the multiplier is a property of the
*ranking rule*, so varying the planner as well would confound the answer, and
the 22-subject ds003029 cohort the ladder wants is not cached here anyway
(one subject is, and its clinical sheet is not). Instead the rule is applied
directly to the ds003498 cohort, which has what a ranking rule needs to be
judged against: 20 patients, the contacts their surgeon removed, and whether
their seizures stopped.

    # both bands, three stricter re-test points, from the cached slices
    ONSET_HFO_OFFLINE=1 python scripts/run_robustness_ablation.py

    # re-print the published tables from the committed extract
    python scripts/run_robustness_ablation.py \
        --from-csv data/outcome/robustness_ablation.csv

What is held fixed, and what varies
-----------------------------------
Everything but the scoring rule. The same prepared recording, the same
detector, the same survey threshold, the same channels, the same metric
functions. Two rules are compared:

    plain       score = survey_rate                  (what S0 and S1 produce)
    multiplied  score = survey_rate x robustness      (what S2 and S3 produce)

The planner's robustness depends on which stricter thresholds the model chose
to re-test at, so that choice is swept rather than fixed: 1.25x, 1.5x and 2.0x
the survey threshold. A rule that only helps at one arbitrary re-test point is
not a rule.

One metric cannot be computed on a multiplied score, and that is a finding
-------------------------------------------------------------------------
`candidate_channels` decides which channels are statistically tied by
overlapping **Poisson** intervals, which needs an integer event count. A score
of `rate x robustness` is not a count and has no such interval, so
`candidates_resected` -- this project's own tie-aware answer to "did the
surgeon remove what the map pointed at" -- is undefined for the multiplied
rule. It is reported for the plain rule and left NaN for the other, rather than
computed on a truncated count that would look like a number.

`tied_set_argmax_resected` covers the same ground in a way that survives the
asymmetry: the tied set is taken from the survey **counts** (valid for both
rules, since both share the same survey pass), and each rule's argmax is then
checked against the resection. That isolates exactly what the multiplier is
for -- reordering channels the data cannot separate.
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
# insert has taken effect.
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 21)]
BANDS = ["fast_ripple", "ripple"]
#: Stricter re-test points, as multiples of the band's survey threshold.
STRICTER = [1.25, 1.5, 2.0]
PLAIN = "plain"
MULTIPLIED = "multiplied"
#: Computable for both rules, so these carry the comparison.
SHARED_METRICS = ["top_channel_resected", "top3_resected", "share_in_rz",
                  "tied_set_argmax_resected"]


def _robustness(survey, stricter):
    """The planner's rule, transcribed: capped at 1, and 1 where nothing was seen.

    `rate_survey <= 0` giving 1.0 rather than 0 is the planner's own choice
    (`planner.py`: ``ratio = 1.0 if base_rate <= 0 else ...``). It means
    *unchallenged*, not *verified*, and it is kept here so the ablation
    measures the rule as shipped rather than a tidier version of it.
    """
    out = {}
    for channel, base in survey.items():
        out[channel] = 1.0 if base <= 0 else min(1.0, stricter[channel] / base)
    return out


def _argmax(score, channels):
    """The busiest channel, ties broken by name so the same data gives the same answer."""
    return max(channels, key=lambda c: (score[c], c))


def _metric_row(rule, score, multiple, where, tied, zones, duration_min):
    """Score one ranking against the resection. `where` carries the row's identity."""
    from onset_hfo.outcome import (
        _candidate_metrics,
        _share_in_resection,
        _top_channel_resected,
        _top_k_resected,
    )

    inside_tied = float("nan")
    if tied:
        # Each rule's own argmax, restricted to the set the survey counts
        # could not separate -- which is the only thing the multiplier is for.
        inside_tied = float(zones.loc[_argmax(score, tied)] == "resected")

    # `_share_in_resection` returns an `n_events` of its own -- the summed
    # score, which for the multiplied rule is score mass rather than a count.
    # It is spread first so `where`'s `n_events_survey` (the real integer count
    # from the shared survey pass) cannot be overwritten by it.
    shares = _share_in_resection(score, zones)
    row = {
        **shares, **where, "rule": rule, "stricter_x": multiple,
        "top_channel_resected": _top_channel_resected(score, zones),
        "top3_resected": _top_k_resected(score, zones, k=3),
        "tied_set_argmax_resected": inside_tied,
    }
    row["score_mass"] = row.pop("n_events")
    for zone in ("resected", "partial", "spared"):
        row[f"score_mass_{zone}"] = row.pop(f"n_events_{zone}")
    if rule == PLAIN:
        # Poisson intervals need integer counts, so the tie-aware metric
        # exists for this rule only. See the module docstring.
        row.update(_candidate_metrics(score, zones, duration_min))
    return row


def screen(bands, subjects, n_windows, stricter_multiples, run="01"):
    """One row per band x subject x window x rule x re-test point."""
    import pandas as pd

    from onset_hfo.clinical import classify_channels, resection_map
    from onset_hfo.config import BANDS as BAND_DEFS
    from onset_hfo.config import DetectorConfig, PipelineConfig
    from onset_hfo.datasets import fetch_slice
    from onset_hfo.detectors import detect_rms
    from onset_hfo.outcome import _threshold_for, candidate_channels
    from onset_hfo.preprocess import prepare
    from onset_hfo.validate import validate_events

    cfg = PipelineConfig()
    rmap = resection_map("ds003498")
    windows = [(w * 60.0, (w + 1) * 60.0) for w in range(n_windows)]
    rows = []

    def counts_at(prep, band, threshold, channels):
        det_cfg = DetectorConfig(band=band, threshold_sd=threshold)
        events = detect_rms(prep, det_cfg, channels=channels,
                            describe_spectrum=False)
        validate_events(events, prep, cfg.validation)
        series = pd.Series({c: 0.0 for c in channels}, dtype=float)
        for event in events:
            if event.accepted:
                series[event.channel] += 1.0
        return series

    for subject in subjects:
        for index, (t_start, t_stop) in enumerate(windows):
            record = fetch_slice(dataset="ds003498", subject=subject, run=run,
                                 t_start=t_start, t_stop=t_stop, verbose=False)
            prep = prepare(record, verbose=False)
            labels = classify_channels(prep.ch_names,
                                       rmap[subject]).set_index("channel")
            reviewed = [c for c in record.reviewed_channels
                        if c in prep.ch_names
                        and not bool(labels.loc[c, "eloquent"])]
            if not reviewed:
                continue
            zones = labels.loc[reviewed, "zone"]
            duration_min = prep.duration / 60.0

            for band_name in bands:
                band = getattr(BAND_DEFS, band_name)
                if not BAND_DEFS.usable(prep.sfreq, band):
                    continue
                survey_sd = _threshold_for(band_name, None)
                survey = counts_at(prep, band, survey_sd, reviewed)

                # The tied set comes from the survey counts, so both rules are
                # judged on the same set of channels the data cannot separate.
                tied = candidate_channels(survey, duration_min)

                where = {"band": band_name, "subject": subject, "window": index,
                         "survey_sd": survey_sd,
                         "n_events_survey": float(survey.sum()),
                         "n_tied": len(tied)}

                rows.append(_metric_row(PLAIN, survey, float("nan"), where,
                                        tied, zones, duration_min))

                for multiple in stricter_multiples:
                    strict = counts_at(prep, band, survey_sd * multiple, reviewed)
                    factor = pd.Series(_robustness(survey, strict))
                    score = survey * factor
                    row = _metric_row(MULTIPLIED, score, multiple, where,
                                      tied, zones, duration_min)
                    row.update({
                        "n_demoted": int((factor.reindex(reviewed) < 1.0).sum()),
                        "mean_robustness": float(factor.reindex(reviewed).mean()),
                        "reordered": float(_argmax(score, reviewed)
                                           != _argmax(survey, reviewed)),
                    })
                    rows.append(row)
                print(f"{band_name} {subject} w{index}: "
                      f"{survey.sum():.0f} events, {len(tied)} tied", flush=True)

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
        for metric in SHARED_METRICS:
            for rule in (PLAIN, MULTIPLIED):
                part = rows[(rows.band == band) & (rows.rule == rule)]
                multiples = ([float("nan")] if rule == PLAIN
                             else sorted(part.stricter_x.unique()))
                for multiple in multiples:
                    arm = (part if rule == PLAIN
                           else part[part.stricter_x == multiple])
                    per_subject = (arm.groupby(["subject", "seizure_free"])[metric]
                                   .mean().reset_index())
                    free = per_subject[per_subject.seizure_free][metric].to_numpy()
                    recurrence = per_subject[~per_subject.seizure_free][metric].to_numpy()
                    stat = rank_comparison(free, recurrence, n_boot=3000)
                    out.append({
                        "band": band, "metric": metric, "rule": rule,
                        "stricter_x": multiple,
                        "n_SF": stat["n_seizure_free"], "n_rec": stat["n_recurrence"],
                        "auc": stat["auc"], "lo": stat["auc_lo"], "hi": stat["auc_hi"],
                        "p": stat["p_permutation"],
                    })
    return pd.DataFrame(out)


def report(rows, result) -> None:
    import pandas as pd

    pd.set_option("display.width", 220)
    for band in rows.band.unique():
        part = rows[rows.band == band]
        mult = part[part.rule == MULTIPLIED]
        print(f"\n=== {band}: {part.subject.nunique()} subjects × "
              f"{part.window.nunique()} windows, survey at "
              f"{part.survey_sd.iloc[0]:g} SD")
        print(f"  events per window (median): "
              f"{part[part.rule == PLAIN].n_events_survey.median():.0f}"
              f"   tied-set size (median): {part[part.rule == PLAIN].n_tied.median():.0f}")
        if len(mult):
            print("  what the multiplier does, by re-test point:")
            for multiple, arm in mult.groupby("stricter_x"):
                print(f"    {multiple:g}x: mean robustness "
                      f"{arm.mean_robustness.mean():.3f}, "
                      f"{arm.n_demoted.mean():.1f} channels demoted per window, "
                      f"leader moved in {100 * arm.reordered.mean():.0f}% of windows")

        table = result[result.band == band]
        print(table.drop(columns="band").round(3).to_string(index=False))
        for metric in SHARED_METRICS:
            arm = table[table.metric == metric]
            plain = arm[arm.rule == PLAIN].auc
            if not len(plain) or not plain.notna().any():
                continue
            base = float(plain.iloc[0])
            best = arm[arm.rule == MULTIPLIED].auc
            if not len(best) or not best.notna().any():
                continue
            print(f"  {metric:26s} plain {base:.3f}  "
                  f"multiplied {best.min():.3f}–{best.max():.3f}  "
                  f"best gain {float(best.max()) - base:+.3f}")

    print("\ncandidates_resected is computed for the plain rule only: Poisson")
    print("intervals need integer counts, and rate x robustness is not one.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_robustness_ablation", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", nargs="+", default=BANDS)
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--windows", type=int, default=5,
                        help="how many cached 60 s windows per subject")
    parser.add_argument("--stricter", nargs="+", type=float, default=STRICTER,
                        help="re-test points, as multiples of the survey threshold")
    parser.add_argument("--from-csv", default=None,
                        help="skip the analysis and report this extract instead")
    parser.add_argument("--out", default=str(REPO / "artifacts" / "results"
                                             / "robustness_ablation.csv"))
    args = parser.parse_args(argv)

    import pandas as pd

    if args.from_csv:
        rows = pd.read_csv(args.from_csv)
    else:
        rows = screen(args.bands, args.subjects, args.windows, args.stricter)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        print(f"\nwrote {len(rows)} rows to {out}")

    report(rows, analyse(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
