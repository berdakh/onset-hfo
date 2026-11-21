"""Rebuild `data/benchmark/per_patient.csv.gz` and the two sweeps' means.

One row per patient, band, detector and threshold over the union of the
grids (`data/benchmark/README.md`), scored by `onset_hfo.benchmark.
benchmark_subject` on the first 60 s of `run-01`. The cohort means in
`agreement_sweep.csv` and `four_detector_sweep.csv` are then the means of
these rows, for exactly the cells those files already hold.

    python scripts/rerun_benchmark_per_patient.py --check-off sub-01
        # with the peak-contrast check off, one patient must reproduce the
        # committed rows exactly: the evidence that this script scores the
        # way the committed table was scored
    python scripts/rerun_benchmark_per_patient.py          # score, one file a patient
    python scripts/rerun_benchmark_per_patient.py --assemble   # rewrite the tables

Scoring writes one shard per patient under `artifacts/rerun/`, so several
invocations (`--subjects ...`) can run side by side.

Needs the 60 s slices (cached under `artifacts/data/` after the first run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from onset_hfo import benchmark  # noqa: E402
from onset_hfo.config import ValidationConfig  # noqa: E402
from onset_hfo.validate import validate_events  # noqa: E402

BENCH = HERE / "data" / "benchmark"
THRESHOLDS = (0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
              8.0, 10.0, 12.0, 15.0)
DETECTORS = ("rms", "line_length", "hilbert", "short_time_energy")
BANDS_ = ("ripple", "fast_ripple")
KEYS = ["subject", "run", "band", "detector", "threshold_sd"]
MEAN_KEYS = ["band", "detector", "threshold_sd"]


def patient_rows(subject: str, thresholds=THRESHOLDS) -> pd.DataFrame:
    run = benchmark.benchmark_subject(subject, thresholds=thresholds, detectors=DETECTORS,
                                      bands=BANDS_, verbose=False)
    scores = pd.DataFrame(run.rows)
    ranks = pd.DataFrame(run.ranks)
    return scores.merge(ranks, on=KEYS, how="left", suffixes=("", "_rank"))


def means(patients: pd.DataFrame) -> pd.DataFrame:
    return patients.groupby(MEAN_KEYS).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        detections=("n_detections", "mean"), rank_rho=("spearman_rho", "mean"),
        top5_overlap=("top5_overlap", "mean")).round(3).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--check-off", nargs="+", metavar="SUBJECT", default=None,
                        help="score these with the peak-contrast check off and compare "
                             "with the committed rows; writes nothing")
    parser.add_argument("--assemble", action="store_true",
                        help="build the committed tables from the per-patient shards")
    args = parser.parse_args()
    committed = pd.read_csv(BENCH / "per_patient.csv.gz", dtype={"run": str})
    columns = list(committed.columns)
    grid = tuple(sorted(committed["threshold_sd"].unique()))

    if args.check_off:
        benchmark.validate_events = lambda events, prep: validate_events(
            events, prep, ValidationConfig(min_peak_contrast_db=0.0))
        worst = 0.0
        for subject in args.check_off:
            mine = patient_rows(subject, grid)[columns].sort_values(KEYS).reset_index(drop=True)
            theirs = committed[committed["subject"] == subject].sort_values(KEYS) \
                .reset_index(drop=True)
            assert len(mine) == len(theirs), (len(mine), len(theirs))
            for column in columns:
                if mine[column].dtype.kind in "fi":
                    gap = (mine[column].astype(float)
                           - theirs[column].astype(float)).abs().max()
                    worst = max(worst, 0.0 if pd.isna(gap) else float(gap))
                    assert mine[column].isna().equals(theirs[column].isna()), column
                else:
                    assert (mine[column].fillna("") == theirs[column].fillna("")).all(), column
            print(f"{subject}: {len(mine)} rows reproduce (largest difference {worst:.2e})")
        return 0

    shards = HERE / "artifacts" / "rerun" / "benchmark_shards"
    if args.assemble:
        frames = [pd.read_csv(path, dtype={"run": str}) for path in sorted(shards.glob("*.csv"))]
        patients = pd.concat(frames, ignore_index=True)[columns]
        missing = set(committed["subject"]) - set(patients["subject"])
        if missing:
            print(f"not assembled: no shard for {', '.join(sorted(missing))}")
            return 1
        patients = patients.drop_duplicates(KEYS).sort_values(KEYS).reset_index(drop=True)
        assert len(patients) == len(committed), (len(patients), len(committed))
    else:
        # One file per patient, so several of these can run at once and a
        # stopped run loses one patient, not all of them.
        shards.mkdir(parents=True, exist_ok=True)
        for subject in args.subjects or sorted(committed["subject"].unique()):
            if (shards / f"{subject}.csv").exists():
                continue
            patient_rows(subject, grid)[columns].to_csv(shards / f"{subject}.csv", index=False)
            print(f"[benchmark] {subject} done", flush=True)
        print("shards written; run again with --assemble once every patient is there")
        return 0
    patients.to_csv(BENCH / "per_patient.csv.gz", index=False)
    rebuilt = means(patients)
    for name in ("agreement_sweep.csv", "four_detector_sweep.csv"):
        old = pd.read_csv(BENCH / name)
        new = old[MEAN_KEYS].merge(rebuilt, on=MEAN_KEYS, how="left")[list(old.columns)]
        assert len(new) == len(old)
        new.to_csv(BENCH / name, index=False)
        print(f"{name}: {len(new)} rows rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
