"""What the peak-contrast check costs and buys against the expert markings.

`ValidationConfig.min_peak_contrast_db` was set on synthetic signals, where
discharges with no oscillation in them score under 5 dB and implanted ripples
a 5th percentile of 7 to 8 dB. This measures it on the recordings the rest of
the project is measured on: ds003498, the first 60 s of `run-01` of every
patient, scored on the channels the annotators reviewed -- the benchmark's own
window and scope (`onset_hfo.benchmark`).

Each detector runs once per band and threshold; the same candidates are then
validated with the check off (floor 0, which is what every committed table
was computed with) and at each floor asked for, so the difference between
two rows is the check and nothing else.

    python scripts/run_peak_contrast_cost.py                 # all 20 patients
    python scripts/run_peak_contrast_cost.py --subjects sub-01 sub-02

Writes ``data/benchmark/peak_contrast.csv`` (one row per patient, band,
detector, threshold and floor) and ``peak_contrast_events.csv.gz`` (the
contrast of every candidate the older checks accept, and whether an expert
marking matches it). Needs the archive: about 40 MB per patient, cached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from onset_hfo.benchmark import _rank_agreement  # noqa: E402
from onset_hfo.config import BANDS, DetectorConfig, ValidationConfig  # noqa: E402
from onset_hfo.datasets import fetch_slice, list_subjects  # noqa: E402
from onset_hfo.detectors import DETECTORS  # noqa: E402
from onset_hfo.evaluate import evaluate_detections  # noqa: E402
from onset_hfo.preprocess import prepare  # noqa: E402
from onset_hfo.validate import validate_events  # noqa: E402

OUT = HERE / "data" / "benchmark"
#: The thresholds the project reports at: its measured optimum for agreement
#: with the annotators in each band, and the literature value for ripples.
OPERATING = {"ripple": (2.0, 5.0), "fast_ripple": (5.0,)}
FLOORS = (0.0, 3.0, 6.0, 9.0)
TOLERANCE_S = 0.02


def _matched(event, marks: pd.DataFrame) -> bool:
    """Whether an expert marking on the same channel overlaps the event."""
    if marks.empty:
        return False
    return bool(((marks["start"] - TOLERANCE_S < event.stop)
                 & (marks["stop"] + TOLERANCE_S > event.start)).any())


def subject_rows(subject: str, floors=FLOORS, verbose: bool = True):
    recording = fetch_slice(dataset="ds003498", subject=subject, run="01",
                            t_start=0.0, t_stop=60.0, verbose=False)
    truth = recording.ground_truth
    if truth is None or not len(truth):
        raise ValueError(f"{subject} carries no expert markings")
    prep = prepare(recording, verbose=False)
    reviewed = [c for c in recording.reviewed_channels if c in prep.ch_names]
    rows, events_out = [], []
    for band_name, thresholds in OPERATING.items():
        band = getattr(BANDS, band_name)
        if not BANDS.usable(prep.sfreq, band):
            continue
        marks = truth[truth["kind"] == band_name]
        by_channel = {c: g for c, g in marks.groupby("channel")}
        expert_rate = marks.groupby("channel").size().reindex(reviewed, fill_value=0)
        for detector in ("rms", "line_length"):
            for threshold in thresholds:
                events = DETECTORS[detector](
                    prep, DetectorConfig(band=band, threshold_sd=float(threshold)),
                    channels=reviewed)
                for floor in floors:
                    validate_events(events, prep,
                                    ValidationConfig(min_peak_contrast_db=float(floor)))
                    score = evaluate_detections(events, truth, kind=band_name,
                                                channels=reviewed)
                    ours = pd.Series({c: sum(1 for e in events
                                             if e.accepted and e.channel == c)
                                      for c in reviewed})
                    rank = _rank_agreement(expert_rate, ours, 5)
                    rows.append({
                        "subject": subject, "band": band_name, "detector": detector,
                        "threshold_sd": float(threshold), "floor_db": float(floor),
                        "accepted": int(sum(e.accepted for e in events)),
                        "precision": score.precision, "recall": score.recall,
                        "f1": score.f1, "rank_rho": rank["spearman_rho"],
                        "top5_overlap": rank["top5_overlap"]})
                    if floor == 0.0:
                        empty = marks.iloc[0:0]
                        for e in events:
                            if e.accepted:
                                events_out.append({
                                    "subject": subject, "band": band_name,
                                    "detector": detector,
                                    "threshold_sd": float(threshold),
                                    "contrast_db": round(float(e.peak_contrast_db), 2),
                                    "matched": _matched(
                                        e, by_channel.get(e.channel, empty))})
    if verbose:
        print(f"[peak-contrast] {subject}: {len(reviewed)} reviewed channels, "
              f"{len(truth)} expert events", flush=True)
    return rows, events_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    subjects = args.subjects or list_subjects("ds003498")
    rows, events = [], []
    for subject in subjects:
        try:
            r, e = subject_rows(subject)
        except Exception as error:                  # noqa: BLE001 -- say, and go on
            print(f"[peak-contrast] skipped {subject}: {type(error).__name__}: {error}",
                  flush=True)
            continue
        rows += r
        events += e
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out / "peak_contrast.csv", index=False)
    pd.DataFrame(events).to_csv(args.out / "peak_contrast_events.csv.gz", index=False)
    summary = (table.groupby(["band", "detector", "threshold_sd", "floor_db"])
               [["accepted", "precision", "recall", "f1", "rank_rho", "top5_overlap"]]
               .mean().round(3))
    print(summary.to_string())
    print(f"\n{table['subject'].nunique()} patient(s) -> {args.out / 'peak_contrast.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
