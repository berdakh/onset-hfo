#!/usr/bin/env python
"""Scalp EEG: our ripple detector against a published one, on ds003555.

ds003555 (Cserpan et al., Zurich) holds sleep scalp EEG from 30 children with
epilepsy. Its derivatives cut 5-minute N3 intervals, re-referenced to 52
nearest-neighbour bipolar channels, and list the ripples (80-250 Hz) the
group's own automated detector found in each, with whether each passed its
artefact rejection. That is a reference this project has never had for scalp
EEG: a detector from the same lineage as the one this project validated on
intracranial EEG, run by its authors, on the same files.

    # the first ten subjects, downloading each interval once (~60 MB each)
    python scripts/run_scalp_comparison.py --subjects 01-10

    # re-print the published table from the committed extract
    python scripts/run_scalp_comparison.py --from-csv data/scalp/scalp_comparison.csv

What is compared
----------------
Per subject, the number of ripples on each of the 52 channels: ours (the RMS
detector, its measured default threshold, the project's quality checks and
validation, nothing re-referenced because the file is already bipolar) and
theirs, both after their artefact rejection (``theirs_passed``) and before it
(``theirs_all``). Spearman's rho across channels says whether the two rank the
channels alike; the busiest channel says whether they would point at the same
place.

What it is not
--------------
Their detector is not ground truth, and nobody here has looked at the events.
Agreement says the two detectors see the same things, not that either is
right. The intervals are their choice of clean N3 sleep, which flatters any
detector.
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "ds003555"
COLUMNS = ["subject", "sfreq", "seconds", "channels", "kept", "ours", "theirs_passed",
           "theirs_all", "rho_passed", "rho_all", "our_top", "their_top"]


def _subjects(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            out += [f"{i:02d}" for i in range(int(low), int(high) + 1)]
        elif part:
            out.append(f"{int(part):02d}")
    return out


def compare(subject: str, cache: Path) -> dict:
    import mne
    from scipy.stats import spearmanr

    from onset_hfo import openneuro
    from onset_hfo.config import PipelineConfig, PreprocessConfig
    from onset_hfo.datasets import Recording
    from onset_hfo.pipeline import run_pipeline

    base = (f"{DATASET}/derivatives/sub-{subject}/ses-01/eeg/"
            f"sub-{subject}_ses-01_task-hfo_run-01")
    edf = cache / f"sub-{subject}_run-01_eeg.edf"
    if not edf.exists():
        edf.parent.mkdir(parents=True, exist_ok=True)
        edf.write_bytes(openneuro._get(base + "_eeg.edf"))
    events = pd.read_csv(io.StringIO(openneuro._get(base + "_events.tsv").decode()), sep="\t")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    # The EDF's unit field is blank, which MNE reads as volts; channels.tsv
    # says µV (the same fix `onset_hfo.openneuro` applies when it fetches).
    if set(getattr(raw, "_orig_units", {}).values()) <= {"n/a", ""}:
        raw._data *= 1e-6
    raw.set_channel_types({n: "eeg" for n in raw.ch_names}, verbose="ERROR")
    raw.info["line_freq"] = 50.0
    record = Recording(raw=raw, source=f"{DATASET}:{subject}", subject=f"sub-{subject}",
                       task="hfo", run="01", t_offset=0.0, line_freq=50.0)
    config = PipelineConfig()
    config.preprocess = PreprocessConfig(modality="eeg", bipolar=False)
    result = run_pipeline(record, config=config, detectors=("rms",), verbose=False)
    ours = result.rates["rms"].set_index("channel")["n_events"]
    passed = events[events["EvPassRejection"] == 1]["strChannelName"].value_counts() \
        .reindex(ours.index).fillna(0)
    every = events["strChannelName"].value_counts().reindex(ours.index).fillna(0)

    def rho(other):
        return round(float(spearmanr(ours, other).statistic), 2) \
            if ours.sum() and other.sum() else np.nan

    return {"subject": subject, "sfreq": float(raw.info["sfreq"]),
            "seconds": round(float(raw.times[-1])), "channels": len(ours),
            "kept": int(result.quality["good"].sum()) if not result.quality.empty else 0,
            "ours": int(ours.sum()), "theirs_passed": int(passed.sum()),
            "theirs_all": int(every.sum()), "rho_passed": rho(passed), "rho_all": rho(every),
            "our_top": ours.idxmax() if ours.sum() else "",
            "their_top": passed.idxmax() if passed.sum() else ""}


def summarise(table: pd.DataFrame) -> str:
    agree = int((table["our_top"] == table["their_top"]).sum())
    return (f"{len(table)} subjects. Per-channel Spearman rho with their ripples after "
            f"artefact rejection: median {table['rho_passed'].median():.2f} "
            f"(range {table['rho_passed'].min():.2f}-{table['rho_passed'].max():.2f}); "
            f"before it: median {table['rho_all'].median():.2f}. Same busiest channel in "
            f"{agree} of {len(table)}. Ours found {int(table['ours'].sum())} events, they "
            f"kept {int(table['theirs_passed'].sum())} of {int(table['theirs_all'].sum())}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subjects", default="01-10")
    parser.add_argument("--cache", type=Path, default=None,
                        help="where the downloaded intervals are kept")
    parser.add_argument("--out", type=Path, default=ROOT / "data/scalp/scalp_comparison.csv")
    parser.add_argument("--from-csv", type=Path, default=None)
    args = parser.parse_args()
    if args.from_csv:
        table = pd.read_csv(args.from_csv, dtype={"subject": str})
    else:
        from onset_hfo import openneuro

        cache = args.cache or openneuro.get_data_home() / DATASET / "derivatives"
        rows = []
        for subject in _subjects(args.subjects):
            rows.append(compare(subject, cache))
            print(rows[-1], flush=True)
        table = pd.DataFrame(rows, columns=COLUMNS)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(summarise(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
