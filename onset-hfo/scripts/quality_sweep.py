"""Does the data-quality stage help, or does it delete findings?

Run over every cached window of ds003498 -- 20 subjects x 5 minutes -- and ask
three questions that a three-subject spot check could not answer.

1. **Does it change any published number?** The rate table and the leading
   channel, computed with the stage on and off, for every window.
2. **What does it act on?** Every contact set aside or flagged, with the
   measurement behind the verdict.
3. **Is a flag a fault, or is it the finding?** This is the one that matters.
   The archive ships expert HFO markings per bipolar channel, so for every
   flagged contact we can ask how many ripples the annotators themselves
   marked on it. A check that flags the contacts the experts marked most
   heavily is not finding noise -- it is finding epilepsy, and would be
   deleting the result if it removed them.

   Question 3 is why the band-power check flags rather than removes. This
   sweep is the cohort-scale version of the sub-13 observation that caused
   that change: `TR1-TR2` and `TR2-TR3` were flagged, and the annotators had
   marked 91, 102 and 164 ripples on those three contacts.

Writes two tables to `artifacts/quality_sweep/`: one row per window, and one
row per verdict. Offline: everything it reads is already cached.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("ONSET_HFO_OFFLINE", "1")   # refuse to fetch, loudly
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from onset_hfo.config import PipelineConfig  # noqa: E402
from onset_hfo.datasets import fetch_slice  # noqa: E402
from onset_hfo.pipeline import run_pipeline  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "quality_sweep"
SUBJECTS = [f"sub-{n:02d}" for n in range(1, 21)]
WINDOWS = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300)]


def expert_counts(record) -> dict[str, int]:
    """Ripples the archive's own annotators marked, per bipolar channel."""
    frame = getattr(record, "ground_truth", None)
    if frame is None or not len(frame) or "channel" not in frame.columns:
        return {}
    return (frame[frame["kind"].astype(str) == "ripple"]
            .groupby("channel").size().to_dict())


def one_window(subject: str, t_start: int, t_stop: int) -> tuple[dict, list[dict]]:
    record = fetch_slice(dataset="ds003498", subject=subject, run="01",
                         t_start=t_start, t_stop=t_stop, verbose=False)
    off = PipelineConfig()
    off.check_quality = False
    without = run_pipeline(record, config=off, verbose=False)
    with_ = run_pipeline(record, verbose=False)

    a = without.rates["rms"].set_index("channel")["rate_per_min"].sort_index()
    b = with_.rates["rms"].set_index("channel")["rate_per_min"].sort_index()
    identical = bool(a.index.equals(b.index)
                     and np.allclose(a.values, b.values, equal_nan=True))

    quality, segments = with_.quality, with_.segments
    marked = expert_counts(record)
    reviewed = set(getattr(record, "reviewed_channels", []) or [])
    ranks = {str(row.channel): i + 1
             for i, row in enumerate(without.rates["rms"].itertuples())}

    verdicts = []
    for row in quality[(~quality["good"]) | quality["flagged"]].itertuples():
        channel = str(row.channel)
        verdicts.append({
            "subject": subject, "t_start": t_start,
            "channel": channel,
            "verdict": "set aside" if not row.good else "flagged",
            "reason": row.reason,
            "amplitude_uv": round(float(row.amplitude_uv), 2),
            "hf_ratio_sd": round(float(row.hf_ratio_sd), 2),
            "line_fraction": round(float(row.line_fraction), 4),
            "clipped_fraction": round(float(row.clipped_fraction), 4),
            # The decisive columns: what the archive's experts said about it.
            "expert_ripples": int(marked.get(channel, 0)),
            "expert_reviewed": channel in reviewed,
            "rank_without_stage": ranks.get(channel, 0),
            "rate_without_stage": round(float(a.get(channel, float("nan"))), 2),
            "n_channels": int(len(quality)),
        })

    bad_seconds = (float((~segments["good"]).mean()) if len(segments) else 0.0)
    return {
        "subject": subject, "t_start": t_start, "t_stop": t_stop,
        "n_channels": int(len(quality)),
        "n_set_aside": int((~quality["good"]).sum()),
        "n_flagged": int(quality["flagged"].sum()),
        "pct_seconds_rejected": round(100.0 * bad_seconds, 3),
        "rates_identical": identical,
        "leader_with": str(with_.rates["rms"]["channel"].iloc[0]),
        "leader_without": str(without.rates["rms"]["channel"].iloc[0]),
        "n_expert_marked_channels": int(sum(1 for v in marked.values() if v)),
    }, verdicts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, verdicts = [], []
    started = time.perf_counter()
    jobs = [(s, a, b) for s in SUBJECTS for a, b in WINDOWS]
    for index, (subject, a, b) in enumerate(jobs, 1):
        try:
            row, found = one_window(subject, a, b)
        except Exception as problem:
            print(f"[{index:3d}/{len(jobs)}] {subject} {a}-{b}s FAILED: "
                  f"{type(problem).__name__}: {problem}", flush=True)
            continue
        rows.append(row)
        verdicts.extend(found)
        print(f"[{index:3d}/{len(jobs)}] {subject} {a}-{b}s: "
              f"{row['n_channels']:3d} ch, aside {row['n_set_aside']}, "
              f"flagged {row['n_flagged']}, "
              f"{row['pct_seconds_rejected']:.2f}% seconds, "
              f"identical {row['rates_identical']}, "
              f"leader {row['leader_without']}"
              + ("" if row["leader_with"] == row["leader_without"]
                 else f" -> {row['leader_with']}  *** LEADER CHANGED ***"),
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "windows.csv", index=False)
        pd.DataFrame(verdicts).to_csv(OUT / "verdicts.csv", index=False)

    print(f"\ndone in {time.perf_counter() - started:.0f} s; "
          f"{len(rows)} windows, {len(verdicts)} verdicts -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
