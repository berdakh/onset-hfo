"""Does the Epileptogenicity Index find the clinicians' onset zone? ds003029.

For every patient in the ictal archive with clinician-marked seizure onset
zone contacts (``sourcedata/clinical_data_summary.xlsx``, via
`onset_hfo.cohort`), up to `max_seizures` seizures: the electrographic onset
is read from the run's own events (`onset_hfo.datasets._seizure_times`), a
window around it is fetched, the bipolar montage built with the project's
default preprocessing, and the index computed (`onset_hfo.ictal`). A bipolar
channel counts as onset zone when either of its contacts is one.

Per seizure, and per patient over the median of their seizures:

* **AUC** -- how well the index separates onset-zone channels from the rest
  (0.5 is chance, 1 is perfect);
* the same for the plain post-change energy ratio, without the timing, to
  see what the timing adds;
* whether the top channel is in the zone.

Across patients: the median AUC with a bootstrap 95% interval (patients
resampled), the share above 0.5, a Wilcoxon signed-rank test against 0.5,
and the same for the seizure-free patients alone, whose zones are the
trustworthy ones (their resection worked).

The clinicians' zone is a clinical judgement made with everything they had,
including this kind of reading of the seizure; agreement with it is not
independent ground truth. Each window's data is deleted after use; the
tables are kept.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["run_study", "auc", "bootstrap_median", "summarise", "WINDOW_BEFORE_S",
           "WINDOW_AFTER_S"]

WINDOW_BEFORE_S = 35.0
WINDOW_AFTER_S = 35.0


def auc(scores, positive) -> float:
    """Mann-Whitney AUC of `scores` for `positive` (ties count half)."""
    from scipy.stats import mannwhitneyu

    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    a, b = scores[positive], scores[~positive]
    if not len(a) or not len(b):
        return float("nan")
    return float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def bootstrap_median(values, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(values):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(n, len(values)), replace=True), axis=1)
    return (float(np.median(values)), float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)))


def _in_zone(channel: str, zone: set[str]) -> bool:
    return any(part.strip().upper() in zone for part in str(channel).split("-"))


def _events(subject: str, run: str, acq: str, spec) -> pd.DataFrame | None:
    from onset_hfo import datasets

    stem = datasets._stem(spec, subject, run, spec.session, spec.task, acq or spec.acq)
    try:
        blob = datasets._http_get(datasets._dataset_url(spec, f"{stem}_events.tsv"))
    except Exception:       # noqa: BLE001 - a run without events is skipped
        return None
    return datasets.read_tsv_text(blob.decode("utf-8", "replace"))


def run_study(out: str | Path, subjects=None, max_seizures: int = 3, settings=None,
              progress=print, keep_data: bool = False, shift_s: float = 0.0) -> dict:
    """Run the study and write per_seizure.csv, per_patient.csv and
    summary.json into `out`. Resumable: seizures already in per_seizure.csv
    are not fetched again.

    `shift_s` > 0 is the **control**: the same seizures, the same method, but
    the "onset" put `shift_s` seconds before the real one, so the whole
    analysed stretch is before the seizure. It must be at least
    `WINDOW_AFTER_S`; seizures with too little recording before them are
    left out. Agreement with the zone there is what the method finds in the
    zone's channels without a seizure."""
    if shift_s and shift_s < WINDOW_AFTER_S:
        raise ValueError(f"a control shift under {WINDOW_AFTER_S:g} s reaches the seizure")
    from onset_hfo import cohort, datasets
    from onset_hfo.config import DATASETS, PreprocessConfig
    from onset_hfo.ictal import IctalSettings, epileptogenicity
    from onset_hfo.preprocess import prepare

    settings = settings or IctalSettings()
    spec = DATASETS["ds003029"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seizure_path = out / "per_seizure.csv"
    done = pd.read_csv(seizure_path, dtype={"run": str}) if seizure_path.exists() \
        else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["subject"], str(r["run"])) for r in rows}
    channel_rows = []
    channel_path = out / "per_channel.csv"
    if channel_path.exists():
        channel_rows = pd.read_csv(channel_path, dtype={"run": str}).to_dict("records")
    cache = Path(tempfile.mkdtemp(prefix="onset-ictal-"))
    subjects = subjects or datasets.list_subjects("ds003029")
    try:
        for subject in subjects:
            labels = cohort.soz_labels(subject)
            if labels.source != "clinical_summary" or not labels.soz_contacts:
                progress(f"{subject}: no clinician onset zone, skipped")
                continue
            runs = datasets.list_runs(subject, dataset="ds003029")
            if runs is None or not len(runs):
                continue
            runs = runs[runs["task"].str.lower() == "ictal"].head(max_seizures)
            for row in runs.itertuples():
                key = (subject, str(row.run))
                if key in seen:
                    continue
                events = _events(subject, str(row.run), str(row.acq), spec)
                onset, _offset, kind = datasets._seizure_times(events, with_kind=True)
                base = {"subject": subject, "run": str(row.run), "acq": row.acq,
                        "site": labels.site, "seizure_free": labels.seizure_free,
                        "engel": labels.engel, "onset_s": onset, "marker": kind}
                if onset is None or onset < WINDOW_BEFORE_S:
                    rows.append({**base, "status": "no usable onset mark"})
                    seen.add(key)
                    continue
                if shift_s:
                    if onset - shift_s < WINDOW_BEFORE_S:
                        rows.append({**base, "status": "too little recording before the "
                                     "seizure for the control"})
                        seen.add(key)
                        continue
                    onset = onset - shift_s
                    base["analysed_onset_s"] = onset
                started = time.monotonic()
                try:
                    rec = datasets.fetch_slice(
                        subject, task="ictal", run=str(row.run), acq=row.acq or None,
                        t_start=round(onset - WINDOW_BEFORE_S, 3),
                        t_stop=round(onset + WINDOW_AFTER_S, 3), dataset="ds003029",
                        cache_dir=cache, verbose=False)
                    prep = prepare(rec, PreprocessConfig(), verbose=False)
                    table = epileptogenicity(prep.data, list(prep.ch_names), prep.sfreq,
                                             onset_s=onset, start_s=prep.t_offset,
                                             settings=settings)
                except Exception as error:      # noqa: BLE001 - a failure is a row
                    rows.append({**base, "status": f"{type(error).__name__}: {error}"[:200]})
                    seen.add(key)
                    continue
                finally:
                    if not keep_data:
                        shutil.rmtree(cache, ignore_errors=True)
                        cache.mkdir(exist_ok=True)
                zone = [_in_zone(ch, labels.soz_contacts) for ch in table["channel"]]
                table = table.assign(subject=subject, run=str(row.run), soz=zone)
                channel_rows += table.to_dict("records")
                top = table.iloc[0]
                rows.append({**base, "status": "ok", "sfreq": prep.sfreq,
                             "n_channels": len(table), "n_soz": int(sum(zone)),
                             "n_detected": int(table["detected"].sum()),
                             "auc_ei": auc(table["ei"], zone),
                             "auc_er": auc(table["er_after"].fillna(0.0), zone),
                             "top_in_zone": bool(top["soz"]),
                             "seconds": round(time.monotonic() - started, 1)})
                seen.add(key)
                progress(f"{subject} run {row.run}: AUC {rows[-1]['auc_ei']:.2f}, "
                         f"top {top['channel']} {'in' if top['soz'] else 'outside'} the zone")
                pd.DataFrame(rows).to_csv(seizure_path, index=False)
                pd.DataFrame(channel_rows).to_csv(channel_path, index=False)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    seizures = pd.DataFrame(rows)
    seizures.to_csv(seizure_path, index=False)
    patients = per_patient(pd.DataFrame(channel_rows), seizures)
    patients.to_csv(out / "per_patient.csv", index=False)
    summary = summarise(patients, seizures, settings)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def per_patient(channels: pd.DataFrame, seizures: pd.DataFrame) -> pd.DataFrame:
    """Each patient over their seizures: the median index per channel, then
    the AUC of that against the zone."""
    from onset_hfo.ictal import EI_CUTOFF

    rows = []
    ok = seizures[seizures.get("status", "") == "ok"] if len(seizures) else seizures
    for subject, part in channels.groupby("subject", sort=True):
        merged = part.groupby("channel").agg(ei=("ei", "median"), soz=("soz", "max"),
                                             er_after=("er_after", "median"),
                                             high=("ei", lambda v: int((v >= EI_CUTOFF).sum())))
        meta = ok[ok["subject"] == subject].iloc[0]
        top = merged.sort_values("ei", ascending=False).iloc[0]
        rows.append({"subject": subject, "site": meta["site"],
                     "seizure_free": meta["seizure_free"], "engel": meta["engel"],
                     "n_seizures": int(part["run"].nunique()), "n_channels": len(merged),
                     "n_soz": int(merged["soz"].sum()),
                     "auc_ei": auc(merged["ei"], merged["soz"]),
                     "auc_er": auc(merged["er_after"].fillna(0.0), merged["soz"]),
                     "top_in_zone": bool(top["soz"])})
    return pd.DataFrame(rows)


def summarise(patients: pd.DataFrame, seizures: pd.DataFrame, settings) -> dict:
    from scipy.stats import wilcoxon

    def block(part: pd.DataFrame, column: str) -> dict:
        values = part[column].dropna().to_numpy(float)
        median, low, high = bootstrap_median(values)
        out = {"n": int(len(values)), "median": median, "ci_low": low, "ci_high": high,
               "share_above_half": float((values > 0.5).mean()) if len(values) else float("nan")}
        if len(values) >= 6:
            out["wilcoxon_p"] = float(wilcoxon(values - 0.5).pvalue)
        return out

    seizure_free = patients[patients["seizure_free"].astype(str) == "True"]
    not_free = patients[patients["seizure_free"].astype(str) == "False"]
    ok = seizures[seizures.get("status", "") == "ok"] if len(seizures) else seizures
    return {
        "when": time.strftime("%Y-%m-%d"),
        "method": settings.describe(),
        "seizures_analysed": int(len(ok)),
        "seizures_skipped": int(len(seizures) - len(ok)),
        "patients": int(len(patients)),
        "ei_all": block(patients, "auc_ei"), "er_all": block(patients, "auc_er"),
        "ei_seizure_free": block(seizure_free, "auc_ei"),
        "ei_not_seizure_free": block(not_free, "auc_ei"),
        "ei_per_seizure": block(ok, "auc_ei"),
        "top_in_zone_patients": float(patients["top_in_zone"].mean()) if len(patients)
        else float("nan"),
        "by_site": {site: block(part, "auc_ei") for site, part in patients.groupby("site")},
    }


if __name__ == "__main__":      # pragma: no cover - a long network run
    import sys

    os.environ.pop("ONSET_HFO_OFFLINE", None)
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/ictal_ds003029")
    shift = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    print(json.dumps(run_study(target, shift_s=shift), indent=1, default=str))
