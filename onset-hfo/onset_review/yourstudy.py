"""Your study and your cohort: the published analyses, run with your choices.

Two things a reader can do that the study pages alone cannot, both kept
apart from the published record and labelled as theirs wherever they show:

**Re-run the study with your settings.** The Detectors study scores the
detectors against the experts' markings on the archive's 20 patients; the
Outcome study asks whether the busiest channels were inside the resection
of the patients who became seizure-free. Both are run here exactly as the
project ran them -- `benchmark_subject` and `outcome_subject`, the same
functions -- with the band, detectors, thresholds and window the reader
chooses. The result is written beside the reader's settings
(``study/your_sweep.csv`` and ``study/your_outcome.csv`` next to the
assistant's settings) and drawn dashed next to the published lines; the
committed tables are never touched. A patient whose recording cannot be
read is skipped and named, never silently dropped.

**Your cohort.** A reader's own recordings, each with the resection and the
outcome the reader enters, measured the way the Outcome study measures a
patient: the channels tied with the busiest (`candidate_channels`), and the
share of them inside the resection. With two or more patients in each group
the comparison is the study's own (`rank_comparison`), with the smallest
effect a cohort that size could detect beside it. Nothing here is
validated; the page says so above every number.

Qt-free; the window's controls are in `onset_review.studypages`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["state_dir", "rerun_detectors", "load_your_sweep", "rerun_outcome",
           "load_your_outcome", "cohort_entries", "add_to_cohort", "remove_from_cohort",
           "cohort_frame", "compare_cohort", "contacts_of", "measure"]

OUTCOMES = {"seizure-free": True, "recurrence": False}


def state_dir() -> Path:
    from onset_review.assistant_config import config_path

    return config_path().with_name("study")


# -- re-running the Detectors study ------------------------------------------------
def rerun_detectors(subjects, thresholds, detectors, bands=("ripple",), t_start=0.0,
                    t_stop=60.0, progress=None, should_stop=None, score=None) -> dict:
    """Score `detectors` at `thresholds` on each of `subjects` against the
    experts, as the published sweep was scored. `progress(i, n, subject)`
    is told of each patient; `should_stop()` ends the run between patients.
    `score` replaces `benchmark_subject` (tests). Returns {summary, rows,
    skipped, settings} and writes them under `state_dir()`."""
    if score is None:
        from onset_hfo.benchmark import benchmark_subject as score
    subjects = list(subjects)
    rows, ranks, skipped = [], [], {}
    for index, subject in enumerate(subjects):
        if should_stop is not None and should_stop():
            skipped.update({s: "stopped before it ran" for s in subjects[index:]})
            break
        if progress is not None:
            progress(index, len(subjects), subject)
        try:
            result = score(subject, t_start=float(t_start), t_stop=float(t_stop),
                           thresholds=tuple(float(t) for t in thresholds),
                           detectors=tuple(detectors), bands=tuple(bands), verbose=False)
        except Exception as error:      # noqa: BLE001 - named in the result, not raised
            skipped[subject] = f"{type(error).__name__}: {error}"
            continue
        rows += result.rows
        ranks += result.ranks
    summary = _sweep_summary(pd.DataFrame(rows), pd.DataFrame(ranks))
    settings = {"subjects": subjects, "thresholds": [float(t) for t in thresholds],
                "detectors": list(detectors), "bands": list(bands),
                "window_s": [float(t_start), float(t_stop)], "skipped": skipped,
                "n_subjects": int(summary["n_subjects"].max()) if len(summary) else 0,
                "when": time.strftime("%Y-%m-%d %H:%M")}
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(folder / "your_sweep.csv", index=False)
    pd.DataFrame(rows).to_csv(folder / "your_sweep_patients.csv", index=False)
    (folder / "your_sweep.json").write_text(json.dumps(settings, indent=1) + "\n")
    return {"summary": summary, "rows": pd.DataFrame(rows), "skipped": skipped,
            "settings": settings}


def _sweep_summary(scores: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    """Cohort means per band, detector and threshold, in the committed
    sweep's columns, with how many patients each mean is over."""
    columns = ["band", "detector", "threshold_sd", "precision", "recall", "f1", "detections",
               "rank_rho", "top5_overlap", "n_subjects"]
    if scores.empty:
        return pd.DataFrame(columns=columns)
    keys = ["band", "detector", "threshold_sd"]
    agg = scores.groupby(keys).agg(precision=("precision", "mean"), recall=("recall", "mean"),
                                   f1=("f1", "mean"), detections=("n_detections", "mean"),
                                   n_subjects=("subject", "nunique"))
    if not ranks.empty:
        rho = ranks.groupby(keys).agg(rank_rho=("spearman_rho", "mean"),
                                      top5_overlap=("top5_overlap", "mean"))
        agg = agg.join(rho)
    out = agg.reset_index()
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns]


def _read(name: str) -> tuple[pd.DataFrame | None, dict]:
    folder = state_dir()
    try:
        frame = pd.read_csv(folder / f"{name}.csv")
        settings = json.loads((folder / f"{name}.json").read_text())
    except (OSError, ValueError):
        return None, {}
    return frame, settings


def load_your_sweep() -> tuple[pd.DataFrame | None, dict]:
    """The last re-run of the Detectors study, and its settings."""
    return _read("your_sweep")


# -- re-running the Outcome study --------------------------------------------------
def rerun_outcome(subjects, detector="rms", band="fast_ripple", threshold_sd=None,
                  t_start=0.0, t_stop=300.0, progress=None, should_stop=None,
                  measure_subject=None, resections=None, participants=None) -> dict:
    """The Outcome study with your detector, band, threshold and window, on
    the reviewed channels as published. Returns {cohort, groups, skipped,
    settings}: `cohort` has the columns the outcome chart reads."""
    from onset_hfo.outcome import compare_groups

    if measure_subject is None:
        from onset_hfo.outcome import outcome_subject as measure_subject
    if resections is None or participants is None:
        from onset_hfo.clinical import fetch_participants, resection_map

        resections = resections if resections is not None else resection_map("ds003498")
        participants = (participants if participants is not None
                        else fetch_participants("ds003498"))
    subjects = list(subjects)
    rows, skipped = [], {}
    for index, subject in enumerate(subjects):
        if should_stop is not None and should_stop():
            skipped.update({s: "stopped before it ran" for s in subjects[index:]})
            break
        if progress is not None:
            progress(index, len(subjects), subject)
        resection = resections.get(subject)
        if resection is None or not len(resection):
            skipped[subject] = "no resected zone in the clinical sheet"
            continue
        try:
            found, _channels, _meta = measure_subject(
                subject, resection, t_start=float(t_start), t_stop=float(t_stop),
                detector=detector, threshold_sd=threshold_sd, bands=(band,), verbose=False)
        except Exception as error:      # noqa: BLE001 - named in the result
            skipped[subject] = f"{type(error).__name__}: {error}"
            continue
        rows += found
    frame = pd.DataFrame(rows)
    groups = compare_groups(frame, participants) if len(frame) else pd.DataFrame()
    cohort = _outcome_cohort(frame, participants, detector, band)
    settings = {"subjects": subjects, "detector": detector, "band": band,
                "threshold_sd": threshold_sd, "window_s": [float(t_start), float(t_stop)],
                "skipped": skipped, "n_subjects": int(len(cohort)),
                "when": time.strftime("%Y-%m-%d %H:%M")}
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(folder / "your_outcome.csv", index=False)
    groups.to_csv(folder / "your_outcome_groups.csv", index=False)
    (folder / "your_outcome.json").write_text(json.dumps(settings, indent=1) + "\n")
    return {"cohort": cohort, "groups": groups, "skipped": skipped, "settings": settings}


def _outcome_cohort(frame: pd.DataFrame, participants: pd.DataFrame, detector: str,
                    band: str) -> pd.DataFrame:
    columns = ["subject", "seizure_free", "candidates_resected_expert",
               "candidates_resected_rms"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    part = frame[(frame["scope"] == "reviewed") & (frame["band"] == band)]
    wide = part.pivot_table(index="subject", columns="source", values="candidates_resected",
                            aggfunc="first")
    out = pd.DataFrame({"subject": wide.index})
    out["candidates_resected_expert"] = wide.get("expert", pd.Series(dtype=float)).to_numpy() \
        if "expert" in wide else np.nan
    # The chart's detector panel reads `_rms`; the column holds whichever
    # detector was chosen, and the settings say which.
    out["candidates_resected_rms"] = wide[detector].to_numpy() if detector in wide else np.nan
    outcome = participants.set_index("subject")["outcome"]
    out["seizure_free"] = [outcome.get(s) == "S" for s in out["subject"]]
    return out[columns]


def load_your_outcome() -> tuple[pd.DataFrame | None, dict]:
    frame, settings = _read("your_outcome")
    groups = None
    try:
        groups = pd.read_csv(state_dir() / "your_outcome_groups.csv")
    except (OSError, ValueError):
        pass
    if settings:
        settings["groups"] = groups
    return frame, settings


# -- your cohort ---------------------------------------------------------------------
def _cohort_file() -> Path:
    return state_dir() / "cohort.json"


def cohort_entries() -> list[dict]:
    try:
        entries = json.loads(_cohort_file().read_text())
        return [e for e in entries if isinstance(e, dict)]
    except (OSError, ValueError, TypeError):
        return []


def _write_cohort(entries: list[dict]) -> None:
    path = _cohort_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=1) + "\n")


def contacts_of(channels) -> list[str]:
    """The contacts behind channel names: both ends of a bipolar pair."""
    out: list[str] = []
    for name in channels:
        for contact in str(name).split("-"):
            contact = contact.strip().upper()
            if contact and contact not in out:
                out.append(contact)
    return out


def measure(session, resected_contacts) -> dict:
    """One patient as the Outcome study measures one: the busiest channel's
    tie-aware set from the window's counts (`candidate_channels`), and the
    share of it inside the resection, by the study's own functions."""
    from onset_hfo.clinical import Resection, classify_channels
    from onset_hfo.outcome import _candidate_metrics, _top_channel_resected, candidate_channels

    findings = session.findings
    counts = findings.set_index("channel")["n_events"].astype(float) \
        if findings is not None and len(findings) else pd.Series(dtype=float)
    channels = list(session.raw.ch_names)
    counts = counts.reindex(channels, fill_value=0.0)
    resection = Resection(subject=session.request.subject,
                          resected=tuple(c.upper() for c in resected_contacts), eloquent=())
    zones = classify_channels(channels, resection).set_index("channel")["zone"]
    seconds = float(session.span[1] - session.span[0]) if session.span else \
        float(session.raw.times[-1])
    minutes = seconds / 60.0
    metrics = _candidate_metrics(counts, zones, minutes)
    return {**metrics, "top_channel_resected": _top_channel_resected(counts, zones),
            "candidates": candidate_channels(counts, minutes),
            "n_resected_channels": int((zones == "resected").sum()),
            "n_channels": len(channels), "seconds": seconds}


def add_to_cohort(session, outcome: str, resected_contacts, label: str = "") -> dict:
    """Measure the open window and keep it in your cohort, replacing an
    earlier entry for the same recording and window."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome is one of {list(OUTCOMES)}, not {outcome!r}")
    resected = [c.upper() for c in resected_contacts]
    if not resected:
        raise ValueError("name at least one resected contact")
    request = session.request
    where = str(request.path) if request.path is not None else \
        f"{request.dataset}/{request.subject}/run-{request.run}"
    entry = {
        "id": f"{where}@{request.t_start:g}-{request.t_stop:g}",
        "label": label or request.subject, "recording": where,
        "window_s": [float(request.t_start), float(request.t_stop)],
        "band": request.band, "detector": request.detectors[0],
        "threshold_sd": request.threshold_sd, "outcome": outcome,
        "seizure_free": OUTCOMES[outcome], "resected": resected,
        "added": time.strftime("%Y-%m-%d %H:%M"),
        **{k: v for k, v in measure(session, resected).items()},
    }
    entries = [e for e in cohort_entries() if e.get("id") != entry["id"]]
    entries.append(entry)
    _write_cohort(entries)
    return entry


def remove_from_cohort(entry_id: str) -> bool:
    entries = cohort_entries()
    kept = [e for e in entries if e.get("id") != entry_id]
    if len(kept) == len(entries):
        return False
    _write_cohort(kept)
    return True


def cohort_frame() -> pd.DataFrame:
    """Your cohort as the outcome chart reads a cohort."""
    entries = cohort_entries()
    columns = ["subject", "seizure_free", "candidates_resected_rms", "n_candidates",
               "top_channel_resected", "band", "detector", "recording", "id"]
    if not entries:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([{
        "subject": e.get("label", ""), "seizure_free": bool(e.get("seizure_free")),
        "candidates_resected_rms": e.get("candidates_resected", np.nan),
        "n_candidates": e.get("n_candidates", 0),
        "top_channel_resected": e.get("top_channel_resected", np.nan),
        "band": e.get("band", ""), "detector": e.get("detector", ""),
        "recording": e.get("recording", ""), "id": e.get("id", "")} for e in entries])
    return frame[columns]


def compare_cohort(frame: pd.DataFrame | None = None) -> dict:
    """Seizure-free against recurrence on the share of the tied set inside
    the resection, by the study's own comparison, with the smallest AUC a
    cohort of this size could have detected. Groups under two say so."""
    from onset_hfo.outcome import min_detectable_auc, rank_comparison

    frame = cohort_frame() if frame is None else frame
    free = frame.loc[frame["seizure_free"].astype(bool), "candidates_resected_rms"]
    recur = frame.loc[~frame["seizure_free"].astype(bool), "candidates_resected_rms"]
    out = rank_comparison(free.to_numpy(float), recur.to_numpy(float), n_boot=2000)
    n1, n2 = out["n_seizure_free"], out["n_recurrence"]
    out["floor"] = min_detectable_auc(n1, n2) if n1 >= 2 and n2 >= 2 else float("nan")
    out["mixed_settings"] = bool(len(frame) and (frame["band"].nunique() > 1
                                                 or frame["detector"].nunique() > 1))
    return out
