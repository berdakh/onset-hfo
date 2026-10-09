"""Which stretches of a case to analyse for interictal markers. Qt-free.

Interictal HFO rates are read from stretches chosen by rule, not from
whatever minute was open: rates differ by sleep stage (they are highest and
steadiest in non-REM sleep), rise and fall around seizures, and an artefact
or a disconnected headbox turns into "events" or into silence. So a
`SegmentRule` says what qualifies --

* **sleep stages** (default non-REM N2 and N3; none ticked means any time);
* **distance from any seizure** (default one hour either side);
* **away from artefacts** the reader marked (with a margin), and away from
  stretches the overview found flat on most contacts;
* **how long** each segment is (5 minutes) and how much in all (30 minutes),

-- and `choose` cuts what qualifies into segments, spread evenly across the
recordings rather than all from the first night, and says how much time each
rule took out. The rule and the segments are saved with the case
(``derivatives/onset/segments.tsv`` and ``segments.json``).

A seizure is the span of a *seizure* mark, or from a *seizure onset* (or
clinical onset) to the next *seizure offset* within half an hour, or, with no
offset, two minutes from the onset. Defaults are starting points to state in
the report, not established constants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SegmentRule", "choose", "describe_choice", "seizure_spans", "save_segments",
           "load_segments", "SEGMENT_COLUMNS", "STAGE_NAMES"]

SEGMENT_COLUMNS = ("segment", "run", "start", "stop", "duration", "stage",
                   "nearest_seizure_s")
STAGE_NAMES = ("sleep-W", "sleep-N1", "sleep-N2", "sleep-N3", "sleep-REM")
#: Without an offset, a seizure is taken to last this long from its onset.
SEIZURE_FALLBACK_S = 120.0
#: An onset and an offset further apart than this are not one seizure.
SEIZURE_PAIR_MAX_S = 1800.0
ARTEFACT_MARGIN_S = 5.0


@dataclass(frozen=True)
class SegmentRule:
    stages: tuple = ("sleep-N2", "sleep-N3")
    min_from_seizure_s: float = 3600.0
    segment_s: float = 300.0
    total_s: float = 1800.0
    min_segment_s: float = 60.0
    avoid_flat: bool = True

    def describe(self) -> str:
        stages = ", ".join(s.replace("sleep-", "") for s in self.stages) or "any stage"
        return (f"{stages}; at least {self.min_from_seizure_s / 3600:g} h from any seizure; "
                f"away from artefacts{' and flat stretches' if self.avoid_flat else ''}; "
                f"{self.segment_s / 60:g}-minute segments, {self.total_s / 60:g} minutes in all")


# -- intervals ----------------------------------------------------------------------------------
def _union(spans):
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(spans, cut):
    out = list(_union(spans))
    for c0, c1 in _union(cut):
        kept = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                kept.append((a, b))
                continue
            if a < c0:
                kept.append((a, c0))
            if c1 < b:
                kept.append((c1, b))
        out = kept
    return out


def _intersect(spans, keep):
    out = []
    for a, b in _union(spans):
        for k0, k1 in _union(keep):
            lo, hi = max(a, k0), min(b, k1)
            if hi > lo:
                out.append((lo, hi))
    return _union(out)


def _length(spans) -> float:
    return float(sum(b - a for a, b in spans))


# -- reading marks --------------------------------------------------------------------------
def _has_marks(marks) -> bool:
    return marks is not None and len(marks) > 0 and "trial_type" in marks


def seizure_spans(marks: pd.DataFrame) -> list[tuple[float, float]]:
    """Seizures as (start, stop) in recording seconds, overlapping ones merged."""
    spans = []
    if not _has_marks(marks):
        return spans
    frame = marks.sort_values("onset")
    kinds = frame["trial_type"].astype(str).to_numpy()
    onsets = frame["onset"].astype(float).to_numpy()
    durations = frame["duration"].astype(float).to_numpy()
    offsets = onsets[kinds == "seizure-offset"]
    for kind, onset, duration in zip(kinds, onsets, durations, strict=True):
        if kind == "seizure":
            spans.append((onset, onset + max(duration, 1.0)))
        elif kind in ("seizure-onset", "seizure-clinical-onset"):
            later = offsets[(offsets > onset) & (offsets - onset <= SEIZURE_PAIR_MAX_S)]
            spans.append((onset, float(later.min()) if len(later)
                          else onset + SEIZURE_FALLBACK_S))
    return _union(spans)


def _spans_of(marks: pd.DataFrame, kinds) -> list[tuple[float, float]]:
    if not _has_marks(marks):
        return []
    part = marks[marks["trial_type"].astype(str).isin(set(kinds))]
    return [(float(o), float(o) + float(d)) for o, d in zip(part["onset"], part["duration"],
                                                            strict=True) if float(d) > 0]


def _stage_at(marks: pd.DataFrame, a: float, b: float) -> str:
    best, overlap = "unscored", 0.0
    for stage in STAGE_NAMES:
        got = _length(_intersect([(a, b)], _spans_of(marks, [stage])))
        if got > overlap:
            best, overlap = stage, got
    return best


# -- choosing ------------------------------------------------------------------------------------
def choose(recordings: list[dict], rule: SegmentRule) -> tuple[pd.DataFrame, dict]:
    """Segments from `recordings` -- each {run, duration, marks (events
    frame), flat (optional list of (start, stop))} -- by `rule`. Returns the
    segments and the seconds each rule removed (`excluded`), plus totals."""
    candidates = []
    removed = {"not in the chosen stages": 0.0, "near a seizure": 0.0, "artefact": 0.0,
               "flat": 0.0, "too short to use": 0.0}
    total = 0.0
    scored_any = False
    for rec in recordings:
        duration = float(rec["duration"])
        marks = rec["marks"]
        total += duration
        allowed = [(0.0, duration)]
        if rule.stages:
            staged = _spans_of(marks, rule.stages)
            scored_any = scored_any or bool(_spans_of(marks, STAGE_NAMES))
            before = _length(allowed)
            allowed = _intersect(allowed, staged)
            removed["not in the chosen stages"] += before - _length(allowed)
        seizures = seizure_spans(marks)
        margin = float(rule.min_from_seizure_s)
        before = _length(allowed)
        allowed = _subtract(allowed, [(a - margin, b + margin) for a, b in seizures])
        removed["near a seizure"] += before - _length(allowed)
        before = _length(allowed)
        allowed = _subtract(allowed, [(a - ARTEFACT_MARGIN_S, b + ARTEFACT_MARGIN_S)
                                      for a, b in _spans_of(marks, ["artefact"])])
        removed["artefact"] += before - _length(allowed)
        if rule.avoid_flat and rec.get("flat"):
            before = _length(allowed)
            allowed = _subtract(allowed, rec["flat"])
            removed["flat"] += before - _length(allowed)
        for a, b in allowed:
            start = a
            while b - start >= rule.min_segment_s:
                stop = min(b, start + rule.segment_s)
                if stop - start < rule.min_segment_s:
                    break
                # Seconds to the nearest seizure, before or after (none overlap:
                # they were taken out).
                nearest = min((max(s0 - stop, start - s1, 0.0) for s0, s1 in seizures),
                              default=float("inf"))
                candidates.append({"run": str(rec["run"]), "start": round(start, 3),
                                   "stop": round(stop, 3), "duration": round(stop - start, 3),
                                   "stage": _stage_at(marks, start, stop),
                                   "nearest_seizure_s": float(nearest)})
                start = stop
            if b - start > 0:
                removed["too short to use"] += b - start
    chosen = candidates
    available = float(sum(c["duration"] for c in candidates))
    if candidates and available > rule.total_s:
        n = max(1, int(round(rule.total_s / rule.segment_s)))
        picks = np.unique(np.round(np.linspace(0, len(candidates) - 1, n)).astype(int))
        chosen = [candidates[i] for i in picks]
    frame = pd.DataFrame(chosen, columns=[c for c in SEGMENT_COLUMNS if c != "segment"])
    frame.insert(0, "segment", [f"seg-{i + 1:02d}" for i in range(len(frame))])
    info = {"excluded": {k: round(v, 1) for k, v in removed.items()},
            "total_s": round(total, 1), "available_s": round(available, 1),
            "chosen_s": round(float(frame["duration"].sum()) if len(frame) else 0.0, 1),
            "n_available": len(candidates),
            "no_sleep_scored": bool(rule.stages) and not scored_any}
    return frame, info


def describe_choice(frame: pd.DataFrame, info: dict, rule: SegmentRule) -> str:
    """The choice in a sentence or three."""
    if not len(frame):
        why = ("No sleep is scored in this case: mark sleep stages on the Annotate step, "
               "or untick the stages to use any time." if info.get("no_sleep_scored") else
               "Nothing qualifies under this rule; loosen it (fewer hours from seizures, "
               "more stages) or mark more of the recordings.")
        return why
    stages = frame["stage"].value_counts()
    parts = [f"{len(frame)} segment(s), {frame['duration'].sum() / 60:.0f} minutes, from "
             f"{frame['run'].nunique()} recording(s): "
             + ", ".join(f"{n} {s.replace('sleep-', '')}" for s, n in stages.items()) + "."]
    excluded = {k: v for k, v in info["excluded"].items() if v > 0}
    if excluded:
        parts.append("Left out: " + "; ".join(f"{v / 3600:.1f} h {k}" if v >= 360 else
                                              f"{v / 60:.0f} min {k}"
                                              for k, v in excluded.items()) + ".")
    if info["available_s"] > info["chosen_s"]:
        parts.append(f"{info['available_s'] / 60:.0f} minutes qualified; "
                     f"{info['chosen_s'] / 60:.0f} were taken, spread evenly in time.")
    return " ".join(parts)


def save_segments(case, frame: pd.DataFrame, rule: SegmentRule, info: dict) -> Path:
    folder = Path(case.derivatives)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "segments.tsv"
    frame.to_csv(path, sep="\t", index=False)
    (folder / "segments.json").write_text(json.dumps(
        {"rule": asdict(rule), "info": info}, indent=1, default=str) + "\n")
    case.record("chose segments", f"{len(frame)} segment(s), "
                f"{frame['duration'].sum() / 60 if len(frame) else 0:.0f} min: "
                f"{rule.describe()}")
    return path


def load_segments(case) -> tuple[pd.DataFrame, SegmentRule | None, dict]:
    folder = Path(case.derivatives)
    path = folder / "segments.tsv"
    if not path.exists():
        return pd.DataFrame(columns=list(SEGMENT_COLUMNS)), None, {}
    frame = pd.read_csv(path, sep="\t", dtype={"run": str, "segment": str})
    meta = json.loads((folder / "segments.json").read_text()) if \
        (folder / "segments.json").exists() else {}
    raw = meta.get("rule") or {}
    rule = SegmentRule(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in raw.items()
                          if k in SegmentRule.__dataclass_fields__}) if raw else None
    return frame, rule, meta.get("info", {})
