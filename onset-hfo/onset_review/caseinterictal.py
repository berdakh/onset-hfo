"""A case's interictal analysis: every chosen segment, analysed and pooled.

Each segment of the case (`onset_hfo.case.segments`) is analysed by the
review window's own path -- `onset_review.session.load_session` on a request
for that stretch of the case's recording, with the case's channel types, its
bad contacts left out, and its analysis settings -- so a segment here and
the same minutes opened in the window are one analysis, not two. The
segments are then pooled (`onset_hfo.case.pooling`).

A segment once analysed is kept (``derivatives/onset/interictal/cache/``,
keyed by everything that changes its numbers), so adding a segment or
re-running costs only what is new. Each run writes a dated folder beside it:
the pooled table, the per-segment counts, every detection with its time in
the recording, the settings and a summary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from onset_hfo.case import bids
from onset_hfo.case.model import new_run_folder
from onset_hfo.case.pooling import pool

__all__ = ["template_for", "save_template", "segment_request", "run_interictal",
           "InterictalResult", "latest_result", "draw_pooled", "DEFAULT_DETECTORS"]

DEFAULT_DETECTORS = ("rms", "line_length")


def template_for(case):
    """The case's analysis settings as a review request (no recording yet)."""
    from onset_review.project import request_from_dict
    from onset_review.session import ReviewRequest

    if case.analysis:
        return request_from_dict(dict(case.analysis))
    return ReviewRequest(dataset="", detectors=DEFAULT_DETECTORS)


def save_template(case, request, by: str = "") -> None:
    from onset_review.project import request_to_dict

    keep = ("detectors", "band", "threshold_sd", "with_spikes", "check_quality",
            "preprocess")
    data = {k: v for k, v in request_to_dict(request).items() if k in keep}
    case.analysis = data
    case.record("set the analysis", json.dumps(data, default=str)[:400], by)


def segment_request(case, run: str, start: float, stop: float, template):
    """The request for one stretch of a case's recording, analysed as the case says."""
    from onset_hfo.config import PreprocessConfig

    rec = case.recording(str(run))
    where = case.where(rec)
    channels = case.channels(rec)
    sidecar = json.loads(where.path("ieeg.json").read_text())
    types = tuple((str(r.name), bids.MNE_TYPES.get(str(r.type).upper(), "misc"))
                  for r in channels.itertuples())
    bad = tuple(str(r.name) for r in channels.itertuples() if str(r.status) == "bad")
    preprocess = template.preprocess or PreprocessConfig()
    if bad:
        preprocess = dataclasses.replace(
            preprocess, exclude=tuple(dict.fromkeys(tuple(preprocess.exclude) + bad)))
    return dataclasses.replace(
        template, dataset="", subject=case.case_id, run=rec.run, path=where.header,
        t_start=float(start), t_stop=float(stop), channel_types=types,
        line_freq=float(sidecar.get("PowerLineFrequency") or 50.0), preprocess=preprocess,
        span_start=None, span_stop=None)


def _key(request) -> str:
    from onset_review.project import request_to_dict

    data = request_to_dict(request)
    data["path"] = Path(str(data.get("path"))).name
    blob = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _measure(session, segment: str, run: str) -> dict:
    """What a segment contributes to the pool, from its session."""
    request = session.request
    detector = request.primary
    findings = session.findings
    channels = list(findings["channel"].astype(str)) if not findings.empty else []
    span = session.span
    seconds = {ch: float(session.clean_seconds.get(ch, 0.0)) if session.clean_seconds
               else float(span[1] - span[0]) for ch in channels}
    counts = {ch: 0 for ch in channels}
    detections = []
    for event in session.events:
        if event.detector == detector and event.accepted and event.channel in counts:
            counts[event.channel] += 1
        detections.append({"segment": segment, "run": run, "channel": event.channel,
                           "onset": round(float(event.start), 4),
                           "duration": round(float(event.stop - event.start), 4),
                           "detector": event.detector, "band": getattr(event, "band", ""),
                           "accepted": bool(event.accepted)})
    return {"segment": segment, "run": run, "start": float(span[0]), "stop": float(span[1]),
            "counts": counts, "seconds": seconds, "tied": list(session.candidates),
            "leader": str((session.leader or {}).get("leader") or ""),
            "detections": detections}


@dataclass
class InterictalResult:
    table: pd.DataFrame
    leader: dict
    per_segment: pd.DataFrame
    folder: Path
    n_segments: int
    minutes: float
    settings: dict

    def statement(self) -> str:
        if not self.leader.get("available"):
            return "No channel could be ranked."
        tied = list(self.table.loc[self.table["tied"], "channel"])
        lead = self.leader
        minutes = f"{self.minutes:.1f}" if self.minutes < 10 else f"{self.minutes:.0f}"
        head = (f"Over {self.n_segments} segment(s), {minutes} minutes: "
                f"{lead['leader']} is the busiest at {lead['leader_rate_per_min']:g}/min "
                f"(interval {lead['leader_ci'][0]:g}–{lead['leader_ci'][1]:g}). ")
        if lead.get("distinguishable"):
            return head + (f"It stands out from the median channel; {len(tied)} channel(s) "
                           f"cannot be told apart from it: {', '.join(tied[:12])}"
                           + ("…" if len(tied) > 12 else "") + ".")
        return head + ("No channel stands out: its interval overlaps the median channel's, "
                       "so the order is counting noise.")


def run_interictal(case, segments: pd.DataFrame, template=None, progress=None,
                   should_stop=None, load=None) -> InterictalResult:
    """Analyse each segment (from the cache when it was analysed before with
    the same settings), pool them, and write the results."""
    if load is None:
        from onset_review.session import load_session as load
    template = template or template_for(case)
    cache = Path(case.derivatives) / "interictal" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    measured = []
    rows = list(segments.itertuples())
    for index, row in enumerate(rows):
        if should_stop is not None and should_stop():
            raise InterruptedError(f"stopped after {index} of {len(rows)} segments")
        request = segment_request(case, str(row.run), float(row.start), float(row.stop),
                                  template)
        path = cache / f"{_key(request)}.json"
        if path.exists():
            item = json.loads(path.read_text())
        else:
            item = _measure(load(request), str(row.segment), str(row.run))
            path.write_text(json.dumps(item))
        item["segment"] = str(row.segment)
        measured.append(item)
        if progress is not None:
            progress((index + 1) / max(len(rows), 1))
    table, leader = pool(measured)
    folder = new_run_folder(Path(case.derivatives) / "interictal")
    per_segment = pd.DataFrame([
        {"segment": m["segment"], "run": m["run"], "start": m["start"], "stop": m["stop"],
         "channel": ch, "n_events": m["counts"].get(ch, 0), "seconds": m["seconds"].get(ch, 0),
         "tied": ch in set(m["tied"])}
        for m in measured for ch in m["seconds"]])
    detections = pd.DataFrame([d for m in measured for d in m["detections"]])
    from onset_review.project import request_to_dict

    settings = {"when": time.strftime("%Y-%m-%d %H:%M"),
                "analysis": {k: v for k, v in request_to_dict(template).items()
                             if k in ("detectors", "band", "threshold_sd", "with_spikes",
                                      "check_quality", "preprocess")},
                "segments": segments.to_dict("records")}
    minutes = float(sum(float(m["stop"]) - float(m["start"]) for m in measured)) / 60.0
    result = InterictalResult(table, leader, per_segment, folder, len(measured), minutes,
                              settings)
    table.to_csv(folder / "pooled.tsv", sep="\t", index=False)
    per_segment.to_csv(folder / "per_segment.tsv", sep="\t", index=False)
    detections.to_csv(folder / "detections.tsv", sep="\t", index=False)
    (folder / "settings.json").write_text(json.dumps(settings, indent=1, default=str) + "\n")
    (folder / "result.json").write_text(json.dumps(
        {"leader": leader, "n_segments": len(measured), "minutes": minutes}, indent=1,
        default=str) + "\n")
    (folder / "summary.md").write_text(_markdown(result), encoding="utf-8")
    case.record("ran the interictal analysis",
                f"{len(measured)} segment(s), {minutes:.0f} min; "
                f"busiest {leader.get('leader', '–')}; results in {folder.name}")
    return result


def _markdown(result: InterictalResult) -> str:
    lines = [f"# Interictal analysis — {result.settings['when']}", "", result.statement(), "",
             "| rank | channel | events | minutes | rate /min | interval | tied | "
             "segments tied |", "|---:|---|---:|---:|---:|---|---|---:|"]
    for r in result.table.itertuples():
        rank = "" if pd.isna(r.rank) else f"{int(r.rank)}"
        rate = "" if pd.isna(r.rate_per_min) else f"{r.rate_per_min:.2f}"
        lines.append(f"| {rank} | {r.channel} | {r.n_events} | {r.minutes:.1f} | {rate} | "
                     f"{r.rate_ci_low:.2f}–{r.rate_ci_high:.2f} | {'yes' if r.tied else ''} | "
                     f"{r.segments_tied} of {r.segments_analysed} |")
    return "\n".join(lines) + "\n"


def latest_result(case) -> InterictalResult | None:
    """The case's most recent interictal run, read back from its folder."""
    folder = Path(case.derivatives) / "interictal"
    runs = sorted(folder.glob("run-*")) if folder.exists() else []
    for run in reversed(runs):
        try:
            table = pd.read_csv(run / "pooled.tsv", sep="\t")
            per_segment = pd.read_csv(run / "per_segment.tsv", sep="\t",
                                      dtype={"segment": str, "run": str})
            meta = json.loads((run / "result.json").read_text())
            settings = json.loads((run / "settings.json").read_text())
        except (OSError, ValueError):
            continue
        table["tied"] = table["tied"].astype(bool)
        return InterictalResult(table, meta.get("leader", {}), per_segment, run,
                                int(meta.get("n_segments", 0)),
                                float(meta.get("minutes", 0.0)), settings)
    return None


def draw_pooled(figure, axis, result: InterictalResult, top: int = 20) -> dict:
    """The busiest channels, pooled: each one's rate with its interval, filled
    when tied with the busiest, and its rate in each segment as small dots --
    so a channel busy in every segment looks different from one busy in one."""
    import numpy as np

    from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

    _quiet(axis)
    axis.yaxis.grid(False)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    table = result.table.dropna(subset=["rate_per_min"]).head(top).reset_index(drop=True)
    y = np.arange(len(table))[::-1]
    per = result.per_segment
    points = {}
    for yi, row in zip(y, table.itertuples(), strict=True):
        axis.plot([row.rate_ci_low, row.rate_ci_high], [yi, yi], color=MUTED, linewidth=1.2)
        if len(per):
            mine = per[per["channel"] == row.channel]
            rates = mine["n_events"] / (mine["seconds"].clip(lower=1e-9) / 60.0)
            axis.scatter(rates, np.full(len(rates), yi + 0.28), s=9, color=MUTED, alpha=0.6,
                         linewidths=0)
        (dot,) = axis.plot([row.rate_per_min], [yi], marker="o", markersize=7,
                           markerfacecolor=SERIES[0] if row.tied else "white",
                           markeredgecolor=SERIES[0], markeredgewidth=1.5, linestyle="none")
        points[row.channel] = dot
    axis.set_yticks(y)
    axis.set_yticklabels(list(table["channel"]), fontsize=8)
    axis.set_ylim(-0.7, len(table) - 0.2)
    axis.set_xlim(left=0)
    axis.set_xlabel("events/min, pooled · line: 95% interval · dots: each segment",
                    fontsize=8, color=MUTED)
    figure.tight_layout()
    return points
