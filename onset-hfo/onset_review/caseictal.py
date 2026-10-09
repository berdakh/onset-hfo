"""A case's ictal onset: the Epileptogenicity Index of every marked seizure.

The seizures are the case's own marks: an electrographic *seizure onset*, or
the start of a *seizure* span (which runs from electrographic onset to
offset). A clinical onset alone is listed and left out: the index dates the
change against the electrographic onset, and the first clinical sign usually
comes seconds after it, inside the seizure.

Each seizure's stretch -- `WINDOW_BEFORE_S` before the onset to
`WINDOW_AFTER_S` after, the same as the archive study
(`onset_hfo.ictal_study`) -- is read from the case's recording and
preprocessed as the case's analysis says (its montage, its bad contacts left
out: `onset_review.caseinterictal.segment_request`), then the index computed
(`onset_hfo.ictal.epileptogenicity`) and the seizures combined
(`onset_hfo.ictal.combine`). Each run writes a dated folder
(``derivatives/onset/ictal/run-*``): the combined table, every seizure's
table, which seizures were analysed and why not, the settings and a summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["seizures_of", "run_ictal", "IctalResult", "latest_ictal", "draw_ictal",
           "SEIZURE_COLUMNS"]

SEIZURE_COLUMNS = ("seizure", "run", "onset", "marker", "usable", "note")
#: An onset mark and a seizure span starting this close are one seizure.
SAME_SEIZURE_S = 10.0


def seizures_of(case) -> pd.DataFrame:
    """Every seizure the case's marks name, in time order per recording,
    with whether it can be analysed and, if not, why."""
    from onset_hfo.ictal_study import WINDOW_BEFORE_S

    rows = []
    for rec in case.recordings:
        marks = case.marks(rec)
        if marks is None or not len(marks) or "trial_type" not in marks:
            continue
        kinds = marks["trial_type"].astype(str)
        onsets = sorted(float(t) for t in marks.loc[kinds == "seizure-onset", "onset"])
        starts = sorted(float(t) for t in marks.loc[kinds == "seizure", "onset"])
        clinical = sorted(float(t) for t in
                          marks.loc[kinds == "seizure-clinical-onset", "onset"])
        found = [(t, "seizure onset") for t in onsets]
        found += [(t, "start of a seizure mark") for t in starts
                  if all(abs(t - o) > SAME_SEIZURE_S for o in onsets)]
        electro = [t for t, _ in found]
        # A clinical onset follows the electrographic one by seconds to a minute.
        found += [(t, "clinical onset only") for t in clinical
                  if all(not (0 <= t - e <= 60.0) for e in electro)]
        for onset, marker in sorted(found):
            usable, note = True, ""
            if marker == "clinical onset only":
                usable = False
                note = ("only the clinical onset is marked: mark the electrographic onset "
                        "on the Annotate step")
            elif onset < WINDOW_BEFORE_S - 5.0:
                usable = False
                note = f"less than {WINDOW_BEFORE_S - 5.0:g} s of recording before it"
            elif onset > float(rec.duration_s) - 10.0:
                usable = False
                note = "too near the end of the recording"
            rows.append({"run": str(rec.run), "onset": round(onset, 3), "marker": marker,
                         "usable": usable, "note": note})
    frame = pd.DataFrame(rows, columns=[c for c in SEIZURE_COLUMNS if c != "seizure"])
    frame.insert(0, "seizure", [f"sz-{i + 1:02d}" for i in range(len(frame))])
    return frame


@dataclass
class IctalResult:
    combined: pd.DataFrame
    per_seizure: pd.DataFrame
    seizures: pd.DataFrame
    folder: Path
    settings: dict

    @property
    def n_analysed(self) -> int:
        return int((self.seizures["status"] == "ok").sum()) if len(self.seizures) else 0

    def consistent(self) -> list[str]:
        """Channels at or above the cutoff in at least half the seizures."""
        from onset_hfo.ictal import EI_CUTOFF

        table = self.combined
        if table.empty:
            return []
        keep = (table["seizures_high"] * 2 >= table["seizures"]) & \
               (table["median_ei"] >= EI_CUTOFF)
        return list(table.loc[keep, "channel"])

    def statement(self) -> str:
        from onset_hfo.ictal import EI_CUTOFF

        n = self.n_analysed
        if not n or self.combined.empty:
            return "No seizure could be analysed."
        top = self.combined.iloc[0]
        if float(top["median_ei"]) <= 0:
            return (f"Over {n} seizure(s), no channel showed a lasting rise of fast over slow "
                    "activity after the marked onset.")
        steady = self.consistent()
        text = (f"Over {n} seizure(s): {top['channel']} has the highest median index "
                f"({top['median_ei']:.2f}) and reached {EI_CUTOFF:g} in "
                f"{int(top['seizures_high'])} of {int(top['seizures'])}. ")
        if steady:
            text += (f"At or above {EI_CUTOFF:g} in at least half the seizures: "
                     + ", ".join(steady[:12]) + ("…" if len(steady) > 12 else "") + ".")
        else:
            text += f"No channel reached {EI_CUTOFF:g} in at least half the seizures."
        if n == 1:
            text += (" One seizure's onset zone is not necessarily the patient's: "
                     "mark more seizures if there are any.")
        return text


def _check_preprocess(preprocess, settings) -> None:
    if preprocess.lowpass and preprocess.lowpass < settings.fast_band[1]:
        raise ValueError(f"the case's low-pass at {preprocess.lowpass:g} Hz removes part of "
                         f"the fast band ({settings.fast_band[0]:g}–"
                         f"{settings.fast_band[1]:g} Hz); raise it on the Preprocess step")


def run_ictal(case, seizures: pd.DataFrame | None = None, settings=None, template=None,
              progress=None, should_stop=None) -> IctalResult:
    """Analyse each usable seizure, combine them, and write the results."""
    from onset_hfo.ictal import IctalSettings, combine, epileptogenicity
    from onset_hfo.ictal_study import WINDOW_AFTER_S, WINDOW_BEFORE_S
    from onset_hfo.preprocess import prepare
    from onset_review.caseinterictal import segment_request, template_for
    from onset_review.project import request_to_dict
    from onset_review.session import laplacian_positions, source_for

    settings = settings or IctalSettings()
    template = template or template_for(case)
    seizures = seizures_of(case) if seizures is None else seizures
    usable = seizures[seizures["usable"].astype(bool)] if len(seizures) else seizures
    tables: dict[str, pd.DataFrame] = {}
    status = []
    for index, row in enumerate(usable.itertuples()):
        if should_stop is not None and should_stop():
            raise InterruptedError(f"stopped after {index} of {len(usable)} seizures")
        rec = case.recording(str(row.run))
        onset = float(row.onset)
        start = max(0.0, onset - WINDOW_BEFORE_S)
        stop = min(float(rec.duration_s), onset + WINDOW_AFTER_S)
        try:
            request = segment_request(case, str(row.run), start, stop, template)
            preprocess = request.pipeline_config().preprocess
            _check_preprocess(preprocess, settings)
            record = source_for(request)(start, stop)
            positions, positions_from = laplacian_positions(request)
            prep = prepare(record, preprocess, verbose=False, positions=positions,
                           positions_from=positions_from)
            table = epileptogenicity(prep.data, list(prep.ch_names), prep.sfreq,
                                     onset_s=onset, start_s=float(prep.t_offset),
                                     settings=settings)
        except (ValueError, OSError, RuntimeError) as error:
            status.append({"seizure": row.seizure, "status": str(error)[:200]})
            continue
        tables[str(row.seizure)] = table
        top = table.iloc[0]
        status.append({"seizure": row.seizure, "status": "ok", "sfreq": float(prep.sfreq),
                       "n_channels": len(table), "n_detected": int(table["detected"].sum()),
                       "top": str(top["channel"]), "first_change_s":
                       float(np.nanmin(table["change_s"])) if table["detected"].any()
                       else float("nan")})
        if progress is not None:
            progress((index + 1) / max(len(usable), 1))
    states = pd.DataFrame(status, columns=["seizure", "status", "sfreq", "n_channels",
                                           "n_detected", "top", "first_change_s"])
    listed = seizures.merge(states, on="seizure", how="left") if len(seizures) else \
        seizures.assign(status=[])
    listed["status"] = listed["status"].fillna("not analysed: " + listed["note"].astype(str))
    combined = combine(tables)
    per_seizure = pd.concat([t.assign(seizure=k) for k, t in tables.items()],
                            ignore_index=True) if tables else pd.DataFrame()
    folder = Path(case.derivatives) / "ictal" / time.strftime("run-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"when": time.strftime("%Y-%m-%d %H:%M"), "method": settings.describe(),
            "settings": settings.to_json(),
            "window_s": [WINDOW_BEFORE_S, WINDOW_AFTER_S],
            "analysis": {k: v for k, v in request_to_dict(template).items()
                         if k in ("preprocess",)}}
    result = IctalResult(combined, per_seizure, listed, folder, meta)
    combined.to_csv(folder / "combined.tsv", sep="\t", index=False)
    per_seizure.to_csv(folder / "per_seizure.tsv", sep="\t", index=False)
    listed.to_csv(folder / "seizures.tsv", sep="\t", index=False)
    (folder / "settings.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    (folder / "summary.md").write_text(_markdown(result), encoding="utf-8")
    case.record("ran the ictal onset analysis",
                f"{result.n_analysed} of {len(seizures)} seizure(s); highest "
                f"{combined.iloc[0]['channel'] if len(combined) else '–'}; "
                f"results in {folder.name}")
    return result


def _markdown(result: IctalResult) -> str:
    lines = [f"# Ictal onset — {result.settings['when']}", "", result.statement(), "",
             f"Method: {result.settings['method']}.", "",
             "| rank | channel | median index | seizures at or above 0.3 | median change (s) |",
             "|---:|---|---:|---:|---:|"]
    for r in result.combined.itertuples():
        change = "" if pd.isna(r.median_change_s) else f"{r.median_change_s:+.1f}"
        lines.append(f"| {r.rank} | {r.channel} | {r.median_ei:.2f} | "
                     f"{r.seizures_high} of {r.seizures} | {change} |")
    lines += ["", "| seizure | run | onset (s) | marker | status |", "|---|---|---:|---|---|"]
    for r in result.seizures.itertuples():
        lines.append(f"| {r.seizure} | {r.run} | {r.onset:.1f} | {r.marker} | {r.status} |")
    return "\n".join(lines) + "\n"


def latest_ictal(case) -> IctalResult | None:
    """The case's most recent ictal run, read back from its folder."""
    folder = Path(case.derivatives) / "ictal"
    runs = sorted(folder.glob("run-*")) if folder.exists() else []
    for run in reversed(runs):
        try:
            combined = pd.read_csv(run / "combined.tsv", sep="\t")
            seizures = pd.read_csv(run / "seizures.tsv", sep="\t",
                                   dtype={"run": str, "seizure": str})
            try:
                per_seizure = pd.read_csv(run / "per_seizure.tsv", sep="\t",
                                          dtype={"seizure": str})
            except pd.errors.EmptyDataError:
                per_seizure = pd.DataFrame()
            settings = json.loads((run / "settings.json").read_text())
        except (OSError, ValueError):
            continue
        return IctalResult(combined, per_seizure, seizures, run, settings)
    return None


def draw_ictal(figure, axis, result: IctalResult, top: int = 20) -> dict:
    """The highest-index channels: each one's median over the seizures,
    filled when it reached the cutoff in at least half of them, with each
    seizure's index as small dots and the cutoff as a line -- so a channel
    that leads every seizure looks different from one that led once."""
    from onset_hfo.ictal import EI_CUTOFF
    from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

    _quiet(axis)
    axis.yaxis.grid(False)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    table = result.combined.head(top).reset_index(drop=True)
    steady = set(result.consistent())
    y = np.arange(len(table))[::-1]
    per = result.per_seizure
    points = {}
    axis.axvline(EI_CUTOFF, color=MUTED, linewidth=1.0, linestyle="--")
    for yi, row in zip(y, table.itertuples(), strict=True):
        if len(per):
            mine = per.loc[per["channel"] == row.channel, "ei"]
            axis.scatter(mine, np.full(len(mine), yi + 0.28), s=9, color=MUTED, alpha=0.7,
                         linewidths=0)
        (dot,) = axis.plot([row.median_ei], [yi], marker="o", markersize=7,
                           markerfacecolor=SERIES[0] if row.channel in steady else "white",
                           markeredgecolor=SERIES[0], markeredgewidth=1.5, linestyle="none")
        points[row.channel] = dot
    axis.set_yticks(y)
    axis.set_yticklabels(list(table["channel"]), fontsize=8)
    axis.set_ylim(-0.7, max(len(table), 1) - 0.2)
    axis.set_xlim(-0.02, 1.05)
    axis.set_xlabel(f"Epileptogenicity Index, median · dots: each seizure · "
                    f"line: {EI_CUTOFF:g}", fontsize=8, color=MUTED)
    figure.tight_layout()
    return points
