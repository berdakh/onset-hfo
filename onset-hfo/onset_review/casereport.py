"""A case's report: what was done, what it found, who signed it, versioned.

The report is the one thing from a case that leaves the department, so it is
built under four rules:

1. **Nothing is computed here.** Every number is read from the steps' own
   result folders (`caseinterictal`, `caseictal`, `casemap`); the report
   cannot disagree with the screen.
2. **It is not produced before the work is.** `readiness` lists what must be
   true first: recordings converted, channels checked, an analysis run, its
   detections reviewed, a de-identification check passed, and an intact
   audit log (`Case.verify_log`).
3. **A sign-off is of content, not of a file.** `sign_off` records who, in
   what role, saying what, against a **fingerprint** of every result,
   setting, contact position and zone the report draws on. Change any of
   them and the sign-off is shown as *superseded*; the report says "not
   signed off for this content" until someone signs again.
4. **Versions are kept.** Each report is a new numbered folder
   (``derivatives/onset/reports/report-vN/``: HTML, PDF, figures and a
   manifest with every file's checksum and the fingerprint); none is ever
   overwritten, and producing one is logged.

The HTML is built here, without Qt; `html_to_pdf` turns it into the PDF
with Qt's own PDF writer.
"""

from __future__ import annotations

import hashlib
import html
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["fingerprint", "readiness", "sign_off", "signoff_status", "produce", "html_to_pdf",
           "report_versions", "ReportResult", "VALIDATION", "SIGNOFF_STATEMENT"]

#: Figure width in the PDF's own pixels: the A4 text width at the writer's
#: 110 dpi, less a margin.
FIGURE_WIDTH = 560
SIGNOFF_STATEMENT = ("I have reviewed the analyses, results and contact positions of this "
                     "case as shown, and they may be used in its discussion.")
#: What each method was measured to do, from the archive studies; a test holds
#: these to the committed tables.
VALIDATION = {
    "ictal": ("Epileptogenicity Index against the clinicians' onset zone, ds003029, 28 "
              "patients: median per-patient AUC 0.80 (95% interval 0.70–0.87); 0.52 on the "
              "same seizures 40 s before onset; top channel in the zone in 15 of 28 "
              "(docs/ICTAL.md)."),
    "template": ("Straight-line plans against 38 patients' real implants on the template "
                 "(ds004100): median 2.5 mm off with each shaft's own spacing, 10.1 mm at a "
                 "3.5 mm pitch; at real positions the atlas names a structure for 51% of "
                 "contacts (docs/TEMPLATE_MAP.md)."),
    "interictal": ("Interictal HFO ranking against surgical outcome, ds003498, 20 patients: "
                   "no comparison reached significance (docs/OUTCOME.md)."),
}


# -- what the report rests on ------------------------------------------------------------------
def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs(case) -> dict:
    """Everything the report draws on, as name -> checksum or value."""
    from onset_hfo.case.electrodes import electrodes_path
    from onset_review.caseictal import latest_ictal
    from onset_review.caseinterictal import latest_result

    items: dict[str, str] = {}
    for label, result in (("interictal", latest_result(case)), ("ictal", latest_ictal(case))):
        if result is not None:
            for path in sorted(Path(result.folder).glob("*")):
                if path.is_file() and path.suffix in (".tsv", ".json"):
                    items[f"{label}/{path.name}"] = _sha(path)
    for path in (electrodes_path(case), Path(case.derivatives) / "segments.tsv",
                 Path(case.derivatives) / "segments.json"):
        if path.exists():
            items[path.name] = _sha(path)
    items["zones"] = json.dumps({k: v.get("contacts", []) for k, v in case.zones.items()},
                                sort_keys=True)
    items["analysis"] = json.dumps(case.analysis, sort_keys=True, default=str)
    items["recordings"] = json.dumps([(r.session, r.run, r.source_sha256)
                                      for r in case.recordings])
    return items


def fingerprint(case) -> str:
    """One checksum of everything the report rests on."""
    blob = json.dumps(_inputs(case), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def readiness(case) -> list[tuple[str, bool, str]]:
    """(what, met, why) for each thing that must hold before a report."""
    from onset_hfo.case.deid import check_case
    from onset_review.caseictal import latest_ictal
    from onset_review.caseinterictal import latest_result

    inter, ictal = latest_result(case), latest_ictal(case)
    found = check_case(case)
    checked = any(e.get("action") == "checked de-identification" for e in case.log)
    log = case.verify_log()
    def done(step: str, todo: str) -> str:
        info = case.steps.get(step, {})
        return f"by {info.get('by', '')} at {info.get('at', '')}" if case.step_done(step) \
            else todo

    rows = [
        ("Recordings converted", bool(case.recordings),
         f"{len(case.recordings)} recording(s)"),
        ("Channels checked", case.step_done("channels"),
         done("channels", "tick Channels checked")),
        ("An analysis run", inter is not None or ictal is not None,
         ", ".join(n for n, r in (("interictal", inter), ("ictal onset", ictal)) if r)
         or "run the Interictal or Ictal onset step"),
    ]
    if inter is not None:
        rows.append(("Detections reviewed", case.step_done("review"),
                     done("review", "tick Review done after reviewing the segments")))
    rows += [
        ("De-identification checked", checked and not found,
         (f"{len(found)} thing(s) look identifying" if found else
          "passed" if checked else "run the check, with the patient's names")),
        ("Audit log intact", log["ok"], log["summary"]),
    ]
    return rows


def sign_off(case, by: str, role: str, statement: str = SIGNOFF_STATEMENT) -> dict:
    if not by.strip():
        raise ValueError("a sign-off needs a name")
    now = fingerprint(case)
    if any(s.get("by") == by.strip() and s.get("fingerprint") == now for s in case.signoffs):
        raise ValueError(f"{by.strip()} has signed off this content already")
    entry = {"by": by.strip(), "role": role.strip(), "statement": statement.strip(),
             "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "fingerprint": now}
    case.signoffs.append(entry)
    case.record("signed off", f"{entry['by']} ({entry['role'] or 'no role given'}) for "
                f"content {entry['fingerprint'][:12]}", by)
    return entry


def signoff_status(case) -> list[dict]:
    now = fingerprint(case)
    return [{**s, "current": s.get("fingerprint") == now} for s in case.signoffs]


# -- the report itself --------------------------------------------------------------------------
@dataclass
class ReportResult:
    version: int
    folder: Path
    html: Path
    pdf: Path | None
    fingerprint: str
    signed: bool


def report_versions(case) -> list[dict]:
    folder = Path(case.derivatives) / "reports"
    out = []
    for path in sorted(folder.glob("report-v*"), key=lambda p: int(p.name.split("-v")[-1])) \
            if folder.exists() else []:
        try:
            meta = json.loads((path / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        out.append({**meta, "folder": path})
    return out


def _esc(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    return html.escape(str(value))


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{_esc(title)}</th>" for _key, title in columns)
    body = "".join("<tr>" + "".join(f"<td>{_esc(r.get(k))}</td>" for k, _t in columns) +
                   "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _figure(path: Path, draw) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.0, 3.6), dpi=130)
    FigureCanvasAgg(figure)
    draw(figure)
    figure.savefig(path, bbox_inches="tight")
    return f'<p><img src="{path.name}" width="{FIGURE_WIDTH}"></p>'


def _analysis_text(analysis: dict) -> str:
    parts = [f"detectors {', '.join(str(d).replace('_', ' ') for d in analysis.get('detectors') or [])}"
             f" (the first ranks)", f"{str(analysis.get('band') or 'ripple').replace('_', ' ')} band",
             f"threshold {analysis['threshold_sd']:g} SD" if analysis.get("threshold_sd")
             else "each detector's default threshold"]
    if analysis.get("check_quality"):
        parts.append("bad contacts and seconds set aside")
    if analysis.get("with_spikes"):
        parts.append("events on spikes noted")
    preprocess = analysis.get("preprocess")
    parts.append(f"reference {preprocess.get('reference', 'bipolar')}" if isinstance(
        preprocess, dict) and preprocess.get("reference") else "the default bipolar montage")
    return "; ".join(parts)


def _fmt(value, spec="{:.2f}"):
    return "–" if value is None or pd.isna(value) else spec.format(value)


def build_html(case, version: int, folder: Path, by: str) -> str:
    from onset_hfo.case.atlas import Atlas
    from onset_hfo.case.electrodes import TEMPLATE_NOTE, load_electrodes
    from onset_hfo.case.segments import load_segments
    from onset_review.caseictal import draw_ictal, latest_ictal
    from onset_review.caseinterictal import draw_pooled, latest_result
    from onset_review.casemap import agreement, combined_table, draw_combined, statement
    from onset_review.report import DISCLAIMER

    print_ = fingerprint(case)
    signoffs = [s for s in signoff_status(case) if s["current"]]
    inter, ictal = latest_result(case), latest_ictal(case)
    parts = [f"<h1>Case {_esc(case.case_id)} — report v{version}</h1>",
             f"<p class='small'>Made {time.strftime('%Y-%m-%d %H:%M')} by {_esc(by)} · content "
             f"{print_[:16]} · {len(case.recordings)} recording(s), "
             f"{sum(r.duration_s for r in case.recordings) / 3600:.1f} h</p>",
             "<p class='warn'>" + _esc(DISCLAIMER.replace("**", "").replace("`", "")) +
             "</p>"]
    if signoffs:
        parts.append("<p><b>Signed off for this content</b> by " + "; ".join(
            f"{_esc(s['by'])} ({_esc(s['role'])}) at {_esc(s['at'])}" for s in signoffs) +
            ".</p>")
    else:
        parts.append("<p class='warn'><b>Not signed off for this content.</b> A draft.</p>")

    # Summary
    parts.append("<h2>Summary</h2>")
    table = combined_table(case)
    if len(table):
        found = agreement(table, ictal.consistent() if ictal else [])
        parts.append(f"<p>{_esc(statement(table, found))}</p>")
    if inter is not None:
        parts.append(f"<p><b>Interictal.</b> {_esc(inter.statement())}</p>")
    if ictal is not None:
        parts.append(f"<p><b>Ictal onset.</b> {_esc(ictal.statement())}</p>")
    zone = case.zone("soz")
    parts.append("<p><b>Onset zone as marked by the clinician:</b> " +
                 (_esc(", ".join(zone)) + f" (by {_esc(case.zones['soz'].get('by'))})"
                  if zone else "none marked") + ".</p>")

    # Recordings and steps
    parts.append("<h2>Recordings</h2>")
    parts.append(_table([{"run": r.run, "session": r.session, "task": r.task,
                          "format": r.source_format, "sfreq": f"{r.sfreq:g} Hz",
                          "channels": r.n_channels, "duration": f"{r.duration_s / 60:.1f} min",
                          "sha": r.source_sha256[:12]} for r in case.recordings],
                        [("run", "Run"), ("session", "Session"), ("task", "Task"),
                         ("format", "Source format"), ("sfreq", "Sampling"),
                         ("channels", "Channels"), ("duration", "Length"),
                         ("sha", "Source SHA-256")]))
    parts.append("<h2>Steps</h2>")
    from onset_hfo.case.model import STEPS

    parts.append(_table([{"step": title, "done": "yes" if case.step_done(key) else "",
                          "by": case.steps.get(key, {}).get("by", ""),
                          "at": case.steps.get(key, {}).get("at", "")}
                         for key, title, _p in STEPS],
                        [("step", "Step"), ("done", "Done"), ("by", "By"), ("at", "At")]))

    # Interictal
    if inter is not None:
        frame, rule, _info = load_segments(case)
        parts.append("<h2>Interictal</h2>")
        if rule is not None:
            parts.append(f"<p class='small'>Segments: {_esc(rule.describe())}; "
                         f"{len(frame)} chosen. Analysis: "
                         f"{_esc(_analysis_text(inter.settings.get('analysis', {})))}.</p>")
        parts.append(_figure(folder / "interictal.png",
                             lambda f: draw_pooled(f, f.add_subplot(111), inter, top=15)))
        rows = [{"rank": "" if pd.isna(r.rank) else int(r.rank), "channel": r.channel,
                 "events": int(r.n_events), "rate": _fmt(r.rate_per_min),
                 "ci": f"{r.rate_ci_low:.2f}–{r.rate_ci_high:.2f}",
                 "tied": "yes" if r.tied else "",
                 "seg": f"{int(r.segments_tied)} of {int(r.segments_analysed)}"}
                for r in inter.table.head(20).itertuples()]
        parts.append(_table(rows, [("rank", "Rank"), ("channel", "Channel"),
                                   ("events", "Events"), ("rate", "Rate /min"),
                                   ("ci", "95% interval"), ("tied", "Tied"),
                                   ("seg", "Segments tied")]))

    # Ictal
    if ictal is not None:
        parts.append("<h2>Ictal onset</h2>")
        parts.append(f"<p class='small'>{_esc(ictal.settings.get('method', ''))}.</p>")
        if len(ictal.combined):
            parts.append(_figure(folder / "ictal.png",
                                 lambda f: draw_ictal(f, f.add_subplot(111), ictal, top=15)))
        rows = [{"rank": int(r.rank), "channel": r.channel, "ei": _fmt(r.median_ei),
                 "high": f"{int(r.seizures_high)} of {int(r.seizures)}",
                 "change": _fmt(r.median_change_s, "{:+.1f}")}
                for r in ictal.combined.head(20).itertuples()]
        parts.append(_table(rows, [("rank", "Rank"), ("channel", "Channel"),
                                   ("ei", "Median index"), ("high", "Seizures ≥ 0.3"),
                                   ("change", "Median change (s)")]))
        parts.append(_table(ictal.seizures.to_dict("records"),
                            [("seizure", "Seizure"), ("run", "Run"), ("onset", "Onset (s)"),
                             ("marker", "Marked as"), ("status", "Status")]))

    # Map
    electrodes = load_electrodes(case)
    if len(table):
        parts.append("<h2>Map</h2>")
        parts.append(f"<p class='warn'>{_esc(TEMPLATE_NOTE)} {len(electrodes)} contact(s) "
                     f"placed: " + _esc(", ".join(f"{n} {s}" for s, n in
                                                  electrodes["source"].value_counts().items()))
                     + ".</p>")
        atlas = Atlas.load() if Atlas.available() else None
        parts.append(_figure(folder / "map.png", lambda f: draw_combined(f, table, atlas)))
        rows = [{"channel": r.channel, "rate": _fmt(r.rate_per_min), "tied": "yes" if r.tied
                 else "", "ei": _fmt(r.median_ei),
                 "where": r.where or ("not placed" if not r.placed else ""),
                 "soz": "yes" if r.soz else ""} for r in table.head(25).itertuples()]
        parts.append(_table(rows, [("channel", "Channel"), ("rate", "Rate /min"),
                                   ("tied", "Tied"), ("ei", "Ictal index"),
                                   ("where", "Probably (template)"), ("soz", "Onset zone")]))

    # Methods, validation, checks
    parts.append("<h2>What the methods were measured to do</h2><ul>" +
                 "".join(f"<li>{_esc(v)}</li>" for v in VALIDATION.values()) + "</ul>")
    parts.append("<h2>Checks</h2>")
    parts.append(_table([{"what": w, "ok": "yes" if ok else "NO", "detail": d}
                         for w, ok, d in readiness(case)],
                        [("what", "Check"), ("ok", "Met"), ("detail", "Detail")]))
    parts.append("<h2>Sign-offs</h2>")
    parts.append(_table([{**s, "current": "yes" if s["current"] else "superseded"}
                         for s in signoff_status(case)],
                        [("by", "By"), ("role", "Role"), ("at", "At"),
                         ("statement", "Statement"), ("current", "For this content")])
                 if case.signoffs else "<p>None.</p>")
    parts.append(f"<p class='small'>Content fingerprint {print_}. Every result, setting, "
                 "contact position and zone this report draws on is in it; a sign-off is "
                 "for one fingerprint.</p>")
    style = ("body{font-family:sans-serif;font-size:9pt;} h1{font-size:15pt;} "
             "h2{font-size:11.5pt;margin-top:14pt;} table{border-collapse:collapse;} "
             "td,th{border:1px solid gray;padding:2px 5px;font-size:8pt;} "
             ".small{color:dimgray;font-size:8pt;} .warn{color:firebrick;}")
    return (f"<html><head><meta charset='utf-8'><title>Case {_esc(case.case_id)} report "
            f"v{version}</title><style>{style}</style></head><body>" + "\n".join(parts) +
            "</body></html>")


def produce(case, by: str, render_pdf=None) -> ReportResult:
    """Write the next report version. `render_pdf(html, folder, path)` makes
    the PDF (the window passes `html_to_pdf`); without one, HTML only."""
    blockers = [what for what, ok, _d in readiness(case) if not ok]
    if blockers:
        raise ValueError("not ready for a report: " + "; ".join(blockers))
    root = Path(case.derivatives) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name.split("-v")[-1]) for p in root.glob("report-v*")
                if p.name.split("-v")[-1].isdigit()]
    version = max(versions, default=0) + 1
    folder = root / f"report-v{version}"
    folder.mkdir()
    text = build_html(case, version, folder, by)
    html_path = folder / "report.html"
    html_path.write_text(text, encoding="utf-8")
    pdf_path = None
    if render_pdf is not None:
        pdf_path = folder / f"case-{case.case_id}-report-v{version}.pdf"
        render_pdf(text, folder, pdf_path)
    print_ = fingerprint(case)
    signed = any(s["current"] for s in signoff_status(case))
    manifest = {"version": version, "made_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "made_by": by,
                "fingerprint": print_, "signed": signed,
                "signoffs": [s for s in signoff_status(case) if s["current"]],
                "files": {p.name: _sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    case.record("produced a report", f"v{version}, content {print_[:12]}, "
                f"{'signed off' if signed else 'not signed off'}", by)
    if signed:
        case.mark_step("report", True, note=f"v{version}", by=by)
    return ReportResult(version, folder, html_path, pdf_path, print_, signed)


def html_to_pdf(text: str, folder: Path, path: Path) -> Path:
    """The report's HTML as an A4 PDF, with Qt's PDF writer (needs a running
    QApplication)."""
    from qtpy.QtCore import QMarginsF, QUrl
    from qtpy.QtGui import QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument

    document = QTextDocument()
    # The figures are handed to the document by name: a relative <img> is not
    # resolved against a base URL when the document prints.
    for image in sorted(Path(folder).glob("*.png")):
        document.addResource(QTextDocument.ImageResource, QUrl(image.name),
                             QImage(str(image)))
    document.setHtml(text)
    writer = QPdfWriter(str(path))
    writer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                     QMarginsF(15, 15, 15, 15), QPageLayout.Millimeter))
    writer.setResolution(110)
    writer.setTitle(Path(path).stem)
    writer.setCreator("onset-review")
    document.print_(writer)
    return Path(path)
