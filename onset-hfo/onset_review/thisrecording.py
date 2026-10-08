"""The open recording, read against the study: Qt-free.

The study pages are the published record and stay as published. What this
module adds is a second layer, always headed *This recording* and always
said to be one window of one patient: where the window that is open sits
against the study, measured the way the study measured.

* **Detectors.** When the window carries expert markings (the archive's 20
  patients do), this window's detections are scored against them exactly as
  the cohort was -- `evaluate_detections` and the same rank agreement, on
  the reviewed channels only -- and drawn as a point on the sweep. For a
  recording without markings there is nothing to score against, so the page
  says that and shows how far the detectors that ran agree with each other.
* **Outcome.** A cohort patient's own row is found and marked. For anyone
  else the page says the plain thing: the study's result describes 20
  patients and cannot be applied to one; nothing here predicts.
* **Patients.** The patient picker opens on this subject.
* **Data.** The open file's own facts beside the two archives.
* **Architecture.** The steps this analysis actually ran, with the settings
  it ran them with.

Nothing here changes a published number, and nothing here is a score of the
method: a 20-patient mean and one window are different quantities, and
every section says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["COHORT_DATASET", "in_cohort", "has_markings", "threshold_for", "score_window",
           "detector_agreement", "facts", "components", "marks", "section", "CAVEAT"]

COHORT_DATASET = "ds003498"

CAVEAT = ("One window of one patient. The study's numbers are means over 20 patients' "
          "first 60 s; this is where the open window sits, measured the same way, not a "
          "score of the method.")


def in_cohort(session) -> bool:
    """Whether the open recording is one of the study's patients."""
    if session is None:
        return False
    request = session.request
    dataset = getattr(getattr(session, "recording", None), "dataset_id", None) or request.dataset
    return request.path is None and dataset == COHORT_DATASET


def _truth(session) -> pd.DataFrame | None:
    frame = getattr(getattr(session, "recording", None), "ground_truth", None)
    if frame is None or not len(frame):
        return None
    return frame


def has_markings(session) -> bool:
    return session is not None and _truth(session) is not None \
        and len(_reviewed(session)) > 0


def _reviewed(session) -> list[str]:
    names = set(session.raw.ch_names) if session.raw is not None else set()
    return [c for c in session.reviewed_channels if c in names]


def _hfo_detectors(session) -> list[str]:
    from onset_hfo.detectors import HFO_DETECTORS

    found = []
    for name in session.request.detectors:
        if name in HFO_DETECTORS and name not in found:
            found.append(name)
    return found


def threshold_for(session, detector: str) -> float:
    """The threshold this window's `detector` ran at, in robust SD."""
    cfg = session.request.pipeline_config()
    return float(getattr(cfg, detector).threshold_sd)


def score_window(session) -> pd.DataFrame | None:
    """This window scored against its expert markings, one row per detector:
    precision, recall, F1, channel-rank ρ, top-5 shared and detections per
    60 s, on the reviewed channels -- the columns of the study's sweep. None
    when the window has no markings."""
    if not has_markings(session):
        return None
    from onset_hfo.benchmark import _rank_agreement
    from onset_hfo.evaluate import evaluate_detections

    truth = _truth(session)
    reviewed = _reviewed(session)
    band = session.request.band
    seconds = float(session.span[1] - session.span[0]) if session.span else \
        float(session.raw.times[-1])
    expert = truth[truth["kind"] == band].groupby("channel").size() \
        .reindex(reviewed, fill_value=0)
    rows = []
    for detector in _hfo_detectors(session):
        events = [e for e in session.events if e.detector == detector]
        score = evaluate_detections(events, truth, kind=band, channels=reviewed)
        ours = pd.Series({c: sum(1 for e in events if e.accepted and e.channel == c)
                          for c in reviewed})
        ranks = _rank_agreement(expert, ours, top_k=5)
        detections = float(ours.sum()) / max(seconds, 1e-9) * 60.0
        rows.append({"detector": detector, "band": band,
                     "threshold_sd": threshold_for(session, detector),
                     "precision": float(score.precision), "recall": float(score.recall),
                     "f1": float(score.f1), "rank_rho": float(ranks["spearman_rho"]),
                     "top5_overlap": float(ranks["top5_overlap"]),
                     "detections": detections, "n_reviewed": len(reviewed),
                     "n_expert": int(expert.sum()), "seconds": seconds})
    return pd.DataFrame(rows) if rows else None


def detector_agreement(session) -> pd.DataFrame | None:
    """For a recording without markings: each pair of the detectors that ran,
    how many events they share and how alike their channel rankings are."""
    from onset_hfo.benchmark import _rank_agreement
    from onset_hfo.metrics import match_events

    names = _hfo_detectors(session) if session is not None else []
    if len(names) < 2:
        return None
    channels = list(session.raw.ch_names)
    by = {n: [e for e in session.events if e.detector == n and e.accepted] for n in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            matched = match_events(by[a], by[b])
            shared = len(matched[0]) if isinstance(matched, tuple) else len(matched)
            counts = {n: pd.Series({c: sum(1 for e in by[n] if e.channel == c)
                                    for c in channels}) for n in (a, b)}
            ranks = _rank_agreement(counts[a], counts[b], top_k=5)
            rows.append({"pair": f"{a} · {b}", "events_a": len(by[a]), "events_b": len(by[b]),
                         "shared": shared, "rank_rho": ranks["spearman_rho"],
                         "top5_overlap": ranks["top5_overlap"]})
    return pd.DataFrame(rows)


def marks(session, metric: str, band: str) -> list[dict]:
    """Points to draw on the sweep chart: this window, one per detector, at
    the threshold it ran at -- only when the chart's band is the window's."""
    scores = score_window(session)
    if scores is None or band != session.request.band or metric not in scores:
        return []
    return [{"detector": r.detector, "threshold_sd": float(r.threshold_sd),
             "value": float(getattr(r, metric)), "label": "this window"}
            for r in scores.itertuples() if np.isfinite(getattr(r, metric))]


def facts(session) -> list[tuple[str, str]]:
    """The open recording in the Data page's terms."""
    request, raw = session.request, session.raw
    dataset = getattr(session.recording, "dataset_id", None) or request.dataset
    source = (f"your file: {request.path.name}" if request.path is not None
              else f"the {dataset} archive" if dataset == COHORT_DATASET
              else f"{dataset} data")
    bad = getattr(session.recording, "bad_channels", None) or []
    out = [("Source", source), ("Subject", request.subject),
           ("Window", f"{request.t_start:g}–{request.t_stop:g} s"),
           ("Channels analysed", f"{len(raw.ch_names)} ({session.montage})"),
           ("Sampling rate", f"{float(session.sfreq):g} Hz"),
           ("Band", request.band_label()),
           ("Detectors", ", ".join(request.detectors)),
           ("Channels the file marks bad", str(len(bad))),
           ("Expert markings", f"{len(_truth(session))} events on "
                               f"{len(_reviewed(session))} reviewed channels"
            if has_markings(session) else "none in this window")]
    if session.quality is not None and len(session.quality):
        verdicts = session.quality.get("verdict")
        if verdicts is not None:
            out.append(("Quality verdicts", ", ".join(
                f"{n} {v}" for v, n in verdicts.value_counts().items())))
    return out


def components(session) -> list[str]:
    """What this analysis ran, in order, as the report prints it."""
    return [str(step) for step in session.steps]


#: How a column is printed when it is not a two-decimal measure.
FORMATS = {"threshold_sd": "{:g}", "top5_overlap": "{:.0f}", "detections": "{:.0f}",
           "events_a": "{:.0f}", "events_b": "{:.0f}", "shared": "{:.0f}",
           "rz_coverage": "{:.0%}"}


def _table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 2) -> str:
    def cell(key, value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "—"
            return FORMATS.get(key, f"{{:.{digits}f}}").format(float(value))
        return str(value)

    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(cell(key, row.get(key, "")) for key, _ in columns) + " |"
            for row in rows]
    return "\n".join([head, rule, *body]) + "\n"


def section(key: str, session, cohort: pd.DataFrame | None = None) -> str:
    """The *This recording* section of study page `key`, as Markdown, or ""
    when there is no recording open."""
    if session is None:
        return ""
    from onset_review.studies import note

    label = session.request.label()
    out = [f"## This recording — {label}\n"]
    if key == "detectors":
        scores = score_window(session)
        if scores is not None:
            out.append(note("info", CAVEAT + " The point marked *this window* on the "
                                             "chart is this table's value."))
            out.append(_table(scores.to_dict("records"), [
                ("detector", "detector"), ("threshold_sd", "threshold (SD)"),
                ("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
                ("rank_rho", "channel-rank ρ"), ("top5_overlap", "top-5 shared"),
                ("detections", "detections / 60 s")]))
            first = scores.iloc[0]
            out.append(f"*{int(first.n_expert)} expert-marked {first.band.replace('_', ' ')}s "
                       f"on {int(first.n_reviewed)} reviewed channels in "
                       f"{first.seconds:g} s, scored with this window's own preprocessing "
                       f"and quality settings.*\n")
        else:
            pairs = detector_agreement(session)
            out.append(note("info", "This recording carries no expert markings, so there "
                                    "is nothing to score its detections against; "
                                    "precision, recall and channel-rank ρ are not "
                                    "defined for it."))
            if pairs is not None:
                out.append("How far the detectors that ran agree with each other — "
                           "agreement, not accuracy:\n")
                out.append(_table(pairs.to_dict("records"), [
                    ("pair", "detectors"), ("events_a", "events, first"),
                    ("events_b", "events, second"), ("shared", "shared"),
                    ("rank_rho", "channel-rank ρ"), ("top5_overlap", "top-5 shared")]))
            else:
                out.append("Only one HFO detector ran; run two from the Recording page's "
                           "settings to see how far they agree.\n")
    elif key == "outcome":
        row = _cohort_row(session, cohort)
        if row is not None:
            outcome = "seizure-free" if bool(row.get("seizure_free")) else "recurrence"
            out.append(note("info", f"{session.request.subject} is one of the study's "
                                    "patients: their dot is ringed on the chart. One dot "
                                    "among twenty; the study's result is the groups', "
                                    "not this patient's."))
            out.append(_table([{**row, "outcome": outcome}], [
                ("subject", "patient"), ("outcome", "outcome"),
                ("candidates_resected_expert", "tied set resected, expert"),
                ("candidates_resected_rms", "tied set resected, our detector"),
                ("rz_coverage", "resection recorded")]))
        else:
            out.append(note("warn", "This patient is not one of the study's 20. The "
                                    "study's result describes those groups and cannot be "
                                    "applied to one patient: nothing on this page, or "
                                    "anywhere in this software, predicts an outcome. "
                                    "Your own patients can be studied the same way under "
                                    "*Your cohort* below, labelled as yours."))
    elif key == "patients":
        if in_cohort(session):
            out.append(f"The picker above is on **{session.request.subject}**, the patient "
                       "open now.\n")
        else:
            out.append("The open recording is not one of the study's patients, so it has "
                       "no row here.\n")
    elif key == "data":
        out.append(_table([{"what": k, "value": v} for k, v in facts(session)],
                          [("what", "what"), ("value", "this recording")]))
    elif key == "architecture":
        steps = components(session)
        out.append("What this analysis actually ran, in order:\n")
        out.extend(f"{i}. {step}" for i, step in enumerate(steps, 1))
        out.append("")
        request = session.request
        out.append(f"*Detectors {', '.join(request.detectors)} in the "
                   f"{request.band_label()} band, at "
                   + ", ".join(f"{d} {threshold_for(session, d):g} SD"
                               for d in _hfo_detectors(session))
                   + f"; quality stage {'on' if request.check_quality else 'off'}.*\n")
    else:
        return ""
    return "\n".join(out) + "\n"


def _cohort_row(session, cohort: pd.DataFrame | None) -> dict | None:
    if not in_cohort(session) or cohort is None or not len(cohort):
        return None
    rows = cohort[cohort["subject"] == session.request.subject]
    return None if not len(rows) else rows.iloc[0].to_dict()
