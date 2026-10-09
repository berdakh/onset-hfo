"""Turning a saved analysis into things a page can show.

Every function here is pure, takes a :class:`~onset_hfo.store.ResultStore`,
and returns plain data or a DataFrame. None of them imports Streamlit, which
is what makes them testable and what keeps the UI a *view*: the page arranges
what these return and does not decide anything.

The one rule this module exists to enforce
------------------------------------------

**The interface must not compute a result.** Every number a reader sees has to
be one the pipeline wrote to disk, because the moment a UI starts deriving its
own figures there are two sources of truth and the one on screen is the one
nobody validated.

There is exactly one exception, and it is deliberate:
:func:`leader_note` calls :func:`onset_hfo.metrics.leader_separation` on the
stored rate table. That function is the missing null hypothesis -- it answers
*does any channel actually stand out, or are they all tied?* -- and a page
that showed a ranking without it would be the single most misleading thing
this project could ship. It derives nothing new: it reads the counts and
intervals already in ``rates_*.csv`` through a library function the tests
cover. Treat any second exception as a bug.

Re-rendering a signal window (``app/signal.py``) is not an exception to
this rule either. The event's times, frequency and amplitude all come from the
store; the signal is fetched again only so the reader can *look* at the window
the numbers describe. Nothing is re-detected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from onset_hfo.config import PROJECT_ROOT, RESULTS_DIR
from onset_hfo.store import ResultStore

#: Where the committed tables live. The checkout's ``data/`` unless
#: ``ONSET_HFO_DATA_ROOT`` says otherwise -- which the desktop reviewer sets
#: when it reads these loaders out of an installed release bundle rather
#: than a checkout.
DATA_ROOT = Path(os.environ.get("ONSET_HFO_DATA_ROOT") or PROJECT_ROOT / "data")

#: A real 60 s analysis that ships with the source, so every page works on a
#: fresh clone with no download: OpenNeuro ds003029, the slice the quickstart
#: documents, gzipped.
EXAMPLE = DATA_ROOT / "example_analysis"

#: Cohort-study tables, committed so the evaluation and outcome pages need no
#: download and no 90-minute rerun.
STUDIES = DATA_ROOT / "stability"

#: The outcome study's **per-subject** tables, committed for the same reason.
#: Group means are what a paper reports; a clinician asks about a patient.
COHORT = DATA_ROOT / "outcome"

#: The detector threshold sweep against expert markings, and the per-subject
#: reference counts it was scored against.
BENCHMARK = DATA_ROOT / "benchmark"

# --------------------------------------------------------------------------
# The standing disclaimer -- the canonical copy
# --------------------------------------------------------------------------
#
# DUPLICATION.md item 5. The same disclaimer is shown by berdakh/onset's
# app/common.py. This is the copy of record: the structure below is fixed, and
# the *only* part the teaching prototype changes is DATA_SENTENCE, because it
# runs on a synthetic cohort and must not claim otherwise. If you edit anything
# but DATA_SENTENCE here, edit the twin too -- see docs/DUPLICATION.md, which
# carries both wordings side by side so neither has to be reconstructed.
#
# It lives here rather than in common.py for the reason this module exists:
# common.py imports Streamlit, CI does not install Streamlit, and a test
# that reaches for the most important sentence in the product must not need
# a browser library to read it.

#: Fixed. Names the thing and refuses the category, in that order.
DISCLAIMER_LEAD = "Research tool — not a medical device."

#: The one line that legitimately differs between the two apps.
DATA_SENTENCE = ("Methods tested on real recordings with expert markings and "
                 "surgical outcomes — and nothing here is validated for clinical use.")

#: Fixed. The evidence rule, then the sentence that matters most: the product
#: contains no recommendation, as against containing one that is hedged.
DISCLAIMER_TAIL = ("Every number cites the window it came from. There is no "
                   "recommendation anywhere in this product; the clinician "
                   "decides.")

__all__ = [
    "EXAMPLE",
    "STUDIES",
    "COHORT",
    "BENCHMARK",
    "DISCLAIMER_LEAD",
    "DATA_SENTENCE",
    "DISCLAIMER_TAIL",
    "sweep",
    "sweep_grid",
    "benchmark_cohort",
    "operating_points",
    "cohort_table",
    "cohort_overview",
    "subject_metrics",
    "subject_channels",
    "subject_caveats",
    "analyses",
    "find_results",
    "detector_band",
    "event_from_record",
    "metadata_rows",
    "ranking_table",
    "leader_note",
    "evidence_table",
    "disagreement_table",
    "agreement_note",
    "citations",
    "reload_spec",
]


def find_results(root: str | Path) -> list[Path]:
    """Saved analyses under ``root``, newest first.

    A directory qualifies when it holds ``events.csv``; the cohort studies
    (benchmark, outcome, stability) live in the same results folder and are
    not single analyses, so they are skipped rather than offered and then
    failing to load.
    """
    root = Path(root)
    if not root.exists():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and (p / "events.csv").exists()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def analyses() -> list[Path]:
    """Every saved analysis: the shipped example first, then anything local.

    Lives here rather than beside the page because it is a fact about the
    filesystem, not a decision about layout -- and because this module must
    import without Streamlit, which is what lets the tests run in CI where
    Streamlit is not installed.
    """
    found = find_results(RESULTS_DIR)
    if EXAMPLE.exists() and EXAMPLE not in found:
        found = [EXAMPLE, *found]
    return found


def metadata_rows(store: ResultStore) -> list[tuple[str, str]]:
    """The header: what was analysed, in the order a reader needs it."""
    meta = store.metadata()
    window = meta.get("analysed_window_s") or [None, None]
    band = meta.get("band_hz") or []
    rows = [
        ("Subject", str(meta.get("subject", "?"))),
        ("Source", str(meta.get("source", "?"))),
        ("Task / run", f"{meta.get('task') or '-'} / {meta.get('run') or '-'}"),
        ("Window analysed",
         f"{window[0]:g}-{window[1]:g} s" if None not in window else "-"),
        ("Sampling rate", f"{meta.get('sampling_rate_hz', '?')} Hz"),
        ("Channels analysed",
         f"{meta.get('channels_analysed', '?')} of "
         f"{meta.get('channels_in_recording', '?')}"),
        ("Band", f"{band[0]:g}-{band[1]:g} Hz" if len(band) == 2 else "-"),
        ("Pipeline version", str(meta.get("pipeline_version", "?"))),
    ]
    if meta.get("seizure_onset_s") is not None:
        rows.insert(4, ("Marked seizure onset", f"{meta['seizure_onset_s']:g} s"))
    return rows


def ranking_table(store: ResultStore, detector: str | None = None) -> pd.DataFrame:
    """Channels by event rate, with the interval that says how firm the order is.

    The interval columns are not decoration. Sixty seconds turns a rate into a
    small count, and two channels whose intervals overlap are tied, not
    ranked -- which is why they sit next to the rate rather than in a tooltip.
    """
    name = detector or (store.detectors()[0] if store.detectors() else None)
    table = store.rates.get(name)
    if table is None or not len(table):
        return pd.DataFrame()
    rows = store.list_channels(name)
    frame = pd.DataFrame(rows)
    lookup = table.set_index("channel")
    for column, label in (("rate_ci_low", "ci_low"), ("rate_ci_high", "ci_high"),
                          ("mean_frequency_hz", "mean_freq_hz"),
                          ("mean_duration_ms", "mean_duration_ms")):
        if column in lookup.columns:
            frame[label] = [lookup.at[c, column] if c in lookup.index else None
                            for c in frame["channel"]]
    return frame.round({"rate_per_min": 2, "ci_low": 2, "ci_high": 2,
                        "mean_freq_hz": 1, "mean_duration_ms": 1})


def leader_note(store: ResultStore, detector: str | None = None) -> dict:
    """Does any channel stand out, or is the ranking sorting noise?

    The documented exception to "the UI computes nothing" -- see the module
    docstring. Returns ``{"available": False, ...}`` when the stored rate
    table has no confidence intervals to reason about.
    """
    from onset_hfo.metrics import leader_separation

    name = detector or (store.detectors()[0] if store.detectors() else None)
    table = store.rates.get(name)
    if table is None or not len(table):
        return {"available": False, "reason": f"no rate table for {name!r}"}
    return leader_separation(table)


def detector_band(store: ResultStore, detector: str) -> tuple[float, float]:
    """The band a detector looked in, as the stored config recorded it.

    Needed to redraw an event: the figure band-passes the signal the same way
    the detector did, and guessing it would show the reader a different filter
    from the one the measurement came through. ``spike`` is stored under
    ``spikes`` in the config, which is the kind of detail that belongs in one
    place rather than in the page.
    """
    config = store.config.get("config", {})
    key = "spikes" if detector == "spike" else detector
    band = config.get(key, {}).get("band")
    if not band:
        band = config.get("bands", {}).get("ripple", [80.0, 250.0])
    return float(band[0]), float(band[1])


def event_from_record(store: ResultStore, record: dict):
    """Rebuild the :class:`~onset_hfo.detectors.base.Event` the store describes.

    Every field is copied from the stored record, and that is the point. The
    event figure prints the peak frequency, the prominence over background and
    the amplitude in its subtitle; building the event from times alone leaves
    those at their defaults, and the figure then says ``peak nan Hz, nan dB``
    directly underneath a panel showing 192 Hz and 12.5 dB. Two numbers
    disagreeing on one screen is the failure mode this whole project is
    organised against, so the mapping lives here, once, with a test.

    Nothing is recomputed: the values are the pipeline's, carried through.
    """
    from onset_hfo.detectors.base import Event

    def _get(key, default):
        value = record.get(key, default)
        return default if value is None else value

    return Event(
        channel=str(record["channel"]),
        start=float(record["start"]),
        stop=float(record["stop"]),
        detector=str(record["detector"]),
        band=detector_band(store, str(record["detector"])),
        score=float(_get("score", 0.0)),
        peak_amplitude_uv=float(_get("peak_amplitude_uv", 0.0)),
        peak_frequency_hz=float(_get("peak_frequency_hz", float("nan"))),
        spectral_prominence_db=float(_get("spectral_prominence_db", float("nan"))),
        n_peaks=int(_get("n_peaks", 0)),
        n_cycles=float(_get("n_cycles", float("nan"))),
        accepted=bool(_get("accepted", True)),
        reject_reason=record.get("reject_reason") or None,
        co_occurs_with_spike=bool(_get("co_occurs_with_spike", False)),
    )


def evidence_table(store: ResultStore, channel: str, detector: str | None = None,
                   k: int = 5) -> pd.DataFrame:
    """The citable windows on one channel, strongest first."""
    rows = store.evidence(channel, detector=detector, k=k)
    return pd.DataFrame(rows)


def disagreement_table(store: ResultStore) -> pd.DataFrame:
    """Channels the two detectors rank very differently.

    Reported rather than resolved, which is the pipeline's position: the
    report states both ranks and picks neither, and so does this page.
    """
    return pd.DataFrame(store.disagreements())


def agreement_note(store: ResultStore) -> dict:
    """Event-by-event agreement between the detectors, as stored."""
    return store.detector_agreement()


def citations(store: ResultStore, evidence_ids: list[str]) -> list[dict]:
    """Resolve what an answer cited, and flag anything that does not resolve.

    An id the store cannot resolve is the interesting case: it means the
    answer referred to a window that was never retrieved. The guard should
    have caught it before the answer was shown, so a row with
    ``resolved=False`` reaching this page is a bug worth seeing rather than
    an error worth hiding.
    """
    out = []
    for evidence_id in evidence_ids:
        record = store.resolve(evidence_id)
        out.append({"evidence_id": evidence_id, "resolved": record is not None,
                    **(record or {})})
    return out


def reload_spec(store: ResultStore) -> dict | None:
    """What :func:`onset_hfo.datasets.fetch_slice` needs to show this signal again.

    ``None`` when the analysis did not come from the public archive -- a
    synthetic run has no archive to fetch from, and the page says so instead
    of offering a button that cannot work.
    """
    provenance = store.provenance
    source = str(provenance.get("source", ""))
    if not source.startswith("openneuro:"):
        return None
    window = [provenance.get("slice_start_s"), provenance.get("slice_stop_s")]
    if None in window:
        return None
    return {
        "dataset": source.split(":", 1)[1],
        "subject": provenance.get("subject"),
        "run": provenance.get("run"),
        "session": provenance.get("session"),
        "task": provenance.get("task"),
        "acq": provenance.get("acq"),
        "t_start": float(window[0]),
        "t_stop": float(window[1]),
    }


# --------------------------------------------------------------------------
# The cohort: one row per patient, rather than one row per group
# --------------------------------------------------------------------------

#: The comparison the outcome study pre-specified, and the one every
#: per-subject view below defaults to. Named once so a page cannot quietly
#: show a different arm than the tables it sits next to.
PRIMARY = {"band": "fast_ripple", "scope": "reviewed"}


def cohort_table(name: str) -> pd.DataFrame:
    """One committed per-subject table, or an empty frame if it is absent.

    Absence is normal on a clone that has not run the outcome study and has
    not got the committed extract either; the page says so rather than
    raising.
    """
    for candidate in (COHORT / name, COHORT / f"{name}.gz"):
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


def cohort_overview(band: str = PRIMARY["band"], scope: str = PRIMARY["scope"]) -> pd.DataFrame:
    """One row per patient: who they were, what was removed, what we found.

    Joins the four committed tables that describe a subject -- the archive's
    participants sidecar, the per-recording summary, the per-subject metrics,
    and the two stability studies -- into the table a clinician would actually
    ask for. Every column is read from disk; nothing here is recomputed.

    ``rz_coverage`` is carried deliberately. In five temporal-lobe subjects
    only a quarter of the listed resected contacts were recorded, so their
    "share inside the resection" describes a quarter of their resection, and a
    view that hid that would be inviting the reader to over-read those rows.
    """
    participants = cohort_table("participants.csv")
    recordings = cohort_table("recordings.csv")
    subjects = cohort_table("subjects.csv")
    if participants.empty or subjects.empty:
        return pd.DataFrame()

    frame = participants.merge(
        recordings[["subject", "n_channels", "n_reviewed", "n_resected_channels",
                    "rz_coverage", "n_expert_events"]],
        on="subject", how="left")

    arm = subjects[(subjects["band"] == band) & (subjects["scope"] == scope)]
    for source in ("expert", "rms"):
        part = arm[arm["source"] == source][
            ["subject", "top_channel_resected", "n_candidates",
             "candidates_resected", "share_in_rz"]]
        frame = frame.merge(part.rename(columns={
            "top_channel_resected": f"top_resected_{source}",
            "n_candidates": f"n_candidates_{source}",
            "candidates_resected": f"candidates_resected_{source}",
            "share_in_rz": f"share_in_rz_{source}",
        }), on="subject", how="left")

    windows = _stability_column("decision_stability.csv", "stable_across_windows")
    runs = _stability_column("decision_stability_across_runs.csv", "stable_across_runs")
    for part in (windows, runs):
        if not part.empty:
            frame = frame.merge(part, on="subject", how="left")

    frame["seizure_free"] = frame["outcome"].map({"S": True, "F": False})
    return frame.sort_values("subject").reset_index(drop=True)


def _stability_column(table: str, prefix: str) -> pd.DataFrame:
    """Per-subject ``stable`` flags from one stability table, one column per arm."""
    import pandas as _pd

    path = STUDIES / table
    if not path.exists():
        return _pd.DataFrame()
    frame = _pd.read_csv(path)
    if "metric" in frame.columns:
        frame = frame[frame["metric"] == "top_channel_resected"]
    wide = frame.pivot_table(index="subject", columns="source", values="stable",
                             aggfunc="first")
    wide.columns = [f"{prefix}_{c}" for c in wide.columns]
    return wide.reset_index()


def subject_metrics(subject: str) -> pd.DataFrame:
    """Every metric computed for one patient, across both bands and both arms.

    The group tables answer "does this work?". This answers "what happened to
    this patient?", which is the question a clinician asks and the one the
    project could not previously show.
    """
    subjects = cohort_table("subjects.csv")
    if subjects.empty:
        return subjects
    rows = subjects[subjects["subject"] == subject]
    return rows.sort_values(["band", "scope", "source"]).reset_index(drop=True)


def subject_channels(subject: str, band: str = PRIMARY["band"],
                     reviewed_only: bool = True) -> pd.DataFrame:
    """One patient's channels: which zone each sits in, and how busy it was.

    This is the row-level evidence under a per-patient claim. A channel is
    ``resected`` when both of its contacts were removed, ``partial`` when one
    was, and ``spared`` when neither was -- so a "partial" channel is the one
    a reader should look at hardest before believing either answer.
    """
    channels = cohort_table("channels.csv")
    if channels.empty:
        return channels
    rows = channels[(channels["subject"] == subject) & (channels["band"] == band)]
    if reviewed_only and "reviewed" in rows.columns:
        rows = rows[rows["reviewed"]]
    return rows.sort_values("expert_events", ascending=False).reset_index(drop=True)


def _is_false(value) -> bool:
    """``value`` is present and falsy -- as against absent, which is not "stable"."""
    return bool(pd.notna(value)) and not bool(value)


def subject_caveats(subject: str, band: str = PRIMARY["band"],
                    scope: str = PRIMARY["scope"]) -> list[str]:
    """Everything about this patient that should stop a reader over-reading them.

    Returned as plain sentences rather than flags, because the page that shows
    a per-patient answer is exactly where a caveat has to be unmissable. An
    empty list means the four checks below found nothing, not that the row is
    safe to act on -- the study is underpowered for every patient in it.
    """
    notes: list[str] = []
    overview = cohort_overview(band=band, scope=scope)
    if overview.empty or subject not in set(overview["subject"]):
        return ["No committed cohort tables for this subject."]
    row = overview[overview["subject"] == subject].iloc[0]

    coverage = row.get("rz_coverage")
    if pd.notna(coverage) and coverage < 1.0:
        notes.append(
            f"Only {coverage:.0%} of the contacts listed as resected were actually "
            f"recorded, so any 'share inside the resection' for this patient "
            f"describes {coverage:.0%} of their resection.")

    for source, arm in (("expert", "expert"), ("rms", "RMS")):
        n = row.get(f"n_candidates_{source}")
        if pd.notna(n) and n > 1:
            notes.append(
                f"The {arm} arm could not separate the busiest channel from "
                f"{int(n) - 1} other{'s' if n > 2 else ''}: its rate interval "
                f"overlaps theirs, so 'the busiest channel' names one of {int(n)}.")
        # ``is False`` would be wrong here and silently so: pandas hands back
        # ``numpy.bool_(False)``, which is not the ``False`` singleton, so an
        # identity test leaves an unstable patient uncaveated. The test named
        # ``test_every_patient_is_either_clean_or_carries_a_reason`` exists
        # because this page shipped that bug once.
        if _is_false(row.get(f"stable_across_windows_{source}")):
            notes.append(
                f"The {arm} arm's inside/outside answer changed between the five "
                f"one-minute windows of this recording.")
        if _is_false(row.get(f"stable_across_runs_{source}")):
            notes.append(
                f"The {arm} arm's answer changed between this patient's "
                f"recordings on different nights.")

    missing = row.get("missing")
    if isinstance(missing, str) and missing.strip():
        notes.append(f"Contacts the clinical sheet lists but the parser could not "
                     f"resolve: {missing}.")
    return notes


# --------------------------------------------------------------------------
# The detector sweep: what the page shows instead of a transcription
# --------------------------------------------------------------------------
#
# This section exists because `4_Detectors` used to carry the sweep as a
# hand-typed table of nine rows. Every number in it was right, and the page
# beside it still quoted a *tenth* number -- the mean expert fast-ripple count
# -- that was wrong in three places at once. A transcription that is correct
# today is a number with no owner; these functions give the page the file.


def sweep() -> pd.DataFrame:
    """The committed threshold sweep, or an empty frame if it is absent.

    One row per band x detector x threshold: agreement with the expert HFO
    markings of ds003498, cohort means over 20 subjects. ``precision`` is
    agreement, never accuracy -- the reference is another detector's validated
    output, so an event we find that it never proposed counts against us
    whether or not it is real.
    """
    path = BENCHMARK / "agreement_sweep.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def benchmark_cohort() -> pd.DataFrame:
    """Per-subject reference counts: what the sweep above was scored against.

    Committed so that a claim about *how many* expert events there were can be
    checked from a clone, rather than transcribed from an untracked artifact
    and then quoted at three different values.
    """
    path = BENCHMARK / "cohort.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def sweep_grid(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """What was actually swept, per arm -- the context a single row hides.

    The four arms were not swept over the same grid: the ripple RMS arm runs
    down to 1.0 SD and the fast-ripple RMS arm up to 10.0, because each was
    extended when its first optimum landed on a boundary. Line length was
    never extended either way. A reader comparing detectors needs to know
    that, or "line length is worse" reads as a measurement when part of it is
    a grid that stops sooner.
    """
    frame = sweep() if frame is None else frame
    if frame.empty:
        return frame
    grid = (frame.groupby(["band", "detector"])["threshold_sd"]
            .agg(n_points="count", lowest="min", highest="max")
            .reset_index())
    return grid.sort_values(["band", "detector"]).reset_index(drop=True)


def operating_points(by: str = "rank_rho") -> pd.DataFrame:
    """The best threshold for each arm, and whether the grid actually found it.

    ``at_boundary`` is the column that matters. An optimum sitting on the edge
    of the swept range is not an optimum -- it is the grid running out, and
    this project has already been wrong that way once: the first ripple sweep
    stopped at 2.0 SD and reported 2.0 as the answer, which is why the
    committed table now runs down to 1.0.
    """
    frame = sweep()
    if frame.empty or by not in frame.columns:
        return pd.DataFrame()
    rows = []
    for (band, detector), part in frame.groupby(["band", "detector"]):
        part = part.sort_values("threshold_sd")
        best = part.loc[part[by].idxmax()]
        threshold = float(best["threshold_sd"])
        rows.append({
            "band": band,
            "detector": detector,
            "threshold_sd": threshold,
            by: float(best[by]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "detections": float(best["detections"]),
            "at_boundary": threshold in (float(part["threshold_sd"].min()),
                                         float(part["threshold_sd"].max())),
        })
    return (pd.DataFrame(rows).sort_values(["band", "detector"])
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# The four-detector family, and two studies that followed from it
# --------------------------------------------------------------------------
#
# Same principle as the sweep above: the page gets the committed file. These
# three extracts each answered a question the Research page used to list as
# open, and each answer is a null, which is exactly the kind of result that
# decays into a vague claim if it is retyped rather than read.


def family_sweep() -> pd.DataFrame:
    """The four-detector sweep, or an empty frame if it is absent.

    Same scoring as :func:`sweep`, extended to the analytic-signal envelope and
    short-time energy and run over a common grid so the four are comparable.
    """
    path = BENCHMARK / "four_detector_sweep.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def family_operating_points(by: str = "rank_rho") -> pd.DataFrame:
    """Each detector at *its own* best threshold, which is the only fair view.

    Comparing four features at a shared threshold measures the threshold, not
    the feature: ``median + 5 robustSD`` sits at about the 98th percentile of an
    RMS trace and the 96th of a squared one, so the same multiplier is a
    materially more permissive operating point on short-time energy. The
    ``threshold_sd`` column is therefore part of the result, not a parameter to
    be held constant.
    """
    frame = family_sweep()
    if frame.empty or by not in frame.columns:
        return pd.DataFrame()
    rows = []
    for (band, detector), part in frame.groupby(["band", "detector"]):
        best = part.loc[part[by].idxmax()]
        rows.append({
            "band": band,
            "detector": detector,
            "threshold_sd": float(best["threshold_sd"]),
            "rank_rho": float(best["rank_rho"]),
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "detections": float(best["detections"]),
        })
    out = pd.DataFrame(rows)
    return (out.sort_values(["band", by], ascending=[True, False])
            .reset_index(drop=True))


def family_margin(by: str = "rank_rho") -> dict:
    """How much the two added features bought, per band. The answer is nothing.

    Returned rather than asserted so the page cannot drift from the file: if a
    future sweep makes one of them win properly, the margin moves and the
    sentence beside it is wrong in a way a reader can see.
    """
    best = family_operating_points(by)
    if best.empty:
        return {}
    original = {"rms", "line_length"}
    out = {}
    for band, part in best.groupby("band"):
        was = part[part.detector.isin(original)][by].max()
        now = part[by].max()
        winner = part.loc[part[by].idxmax(), "detector"]
        out[band] = {"best_original": float(was), "best_overall": float(now),
                     "winner": str(winner), "margin": float(now - was)}
    return out


def subpopulation_outcome(band: str = "ripple") -> pd.DataFrame:
    """Do the ripple sub-populations localise the resection better? They do not.

    One row per metric x population, pooled over five 60 s windows per patient.
    Reported in the ripple band because the fast-ripple spike-coupled
    population has a median of one event per window, which cannot support an
    argmax; ``docs/EVALUATION.md`` §3b has that argument in full.
    """
    path = COHORT / "subpopulation_screen.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    return rows[rows["band"] == band].reset_index(drop=True)


def subpopulation_measurability() -> pd.DataFrame:
    """Events per window per population, which is why the band choice was forced.

    The column to read is ``median``: a population with one event per minute
    cannot rank channels, and an argmax over it produced the most flattering
    numbers in the whole screen.
    """
    path = COHORT / "subpopulation_screen.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    out = (rows.groupby(["band", "population"])["n_events"]
           .agg(median="median", q1=lambda s: s.quantile(0.25), maximum="max")
           .reset_index())
    empty = (rows.assign(sparse=rows["n_events"] < 5)
             .groupby(["band", "population"])["sparse"].sum()
             .reset_index(name="windows_under_5"))
    return out.merge(empty, on=["band", "population"])


def subpopulation_groups(band: str = "ripple") -> pd.DataFrame:
    """The §3b group table, as committed, so the page need not recompute it.

    The per-window rows in :func:`subpopulation_outcome` take a bootstrap and an
    exact permutation to reduce, which is seconds of work and not something to
    do on a page load. This is the reduced table, committed for the same reason
    `data/stability/` holds the group tables beside the per-subject ones.
    """
    path = COHORT / "subpopulation_groups.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    return rows[rows["band"] == band].reset_index(drop=True)


def robustness_groups() -> pd.DataFrame:
    """The §6c group table, as committed. See :func:`subpopulation_groups`."""
    path = COHORT / "robustness_groups.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def robustness_ablation() -> pd.DataFrame:
    """The re-testing rule the planner uses, measured against outcome.

    ``rule`` is ``plain`` (score = rate, what the fixed rungs produce) or
    ``multiplied`` (score = rate x robustness, what the re-planning rungs
    produce). ``retest_silent == 1`` marks a window where the stricter pass
    found nothing on any channel; the rule no longer fires on those.
    ``score_mass == 0`` marks a window with no events at the survey threshold
    either, where zero is the measurement rather than a destroyed ranking.
    """
    path = COHORT / "robustness_ablation.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def montage_screen() -> pd.DataFrame:
    """The montage comparison: the same pipeline under three references.

    One row per ``band`` x ``subject`` x ``window`` x ``montage`` (``bipolar``,
    ``referential``, ``average``), scored against the resection with the
    plain rule and against the expert markings. ``event_scoring`` says whether
    precision and recall were measured on the arm's own channels (``direct``,
    bipolar only) or on detections projected onto the reviewed pairs
    (``projected``); ``rho_vs_bipolar`` is how far the arm's ranking moved
    from the shipped default's, on the pairs both can rank.
    """
    path = COHORT / "montage_screen.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def montage_groups() -> pd.DataFrame:
    """The reduced montage table: outcome AUCs and cohort-mean agreement per
    band x montage, as committed. See :func:`subpopulation_groups`."""
    path = COHORT / "montage_groups.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def robustness_collapse() -> dict:
    """How often the ranking has no leader at all, by band.

    Since the rule stopped firing on a silent re-test these are windows that
    were empty to begin with, not rankings the rule destroyed -- so this is now
    a statement about how sparse the fast-ripple band is, which is still worth
    seeing before the AUC table.
    """
    rows = robustness_ablation()
    if rows.empty or "score_mass" not in rows.columns:
        return {}
    mult = rows[rows["rule"] == "multiplied"]
    if not len(mult):
        return {}
    collapsed = mult[mult["score_mass"] == 0]
    by_band = collapsed.groupby("band").size().to_dict()
    return {"collapsed": int(len(collapsed)), "windows": int(len(mult)),
            "by_band": {str(k): int(v) for k, v in by_band.items()}}
