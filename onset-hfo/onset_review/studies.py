"""The results site's study pages, as Markdown built from the committed tables.

The Streamlit site (``app/``) has six pages that report the cohort studies:
Detectors, Outcome, Patients, Data, Architecture, Research. They are the
public record of what this project measured, and the desktop window shows
the same pages so that a reviewer never has to leave it to find out how much
a number can be trusted.

Each builder here returns one Markdown document. The numbers come from the
same committed tables the site reads, through the same loaders
(``app.panels``), so a corrected extract corrects both. The prose is the
site's, carried over; where the site derives a sentence from a table, so does
this. No Qt in this module: the text is testable without a display, and the
page widget in :mod:`onset_review.studypages` only renders it.

The site and its tables are part of the repository, not of the wheel. On an
install that lacks them the builders say so, and the pages point at the site.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["STUDIES", "available", "build", "controls", "patient_rows",
           "cached_window_for", "site_root", "key_message", "fold_tables",
           "FOLD_ROWS", "chart_data"]

#: (key, sidebar label, one-line description)
STUDIES = (
    ("detectors", "Detectors", "how the detectors score against expert markings"),
    ("outcome", "Outcome", "did the map point at the tissue whose removal cured the patient?"),
    ("patients", "Patients", "the same study one patient at a time, with the caveats"),
    ("data", "Data", "the two archives, and what each one carries"),
    ("architecture", "Architecture", "every component, and which ones this release includes"),
    ("research", "Research", "the measured state of the work, and what is open"),
)

BANDS = {"ripple": "Ripples (80–250 Hz)", "fast_ripple": "Fast ripples (250–500 Hz)"}
METRICS = {
    "rank_rho": "channel-rank ρ",
    "f1": "F1",
    "precision": "precision (agreement)",
    "recall": "recall",
    "detections": "detections per 60 s",
    "top5_overlap": "top-5 channels shared, out of 5",
}
SCOPES = {"reviewed": "Reviewed by the annotators", "all": "Every recorded channel"}

SITE = "https://berdakh.github.io/onset-hfo/"


# -- the data layer --------------------------------------------------------

#: Where an installed release bundle keeps the site's loaders and tables
#: (``install.sh`` copies the bundle's ``site/`` here).
INSTALLED_SITE = Path.home() / ".local" / "share" / "onset-review" / "site"


def site_root() -> Path | None:
    """The directory holding ``app/panels.py`` and ``data/``, or None.

    Three places, in order: ``ONSET_REVIEW_SITE_DIR``; the checkout this
    package was imported from; the installed bundle's ``site/``.
    """
    import os

    from onset_hfo.config import PROJECT_ROOT

    named = os.environ.get("ONSET_REVIEW_SITE_DIR")
    candidates = ([Path(named).expanduser()] if named else []) + [PROJECT_ROOT, INSTALLED_SITE]
    for root in candidates:
        if (root / "app" / "panels.py").exists():
            return root
    return None


_loaded: dict = {}


def _panels():
    """The site's loaders, or None where the site is not on this machine.

    ``app/panels.py`` is loaded from its file rather than imported as
    ``app.panels``: the site's package ``__init__`` imports Streamlit, which
    the desktop does not ship, and the loaders themselves need only pandas
    and this project. The data root is pointed at the same place first, so
    a bundle's tables are found beside a bundle's loaders.
    """
    root = site_root()
    if root is None:
        return None
    if _loaded.get("root") == root:
        return _loaded["module"]
    import importlib.util
    import os

    from onset_hfo.config import PROJECT_ROOT

    if root != PROJECT_ROOT:
        os.environ["ONSET_HFO_DATA_ROOT"] = str(root / "data")
    else:
        # A checkout reads its own data/; a value left over from an earlier
        # site must not redirect it.
        os.environ.pop("ONSET_HFO_DATA_ROOT", None)
    try:
        spec = importlib.util.spec_from_file_location("onset_site_panels",
                                                      root / "app" / "panels.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:       # noqa: BLE001 - absent, broken: both mean "not here"
        return None
    _loaded.update(root=root, module=module)
    return module


def available() -> bool:
    panels = _panels()
    if panels is None:
        return False
    return Path(panels.COHORT).exists() or Path(panels.BENCHMARK).exists()


def _image(name: str) -> Path | None:
    """A figure of the site's docs, wherever the site is."""
    root = site_root()
    path = (root / "docs" / "img" / name) if root is not None else None
    return path if path is not None and path.exists() else None


def _missing(name: str) -> str:
    return (f"# {name}\n\n> **The study tables are not on this machine.** These "
            f"pages read the committed extracts under `data/` through the site's "
            f"own loaders (`app/panels.py`), which ship with a checkout of the "
            f"repository and with the release bundle's `site/` (installed by "
            f"`install.sh`), not with the bare wheel. The same page is on the "
            f"results site: {SITE}\n")


# -- Markdown helpers ----------------------------------------------------------


def _cell(value) -> str:
    import math

    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if value.is_integer() and abs(value) < 1e6:
            return f"{int(value)}"
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(frame, columns: list[str] | None = None, rename: dict | None = None,
          limit: int | None = None) -> str:
    """A GitHub-flavoured Markdown table from a DataFrame. Empty → a sentence."""
    if frame is None or len(frame) == 0:
        return "*(no rows)*\n"
    shown = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    if rename:
        shown = shown.rename(columns=rename)
    if limit is not None:
        shown = shown.head(limit)
    heads = [str(c) for c in shown.columns]
    cells = [[_cell(v) for v in row] for row in shown.itertuples(index=False)]
    # Numbers right-aligned, as a reader expects of a column of figures;
    # GitHub's `---:` is the dialect's own way of saying so.
    rules = []
    for index in range(len(heads)):
        column = [row[index] for row in cells if row[index] != "—"]
        rules.append("---:" if column and all(_numeric(c) for c in column) else "---")
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join(rules) + "|"]
    for row in cells:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _numeric(cell: str) -> bool:
    try:
        float(cell.replace(",", ""))
    except ValueError:
        return False
    return True


#: A table longer than this many rows is folded on a study page until asked
#: for: the page is read first and consulted second.
FOLD_ROWS = 10


def fold_tables(text: str, max_rows: int = FOLD_ROWS) -> tuple[str, int]:
    """Replace every Markdown table longer than `max_rows` with a line saying
    so. Returns the text and how many tables were folded."""
    lines = text.splitlines()
    out: list[str] = []
    folded = 0
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("|"):
            start = index
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                index += 1
            block = lines[start:index]
            rows = max(0, len(block) - 2)
            if rows > max_rows:
                folded += 1
                out.append(f"> *A table of {rows} rows is folded. Tick **Show every "
                           f"table** above to see it.*")
            else:
                out.extend(block)
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), folded


def figure_dir() -> Path:
    """Where the pages' own figures are written: the user's cache, or the
    temporary directory where there is none."""
    import os
    import tempfile

    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    for candidate in (root / "onset-review" / "figures",
                      Path(tempfile.gettempdir()) / "onset-review-figures"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def chart(name: str, draw, alt: str) -> str:
    """A figure drawn by `draw(path)` as a Markdown image line, or nothing:
    a page must build whether or not a figure can be drawn here."""
    try:
        path = draw(figure_dir() / f"{name}.png")
    except Exception:       # noqa: BLE001 - a figure is a bonus, never a failure
        return ""
    return f"![{alt}]({Path(path).as_posix()})\n"


def note(kind: str, text: str) -> str:
    """A callout: `info`, `warn`, `bad` or `ok`, as a blockquote with a lead."""
    lead = {"info": "Note", "warn": "Caution", "bad": "Not supported", "ok": "Result"}
    body = " ".join(line.strip() for line in text.strip().splitlines())
    return f"> **{lead.get(kind, 'Note')}.** {body}\n"


# -- the key message ---------------------------------------------------------

def key_message(key: str, **choices) -> dict:
    """What a page shows, for someone who reads nothing else on it.

    {"lead": two or three plain sentences, "tiles": [(value, caption), ...]}.
    Every number is read from the same committed tables the page renders,
    through the same loaders, so the message cannot drift from the page. With
    the tables absent the message says so rather than invent a figure.
    """
    builders = {"detectors": _key_detectors, "outcome": _key_outcome,
                "patients": _key_patients, "data": _key_data,
                "architecture": _key_architecture, "research": _key_research}
    if key not in builders:
        raise KeyError(f"no study page {key!r}")
    try:
        return builders[key](**choices)
    except Exception as error:      # noqa: BLE001 - a message must never break a page
        return {"lead": f"The key message could not be computed: {error}", "tiles": []}


_ABSENT = {"lead": "The committed tables are not on this machine, so this page can "
                   "only point at the results site.", "tiles": []}


def _key_detectors(band: str = "ripple", criterion: str = "rank_rho", **_) -> dict:
    panels = _panels()
    if panels is None:
        return _ABSENT
    best = panels.operating_points(criterion)
    best = best[best["band"] == band]
    cohort = panels.benchmark_cohort()
    if best is None or len(best) == 0:
        return _ABSENT
    top = best.sort_values("rank_rho", ascending=False).iloc[0]
    spread = float(best["rank_rho"].max() - best["rank_rho"].min())
    n = len(cohort) if cohort is not None else 0
    events = int(cohort["n_expert_events"].sum()) if cohort is not None else 0
    lead = (f"Four plain detectors, each thresholding one feature of the signal, rank the "
            f"contacts of {n} patients almost identically: the best reaches channel-rank "
            f"ρ {top['rank_rho']:.2f} against the expert markings and the others sit within "
            f"{spread:.3f} of it. Precision near {top['precision']:.2f} is a floor set by the "
            f"reference, which is another detector's output, not a measure of accuracy.")
    return {"lead": lead, "tiles": [
        (f"{top['rank_rho']:.2f}", "channel-rank ρ, best detector"),
        (f"{top['precision']:.2f}", "precision at that threshold"),
        (f"{n}", "patients"),
        (f"{events:,}", "expert-marked events"),
    ]}


def _key_outcome(**_) -> dict:
    panels = _panels()
    if panels is None:
        return _ABSENT
    import pandas as pd

    path = Path(panels.STUDIES) / "outcome_groups_300s.csv"
    if not path.exists():
        return _ABSENT
    groups = pd.read_csv(path).query(
        "scope == 'reviewed' and band == 'fast_ripple' and metric == 'top_channel_resected'")
    expert = groups[groups["source"] == "expert"].iloc[0]
    rms = groups[groups["source"] == "rms"].iloc[0]
    n_sf, n_rec = int(expert["n_seizure_free"]), int(expert["n_recurrence"])
    lead = (f"In {n_sf + n_rec} patients, the busiest fast-ripple channel sat inside the "
            f"removed tissue more often in those who became seizure-free: "
            f"{expert['mean_seizure_free']:.0%} of them against "
            f"{expert['mean_recurrence']:.0%} of those whose seizures returned, on the "
            f"expert markings. With {n_sf} patients against {n_rec} that does not reach "
            f"significance (p = {expert['p_permutation']:.2f}), and a shorter analysis "
            f"window once said otherwise.")
    return {"lead": lead, "tiles": [
        (f"{expert['auc']:.2f}", "AUC, expert markings"),
        (f"{rms['auc']:.2f}", "AUC, our detector"),
        (f"{n_sf} vs {n_rec}", "seizure-free vs recurrence"),
        (f"{expert['p_permutation']:.2f}", "permutation p"),
    ]}


def _key_patients(band: str = "fast_ripple", scope: str = "reviewed", **_) -> dict:
    panels = _panels()
    if panels is None:
        return _ABSENT
    cohort = panels.cohort_overview(band=band, scope=scope)
    if cohort is None or len(cohort) == 0:
        return _ABSENT
    n = len(cohort)
    free = cohort["seizure_free"].astype(bool)
    hit = cohort["top_resected_expert"] == 1
    agree = int((cohort["top_resected_expert"] == cohort["top_resected_rms"]).sum())
    hit_sf, hit_rec = int((hit & free).sum()), int((hit & ~free).sum())
    n_sf, n_rec = int(free.sum()), int((~free).sum())
    lead = (f"The experts' busiest channel was inside the resection in {hit_sf} of the "
            f"{n_sf} patients who became seizure-free and in {hit_rec} of the {n_rec} whose "
            f"seizures returned. Our detector reaches the same inside-or-outside answer as "
            f"the experts in {agree} of {n} patients: the two disagree about rates more "
            f"than about which channel leads. A result four patients wide moves when one "
            f"row does.")
    return {"lead": lead, "tiles": [
        (f"{hit_sf}/{n_sf}", "seizure-free: busiest channel resected"),
        (f"{hit_rec}/{n_rec}", "recurrence: busiest channel resected"),
        (f"{agree}/{n}", "detector agrees with the experts"),
        (f"{n}", "patients"),
    ]}


def _key_data(**_) -> dict:
    panels = _panels()
    cohort = panels.benchmark_cohort() if panels is not None else None
    if cohort is None or len(cohort) == 0:
        return {"lead": "Two public archives of real patients, no simulator. ds003498 carries "
                        "expert HFO markings, the contacts the surgeon removed and the "
                        "outcome; ds003029 adds seizure recordings from many centres.",
                "tiles": [("2", "public archives")]}
    n = len(cohort)
    events = int(cohort["n_expert_events"].sum())
    sfreq = float(cohort["sfreq_hz"].iloc[0])
    lead = (f"Two public archives of real patients, no simulator. ds003498 is the one that "
            f"matters: {n} patients recorded at {sfreq:.0f} Hz in slow-wave sleep, with "
            f"{events:,} expert-marked HFOs, the contacts the surgeon removed, and whether "
            f"the patient became seizure-free. ds003029 adds seizure recordings from 35 "
            f"patients across centres.")
    return {"lead": lead, "tiles": [
        ("2", "public archives"), (f"{n}", "patients with outcome"),
        (f"{events:,}", "expert-marked HFOs"), (f"{sfreq:.0f} Hz", "sampling rate"),
    ]}


def _key_architecture(**_) -> dict:
    return {"lead": ("One pipeline, read by two readers. The signal is filtered and "
                     "re-referenced, four detectors each threshold one feature, one "
                     "artifact-rejection stage removes filter ringing, and every "
                     "measurement is written once. The assistant and this window read "
                     "those results through the same read-only store and cannot change "
                     "a number."),
            "tiles": [("4", "detectors"), ("1", "artifact-rejection stage"),
                      ("8", "read-only tools for the assistant"),
                      ("0", "numbers a reader can change")]}


def _key_research(**_) -> dict:
    panels = _panels()
    if panels is None:
        return _ABSENT
    import pandas as pd

    best = panels.operating_points("rank_rho")
    rho = float(best[best["band"] == "ripple"]["rank_rho"].max())
    path = Path(panels.STUDIES) / "outcome_groups_300s.csv"
    groups = pd.read_csv(path).query(
        "scope == 'reviewed' and band == 'fast_ripple' and metric == 'top_channel_resected' "
        "and source == 'expert'").iloc[0]
    cohort = panels.cohort_overview()
    stable = int(cohort["stable_across_runs_expert"].astype(bool).sum())
    n = len(cohort)
    lead = (f"It detects HFOs in real recordings: channel ranking agrees with the experts "
            f"at ρ ≈ {rho:.2f}. It does not beat a published detector and does not claim "
            f"to. The map points in the right direction for surgical outcome (AUC "
            f"{groups['auc']:.2f}) without reaching significance on "
            f"{int(groups['n_seizure_free'])} against {int(groups['n_recurrence'])} "
            f"patients. The ranking is stable across nights in {stable} of {n} patients, "
            f"and not within a single minute. Nothing here is validated for clinical use.")
    return {"lead": lead, "tiles": [
        (f"{rho:.2f}", "channel-rank ρ vs experts"),
        (f"{groups['auc']:.2f}", "outcome AUC, not significant"),
        (f"{stable}/{n}", "stable across nights"),
        ("No", "validated for clinical use"),
    ]}


# -- the pages -------------------------------------------------------------


def controls(key: str) -> dict:
    """The choices a page offers, as {name: (label, {value: text}, default)}."""
    if key == "detectors":
        return {"band": ("Band", BANDS, "ripple"),
                "metric": ("Plot", METRICS, "rank_rho"),
                "criterion": ("Best by", {"rank_rho": METRICS["rank_rho"],
                                          "f1": METRICS["f1"]}, "rank_rho")}
    if key == "patients":
        return {"band": ("Band", {"fast_ripple": BANDS["fast_ripple"],
                                  "ripple": BANDS["ripple"]}, "fast_ripple"),
                "scope": ("Channels", SCOPES, "reviewed"),
                "subject": ("Patient", {}, "")}
    return {}


def patient_rows(band: str = "fast_ripple", scope: str = "reviewed"):
    """The cohort table, for the Patients page's subject picker."""
    panels = _panels()
    if panels is None:
        return None
    try:
        return panels.cohort_overview(band=band, scope=scope)
    except Exception:       # noqa: BLE001
        return None


def cached_window_for(subject: str, windows) -> dict | None:
    """The first cached window of `subject`, or None: what the Patients page
    opens when a patient is picked, the one link the site cannot make."""
    if windows is None or len(windows) == 0:
        return None
    rows = windows[(windows["subject"] == subject) & (windows["dataset"] == "ds003498")]
    if len(rows) == 0:
        return None
    rows = rows.sort_values(["t_start"])
    return rows.iloc[0].to_dict()


def build(key: str, session=None, **choices) -> str:
    """One page's Markdown. Unknown keys raise; missing data explains itself.

    With `session` (the recording open in the window) the page gains a
    *This recording* section under its title (`onset_review.thisrecording`),
    and the Detectors and Outcome charts mark it. A reader's own re-run and
    cohort (`onset_review.yourstudy`) are added under *Yours* when there are
    any. The published text and tables are the same either way."""
    builders = {"detectors": _detectors, "outcome": _outcome, "patients": _patients,
                "data": _data, "architecture": _architecture, "research": _research}
    if key not in builders:
        raise KeyError(f"no study page {key!r}; one of {[k for k, _, _ in STUDIES]}")
    if key in ("detectors", "outcome"):
        text = builders[key](session=session, **choices)
    else:
        text = builders[key](**choices)
    extra = _this_recording(key, session)
    if extra:
        head, _, rest = text.partition("\n")
        text = f"{head}\n\n{extra}\n{rest}"
    return text


def _this_recording(key: str, session) -> str:
    if session is None:
        return ""
    from onset_review import thisrecording

    cohort = None
    if key == "outcome":
        panels = _panels()
        try:
            cohort = panels.cohort_overview() if panels is not None else None
        except Exception:       # noqa: BLE001 - the section says what it can without it
            cohort = None
    try:
        return thisrecording.section(key, session, cohort)
    except Exception as error:       # noqa: BLE001 - a section must not cost the page
        return note("warn", f"This recording could not be set against the study: {error}")


def _stamp(*parts) -> str:
    """A short digest of what a figure shows, so a changed figure gets a new
    file name and the page never shows the one cached before it."""
    import hashlib

    return hashlib.sha1(repr(parts).encode()).hexdigest()[:10]


def chart_data(key: str, session=None, **choices) -> dict | None:
    """What a page's chart is drawn from, for the window that draws it live
    (`onset_review.chartview`). None for a page without a chart or data."""
    panels = _panels()
    if panels is None:
        return None
    if key == "detectors":
        from onset_review import thisrecording, yourstudy

        band = choices.get("band", "ripple")
        metric = choices.get("metric", "rank_rho")
        frame = panels.sweep()
        if frame is None or not len(frame):
            return None
        yours, _settings = yourstudy.load_your_sweep()
        start = None
        if session is not None:
            detector = session.request.detectors[0]
            try:
                start = thisrecording.threshold_for(session, detector)
            except Exception:       # noqa: BLE001
                start = None
        return {"kind": "sweep", "frame": frame, "band": band, "metric": metric,
                "best": panels.operating_points(choices.get("criterion", "rank_rho")),
                "marks": thisrecording.marks(session, metric, band) if session else [],
                "yours": yours, "start_threshold": start}
    if key == "outcome":
        import pandas as pd

        from onset_review import thisrecording, yourstudy

        groups_path = Path(panels.STUDIES) / "outcome_groups_300s.csv"
        if not groups_path.exists():
            return None
        groups = pd.read_csv(groups_path).query("scope == 'reviewed' and band == 'fast_ripple'")
        mine = yourstudy.cohort_frame()
        return {"kind": "outcome", "cohort": panels.cohort_overview(), "groups": groups,
                "highlight": session.request.subject
                if thisrecording.in_cohort(session) else None,
                "yours": mine if len(mine) else None}
    return None


def _detectors(band: str = "ripple", metric: str = "rank_rho",
               criterion: str = "rank_rho", session=None, **_) -> str:
    panels = _panels()
    if panels is None:
        return _missing("Detectors and how they are scored")
    from onset_hfo.config import THRESHOLDS

    frame = panels.sweep()
    cohort = panels.benchmark_cohort()
    out = ["# Detectors and how they are scored\n", """
Four deliberately plain detectors that differ in **one thing only** — the
feature they threshold — so a disagreement between them is attributable to the
feature rather than to four implementations drifting apart.

- **RMS energy** (Staba et al. 2002) — root-mean-square of the band-passed signal.
- **Line length** (Gardner et al. 2007) — cumulative absolute difference.
- **Hilbert envelope** — magnitude of the analytic signal.
- **Short-time energy** — mean square, which is RMS *before* the square root.

All four then pass every candidate through the same **artifact rejection**, which
is where filter ringing is removed and where almost all of the precision comes
from.
"""]
    if frame is None or len(frame) == 0:
        out.append(note("bad", "`data/benchmark/agreement_sweep.csv` is missing. "
                               "Regenerate with `python -m onset_hfo.cli benchmark`."))
        return "\n".join(out)

    out.append("## The reference, and why it is agreement rather than accuracy\n")
    if cohort is not None and len(cohort):
        out.append(
            f"**{len(cohort)} patients**, first {cohort['duration_s'].iloc[0]:.0f} s of "
            f"run-01 each, {cohort['sfreq_hz'].iloc[0]:.0f} Hz, slow-wave sleep · "
            f"**{int(cohort['n_expert_events'].sum()):,} expert-marked events** "
            f"({int(cohort['expert_ripples'].sum()):,} ripples, "
            f"{int(cohort['expert_fast_ripples'].sum()):,} fast ripples) · "
            f"**{int(cohort['n_reviewed_channels'].min())}–"
            f"{int(cohort['n_reviewed_channels'].max())} reviewed channels** per "
            f"patient — scoring is restricted to these; a detection elsewhere is "
            f"unjudged, not wrong.\n")
        out.append(f"""
Scored against expert HFO markings on the 20 patients of
[ds003498](https://openneuro.org/datasets/ds003498). The reference is the
validated output of *another detector*, not a census of every oscillation, so
an event we find that it never proposed counts against us whether or not it is
real. **These are agreement numbers. Precision around 0.6 is a floor, not a
measurement.**

Per patient per 60 s that is a mean of
**{cohort['expert_ripples'].mean():,.0f} marked ripples** and
**{cohort['expert_fast_ripples'].mean():.0f} marked fast ripples** — the
denominator every recall figure below is a fraction of.
""")

    out.append(f"## The sweep, as committed — {BANDS.get(band, band)}, {METRICS.get(metric, metric)}\n")
    from onset_review import studycharts, thisrecording, yourstudy

    best_points = panels.operating_points(criterion)
    marks = thisrecording.marks(session, metric, band) if session is not None else []
    yours, your_settings = yourstudy.load_your_sweep()
    out.append(chart(f"sweep-{band}-{metric}-{criterion}-{_stamp(marks, your_settings)}",
                     lambda path: studycharts.sweep_chart(frame, band, metric, best_points,
                                                          path, marks=marks, yours=yours),
                     "The threshold sweep, one line per detector"))
    if marks or yours is not None:
        out.append("*Stars: this window, measured the same way. Dashed: your re-run, "
                   "below. Solid lines are the published sweep.*\n")
    part = frame[frame["band"] == band]
    pivot = part.pivot(index="threshold_sd", columns="detector", values=metric).reset_index()
    out.append(table(pivot.rename(columns={"threshold_sd": "threshold (SD)"})))
    out.append(f"*{METRICS.get(metric, metric)} at each threshold, cohort means over "
               f"20 patients. `channel-rank ρ` is the Spearman correlation between "
               f"our per-channel rates and the experts'.*\n")
    out.append("### Every column\n")
    out.append(table(
        part.drop(columns="band").sort_values(["detector", "threshold_sd"]),
        rename={"threshold_sd": "threshold (SD)", "f1": "F1", "rank_rho": "channel-rank ρ",
                "detections": "detections / 60 s", "top5_overlap": "top-5 shared"}))

    out.append(f"## Where each arm peaks — best by {METRICS.get(criterion, criterion)}\n")
    best = panels.operating_points(criterion)
    shown = best.rename(columns={
        "threshold_sd": "best threshold (SD)", "rank_rho": "channel-rank ρ",
        "f1": "F1", "detections": "detections / 60 s"}).round(3)
    shown["on the edge of the grid?"] = best["at_boundary"].map(
        {True: "yes — not yet measured", False: "no — interior"})
    out.append(table(shown.drop(columns="at_boundary")))
    edge = best[best["at_boundary"]]
    if len(edge):
        out.append(note("warn",
            "**An optimum on the edge of the swept grid is not an optimum, it is "
            "the grid running out.** " + "; ".join(
                f"`{r.detector}` in the {r.band} band peaks at {r.threshold_sd:g} SD, "
                f"the {'lowest' if r.threshold_sd <= 2.0 else 'highest'} value swept"
                for r in edge.itertuples()) +
            ". This project has already been wrong that way once — the first ripple "
            "sweep stopped at 2.0 SD and reported 2.0 as the answer, which is why "
            "the RMS arm now runs down to 1.0. Read those rows as *not yet "
            "measured* rather than as a detector that lost."))
    else:
        out.append(note("ok", "Every arm's optimum is interior to its swept grid, so "
                              "none of them is the grid running out."))
    out.append("### What was actually swept, per arm\n")
    out.append(table(panels.sweep_grid().rename(columns={
        "n_points": "thresholds tried", "lowest": "lowest SD", "highest": "highest SD"})))
    out.append("*The four arms do not share a grid: each RMS arm was extended when "
               "its first optimum landed on a boundary, and neither line-length arm "
               "was. A comparison across arms is a comparison across grids.*\n")

    out.append("## The whole family: does the feature choice matter?\n")
    family = panels.family_operating_points()
    if family is None or len(family) == 0:
        out.append(note("info", "`data/benchmark/four_detector_sweep.csv` is missing, "
                                "so the four-detector comparison is not shown."))
    else:
        margin = panels.family_margin()
        out.append("""
Two more features were added and swept on the same 20 patients over a common
grid, so that four are comparable under identical downstream processing. Each
row is that detector at **its own** best threshold, because comparing features
at a shared threshold measures the threshold rather than the feature.
""")
        out.append(table(family.rename(columns={
            "threshold_sd": "its best threshold (SD)", "rank_rho": "channel-rank ρ",
            "f1": "F1", "detections": "detections / 60 s"}).round(3)))
        for band_name, m in sorted((margin or {}).items()):
            out.append(f"- **{BANDS.get(band_name, band_name)} — gain from adding two "
                       f"features: {m['margin']:+.3f} ρ** (best of the original pair "
                       f"{m['best_original']:.3f}, best of all four "
                       f"{m['best_overall']:.3f}, `{m['winner']}`)")
        out.append("")
        out.append(note("ok", """
**Both margins are ties.** Two more detectors, each given its own sweep on real
data, bought no measurable improvement in channel ranking over the two already
there. That is a useful thing to have paid for: it says the ceiling here is not
the choice of feature. What is left is the reference standard — the markings are
another detector's validated output, so precision above about 0.6 is not
available — and the unit of analysis, which the Outcome page shows dominates
everything else measured here.
"""))
        out.append("""
**A threshold in robust SD is not portable between features.** Short-time energy
is RMS before the square root, so under a *fixed* threshold the two would select
identical samples. They do not, because `median + 5 robustSD` sits at about the
98th percentile of an RMS trace and the 96th of a squared one: squaring is not
affine and stretches the upper tail relative to the median. Run short-time
energy at RMS's fast-ripple threshold and its rank agreement falls from 0.601 to
**0.485**. Each feature needs its own sweep, and the table above is what that
looks like.
""")

    rms = frame[(frame["band"] == "ripple") & (frame["detector"] == "rms")]
    at = {row.threshold_sd: row for row in rms.itertuples()}
    fr = frame[(frame["band"] == "fast_ripple") & (frame["detector"] == "rms")]
    fr_at = {row.threshold_sd: row for row in fr.itertuples()}
    t_r, t_fr = THRESHOLDS["interictal-agreement"], THRESHOLDS["interictal-agreement-fast-ripple"]
    if 5.0 in at and t_r in at and t_fr in fr_at and t_r in fr_at and cohort is not None:
        default, tuned, fr_tuned, fr_wrong = at[5.0], at[t_r], fr_at[t_fr], fr_at[t_r]
        out.append("## What the sweep changed\n")
        out.append(f"""
**The shipped default was wrong for interictal work, and now that is measured.**
At 5.0 SD — Staba's published value, which this pipeline inherited — the ripple
detector finds {default.recall:.0%} of the expert-marked events and ranks
channels at ρ = {default.rank_rho:.2f}. At {t_r:g} SD it finds {tuned.recall:.0%}
and ranks at ρ = {tuned.rank_rho:.2f}. The default stays at 5.0 so that published
ictal results do not silently change; `--threshold interictal-agreement` selects
the measured one.

**The two bands want different operating points, by a factor of {t_fr / t_r:g}.**
Ripples rank best at {t_r:g} SD, fast ripples at {t_fr:g} SD. Running the ripple
value in the fast-ripple band drops precision from {fr_tuned.precision:.2f} to
**{fr_wrong.precision:.2f}** — fewer than one detection in ten is an
expert-marked event, at a mean of {fr_wrong.detections:,.0f} detections per 60 s
against {cohort['expert_fast_ripples'].mean():.0f} marked ones. One threshold
for both bands is a bug, not a simplification.

**Channel ranking is the number that matters.** Nobody operates on an event;
they operate on tissue. ρ is the agreement between our per-channel rates and the
experts', and it peaks at a different threshold from F1 — which is why both are
reported instead of one headline.
""")
    out.append(note("info", "Reproduce the sweep: `python -m onset_hfo.cli benchmark`, "
                            "about an hour after the first fetch. `data/benchmark/` is "
                            "the committed extract this page reads. Full tables and the "
                            "reasoning: `docs/EVALUATION.md`."))
    out.append(_your_sweep_section(yours, your_settings, band))
    return "\n".join(out)


def _your_sweep_section(yours, settings: dict, band: str) -> str:
    """Your re-run of the sweep, labelled, beside the published one."""
    if yours is None or not len(yours):
        return ""
    out = ["## Your re-run — not the published study\n"]
    window = settings.get("window_s", [0, 60])
    out.append(note("warn",
        f"Run on this machine on {settings.get('when', '?')} with your settings: "
        f"{', '.join(settings.get('detectors', []))} at "
        f"{', '.join(f'{t:g}' for t in settings.get('thresholds', []))} SD, "
        f"{window[0]:g}–{window[1]:g} s, on {settings.get('n_subjects', 0)} of "
        f"{len(settings.get('subjects', []))} patients. Means over the patients that "
        "ran; not reviewed, not published."))
    skipped = settings.get("skipped") or {}
    shown = yours[yours["band"] == band] if band in set(yours["band"]) else yours
    out.append(table(shown.round(3), rename={
        "threshold_sd": "threshold (SD)", "f1": "F1", "rank_rho": "channel-rank ρ",
        "detections": "detections / 60 s", "top5_overlap": "top-5 shared",
        "n_subjects": "patients"}))
    if skipped:
        out.append("Skipped: " + "; ".join(f"{s} ({why})" for s, why in skipped.items())
                   + "\n")
    return "\n".join(out)


def _outcome(session=None, **_) -> str:
    panels = _panels()
    if panels is None:
        return _missing("Surgical outcome, and whether any of it is stable")
    import pandas as pd

    groups_path = Path(panels.STUDIES) / "outcome_groups_300s.csv"
    groups = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame()
    out = ["# Surgical outcome, and whether any of it is stable\n",
           "*20 patients of ds003498 · whole 300 s runs · expert markings and our "
           "detector on the same channels · outcome from participants.tsv*\n", """
Every other number in this project compares an algorithm to another algorithm.
This one compares it to **what happened to the patient after surgery** — the
only reference standard in epilepsy surgery that is not another opinion.
""", "## Was the busiest fast-ripple channel inside the resection?\n"]
    if len(groups):
        from onset_review import studycharts, thisrecording, yourstudy

        cohort = panels.cohort_overview()
        view_groups = groups.query("scope == 'reviewed' and band == 'fast_ripple'")
        highlight = session.request.subject if thisrecording.in_cohort(session) else None
        mine = yourstudy.cohort_frame()
        mine = mine if len(mine) else None
        out.append(chart(f"outcome-patients-{_stamp(highlight, None if mine is None else mine.to_dict())}",
                         lambda path: studycharts.outcome_chart(cohort, view_groups, path,
                                                                highlight=highlight,
                                                                yours=mine),
                         "Every patient: the share of the tied busiest channels inside "
                         "the resection, by outcome"))
        if highlight or mine is not None:
            out.append("*" + " ".join(part for part in (
                f"The ringed dot is {highlight}, open now." if highlight else "",
                "Hollow squares on the detector's panel are your cohort, below; they are "
                "not in the published groups or their AUC." if mine is not None else "")
                if part) + "*\n")
        out.append("*Each dot is one patient; the bar is the group's mean. The AUC "
                   "is how often a seizure-free patient sits to the right of a "
                   "recurrence.*\n")
        view = groups.query(
            "scope == 'reviewed' and band == 'fast_ripple' and "
            "metric in ['top_channel_resected', 'candidates_resected']")
        view = view.assign(metric=view["metric"].map(
            {"top_channel_resected": "busiest channel resected",
             "candidates_resected": "tied set resected"}).fillna(view["metric"]),
            source=view["source"].map({"expert": "expert markings",
                                       "rms": "our RMS detector"}).fillna(view["source"]))
        out.append(table(view.round(3), columns=[
            "source", "metric", "mean_seizure_free", "mean_recurrence", "auc",
            "auc_lo", "auc_hi", "p_permutation"],
            rename={"mean_seizure_free": "seizure-free, mean", "mean_recurrence": "recurrence, mean",
                    "auc": "AUC", "auc_lo": "AUC low", "auc_hi": "AUC high",
                    "p_permutation": "p (permutation)"}))
    else:
        out.append(note("info", "`data/stability/outcome_groups_300s.csv` is missing."))
    out.append("""
The direction is the one [Fedele et al. 2017](https://www.nature.com/articles/s41598-017-13064-1)
predicts — the study this dataset comes from — and with **13 seizure-free
patients against 7 recurrences it does not reach significance**. Nothing in the
whole study survives correction for its 36 comparisons, and
`min_detectable_auc(13, 7)` is **0.85**, so this cohort could not have
established either number.

**Concentration localises; proportion does not.** The share of a patient's HFOs
inside the resection shows nothing even for the experts (AUC 0.48 — a coin
flip), while *which channel is busiest* on the same events gives 0.71. "Most of
this patient's HFOs were inside the resection" is largely a statement about how
big the resection was.

**And the pipeline stopped picking a winner.** The data picks a single channel
in fewer than half the patients — worst case, 24 tied channels out of 37 — so
`candidates_resected` reports the share of the *tied set* that was removed.
Being honest about ties costs 0.017 AUC.
""")
    out.append("## Does the window matter? The result that did not hold up\n")
    figure = _image("window_stability.png")
    if figure is not None:
        out.append(f"![Window stability]({figure.as_posix()})\n")
    out.append("""
The first version of this study used the **first 60 seconds** of each recording
and reported AUC **0.82, p = 0.007**. That number reached this project's README
and its landing page. Re-run on the whole 300-second run, the same code and the
same pre-specified metric give **0.71, p = 0.12**.

It was not cherry-picked: 60 s was chosen for download size before any outcome
data was touched, and the metric and band were pre-specified. It is something
more ordinary and easier to miss — **an analysis window short enough to change
the conclusion, never checked.**

Across five *disjoint* minutes of the same recordings the expert AUC spans
0.566–0.819 and its p-value **0.007 to 0.613**. The published 0.007 was the best
of five minutes.

One thing the study was not designed to find: **our detector is more stable
across windows than the expert markings** — AUC spread 0.09 against 0.25, same
busiest channel in 12/20 patients against 7/20. Read it as reproducibility, not
accuracy.
""")
    out.append("## Does the night matter? A whole run is a stable unit; a minute is not\n")
    figure = _image("run_stability.png")
    if figure is not None:
        out.append(f"![Run stability]({figure.as_posix()})\n")
    out.append(table(pd.DataFrame([
        ("expert", "9/20", "18/20"), ("our RMS detector", "16/20", "16/20")],
        columns=["source", "same answer across 5 minutes of one run",
                 "same answer across 5 runs (different nights)"])))
    out.append("""
The archive holds **385 runs** across the 20 subjects — 1 to 39 each — because a
night contributes several five-minute segments. The expert markings **doubled
their agreement with themselves** once the unit of analysis became a whole run
instead of a minute of one. Pooling a patient's runs also resolves the ties:
median candidate set 2 → 1, worst case 24 channels → 5.

**So the quantity this pipeline measures is a property of the patient, not of
the recording session.** That is the precondition for anything clinical, and it
now holds. Whether it *predicts outcome* is what twenty patients cannot settle.
""")
    out.append("## Does splitting the ripples help?\n")
    out.append("""
Every rate on this page merges two populations that mean opposite things: a
ripple in healthy cortex is not a finding, and nothing in this pipeline can tell
it from an epileptic one. The pipeline does the available half — it splits
events by whether they ride an interictal discharge — and **measures whether the
split is worth anything**: does either population predict the resected zone
better than the merged rate? Screened across all 20 patients, five 60 s windows
each, pooled by subject mean.
""")
    split = panels.subpopulation_groups("ripple")
    if split is None or len(split) == 0:
        out.append(note("info", "`data/outcome/subpopulation_groups.csv` is missing."))
    else:
        out.append(table(split.round(3), columns=[
            "metric", "population", "n_SF", "n_rec", "auc", "p", "p_bonferroni"],
            rename={"n_SF": "n seizure-free", "n_rec": "n recurrence", "auc": "AUC",
                    "p_bonferroni": "Bonferroni"}))
        gains = []
        for metric, part in split.groupby("metric", sort=False):
            merged = float(part[part.population == "merged"].auc.iloc[0])
            best = float(part[part.population != "merged"].auc.max())
            gains.append(f"{best - merged:+.3f} on `{metric}`")
        out.append(note("bad",
            "**It does not help.** Every sub-population arm is at or below the "
            "merged rate: best gains of " + ", ".join(gains) + ". The spike-coupled "
            "arm sits at chance on all three."))
        meas = panels.subpopulation_measurability()
        if meas is not None and len(meas):
            out.append("*Why the ripple band and not the pre-specified fast-ripple "
                       "one: the spike-coupled population in the fast-ripple band is "
                       "too small to rank channels with.*\n")
            out.append(table(meas.round(1), rename={
                "median": "median events / window", "q1": "lower quartile",
                "maximum": "max", "windows_under_5": "windows with <5 events"}))
    out.append("## What it cannot support\n")
    out.append(note("bad", """
**Nothing here survives correction.** One of the study's 36 comparisons reaches
p < 0.05 uncorrected — our detector, *ripple* band, `candidates_resected`, AUC
0.786, p = 0.034, **Bonferroni 1.00**. In 36 uncorrected tests chance alone
produces about two such rows, so one is *fewer* than expected.
"""))
    out.append("""
- **Thirteen against seven is a very small study.** Only a very large separation
  (AUC ≥ 0.85) could reach 80% power. A p above 0.05 here means *underpowered*,
  not *no effect*.
- **Thirty-six comparisons, uncorrected.** Nothing survives the Bonferroni column.
- **The published headline changed once already** (0.82 → 0.71). Treat that as
  the calibration for how much weight any single number here can carry.
- **Resected contacts that were never recorded cannot be scored.** In five
  temporal-lobe patients only 4 of 16 appear in the recording.
- **Retrospective, one centre, one surgical team.** The randomised
  [HFO Trial](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(22)00311-8/fulltext)
  (Lancet Neurology 2022) tested HFO-guided resection prospectively and did not
  find the benefit retrospective series report. This is a retrospective series.
""")
    out.append(note("info", "Full design, every table, and the list of what this cannot "
                            "support: `docs/OUTCOME.md`."))
    out.append(_your_outcome_section())
    out.append(_your_cohort_section())
    return "\n".join(out)


def _your_outcome_section() -> str:
    """Your re-run of the Outcome study, labelled, if there is one."""
    from onset_review import studycharts, yourstudy

    cohort, settings = yourstudy.load_your_outcome()
    if cohort is None or not len(cohort):
        return ""
    window = settings.get("window_s", [0, 300])
    out = ["## Your re-run — not the published study\n",
           note("warn", f"Run on this machine on {settings.get('when', '?')}: detector "
                        f"{settings.get('detector')}, {settings.get('band', '').replace('_', ' ')} "
                        f"band, threshold {settings.get('threshold_sd') or 'its measured default'}"
                        f", {window[0]:g}–{window[1]:g} s, on {len(cohort)} of "
                        f"{len(settings.get('subjects', []))} patients. Not reviewed, not "
                        "published.")]
    groups = settings.get("groups")
    view = None
    if groups is not None and len(groups):
        view = groups.query("scope == 'reviewed'")
        detector = settings.get("detector", "rms")
        view = view.assign(source=view["source"].replace({detector: "rms"}))
    out.append(chart(f"your-outcome-{_stamp(settings.get('when'), len(cohort))}",
                     lambda path: studycharts.outcome_chart(cohort, view, path),
                     "Your re-run: every patient by outcome"))
    if view is not None and len(view):
        rows = view[view["metric"] == "candidates_resected"]
        out.append(table(rows.round(3), columns=[
            "source", "mean_seizure_free", "mean_recurrence", "auc", "auc_lo", "auc_hi",
            "p_permutation"], rename={"mean_seizure_free": "seizure-free, mean",
                                      "mean_recurrence": "recurrence, mean", "auc": "AUC",
                                      "auc_lo": "AUC low", "auc_hi": "AUC high",
                                      "p_permutation": "p (permutation)"}))
    skipped = settings.get("skipped") or {}
    if skipped:
        out.append("Skipped: " + "; ".join(f"{s} ({why})" for s, why in skipped.items()) + "\n")
    return "\n".join(out)


def _your_cohort_section() -> str:
    """The reader's own patients, measured as the study measures one."""
    from onset_review import yourstudy

    frame = yourstudy.cohort_frame()
    out = ["## Your cohort\n"]
    if not len(frame):
        out.append("Your own patients can be measured here the way the study measures "
                   "one: open a recording, then *Add this recording to your cohort* "
                   "above, with the contacts that were resected and the outcome. Nothing "
                   "is added without you.\n")
        return "\n".join(out)
    out.append(note("warn", "Your patients, your resections, your outcomes, measured on "
                            "the windows you added. Not validated, not reviewed; a "
                            "handful of patients cannot establish an effect either way."))
    out.append(table(frame.drop(columns=["id"]).round(2), rename={
        "subject": "patient", "seizure_free": "seizure-free",
        "candidates_resected_rms": "tied set resected", "n_candidates": "tied set size",
        "top_channel_resected": "busiest resected"}))
    result = yourstudy.compare_cohort(frame)
    n1, n2 = result["n_seizure_free"], result["n_recurrence"]
    if n1 >= 2 and n2 >= 2:
        floor = result["floor"]
        reach = (f"The smallest AUC groups this size could detect with 80% power is "
                 f"**{floor:.2f}**." if floor == floor else
                 "Groups this size cannot reach 80% power for any effect, however large: "
                 "no result here could be significant.")
        out.append(f"Seizure-free {n1}, recurrence {n2}: AUC **{result['auc']:.2f}** "
                   f"({result['auc_lo']:.2f}–{result['auc_hi']:.2f}), permutation p = "
                   f"{result['p_permutation']:.2f}. {reach}\n")
    else:
        out.append(f"Seizure-free {n1}, recurrence {n2}: a comparison needs at least two "
                   "patients in each group.\n")
    if result.get("mixed_settings"):
        out.append(note("warn", "Your patients were measured with different bands or "
                                "detectors; the comparison mixes them."))
    return "\n".join(out)


def _patients(band: str = "fast_ripple", scope: str = "reviewed",
              subject: str = "", **_) -> str:
    panels = _panels()
    if panels is None:
        return _missing("Twenty patients, one at a time")
    import pandas as pd

    cohort = panels.cohort_overview(band=band, scope=scope)
    out = ["# Twenty patients, one at a time\n",
           "*ds003498 · whole 300 s runs · pre-specified arm: fast ripples, reviewed "
           "channels · expert markings and our RMS detector on the same channels*\n"]
    primary = (band, scope) == (panels.PRIMARY["band"], panels.PRIMARY["scope"])
    if not primary:
        out.append(note("warn",
            f"You are looking at **{band} / {scope}**, which is not the arm the study "
            f"pre-specified (fast_ripple / reviewed). Every number on the Outcome "
            f"page comes from the pre-specified arm; these are among the same 36 "
            f"comparisons, none of which survives correction."))
    if cohort is None or len(cohort) == 0:
        out.append(note("bad", "No committed per-subject tables found in `data/outcome/`."))
        return "\n".join(out)

    out.append("## Was the busiest channel inside the tissue that was removed?\n")
    n_sf = int(cohort["seizure_free"].sum())
    counts = []
    for source, label in (("expert", "expert markings"), ("rms", "our RMS detector")):
        hit = cohort[f"top_resected_{source}"] == 1
        counts.append((label, f"{int(hit.sum())}/{len(cohort)}",
                       f"{int((hit & cohort['seizure_free']).sum())}/{n_sf}",
                       f"{int((hit & ~cohort['seizure_free']).sum())}/{len(cohort) - n_sf}",
                       int((cohort[f"n_candidates_{source}"] > 1).sum())))
    out.append(table(pd.DataFrame(counts, columns=[
        "source", "busiest channel was resected", "among seizure-free",
        "among recurrence", "patients with no single busiest channel"])))
    agree = int((cohort["top_resected_expert"] == cohort["top_resected_rms"]).sum())
    provenance = ("*is* the AUC on the Outcome page, written out as patients" if primary
                  else "is one of the study's 36 comparisons; the Outcome page reports "
                       "the pre-specified fast-ripple arm, not this one")
    out.append(f"""
That gap — **{counts[0][2]} of the seizure-free patients against {counts[0][3]}
of the recurrences** — {provenance}. Either way it is a result four patients
wide: move one row and it moves. The two sources reach the same inside/outside
answer in **{agree} of {len(cohort)}** patients: our detector and the experts
mostly disagree about *rates*, and mostly agree about *which channel is busiest*.
""")
    out.append("### Every patient\n")
    show = cohort.rename(columns={
        "epilepsy": "type", "ilae": "ILAE", "rz_coverage": "resection recorded",
        "n_reviewed": "reviewed ch", "n_resected_channels": "resected ch",
        "top_resected_expert": "resected? (expert)", "top_resected_rms": "resected? (RMS)",
        "n_candidates_expert": "ties (expert)", "n_candidates_rms": "ties (RMS)",
        "share_in_rz_expert": "share inside (expert)", "share_in_rz_rms": "share inside (RMS)"})
    out.append(table(show.round(2), columns=[
        "subject", "type", "ILAE", "outcome", "reviewed ch", "resected ch",
        "resection recorded", "resected? (expert)", "ties (expert)", "share inside (expert)",
        "resected? (RMS)", "ties (RMS)", "share inside (RMS)"]))
    out.append("*`outcome`: S = seizure-free (ILAE 1–2), F = recurrence. `resected?` is 1 "
               "when the busiest channel lay inside the resection. `ties` above 1 means "
               "no single busiest channel. `resection recorded` below 1.0: read every "
               "share with care.*\n")
    thin = cohort[cohort["rz_coverage"] < 1.0]["subject"].tolist()
    if thin:
        out.append(note("warn",
            f"**{len(thin)} patients' resections were mostly not recorded** — "
            f"{', '.join(thin)}, at 25% coverage. Their `share inside` describes a "
            f"quarter of the tissue that was removed. They are in the study because "
            f"excluding them after seeing the numbers would be worse."))

    if subject and subject in set(cohort["subject"]):
        row = cohort[cohort["subject"] == subject].iloc[0]
        out.append(f"## One patient: {subject}\n")
        out.append(
            f"**Outcome:** {'Seizure-free' if row['seizure_free'] else 'Recurrence'} "
            f"(ILAE {int(row['ilae'])}, {int(row['months_follow_up'])} months of "
            f"follow-up) · **Epilepsy:** {row['epilepsy']}, "
            f"{'lesion on imaging' if row['lesion'] == 1 else 'no lesion found'} · "
            f"**Channels:** {int(row['n_reviewed'])} reviewed, "
            f"{int(row['n_resected_channels'])} inside the resection, "
            f"{int(row['n_channels'])} recorded in all.\n")

        def word(value, when_true, when_false):
            return "—" if pd.isna(value) else (when_true if value else when_false)

        def percent(value):
            return "—" if pd.isna(value) else f"{value:.0%}"

        answers = []
        for source, label in (("expert", "expert markings"), ("rms", "our RMS detector")):
            ties = row.get(f"n_candidates_{source}")
            answers.append((
                label,
                word(row.get(f"top_resected_{source}"), "inside the resection", "outside it"),
                "—" if pd.isna(ties) else ("a single channel" if ties == 1
                                           else f"{int(ties)} tied channels"),
                percent(row.get(f"candidates_resected_{source}")),
                percent(row.get(f"share_in_rz_{source}")),
                word(row.get(f"stable_across_windows_{source}"), "same answer", "changed"),
                word(row.get(f"stable_across_runs_{source}"), "same answer", "changed")))
        out.append("### What each source said about this patient\n")
        out.append(table(pd.DataFrame(answers, columns=[
            "source", "busiest channel was", "and it was", "of the tied set resected",
            "share of events inside", "across 5 minutes of this run",
            "across this patient's other nights"])))
        out.append("### Before reading anything above as a finding\n")
        caveats = panels.subject_caveats(subject, band=band, scope=scope)
        if caveats:
            for item in caveats:
                out.append(note("warn", item))
        else:
            out.append(note("info",
                "None of this screen's four per-patient checks fired for this "
                "patient: their resection was fully recorded, both sources picked a "
                "single busiest channel, and both gave the same answer across the five "
                "minutes of this run and across their other nights. It is still one "
                "patient in a study of twenty that is underpowered for all of them."))
        out.append("### The channels underneath it\n")
        rows = panels.subject_channels(subject, band=band)
        if rows is None or len(rows) == 0:
            out.append(note("info", "No committed channel table for this patient."))
        else:
            shown = rows.rename(columns={"expert_events": "expert events",
                                         "rms_events": "RMS events"})
            shown["eloquent"] = shown["eloquent"].map({True: "eloquent", False: ""})
            out.append(table(shown, columns=["channel", "zone", "eloquent",
                                             "expert events", "RMS events"]))
            out.append(f"*{len(rows)} reviewed channels, busiest first by expert count. "
                       f"A **partial** channel is the one to look at hardest: half of it "
                       f"was removed, so it counts as neither in nor out.*\n")

    out.append("## What this screen is not\n")
    out.append(note("bad", """
**This screen cannot tell you anything about a patient.** It is a per-patient
*view of a group result*, and the group result does not reach significance:
13 seizure-free against 7 recurrences, `min_detectable_auc(13, 7) = 0.85`, and
nothing among the study's 36 comparisons survives Bonferroni correction.
"""))
    out.append("""
- **No confidence interval on a single patient's answer.** `top_channel_resected`
  is one bit. The interval is on the cohort, and it spans a coin flip.
- **No prediction.** Nothing here was fitted, so nothing can be applied to a new
  patient.
- **Not the tissue the surgeon should have removed.** The resection is the input,
  not the output.
- **Not why a patient's answer changed between windows or nights.** The stability
  columns record *that* it changed.

The caveats above are computed from the committed tables, not written by hand:
resection coverage, tie sets, window stability and run stability, per patient.
Design and every group table: `docs/OUTCOME.md`; the tables behind this page:
`data/outcome/README.md`.
""")
    return "\n".join(out)


def _data(**_) -> str:
    import pandas as pd

    out = ["# The data — real recordings, real patients\n", """
No simulator behind these pages. Two public, CC0, BIDS-formatted archives of
intracranial recordings from people who had epilepsy surgery.
"""]
    out.append(table(pd.DataFrame([
        ("ds003498", "Zurich interictal slow-wave sleep", "20", "2000 Hz",
         "expert HFO markings · resected contacts · surgical outcome",
         "everything measured in this project"),
        ("ds003029", "Epilepsy-iEEG multicentre", "35+", "250–1000 Hz",
         "clinician seizure markers · curated SOZ contacts",
         "the ictal quickstart and the learned per-contact model")],
        columns=["archive", "what it is", "subjects", "sampling", "what it carries",
                 "used for"])))
    out.append("""
## Why ds003498 is the one that matters

Almost no public iEEG dataset carries all three of the things an outcome study
needs. This one does:

| Ingredient | Where it lives | Why it is there |
|---|---|---|
| **Expert HFO markings** per channel | `*_events.tsv` | turns "we detected 40 ripples" into a score against a human-validated reference |
| **The resected contacts** | `sourcedata/clinical_ch_sheet_zurich.xlsx` | free-text ranges like `"ahr1-4, ar1-4"`, parsed by `onset_hfo/clinical.py` |
| **Surgical outcome** | `participants.tsv` | `S` seizure-free (13) / `F` recurrence (7), ILAE class, 10–46 months follow-up |
| 2000 Hz sampling | — | fast ripples (250–500 Hz) are analysable at all; the other archive is 1000 Hz and cannot support them |
| Interictal slow-wave sleep | — | the setting the clinical HFO literature actually uses |

Two things to know before trusting a number from it:

- **Recorded channels are a subset of implanted contacts.** In 15 of 20 subjects
  every resected contact appears in the recording; in the other five — all
  temporal-lobe cases, where the study kept only the three most mesial bipolar
  channels — only 4 of 16 do. `rz_coverage` reports this per subject.
- **The clinical sheet has a typo.** Subject 15's excluded list reads `1ll22-24`
  where every other token on the row says `tll`. The parser surfaces unparsable
  chunks rather than quietly returning a smaller resection.

## How 24 MB is downloaded instead of 105 MB

BrainVision stores samples **multiplexed** — channel 1 at time 1, channel 2 at
time 1, …, then time 2 — with no per-sample header. So sample *k* begins at byte
`k × n_channels × bytes_per_sample`, which makes **a byte range a time range**.

`onset_hfo.datasets.fetch_slice` downloads the three small text sidecars in
full, computes the byte offsets for the window you asked for, and issues one
HTTP Range request. A notebook starts in seconds instead of minutes, and a
laptop can work on an archive it could never hold. The same trick is what makes
the 92-run stability study affordable: each slice is fetched, analysed and
deleted, so peak disk stays flat against an archive that is 46 GB in total.

## Licence and citation

Both archives are **CC0**. Work using them is asked to cite:

- Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T, Sarnthein J.
  *Resection of high frequency oscillations predicts seizure outcome in the
  individual patient.* Sci Rep 7:13836 (2017). doi:10.1038/s41598-017-13064-1 —
  OpenNeuro ds003498, BIDS conversion by A. Li (mne-hfo).
- Li A, Inati S, Zaghloul K, *et al.* *Epilepsy-iEEG-Multicenter-Dataset*,
  OpenNeuro (2021), doi:10.18112/openneuro.ds003029.v1.0.3 — and
  *Neural fragility as an EEG marker of the seizure onset zone*, doi:10.1101/862797.

**Your own recordings.** De-identification, ethics approval and data governance
are entirely your responsibility — see `docs/DATA.md`. This release has no
security model, no authentication and no audit log, and must not be pointed at
identifiable data.
""")
    return "\n".join(out)


def _architecture(**_) -> str:
    return """# Architecture: the full system and what this release includes

```
Public archive (OpenNeuro, BIDS, byte-range slices) ─┐
Simulator (labelled events + the traps) ─────────────┴─► Preprocess ─► Detectors ─► Artifact rejection ─► Measure ─► artifacts/results/
   high-pass · notch · bipolar          RMS · line length      spectral prominence     rates · Poisson CIs     events · rates · report
                                        envelope · energy      over 1/f                agreement               (immutable)
                                        spikes                                                                       │
                                                                     ┌───────────────────────────────────────────────┼────────────────┐
                                                                     ▼                                               ▼                ▼
                                                              Agent: 8 read-only tools,                      Reading interface    Cohort studies
                                                              cites or refuses ─────────────────────────────► shows, decides      benchmark · outcome
                                                                                                              nothing              stability
```

Both readers — the agent and the interface — go through the same read-only
`ResultStore`. Neither can change a number. That is the whole reason it exists:
a second reader is where a project usually grows a second source of truth, and
here it structurally cannot.

## Full system versus this release

| Component | Full system | This release |
|---|---|---|
| Data | BIDS ingest, de-identification, Postgres | two public archives, byte-range slices, on disk |
| Preprocessing | versioned, manifested | same definitions, config stamped into every result |
| Detectors | registry, contracts, learned models | two classical detectors behind one shared engine + a spike detector |
| Artifact rejection | same, plus muscle/stimulation | spectral prominence over 1/f, cycle count — the ablation is published |
| Evaluation | prospective, multi-centre | agreement with expert markings on 20 patients; accuracy on synthetic truth |
| Outcome | prospective trial | retrospective, 20 patients, nothing significant — and the window study that corrected it |
| Predictions | immutable rows with evidence windows | immutable CSV/JSON with evidence windows |
| Report | local LLM writes, validator checks every citation | deterministic template, same schema, same rules, no recommendation field |
| Agent | local open-weight model over read-only tools | the same, plus a scripted backend so the loop runs with no model at all |
| Interface | clinician UI, roles, audit | this window and the site: shows, decides nothing |
| Security | roles, audit middleware, row-level security, on-prem | **not included** — single user, no PHI |
| Operations | containers, traces, budgets, nightly chain | **not included** |

## The contracts that hold it together

1. **Every score carries the window it looked at.** An event's evidence id is
   `subject|channel|detector|start`, and it resolves or the answer is discarded.
2. **The agent cannot compute.** Eight read-only tools over one saved analysis.
   No tool runs a detector, writes a file, or takes another patient's id.
3. **Scope refusals happen before the model runs.** Treatment and diagnosis
   questions never reach a model that could be persuaded.
4. **Disagreement is reported, not averaged away.** Both detector ranks are
   shown; neither is preferred.
5. **There is no recommendation field.** Not empty — absent from the schema.

*This release keeps every rule that matters to a clinician — cited evidence,
disagreements stated, uncertainty shown, no recommendation — and leaves out
everything that only matters once real users and identifiable data exist.*
"""


def _research(**_) -> str:
    panels = _panels()
    out = ["# Research programme\n", """
**Where this sits.** Onset-HFO is the signal-processing and evidence half of the
[Onset](https://berdakh.github.io/onset/) project, built on **public recordings
of real patients** rather than a simulator, so that every claim can be checked
against a reference someone else published.

## What is measured, honestly

| Claim | Status |
|---|---|
| Detects HFOs in real interictal recordings | **Yes** — agreement with expert markings, ρ ≈ 0.66 on channel ranking |
| Beats a published detector | **No, and not claimed.** The reference *is* a published detector; these are agreement numbers |
| The map relates to surgical outcome | **Direction yes, significance no.** AUC 0.71, p = 0.12 on 13 vs 7 patients; nothing in the study survives correction for its 36 comparisons |
| The measurement is stable | **Across nights yes** (18/20 patients), **within a minute no** — a 60 s window changed the headline once |
| Distinguishes epileptic from physiological ripples | **No.** Nothing in the pipeline tries. The one available proxy — whether a ripple rides a discharge — was screened against outcome and does **not** localise better than the merged rate |
| The choice of detector is the limiting factor | **No, measured.** Four features, each swept on real data, rank channels within 0.003 ρ of one another |
| The montage is the limiting factor | **No, measured.** Bipolar beats the archive's reference and a common average on both yardsticks, in both bands |
| Validated for clinical use | **No.** Not a medical device |

## Two results this project reversed on itself
"""]
    out.append(note("info", """
**The 60-second headline.** An outcome result of AUC 0.82, p = 0.007 reached the
README and the landing page before the study was re-run on whole recordings,
where it became 0.71, p = 0.12. Corrected everywhere, with both tables kept side
by side rather than one quietly replacing the other.
**The detector gap.** That same 60-second run appeared to show our detector
trailing the experts, which made "add a morphology criterion" look like the top
priority. On whole runs the gap largely closed — because the *expert* number
came down, not because ours went up — so the priority changed to ranking
stability instead.
"""))
    out.append("""
## What is open, in order

1. **A real model driving the analysis ladder.** Every orchestration number so
   far comes from a scripted planner, so it measures the script rather than the
   thesis. Needs a GPU afternoon; the harness is built and tested.
2. **A second cohort.** Everything rests on one centre, one annotation protocol,
   one surgical team. Needs an archive with HFO markings *and* resection *and*
   outcome — ds003498 is the only public one known to have all three.
3. **A hand-annotated benchmark.** The only way to turn *agreement* into
   *accuracy*: a few hundred expert-marked events on real data, reviewed here.
4. **Electrode geometry.** Bipolar pairs are formed by contact number, which is
   wrong where a grid row wraps. Needs an archive that publishes coordinates.
5. **Physiological versus epileptic ripples** remains the biggest scientific
   gap, but it is no longer an experiment this project can run: the only proxy
   available in public data has now been measured and does not localise better
   than the merged rate. Progress needs a labelled archive, not more analysis.

## Four questions that came off this list by being answered
""")
    robustness = panels.robustness_groups() if panels is not None else None
    montage = panels.montage_groups() if panels is not None else None
    fr = ""
    if robustness is not None and len(robustness):
        arm = robustness[(robustness.band == "fast_ripple")
                         & (robustness.metric == "top_channel_resected")]
        plain = arm[arm.rule == "plain"].auc
        mult = arm[arm.rule == "multiplied"].auc
        if len(plain) and len(mult):
            fr = (f" — {float(plain.iloc[0]):.3f} plain against "
                  f"{float(mult.min()):.3f}–{float(mult.max()):.3f} multiplied")
    mont = ""
    if montage is not None and len(montage):
        rho = montage[(montage.kind == "agreement") & (montage.metric == "spearman_rho")]
        if len(rho):
            cells = {(r.band, r.montage): r.value for r in rho.itertuples()}
            mont = (f" Cohort-mean Spearman with the expert ranking: "
                    f"{cells.get(('ripple', 'bipolar'), float('nan')):.3f} bipolar against "
                    f"{cells.get(('ripple', 'referential'), float('nan')):.3f} referential and "
                    f"{cells.get(('ripple', 'average'), float('nan')):.3f} average in the "
                    f"ripple band.")
    out.append(note("ok", f"""
**More than two detectors.** Four features were implemented and swept on real
data; every pairwise Jaccard overlap sits between 0.47 and 0.79, and the two
added features bought 0.003 ρ in each band, so the choice of feature is not
where the headroom is.
**Do the ripple sub-populations localise differently?** No. Splitting events by
discharge coupling leaves every arm at or below the merged rate.
**Is the re-planning ranking rule the right rule?** Conditionally: it helps in
the ripple band at every re-test point and hurts in the fast-ripple band{fr},
now measured on every patient after the rule stopped firing on silent re-tests.
**Does the bipolar montage help or hurt for rate?** It helps, on both yardsticks
and in both bands; the archive's own reference is last on all eight outcome
cells.{mont}
"""))
    out.append("""
## Principles inherited from the Onset project

1. No patient on both sides of a split; normalisation fit on training patients only.
2. Every score comes with the window it looked at, or an empty evidence list and
   a note saying so.
3. Disagreement between methods is reported, not averaged away.
4. Nothing in the system recommends treatment; the clinician decides.
5. Public data first; partner-centre data only through de-identification.
6. **When a number changes, the old one stays visible next to the new one.**

*Code: github.com/berdakh/onset-hfo · MIT licence. The full programme, with the
lab and the people behind it, is on the results site.*
""")
    return "\n".join(out)
