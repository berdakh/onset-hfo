"""The study pages for the clinic edition's two measured methods. No Qt.

* **Ictal onset** -- does the Epileptogenicity Index find the clinicians'
  onset zone? ds003029 (`docs/ICTAL.md`, ``data/ictal/``), its pre-seizure
  control, and the replication on a second archive, HUP ds004100
  (``data/hup/``), when it is there.
* **Template map** -- how far a straight-line plan lands from real implants
  on the template, and what an atlas label is worth (`docs/TEMPLATE_MAP.md`,
  ``data/template/``).

Built as `onset_review.studies` builds the rest: every number from the
committed tables, a key message whose numbers are on the page, a chart that
is a bonus and never a failure.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["hup_interictal_section", "ictal_page", "template_page", "key_ictal", "key_template"]


def _root() -> Path | None:
    from onset_review import studies

    panels = studies._panels()
    return Path(panels.DATA_ROOT) if panels is not None else None


def _json(*parts) -> dict | None:
    root = _root()
    path = root.joinpath(*parts) if root is not None else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _csv(*parts):
    import pandas as pd

    root = _root()
    path = root.joinpath(*parts) if root is not None else None
    return pd.read_csv(path) if path is not None and path.exists() else None


def _interval(block: dict) -> str:
    return f"{block['median']:.2f} ({block['ci_low']:.2f}–{block['ci_high']:.2f})"


# -- Ictal onset ----------------------------------------------------------------------------
def key_ictal(**_) -> dict:
    main = _json("ictal", "summary.json")
    control = _json("ictal", "control_summary.json")
    if main is None:
        from onset_review.studies import _ABSENT

        return _ABSENT
    block = main["ei_all"]
    top = round(main["top_in_zone_patients"] * main["patients"])
    lead = (f"On {main['patients']} patients the index ranks the clinicians' onset-zone "
            f"channels above the rest: median AUC {block['median']:.2f}, interval "
            f"{block['ci_low']:.2f}–{block['ci_high']:.2f}. Moved 40 s before the seizure "
            f"it falls to {control['ei_per_seizure']['median']:.2f}, so the agreement comes "
            f"from the seizure. Its single top channel was in the zone in {top} of "
            f"{main['patients']}, and the plain energy ratio does almost as well.")
    tiles = [(f"{block['median']:.2f}", "AUC, onset zone vs the rest"),
             (f"{control['ei_per_seizure']['median']:.2f}", "40 s before the seizure"),
             (f"{top}/{main['patients']}", "top channel in the zone")]
    hup = _json("hup", "ictal", "summary.json")
    if hup is not None and hup.get("patients"):
        tiles.append((f"{hup['ei_all']['median']:.2f}", "AUC on a second archive (HUP)"))
    return {"lead": lead, "tiles": tiles}


def _strip_chart(groups: list[tuple[str, list[float]]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from onset_hfo.ictal_study import bootstrap_median
    from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

    figure = Figure(figsize=(6.4, 3.6), dpi=130)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    _quiet(axis)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    rng = np.random.default_rng(0)
    colours = [SERIES[0], SERIES[1], MUTED, SERIES[0]]
    for i, (_label, values) in enumerate(groups):
        values = [v for v in values if np.isfinite(v)]
        axis.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=16,
                     color=colours[i % len(colours)], alpha=0.75, linewidths=0)
        median, low, high = bootstrap_median(values)
        axis.plot([i + 0.25, i + 0.25], [low, high], color="black", linewidth=1.3)
        axis.plot([i + 0.18, i + 0.32], [median, median], color="black", linewidth=2.0)
        axis.text(i + 0.36, median, f"{median:.2f}\nn={len(values)}", fontsize=7,
                  va="center")
    axis.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.0)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    axis.set_ylim(0, 1.02)
    axis.set_xlim(-0.5, len(groups) - 0.2)
    axis.set_ylabel("AUC per patient, onset zone vs the rest", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    return path


def ictal_page(**_) -> str:
    import pandas as pd

    from onset_review.studies import _missing, _stamp, chart, note, table

    main = _json("ictal", "summary.json")
    if main is None:
        return _missing("Ictal onset")
    control = _json("ictal", "control_summary.json")
    patients = _csv("ictal", "per_patient.csv")
    control_patients = _csv("ictal", "control_per_patient.csv")
    hup = _json("hup", "ictal", "summary.json")
    hup_patients = _csv("hup", "ictal", "per_patient.csv")
    hup_control = _json("hup", "ictal", "control_summary.json")
    hup_control_patients = _csv("hup", "ictal", "control_per_patient.csv")
    out = ["# Ictal onset: does the index find the clinicians' zone?\n", """
The case window's **Ictal onset** step ranks channels by the
**Epileptogenicity Index** (Bartolomei, Chauvel & Wendling, *Brain* 2008): how
much the fast-over-slow energy ratio rises after the marked onset, and how soon
after the first channel's change. Here it is scored, channel by channel,
against the onset zone clinicians marked. The settings are the method's own,
untuned: 12.4–97 Hz over 3.5–12.4 Hz in 1 s windows, a Page–Hinkley change
test, 5 s of energy, and a time constant of 1 s.
"""]
    rows = [("ds003029, at the seizure", main["ei_all"]["n"], _interval(main["ei_all"]),
             f"{main['ei_all']['share_above_half']:.0%}"),
            ("energy ratio alone, same seizures", main["er_all"]["n"],
             _interval(main["er_all"]), f"{main['er_all']['share_above_half']:.0%}"),
            ("seizure-free patients only", main["ei_seizure_free"]["n"],
             _interval(main["ei_seizure_free"]),
             f"{main['ei_seizure_free']['share_above_half']:.0%}"),
            ("40 s before the seizure (control)", control["ei_all"]["n"],
             _interval(control["ei_all"]), f"{control['ei_all']['share_above_half']:.0%}")]
    if hup is not None and hup.get("patients"):
        rows.append(("HUP ds004100, at the seizure", hup["ei_all"]["n"],
                     _interval(hup["ei_all"]), f"{hup['ei_all']['share_above_half']:.0%}"))
    if hup_control is not None and hup_control.get("patients"):
        rows.append(("HUP, 40 s before (control)", hup_control["ei_all"]["n"],
                     _interval(hup_control["ei_all"]),
                     f"{hup_control['ei_all']['share_above_half']:.0%}"))
    out.append("## Per patient\n")
    out.append(table(pd.DataFrame(rows, columns=["", "patients", "median AUC (95% interval)",
                                                 "above 0.5"])))
    groups = [("ds003029\nat the seizure", list(patients["auc_ei"])),
              ("energy ratio\nalone", list(patients["auc_er"])),
              ("40 s before\n(control)", list(control_patients["auc_ei"]))]
    if hup_patients is not None and len(hup_patients):
        groups.append(("HUP\nat the seizure", list(hup_patients["auc_ei"])))
    if hup_control_patients is not None and len(hup_control_patients):
        groups.append(("HUP 40 s\nbefore", list(hup_control_patients["auc_ei"])))
    out.append(chart(f"ictal-auc-{_stamp(len(patients), None if hup is None else hup['patients'])}",
                     lambda path: _strip_chart(groups, path),
                     "Per-patient AUC: the index at the seizure, the energy ratio alone, "
                     "the index 40 s before, and on the second archive"))
    top = round(main["top_in_zone_patients"] * main["patients"])
    out.append(note("info", f"""
The control ran on {control['seizures_analysed']} seizures with enough recording
before them. There the per-seizure median is
{control['ei_per_seizure']['median']:.2f}, against
{main['ei_per_seizure']['median']:.2f} at the seizure. The top channel is in the
zone in {top} of {main['patients']} patients, so the index ranks the zone as a
set better than it names one contact."""))
    if hup is not None and hup.get("patients"):
        hup_top = round(hup["top_in_zone_patients"] * hup["patients"])
        out.append("## A second archive\n")
        out.append(f"""
The same index, unchanged, on HUP (ds004100): {hup['seizures_analysed']}
seizures from {hup['patients']} patients, against the onset-zone contacts in the
archive's own channel tables. The median per-patient AUC is
**{_interval(hup['ei_all'])}**, and the top channel is in the zone in
{hup_top} of {hup['patients']}.""" + ("" if hup_control is None else f"""
Moved 40 s before the onset, as on ds003029, it falls to
{_interval(hup_control['ei_all'])}.""") + """ By implant:
""")
        out.append(table(pd.DataFrame(
            [(site, b["n"], _interval(b)) for site, b in hup.get("by_site", {}).items()],
            columns=["implant", "patients", "median AUC (95% interval)"])))
    out.append("## Why it is not ground truth\n")
    out.append("""
The clinicians drew the zone with everything they had, including reading these
seizures by eye, so agreement with them is agreement with a careful reading of
the same signal, not with an outcome. Most of ds003029's onset marks are not
labelled electrographic, and the top channel's change usually came about 5 s
before the mark. The index is only as good as the time it is anchored to. The
full account, with what was left out and why, is in `docs/ICTAL.md`.
""")
    shown = patients[["subject", "site", "seizure_free", "n_seizures", "n_channels",
                      "n_soz", "auc_ei", "auc_er", "top_in_zone"]]
    out.append("## Every patient (ds003029)\n")
    out.append(table(shown))
    return "\n".join(out)


def hup_interictal_section() -> str:
    """The outcome study's question, asked of HUP (ds004100): "" when the
    tables are not there."""
    import pandas as pd

    from onset_review.studies import table

    summary = _json("hup", "interictal", "summary.json")
    patients = _csv("hup", "interictal", "per_patient.csv")
    if summary is None or patients is None:
        return ""
    ripple = summary["bands"].get("ripple")
    if not ripple:
        return ""
    status = patients[patients["band"] == "ripple"]["status"].value_counts()
    rates = ", ".join(f"{n} at {s.replace('not analysable at ', '')}"
                      for s, n in status.items() if s != "ok")
    outcome = ripple["outcome"]
    rows = [(name, block["n_seizure_free"], block["n_recurrence"], f"{block['auc']:.2f}",
             f"{block['auc_lo']:.2f}–{block['auc_hi']:.2f}", f"{block['p_permutation']:.2f}")
            for name, block in (("share of events in the resection", outcome["share_in_rz"]),
                                ("busiest channel resected", outcome["top_channel_resected"]),
                                ("top three resected", outcome["top3_resected"]))]
    return "\n".join([
        "## A second archive (HUP)\n", f"""
The same detector (RMS, at this study's operating point) on HUP's five-minute
interictal recordings (ds004100). Most of them were sampled too slowly for
ripples ({rates}): **{ripple['patients']} patients** were analysable, and none
for fast ripples, which need more than 1000 Hz. Against the onset-zone contacts,
the ripple rate gives a median per-patient AUC of
**{_interval(ripple['auc_soz'])}**. Against outcome, with
{outcome['share_in_rz']['n_seizure_free']} seizure-free patients and
{outcome['share_in_rz']['n_recurrence']} with recurrence, nothing separates:
""", table(pd.DataFrame(rows, columns=["", "seizure-free", "recurrence", "AUC",
                                         "95% interval", "p (permutation)"])), """
On ds003498 nothing separated the outcome groups after correction either, with
13 patients against 7; five against ten is smaller still. What HUP adds is
that the ripple rate ranks the clinicians' onset zone above the rest at all.
"""])


# -- Template map ---------------------------------------------------------------------------
def key_template(**_) -> dict:
    summary = _json("template", "summary.json")
    if summary is None:
        from onset_review.studies import _ABSENT

        return _ABSENT
    own = summary["variants"]["own"]
    pitch = summary["variants"]["3.5mm"]
    labels = summary["real_labels"]["in"]
    lead = (f"On {summary['patients']} patients' real implants, a straight line from the "
            f"deepest contact lands a median {own['median_mm']:.1f} mm off with the shaft's "
            f"own spacing, and {pitch['median_mm']:.1f} mm off at a 3.5 mm catalogue pitch. "
            f"Even at the real positions the atlas names a structure for "
            f"{labels:.0%} of contacts, so every label is a probable one.")
    return {"lead": lead, "tiles": [
        (f"{own['median_mm']:.1f} mm", "median error, own spacing"),
        (f"{own['within_5mm']:.0%}", "within 5 mm"),
        (f"{pitch['median_mm']:.1f} mm", "at a 3.5 mm pitch"),
        (f"{labels:.0%}", "real positions in a named structure"),
    ]}


def _error_chart(contacts, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from onset_review.studycharts import GRID, MUTED, SERIES, _quiet

    figure = Figure(figsize=(6.4, 3.6), dpi=130)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    _quiet(axis)
    axis.grid(True, color=GRID, linewidth=0.8)
    after = contacts[contacts["index"] > 0]
    edges = np.arange(0, 65, 5.0)
    middle = (edges[:-1] + edges[1:]) / 2
    for key, label, colour, style in (("own", "each shaft's own spacing", SERIES[0], "-"),
                                      ("common", "one spacing for all", SERIES[0], "--"),
                                      ("5mm", "5 mm", SERIES[1], "-"),
                                      ("3.5mm", "3.5 mm", SERIES[1], "--")):
        values = []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            part = after[(after.from_tip_mm >= low) & (after.from_tip_mm < high)][
                f"error_{key}_mm"].dropna()
            values.append(part.median() if len(part) >= 20 else np.nan)
        axis.plot(middle, values, style, color=colour, linewidth=1.8, label=label)
    axis.axhline(5, color=MUTED, linestyle=":", linewidth=0.9)
    axis.set_xlabel("distance of the contact from the tip (mm)", fontsize=8)
    axis.set_ylabel("median distance, planned to real (mm)", fontsize=8)
    axis.set_xlim(0, 62)
    axis.set_ylim(bottom=0)
    axis.legend(fontsize=7, frameon=False, title="spacing used", title_fontsize=7)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    return path


def template_page(**_) -> str:
    import pandas as pd

    from onset_review.studies import _missing, chart, note, table

    summary = _json("template", "summary.json")
    contacts = _csv("template", "per_contact.csv")
    if summary is None or contacts is None:
        return _missing("Template map")
    out = ["# Template map: how far off, and what a label is worth\n", f"""
Without the patient's own CT and MRI, the **Map** step places contacts on a
template brain (MNI152): imported from a planning system, or planned as a
straight line from the target towards the entry. Here those plans are measured
against real implants on the same template: **{summary['patients']} patients,
{summary['shafts']} depth shafts, {summary['contacts']:,} contacts** from HUP
(ds004100), localised on each patient's own imaging. Each plan was given the
real deepest contact and the real last contact, the best a person could know.
Only the spacing was chosen.
"""]
    names = {"own": "each shaft's own spacing", "common":
             f"one spacing for all ({summary['common_spacing_mm']:.1f} mm)",
             "5mm": "5 mm", "3.5mm": "3.5 mm"}
    rows = [(names[k], v.get("n", ""), f"{v['median_mm']:.1f} mm", f"{v['p90_mm']:.1f} mm",
             f"{v['within_5mm']:.0%}", f"{v['same_label']:.0%}")
            for k, v in summary["variants"].items() if k in names]
    out.append("## Planned against real\n")
    out.append(table(pd.DataFrame(rows, columns=["spacing", "contacts", "median error",
                                                 "90th percentile", "within 5 mm",
                                                 "same atlas label"])))
    out.append(chart("template-error", lambda path: _error_chart(contacts, path),
                     "Median planning error by distance from the tip, for four spacings"))
    spacing = summary["spacing_mm"]
    out.append(note("info", f"""
On the template, neighbouring contacts were {spacing['p10']:.1f}–{spacing['p90']:.1f} mm
apart for most shafts (median {spacing['median']:.1f} mm). A template brain is not
the patient's size, so a plan at the catalogue pitch drifts by about 2 mm for
every 5 mm along the shaft. Set the spacing on the template, not from the
catalogue."""))
    out.append("## What the atlas says at the real positions\n")
    out.append(table(pd.DataFrame([(k, f"{v:.0%}") for k, v in summary["real_labels"].items()],
                                  columns=["label", "share of contacts"])))
    out.append("""
A label is *in* a named structure, *near* one (within 5 mm, with the distance),
*white matter*, or *outside the brain*. Contacts outside the template brain sit
mostly in the outer third of their shaft, near the entry. Read every label as
"probably". The full account is in `docs/TEMPLATE_MAP.md`.
""")
    return "\n".join(out)
