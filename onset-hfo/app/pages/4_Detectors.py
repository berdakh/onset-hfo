"""How the detectors score against expert HFO markings on 20 real patients.

Reads `data/benchmark/agreement_sweep.csv` rather than restating it. The
previous version of this page carried the sweep as nine hand-typed rows; every
one of them was right, and the sentence beneath them quoted a mean expert
fast-ripple count that was wrong -- in this file, in `docs/EVALUATION.md` and
in `onset_hfo/config.py`, at two different wrong values. That is what a
transcription costs, and it is why the numbers here now come from the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import panels  # noqa: E402
from app.common import banner  # noqa: E402
from onset_hfo.config import THRESHOLDS  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Detectors", layout="wide")
banner()

BANDS = {"ripple": "Ripples (80–250 Hz)", "fast_ripple": "Fast ripples (250–500 Hz)"}
METRICS = {
    "rank_rho": "channel-rank ρ",
    "f1": "F1",
    "precision": "precision (agreement)",
    "recall": "recall",
    "detections": "detections per 60 s",
    "top5_overlap": "top-5 channels shared, out of 5",
}


@st.cache_data(show_spinner=False)
def sweep() -> pd.DataFrame:
    return panels.sweep()


st.title("Detectors and how they are scored")
st.markdown("""
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

The first two were the original pair and carry the sweep below, which has the
finer grid and the longer history. The second two were added later to test
whether the choice of feature is where the remaining headroom lies; the
[family comparison](#the-whole-family-does-the-feature-choice-matter) says it
is not.
""")

frame = sweep()
cohort = panels.benchmark_cohort()
if frame.empty:
    st.error("`data/benchmark/agreement_sweep.csv` is missing. "
             "Regenerate with `python -m onset_hfo.cli benchmark`.")
    st.stop()

# --------------------------------------------------------------------------
# The reference, stated from the table it was scored against
# --------------------------------------------------------------------------

st.subheader("The reference, and why it is agreement rather than accuracy")
if not cohort.empty:
    left, mid, right = st.columns(3)
    left.metric("Patients", len(cohort))
    left.caption(f"first {cohort['duration_s'].iloc[0]:.0f} s of run-01 each, "
                 f"{cohort['sfreq_hz'].iloc[0]:.0f} Hz, slow-wave sleep")
    mid.metric("Expert-marked events", f"{int(cohort['n_expert_events'].sum()):,}")
    mid.caption(f"{int(cohort['expert_ripples'].sum()):,} ripples · "
                f"{int(cohort['expert_fast_ripples'].sum()):,} fast ripples")
    right.metric("Reviewed channels",
                 f"{int(cohort['n_reviewed_channels'].min())}–"
                 f"{int(cohort['n_reviewed_channels'].max())}")
    right.caption("per patient — scoring is restricted to these; a detection "
                  "elsewhere is unjudged, not wrong")

st.markdown(f"""
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
""" if not cohort.empty else "")

# --------------------------------------------------------------------------
# The sweep itself
# --------------------------------------------------------------------------

st.subheader("The sweep, as committed")
band = st.radio("Band", list(BANDS), horizontal=True, format_func=BANDS.get)
metric = st.selectbox("Plot", list(METRICS), format_func=METRICS.get)

part = frame[frame["band"] == band]
chart = part.pivot(index="threshold_sd", columns="detector", values=metric)
st.line_chart(chart, x_label="threshold (robust SD of the channel's own feature)",
              y_label=METRICS[metric])

st.dataframe(
    part.drop(columns="band").sort_values(["detector", "threshold_sd"]).rename(
        columns={"threshold_sd": "threshold (SD)", "f1": "F1",
                 "rank_rho": "channel-rank ρ", "detections": "detections / 60 s",
                 "top5_overlap": "top-5 shared"}),
    width="stretch", hide_index=True)
st.caption("Cohort means over 20 patients. `channel-rank ρ` is the Spearman "
           "correlation between our per-channel rates and the experts'.")

# --------------------------------------------------------------------------
# What the sweep says, derived rather than asserted
# --------------------------------------------------------------------------

st.subheader("Where each arm peaks")
criterion = st.radio("Best by", ["rank_rho", "f1"], horizontal=True,
                     format_func=METRICS.get)
best = panels.operating_points(criterion)
shown = best.rename(columns={
    "threshold_sd": "best threshold (SD)", "rank_rho": "channel-rank ρ",
    "f1": "F1", "detections": "detections / 60 s"}).round(3)
# A disabled checkbox reads as a control the reader cannot use; this is a fact.
shown["on the edge of the grid?"] = best["at_boundary"].map(
    {True: "yes — not yet measured", False: "no — interior"})
shown = shown.drop(columns="at_boundary")
st.dataframe(shown, width="stretch", hide_index=True)

edge = best[best["at_boundary"]]
if len(edge):
    st.warning(
        "**An optimum on the edge of the swept grid is not an optimum, it is "
        "the grid running out.** " + "; ".join(
            f"`{r.detector}` in the {r.band} band peaks at {r.threshold_sd:g} SD, "
            f"the {'lowest' if r.threshold_sd <= 2.0 else 'highest'} value swept"
            for r in edge.itertuples()) +
        ". This project has already been wrong that way once — the first ripple "
        "sweep stopped at 2.0 SD and reported 2.0 as the answer, which is why "
        "the RMS arm now runs down to 1.0. The arms above have not been "
        "extended, so read those rows as *not yet measured* rather than as a "
        "detector that lost.")
else:
    st.success("Every arm's optimum is interior to its swept grid, so none of "
               "them is the grid running out.")

with st.expander("What was actually swept, per arm"):
    st.dataframe(panels.sweep_grid().rename(columns={
        "n_points": "thresholds tried", "lowest": "lowest SD",
        "highest": "highest SD"}), width="stretch", hide_index=True)
    st.caption("The four arms do not share a grid: each RMS arm was extended "
               "when its first optimum landed on a boundary, and neither line-"
               "length arm was. A comparison across arms is a comparison "
               "across grids.")

# --------------------------------------------------------------------------
# All four features, on a common grid
# --------------------------------------------------------------------------

st.subheader("The whole family: does the feature choice matter?")

# Streamlit re-runs a changed page script but can keep an already-imported local
# module in sys.modules, so a deploy that adds a `panels` function can leave a
# new page calling an old module. That used to raise AttributeError and blank
# the whole page; now it degrades to this one section and names the remedy.
_needed = ("family_operating_points", "family_margin")
_missing = [name for name in _needed if not hasattr(panels, name)]

if _missing:
    family, margin = pd.DataFrame(), {}
    st.warning(
        "This section needs `panels." + "`, `panels.".join(_missing) + "`, which "
        "the running app has not loaded. That happens when a deploy picks up a "
        "changed page while keeping the previously imported module, and it "
        "clears on a restart: **Manage app → Reboot**. The rest of this page is "
        "unaffected.")
elif (family := panels.family_operating_points()).empty:
    margin = {}
    st.info("`data/benchmark/four_detector_sweep.csv` is missing, so the "
            "four-detector comparison is not shown.")
else:
    margin = panels.family_margin()
    st.markdown("""
Two more features were added and swept on the same 20 patients over a common
grid, so that four are comparable under identical downstream processing. Each
row is that detector at **its own** best threshold, because comparing features
at a shared threshold measures the threshold rather than the feature.
""")
    st.dataframe(family.rename(columns={
        "threshold_sd": "its best threshold (SD)", "rank_rho": "channel-rank ρ",
        "f1": "F1", "detections": "detections / 60 s"}).round(3),
        width="stretch", hide_index=True)
    st.caption(
        "**Why line length peaks lower here than in the table above.** These are "
        "two different grids, and on the 27 thresholds they share they agree "
        "*exactly* — the largest difference in any metric is 0. The older sweep "
        "simply never ran line length below 2.0 SD, so its optimum of 2.5 was "
        "the grid stopping too high; this one runs down to 0.5 and finds 1.5. "
        "That is the boundary problem the warning above describes, caught on an "
        "arm that had not been extended.")

    if margin:
        cols = st.columns(len(margin))
        for col, (band, m) in zip(cols, sorted(margin.items()), strict=False):
            col.metric(f"{BANDS.get(band, band)} — gain from adding two features",
                       f"{m['margin']:+.3f} ρ")
            col.caption(f"best of the original pair {m['best_original']:.3f}, "
                        f"best of all four {m['best_overall']:.3f} "
                        f"(`{m['winner']}`)")

    st.success("""
**Both margins are ties.** Two more detectors, each given its own sweep on real
data, bought no measurable improvement in channel ranking over the two already
there. That is a useful thing to have paid for: it says the ceiling here is not
the choice of feature. What is left is the reference standard — the markings are
another detector's validated output, so precision above about 0.6 is not
available — and the unit of analysis, which the Outcome page shows dominates
everything else measured here.
""")

    st.markdown("""
**A threshold in robust SD is not portable between features.** Short-time energy
is RMS before the square root, so under a *fixed* threshold the two would select
identical samples. They do not, because `median + 5 robustSD` sits at about the
98th percentile of an RMS trace and the 96th of a squared one: squaring is not
affine and stretches the upper tail relative to the median. Run short-time
energy at RMS's fast-ripple threshold and its rank agreement falls from 0.601 to
**0.485**. The Hilbert envelope, by contrast, wants almost exactly what RMS
wants. Each feature needs its own sweep, and the table above is what that looks
like.
""")

    with st.expander("How much the four detectors disagree with each other"):
        st.dataframe(pd.DataFrame([
            ("RMS", 1.000, 0.588, 0.788, 0.471),
            ("line length", 0.588, 1.000, 0.574, 0.500),
            ("Hilbert envelope", 0.788, 0.574, 1.000, 0.506),
            ("short-time energy", 0.471, 0.500, 0.506, 1.000),
        ], columns=["detector", "RMS", "line length", "Hilbert", "short-time energy"]),
            width="stretch", hide_index=True)
        st.caption(
            "Mean Jaccard overlap over three synthetic seeds. Every off-diagonal "
            "cell sits between 0.47 and 0.79: no two of the four agree on more "
            "than four events in five, and the worst pair agrees on fewer than "
            "half. The disagreement is structural rather than a quirk of one "
            "pair, so a single-detector rate table is less certain than it looks "
            "— and that is now four measurements rather than one. Note which "
            "pair is least similar: short-time energy was added *expecting* "
            "near-redundancy with RMS, and the prediction was wrong.")

# --------------------------------------------------------------------------

rms = frame[(frame["band"] == "ripple") & (frame["detector"] == "rms")]
at = {row.threshold_sd: row for row in rms.itertuples()}
default, tuned = at[5.0], at[THRESHOLDS["interictal-agreement"]]
fr = frame[(frame["band"] == "fast_ripple") & (frame["detector"] == "rms")]
fr_at = {row.threshold_sd: row for row in fr.itertuples()}
fr_tuned = fr_at[THRESHOLDS["interictal-agreement-fast-ripple"]]
fr_wrong = fr_at[THRESHOLDS["interictal-agreement"]]

st.subheader("What the sweep changed")
st.markdown(f"""
**The shipped default was wrong for interictal work, and now that is measured.**
At 5.0 SD — Staba's published value, which this pipeline inherited — the ripple
detector finds {default.recall:.0%} of the expert-marked events and ranks
channels at ρ = {default.rank_rho:.2f}. At
{THRESHOLDS["interictal-agreement"]:g} SD it finds {tuned.recall:.0%} and ranks
at ρ = {tuned.rank_rho:.2f}. The default stays at 5.0 so that published ictal
results do not silently change; `--threshold interictal-agreement` selects the
measured one.

**The two bands want different operating points, by a factor of
{THRESHOLDS["interictal-agreement-fast-ripple"] / THRESHOLDS["interictal-agreement"]:g}.**
Ripples rank best at {THRESHOLDS["interictal-agreement"]:g} SD, fast ripples at
{THRESHOLDS["interictal-agreement-fast-ripple"]:g} SD. Running the ripple value
in the fast-ripple band drops precision from {fr_tuned.precision:.2f} to
**{fr_wrong.precision:.2f}** — fewer than one detection in ten is an
expert-marked event, at a mean of {fr_wrong.detections:,.0f} detections per 60 s
against {cohort['expert_fast_ripples'].mean():.0f} marked ones. One threshold
for both bands is a bug, not a simplification, and it flattened the outcome
result on the next page until it was found.

**Channel ranking is the number that matters.** Nobody operates on an event;
they operate on tissue. ρ is the agreement between our per-channel rates and the
experts', and it peaks at a different threshold from F1 — which is why both are
reported instead of one headline.
""")

# --------------------------------------------------------------------------

st.subheader("Against known truth, where accuracy *can* be measured")
st.dataframe(pd.DataFrame([
    ("RMS energy (ripples)", "0.956 ± 0.017", "0.528 ± 0.073", 0.679),
    ("Line length (ripples)", "0.957 ± 0.017", "0.550 ± 0.078", 0.697),
    ("Hilbert envelope (ripples)", "0.932", "0.556", 0.696),
    ("Short-time energy (ripples)", "0.835", "0.808", 0.821),
    ("Interictal discharges", "0.998 ± 0.004", "0.844 ± 0.037", 0.914),
], columns=["detector", "precision", "recall", "F1"]),
    width="stretch", hide_index=True)
st.caption(
    "Three synthetic recordings where every event is known by construction — "
    "the one table on this page that is **not** read from a committed file, "
    "because `evaluate` regenerates its data in seconds from a seed rather "
    "than downloading it. Reproduce with `python -m onset_hfo.cli evaluate "
    "--seeds 1 7 42`. Artifact rejection is what earns that precision: before "
    "it the RMS detector scores 0.63 — nearly all of its false positives are "
    "large transients ringing through the band-pass — and after it, 0.97, for "
    "about one point of recall.")
st.warning(
    "**Do not read short-time energy's F1 of 0.821 as a better detector.** All "
    "four run at their *default* 5.0 SD here, and 5.0 SD is a materially more "
    "permissive point on a squared feature — it finds 294 events on one "
    "recording where RMS finds 132. On real data at its own tuned threshold it "
    "matches RMS rather than beating it. A detector comparison at a shared "
    "threshold is a comparison of thresholds.")

st.info(
    "Reproduce the sweep: `python -m onset_hfo.cli benchmark`, about an hour "
    "after the first fetch. It writes to `artifacts/` — untracked — and "
    "`data/benchmark/` is the committed extract this page reads, so every "
    "number above can be checked from a clone with no network. Full tables and "
    "the reasoning: `docs/EVALUATION.md`.")
