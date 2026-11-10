"""The lab, the measured state of the work, and what is open."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import banner  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Research", layout="wide")
banner()

st.title("Research programme")

st.markdown("""
**Lab.** Brain–Machine Interfaces Laboratory, School of Computing and Artificial
Intelligence, Nazarbayev University. PI: Berdakh Abibullaev. Focus: deep
learning for EEG/iEEG signal processing, BCI, and clinical translation for
epilepsy.

**Where this sits.** Onset-HFO is the signal-processing and evidence half of the
[Onset](https://berdakh.github.io/onset/) project, built on **public recordings
of real patients** rather than a simulator, so that every claim can be checked
against a reference someone else published.
""")

st.subheader("What is measured, honestly")
st.markdown("""
| Claim | Status |
|---|---|
| Detects HFOs in real interictal recordings | **Yes** — agreement with expert markings, ρ ≈ 0.7 on channel ranking |
| Beats a published detector | **No, and not claimed.** The reference *is* a published detector; these are agreement numbers |
| The map relates to surgical outcome | **Direction yes, significance no.** AUC 0.71, p = 0.12 on 13 vs 7 patients; none of the study's 34 comparisons reaches p < 0.05 even before correction |
| The measurement is stable | **Across whole runs mostly** (16 of the 18 patients with more than one run, expert markings; 14 of 18 for our detector), **within a minute no** — a 60 s window changed the headline once |
| Distinguishes epileptic from physiological ripples | **No.** Nothing in the pipeline tries. The one available proxy — whether a ripple rides a discharge — was screened against outcome and does **not** localise better than the merged rate in ripples; in fast ripples one arm separates by one metric and falls below chance by the tie-aware one |
| The choice of detector is the limiting factor | **No, measured.** Four features, each swept on real data: at its best threshold none ranks channels better than RMS by more than 0.021 ρ, inside what twenty patients can resolve. The ceiling is the reference standard and the unit of analysis |
| Validated for clinical use | **No.** Not a medical device |
""")

st.subheader("Two results this project reversed on itself")
st.info("""
**The 60-second headline.** An outcome result of AUC 0.82, p = 0.007 reached the
README and the landing page before the study was re-run on whole recordings,
where it became 0.71, p = 0.12. Corrected everywhere, with both tables kept side
by side rather than one quietly replacing the other.

**The detector gap.** That same 60-second run appeared to show our detector
trailing the experts, which made "add a morphology criterion" look like the top
priority. On whole runs the gap closed — because the *expert* number
came down, not because ours went up — so the priority changed to ranking
stability instead.
""")

st.subheader("What is open, in order")
st.markdown("""
1. **A real model driving the analysis ladder.** Every orchestration number so
   far comes from a scripted planner, so it measures the script rather than the
   thesis. Needs a GPU afternoon; the harness is built and tested.
2. **A second cohort.** Everything rests on one centre, one annotation protocol,
   one surgical team. Needs an archive with HFO markings *and* resection *and*
   outcome — ds003498 is the only public one known to have all three.
3. **A hand-annotated benchmark.** The only way to turn *agreement* into
   *accuracy*: a few hundred expert-marked events on real data, reviewed here.
   The sampling, agreement and scoring harness is built; the marking needs two
   reviewers' afternoons.
4. **Electrode geometry.** Bipolar pairs are formed by contact number, which is
   wrong where a grid row wraps. Needs an archive that publishes coordinates.
5. **Physiological versus epileptic ripples** remains the biggest scientific
   gap, but it is no longer an experiment this project can run: the only proxy
   available in public data has now been measured: no gain in the ripple band,
   and in fast ripples one arm that the tie-aware metric contradicts. Progress
   needs a labelled archive, not more analysis.
""")

st.subheader("Four questions that came off this list by being answered")
st.success("""
**More than two detectors.** Two detectors matching about half their events was
a finding; the question was whether that disagreement is structural. Four
features were implemented and swept on real data, and it is: every pairwise
Jaccard overlap sits between 0.47 and 0.79. The two added features bought
**0.021 ρ** in ripples and **0.006** in fast ripples, both inside the per-patient
bootstrap interval, so the choice of feature is not where the headroom is.

**Do the ripple sub-populations localise differently?** Not in the ripple band:
splitting events by discharge coupling leaves every arm at or below the merged
rate. In fast ripples the spike-coupled busiest-channel arm gives AUC 0.875,
p = 0.004 (Bonferroni 0.033 across that screen's nine rows), and is reported as
measured; the tie-aware metric puts the same arm at 0.446, it collapses to 0.533
in ripples, and it is one of many comparisons. See the Outcome page.

**Is the re-planning ranking rule the right rule?** Conditionally. It helps in
the ripple band at every re-test point — the first evidence here that the
mechanism separating the re-planning rungs from the fixed ones carries
information — and in the fast-ripple band it hurts, 0.791 plain against
0.610–0.665 multiplied, now measured on every patient. It used to do worse than
that: a silent re-test zeroed every channel's score, and because a score-ordered
list with no variance falls through to the alphabetical tie-break, the result
was not a flat list but a confident ranking in the wrong order. That is fixed —
the rule refuses to fire on a re-test that measured nothing — and no patient is
dropped from any arm any more.

**Does the bipolar montage help or hurt for rate?** It helps against the
archive's own reference. The same pipeline
was run under the shipped bipolar montage, the archive's own reference and a
common average, on the same windows. Against the expert ranking the bipolar
arm has the highest Spearman in 15 of 20 patients (ripple) and 19 of 20 (fast
ripple); against outcome it is first on four of eight AUC cells and the common
average on the other four, and the archive's own reference is last on all
eight. The referential arms find more
events and agree less — noise shared through the reference, counted as events.
No single difference is significant at 13 against 7; the direction is the
result, and the default stays.
""")

st.subheader("Principles inherited from the Onset project")
st.markdown("""
1. No patient on both sides of a split; normalisation fit on training patients only.
2. Every score comes with the window it looked at, or an empty evidence list and
   a note saying so.
3. Disagreement between methods is reported, not averaged away.
4. Nothing in the system recommends treatment; the clinician decides.
5. Public data first; partner-centre data only through de-identification.
6. **When a number changes, the old one stays visible next to the new one.**
""")

st.caption("Contact the PI for collaboration or to see the full system. "
           "Code: github.com/berdakh/onset-hfo · MIT licence.")
