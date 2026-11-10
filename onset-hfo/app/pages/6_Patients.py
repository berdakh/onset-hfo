"""The outcome cohort one patient at a time, rather than one row per group.

Every other page in this app reports a group: an AUC, a cohort mean, a
threshold sweep. That is what a paper needs and it is not what a clinician
asks. This page answers "what happened to *this* patient, and how much should
I believe it" — which, on a cohort of twenty, is mostly a question about
caveats, so the caveats are the part that is hardest to miss here.

Named ``Patients`` rather than ``Cohort`` on purpose: ``data/cohort/`` already
holds the ds003029 feature table for the learned model, and a page called
Cohort that read ``data/outcome/`` would be a trap for the next reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import banner  # noqa: E402
from onset_review import studytables as panels  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Patients", layout="wide")
banner()

st.title("Twenty patients, one at a time")
st.caption("ds003498 · whole 300 s runs · pre-specified arm: fast ripples, "
           "reviewed channels · expert markings and our RMS detector on the "
           "same channels")


@st.cache_data(show_spinner=False)
def overview(band: str, scope: str) -> pd.DataFrame:
    return panels.cohort_overview(band=band, scope=scope)


@st.cache_data(show_spinner=False)
def channels(subject: str, band: str) -> pd.DataFrame:
    return panels.subject_channels(subject, band=band)


band = st.sidebar.selectbox("Band", ["fast_ripple", "ripple"],
                            format_func=lambda b: {"fast_ripple": "Fast ripples (250–500 Hz)",
                                                   "ripple": "Ripples (80–250 Hz)"}[b])
scope = st.sidebar.selectbox("Channels", ["reviewed", "all"],
                             format_func=lambda s: {"reviewed": "Reviewed by the annotators",
                                                    "all": "Every recorded channel"}[s])
if (band, scope) != (panels.PRIMARY["band"], panels.PRIMARY["scope"]):
    st.warning(f"You are looking at **{band} / {scope}**, which is not the arm the "
               f"study pre-specified (fast_ripple / reviewed). Every number on the "
               f"Outcome page comes from the pre-specified arm; these are among the "
               f"same 34 distinct comparisons, none of which reaches p < 0.05 "
               f"even before correction.")

cohort = overview(band, scope)
if cohort.empty:
    st.error("No committed per-subject tables found in `data/outcome/`. "
             "Regenerate with `python -m onset_hfo.cli outcome`.")
    st.stop()

tab_cohort, tab_patient, tab_limits = st.tabs(
    ["The twenty", "One patient", "What this screen is not"])


# --------------------------------------------------------------------------

with tab_cohort:
    st.subheader("Was the busiest channel inside the tissue that was removed?")

    n_sf = int(cohort["seizure_free"].sum())
    counts = []
    for source, label in (("expert", "expert markings"), ("rms", "our RMS detector")):
        top = cohort[f"top_resected_{source}"]
        hit = top == 1
        counts.append((
            label,
            f"{int(hit.sum())}/{len(cohort)}",
            f"{int((hit & cohort['seizure_free']).sum())}/{n_sf}",
            f"{int((hit & ~cohort['seizure_free']).sum())}/{len(cohort) - n_sf}",
            int((cohort[f"n_candidates_{source}"] > 1).sum()),
        ))
    st.dataframe(pd.DataFrame(counts, columns=[
        "source", "busiest channel was resected", "among seizure-free",
        "among recurrence", "patients with no single busiest channel"]),
        width="stretch", hide_index=True)

    agree = int((cohort["top_resected_expert"] == cohort["top_resected_rms"]).sum())
    primary = (band, scope) == (panels.PRIMARY["band"], panels.PRIMARY["scope"])
    # Only the pre-specified arm is the one the Outcome page reports, so only
    # there may this tab claim to be showing the same number twice.
    provenance = ("*is* the AUC on [the Outcome page](/Outcome), written out as "
                  "patients" if primary else
                  "is one of the study's 34 comparisons; the Outcome page reports "
                  "the pre-specified fast-ripple arm, not this one")
    st.markdown(f"""
That gap — **{counts[0][2]} of the seizure-free patients against
{counts[0][3]} of the recurrences** — {provenance}. Either way it is a result
four patients wide: move one row and it moves.

The two sources reach the same inside/outside answer in **{agree} of
{len(cohort)}** patients, which is the more interesting number on this tab: our
detector and the experts mostly disagree about *rates*, and mostly agree about
*which channel is busiest*.
""")

    st.markdown("#### Every patient")
    show = cohort.rename(columns={
        "epilepsy": "type", "ilae": "ILAE", "rz_coverage": "resection recorded",
        "n_reviewed": "reviewed ch", "n_resected_channels": "resected ch",
        "top_resected_expert": "resected? (expert)",
        "top_resected_rms": "resected? (RMS)",
        "n_candidates_expert": "ties (expert)", "n_candidates_rms": "ties (RMS)",
        "share_in_rz_expert": "share inside (expert)",
        "share_in_rz_rms": "share inside (RMS)",
    })
    st.dataframe(
        show[["subject", "type", "ILAE", "outcome", "reviewed ch", "resected ch",
              "resection recorded", "resected? (expert)", "ties (expert)",
              "share inside (expert)", "resected? (RMS)", "ties (RMS)",
              "share inside (RMS)"]].round(2),
        width="stretch", hide_index=True,
        column_config={
            "outcome": st.column_config.TextColumn(
                "outcome", help="S = seizure-free (ILAE 1–2) · F = recurrence"),
            "resection recorded": st.column_config.NumberColumn(
                help="fraction of the contacts listed as resected that appear in "
                     "the recording at all — below 1.0, read every share with care",
                format="%.2f"),
            "ties (expert)": st.column_config.NumberColumn(
                help="channels statistically tied for busiest; above 1 there is no "
                     "single busiest channel"),
            "ties (RMS)": st.column_config.NumberColumn(
                help="channels statistically tied for busiest; above 1 there is no "
                     "single busiest channel"),
        })
    st.caption(
        "`resected?` is 1 when the busiest channel lay inside the resection, 0 when "
        "it did not. `share inside` is the proportion of that patient's events "
        "inside it — which localises nothing" +
        (" (cohort AUC 0.53, against 0.71 for `resected?` on the same events)"
         if primary else "") +
        " and is here so you can see that for yourself.")

    # Empty in the committed tables: every resected contact is recorded. Through
    # 0.3.24 five patients read 0.25 here because their electrodes are named
    # differently in the resection sheet and the recording; the check stays for
    # data where a resection really was only partly recorded.
    thin = cohort[cohort["rz_coverage"] < 1.0]
    if len(thin):
        listed = ", ".join(f"{name} ({coverage:.0%})" for name, coverage
                           in zip(thin["subject"], thin["rz_coverage"], strict=True))
        st.warning(f"**{len(thin)} patients' resections were not fully recorded** — "
                   f"{listed} of the resected contacts appear in the recording. "
                   f"Their `share inside` describes only that part of the tissue "
                   f"that was removed. They are in the study because excluding "
                   f"them after seeing the numbers would be worse.")


# --------------------------------------------------------------------------

with tab_patient:
    subject = st.selectbox("Patient", cohort["subject"].tolist())
    row = cohort[cohort["subject"] == subject].iloc[0]

    # ``st.metric``'s delta would draw an up-arrow beside each of these, which
    # on a clinical screen reads as "improved". They are facts, not changes.
    left, mid, right = st.columns(3)
    left.metric("Outcome", "Seizure-free" if row["seizure_free"] else "Recurrence")
    left.caption(f"ILAE {int(row['ilae'])} · {int(row['months_follow_up'])} months "
                 f"of follow-up")
    mid.metric("Epilepsy", row["epilepsy"])
    mid.caption("lesion on imaging" if row["lesion"] == 1 else "no lesion found")
    right.metric("Channels analysed", f"{int(row['n_reviewed'])} reviewed")
    right.caption(f"{int(row['n_resected_channels'])} inside the resection · "
                  f"{int(row['n_channels'])} recorded in all")

    st.markdown("#### What each source said about this patient")

    def word(value, when_true: str, when_false: str) -> str:
        """A missing value prints as an em dash rather than as ``False``."""
        return "—" if pd.isna(value) else (when_true if value else when_false)

    def percent(value) -> str:
        return "—" if pd.isna(value) else f"{value:.0%}"

    answers = []
    for source, label in (("expert", "expert markings"), ("rms", "our RMS detector")):
        ties = row.get(f"n_candidates_{source}")
        answers.append((
            label,
            word(row.get(f"top_resected_{source}"), "inside the resection", "outside it"),
            "—" if pd.isna(ties) else
            ("a single channel" if ties == 1 else f"{int(ties)} tied channels"),
            percent(row.get(f"candidates_resected_{source}")),
            percent(row.get(f"share_in_rz_{source}")),
            word(row.get(f"stable_across_windows_{source}"), "same answer", "changed"),
            word(row.get(f"stable_across_runs_{source}"), "same answer", "changed"),
        ))
    st.dataframe(pd.DataFrame(answers, columns=[
        "source", "busiest channel was", "and it was", "of the tied set resected",
        "share of events inside", "across 5 minutes of this run",
        "across this patient's other nights"]), width="stretch", hide_index=True)

    caveats = panels.subject_caveats(subject, band=band, scope=scope)
    st.markdown("#### Before reading anything above as a finding")
    if caveats:
        for note in caveats:
            st.warning(note)
    else:
        st.info("None of this screen's four per-patient checks fired for this "
                "patient: their resection was fully recorded, both sources picked "
                "a single busiest channel, and both gave the same answer across "
                "the five minutes of this run and across their other nights. That "
                "is the cleanest case in the cohort, and it is still one patient "
                "in a study of twenty that is underpowered for all of them.")

    st.markdown("#### The channels underneath it")
    rows = channels(subject, band)
    if rows.empty:
        st.info("No committed channel table for this patient.")
    else:
        shown = rows.rename(columns={"expert_events": "expert events",
                                     "rms_events": "RMS events"})
        shown["eloquent"] = shown["eloquent"].map({True: "eloquent", False: ""})
        st.dataframe(
            shown[["channel", "zone", "eloquent", "expert events", "RMS events"]],
            width="stretch", hide_index=True, height=340,
            column_config={"zone": st.column_config.TextColumn(
                help="resected = both contacts removed · partial = one · "
                     "spared = neither")})
        st.caption(f"{len(rows)} reviewed channels, busiest first by expert count. "
                   f"A **partial** channel is the one to look at hardest: half of it "
                   f"was removed, so it counts as neither in or out, and the study "
                   f"reports it both ways (`share_in_rz` and "
                   f"`share_in_rz_incl_partial`).")

    with st.expander("Every metric computed for this patient, both bands, both scopes"):
        st.dataframe(panels.subject_metrics(subject).round(3), width="stretch",
                     hide_index=True)


# --------------------------------------------------------------------------

with tab_limits:
    st.error("""
**This screen cannot tell you anything about a patient.** It is a per-patient
*view of a group result*, and the group result does not reach significance:
13 seizure-free against 7 recurrences, `min_detectable_auc(13, 7) = 0.85`, and
none of the study's 34 distinct comparisons reaches p < 0.05 even before
correction. A row
here is one observation from a cohort too small to have established the effect
it is an observation of.
""")
    st.markdown("""
Four things this page does *not* show, because they do not exist:

- **A confidence interval on a single patient's answer.** `top_channel_resected`
  is one bit. The interval is on the cohort, and it spans a coin flip.
- **A prediction.** Nothing here was fitted, so nothing here can be applied to
  a new patient. The two sources are compared on the same channels; neither is
  a model of outcome.
- **The tissue the surgeon should have removed.** The resection is the input,
  not the output — we observe whether the busiest channel was inside it.
- **Why a patient's answer changed between windows or nights.** The stability
  columns record *that* it changed. Which minute was right is not knowable from
  this data, and that is the point of
  [the Outcome page](/Outcome) rather than a defect of this one.

**The rows are honest about themselves, which is the most this can offer.** The
caveats on the second tab are computed from the committed tables, not written
by hand: resection coverage, tie sets, window stability and run stability, per
patient. Five of the twenty patients trip none of them; the other fifteen do,
and a screen that only showed the five would be the same mistake as reporting
the best of five minutes.
""")
    st.info("Design, every group table, and the full list of what the study "
            "cannot support: `docs/OUTCOME.md`. The tables behind this page and "
            "the demographic columns deliberately left out of them: "
            "`data/outcome/README.md`.")
