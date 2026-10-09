"""Onset-HFO — the reading interface.

    streamlit run app/Home.py

Same page order as the Onset prototype (`berdakh/onset`), so the two can be
read side by side and eventually merged. The difference is the data: that one
builds a synthetic cohort at startup, this one reads analyses and cohort
studies computed from **public recordings of real patients** — with expert
HFO markings, the contacts the surgeon removed, and whether the patient became
seizure-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.common import banner, pick_analysis, study  # noqa: E402

st.set_page_config(page_title="Onset-HFO", page_icon="🧠", layout="wide")
banner()

st.title("Onset-HFO")
st.subheader("An AI-assisted research tool for intracranial EEG")

st.markdown("""
**What this is.** An **AI-assisted research tool** for intracranial EEG. It
detects high-frequency oscillations (ripples 80–250 Hz, fast ripples 250–500
Hz) and interictal discharges, and the desktop app runs it on your own
recordings as well as public ones. These pages show it on **two public
archives of real patients**: it scores itself against **expert HFO markings
on 20 patients** and tests its map against **what happened to those patients
after surgery**. An agent on an open-weight model reads the results, must cite
them, and is refused by code when it strays.

**What it is not.** Not a diagnostic device, not validated on patients, not a
seizure-onset-zone finder. A high event rate is a measurement; physiological
ripples occur in healthy tissue. Onset-HFO organises evidence. The clinician
decides. **There is no recommendation anywhere in this product.**

**How to read the pages (left sidebar):**

1. **Recording** — channel ranking from the RMS and line-length detectors side by side, disagreement highlighted, and the signal window behind any event.
2. **Report** — the structured, cited report: findings, disagreements, data quality, limitations.
3. **Assistant** — ask about a channel or the evidence; try asking what to resect.
4. **Detectors** — how they score against expert markings: precision, recall, channel-rank agreement, and the operating point the data prefers.
5. **Outcome** — did the HFO map point at the tissue whose removal cured the patient? Including the two results that did *not* hold up.
6. **Patients** — the same outcome study one patient at a time, with the caveats that apply to each of them.
7. **Data** — the two archives, what each one carries, and how 24 MB is downloaded instead of 105 MB.
8. **Architecture** — every component, and which ones this release includes.
9. **Research** — the lab, the measured state of the work, and what is open.
""")

scores = study("outcome_groups_300s.csv")
curve = study("curve.csv")
store = pick_analysis()

columns = st.columns(4)
columns[0].metric("Patients with expert HFO markings", 20)
columns[1].metric("Runs in the archive", 385)
columns[2].metric("Public archives", 2)
columns[3].metric("Offline tests", 281)

st.info(
    "Nothing on these pages is simulated. The example recording ships with the "
    "source (60 s of OpenNeuro ds003029), and the cohort studies read tables "
    "committed to `data/stability/` — so every page works on a fresh clone with "
    "no download.")

if store is not None:
    st.caption(f"Loaded: **{store.subject}** · {store.metadata().get('source')} · "
               f"{store.metadata().get('duration_s', '?')} s · "
               f"{len(store.channels())} channels analysed")
if len(scores) and len(curve):
    st.caption("Cohort studies loaded: detector benchmark, surgical outcome, "
               "window stability, across-runs stability.")
