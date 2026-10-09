"""The two public archives its methods are tested on — and what each one carries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import banner, pick_analysis  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Data", layout="wide")
banner()

st.title("The data — real recordings, real patients")
st.markdown("""
No simulator behind these pages. Two public, CC0, BIDS-formatted archives of
intracranial recordings from people who had epilepsy surgery.
""")

st.dataframe(pd.DataFrame([
    ("ds003498", "Zurich interictal slow-wave sleep", "20", "2000 Hz",
     "expert HFO markings · resected contacts · surgical outcome",
     "everything measured in this project"),
    ("ds003029", "Epilepsy-iEEG multicentre", "35+", "250–1000 Hz",
     "clinician seizure markers · curated SOZ contacts",
     "the ictal quickstart and the learned per-contact model"),
], columns=["archive", "what it is", "subjects", "sampling", "what it carries",
            "used for"]), width="stretch", hide_index=True)

st.subheader("Why ds003498 is the one that matters")
st.markdown("""
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
""")

st.subheader("How 24 MB is downloaded instead of 105 MB")
st.markdown("""
BrainVision stores samples **multiplexed** — channel 1 at time 1, channel 2 at
time 1, …, then time 2 — with no per-sample header. So sample *k* begins at byte
`k × n_channels × bytes_per_sample`, which makes **a byte range a time range**.

`onset_hfo.datasets.fetch_slice` downloads the three small text sidecars in
full, computes the byte offsets for the window you asked for, and issues one
HTTP Range request. A notebook starts in seconds instead of minutes, and a
laptop can work on an archive it could never hold.

The same trick is what makes the 92-run stability study affordable: each slice
is fetched, analysed and deleted, so peak disk stays flat against an archive
that is 46 GB in total.
""")

store = pick_analysis()
if store is not None:
    meta = store.metadata()
    st.subheader(f"Provenance of the loaded recording — {store.subject}")
    st.json({k: v for k, v in meta.items() if k not in ("notes",)})
    if meta.get("notes"):
        st.caption("Notes: " + "; ".join(meta["notes"]))
    st.caption(meta.get("citation", ""))

st.subheader("Licence and citation")
st.markdown("""
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
