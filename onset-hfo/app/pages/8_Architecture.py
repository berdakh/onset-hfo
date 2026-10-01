"""Every component, and which ones this prototype includes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import banner  # noqa: E402

st.set_page_config(page_title="Onset-HFO · Architecture", layout="wide")
banner()

st.title("Architecture: the full system and what this prototype includes")

st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F7F6F2",
                              fontname="Helvetica", fontsize=11];
  arch   [label="Public archive\\nOpenNeuro (BIDS)\\nbyte-range slices"];
  sim    [label="Simulator\\nlabelled events\\n+ the traps", fillcolor="#EEEDFE"];
  pre    [label="Preprocess\\nhigh-pass · notch\\nbipolar montage"];
  det    [label="Detectors\\nRMS · line length\\nenvelope · energy\\nspikes"];
  val    [label="Artifact rejection\\nspectral prominence\\nover 1/f"];
  meas   [label="Measure\\nrates · Poisson CIs\\nagreement"];
  store  [label="artifacts/results/\\nevents · rates · report\\n(immutable)"];
  agent  [label="Agent\\n8 read-only tools\\ncites or refuses", fillcolor="#E1F5EE"];
  ui     [label="app/  reading interface\\nshows · decides nothing", fillcolor="#E1F5EE"];
  studies[label="Cohort studies\\nbenchmark · outcome\\nstability", fillcolor="#FAEEDA"];
  arch -> pre; sim -> pre; pre -> det -> val -> meas -> store;
  store -> agent -> ui; store -> ui; store -> studies -> ui;
}
""")

st.markdown("""
Both readers — the agent and this interface — go through the same read-only
`ResultStore`. Neither can change a number. That is the whole reason it exists:
a second reader is where a project usually grows a second source of truth, and
here it structurally cannot.
""")

st.subheader("Full system versus this prototype")
st.markdown("""
| Component | Full system | This prototype |
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
| Interface | clinician UI, roles, audit | these pages: shows, decides nothing |
| Security | roles, audit middleware, row-level security, on-prem | **not included** — single user, no PHI |
| Operations | containers, traces, budgets, nightly chain | **not included** |
""")

st.subheader("The contracts that hold it together")
st.markdown("""
1. **Every score carries the window it looked at.** An event's evidence id is
   `subject|channel|detector|start`, and it resolves or the answer is discarded.
2. **The agent cannot compute.** Eight read-only tools over one saved analysis.
   No tool runs a detector, writes a file, or takes another patient's id.
3. **Scope refusals happen before the model runs.** Treatment and diagnosis
   questions never reach a model that could be persuaded.
4. **Disagreement is reported, not averaged away.** Both detector ranks are
   shown; neither is preferred.
5. **There is no recommendation field.** Not empty — absent from the schema.
""")

st.caption(
    "The prototype keeps every rule that matters to a clinician — cited evidence, "
    "disagreements stated, uncertainty shown, no recommendation — and leaves out "
    "everything that only matters once real users and identifiable data exist.")
