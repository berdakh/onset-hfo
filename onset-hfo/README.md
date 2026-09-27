# Onset-HFO — a first prototype for HFO and epileptiform-discharge detection

**Two halves, deliberately separate.**

1. **A signal-processing pipeline** that takes one minute of real, public
   intracranial EEG and reports where high-frequency oscillations (ripples,
   80–250 Hz) and interictal epileptiform discharges occur, with the exact
   signal window behind every number.
2. **An agent built on an open-weight language model** that can read what the
   pipeline produced, must cite it, and is refused, checked and contradicted
   by code whenever it strays — and that can also *drive* the pipeline,
   choosing which analyses to run and at what thresholds, then being made to
   resolve every claim it writes back to the run that produced it.

It is a **prototype**: small, readable, measured, and honest about what it
cannot do. It is not a medical device and it makes no clinical claim. It is
the smallest thing that is genuinely *useful to argue with*, built so that the
next person can extend it — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

This sits under the [Onset](https://berdakh.github.io/onset/) project
(Brain–Machine Interfaces Lab, Nazarbayev University) and inherits its
principles: every score carries the window it looked at, disagreement is
reported rather than averaged away, and **no recommendation exists anywhere in
the schema**.

---

## Start here

**[Open the live app](https://onsetnu.streamlit.app/)** — ten pages over real recordings, no install.
A channel's rate leads to the events behind it, an event leads to the signal it
was measured on, and the assistant's every citation opens to both. Ask it which
channels to resect and watch it refuse.

New to the project? [**Where do seizures start?**](https://berdakh.github.io/onset-hfo/TUTORIAL.html)
([source](../site/TUTORIAL.html)) is a standalone
walkthrough — the research question, the signal, the traps, every measured
result and what it does and does not support. One file, opens in a browser.

Otherwise, run something:

## Five notebooks, no setup

| | Notebook | What it does | Needs |
|---|---|---|---|
| 1 | [**HFO detection quickstart**](notebooks/01_hfo_detection_quickstart.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/01_hfo_detection_quickstart.ipynb) | Real public iEEG → detections → figures → cited report | ~24 MB download |
| 2 | [**Agentic analysis**](notebooks/02_agentic_analysis.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/02_agentic_analysis.ipynb) | An open-weight model answering questions about those results, with citations, refusals and guards | nothing (a model is optional) |
| 3 | [**Validation and benchmark**](notebooks/03_validation_and_benchmark.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/03_validation_and_benchmark.ipynb) | Precision/recall against known truth, threshold curves, what each check buys | nothing |
| 4 | [**Orchestration and localization**](notebooks/04_orchestration_and_localization.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/04_orchestration_and_localization.ipynb) | The agent choosing what to measure, a learned per-contact model, conformal sets, and trying to break all of it | nothing |
| 5 | [**Surgical outcome study**](notebooks/05_surgical_outcome_study.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/onset-hfo/blob/master/onset-hfo/notebooks/05_surgical_outcome_study.ipynb) | Did the HFO map point at the tissue whose removal cured the patient? Reproduces a published finding, then shows our detector missing it | ~480 MB download |

## Or, locally, in two minutes

```bash
git clone https://github.com/berdakh/onset-hfo.git
cd onset-hfo/onset-hfo
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# 1. offline: labelled synthetic data, no download, ~10 seconds
python -m onset_hfo.cli run --synthetic --figures

# 2. the real thing: 60 s of public iEEG from OpenNeuro ds003029 (~24 MB)
python -m onset_hfo.cli run --subject sub-pt01 --task ictal --run 01 \
       --start 50 --stop 110 --figures

# 3. ask the agent about the results (no model needed for the scripted policy)
python -m onset_agent.cli --results artifacts/results/sub-pt01_ictal_run-01 --demo

# 4. with a real open-weight model
ollama pull qwen2.5:7b-instruct && ollama serve &
python -m onset_agent.cli --results artifacts/results/sub-pt01_ictal_run-01 \
       --backend ollama --chat

# 5. let the model choose and parameterise the analyses, not just read them:
#    four rungs of increasing model control over the SAME analyzers
python -m onset_agent.orchestrate --subject sub-pt01 --task ictal --run 01 \
       --start 50 --stop 110 --score

# 6. learn a per-contact model across a cohort, and find out what transfers
python -m onset_hfo.learn cohort --dry-run   # the plan; downloads nothing
python -m onset_hfo.learn cohort             # ~24 MB per subject, resumable
python -m onset_hfo.learn evaluate           # within-subject vs LOPO vs cross-site
python -m onset_hfo.learn acquire            # which contacts to label first
python -m onset_hfo.learn uncertainty        # calibration, conformal coverage

# 7. measure the detectors against known truth
python -m onset_hfo.cli evaluate --seeds 1 7 42

# 8. score the detectors against expert HFO markings on 20 real subjects
python -m onset_hfo.cli benchmark

# 9. test the HFO map against what happened to the patients after surgery
python -m onset_hfo.cli outcome

# 10. read a saved analysis on a page, with the signal behind every number
#     (or just open https://onsetnu.streamlit.app/)
pip install -e ".[app]" && streamlit run app/Home.py

pytest -q        # 379 tests, all offline
```

## What it actually does

```
 public archive (OpenNeuro ds003029, CC0)
        │  byte-range download of 60 s  (datasets.py)
        ▼
 preprocess  ── 1 Hz high-pass · narrow 60 Hz notches · bipolar montage
        │                                              (preprocess.py)
        ▼
 detect ──┬── RMS energy        (Staba 2002)      ─┐
          ├── line length       (Gardner 2007)    ─┤ ripple band, 80–250 Hz
          └── spikes: amplitude + sharpness       ─┘ discharge band, 5–60 Hz
        │                                      (detectors/)
        ▼
 validate ── cycle count · spectral peak above the 1/f background
        │    (this is where filter ringing is removed)   (validate.py)
        ▼
 measure ── rates per channel with Poisson intervals · ranks ·
        │   detector agreement · rate before vs during the seizure  (metrics.py)
        ▼
 report ── findings with evidence windows · disagreements stated ·
        │  data quality · limitations · NO recommendation field    (report.py)
        ▼
 saved as CSV + JSON  ──►  the agent can read this, and nothing else
                                                  (store.py, onset_agent/)
```

And, in the other direction — the agent deciding what the pipeline measures:

```
 planner ── chooses a tool and its parameters          (planner.py)
    │       "survey every channel"  →  "now re-run the leaders at 7 SD"
    ▼
 tool registry ── strict JSON in, strict JSON out, one run_id per call
    │             every call RUNS the real pipeline   (contract.py, analysis.py)
    ▼
 evidence store ── append-only ledger: input, output, run_id, runtime
    │              failed calls kept too               (evidence.py)
    ▼
 verifier ── every number in the report must resolve to a run_id,
    │        or the sentence is struck and recorded    (verifier.py)
    ▼
 ranking + report + audit trail  ──►  scored against the archive's
                                      clinician SOZ labels (cohort.py, scoring.py)
```

## Results you can check

**Against surgical outcome** — the strongest test here, and the only one whose
reference standard is not another algorithm. 20 patients of ds003498, the whole
interictal-sleep run each (300 s), resected contacts from the archive's
clinical sheet, seizure outcome from `participants.tsv`. The question: *was the
channel generating the most fast ripples inside the tissue the surgeon
removed?*

| source | seizure-free (n=13) | recurrence (n=7) | AUC (95% CI) | p |
|---|---|---|---|---|
| expert markings | 11/13 | 3/7 | 0.71 (0.50–0.92) | 0.12 |
| our RMS detector | 10/13 | 3/7 | 0.67 (0.45–0.89) | 0.17 |

The direction is the one [Fedele et al. 2017](https://www.nature.com/articles/s41598-017-13064-1)
predicts — the study this dataset comes from — and with 13 patients against 7
it does not reach significance. `min_detectable_auc(13, 7)` is **0.85**, so
this cohort could not have established either number. Nothing in the study
survives correction for the 36 comparisons the study runs: one row reaches
p = 0.034 uncorrected, and chance alone produces about two.

**The most useful result is that an earlier version of this table said
something stronger.** On the first 60 seconds of the same recordings, the same
code and the same pre-specified metric, the expert arm gave AUC **0.82,
p = 0.007** — and that is what this README claimed until the study was re-run
on the full recording. Two patients account for the whole difference. A
conclusion that changes between minute one and minutes one-to-five is not yet
a measurement, and channel-ranking stability across windows is now the most
concrete open problem in this repository. The full accounting, both tables side
by side, is in [`docs/OUTCOME.md`](docs/OUTCOME.md).

**How unstable?** Now measured. Across five disjoint 60-second windows of the
same recordings, the expert AUC spans 0.566–0.819 and its p-value 0.007–0.613 —
the published 0.007 was the best of five minutes. The growing-window curve does
settle from 180 s, so the whole-run number above is a settled estimate of one
recording. And the result that was not designed for: **our detector is more
stable across windows than the expert markings** (AUC spread 0.09 vs 0.25, same
busiest channel in 12/20 patients vs 7/20). Figure and tables in
[`docs/OUTCOME.md`](docs/OUTCOME.md).

```bash
python -m onset_hfo.cli outcome            # whole runs, ~2.2 GB, ~25 minutes
python -m onset_hfo.cli outcome --stop 60  # the first minute: a different answer
python -m onset_hfo.cli stability          # 11 windows: does any of it hold?
```

**So the pipeline stopped picking a winner.** `candidates_resected` reports
the share of the channels that *cannot be told apart from the busiest one*
(overlapping Poisson rate intervals) which the surgeon removed. On whole runs
the data picks a single channel in only 9 of 20 patients — worst case, 24 tied
channels out of 37 — and reporting the set instead of a winner costs 0.017 AUC.
Never quote the single-channel number without `n_candidates` beside it.

**And the reassuring one.** Across five *different nights* the per-patient
answer holds for 18 of 20 patients (experts) and 16 of 20 (ours) — against
9 and 16 across five *minutes* of a single run. Pooling a patient's runs cuts
the median candidate set from 2 channels to 1 and the worst case from 24 to 5.
So the quantity measured here is a property of the **patient**, not of the
recording session; a minute was simply too short a unit. It still does not
predict outcome on twenty patients — nothing here survives correction.

```bash
python -m onset_hfo.cli stability --across-runs 5   # 92 runs, disk-pruned
```

**Against expert HFO markings** — 20 subjects of [ds003498](https://openneuro.org/datasets/ds003498)
(Zurich interictal slow-wave sleep, 2000 Hz), 41,187 expert-marked events,
scored only on the channels the annotators reviewed:

| band | detector | threshold | precision | recall | F1 | channel-rank ρ |
|---|---|---|---|---|---|---|
| ripple | RMS | 1.5 SD | 0.43 | 0.48 | **0.42** | 0.59 |
| ripple | RMS | 2.0 SD | 0.51 | 0.38 | 0.40 | **0.66** |
| ripple | RMS | 5.0 SD *(default)* | 0.59 | 0.12 | 0.17 | 0.37 |
| fast ripple | RMS | 3.5 SD | 0.33 | 0.29 | **0.30** | 0.59 |
| fast ripple | RMS | 5.0 SD | 0.54 | 0.20 | 0.26 | **0.61** |
| fast ripple | RMS | 2.0 SD | 0.09 | 0.53 | 0.15 | 0.44 |

Read that as **agreement, not accuracy**: the reference is the validated
output of another detector, so an event we find that it never proposed counts
against us either way. The useful sentence is *we reproduce about half of a
published detector's validated events and rank the same channels active at
ρ ≈ 0.66*.

It also shows the shipped default is wrong for this task — 5.0 SD comes from
the ictal literature and finds 12% of interictal ripple markings. `--threshold
interictal-agreement` (2.0 SD) is the measured operating point **for ripples**.
Fast ripples want 5.0 SD on the same cohort, and running 2.0 SD there drops
precision to 0.09 and erases the outcome signal above. One threshold for both
bands is a bug, not a simplification.

**Against synthetic ground truth**, where every event is known by construction
(three seeds, `python -m onset_hfo.cli evaluate`):

| detector | precision | recall | F1 |
|---|---|---|---|
| RMS energy (ripples) | 0.956 ± 0.017 | 0.528 ± 0.073 | 0.679 |
| line length (ripples) | 0.957 ± 0.017 | 0.550 ± 0.078 | 0.697 |
| interictal discharges | 0.998 ± 0.004 | 0.844 ± 0.037 | 0.914 |

Artifact rejection is what earns that precision: **before** it, the RMS
detector scores 0.63 (the false positives are filter ringing from large
transients); **after** it, 0.97, at a cost of about one point of recall.

On the **ictal recording** (`sub-pt01`, 60 s, 71 bipolar channels) the
pipeline runs in about 7 seconds, names seven channels the two detectors rank
very differently, and shows ripple rates going from 0 before the marked onset
to ~145/min during the seizure. It also reports, because it is true, that its
top-ranked channels do not overlap the contacts the clinician named more than
chance — see [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Documentation

| Document | Read it when you want to know |
|---|---|
| [`docs/README.md`](docs/README.md) | where to start, and what each file in the repository is for |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how the pieces fit, and why the two halves are separate |
| [`docs/DATA.md`](docs/DATA.md) | the dataset, its licence, its annotations, and how to use your own data |
| [`docs/METHODS.md`](docs/METHODS.md) | every algorithm, every threshold, and the paper it came from |
| [`docs/AGENT.md`](docs/AGENT.md) | how the agent is constrained, its threat model, and how to add a tool |
| [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) | the tool contract, the evidence store, the S0–S3 ladder, and what verification costs |
| [`docs/LOCALIZATION.md`](docs/LOCALIZATION.md) | the cohort, the learned per-contact model, calibration and conformal sets, and how much of it transfers |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | what was measured, how, and what the numbers mean |
| [`docs/OUTCOME.md`](docs/OUTCOME.md) | the surgical-outcome study and the window-stability study: design, results, and what they cannot support |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | what this must not be used for |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | the clinical and signal-processing vocabulary, defined |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | what to build next, in order, with the reasoning |
| [**the live app**](https://onsetnu.streamlit.app/) | the interface itself, deployed — ten pages over real recordings |
| [`app/README.md`](app/README.md) | the reading interface: what it shows, the one rule it is built on, and how to deploy it |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | how to add a detector, a dataset or a tool without breaking the contracts |

## Layout

```
onset_hfo/            the pipeline
  config.py           every threshold and band, in one place, documented
  datasets.py         byte-range loader for the public archive + provenance
  synthetic.py        labelled simulator, including the traps
  preprocess.py       channel selection, filtering, bipolar montage
  detectors/          base.py (primitives) · engine.py (the shared loop)
                      rms.py · line_length.py · spike.py
  spectral.py         the event spectrum: peak frequency and prominence
  validate.py         artifact rejection, with reasons kept
  metrics.py          rates, intervals, ranks, agreement, rate change
  evaluate.py         precision/recall against truth; the clinician-marker check
  report.py           the structured, cited report (no recommendation field)
  viz.py              the four figures
  pipeline.py         end to end
  store.py            the read-only view the agent is given
  cohort.py           clinician SOZ contacts, outcome and site from the archive
  batch.py            the pipeline across a cohort -> one labelled feature table
  clinical.py         the resected zone and outcome sidecars -> bipolar channels
  outcome.py          does the HFO map point at the tissue that was removed?
  stability.py        does any of that survive a change of analysis window?
  models.py           the learned per-contact model; within-subject / LOPO / cross-site
  uncertainty.py      calibration, split conformal sets, exchangeability stress test
  learn.py            python -m onset_hfo.learn ...
  cli.py              python -m onset_hfo.cli ...

onset_agent/          the agent
  tools.py            eight read-only tools + strict argument validation
  prompts.py          the system prompt, the answer contract, the planner prompt
  guard.py            scope refusals, citation checks, number verification
  backends.py         scripted · ollama · OpenAI-compatible · transformers
  agent.py            the question-answering loop
  cli.py              python -m onset_agent.cli ...

  -- the model driving the analysis, not just reading it --
  contract.py         the frozen JSON tool contract; run ids; validation
  analysis.py         nine LIVE tools: each one re-runs the real pipeline
  evidence.py         the append-only ledger every claim resolves against
  planner.py          the S0-S3 ladder, three stopping rules, a scripted planner
  verifier.py         deterministic + language-model verifiers, and their delta
  scoring.py          ranking vs clinician SOZ labels, with a permutation null
  orchestrate.py      python -m onset_agent.orchestrate ...

app/                  the reading interface (Streamlit), ten pages
  Home.py             the landing page
  common.py           banner, analysis picker, committed-study loader
  panels.py           everything the page decides; imports no Streamlit, so it is tested
  signal.py           the one place the interface touches the recording again
  pages/              Recording · Report · Assistant · Detectors · Outcome
                      Patients · Data · Architecture · Research

data/outcome/         the outcome study's per-subject tables, for the Patients page

data/example_analysis/  a real 60 s analysis, so the app works on a fresh clone

notebooks/            the five Colab notebooks (built by scripts/build_notebooks.py)
tests/                379 offline tests (synthetic data + a mock model server)
docs/                 everything above
```

## Principles this prototype is built on

1. **Every number carries the window it came from.** A finding without
   evidence cannot be constructed.
2. **Disagreement is reported, not resolved.** Two detectors run; where they
   differ, the report says so.
3. **Rejected events are kept, with reasons.** Nothing is silently dropped.
4. **The language model never produces a number.** It chooses what to look up
   and how to phrase it; the pipeline decides what is true, and a checker
   proves it for every answer.
5. **There is no recommendation field.** Not empty — absent.
6. **Public data first.** CC0, cited, and downloaded in the smallest slice
   that demonstrates the point.

## Licence and citation

MIT for the code. The data is CC0 from OpenNeuro `ds003029`; if you publish
anything derived from it, cite the dataset and its paper — the citation is
printed in every report and stored in `provenance.json`.
