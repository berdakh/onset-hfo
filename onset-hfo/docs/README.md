# Documentation index

Start wherever your question is.

## "Explain the whole thing to me once"

Open [**Where do seizures start?**](https://berdakh.github.io/onset-hfo/TUTORIAL.html)
— it lives in [`site/`](../../site/TUTORIAL.html) because that is the directory
CI publishes — a standalone walkthrough
of the entire project in eleven parts: the clinical problem, what the signal
looks like, why filter ringing makes this hard, the cohort, the measured
transfer gap, what a handful of labels buys, calibration, the agent, and the
attempts to break it. Every section ends with what the result actually
justifies claiming. It is self-contained — one file, no build step.

## "I just want to click around in it"

**[https://onsetnu.streamlit.app/](https://onsetnu.streamlit.app/)** — deployed, no install, no account.

Or locally:

```bash
pip install -e ".[app]"
streamlit run app/Home.py
```

The reading interface: a saved analysis on a page, where a rate leads to the
events behind it and an event leads to the signal it was measured on. See
[`app/README.md`](../app/README.md).

## "I just want to see it work"

Open [`notebooks/01_hfo_detection_quickstart.ipynb`](../notebooks/01_hfo_detection_quickstart.ipynb)
in Colab and run it top to bottom. Ten minutes, no setup, real data.

## "I want to understand what it does"

Read in this order:

1. [**ARCHITECTURE.md**](ARCHITECTURE.md) — the shape of the system: what the
   pipeline is, what the agent is, and why they are kept apart.
2. [**DATA.md**](DATA.md) — the recording: where it comes from, what its
   annotations mean, and how to point the loader at your own data.
3. [**METHODS.md**](METHODS.md) — every algorithm and every threshold, with
   the paper each came from and the reason for each deviation.
4. [**EVALUATION.md**](EVALUATION.md) — what has been measured, on what, and
   what the numbers do and do not support.
5. [**OUTCOME.md**](OUTCOME.md) — the one test whose reference standard is
   not another algorithm: did the HFO map point at the tissue whose removal
   made the patient seizure-free? Includes the expert positive control that
   tells an underpowered null apart from a detector that does not work, and
   the window study that measures whether any of it survives looking at a
   different minute of the same recording.
6. [**AGENT.md**](AGENT.md) — how a language model is allowed near clinical
   data at all: the tools, the guards, and the threat model.
7. [**ORCHESTRATION.md**](ORCHESTRATION.md) — the other direction: the model
   choosing and parameterising the analyses, the frozen tool contract, the
   evidence store, the S0–S3 ablation ladder, and what verification costs.
8. [**LOCALIZATION.md**](LOCALIZATION.md) — the learned half: the cohort, the
   per-contact model, calibration and conformal prediction sets, and the
   measured gap between a subject-specific model and one that has to work on
   a new patient.
9. [**LIMITATIONS.md**](LIMITATIONS.md) — read before quoting any number from
   this repository to anyone.

## "I am joining the project and need to do something useful"

* [**ROADMAP.md**](ROADMAP.md) — what is still open, in priority order, with
  what each item is blocked on; then the history of the finished ones, kept
  because what an item found is worth reading before starting the next.
* [**`onset_agent/benchmark.py`**](../onset_agent/benchmark.py) — the harness
  for the one measurement this project most conspicuously lacks: what a *real*
  open-weight model does on the S0–S3 ladder, swept over quantizations.
  `notebooks/06_agent_benchmark.ipynb` runs it on a Colab GPU and resumes
  after a disconnect; `python -m onset_agent.benchmark --backend scripted
  --synthetic` runs the whole thing offline in seconds. The harness is done;
  **the numbers are not**.
* [**CONTRIBUTING.md**](CONTRIBUTING.md) — how to add a detector, a dataset or
  an agent tool without breaking the contracts the rest of the code relies on.
* [**GLOSSARY.md**](GLOSSARY.md) — if "bipolar montage", "ictal" or "ripple
  band" are not yet second nature.

## "I am presenting this to people"

* [**EXPO.md**](EXPO.md) — the medical-expo brief: the 90-second pitch, a
  rehearsed five-minute demo path through the live app, the questions a
  clinician will ask with honest answers, and the claims never to make.
* [**POSTER.md**](POSTER.md) — the one claim a poster is allowed to make, the
  evidence behind each clause of it, the non-claims, an abstract, and the
  panel plan.
* [**DUPLICATION.md**](DUPLICATION.md) — why there are two repositories, what
  is actually duplicated between them, what must not be merged, and the plan.
* The printable one-page handout is [`site/handout.html`](../../site/handout.html),
  live at <https://berdakh.github.io/onset-hfo/handout.html>.

## "Where is X?"

| Question | File |
|---|---|
| Which thresholds are used, and why those? | [`onset_hfo/config.py`](../onset_hfo/config.py) — every parameter is documented at its definition |
| How is the data downloaded without pulling 105 MB? | [`onset_hfo/datasets.py`](../onset_hfo/datasets.py) |
| Where does a detection actually get decided? | [`onset_hfo/detectors/engine.py`](../onset_hfo/detectors/engine.py) |
| How are filter-ringing false positives removed? | [`onset_hfo/validate.py`](../onset_hfo/validate.py) and [`onset_hfo/spectral.py`](../onset_hfo/spectral.py) |
| What exactly can the agent see? | [`onset_hfo/store.py`](../onset_hfo/store.py) and [`onset_agent/tools.py`](../onset_agent/tools.py) |
| What stops the model inventing a number? | [`onset_agent/guard.py`](../onset_agent/guard.py) |
| What does the report contain? | [`onset_hfo/report.py`](../onset_hfo/report.py) |
| Which contacts were resected, and how is that mapped to channels? | [`onset_hfo/clinical.py`](../onset_hfo/clinical.py) |
| How is the outcome study computed, and what are its statistics? | [`onset_hfo/outcome.py`](../onset_hfo/outcome.py) |
| What does the interface decide, and what does it only display? | [`app/panels.py`](../app/panels.py) |
| Does a result survive a change of analysis window? | [`onset_hfo/stability.py`](../onset_hfo/stability.py) |

## A note on how this repository is written

Every module starts with a docstring that explains *why it exists*, not just
what it contains, and every non-obvious parameter is documented where it is
defined rather than in a separate wiki that will drift. If you find code here
whose purpose is not explained in the file itself, that is a bug — please fix
it or open an issue.
