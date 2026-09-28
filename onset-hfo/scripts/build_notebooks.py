"""Generate the Colab notebooks in ``notebooks/``.

The notebooks are committed to the repository (so that the Colab badges work),
but they are *written* here, as plain Python, because reviewing a diff of
notebook JSON is miserable and because it keeps the notebooks consistent
with each other.

    python scripts/build_notebooks.py                  # write the JSON
    python scripts/build_notebooks.py --execute        # write it and run it

``--execute`` runs each notebook top to bottom and stores the outputs, which
is how the committed copies get their figures and tables. It needs the data
cache (or network access) and takes a while; pass notebook numbers to limit
it, e.g. ``--execute 5``.

Every notebook is designed to run top to bottom in a free Colab instance with
no configuration. Notebook 1 downloads ~24 MB of public data and notebook 5
about 480 MB (twenty subjects); notebooks 2, 3 and 4 need no download at all
(notebook 2 can optionally pull an open-weight model).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
REPO = "https://github.com/berdakh/onset-hfo.git"
BRANCH = "master"   # the repository default branch

SETUP = f'''# Colab setup. On your own machine, skip this cell and run
#   pip install -e ".[dev]"  from the onset-hfo directory instead.
import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO = "{REPO}"
BRANCH = "{BRANCH}"   # change to the default branch once this work is merged

if IN_COLAB and not os.path.exists("onset-hfo"):
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, "--depth", "1", REPO], check=True)
if IN_COLAB:
    os.chdir("/content/onset-hfo/onset-hfo")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
print("working directory:", os.getcwd())'''


SETUP_ML = SETUP.replace('"-q", "-e", "."', '"-q", "-e", ".[ml]"').replace(
    'pip install -e ".[dev]"', 'pip install -e ".[dev,ml]"')

_COUNTER = itertools.count(1)


def _cell_id() -> str:
    """nbformat 4.5 requires a stable id on every cell."""
    return f"cell-{next(_COUNTER):03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {},
            "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def write(name: str, cells: list[dict]) -> Path:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    return path


def badge(name: str) -> str:
    url = f"https://colab.research.google.com/github/berdakh/onset-hfo/blob/{BRANCH}/onset-hfo/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


# =========================================================================
# 1. Detection quickstart, on real public data
# =========================================================================

NB1 = "01_hfo_detection_quickstart.ipynb"
nb1 = [
    md(f"""
# Onset-HFO 1 — Detecting ripples and spikes in real intracranial EEG

{badge(NB1)}

**What you will do in the next ten minutes.** Download one minute of a real
intracranial EEG recording from a public archive, filter it the way an
epileptologist's software would, run two classical high-frequency-oscillation
(HFO) detectors and one interictal-spike detector over it, look at what they
found, and produce a structured report that cites the exact signal windows
behind every number.

**Why this matters.** In epilepsy surgery the question is *which tissue is
generating the seizures*. Ripples (80–250 Hz oscillations lasting a few tens
of milliseconds) and interictal epileptiform discharges are two of the
markers used to answer it. Counting them by eye across 100 channels is
impossible, so detection is automated — and automated detectors are famously
easy to fool.

**What this is not.** Not a medical device, not validated, not a diagnosis.
It is a transparent first prototype whose numbers you can check.

**The vocabulary**, if any of it is new:

| term | meaning |
|---|---|
| iEEG / ECoG / SEEG | electrodes recording *inside* the skull: grids on the surface, depth electrodes in tissue |
| contact | one recording point on an electrode, e.g. `ATT3` |
| bipolar pair | the difference between two neighbouring contacts, e.g. `ATT3-ATT4` — the channel a detector actually sees |
| HFO / ripple | an 80–250 Hz oscillation of a few tens of milliseconds |
| IED / spike | an interictal epileptiform discharge: a sharp 20–70 ms transient |
| ictal / interictal | during a seizure / between seizures |
| SOZ | seizure onset zone — what clinicians try to find, and what this prototype does **not** claim to identify |
"""),
    code(SETUP),
    md("""
## 1. The data

[OpenNeuro **ds003029**](https://openneuro.org/datasets/ds003029) —
*Epilepsy-iEEG-Multicenter-Dataset*, CC0 licensed, four clinical centres,
100 subjects, BIDS format, with clinician markers for seizure onset and
offset.

We use subject `sub-pt01`, an ECoG recording sampled at 1000 Hz, and we take
**60 seconds** of it: 26 s before the marked electrographic onset and 34 s
after. The loader downloads only those seconds (~24 MB) by asking the archive
for a byte range of the binary file — the whole run is 105 MB and the whole
dataset is far larger.

One consequence of 1000 Hz sampling, stated up front: the Nyquist frequency
is 500 Hz, so **ripples (80–250 Hz) can be analysed and fast ripples
(250–500 Hz) cannot**. The pipeline refuses to pretend otherwise.
"""),
    code("""
from onset_hfo.datasets import fetch_slice, list_runs

# What else is available for this subject (not downloaded, just listed):
print(list_runs("sub-pt01").to_string(index=False))

recording = fetch_slice(subject="sub-pt01", task="ictal", run="01",
                        t_start=50.0, t_stop=110.0)
"""),
    code("""
import json
print(json.dumps(recording.provenance(), indent=2)[:1200])
print("\\nClinician free-text markers name these contacts near onset:")
print(recording.marked_contacts)
"""),
    md("""
Those contact names come from what the reviewer typed during the seizure
(`"AD1-4, ATT1,2"`). They are **not** a curated seizure-onset-zone label — the
dataset does not ship one — and we will treat them accordingly later.

## 2. What the raw signal looks like

Before any processing. Note the amplitude scale (hundreds of microvolts) and
that the interesting structure is invisible at this zoom: ripples are 30 ms
long and buried under much larger slow activity.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np

sf = recording.sfreq
data = recording.raw.get_data(picks=["ATT6", "ATT7", "PST2"]) * 1e6   # volts -> microvolts
t = recording.t_offset + np.arange(data.shape[1]) / sf

fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
for ax, row, name in zip(axes, data, ["ATT6", "ATT7", "PST2"], strict=False):
    ax.plot(t, row, linewidth=0.5, color="#0b0b0b")
    ax.set_ylabel(name)
    ax.axvline(recording.seizure[0], color="#e34948", linewidth=1.5)
axes[-1].set_xlabel("time (s in the original recording); red line = marked seizure onset")
fig.tight_layout()
"""),
    md("""
## 3. Preprocessing

Four steps, each of which is recorded in the report so a reader knows exactly
what the detector saw:

1. keep intracranial channels, drop DC/trigger/ECG and the channels the
   dataset flagged `bad` (white matter, outside the brain, noisy);
2. high-pass at 1 Hz (drift);
3. notch at 60 Hz and harmonics — **narrow** notches, because the 180 Hz and
   240 Hz harmonics sit inside the ripple band and a wide notch would carve a
   hole in the signal we want to measure;
4. **bipolar montage**: subtract neighbouring contacts on the same electrode.
   This is standard for HFO work — a shared reference spreads its own noise
   across every channel and produces "HFOs" that appear everywhere at once.
"""),
    code("""
from onset_hfo.preprocess import prepare

prepared = prepare(recording)
print("\\nchannels:", prepared.ch_names[:8], "...", prepared.n_channels, "total")
"""),
    md("""
## 4. The detectors

Two HFO detectors, deliberately simple and deliberately different:

* **RMS energy** (Staba et al., 2002) — band-pass 80–250 Hz, root-mean-square
  in a 3 ms window, threshold at 5 robust SDs of the channel's own energy,
  and then the criterion that makes it an *oscillation*: at least 6 rectified
  peaks above a secondary threshold.
* **Line length** (Gardner et al., 2007) — the same pipeline with the mean
  absolute sample-to-sample change instead of energy. It responds to fast
  low-amplitude activity that RMS misses, and to sharp edges that RMS ignores.

Plus an **interictal discharge detector**: amplitude and sharpness in the
5–60 Hz band, with an explicit check against discontinuities (an electrode pop
looks like a beautiful spike once you filter it).

Two detectors rather than one, because *a single detector cannot disagree with
anything*, and disagreement is information this prototype reports instead of
hiding.
"""),
    code("""
from onset_hfo.pipeline import run_pipeline

result = run_pipeline(recording, save_to="artifacts/results")
"""),
    md("""
Notice the two numbers printed for each detector: **candidates** and
**accepted**. Every candidate is kept in the table with the reason it was
rejected. The rejection stage is the difference between a prototype and a
demo — see section 6.

## 5. What did it find?
"""),
    code("""
result.rates["rms"].head(8).round(2)
"""),
    code("""
from onset_hfo.viz import plot_channel_rates, plot_rate_timecourse, plot_detector_comparison

fig = plot_channel_rates(result.rates["rms"], top_k=15, label="rms")
"""),
    md("""
Read the confidence intervals, not just the bars. Sixty seconds of recording
turns a rate into a small count: 30 events/min *is* 30 events, and the
interval around it is wide. Two channels whose intervals overlap are not
ranked, they are tied.
"""),
    code("""
top = list(result.rates["rms"]["channel"].head(3))
fig = plot_rate_timecourse(result.events, prepared.t_offset,
                           prepared.t_offset + prepared.duration,
                           channels=top, seizure=recording.seizure)
"""),
    md("""
The shaded region is the clinician-marked seizure. In this recording the
detected ripple rate on the leading channels is essentially zero before onset
and high during the seizure — these are *ictal* HFOs.

That is worth pausing on, because it is also a limitation: clinical HFO work
is usually done on **interictal** recordings (between seizures, often during
sleep), where a high rate is thought to mark epileptogenic tissue. This
dataset only publishes ictal snapshots, so what we measure here is "where the
seizure is loudest in the ripple band", which is a different claim.
"""),
    code("""
fig = plot_detector_comparison(result.comparison, "rms", "line_length")
"""),
    code("""
# The channels the two detectors rank very differently, stated rather than averaged:
result.comparison[result.comparison["disagrees"]][
    ["channel", "rank_rms", "rank_line_length", "rank_gap",
     "rate_per_min_rms", "rate_per_min_line_length"]].round(1)
"""),
    md("""
## 6. One event, up close

This is the figure that makes a detection arguable. Three panels: the
wideband signal, the band-passed signal, and the spectrum of the event window
against the recording's own 1/f background.

A **real oscillation** leaves a bump in the spectrum. **Filter ringing** —
what a sharp transient becomes after an 80–250 Hz filter — does not: its
spectrum is just the background pushed up. That difference is the single most
important quality check in HFO detection, and it is why the pipeline measures
*spectral prominence* for every event.
"""),
    code("""
from onset_hfo.viz import plot_event

best = result.evidence(top[0], k=1)[0]
fig = plot_event(result.prepared, best)
"""),
    code("""
# ... and an event the validation stage threw away, with its reason:
rejected = sorted([e for evs in result.events.values() for e in evs if not e.accepted],
                  key=lambda e: -e.score)
print(rejected[0].reject_reason)
fig = plot_event(result.prepared, rejected[0])
"""),
    md("""
## 7. The report

A structured object, not prose: findings with rates, ranks, confidence
intervals and evidence windows; disagreements between detectors stated and
left unresolved; data quality; method; limitations. There is **no
recommendation field** — not empty, absent.
"""),
    code("""
from IPython.display import Markdown
Markdown(result.report.to_markdown())
"""),
    md("""
## 8. A weak reality check

This ictal recording has no HFO markings. (The interictal dataset does --
see notebook 3 and `onset-hfo benchmark`.) What we have here is the list of
contacts the clinician named in the seizure markers. Are our top-ranked
channels drawn from them more often than chance?

The honest way to ask is a permutation test, and the honest way to read the
answer is with both hands: the markers are not an SOZ label, they are
incomplete, and this is a 60-second ictal window.
"""),
    code("""
from onset_hfo.evaluate import marked_contact_check

check = marked_contact_check(result.rates["rms"], recording.marked_contacts, k=10)
print(json.dumps(check, indent=2))
"""),
    md("""
Whatever this prints, do not over-read it. If the overlap is no better than
chance that is unsurprising for ictal data and a rate-only ranking; if it is
better, it is mild encouragement and nothing more. Precision and recall are
measured in notebook 3, on synthetic data where the truth is known.

## 9. Where the results went

Everything is saved as CSV and JSON. That directory is also the *only* thing
the agent in notebook 2 can read.
"""),
    code("""
import os
out = "artifacts/results/sub-pt01_ictal_run-01"
print("\\n".join(sorted(os.listdir(out))))
"""),
    md("""
## What to try next

* Change the window (`t_start` / `t_stop`) to a purely pre-ictal stretch and
  see whether the ranking changes. If a channel's rank depends on which minute
  you picked, that is worth knowing before anyone builds on it.
* Raise `threshold_sd` from 5 to 7 in `PipelineConfig` and watch the rates
  fall. Which channels survive?
* Run another subject (`list_runs("sub-jh103")`) and compare.
* Open **notebook 2** to ask an open-weight language model about these
  results, and **notebook 3** to measure how good the detectors actually are.

**Reading**: `docs/METHODS.md` (the algorithms and their references),
`docs/DATA.md` (what is and is not in this dataset), `docs/LIMITATIONS.md`
(what this prototype must not be used for).
"""),
]

# =========================================================================
# 2. The agent
# =========================================================================

NB2 = "02_agentic_analysis.ipynb"
nb2 = [
    md(f"""
# Onset-HFO 2 — An evidence-only agent on an open-weight model

{badge(NB2)}

Notebook 1 produced tables. This notebook puts a language model in front of
them — carefully.

**The design in one paragraph.** The agent can call eight read-only tools
over one saved analysis. It cannot run a detector, change a threshold, open a
file or reach the network. Every factual claim it makes must cite an
`evidence_id` that a tool actually returned, and every number it states must
appear in a tool result. Those two checks run *after* the model answers; if
either fails, the answer is discarded and the agent says it cannot answer.
Questions about treatment, diagnosis or another patient are refused *before*
the model is called at all.

**Why bother with an agent then?** Because choosing which question to ask of a
dataset — "is this channel's lead real, or do the detectors disagree about
it?" — is exactly the kind of multi-step, ill-specified work a language model
is good at, and exactly the kind of work that must never be allowed to invent
a number. The split is: the model decides *what to look up and how to say
it*; the pipeline decides *what is true*.

Everything runs on **open weights** (Qwen2.5-Instruct by default). Nothing in
this project requires a hosted proprietary model.
"""),
    code(SETUP),
    md("""
## 1. Get an analysis to talk about

If you ran notebook 1, its results are already on disk. If not, this cell
builds one from the synthetic recording in a few seconds — no download.
"""),
    code("""
import os
from onset_hfo.store import ResultStore

REAL = "artifacts/results/sub-pt01_ictal_run-01"
if os.path.exists(os.path.join(REAL, "events.csv")):
    results_dir = REAL
else:
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.synthetic import make_synthetic_recording
    recording = make_synthetic_recording(duration_s=60)
    results_dir = str(run_pipeline(recording).save("artifacts/results"))

store = ResultStore(results_dir)
store
"""),
    md("""
## 2. What the agent can see

The store is the whole world the agent has access to. Look at the tools it
exposes — the descriptions below are literally what the model is shown.
"""),
    code("""
from onset_agent.tools import TOOLS

for name, tool in TOOLS.items():
    args = ", ".join(tool.parameters["properties"]) or "-"
    print(f"{name:26s} args: {args}")
    print(f"{'':26s} {tool.description[:110]}...")
"""),
    code("""
# There is no tool that takes a patient argument: scope belongs to the
# application, not to model output. And there is no tool that writes anything.
import json
print(json.dumps(TOOLS["get_evidence"].schema(), indent=2))
"""),
    md("""
## 3. First, with no model at all

The `scripted` backend is a deterministic keyword policy that speaks the same
tool-calling protocol. It is **not** a language model — it exists so the
machinery can be demonstrated and tested offline, and so you can see the loop
before any weights are downloaded.
"""),
    code("""
from onset_agent import OnsetAgent, make_backend

agent = OnsetAgent(store, make_backend("scripted"))
answer = agent.ask("Which channels have the highest ripple rate?")
print(answer)
"""),
    code("""
answer = agent.ask("What is the evidence for the top channel?")
print(answer.text)
print("\\ncitations:", answer.evidence_ids)
print("resolved:", [store.resolve(e) is not None for e in answer.evidence_ids])
"""),
    md("""
## 4. Now with a real open-weight model

Two ways, pick one.

**A. Ollama** (best on your own machine, also works in Colab):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
ollama serve &
```

then `make_backend("ollama", model="qwen2.5:7b-instruct")`.

**B. Hugging Face transformers** (in-process, no server — the easy path in
Colab). On a free CPU instance use the 1.5B model; with a GPU runtime use the
7B, which is much better at tool calling.

The cell below tries transformers and falls back to the scripted policy if the
packages or the weights are unavailable, so the notebook always runs.
"""),
    code("""
USE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # try "Qwen/Qwen2.5-7B-Instruct" on a GPU runtime

backend = None
try:
    import torch, transformers  # noqa: F401
    backend = make_backend("transformers", model=USE_MODEL)
    print("using", backend.describe())
except Exception as exc:
    print("could not load a language model:", type(exc).__name__, exc)
    print("falling back to the scripted policy (install with: pip install -e '.[llm]')")
    backend = make_backend("scripted")

llm_agent = OnsetAgent(store, backend, verbose=True)
"""),
    code("""
answer = llm_agent.ask("Which channels have the highest ripple rate, and how sure can we be?")
print("\\n", answer.text)
print("tools:", answer.tools_called, "| verified:", answer.verified)
"""),
    md("""
If you are running the 1.5B model, there is a good chance the answer above was
*refused* rather than produced. That is not a bug in the notebook — it is the
system working. A small model fumbles the tool protocol or states a number it
did not retrieve, the guards catch it, and the agent declines instead of
publishing a plausible fabrication. Swap in the 7B model and the same
questions start going through.

This is the most useful thing this notebook can show you: **what the failure
looks like, and that it is contained.**

## 5. The refusals

Three kinds, all checked before the model runs.
"""),
    code("""
for question in ["Which contacts should we resect?",
                 "Does this patient have epilepsy?",
                 "What did you find in patient sub-pt02?",
                 "Where do the seizures start?"]:
    a = agent.ask(question)
    print(f"Q: {question}\\n   [{'refused' if a.refused else 'answered'}] {a.text}\\n"
          f"   tools called: {a.tools_called or 'none'}\\n")
"""),
    md("""
Note the last line of each: no tool was called. The refusal costs nothing and
cannot be argued out of, because there is no model in the path to argue with.

## 6. Watching a guard catch a fabrication

Let us make a model that lies, and check that the system holds. This backend
retrieves real data and then states a rate nobody measured.
"""),
    code("""
import json
from onset_agent.backends import AssistantMessage, Backend, ToolCall

class LyingBackend(Backend):
    name = "deliberately-wrong"
    def __init__(self): self.turn = 0
    def chat(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return AssistantMessage(tool_calls=[ToolCall("top_channels", {"k": 3}, "c1")])
        return AssistantMessage(content=json.dumps({
            "answer": "The leading channel fires at 999.9 events/min, far above the others.",
            "evidence_ids": ["sub-xx|MADE-UP|rms|0.000"]}))

bad = OnsetAgent(store, LyingBackend())
answer = bad.ask("How often does the top channel fire?")
print(answer.text, "\\n")
print(json.dumps([s for s in answer.trace if s["type"] == "answer"], indent=2))
"""),
    md("""
Both problems are named: the citation was never returned by a tool, and the
number appears in no tool result. The answer is dropped. Note what the agent
says instead — it points at the report, which *is* authoritative.

## 7. The trace

Every answer carries the full sequence of tool calls, arguments and
verification outcomes. This is what you would show a reviewer who asks "where
did this sentence come from?".
"""),
    code("""
answer = agent.ask("Where do the two detectors disagree?")
print(answer.text, "\\n")
print(json.dumps(answer.as_dict(), indent=2)[:1800])
"""),
    md("""
## 8. The full demonstration set
"""),
    code("""
from onset_agent.prompts import EXAMPLE_QUESTIONS

for a in agent.ask_many(EXAMPLE_QUESTIONS):
    status = "REFUSED" if a.refused else "answer "
    print(f"[{status}] {a.question}\\n           {a.text[:220]}\\n")
"""),
    md("""
## What to try next

* Ask something the tools genuinely cannot answer ("what is the patient's
  age?") and watch it decline rather than guess.
* Add a tool in `onset_agent/tools.py` — say, one that returns the rejected
  events and their reasons — and see the model start using it. The schema is
  the only thing the model knows about it.
* Try a different open-weight model (`llama3.1:8b-instruct`,
  `mistral-nemo`) and compare how often the guards fire. That number is a
  useful, cheap benchmark of a model's tool discipline.
* Read `docs/AGENT.md` for the threat model — including why tool *results*
  are treated as data and never as instructions.
"""),
    md("""
---
## Where this goes next

This notebook's agent *reads* an analysis somebody already ran. The next one
lets it **choose** the analysis -- re-running a detector at a stricter
threshold to see whether a finding survives -- then adds a learned
per-contact model with calibrated uncertainty that the planner can stop on,
and a set of tests that try to make the whole thing confidently wrong.

**[Notebook 4 -- orchestration and localization](04_orchestration_and_localization.ipynb)**
"""),
]

# =========================================================================
# 3. Validation
# =========================================================================

NB3 = "03_validation_and_benchmark.ipynb"
nb3 = [
    md(f"""
# Onset-HFO 3 — How good are these detectors, actually?

{badge(NB3)}

The public recording has no HFO labels. Nobody has marked every ripple in it,
so on real data we can report rates, rankings and agreement, but **not**
precision and recall. Claiming otherwise would be the most common way
detection papers mislead.

So we build a recording where the truth is known: 1/f background, mains noise,
implanted ripples at a controlled signal-to-noise ratio, interictal discharges
(some with ripples riding on them, as in real epileptic tissue), and — most
importantly — **the traps**: large sharp transients that ring through an
80–250 Hz filter and look exactly like ripples.

Everything in this notebook runs offline in about a minute.
"""),
    code(SETUP),
    md("""
## 1. Build a labelled recording
"""),
    code("""
from onset_hfo.synthetic import make_synthetic_recording

recording = make_synthetic_recording(duration_s=60, seed=7)
recording.ground_truth.groupby("kind").agg(
    n=("kind", "size"), mean_amplitude_uv=("amplitude_uv", "mean")).round(1)
"""),
    code("""
print("\\n".join(recording.notes))
"""),
    md("""
The "hot" contacts carry ~20x the event rate of the others, and their rate
rises again inside a simulated seizure window. A detector that cannot find
them is not worth discussing.

## 2. Look at the traps

Three windows: a clean implanted ripple, a discharge, and an artifact. The
artifact has no oscillation of its own — but watch what the band-pass does
to it.
"""),
    code("""
import matplotlib.pyplot as plt
import numpy as np
from onset_hfo.detectors.base import bandpass

sf = recording.sfreq
raw = recording.raw.get_data() * 1e6
names = list(recording.ch_names)
gt = recording.ground_truth

fig, axes = plt.subplots(3, 2, figsize=(11, 6.5))
for row, kind in enumerate(["ripple", "spike", "artifact"]):
    event = gt[gt["kind"] == kind].iloc[3]
    ch = names.index(event["contact"])
    i0 = int((event["start"] - 0.05) * sf); i1 = int((event["stop"] + 0.05) * sf)
    seg = raw[ch, i0:i1]
    t = np.arange(seg.size) / sf * 1000
    axes[row, 0].plot(t, seg, color="#0b0b0b", linewidth=1)
    axes[row, 0].set_ylabel(kind)
    axes[row, 1].plot(t, bandpass(seg[None, :], sf, (80, 250))[0], color="#eb6834", linewidth=1)
axes[0, 0].set_title("wideband")
axes[0, 1].set_title("80-250 Hz - the detector's view")
axes[2, 0].set_xlabel("ms"); axes[2, 1].set_xlabel("ms")
fig.tight_layout()
"""),
    md("""
The bottom right panel is the whole problem in one picture: a step with no
oscillation in it becomes a burst of 80–250 Hz "activity" after filtering.
Any energy-threshold detector will report it. The only way to tell it apart
from the top right panel is to look at the *spectrum* — a real oscillation
leaves a bump above the 1/f background, ringing does not.

## 3. Run the pipeline and score it
"""),
    code("""
from onset_hfo.evaluate import evaluate_detections
from onset_hfo.pipeline import run_pipeline

result = run_pipeline(recording, verbose=False)
for name, events in result.events.items():
    print(evaluate_detections(events, recording.ground_truth, detector=name).summary())
print(evaluate_detections(result.spikes, recording.ground_truth,
                          detector="spike", kind="spike").summary())
"""),
    md("""
Recall around 0.5 for the HFO detectors is not a bug to be tuned away. A
5-SD threshold detector finds the ripples that are clearly above the noise
and misses the marginal ones; the next cell shows exactly that trade-off.
What *would* be a bug is high recall bought with false positives, which is
what the artifact rejection stage prevents — see section 5.

## 4. What the threshold buys you
"""),
    code("""
from onset_hfo.detectors import detect_rms
from onset_hfo.evaluate import sweep_threshold
from onset_hfo.preprocess import prepare

prepared = prepare(recording, verbose=False)
sweep = sweep_threshold(prepared, recording.ground_truth, detect_rms,
                        values=[3, 4, 5, 6, 7, 8], name="rms")
sweep[["threshold_sd", "n_detections", "precision", "recall", "f1"]].round(3)
"""),
    code("""
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sweep["threshold_sd"], sweep["recall"], marker="o", color="#2a78d6", label="recall")
ax.plot(sweep["threshold_sd"], sweep["precision"], marker="s", color="#eb6834", label="precision")
ax.plot(sweep["threshold_sd"], sweep["f1"], marker="^", color="#1baf7a", label="F1")
ax.axvline(5, color="#52514e", linestyle="--", linewidth=1)
ax.text(5.05, 0.05, " default (5 SD)", color="#52514e", fontsize=9)
ax.set_xlabel("detection threshold (robust SDs)"); ax.set_ylabel("score")
ax.set_ylim(0, 1.05); ax.legend(frameon=False)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.grid(color="#dcdbd6", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout()
"""),
    md("""
This curve is the honest description of a threshold detector: not "F1 = 0.74"
but "here is the trade-off, and here is why we chose 5". Quote the curve, not
the point.

## 5. Does artifact rejection earn its place?

The pipeline throws away a third of its own detections. That is only
justifiable if the ones it throws away are worse than the ones it keeps.
"""),
    code("""
from onset_hfo.evaluate import validation_benefit

validation_benefit(result.events["rms"], recording.ground_truth).round(3)
"""),
    md("""
Precision roughly 0.6 before, roughly 0.98 after; recall falls by a couple of
points. The `false_positive_causes` column names what was being confused —
`artifact` for filter ringing, `spike` for a discharge misread as an
oscillation. If that column ever showed mostly `background`, the detector
would be firing on noise and the thresholds would need to move.

## 6. Which criterion does the work?

Each check can be switched off. If a criterion costs recall and buys nothing,
it should be deleted rather than defended.
"""),
    code("""
import pandas as pd
from onset_hfo.config import ValidationConfig
from onset_hfo.validate import validate_events

rows = []
for label, cfg in [
    ("no rejection at all", None),
    ("cycles only", ValidationConfig(min_peak_prominence_db=-99)),
    ("spectral peak only", ValidationConfig(min_cycles=0)),
    ("both (default)", ValidationConfig()),
]:
    events = detect_rms(prepared)
    if cfg is not None:
        validate_events(events, prepared, cfg)
    scores = evaluate_detections(events, recording.ground_truth, accepted_only=cfg is not None)
    rows.append({"stage": label, "kept": sum(e.accepted for e in events) if cfg else len(events),
                 "precision": round(scores.precision, 3), "recall": round(scores.recall, 3),
                 "f1": round(scores.f1, 3)})
pd.DataFrame(rows)
"""),
    md("""
## 7. Sensitivity to the recording, not just the settings

One seed is an anecdote. Three seeds with different noise and event placement
tell you whether the numbers are a property of the detector or of the
recording it happened to see.
"""),
    code("""
rows = []
for seed in (1, 7, 42):
    rec = make_synthetic_recording(duration_s=60, seed=seed, verbose=False)
    res = run_pipeline(rec, verbose=False)
    for name, events in res.events.items():
        s = evaluate_detections(events, rec.ground_truth, detector=name)
        rows.append({"seed": seed, "detector": name, "precision": s.precision,
                     "recall": s.recall, "f1": s.f1})
    s = evaluate_detections(res.spikes, rec.ground_truth, detector="spike", kind="spike")
    rows.append({"seed": seed, "detector": "spike", "precision": s.precision,
                 "recall": s.recall, "f1": s.f1})
scores = pd.DataFrame(rows)
scores.groupby("detector")[["precision", "recall", "f1"]].agg(["mean", "std"]).round(3)
"""),
    md("""
## 8. How hard is the problem? Recall against SNR

The one parameter that matters most is how far the ripple stands above the
background. This is the curve to quote when someone asks "will it work on our
data?" — the answer depends on their signal-to-noise ratio, not on ours.
"""),
    code("""
rows = []
for snr in (4, 5, 6, 7, 9, 12):
    rec = make_synthetic_recording(duration_s=45, seed=3, ripple_snr=snr, verbose=False)
    res = run_pipeline(rec, verbose=False)
    s = evaluate_detections(res.events["rms"], rec.ground_truth, detector="rms")
    rows.append({"ripple_snr": snr, "precision": s.precision, "recall": s.recall, "f1": s.f1})
snr_curve = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(snr_curve["ripple_snr"], snr_curve["recall"], marker="o", color="#2a78d6", label="recall")
ax.plot(snr_curve["ripple_snr"], snr_curve["precision"], marker="s", color="#eb6834", label="precision")
ax.set_xlabel("implanted ripple amplitude / band-limited background RMS")
ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.grid(color="#dcdbd6", linewidth=0.6); ax.set_axisbelow(True)
fig.tight_layout()
snr_curve.round(3)
"""),
    md("""
## What this does and does not establish

**Does:** the detectors find implanted oscillations well above chance; the
artifact rejection stage raises precision sharply at a small cost in recall;
the spike detector's discontinuity check removes the electrode-pop
false positives; the numbers are stable across recordings.

**Does not:** say anything about clinical performance. Real ripples are not
Gaussian-windowed sinusoids, real artifacts are more varied and more
creative, and real recordings have physiological ripples in healthy tissue
that no simulator here models. These scores are a floor on sanity, not a
measure of clinical utility.

The next step for this project — see `docs/ROADMAP.md` — is a small
hand-annotated subset of real recordings. That is the only thing that
converts these numbers into a claim about patients.
"""),
]




# =========================================================================
# 4. The model drives the analysis, and a model learns from the cohort
# =========================================================================

NB4 = "04_orchestration_and_localization.ipynb"
nb4 = [
    md(f"""
# Onset-HFO 4 — The agent drives the analysis, and a model learns from the cohort

{badge(NB4)}

Notebooks 1–3 cover a **fixed pipeline** and an agent that can *read* what it
produced. This one covers the two halves added since: an agent that decides
*what to measure*, and a learned per-contact model with calibrated
uncertainty — plus the place where they meet.

**Nothing here downloads a recording.** The orchestration runs on simulated
data and the modelling runs on a cohort feature table committed to the
repository (269 KB), so the whole notebook executes in a free Colab instance
in a couple of minutes.

| part | what it shows |
|---|---|
| A | the agent re-running a detector at a stricter threshold — and changing its mind |
| B | every claim resolving to a run id, or being struck |
| C | a learned model, and the gap between subject-specific and deployable |
| D | calibration, and a conformal guarantee that does not survive a new patient |
| E | the coupling: the model as a tool the planner stops on |
| F | trying to break the whole thing |

**Read the honest summary first.** The learned model barely beats the ripple
rate it was built from. The interesting numbers here are gaps and failures,
not scores.
"""),
    code(SETUP_ML),
    md("""
---
## Part A — the agent chooses the analysis

The original agent had eight read-only tools over a *saved* analysis. It could
report that a channel had 54 ripples/min at a threshold somebody chose hours
ago. It could not ask the question a reviewer asks next:

> *Does that survive a stricter threshold?*

Every tool below executes the real pipeline at parameters the planner picks at
call time. Let's watch it happen.
"""),
    code("""
from onset_hfo.synthetic import make_synthetic_recording
from onset_agent.analysis import AnalysisSession, build_registry

recording = make_synthetic_recording(seed=7, duration_s=30, verbose=False)
session = AnalysisSession(recording)
tools = build_registry(session)

print("tools the planner may call:")
for name in tools.names():
    print("  -", name)
"""),
    md("""
### The worked example: survey, then challenge

A survey at the conventional 5 robust SD, then a re-test of the leader at 7.
Note the timing: the survey pays for the band-pass, the re-test is nearly
free, which is what makes re-planning practical rather than theoretical.
"""),
    code("""
survey = tools.run("detect_hfo", {"detector": "rms", "threshold_sd": 5, "k": 4})
leader = next(iter(survey.output["channels"]))
print(f"[{survey.run_id}] survey at 5 SD  ({survey.runtime_s:.2f} s)")
for ch, row in survey.output["channels"].items():
    print(f"    {ch:10s} {row['rate_per_min']:6.1f}/min   CI {row['rate_ci']}")

retest = tools.run("detect_hfo", {"channels": [leader], "threshold_sd": 7, "k": 4})
after = retest.output["channels"][leader]["rate_per_min"]
before = survey.output["channels"][leader]["rate_per_min"]
print(f"\\n[{retest.run_id}] {leader} re-tested at 7 SD  ({retest.runtime_s:.2f} s)")
print(f"    {before:.1f}/min  ->  {after:.1f}/min   (survives {after / before:.0%})")
"""),
    md("""
### Does the extra control change the answer?

Four rungs, the **same analyzers**, the same recording. Only who decides what
runs changes:

- **S0** a fixed sequence of calls, no model at all — the prior-work baseline
- **S1** the model picks the analyses once, never sees the results
- **S2** the model sees each result before choosing the next call
- **S3** as S2, plus every claim must resolve to a run id

A channel's score is its survey rate multiplied by how well that rate survived
any stricter threshold, capped at 1 — scrutiny can lower confidence, never
raise it. S0 and S1 never re-test, so they cannot produce a robustness below
1. That is the mechanism by which orchestration can change the answer at all.
"""),
    code("""
from onset_agent.planner import Rung, run_rung, ModelJudged

for rung in [Rung.S0, Rung.S1, Rung.S2, Rung.S3]:
    session.reset_memo()          # each rung pays for its own detections
    result = run_rung(session, rung, stop_rule=ModelJudged(max_calls=12), max_steps=12)
    top = [(r.channel, round(r.score, 1), round(r.robustness, 2)) for r in result.ranking[:3]]
    print(f"{rung.value}: {result.cost['n_tool_calls']:2d} calls, "
          f"{result.n_retested} channel(s) challenged")
    print(f"     top-3 (channel, score, robustness): {top}")
"""),
    md("""
Read the `robustness` column. The re-planning rungs demote a channel whose
rate collapses under a stricter threshold; the fixed rungs cannot, because
they never look again.

**A caution the repository insists on.** On the real `sub-pt01` recording this
same mechanism re-orders the top five and the re-ordering *means nothing* —
those channels sit between 70 and 83 events/min with heavily overlapping
intervals, so they are tied, not ranked. A mechanism that works is not the
same as a mechanism that helped.
"""),
    md("""
---
## Part B — every number resolves to a run, or it is struck

The evidence store is an append-only ledger of every tool call. A sentence
stating a number is admissible only if some tool produced it.
"""),
    code("""
from onset_agent.verifier import DeterministicVerifier

session.reset_memo()
result = run_rung(session, Rung.S3, stop_rule=ModelJudged(max_calls=10))
store = result.store

print(f"{len(store)} tool runs recorded\\n")
print(store.digest(max_runs=3, per_run_chars=110))
print("\\nwhere did 54.0 come from? ->", store.find_number(54.0))
print("where did 99999 come from? ->", store.find_number(99999) or "nowhere: unsupported")
"""),
    code("""
honest = result.report
fabricated = honest + " Channel ZZ1-ZZ2 reached 4210.5 ripples/min [hfo_001]."

check = DeterministicVerifier()(fabricated, store)
print(f"claims checked: {check.n_claims}   struck: {check.n_struck}")
for struck in check.struck:
    print("  STRUCK:", struck["sentence"])
    print("  reason:", struck["reason"])
print("\\nprovenance coverage of what survived:", round(check.provenance_coverage, 3))
"""),
    md("""
The research question is not whether verification removes hallucinations — it
does, by construction. It is what verification **costs**, because a verifier
also strikes true statements it cannot resolve. `compare_verifiers` runs both
the arithmetic checker and a language-model checker and reports the
off-diagonal cells: true statements the model removed, and fabrications it let
through.
"""),
    md("""
---
## Part C — a learned per-contact model

Everything above ranks channels by a threshold. Can a model trained on the
cohort do better?

The cohort: **22 subjects, 1466 channels, 301 labelled seizure-onset contacts**
from the archive's own clinical spreadsheet. The table is committed, so this
runs with no download.
"""),
    code("""
import pandas as pd
from onset_hfo.learn import SHIPPED_COHORT

features = pd.read_csv(SHIPPED_COHORT / "features.csv.gz")
print(f"{features['subject'].nunique()} subjects, {len(features)} channels, "
      f"{int(features['is_soz'].sum())} labelled SOZ "
      f"({features['is_soz'].mean():.1%} prevalence)")
features.groupby("site").agg(subjects=("subject", "nunique"),
                             channels=("channel", "size"), soz=("is_soz", "sum"))
"""),
    md("""
### Three protocols, and the gap between them is the result

| protocol | trains on | answers |
|---|---|---|
| **within-subject** | other contacts of the *same* patient | are the contacts separable at all? a **ceiling** |
| **leave-one-patient-out** | every *other* patient | the deployable number |
| **leave-one-site-out** | patients at the *other* hospitals | what survives a change of centre |

Within-subject is an upper bound, **not a deployable model**: using it would
require already knowing some of that patient's answer.

Class imbalance is severe, so AUPRC against the prevalence baseline is the
number to read — accuracy is meaningless and AUROC is optimistic.
"""),
    code("""
from onset_hfo.models import evaluate, baseline_scores

rows = [baseline_scores(features, col, "raw").summary()
        for col in ["rms_rate_per_min", "ll_rate_per_min"]]
for protocol in ["within_subject", "lopo", "loso"]:
    rows.append(evaluate(features, "gradient_boosting", "raw", protocol).summary())

table = pd.DataFrame(rows)[["protocol", "model", "prevalence", "auprc",
                            "auprc_lift_over_prevalence", "auroc", "precision_at_5"]]
table
"""),
    md("""
**Read the gap, not the best row.**

The learned model buys about one AUPRC point over simply ranking channels by
line-length rate — thirteen features and a cross-validation harness for almost
nothing. The within-subject ceiling is far above both: the features *are*
separable inside a recording, and most of that does not survive the move to a
new patient. Cross-site costs almost nothing on top of cross-patient, which
puts the transfer problem at the patient level rather than the hospital level.

### So how much does a clinician's handful of labels buy?

The ceiling needs labels you do not have. But a reviewer looking at a new
implantation can genuinely point at a few contacts. Below, the target patient
contributes *k* labelled contacts chosen **before anything is predicted**, and
the model is scored only on the contacts it was *not* given. At `k = 0` this
is exactly leave-one-patient-out, so both ends of the curve are the same
experiment.
"""),
    code("""
from onset_hfo.models import label_budget_curve

curve = label_budget_curve(features, budgets=(0, 2, 5, 10, 20), model="logistic",
                           normalisation="raw", n_repeats=3)
curve[["n_labels", "labelled_fraction", "auprc", "gap_closed", "precision_at_5"]]
"""),
    code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(curve["n_labels"], curve["auprc"], "o-", color="#3b6ea5", lw=2)
ax.axhline(curve["auprc"].iloc[0], ls="--", c="#999",
           label=f"leave-one-patient-out ({curve['auprc'].iloc[0]:.3f})")
ax.set_xlabel("contacts the clinician labels first")
ax.set_ylabel("AUPRC")
ax.set_title("What a handful of labels buys")
ax.legend(frameon=False, fontsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.tight_layout()
"""),
    md("""
Five contacts — under a tenth of a typical implantation — recovers a large
part of what is lost moving to a new patient. `labelled_fraction` is in the
table on purpose: *40 labels* sounds modest until it is 60% of the electrodes.
"""),
    md("""
---
## Part D — calibrated probabilities, and a guarantee that breaks

A ranking is not enough to act on. "These contacts, ranked" invites the reader
to draw their own line; "these six contacts, with 90% coverage" is a statement
with a guarantee attached.

Split conformal prediction gives that guarantee **without assuming the model
is right** — it assumes only that calibration and test data are
*exchangeable*. Patients are not.
"""),
    code("""
from onset_hfo.uncertainty import conformal_coverage_report, expected_calibration_error

lopo = evaluate(features, "gradient_boosting", "raw", "lopo")
print("expected calibration error (uncalibrated):",
      round(expected_calibration_error(lopo.scores, lopo.truth), 4))

report = conformal_coverage_report(lopo, alpha=0.1, seed=0)
for key in ["nominal_coverage", "empirical_coverage", "mean_set_size",
            "frac_singleton", "frac_ambiguous", "n_calibration_patients",
            "n_test_patients"]:
    print(f"  {key:24s} {report[key]}")
"""),
    md("""
**It undercovers, and that is the finding.** The guarantee does not fail
loudly — it produces tight-looking sets that cover less often than advertised,
because the calibration patients are not exchangeable with the test patients.

The clinically meaningful object is one number per patient: how many contacts
**cannot be ruled out**. It is brutal reading.
"""),
    code("""
pd.DataFrame(report["candidate_sets"])[
    ["subject", "n_contacts", "n_candidates", "candidate_fraction",
     "n_true_soz", "n_soz_covered"]]
"""),
    md("""
A patient whose candidate set is four contacts has an actionable result. A
patient whose set is forty has not been localized, however confident any
individual score looked — and one patient here has a set of zero, having ruled
out every contact including all the true ones.
"""),
    md("""
---
## Part E — the coupling

This is where the two halves meet, and it is deliberately unremarkable: the
learned model is **a tool in the registry like any other**, returning JSON.
The planner does not know how it works. What it consumes is the *width* of the
candidate set — and it keeps gathering evidence while that set is too wide to
act on.
"""),
    code("""
from onset_hfo.models import fit_soz_model
from onset_agent.planner import ConformalWidth

model = fit_soz_model(features, model="logistic", normalisation="raw", alpha=0.1)
print("model trained on", model.n_train_patients, "patients;",
      "conformal quantile", round(model.conformal_quantile, 3))

coupled = AnalysisSession(recording, soz_model=model)
result = run_rung(coupled, Rung.S2, stop_rule=ConformalWidth(max_width=3, max_calls=12))
print("\\nstopped because:", result.stop_reason)

soz_runs = result.store.runs("estimate_soz_probability", ok_only=True)
if soz_runs:
    out = soz_runs[-1].output
    print(f"candidate set: {out['candidate_set_size']}/{out['n_channels']} channels "
          f"at {out['nominal_coverage']:.0%} coverage")
"""),
    md("""
**Look at why it stopped.** The model was fitted on real ECoG from 22
patients and we have just pointed it at a *simulation* — a different montage,
a different sampling rate, a different everything. It finds neither label
plausible for any channel, so the candidate set comes back **empty**.

An empty set is not a narrow one. Writing this notebook is what surfaced the
bug: the rule originally treated zero candidates as "narrow enough to act on"
and stopped with a message that read like success. It now says what actually
happened. The most obvious failure mode of a deployed model should not be its
success condition.
"""),
    code("""
# Without a model the tool refuses with something the planner can act on,
# and everything else still runs. A missing model degrades the agent; it
# does not break it.
bare = build_registry(AnalysisSession(recording)).run("estimate_soz_probability", {})
print("ok:", bare.ok)
print(bare.error[:200])
"""),
    md("""
---
## Part F — trying to break it

Everything above measures how well something works. These measure whether it
can be made **confidently wrong**. Each test states its expectation before it
runs: a test that can be passed by any outcome is not a test.
"""),
    code("""
from onset_agent.falsify import run_falsification_suite
from onset_hfo.cohort import soz_labels

labels = soz_labels(recording.subject, recording=recording)
for check in run_falsification_suite(recording, labels, repeats=2):
    print(f"[{check.verdict}] {check.name}")
    print(f"        {check.reading}")
"""),
    md("""
One of these failed when it was first written. Given a simulation containing
**no epileptic contacts at all**, the pipeline ranked a channel at 6/min and
nothing in its output said the recording was empty. There was no null
hypothesis — every ranking function sorts noise.

The fix was not a threshold. A leader must be distinguishable from the
*median* channel's confidence interval, built from quantities the pipeline
already computed, so it tightens by itself as the window grows. Check that it
discriminates — a null that always fires is worthless:
"""),
    code("""
from onset_hfo.metrics import leader_separation
from onset_hfo.pipeline import run_pipeline

for label, rec in [("with implanted ripples", recording),
                   ("with nothing in it",
                    make_synthetic_recording(seed=3, duration_s=30, hot_leads=0,
                                             verbose=False))]:
    sep = leader_separation(run_pipeline(rec, verbose=False).rates["rms"])
    print(f"{label:24s} leader {sep['leader_rate_per_min']:5.1f}/min  "
          f"median {sep['median_rate_per_min']:4.1f}  "
          f"tied {sep['n_tied_with_leader']}/{sep['n_channels']}  "
          f"-> stands out: {sep['distinguishable']}")
"""),
    md("""
---
## What none of this establishes

- **Every orchestration number above comes from a deterministic scripted
  planner**, not a language model. It is the control, not the result. The
  anonymised-names falsification test exists specifically to catch a model
  reciting priors about electrode naming, and a script cannot fail it.
- **This is ictal data.** The clinical HFO literature measures *interictal*
  rate; every feature here comes from a 60-second window around a seizure.
- **The model is not clinical.** Fitted on 22 recordings from three centres,
  scored against a retrospective record, on patients whose outcome is already
  known.
- **Nothing here identifies a seizure onset zone.** The agent's guard still
  refuses SOZ questions; localization is an offline metric computed by code,
  never a claim the model is allowed to make.

`docs/ORCHESTRATION.md` and `docs/LOCALIZATION.md` carry the full design and
end with explicit "what is not done" sections.
"""),
]


# =========================================================================
# 5. The outcome study: HFOs against what happened to the patient
# =========================================================================

NB5 = "05_surgical_outcome_study.ipynb"
nb5 = [
    md(f"""
# Onset-HFO 5 — Did the HFO map point at the tissue whose removal cured the patient?

{badge(NB5)}

Every other notebook here compares an algorithm to another algorithm, or to a
human reading the same screen. This one compares it to **what happened to the
patient after surgery** — the only reference standard in epilepsy surgery that
is not another opinion.

**The dataset makes it possible.** OpenNeuro `ds003498` (Zurich, 20 patients,
interictal slow-wave sleep, 2000 Hz) ships three things in the same archive:
expert HFO markings per channel, the **resected contacts** for each patient,
and whether that patient became **seizure-free**. Almost no public iEEG
dataset carries all three.

**What you will do.** Test a published finding on whole recordings, watch our
own detector land close behind the expert markings, and then see the same
analysis on the first 60 seconds give a *stronger* answer than the full five
minutes — which is the most useful result in the notebook, because it says the
number is not stable yet.

**Runtime.** About 45 minutes on a free Colab instance, most of it downloading
~2.2 GB (every run is exactly 300 s, so this is the whole recording and every
expert marking in it). Everything is cached, so a second run is compute-only.
"""),
    code(SETUP),
    md("""
---
## 1. The three ingredients

Read them straight out of the archive before running anything, so you can see
what the study is actually made of.
"""),
    code("""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from onset_hfo.clinical import fetch_participants, resection_map

participants = fetch_participants("ds003498")
print(participants[["subject", "epilepsy", "ilae", "outcome",
                    "months_follow_up"]].to_string(index=False))
print("\\nseizure-free (S) vs recurrence (F):",
      participants["outcome"].value_counts().to_dict())
"""),
    md("""
`S` means **success** — seizure free. `F` means **failure** — seizures
returned. Reading `F` as "free" inverts every label in the cohort, which is
why `onset_hfo.clinical` never exposes the raw letter without the mapping
beside it.

Now the resected zone. The archive stores it as a clinician's free text, one
row per patient:
"""),
    code("""
resections = resection_map("ds003498")
for subject in ["sub-01", "sub-12", "sub-15"]:
    r = resections[subject]
    print(f"{subject}: rz text {r.rz_text!r}")
    print(f"          -> {len(r.resected)} contacts: {', '.join(r.resected[:8])}"
          + (" ..." if len(r.resected) > 8 else ""))
    if r.eloquent:
        print(f"          eloquent (excluded by the source study): {', '.join(r.eloquent)}")
    if r.unparsed:
        print(f"          COULD NOT PARSE: {r.unparsed}  <- reported, not silently dropped")
"""),
    md("""
That last line is the point of the `unparsed` field. Subject 15's sheet says
`1ll22-24` where every other token on the row says `tll` — a typo in the
original clinical data. A parser that quietly returned the other three ranges
would shrink that patient's excluded set with nothing to show for it.

## 2. Contacts are not channels

The sheet names **contacts** (`AHR1`). The analysis runs on **bipolar
channels** (`AHR1-AHR2`), each spanning two contacts. So a channel is inside
the resection, outside it, or — at the margin — straddling it:
"""),
    code("""
from onset_hfo.clinical import classify_channels
from onset_hfo.datasets import fetch_slice
from onset_hfo.preprocess import prepare

rec = fetch_slice(dataset="ds003498", subject="sub-01", run="01",
                  t_start=0, t_stop=60, verbose=False)
prep = prepare(rec, verbose=False)
labels = classify_channels(prep.ch_names, resections["sub-01"])

print(labels["zone"].value_counts().to_string())
print("\\nthe resection margin (one contact in, one out):")
print(labels[labels["zone"] == "partial"][["channel", "contact_a", "contact_b"]]
      .to_string(index=False))
print("\\nhow much of the resection this recording can even see:")
print(resections["sub-01"].coverage(prep.ch_names))
"""),
    md("""
**`partial` is kept as its own label on purpose.** Folding margin channels
into "resected" inflates every "we got it all" number; folding them into
"spared" inflates the opposite one. In the primary metric they sit in the
denominator and not the numerator, so a detector firing along the margin gets
no credit for it.

**`rz_coverage` is the honest denominator.** In five of the twenty patients —
all temporal-lobe cases, where the source study kept only the three most
mesial bipolar channels — only 4 of 16 resected contacts were recorded at all.
Their "share inside the resection" describes a quarter of a resection, and the
study writes that number to `recordings.csv` rather than burying it.

## 3. The design, before any numbers

Four choices, each of which can only make the result *worse*:

| Choice | Why |
|---|---|
| **An expert positive control** | Every number is computed twice — from the published expert markings and from our detector, on the same channels. Without it, a null is unreadable: it could be the detector or it could be 20 patients. Section 7 shows this control earning its keep. |
| **Operating points fixed in advance** | 2.0 SD for ripples, 5.0 SD for fast ripples, both chosen in `benchmark.py` on *channel-rank agreement with the experts* — a question that says nothing about surgery. Tuning a threshold against outcome and then reporting the outcome would be circular. |
| **Margin channels not claimed** | as above. |
| **Both channel scopes reported** | `reviewed` (what the annotators marked) and `all` (what a deployed tool would face). Reporting only the flattering one is a choice made after seeing both. |

## 4. Run it
"""),
    code("""
from onset_hfo.outcome import outcome_study

# The default window is the whole run (300 s). ~2.2 GB cold, ~25 minutes cached.
result = outcome_study(verbose=True)
result.save()
"""),
    md("""
## 5. The metric that carries the signal

Four metrics were computed. They disagree, and the disagreement is itself a
finding.
"""),
    code("""
pd.set_option("display.width", 220)
print("--- was the single busiest channel resected? ---")
print(result.summary("top_channel_resected")
      .query("scope == 'reviewed'").to_string(index=False))
print("\\n--- what fraction of all HFO events was inside the resection? ---")
print(result.summary("share_in_rz")
      .query("scope == 'reviewed'").to_string(index=False))
"""),
    md("""
Read the `expert` / `fast_ripple` row of the first table first. The busiest
fast-ripple channel was inside the resection in **11 of 13** patients who
became seizure-free and **3 of 7** whose seizures returned — AUC 0.71,
permutation p = 0.12. The direction is the one Fedele et al. 2017 predicts, and
with 13 patients against 7 it does not reach significance. Nothing in this
study does.

Then read the same row of the second table: `share_in_rz` shows **nothing** in
the expert arm — AUC 0.48, a coin flip — on the same events. **Concentration
localises; proportion does not.** "Most of this patient's HFOs were inside the
resection" is largely a statement about how big the resection was. "The one
place generating the most fast ripples was removed" is the clinically useful
sentence — and it means any report built on this pipeline should show a
*ranking*, not a percentage. This is the one conclusion in the notebook that
holds at every window length.

## 6. Where our detector stands
"""),
    code("""
print(result.verdict())          # the pre-specified comparison
print()
for key, text in result.verdicts().items():
    if key.startswith("top_channel"):
        print(f"{key}:\\n  {text}\\n")
"""),
    code("""
subjects = result.subjects.merge(result.participants[["subject", "outcome"]], on="subject")
view = subjects.query("band == 'fast_ripple' and scope == 'reviewed'")
print(view[["subject", "outcome", "source", "n_events",
            "share_in_rz", "top_channel_resected"]]
      .sort_values(["outcome", "subject", "source"]).to_string(index=False))
"""),
    md("""
**Our detector lands just behind the expert markings** — 10 of 13 versus 3 of
7, AUC 0.67 against their 0.71, with intervals that overlap almost entirely.
Both are null. Over 300 s every subject produces fast-ripple detections, so
unlike the 60-second version of this analysis no patient drops out.

Keep that 0.67-against-0.71 in mind for the next section, because on the first
minute of the same recordings the two numbers were 0.70 and **0.82**.

## 7. The window changes the answer

This is the most important cell in the notebook. Same code, same pre-specified
metric, same patients — only the analysis window differs.
"""),
    code("""
short = outcome_study(t_stop=60.0, bands=("fast_ripple",), verbose=False)
for label, res in [("first 60 s", short), ("whole 300 s", result)]:
    row = res.summary("top_channel_resected").query(
        "scope == 'reviewed' and band == 'fast_ripple'")
    print(f"--- {label} ---")
    print(row[["source", "mean_seizure_free", "mean_recurrence",
               "auc", "auc_lo", "auc_hi", "p_permutation"]].to_string(index=False))
"""),
    code("""
# Which patients moved, and in which direction?
key = ["subject", "source"]
q = "band == 'fast_ripple' and scope == 'reviewed'"
moved = (short.subjects.query(q)[key + ["n_events", "top_channel_resected"]]
         .merge(result.subjects.query(q)[key + ["n_events", "top_channel_resected"]],
                on=key, suffixes=("_60", "_300"))
         .merge(result.participants[["subject", "outcome"]], on="subject"))
print(moved[moved["top_channel_resected_60"] != moved["top_channel_resected_300"]]
      .to_string(index=False))
"""),
    md("""
**Five times the data, a weaker result.** The expert arm falls from AUC 0.82
(p = 0.007) to 0.71 (p = 0.12); ours barely moves, 0.70 to 0.67. Two patients
account for all of it: `sub-18` (a recurrence whose busiest channel turns out
to be *inside* the resection once you look past the first minute — a flip that
costs twice) and `sub-15` (a seizure-free patient going the other way).

Three things follow, and they are why this notebook exists:

- **The 60-second number should never have been the headline**, and it was:
  the project's README and landing page carried AUC 0.82, p = 0.007 until this
  was run. 60 s was chosen for download size before any outcome data was
  touched, and the metric and band were pre-specified — so this is not
  cherry-picking. It is something worse and more common: an analysis window
  short enough to change the conclusion, never checked.
- **"Which channel is busiest" is fragile.** It is an argmax over 6–65 channels
  whose Poisson rate intervals overlap. `metrics.py` already refuses to rank
  channels whose intervals overlap when it reports rates; this metric does not,
  and it should.
- **The positive control earned its place** — just not in the way it was
  designed to. It was built to tell "our detector is worse" apart from "this
  study is underpowered". The 60-second run looked like the first. The full run
  says the second.

## 8. Why the band matters, demonstrated

One more parameter that decides whether there is a signal at all. The first
version of this analysis used 2.0 SD in **both** bands, because that is what
the ripple benchmark prefers. Here is what that does in the fast-ripple band:
"""),
    code("""
wrong = outcome_study(threshold_sd=2.0,   # one threshold everywhere
                      bands=("fast_ripple",), verbose=False)
print("2.0 SD in both bands:")
print(wrong.summary("top_channel_resected").query("scope == 'reviewed'").to_string(index=False))
print("\\nmeasured per-band operating points:")
print(result.summary("top_channel_resected")
      .query("scope == 'reviewed' and band == 'fast_ripple'").to_string(index=False))
"""),
    md("""
At 2.0 SD the fast-ripple detector runs at precision 0.086 — a mean of 1,142
detections per 60 s against a mean of 228 expert-marked events. Nothing about
the outcome data was used to choose either threshold; both come from
channel-rank agreement with the experts, in
`notebooks/03_validation_and_benchmark.ipynb`. **One threshold for both bands
is a bug, not a simplification.**

## 9. Read this before quoting any of it

- **Thirteen versus seven is a very small study.** `min_detectable_auc(13, 7)`
  returns **0.85** — with these group sizes only a very large separation
  reaches 80% power. A p above 0.05 here means *underpowered*, not *no effect*.
- **Nothing in the full-run study reaches p < 0.05**, and the Bonferroni column
  in `groups.csv` corrects across 24 comparisons. Read every row as
  hypothesis-generating.
- **One full run of one night**, against several whole nights in the source
  study. The archive has 1–6 runs per subject; combining them is the obvious
  extension, and given §7 it should happen before any number here is called
  stable.
- **Retrospective, one centre, one surgical team.** The
  [HFO Trial](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(22)00311-8/fulltext)
  (Lancet Neurology 2022) tested HFO-guided resection prospectively and did not
  find the benefit retrospective series report. This is a retrospective series.
- **Reproducing a retrospective result is evidence that the analysis is sound,
  not that the clinical claim is.**

`docs/OUTCOME.md` carries the full design, every table, and the list of what
this cannot support.
"""),
    code("""
from onset_hfo.outcome import min_detectable_auc
print("smallest AUC detectable at 80% power with 13 vs 7:",
      min_detectable_auc(13, 7))
print("\\nfiles written:")
import pathlib
for f in sorted(pathlib.Path("artifacts/results/outcome_ds003498").iterdir()):
    print("  ", f.name)
"""),
]


# =========================================================================
# 7. The hand-annotated benchmark
# =========================================================================

NB7 = "07_annotation.ipynb"
nb7 = [
    md(f"""
# Onset-HFO 7 — Mark a few hundred windows, and find out what the reference is worth

{badge(NB7)}

Every accuracy number in this project is agreement with *another algorithm* —
the archive's own Morphology detector — or accuracy against a simulator that
can only measure what it simulates. "Precision 0.97 on synthetic ripples" is a
statement about the generator.

**This notebook turns that into a statement about recordings**, by having two
people mark real windows. It is the one measurement here that needs humans,
and it takes two reviewers an afternoon each.

**The three things that make such a benchmark worth doing** are all in the
sampling rather than the marking, and `onset_hfo.review` enforces them:

1. **Windows the detector never proposed are in the sample.** Review only the
   detector's own hits and you can measure precision and *never* recall — and
   the reference becomes a function of the thing you are scoring.
2. **The reviewer cannot see the detector's verdict.** The manifest carries
   four columns: an opaque id, a channel, and two times. The stratum and the
   detector's opinion go in a separate key file, named so that handing over
   the wrong one is a visible mistake.
3. **Agreement is chance-corrected.** Most windows are not ripples, so two
   reviewers who say "no" to everything agree ~90% of the time. Cohen's kappa,
   with a bootstrap interval, and the prevalence beside it.

**And the ceiling is the finding.** Two reviewers who agree at kappa 0.6 have
defined a reference no detector can be scored against more finely than that.
The notebook prints that sentence before it prints any score.

**Runtime.** Minutes of compute; the marking is however long you take.
Progress is appended to a CSV after every window, so you can close the tab.
"""),
    code(SETUP),
    md("""
## 1. Something to review

Any saved analysis will do. The shipped example is a real 60-second recording
from OpenNeuro `ds003029` and needs no download.
"""),
    code("""from onset_hfo.store import ResultStore
from onset_hfo.detectors.base import Event

store = ResultStore("data/example_analysis")
events = store.events
by_detector = {}
for name in ("rms", "line_length"):
    rows = events[events.detector == name]
    by_detector[name] = [
        Event(channel=r.channel, start=r.start, stop=r.stop, detector=name,
              band=(r.band_low, r.band_high), accepted=bool(r.accepted))
        for r in rows.itertuples()]

channels = sorted(events.channel.unique())
duration = float(store.provenance.get("slice_stop_s", 60)
                 - store.provenance.get("slice_start_s", 0))
print(f"{sum(len(v) for v in by_detector.values())} candidates on "
      f"{len(channels)} channels over {duration:.0f} s")"""),
    md("""
## 2. Draw a blinded, stratified sample

Equal numbers from three strata **by design, not by prevalence**: background
outnumbers candidates enormously in a real recording, and sampling it
proportionally would spend the whole afternoon on empty windows. The imbalance
is corrected when scoring, which is why the stratum is recorded.

Set `N_PER_STRATUM` to what your reviewers will actually finish. 70 each means
210 windows, which is roughly an afternoon.
"""),
    code("""from pathlib import Path
from onset_hfo.review import sample_windows, write_manifest

N_PER_STRATUM = 70
OUT = Path("artifacts/review")

windows = sample_windows(by_detector, duration_s=duration, channels=channels,
                         n_per_stratum=N_PER_STRATUM, seed=0)
manifest_path = write_manifest(windows, OUT / "manifest.csv")

import pandas as pd
print(pd.Series([w.stratum for w in windows]).value_counts().to_string())
print()
print("give the reviewer:", manifest_path)
print("do NOT give them:", manifest_path.with_suffix(".key.csv"))"""),
    md("""
## 3. Mark them

One window at a time: the wideband trace, the band-passed trace, and the
event's own spectrum. The question is whether it is an oscillation with a
spectral bump of its own, or a sharp transient the filter turned into one.

**`unsure` is a real answer.** Forcing a binary choice on an ambiguous window
manufactures agreement that is not there, and the analysis drops those
windows explicitly and counts them.

Set `REVIEWER` to your own initials. Re-running the cell resumes where you
stopped.
"""),
    code("""REVIEWER = "A"   # <- change per reviewer

import ipywidgets as widgets
from IPython.display import clear_output, display
import matplotlib.pyplot as plt

from onset_hfo.preprocess import prepare
from onset_hfo.review import AnnotationLog, read_manifest
from onset_hfo.viz import plot_event

manifest = read_manifest(manifest_path)
log = AnnotationLog(OUT / "annotations.csv", reviewer=REVIEWER)
todo = log.remaining(manifest)
print(f"{log.done} done, {len(todo)} to go")

# The signal behind the windows. `reload_spec` on a saved analysis says which
# slice to fetch; here we use the analysis the store was built from.
prep = prepare(store.recording(), verbose=False) if hasattr(store, "recording") else None"""),
    md("""
If `prep` is `None` above, the saved analysis does not carry its signal and
the windows cannot be drawn — run `python -m onset_hfo.cli run` first, or
point `ResultStore` at an analysis you produced locally.

The loop below draws one window and waits for a click.
"""),
    code("""out = widgets.Output()
state = {"i": 0}

def show():
    with out:
        clear_output(wait=True)
        if state["i"] >= len(todo):
            print(f"Done — {log.done} windows marked by {REVIEWER}.")
            return
        row = manifest[manifest.window_id == todo[state["i"]]].iloc[0]
        mid = 0.5 * (row.t_start + row.t_stop)
        window = Event(channel=row.channel, start=mid - 0.015, stop=mid + 0.015,
                       detector="", band=(80.0, 250.0))
        print(f"{row.window_id}   ({state['i'] + 1} of {len(todo)})")
        plot_event(prep, window, context_s=row.t_stop - row.t_start)
        plt.show()

def mark(label):
    def handler(_):
        log.record(todo[state["i"]], label)
        state["i"] += 1
        show()
    return handler

buttons = []
for label, style in (("hfo", "success"), ("not_hfo", "danger"), ("unsure", "warning")):
    b = widgets.Button(description=label, button_style=style)
    b.on_click(mark(label))
    buttons.append(b)

display(widgets.HBox(buttons), out)
show()"""),
    md("""
## 4. What the reference is worth

Run this once **both** reviewers have finished. The ceiling sentence comes
first on purpose: every score after it has to be read against that number.
"""),
    code("""from onset_hfo.review import agreement, ceiling_note

annotations = pd.read_csv(OUT / "annotations.csv")
stats = agreement(annotations)
print(ceiling_note(stats))
print()
for k, v in stats.items():
    print(f"  {k}: {v}")"""),
    md("""
## 5. Score the detector against it

Per stratum, never pooled: the strata were sampled in equal numbers rather
than in proportion, so a pooled figure would weight a background window as
heavily as a candidate and describe a recording that does not exist.

Both consensus rules are printed because they bracket the answer. Quoting
whichever is kinder is how a detector's precision gets published.
"""),
    code("""from onset_hfo.review import score_against_annotations, save_study

key = pd.read_csv(manifest_path.with_suffix(".key.csv"))
for rule in ("both", "either"):
    scores = score_against_annotations(by_detector["rms"], manifest, annotations,
                                       key, consensus=rule, detector="rms")
    print(f"consensus = {rule}")
    for stratum, s in scores["strata"].items():
        print(f"  {stratum:11s} n={s['n']:3d}  tp={s['true_positive']:3d} "
              f"fp={s['false_positive']:3d} fn={s['false_negative']:3d} "
              f"tn={s['true_negative']:3d}  reviewers called "
              f"{s['reviewer_hfo_rate']:.0%} of them HFOs")
    print()

save_study(OUT, stats, scores)
print("written to", OUT)"""),
    md("""
## What to do with it

**Publish the annotations.** The marks are the contribution — a scored
detector is reproducible from them, and nobody else has to spend the
afternoon. Include the manifest, the annotations and the key.

**Publish the kappa next to every score.** A detector reported at precision
0.8 against a reference whose reviewers agreed at kappa 0.4 has been measured
against something largely made of disagreement, and the number will not
survive a second pair of reviewers.

**Do not pool the strata.** The sample is not a recording.
"""),
]


def execute(path: Path, timeout: int = 5400) -> None:
    """Run a notebook in place and keep its outputs.

    The committed notebooks carry their outputs so that a reader browsing on
    GitHub sees the numbers without running anything. Regenerating a notebook
    without re-executing it silently deletes those outputs, so this is how a
    rebuild is meant to be done.
    """
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent.parent)}})
    client.execute()
    nbformat.write(notebook, path)


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    run = "--execute" in argv
    wanted = {int(a) for a in argv if a.isdigit()}
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for number, (name, cells) in enumerate(
            [(NB1, nb1), (NB2, nb2), (NB3, nb3), (NB4, nb4), (NB5, nb5),
             (None, None), (NB7, nb7)], 1):
        if name is None:
            # 06_agent_benchmark.ipynb is committed but was never added here.
            # Numbering is kept aligned so `--execute 7` means notebook 7.
            continue
        if wanted and number not in wanted:
            continue
        path = write(name, cells)
        print(f"wrote {path} ({len(cells)} cells)")
        if run:
            print(f"  executing {name} ...", flush=True)
            execute(path)
            print(f"  done: {name}")


if __name__ == "__main__":
    main()
