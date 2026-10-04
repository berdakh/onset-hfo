# The agent

## What it is

An assistant that answers questions about **one saved analysis** by calling
read-only tools over the files the pipeline wrote, and that is checked by code
both before and after the model runs.

```
question
   │
   ├─ scope check ─────────────► refuse (no model called)
   │   treatment · diagnosis · another patient
   ▼
system prompt + question
   │
   ├─► model picks a tool ──► strict validation ──► ResultStore query ──┐
   │        ▲                                                           │
   │        └───────────────── tool result as JSON ◄────────────────────┘
   │   (up to max_steps times)
   ▼
model returns JSON  {"answer", "evidence_ids"}  or  {"refusal"}
   │
   ├─ citation check: every id must have been returned by a tool in this session
   ├─ number check:   every number must appear in a tool result
   │
   ├─ pass ──► answer
   └─ fail ──► tell the model what was wrong, retry (twice), then refuse
```

## Why an agent at all

Reading a detection table well is genuinely multi-step work: look at the
ranking, notice that the two detectors disagree about the third channel, pull
the evidence windows, check whether those events co-occur with discharges,
qualify the answer with the confidence interval. A language model is good at
deciding which of those steps a given question needs.

It is also perfectly capable of writing "ATT6-ATT7 shows 84 ripples per
minute" when the table says 75. So the split is:

> **The model decides what to look up and how to say it. The pipeline decides
> what is true. A checker proves it, for every answer.**

## What it can see

Eight tools, all read-only, all over one `ResultStore`
([`onset_agent/tools.py`](../onset_agent/tools.py)):

| Tool | Returns |
|---|---|
| `get_recording_metadata` | subject, source, sampling rate, analysed window, seizure markers, detectors, excluded channels |
| `list_channels` | every analysed channel with count and rate |
| `top_channels` | highest-rate channels with 95% intervals |
| `channel_summary` | one channel across all detectors: rate, rank, interval, mean frequency/amplitude, spike co-occurrence |
| `get_evidence` | the strongest events on a channel, each with an `evidence_id` |
| `detector_disagreements` | channels the two detectors rank very differently, plus event-level agreement |
| `rate_change` | rate before versus during the marked seizure |
| `report_section` | one section of the structured report |

There is **no** tool that runs a detector, changes a parameter, writes a file,
executes code or fetches a URL. There is no `subject` argument anywhere: the
patient is fixed by the application when it opens the store, so no model
output can change which patient is being discussed (`test_no_tool_can_change_the_patient`).

## The three guards

### 1. Scope refusal, before the model runs

[`guard.check_question`](../onset_agent/guard.py) refuses:

* **treatment** — resect, ablate, remove, operate, medication, dose, "should
  we", "what would you do", "recommend";
* **diagnosis and prognosis** — "does this patient have epilepsy", "where do
  the seizures start", "seizure onset zone", "will they be seizure free";
* **other patients** — any subject identifier that is not the loaded one.

Refusing here rather than in the prompt matters: there is no model in the path
to be persuaded, the refusal costs nothing, and it cannot be talked around.
The agent says what it *can* do instead.

### 2. Citation verification

Every `evidence_id` in an answer must (a) have been returned by a tool during
this conversation and (b) resolve in the store. An id the model invented, or
one it copied from its own earlier output, fails.

### 3. Number verification

Every number in the answer that carries a unit (`/min`, `Hz`, `ms`, `s`, `dB`,
`µV`, `%`, `x`) and every bare number larger than 20 must appear among the
numbers the tools returned, allowing for rounding. Small integers without a
unit are allowed through — "the top 3 channels", "two detectors" — and that
exception is the known soft spot in this check.

On failure the model is told exactly what was wrong and gets two more attempts;
then the agent declines and points at the report, which is authoritative.

Notebook 2 demonstrates this with a backend that deliberately lies:

```
problems: ["citation 'sub-xx|MADE-UP|rms|0.000' was never returned by a tool",
           "the value '999.9 /min' does not appear in any tool result"]
```

## Threat model

| Risk | What stops it |
|---|---|
| Model invents a rate | number verification; the answer is dropped |
| Model invents a citation | citation verification against retrieved ids |
| Model is talked into a treatment answer | scope refusal before the model runs |
| Model is asked about another patient | scope refusal; no tool takes a subject |
| **Prompt injection through data** — a channel name or report string that says "ignore your instructions" | tool results are passed as JSON and the prompt states they are data; more importantly, nothing the model says survives the citation and number checks, so injected text cannot become a false claim |
| Model calls a tool that does not exist | dispatcher rejects it; the error goes back as a tool result |
| Model passes a malicious argument | strict schema validation: unknown keys rejected, enums enforced, integers clamped; arguments are never `eval`ed or shelled out |
| Model loops forever | `max_steps` (6) and at most 3 calls per turn |
| Small model garbles the protocol | tool calls are recovered from message content; malformed JSON gets one corrective retry; otherwise refusal |

What is **not** defended against, and should not be pretended otherwise: this
is research code with no authentication, no audit log, no rate limiting and no
protection of the results directory itself. Anyone who can write to that
directory can change what the agent believes.

## Running it with an open-weight model

| Backend | When | Command |
|---|---|---|
| `scripted` | tests, CI, demonstration with no model at all | `--backend scripted` (default) |
| `ollama` | a local machine | `ollama pull qwen2.5:7b-instruct && ollama serve`, then `--backend ollama` |
| `openai_compat` | vLLM, llama.cpp server, LM Studio, a GPU box | `--backend openai_compat --model <name> --base-url http://host:8000/v1` |
| `transformers` | Colab, in-process | `--backend transformers --model Qwen/Qwen2.5-7B-Instruct` |
| `auto` | you do not know what your machine can run | `--backend auto` |

```bash
python -m onset_agent.cli --results artifacts/results/sub-pt01_ictal_run-01 \
       --backend ollama --model qwen2.5:7b-instruct --chat
```

### Letting it choose the model

`--backend auto` looks at the machine, picks the largest Qwen2.5-Instruct that
will actually run on it, downloads the weights if they are not cached, and
loads them. To see the decision without committing to a download:

```bash
python -m onset_agent.cli --hardware          # needs no --results
```

```
[onset-agent] cuda, 8 core(s), 32.0 GB RAM, 500.0 GB free
[onset-agent] GPU: NVIDIA GeForce RTX 4090, 24.0 GB

model        Qwen/Qwen2.5-7B-Instruct
load as      fp16 on cuda
needs        18.1 GB of a 24.0 GB budget
download     15.3 GB
fits         yes
```

The sizing lives in [`onset_agent/hardware.py`](../onset_agent/hardware.py) as
a pure function of a described machine, so the policy is tested against a 4 GB
laptop GPU, an Apple unified-memory box and a Raspberry Pi without owning any
of them.

**What the policy is, and why it is not "biggest that fits".**

1. Among models that fit at **fp16**, take the largest. Quantization trades
   accuracy for room; it is a concession to a small card, not a way to claim a
   bigger parameter count.
2. If nothing at or above 7B fits unquantised, take **7B quantised** rather
   than a larger model at a more aggressive quantization. On a 16 GB card both
   7B-at-8-bit and 14B-at-4-bit fit — and 7B is the only size whose tool
   calling this project has actually watched work.
3. Only then go smaller, and say so bluntly.

**Three things it will not do.** It never selects 8-bit or 4-bit without CUDA,
because `bitsandbytes` has no CPU or Metal path and discovering that *after* a
15 GB download is a bad afternoon. It refuses a download before spending a byte
if the disk cannot hold the result. And it never silently substitutes a smaller
model for one you asked for by name — a caller who requested 14B and quietly
got 1.5B would draw conclusions about weights they never ran.

**What it does not tell you.** Whether the chosen size is *good enough at this
task*. Fitting in memory and being competent at tool-constrained evidence work
are different properties, and this project has measured the second for no Qwen
size at all — [`onset_agent/benchmark.py`](../onset_agent/benchmark.py) exists
to measure it and has never been run under a real model. Every choice carries
that warning in its own output, because a 3B model that loads cleanly and then
fails the guards looks like broken software rather than a model out of its
depth.

**On model size.** Qwen2.5-7B-Instruct calls these tools reliably.
Qwen2.5-1.5B-Instruct (the free-Colab default) does not, and that is worth
seeing: it fumbles the protocol, states numbers it did not retrieve, and gets
caught. A system whose failure mode is "declines to answer" is usable; one
whose failure mode is "confidently wrong rate" is not.

**The scripted backend is not a language model.** It is a keyword policy that
speaks the same protocol so the loop can be tested offline. The CLI says so
every time it runs, and so does this sentence.

## Adding a tool

1. Write a method on `ResultStore` that returns JSON-safe primitives.
2. Add a `Tool(...)` entry in `onset_agent/tools.py` with a strict schema
   (`additionalProperties: false`, enums where possible).
3. Write the description as if for a colleague who has never seen the
   pipeline — it is the only thing the model knows about your tool.
4. Add a test: a valid call, an invalid call, and (if it returns evidence) a
   check that the ids resolve.

Do not add a tool that computes anything new. If a number is worth reporting,
the pipeline should compute it, store it, and the tool should read it. The
moment a tool starts calculating, the "the pipeline decides what is true"
guarantee weakens.
