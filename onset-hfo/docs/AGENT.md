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

On a Linux desktop the installer does all of this once:
**From inside the window.** The reviewer's Assistant page carries a *Local
model on this machine* box (`onset_review/modelsetup.py`). It probes the
machine and runs the same `choose(route="ollama")` as the CLI, asks the
server what it holds, and offers one button: *Download and use* streams
`/api/pull` with a progress bar on a worker thread, writes the assistant
defaults, and switches the panel to the model; *Use this model* when the tag
is already there; and when nothing answers on Ollama's port it gives the
download address and the two commands, with *Check again* for afterwards.
`tests/test_model_setup.py` drives it against the stand-in server: the pull
goes to the server once, the defaults file names the chooser's tag, and the
panel switches. The one thing it does not do is install Ollama, for the same
reason the CLI does not.

`./packaging/install-ubuntu.sh --with-assistant` installs Ollama, sets its
context window, runs the same chooser, pulls the model and records the choice so
the reviewer's assistant panel opens on it. See [`INSTALL.md`](INSTALL.md).

`--backend auto` looks at the machine, picks the largest Qwen that will
actually run on it, fetches the weights if they are not already there, and
loads them. To see the decision without committing to anything:

```bash
python -m onset_agent.cli --hardware          # needs no --results
```

On a laptop or desktop without an NVIDIA card — most machines — the answer is
not transformers at all:

```
[onset-agent] cpu, 8 core(s), 16.0 GB RAM, 200.0 GB free

model        Qwen/Qwen2.5-7B-Instruct
run via      Ollama / llama.cpp, tag qwen2.5:7b-instruct
load as      Q4 GGUF on cpu
needs        5.8 GB of a 12.0 GB budget
pull         4.6 GB  (ollama pull qwen2.5:7b-instruct)
fits         yes
then         onset-agent --backend ollama --model qwen2.5:7b-instruct
```

**Why a served GGUF off a CUDA card.** On a CPU, `transformers` loads float32:
twice the memory of the fp16 figure usually quoted and a fraction of the speed.
The same 16 GB machine that can hold only a 1.7B in-process runs a **7B** as a
`Q4_K_M` GGUF under Ollama or a llama.cpp server — several times faster, and
with no torch installed at all. So off CUDA the default route is `ollama`, and
`--backend auto` does the rest: it checks for a running server, pulls the tag
itself if it is missing, and hands the agent an `OllamaBackend`. What it cannot
do is install Ollama — that is a daemon, not a package — so a machine without
one is given the exact three commands (`ollama serve`, the pull, the run) and
offered `--route transformers` as the slower in-process alternative. On a CUDA
card the in-process route is kept, because that is where bitsandbytes and
bf16 pay off and where the benchmark runs. `--route` overrides either way.

On an NVIDIA card:

```
[onset-agent] cuda, 8 core(s), 32.0 GB RAM, 500.0 GB free
[onset-agent] GPU: NVIDIA GeForce RTX 4090, 24.0 GB

model        Qwen/Qwen3-8B
run via      transformers, in-process
load as      fp16 on cuda
needs        19.5 GB of a 24.0 GB budget
download     16.4 GB
fits         yes
```

The sizing lives in [`onset_agent/hardware.py`](../onset_agent/hardware.py) as
a pure function of a described machine, so the policy is tested against a 4 GB
laptop GPU, an Apple unified-memory box and a Raspberry Pi without owning any
of them.

The catalogue, the sizing rule of thumb and the per-device dtype rule follow
[`berdakh/ROBT613`](https://github.com/berdakh/ROBT613)'s `qwen_workshop`
(`config.py`, `env.py`, `loading.py`), so the two agree rather than drifting.
That is also where a bug in the first version of this module came from — see
below.

**What the policy is, and why it is not "biggest that fits".**

1. Among models that fit at **fp16**, take the largest. Quantization trades
   accuracy for room; it is a concession to a small card, not a way to claim a
   bigger parameter count.
2. If nothing at or above 7B fits unquantised, take **7B quantised** rather
   than a larger model at a more aggressive quantization. On a 16 GB card both
   7B-at-8-bit and 14B-at-4-bit fit — and 7B is the only size whose tool
   calling this project has actually watched work.
3. Only then go smaller, and say so bluntly.

On the Ollama route every option is Q4, so step 1 is moot and step 2 decides —
which is why a 64 GB Mac and a 16 GB laptop both land on the same 7B unless a
model is named. On a *CPU* there is one more rule: the reference size is also
the ceiling, for patience rather than memory. The agent makes several model
calls per question and a 14B at Q4 answers at a couple of tokens a second on a
CPU; name it with `--model` and the cap lifts. Apple silicon is exempt — Metal
is fast enough for the budget to rule.

**A CPU is sized for fp32, not fp16.** `transformers` loads float32 on CPU:
fp16 matmuls on a consumer CPU are slower than fp32 and often numerically
unstable. That doubles the requirement against the fp16 figure usually quoted,
and the first version of this module got it wrong — it promised an 8 GB laptop
Qwen2.5-1.5B at "4.0 GB" when the real cost was 7.5 GB, which is exactly the
out-of-memory crash the module exists to prevent. An 8 GB CPU machine now gets
Qwen3-0.6B, which is also what the workshop treats as the laptop default.
Similarly `bfloat16` is only offered to a card with compute capability ≥ 8
(Ampere); a T4 is given `float16` explicitly, because most Qwen configs declare
bfloat16 and `dtype="auto"` would honour a declaration the card cannot execute.

**Three things it will not do.** It never selects 8-bit or 4-bit without CUDA,
because `bitsandbytes` has no CPU or Metal path and discovering that *after* a
15 GB download is a bad afternoon. It refuses a download before spending a byte
if the disk cannot hold the result. And it never silently substitutes a smaller
model for one you asked for by name — a caller who requested 14B and quietly
got 1.5B would draw conclusions about weights they never ran.

**What has actually been run.** The sizing arithmetic, the refusals and the
error paths are covered by 90 tests. The *loader* is covered too, by
[`tests/test_local_model.py`](../tests/test_local_model.py), which builds a real
Qwen2 checkpoint at toy size — two layers, 64 hidden, ~107k random parameters —
and drives the shipped code through it: that `choose().backend_kwargs()` are
accepted verbatim, that a CPU really does end up with `float32` tensors, that
the chat template reaches the model *with the tools attached*, and that a
generation round trip returns the shape the agent loop consumes. It skips
without the `llm` extra, so CI does not pay for a torch download; run
`pip install -e '.[llm]'` to include it. Checked against `torch 2.14` and
`transformers 5.18`.

The last test in that file is the one worth reading. A 107k-parameter model with
random weights is the worst possible agent, and it must be *caught*: the
citation and number checks reject it and the CLI refuses. Run by hand:

```
[onset-agent] backend: transformers:/tmp/tinyqwen (in-process)
Q: Which channel had the highest ripple rate?
[refused] I could not produce an answer I can stand behind: the checks on
          citations and numbers did not pass.
  reason: verification failed
```

**The served route has been run too, against a real llama.cpp server.** The
same toy Qwen2 was converted to GGUF with llama.cpp's own converter and served
by `llama_cpp.server` (llama-cpp-python 0.3.36, built from source on CPU) —
the OpenAI-compatible wire protocol Ollama speaks. The shipped
`OpenAICompatBackend` via `make_backend("llamacpp", ...)` did a tool-attached
round trip, the real agent loop ran against a saved analysis, and the guards
refused the random model: `verification failed`. Its raw output had been
`functions.top_channels:` — a malformed attempt at a tool call, which the
parser correctly declined to treat as one. (Converting a synthetic toy needs
its pre-tokenizer hash mapped to `qwen2`; that was done in a scratch script and
touches nothing here. A real Qwen GGUF converts as-is.)

**Give a served model enough context.** That run found a defect by hitting it.
The agent's opening turn is the system prompt plus eight tool schemas — about
5,500 characters, which the toy's 342-token vocabulary made 3,521 tokens; a
real Qwen tokenizer will spend fewer, but the tool results that follow add more
each step. With a 1024-token context the server answered
`400 context_length_exceeded`, and the backend reported *"could not reach the
model server, start one"* — it had been reached, and the reason was in the body
it threw away. The backend now quotes the server's message and, for a context
overflow, names the fix. **Ollama's default `num_ctx` is 2048**, so this is a
hazard on a real machine, not just a toy: set `OLLAMA_CONTEXT_LENGTH=8192`
before `ollama serve`, or `PARAMETER num_ctx 8192` in a Modelfile; `-c 8192`
for `llama-server`; `--max-model-len` for vLLM. `--hardware` says so on the
Ollama route.

**The assistant answering correctly through a served model has been run too,
and is a permanent test.** Random weights can only show the refusal path, so
[`tests/_fake_ollama.py`](../tests/_fake_ollama.py) is a protocol-faithful
stand-in for Ollama — not a language model but an oracle that does what the
system prompt asks: `top_channels`, then `get_evidence` for the leader, then an
answer citing it. Everything between the reviewer and it is the shipped code.
[`tests/test_served_assistant.py`](../tests/test_served_assistant.py) drives
`--backend auto` through it (discovery, no needless pull, a correct answer with
a real evidence id, the CLI printing `[answer]`), and a lying variant that
states a rate it never retrieved is **refused over the same wire** — without
that, the honest run would prove only that the oracle is polite. The Qt suite
does the same through the real desktop panel: it discovers the server by probe
with nothing configured, opens on its model, asks through the worker thread,
and shows a clickable citation. That test found a bug on its first run: the
panel discovered a server on a moved port and then dialled the hard-coded
default. `OllamaBackend` now follows **`OLLAMA_HOST`** — Ollama's own
variable, in every spelling Ollama accepts — read when the backend is built.

**What has not been run: a real download.** No Qwen weights have been fetched
in this project's development container — `huggingface.co` is blocked there, and
the 403 is what verifies the "the Hub may be blocked" error path rather than
leaving it imagined. So `ensure_model` against the real Hub is the one step in
the chain still unexercised. If it misbehaves, that is where to look first.

**The catalogue is not yet pinned, for the same reason.** Every entry carries
`revision="main"`, because a commit hash that was never looked up would be an
invention. [`scripts/pin_catalogue.py`](../scripts/pin_catalogue.py) resolves
each model's current Hub commit and writes it into `hardware.py` as one
`revision=` line per entry; `--dry-run` shows what would change and `--check`
exits non-zero once the pins have drifted from the Hub. Run it once on a
machine that can see `huggingface.co`, commit the result, and delete the test
in `tests/test_pin_catalogue.py` that documents the unpinned state. The
rewrite itself is tested on the shipped source with stand-in hashes; only the
lookup is unexercised here. Ollama tags are a separate matter: that registry
keys models by its own digests, so a served-route result is pinned by the
digest `ollama show` prints, not by a Hub commit.

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
