# Contributing

## Ground rules

1. **Every number in the output must be traceable to signal.** If you add a
   quantity, it must be computed in the pipeline, written to the result
   directory, and carry the window it came from.
2. **Never add a recommendation.** Not to the report schema, not to the
   agent's prompt, not as a "suggested" field. This is a hard rule and the
   test suite enforces it.
3. **The model never produces a number.** This replaced an earlier rule that
   said the agent may only *read* a saved analysis — which was true until
   `onset_agent/analysis.py` gave it tools that run the pipeline live, and
   would now forbid half the repository. The invariant that actually holds,
   and that the tests enforce, is narrower and stronger: a language model
   chooses *which* measurements to make and how to phrase them; deterministic,
   separately tested code decides what is true; and every number in an answer
   is resolved back to the tool run that produced it or the sentence is
   struck. A tool may compute. It may not be non-deterministic, unrecorded, or
   outside the JSON contract.
4. **Document where the code lives.** Module docstrings explain *why the
   module exists*; parameters are documented at their definition in
   `config.py`. Do not start a separate wiki — it will drift.
5. **Offline tests, enforced.** `tests/conftest.py` sets `ONSET_HFO_OFFLINE`
   for the whole session and the single network choke point in
   `onset_hfo.datasets` refuses every outbound request, so a test that reaches
   for the archive fails loudly rather than making a green CI depend on S3.
   Use the simulator, the committed cohort table in `data/cohort/`, the mock
   model server in `tests/test_agent.py`, and the fake backends in
   `tests/test_orchestration.py`.

## Setup

```bash
cd onset-hfo
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # 468 tests, offline and enforced
# sklearn and openpyxl come with [dev]; ".[ml]" is the same set without pytest
ruff check .
```

## Adding a detector

1. Write the feature function — it takes `(signal, window_samples)` and
   returns a trace of the same length.
2. Add a thin module in `onset_hfo/detectors/` that calls
   `detect_with_feature` with it, plus a docstring naming the paper and the
   parameters you changed.
3. Register it in `detectors/__init__.py` and in `pipeline.HFO_DETECTORS`.
4. Add a config dataclass field if your detector needs its own thresholds.
5. Tests: it finds the hot channels on synthetic data; it returns nothing on a
   flat channel; more events at a lower threshold than a higher one.
6. Measure it: `python -m onset_hfo.cli evaluate --seeds 1 7 42` and put the
   numbers in `docs/EVALUATION.md`. A detector with no measured scores does
   not go in.

## Adding a dataset

Build a `Recording` (see `docs/DATA.md` for the three routes). The pipeline
needs a preloaded MNE `Raw` in volts, channel names as `LETTERS + NUMBER`, the
right mains frequency, and `t_offset` if you loaded a slice. Set `citation` —
every report prints it.

## Adding a read-only agent tool

For the question-answering agent over a *saved* analysis (`onset_agent/
tools.py`). To add a tool the planner can *run*, see the next section.

See `docs/AGENT.md` § *Adding a tool*. In short: a `ResultStore` method that
returns JSON-safe primitives, a strict schema with
`additionalProperties: false`, a description written for someone who has never
seen the pipeline, and tests for a valid call, an invalid call, and (if it
returns evidence) that the ids resolve.

## Adding a live analysis tool

The tools in `onset_agent/analysis.py` run the real pipeline at parameters the
planner picks. See `docs/ORCHESTRATION.md` §2 for the contract. In short:

1. A handler `f(session, **kwargs) -> dict` on `AnalysisSession`, returning
   JSON-safe primitives and nothing else.
2. A `ToolSpec` with a strict schema (`additionalProperties: false`), a
   `run_prefix` that will read well in a trace (`hfo_003`, not `tool_003`),
   and `recomputes=True` if it does.
3. **Memoise on everything that changes the answer.** Two identical calls must
   return identical numbers, or run-to-run stability stops measuring the
   planner and starts measuring the signal processing.
4. Raise `ContractError` for anything the planner could fix — an unknown
   channel, an unusable band. That message goes to the model verbatim, so
   write it for the model: say what is wrong *and* which tool would list the
   valid values.
5. No tool takes a subject argument. The patient is fixed when the session
   opens, so no model output can change which recording is analysed. A test
   asserts this across every registered tool.

## Adding a stopping rule

Subclass `StopRule` in `onset_agent/planner.py` with a `should_stop(store,
model_said_done) -> (bool, str)`. The reason string is read by humans in the
results table, so make it say what happened rather than that something
happened.

Set `guarantees_coverage` honestly. `TiedSetWidth` sets it `False` because a
tied-rank set from Poisson intervals is not a conformal set; `ConformalWidth`
sets it `True` because it rests on a real split-conformal threshold — and
still treats an *empty* candidate set as a failure, because a model that rules
out every channel is out of distribution, not finished.

## Adding an acquisition strategy

`ACQUISITION` in `onset_hfo/models.py`, plus a branch in `_acquire`. Two rules,
both learned the hard way:

* **Never let the sampler see the evaluation contacts.** Every strategy picks
  from a labelling pool split off before anything is predicted, and all
  strategies are scored on the same held-out contacts. Without that you are
  comparing four different exams.
* **Seed with `_stable_seed`, never `hash()`.** Python salts `hash()` on
  strings per interpreter, so a split seeded with it is fixed inside one
  process and different in the next — which made a published table
  irreproducible and failed CI on one Python version while passing on
  another.

Then re-run `python -m onset_hfo.learn acquire` over **several seeds**. One
seed is not a result here: the evaluation split moves the lift by more than
the difference between strategies.

## Adding a falsification test

`onset_agent/falsify.py`. State the expectation **before** the test runs and
return a verdict against it — a test that can be passed by any outcome is not
a test. Two of the existing five were themselves wrong when first written, and
both failed on the real recording while passing on synthetic data, so run new
ones on both.

If a test fails, suspect the test first. The `no pathology` failure was real
and exposed a missing null hypothesis; the `anonymised channel names` and
`leading channel removed` failures were artifacts of how the tests damaged the
recording.

## Changing a threshold

Change it in `config.py`, update the docstring where it is defined, re-run
`python -m onset_hfo.cli evaluate --seeds 1 7 42`, and update the numbers in
`docs/EVALUATION.md` in the same commit. A parameter change without a
re-measurement is not reviewable.

## Notebooks

The four notebooks are generated:

```bash
python scripts/build_notebooks.py     # writes notebooks/*.ipynb without outputs
```

Edit `scripts/build_notebooks.py`, not the JSON. The committed copies include
executed outputs so that they read well on GitHub; regenerate and re-execute
them when you change something they demonstrate:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Style

* Comments explain *why*, not *what*. The code already says what.
* No parameter appears outside `config.py`.
* Times are always original-recording seconds.
* Prefer a boring, readable implementation to a clever one: this codebase's
  main job is to be argued with.

## Four things that must stay in step with `berdakh/onset`

There are two repositories under one project name: this one, and the
[teaching prototype](https://github.com/berdakh/onset) on a synthetic cohort.
[`DUPLICATION.md`](DUPLICATION.md) measures what is actually shared and says
what must **not** be merged. Four things must match across them. **Anything not
on this list is allowed to differ.**

1. The **standing disclaimer**'s structure and its no-recommendation sentence.
   The canonical copy is `app/common.py`'s `DISCLAIMER_LEAD` /
   `DATA_SENTENCE` / `DISCLAIMER_TAIL`; only `DATA_SENTENCE` differs between
   the two apps, because only one of them runs on real recordings.
2. The **shared page names**: `Home`, `Report`, `Assistant`, `Data`,
   `Architecture`, `Research`.
3. The **sidebar link row** — same destinations, same names, same order:
   Clinical guide · Implementation walkthrough · Results & docs · Onset
   project. The entries after those four differ by design.
4. `.streamlit/config.toml`, which carries a `TWIN FILE` header saying so.

No mechanism can enforce this across two repositories, and a submodule or a
published package would cost more than four items are worth. The honest
control is that the list is short, written down in both places, and each item
carries a comment naming its twin. **If you change one of the four, change it
in the other repository in the same sitting.**

## Review checklist

- [ ] `pytest -q` and `ruff check .` pass
- [ ] new numbers are measured, not asserted, and appear in `docs/EVALUATION.md`
- [ ] new parameters are documented in `config.py`
- [ ] the report still contains no recommendation, and every finding still
      carries evidence
- [ ] if the agent surface changed: schemas are strict, refusal and
      verification tests still pass, and no tool takes a subject argument
- [ ] if a live tool changed: identical calls still return identical numbers
- [ ] if a model or acquisition result changed: it was measured over several
      seeds, and the seed spread is reported next to the mean
- [ ] nothing reaches the network from a test (`ONSET_HFO_OFFLINE` is set for
      the whole suite; a new download will fail loudly)
- [ ] `docs/LIMITATIONS.md` updated if the change alters what the results mean
- [ ] if one of the four items above changed: the twin in `berdakh/onset`
      changed too
