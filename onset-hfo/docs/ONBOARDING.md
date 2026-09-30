# Onboarding

**For someone joining this project.** The other documents explain *what was
measured*; this one explains *where things are, what will bite you, and what to
work on first*. Read it once, then use it as a map.

If you read nothing else here, read [**Traps**](#traps-that-will-bite-you).
Every item in that list has already cost someone a day.

---

## What this project is

Two halves, kept deliberately separate:

1. **A signal-processing pipeline** (`onset_hfo/`) that takes real public
   intracranial EEG and reports where ripples (80–250 Hz), fast ripples
   (250–500 Hz) and interictal epileptiform discharges occur, keeping the exact
   signal window behind every number.
2. **An agent on an open-weight language model** (`onset_agent/`) that reads
   what the pipeline produced, must cite it, and is refused and contradicted by
   code when it strays — and that can also *drive* the pipeline, choosing which
   analyses to run and at what thresholds, with every claim resolved back to the
   run that produced it.

It is a **prototype**. Not a medical device, no clinical claim, no
recommendation anywhere in the schema.

### The culture is the part that takes longest to absorb

This repository publishes its own negative results and retracts its own
apparent findings. [`OUTCOME.md`](OUTCOME.md) opens by saying nothing separates
the groups after correction, and that *the most important finding is that an
earlier version of this analysis appeared to*. Three recent studies concluded
"this buys nothing" and were written up in full anyway
([`EVALUATION.md`](EVALUATION.md) §1b, §3b, §6c).

So the question you will be asked about your work is not "did it work?" but
**"what would have falsified this, and did you check?"** Two habits follow:

- **Quote no number you have not verified against the file it came from.** The
  committed extracts in `data/` exist for exactly this. Numbers have been
  mis-transcribed here three times, and each time a programmatic check caught
  it.
- **State the power before the result.** `outcome.min_detectable_auc(n1, n2)`
  gives the smallest effect the sample could resolve. At 13 seizure-free
  against 7 recurrences it is **0.85**, which is above almost everything this
  cohort produces. A result under the floor is *not shown*, not *not there* —
  and saying which is the whole job.

---

## First hour

```bash
git clone https://github.com/berdakh/onset-hfo.git
cd onset-hfo/onset-hfo            # note: the package lives one level in
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest -q                          # ~4 min, offline and enforced
ruff check .

# the smallest end-to-end loop: synthetic signal -> detections -> cited report
python -m onset_hfo.cli run --synthetic --duration 30 --out artifacts/demo
python -m onset_agent.cli --results artifacts/demo/sim-01_simulated_run-01 \
    --question "which channels have the highest ripple rate?"
```

Then open the [live app](https://onsetnu.streamlit.app/) and ask the assistant
which channels to resect. Watching it refuse conveys the design philosophy
faster than any document. Locally: `streamlit run app/Home.py` (needs
`pip install -e ".[app]"`).

For the research framing rather than the code, read
[**Where do seizures start?**](https://berdakh.github.io/onset-hfo/TUTORIAL.html)
— one self-contained HTML page.

---

## Directory map

| path | what is in it |
|---|---|
| `onset_hfo/` | the pipeline — see the table below |
| `onset_agent/` | the agent and the S0–S3 orchestration ladder |
| `app/` | Streamlit app: `Home.py` plus nine pages |
| `docs/` | seventeen documents; reading order at the end of this file |
| `data/` | **committed extracts** so every published number can be checked from a fresh clone with no download |
| `artifacts/` | **gitignored** — where runs actually write, and the dataset cache |
| `notebooks/` | seven notebooks; the first five need no setup |
| `scripts/` | studies too long for a test: the model ladder, the sub-population screen, the multiplier ablation |
| `tests/` | thirteen files, 468 tests, all offline |
| `paper/` | **the preprint — untracked on purpose. Never commit it.** |

### `onset_hfo/` — the modules worth knowing first

| module | why it exists |
|---|---|
| `datasets.py` | **the single network choke point.** Every fetch goes through here, which is what makes offline enforcement possible |
| `config.py` | every tunable parameter, **documented at its definition**. `THRESHOLDS`, `BANDS`, `PipelineConfig` |
| `preprocess.py` | filtering, bad-channel handling, the bipolar montage |
| `detectors/` (package) | `rms`, `line_length`, `hilbert`, `short_time_energy`, `spike`. `HFO_DETECTORS` is the four HFO ones; `DETECTORS` adds spikes |
| `validate.py` | artifact rejection — the spectral-peak check does nearly all the work (§2) |
| `metrics.py` | rates, ranking, Poisson intervals, `leader_separation` |
| `outcome.py` | the surgical-outcome study. `rank_comparison`, `candidate_channels`, `min_detectable_auc` |
| `clinical.py` | parses the resected-contact spreadsheet and maps it to channels |
| `cohort.py` | ds003029 SOZ labels, Engel/ILAE outcome, recording site |
| `populations.py` | splits events by whether they ride an interictal discharge |
| `stability.py` | does the answer hold across minutes and across nights? |
| `streaming.py` | analyse a recording longer than memory, in chunks that cannot change the answer |
| `synthetic.py` | the simulator — the only place "accuracy" is honest rather than "agreement" |

### `onset_agent/` — the parts that constrain each other

| module | why it exists |
|---|---|
| `tools.py` | the **frozen JSON tool contract**. Changing it breaks the published orchestration numbers |
| `analysis.py` | the live tools the agent may call |
| `planner.py` | the S0–S3 ladder, the stopping rules, and `rank_channels` — the scoring rule every orchestration number depends on |
| `verifier.py` | resolves every numeric claim to a run id, or strikes the sentence |
| `guard.py` | refusals: no recommendations, no numbers the tools did not produce |
| `falsify.py` | deliberate attempts to break the system, including renaming channels to see whether the model ranks on anatomical priors instead of evidence |

---

## The data

| dataset | what it gives you | cost |
|---|---|---|
| **`ds003498`** (Zurich) | 20 patients, interictal sleep, 2000 Hz, **expert HFO markings + resected contacts + surgical outcome**. The workhorse — anything needing a reference standard uses this | ~2.2 GB whole, ~24 MB per 60 s slice |
| **`ds003029`** | ictal, multicentre, 1000 Hz, clinician SOZ contacts, Engel/ILAE, four recording sites | large |
| **`onset_hfo.synthetic`** | implanted ground truth, so precision and recall are *accuracy* rather than agreement | none |

Both archives are CC0. Anything published from ds003498 should cite the dataset
and Fedele et al. 2017; the citation is in `provenance.json` of every analysis.

**`data/` holds the committed extracts.** Each subdirectory has a README
explaining every column, and they are worth reading before you touch the
corresponding study: `data/outcome/` (per-patient outcome tables, the
sub-population screen, the multiplier ablation), `data/benchmark/` (threshold
sweeps against expert markings), `data/stability/` (group tables),
`data/cohort/` (the learned per-contact model), `data/example_analysis/` (one
shipped 60 s analysis the app and tests read).

---

## Results you should know before changing anything

**Against expert HFO markings** (20 patients, 41,187 marked events, §0): the two
bands want operating points a factor of 2.5 apart — ripples rank best at
**2.0 SD** (ρ = 0.655), fast ripples at **5.0 SD** (ρ = 0.610). The shipped
default of 5.0 is *known to be wrong for ripple work*: it finds 12% of the
expert-marked events and ranks at ρ = 0.37. It stays 5.0 so that published
ictal results do not silently change, and `BAND_THRESHOLD_SD` carries the
measured values.

**Against surgical outcome** ([`OUTCOME.md`](OUTCOME.md)): the expert
fast-ripple leader was inside the resection in 11 of 13 seizure-free patients
and 3 of 7 recurrences — AUC 0.71, p = 0.12. Our detector: 0.67, p = 0.17.
**Nothing survives correction**, and the power floor of 0.85 is why.

**Three measured nulls**, each worth reading as a model of how to report one:

| study | answer |
|---|---|
| §1b — two extra detectors (`hilbert`, `short_time_energy`) | buy nothing over RMS |
| §3b — splitting ripples by discharge coupling | does not localise the resection better |
| §6c — the agent's robustness multiplier | helps in the ripple band, *annihilates the ranking* in the fast-ripple band |

---

## Traps that will bite you

All of these are load-bearing and none is obvious from reading the code.

1. **`'S'` means success (seizure-free) and `'F'` means failure.** Reading `F`
   as "free" inverts every label in the cohort. `cohort.decode_outcome` is the
   only place that mapping is written down, and a test pins it against the
   Engel scores.
2. **One threshold for both bands is a bug, not a simplification.** Running the
   ripple value in the fast-ripple band costs precision 0.543 → 0.086. Use
   `outcome.BAND_THRESHOLD_SD`.
3. **`candidate_channels` needs integer event counts.** It decides which
   channels are statistically tied using **Poisson** intervals. Hand it a
   weighted score and it truncates to `int` and returns something meaningless
   that still looks like an answer. This is why §6c reports
   `candidates_resected` for one arm only.
4. **Tests must never import the Streamlit half of the app.** CI installs only
   `.[dev]`, which has no Streamlit, so `import app.common` fails there while
   passing locally. Logic lives in `app/panels.py`, which imports no Streamlit;
   a test enforces this. To reproduce CI locally, put a `streamlit.py` that
   raises `ImportError` on `PYTHONPATH`.
5. **`ONSET_HFO_OFFLINE=1` plus explicit `--subjects` runs most studies with no
   network at all**, because only the subject *listing* touches the archive.
   Two roadmap items sat marked "blocked on network" for weeks because nobody
   noticed this.
6. **`onset_agent/tools.py` is frozen.** `analysis.py`'s `_HFO_DETECTORS` is
   deliberately pinned to two detectors although five exist, because the tool
   contract enumerates them and a stable contract is worth more than a tidier
   schema. Do not "fix" it.
7. **Never add a recommendation** — not to the report schema, not to a prompt,
   not as a "suggested" field. Hard rule, enforced by tests.
8. **`artifacts/` is gitignored; `data/` is what is committed.** If you publish
   a number, commit the extract it came from, and say in the README how to
   regenerate it.
9. **`paper/` and `onset-hfo-preprint.zip` are excluded from git** via
   `.git/info/exclude` at the author's request. They are not on GitHub and must
   not be pushed there. Keep your own backup: an untracked file survives
   nothing.
10. **Comparing arms that dropped different patients is not a comparison.**
    A metric that returns NaN for a patient silently shrinks the sample, which
    changes the power floor and breaks the comparison. Carry the per-arm `n`
    into every table — §3b and §6c both do, because both were briefly wrong
    without it.

---

## What to work on

**The board in [`ROADMAP.md`](ROADMAP.md) has five open items and every one of
them is blocked on something you cannot buy with an afternoon**: a GPU (run the
agent ladder under a real model — this is item 1 and item 3 is deliberately
gated on it), a second public archive, an archive that publishes electrode
coordinates, and two clinicians' time for the hand-annotated benchmark.

So realistic first work comes from the **open questions** at the foot of the
same file:

| | runnable now? |
|---|---|
| **Bipolar vs referential montage for ripple rate** | Yes — `config.py` already has the flag, the slices are cached. Best first project: small, self-contained, genuinely unmeasured, and it feeds a limitation that is already documented |
| **Ablate the cycle-count criterion on real data** | Yes — §2 decided *keep* on synthetic evidence only |
| **Implement the silent-re-test fix in `rank_channels`** | Yes, but not small. `test_a_silent_stricter_pass_annihilates_the_ranking` is written so the fix has a failing assertion to flip; the catch is that `rank_channels` moves every orchestration number in the repo, so §6b's ladder table needs regenerating in the same change |

A good pattern to copy for any of these:
`scripts/run_subpopulation_outcome.py` and `scripts/run_robustness_ablation.py`
both run a cohort study offline, write a committed extract, and take
`--from-csv` to re-print their published tables with no re-analysis. That last
flag is what lets a reviewer check your numbers in ten seconds.

---

## Reading order

1. [`../README.md`](../README.md) — what the project is, and the two-minute local run
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the halves fit together
3. [`CONTRIBUTING.md`](CONTRIBUTING.md) — **the ground rules; read before your first PR**
4. [`GLOSSARY.md`](GLOSSARY.md) — if ripple, SOZ, Engel or ILAE are new to you
5. [`DATA.md`](DATA.md) — the archives, their licences, what is and is not in them
6. [`EVALUATION.md`](EVALUATION.md) §0 — the headline benchmark, then skim the rest
7. [`OUTCOME.md`](OUTCOME.md) — the only reference standard that is not another opinion
8. [`LIMITATIONS.md`](LIMITATIONS.md) — **read this before quoting any number**
9. [`ROADMAP.md`](ROADMAP.md) — what is left, and what each finished item cost

Then, as needed: [`METHODS.md`](METHODS.md),
[`ORCHESTRATION.md`](ORCHESTRATION.md), [`AGENT.md`](AGENT.md),
[`LOCALIZATION.md`](LOCALIZATION.md).
