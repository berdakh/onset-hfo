# Evaluation

## The honest summary

Three sources of truth, answering three different questions. Read the label
on each number before quoting it.

| Data | What is known | What can be measured |
|---|---|---|
| **Synthetic** (`onset_hfo.synthetic`) | every implanted event, exactly | precision, recall, F1 against ground truth |
| **`ds003498`** — Zurich interictal sleep, 20 subjects | expert-validated HFO markings, per channel | **agreement with expert markings**: event-level P/R/F1 and channel-rank correlation |
| **`ds003029`** — ictal, 1 subject | clinician seizure markers only | rates, rankings, detector agreement, rate change |

The middle row is new, and it retires the sentence this document used to open
with — *"no HFO ground truth exists, so precision and recall are not
reported"*. That was true of the ictal dataset and false of the archive:
[ds003498](https://openneuro.org/datasets/ds003498) ships expert markings, in
a format the loader already reads.

One distinction runs through everything below. Against synthetic data we
measure **accuracy**, because the truth is constructed. Against ds003498 we
measure **agreement**, because the reference is itself the validated output of
another detector. Agreement is the more useful number and the weaker claim,
and conflating the two would be the easiest way to oversell this software.

Reproduce:

```bash
python -m onset_hfo.cli benchmark                      # real data, expert markings
python -m onset_hfo.cli evaluate --seeds 1 7 42        # synthetic ground truth
```

or run [`notebooks/03_validation_and_benchmark.ipynb`](../notebooks/03_validation_and_benchmark.ipynb).

---

## 0. Scored against expert HFO markings (the headline)

**Cohort**: all 20 subjects of ds003498, first 60 s of run-01 each, 2000 Hz,
slow-wave sleep. **41,187 expert-marked events** in total (35,620 ripples,
5,567 fast ripples), on the 6–65 channels per subject that the annotators
reviewed. Scoring is restricted to those channels: a detection elsewhere is
unjudged, not wrong.

### Ripple band (80–250 Hz), cohort means

| detector | threshold | precision | recall | F1 | rank ρ | top-5 shared |
|---|---|---|---|---|---|---|
| RMS | 1.5 SD | 0.432 | 0.478 | **0.418** | 0.592 | 3.05 / 5 |
| RMS | 2.0 SD | 0.507 | 0.381 | 0.400 | **0.655** | 3.30 / 5 |
| RMS | 3.0 SD | 0.590 | 0.231 | 0.296 | 0.529 | 2.90 / 5 |
| RMS | **5.0 SD (shipped default)** | 0.591 | 0.118 | 0.165 | 0.368 | 2.55 / 5 |
| line length | 2.0 SD | 0.572 | 0.272 | 0.332 | 0.514 | 2.95 / 5 |
| line length | 5.0 SD | 0.557 | 0.077 | 0.106 | 0.354 | 2.20 / 5 |

### Fast ripple band (250–500 Hz), cohort means

Analysable at last: these recordings are 2000 Hz, the ictal one was 1000.

| detector | threshold | precision | recall | F1 | rank ρ | top-5 shared |
|---|---|---|---|---|---|---|
| RMS | 2.0 SD | 0.086 | 0.529 | 0.145 | 0.436 | 3.40 / 5 |
| RMS | 3.0 SD | 0.235 | 0.351 | 0.275 | 0.520 | 3.45 / 5 |
| RMS | 3.5 SD | 0.329 | 0.291 | **0.296** | 0.590 | 3.70 / 5 |
| RMS | 4.0 SD | 0.411 | 0.250 | 0.291 | 0.585 | 3.65 / 5 |
| RMS | **5.0 SD** | 0.543 | 0.204 | 0.264 | **0.610** | 3.55 / 5 |
| RMS | 6.0 SD | 0.632 | 0.170 | 0.230 | 0.580 | 3.35 / 5 |
| RMS | 8.0 SD | 0.720 | 0.112 | 0.161 | 0.543 | 3.05 / 5 |
| RMS | 10.0 SD | 0.710 | 0.083 | 0.122 | 0.509 | 3.20 / 5 |
| line length | 5.0 SD | 0.643 | 0.173 | 0.234 | 0.560 | 3.45 / 5 |

**The two bands want different operating points, by a factor of two and a
half.** Ripples rank best at 2.0 SD, fast ripples at 5.0 SD — an interior
maximum of a 2–10 sweep, not a boundary. Running the ripple value in the
fast-ripple band costs precision 0.543 → 0.086: a mean of 1,142 detections
per 60 s against a mean of 278 expert-marked fast ripples. A single
threshold for both bands is not a simplification, it is a bug, and
[OUTCOME.md](OUTCOME.md) shows what it costs downstream. `config.THRESHOLDS`
now carries `interictal-agreement-fast-ripple` alongside the ripple values,
and `onset_hfo.outcome.BAND_THRESHOLD_SD` maps bands to them.

### What these numbers say

**The shipped default is wrong for interictal HFO work, and now we know by how
much.** At 5.0 SD — Staba's published value, which this pipeline inherited —
the ripple detector finds 12% of the expert-marked events and ranks channels
at ρ = 0.37. At 2.0 SD it finds 38% and ranks at ρ = 0.66. The threshold was
never tuned to the simulator on principle; it turns out the principle
protected a value that real data does not support. `config.THRESHOLDS` now
carries both, measured and named, and `--threshold interictal-agreement`
selects the better one. The default stays 5.0 so that published ictal results
do not silently change; a future release should split the defaults by task.

**The two criteria disagree slightly, and both are reported.** F1 peaks at
1.5 SD, channel-rank agreement at 2.0 SD. Both are interior maxima — the sweep
was extended downwards precisely because the first grid put its optimum on
the edge, which is not an optimum but a boundary. Channel ranking is the more
clinically meaningful of the two: nobody operates on an event.

**Precision plateaus around 0.6, and that is expected.** The reference is the
validated output of the original study's Morphology detector — not a census of
every oscillation. Events we find that it did not are counted as false
positives whether or not they are real, so 0.6 is a floor on our precision,
not a measurement of it. The honest reading: *we reproduce roughly half of a
different detector's validated events and rank the same channels as active at
ρ ≈ 0.65.*

**RMS beats line length on real data**, reversing their order on synthetic
data (0.679 vs 0.697 F1 there). A detector comparison that holds only on
simulated signal is a comparison of simulators.

**Per-subject variability is large** and the cohort mean hides it: reviewed
channels range from 6 to 65, expert events per 60 s from 644 to 7,935, and
per-subject best F1 from 0.14 to 0.56 (median 0.48). `artifacts/results/benchmark_ds003498/scores.csv`
has every row.

---

## 0b. Scored against surgical outcome

The strongest test in the repository lives in its own document, because it
compares the pipeline to something no algorithm produced:
**[OUTCOME.md](OUTCOME.md)** — did the HFO map point at the tissue whose
removal made the patient seizure-free?

The short version, on whole 300-second runs: expert markings, fast-ripple
band, the busiest channel was inside the resection in 11 of 13 seizure-free
patients and 3 of 7 recurrences (AUC 0.71, p = 0.12); our detector on the same
channels, 10 of 13 versus 3 of 7 (AUC 0.67, p = 0.17). Nothing reaches
p < 0.05 after correction; one of 36 uncorrected comparisons reaches 0.034,
which is fewer than chance produces.

**The number that matters is the one that moved.** On the first 60 seconds of
the same recordings the expert arm gave AUC 0.82, p = 0.007 — and that is what
this document and the README reported until the study was re-run on the full
recording. Two patients account for the entire difference. A conclusion that
changes between minute one and minutes one-to-five is not yet a measurement,
and channel-ranking stability across windows is now the most concrete open
problem in this repository.

```bash
python -m onset_hfo.cli outcome
```

---

## 1. Detector scores on synthetic data

Three recordings (seeds 1, 7, 42), 60 s each, 18 contacts → 15 bipolar
channels, 2000 Hz, ~200 implanted ripples, ~120 discharges, ~70 artifacts.

| detector | precision (mean ± sd) | recall | F1 |
|---|---|---|---|
| RMS energy (ripples) | 0.956 ± 0.017 | 0.528 ± 0.073 | 0.679 ± 0.065 |
| line length (ripples) | 0.957 ± 0.017 | 0.550 ± 0.078 | 0.697 ± 0.067 |
| interictal discharges | 0.998 ± 0.004 | 0.844 ± 0.037 | 0.914 ± 0.023 |

**How matching works.** A detection matches a truth event when they overlap in
time (20 ms tolerance) and the truth contact is one of the contacts the
detection's bipolar channel is built from — a pair `SA2-SA3` may legally claim
an event implanted on `SA2` or `SA3`. Recall counts truth events matched at
least once (an event found on two overlapping pairs counts once); precision
counts detections that match some truth event.

**Recall around 0.5 is the expected behaviour of a 5-SD threshold detector**,
not a defect. It finds events that are clearly above the background and misses
marginal ones. Section 3 shows exactly how that trades off.

---

## 1b. Four detectors, and what the extra two are worth

Two detectors matching about half their events is a finding. Three or four
answer the question that one pair cannot: is the disagreement **structural**,
or an idiosyncrasy of RMS-versus-line-length? `detect_hilbert` (the smoothed
analytic-signal envelope) and `detect_short_time_energy` (the sum of squares)
were added for that, and are **opt-in** — `run_pipeline` still defaults to two,
because their thresholds are inherited rather than measured and turning them on
by default would change every number above without anyone deciding to.

```python
res = run_pipeline(rec, detectors=("rms", "line_length", "hilbert", "short_time_energy"))
metrics.agreement_matrix(res.events)          # who agrees with whom
metrics.consensus_ranking(res.events, res.duration_s)   # channels by votes, not rates
```

### Against synthetic ground truth

Three seeds, 60 s, 15 bipolar channels, ripple band, all four at their default
thresholds. The first two rows reproduce §1 exactly, which is the check that
adding detectors perturbed nothing.

| detector | threshold | precision | recall | F1 | detections |
|---|---|---|---|---|---|
| RMS | 5.0 SD | 0.956 | 0.528 | 0.679 | 138 |
| line length | 3.0 SD | 0.957 | 0.550 | 0.697 | 150 |
| Hilbert envelope | 5.0 SD | 0.932 | 0.556 | 0.696 | 149 |
| short-time energy | 5.0 SD | 0.835 | **0.808** | **0.821** | 291 |

### The agreement matrix

Mean Jaccard over the same three seeds:

| | RMS | line length | Hilbert | short-time energy |
|---|---|---|---|---|
| **RMS** | 1.000 | 0.588 | **0.788** | **0.471** |
| **line length** | 0.588 | 1.000 | 0.574 | 0.500 |
| **Hilbert** | 0.788 | 0.574 | 1.000 | 0.506 |
| **short-time energy** | 0.471 | 0.500 | 0.506 | 1.000 |

**The disagreement is structural, not a quirk of one pair.** Every off-diagonal
cell sits between 0.47 and 0.79. No two of these four detectors agree on more
than four events in five, and the worst pair agrees on fewer than half. A
single-detector rate table is less certain than it looks, and now that is four
measurements rather than one.

### A prediction made here was wrong, and that is the useful part

`short_time_energy` was added *expecting it to be nearly redundant* with RMS:
`energy = window × rms²` is a monotone transform, so under a **fixed** threshold
the two would select identical samples. The docstring said so before the
measurement, which is the only reason the correction is checkable.

It is the **least** similar pair in the table (0.471), and the closest pair is
RMS with the Hilbert envelope (0.788). The cause is direct and measurable:
`median + 5 robustSD` sits at about the **98th percentile** of an RMS trace and
the **96th** of an energy trace, because squaring is not affine and stretches
the upper tail relative to the median. So the same `k` is a materially more
permissive operating point on the squared feature. On one recording
short-time energy at 5.0 SD finds 294 events where RMS at 5.0 SD finds 132, and
RMS has to come down to about 2.5 SD (250 events) before the counts are
comparable. A test pins the asymmetry: at 50 robust SD the energy detector is
silent and the squared one is still firing.

**So do not read its F1 of 0.821 as a better detector.** §3 already showed that
lower thresholds score higher F1 *on this simulator*, whose signal-to-noise
distribution is a guess. Short-time energy is mostly the energy detector at an
untuned, lower operating point. The generalisable conclusion is the one §3
draws, strengthened: **a threshold in robust SDs is not a portable operating
point between features.** Each feature needs its own sweep, and the two new
detectors have not had one.

### Ranking by votes rather than by rate

`consensus_ranking` ranks channels by **how many detectors place them in their
own top five**, not by an average of the four rates. Averaging would be wrong
twice over: the features are on different scales — short-time energy is in
µV²·samples — so a mean would be dominated by whichever has the largest units;
and a detector that nearly duplicates another would be counted twice in it.
Here each detector gets one vote, which is the right amount of influence for a
fourth opinion that might be a second copy of the first.

On seed 1, the three channels carrying implanted events take 4/4 votes and the
top three places; the next two channels take 3/4 and carry nothing. Read the
vote next to the matrix above: **unanimity among detectors that agree with each
other on every event would not be unanimity, it would be one detector.** These
four do not, which is what makes the vote worth counting.

### What is missing

All of the above is **synthetic**. The real-data agreement matrix — the one
that matters, on ds003498 against the expert markings — needs one
`onset-hfo benchmark` run on a machine with network access, and a threshold
sweep for each of the two new features. Neither has been done. Until then, the
two added detectors are machinery with a synthetic sanity check, not a result.

---

## 2. What artifact rejection buys

Same recording, RMS detector, with and without the validation stage:

| stage | detections | precision | recall | false positives caused by |
|---|---|---|---|---|
| raw detector output | 238 | 0.630 | 0.614 | 86 artifacts, 2 spikes |
| after artifact rejection | 151 | 0.974 | 0.605 | 3 artifacts, 1 spike |

Precision 0.63 → 0.97 for one point of recall. The false positives it removes
are **filter ringing**: large sharp transients that an 80–250 Hz band-pass
turns into convincing ripples. This is the single most important stage in the
pipeline, and the reason the simulator implants artifacts at realistic
amplitudes (electrode pops in real recordings reach hundreds of microvolts).

### Which criterion actually does the work

| configuration | kept | precision | recall | F1 |
|---|---|---|---|---|
| no rejection at all | 238 | 0.630 | 0.614 | 0.622 |
| cycle count only | 237 | 0.629 | 0.610 | 0.619 |
| **spectral peak only** | 152 | **0.974** | 0.610 | 0.750 |
| both (the default) | 151 | 0.974 | 0.605 | 0.746 |

The spectral-peak check is responsible for essentially all of the gain; the
cycle count is a cheap guard that rarely fires. Both are kept — the cycle
count costs almost nothing and catches a failure mode the spectral check
cannot (an event too short to have a spectrum) — but nobody should believe the
cycle criterion is doing important work, and this table is here so nobody has
to take that on trust.

---

## 3. The threshold is a choice, not a fact

RMS detector, synthetic data, sweeping `threshold_sd`:

| threshold (robust SD) | detections | precision | recall | F1 |
|---|---|---|---|---|
| 3 | 215 | 0.916 | 0.719 | **0.806** |
| 4 | 185 | 0.968 | 0.662 | 0.786 |
| **5 (default)** | 151 | 0.974 | 0.605 | 0.746 |
| 6 | 113 | 0.973 | 0.486 | 0.648 |
| 7 | 69 | 0.957 | 0.338 | 0.500 |
| 8 | 39 | 0.949 | 0.210 | 0.343 |

**The default is not the best-scoring value here, on purpose.** A threshold of
3–4 SD scores higher F1 *on this simulator*, whose signal-to-noise
distribution is a guess. Tuning the default to it would be fitting the
detector to the fiction. Five robust SDs is Staba's published value; it is
what a reviewer will expect; and the curve is published here so anyone can
choose differently for their own data:

```python
from onset_hfo.config import PipelineConfig
cfg = PipelineConfig()
cfg.rms.threshold_sd = 4.0
```

---

## 4. How hard is the problem? Recall against SNR

`ripple_snr` is the implanted ripple's peak amplitude divided by the RMS the
ripple band already carries.

| SNR | precision | recall | F1 |
|---|---|---|---|
| 4 | 0.821 | 0.195 | 0.315 |
| 5 | 0.896 | 0.283 | 0.430 |
| 6 | 0.938 | 0.403 | 0.563 |
| 7 | 0.960 | 0.585 | 0.727 |
| 9 | 0.972 | 0.774 | 0.861 |
| 12 | 0.978 | 0.918 | 0.947 |

This is the curve to quote when someone asks "will it work on our
recordings?". The answer depends on their signal-to-noise ratio, not on ours.
Precision stays high throughout — when this detector fires, it is usually
right; what changes with SNR is how much it misses.

---

## 5. The spike detector's discontinuity check

Before adding it, the discharge detector scored precision ≈ 0.58, and **every**
false positive was an artifact step (a 5–60 Hz band-pass turns an electrode
pop into a textbook sharp wave). Measuring the maximum sample-to-sample jump
in the *unfiltered* signal, in robust SDs of that channel's derivative:

| population | 10th percentile | median | 90th percentile |
|---|---|---|---|
| genuine discharges | 2.8 | 3.6 | 4.9 |
| artifact steps | 24.6 | 45.5 | 63.9 |

A threshold of 10 separates them cleanly. Precision went to ≈ 1.00 with no
measurable loss of recall. The lesson generalises: **when a filter hides the
evidence, test on the raw samples.**

---

## 6. What can be checked on the ictal recording

`sub-pt01`, ictal run 01, 50–110 s, 71 bipolar channels, 1000 Hz. The whole
run takes about 7 seconds and produces 1732 RMS candidates (1549 accepted),
2387 line-length candidates (2188 accepted) and 490 interictal discharges.
Detected ripples have a median peak frequency of 146 Hz (10th–90th percentile
80–235 Hz) and a median spectral prominence of 8.9 dB above the recording's
own background.

**Rate change across the marked onset.** On the leading channels the ripple
rate is 0 before the clinician-marked onset and 144-146/min during the seizure
(`PST2-PST3`, `ATT7-ATT8`, `ATT6-ATT7`).
This is consistent with ictal HFOs, and it is also a reminder that this is
*ictal* data: the classical HFO literature measures interictal rate, which is
a different quantity (see `DATA.md`).

**Detector agreement.** The two detectors match about half their events
(Jaccard = 0.53 on this recording) and rank seven leading channels very
differently. The report names those channels rather than averaging them. Two
simple detectors on the same band disagreeing this much is itself a finding,
and the honest reading is: a single-detector rate table is less certain than
it looks.

**The clinician-marker check.** The events file names contacts near seizure
onset (`AD1-4, ATT1,2, G16, AST1,3, SLT1-3`). Permutation test on our ranking:

| | k = 5 | k = 10 |
|---|---|---|
| top channels touching a named contact | 1 | 2 |
| expected by chance | 1.05 | 2.15 |
| permutation p | 0.71 | 0.68 |

**No better than chance.** That result is reported here rather than buried,
and it is not evidence that the detector is broken. Reasons it is expected:
the markers are not a curated seizure-onset zone; the clinician was naming
what was visible at onset in the wideband signal, not where ripples were
densest; this is one 60-second ictal window; and ictal ripple energy spreads
far beyond the onset region. What it *does* establish is that **nothing in
this repository should be read as identifying a seizure-onset zone**.

The way to turn this into a real result is in the roadmap: interictal
recordings, several patients, and resection outcome as the reference.

---

## 6b. The orchestration ladder

`onset_agent/orchestrate.py` runs four configurations over the *same*
analyzers and the same recording, so any difference is attributable to who
decided what to run. See [`ORCHESTRATION.md`](ORCHESTRATION.md) for the
design. Reproduce with:

```bash
python -m onset_agent.orchestrate --subject sub-pt01 --task ictal --run 01 \
    --start 50 --stop 110 --score
```

**On the real recording** (`sub-pt01`, 50–110 s, 71 bipolar channels,
deterministic scripted planner):

| rung | tool calls | wall-clock | channels re-tested | top-5 |
|---|---|---|---|---|
| S0 fixed pipeline | 5 | 6.2 s | 0 | PST2-PST3, ATT6-ATT7, ATT7-ATT8, AST2-AST3, ATT5-ATT6 |
| S1 single-shot | 4 | 3.4 s | 0 | *(same)* |
| S2 re-planning | 7 | 3.8 s | 2 | ATT6-ATT7, AST2-AST3, ATT7-ATT8, PST2-PST3, ATT5-ATT6 |
| S3 + verifier | 7 | 3.5 s | 2 | *(same as S2)* |

**Read this table carefully, because the obvious reading is wrong.** S2 did
re-order the top five — but look at what it measured. `PST2-PST3` went from
83.0/min at 5 robust SD to 79.0/min at 7 SD: a robustness of 0.95, which is a
channel *passing* its robustness check, not failing it. The five leading
channels sit between 70 and 83 events/min with heavily overlapping Poisson
intervals; they are **tied, not ranked**, and a 5% penalty is enough to
shuffle them. The honest statement is that on this recording re-planning found
nothing to demote, and the re-ordering it produced carries no information.

The mechanism does work when there is something to catch. On synthetic data
with implanted artifacts, `SA2-SA3` falls from 54.0/min at 5 SD to 30.0/min at
7 SD — robustness 0.56 — and drops below a channel it had led. S0 and S1
cannot produce a robustness below 1.0 at all, because they never re-test.

**Scored against the archive's clinician SOZ contacts** (`sub-pt01`: 10
labelled contacts, Engel 1, seizure free, NIH — from
`sourcedata/clinical_data_summary.xlsx`, see [`DATA.md`](DATA.md)):

| rung | hits @ 5 | expected by chance | permutation p |
|---|---|---|---|
| S0 | 0 | 1.03 | 1.00 |
| S1 | 0 | 0.84 | 1.00 |
| S2 | 0 | 0.84 | 1.00 |
| S3 | 0 | 0.84 | 1.00 |

**No better than chance, for every rung.** This replaces the weaker check in
§6 — which used the reviewer's free-text markers — with the curated label, and
the answer does not change. The reasons given in §6 still apply and are worth
repeating: this is one 60-second **ictal** window, ictal ripple energy spreads
far beyond the onset region, and the classical HFO literature measures
interictal rate. Orchestration cannot fix a measurement that is the wrong
measurement for the question; it is not supposed to, and a ladder that
appeared to would be measuring something else.

Every number above comes from the deterministic scripted planner. It is the
control, not the result: what a real open-weight model does to these rows is
unmeasured, and is the first experiment to run.

## 7. Test suite

`pytest -q` — 403 tests, entirely offline. They cover the
primitives (robust scale, sliding features, threshold segmentation, bipolar
pairing), the detectors (hot channels found, events are oscillations, a flat
channel yields nothing, thresholds behave monotonically, reruns are
identical), validation (precision rises, recall survives, pops are rejected),
the report (no recommendation field, every finding carries evidence), and the
whole agent (tool schemas, argument validation, scope refusals, citation and
number verification, and the language-model path against a mock
OpenAI-compatible server that replies the way Qwen and Llama servers do —
including the two ways small models get it wrong).

`tests/test_orchestration.py` adds 70 of those and `tests/test_localization.py` 60, covering the orchestration
half: the label layer (including the `S`/`F` outcome inversion and the
subject-id mismatch), the tool contract's validation and clamping, the
evidence store's number resolution, the live tools (a stricter threshold finds
fewer events; the same call twice gives the same numbers; an unknown channel
is refused with a usable message), all four rungs, the three stopping rules,
both verifiers and the delta between them, and the permutation scoring. The
language-model paths run against small fake backends that reply the way a
served model does, including a babbling one and a verifier that strikes a true
sentence while missing a fabricated one.

## 8. What none of this establishes

Nothing here is a claim about clinical performance. Real ripples are not
Gaussian-windowed sinusoids; real artifacts are more varied; real recordings
contain physiological ripples in healthy tissue that this simulator does not
model at all. These numbers establish that the code does what it says on data
where the truth is known. They say nothing about patients.
