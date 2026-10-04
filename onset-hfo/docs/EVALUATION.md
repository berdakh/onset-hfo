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

### On real data: the two extra detectors buy nothing

The sweep above is synthetic. Here is the same four-detector comparison on
**ds003498 — 20 patients, first 60 s of run-01 each, scored against 41,187
expert markings on reviewed channels only**, swept over threshold in both
bands. Cohort means; committed at
[`data/benchmark/four_detector_sweep.csv`](../data/benchmark/four_detector_sweep.csv).

Each detector at **its own best threshold**, by channel-rank agreement — the
metric that matters, because nobody operates on an event, they operate on
tissue:

**Ripple band (80–250 Hz)**

| detector | best SD | rank ρ | F1 | precision | recall | detections/60 s |
|---|---|---|---|---|---|---|
| short-time energy | 4.0 | **0.658** | 0.414 | 0.513 | 0.396 | 1,081 |
| RMS | 2.0 | **0.655** | 0.400 | 0.507 | 0.381 | 1,035 |
| line length | 1.5 | 0.602 | 0.402 | 0.505 | 0.391 | 1,055 |
| Hilbert envelope | 1.5 | 0.578 | 0.385 | 0.456 | 0.404 | 1,163 |

**Fast ripple band (250–500 Hz)**

| detector | best SD | rank ρ | F1 | precision | recall | detections/60 s |
|---|---|---|---|---|---|---|
| Hilbert envelope | 5.0 | **0.613** | 0.270 | 0.546 | 0.208 | 71 |
| RMS | 5.0 | **0.610** | 0.264 | 0.543 | 0.204 | 71 |
| short-time energy | 12.0 | 0.601 | 0.259 | 0.583 | 0.200 | 69 |
| line length | 5.0 | 0.560 | 0.234 | 0.643 | 0.173 | 58 |

**Both margins are ties.** 0.658 against 0.655 in ripples, 0.613 against 0.610
in fast ripples — three thousandths of a Spearman correlation across twenty
patients. Two more detectors, each tuned on real data, bought **no measurable
improvement in channel ranking** over the two that were already there. That is
a useful thing to have paid for: it says the ceiling here is not the choice of
feature.

### And the threshold is not portable between features — measured

The synthetic section predicted this; the cohort confirms it, and more sharply.
Optimal threshold by detector and band, in robust SDs of that detector's own
feature:

| | RMS | line length | Hilbert | short-time energy |
|---|---|---|---|---|
| **ripple** | 2.0 | 1.5 | 1.5 | **4.0** |
| **fast ripple** | 5.0 | 5.0 | 5.0 | **12.0** |

Short-time energy needs **twice** the ripple threshold and **2.4×** the
fast-ripple threshold that RMS does. Run it at RMS's 5.0 SD in the fast-ripple
band — the obvious thing to do — and rank agreement falls from 0.601 to 0.485.
The Hilbert envelope, by contrast, wants almost exactly what RMS wants (1.5
against 2.0; 5.0 against 5.0), which is the same story the synthetic agreement
matrix told: those two are the closest pair in the family.

**A claim in this section was too harsh and is corrected.** It read: *"Short-time
energy is mostly the energy detector at an untuned, lower operating point."*
The first half is right — at a shared threshold it is badly mis-set — but at
**its own** tuned threshold it matches RMS rather than merely trading recall
for precision. It is a competitive detector that needs a different number, not
a mis-set copy of another one.

### Four of sixteen arms first peaked on the edge of the grid

The first pass swept 1.0–8.0 SD. Four of the sixteen (detector × band ×
criterion) optima landed on an endpoint, which by this project's own standard
is the grid running out rather than an optimum — the lesson from §0, where the
ripple sweep had to be extended downwards for exactly this reason.

Extending (down to 0.5 for the ripple arms, up to 15.0 for short-time energy in
fast ripples) moved one of them materially: **short-time energy's fast-ripple
rank agreement went from 0.570 at 8.0 SD to 0.601 at 12.0 SD.** Had the grid
not been extended, that detector would have been reported a full 0.03 worse
than it is. All sixteen optima are now interior.

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

## 3b. Physiological versus epileptic ripples: what the split is worth

Roadmap item 6. Every rate this project has published merges two populations
that mean opposite things — a ripple in healthy cortex is not a finding, and
nothing here can tell it from an epileptic one. No public archive carries the
label, so the pipeline does the honest half instead: it splits events by
whether they ride an interictal discharge, reports the sub-populations side by
side (`populations_<detector>.csv`), and **measures whether that split
separates anything at all**.

**Cohort.** The shipped 60-second analysis, `data/example_analysis/`
(ds003029, real patient recording). RMS: 95 spike-coupled events against
1,454 independent. Line length: 181 against 2,007. Five morphology features
per event, Bonferroni-corrected across the five.

### RMS — nothing survives correction, and the sample is borderline

| feature | median coupled | median independent | AUC | 95% CI | p | Bonferroni |
|---|---|---|---|---|---|---|
| duration (ms) | 47.0 | 55.0 | 0.437 | 0.379–0.496 | 0.039 | 0.193 |
| cycles | 6.57 | 7.69 | 0.425 | 0.366–0.478 | 0.014 | **0.072** |
| peak frequency (Hz) | 136.7 | 148.4 | 0.459 | 0.395–0.518 | 0.185 | 0.925 |
| spectral prominence (dB) | 9.31 | 8.93 | 0.499 | 0.444–0.554 | 0.983 | 1.000 |
| peak amplitude (µV) | 106.3 | 141.2 | 0.447 | 0.386–0.509 | 0.081 | 0.404 |

`min_detectable_auc(95, 1454) = 0.59`. The largest observed deviation from
chance is 0.425, equivalent to 0.575 — **just under the floor.** So this is
*not shown*, not *not there*: the sample could not have resolved the effect it
appears to be seeing.

### Line length — one feature separates, and it is weak

| feature | median coupled | median independent | AUC | 95% CI | p | Bonferroni |
|---|---|---|---|---|---|---|
| duration (ms) | 53.0 | 54.0 | 0.484 | 0.440–0.530 | 0.465 | 1.000 |
| cycles | 8.73 | 7.85 | 0.549 | 0.508–0.592 | 0.029 | 0.146 |
| **peak frequency (Hz)** | **174.8** | **148.4** | **0.585** | 0.542–0.625 | <0.001 | **0.001** |
| spectral prominence (dB) | 9.84 | 9.87 | 0.456 | 0.420–0.492 | 0.048 | 0.238 |
| peak amplitude (µV) | 66.0 | 85.4 | 0.443 | 0.406–0.481 | 0.012 | 0.058 |

`min_detectable_auc(181, 2007) = 0.57`, and the observed 0.585 clears it. So
this one is real on this recording: spike-coupled events peak about 26 Hz
higher. **AUC 0.585 still cannot classify an individual event** — it is a
distributional difference, not a label.

### The finding that matters is that the two detectors disagree

On peak frequency, RMS gives AUC **0.459** — coupled events slightly *lower* —
and line length gives **0.585** — coupled events clearly *higher*. Same
recording, same discharges, opposite sign. Two detectors that differ only in
the feature they threshold should not disagree about the direction of a
property of the underlying ripples, and the most economical explanation is
that this is a fact about *which events each detector selects* rather than
about the ripples themselves.

**So the pipeline reports two rates and does not label either one.** A split
that cannot be seen consistently in the waveform is a split you carry forward,
not one you resolve.

### Does the split localise the resection better? Measured: no

The question that would make the split clinically interesting is not whether
the two populations *differ* — measured above, weakly — but whether either one
predicts the resected zone better than the merged rate does. Screened across
all 20 ds003498 patients, on the five cached 60 s windows that tile each
`run-01`, pooled by subject mean, with the outcome study's own metric
functions so the `merged` arm is comparable to §0b. Committed extract:
[`data/outcome/subpopulation_screen.csv`](../data/outcome/subpopulation_screen.csv).

**Ripple band** (80–250 Hz). Bonferroni across the nine rows; `n` is
seizure-free against recurrence, and `floor` is that arm's own
`min_detectable_auc`:

| metric | population | n | mean SF | mean rec | AUC | 95% CI | p | Bonferroni | floor |
|---|---|---|---|---|---|---|---|---|---|
| top_channel_resected | merged | 13/7 | 0.600 | 0.229 | 0.747 | 0.527–0.934 | 0.072 | 0.644 | 0.85 |
| | spike_coupled | 13/6 | 0.600 | 0.667 | **0.487** | 0.231–0.737 | 0.950 | 1.000 | 0.87 |
| | independent | 13/7 | 0.538 | 0.171 | 0.736 | 0.516–0.923 | 0.073 | 0.656 | 0.85 |
| candidates_resected | merged | 13/7 | 0.560 | 0.369 | 0.648 | 0.396–0.868 | 0.311 | 1.000 | 0.85 |
| | spike_coupled | 13/6 | 0.509 | 0.488 | **0.513** | 0.218–0.795 | 0.949 | 1.000 | 0.87 |
| | independent | 13/7 | 0.515 | 0.363 | 0.621 | 0.363–0.852 | 0.403 | 1.000 | 0.85 |
| share_in_rz | merged | 13/7 | 0.379 | 0.393 | 0.527 | 0.253–0.791 | 0.877 | 1.000 | 0.85 |
| | spike_coupled | 13/6 | 0.508 | 0.536 | **0.474** | 0.218–0.731 | 0.898 | 1.000 | 0.87 |
| | independent | 13/7 | 0.375 | 0.385 | 0.516 | 0.253–0.780 | 0.938 | 1.000 | 0.85 |

**Every sub-population gain over the merged rate is negative**: −0.011,
−0.027, −0.011. The spike-coupled arm sits at chance on all three metrics, and
on `top_channel_resected` the sign is *reversed* — 0.600 in the seizure-free
patients against 0.667 in the recurrences.

**The arms are not scored on quite the same patients**, which the `n` column is
there to show. A patient whose spike-coupled population is empty in all five
windows has no share and no busiest channel in that arm, so it drops out of
that arm alone — and putting a 0 there instead would read as "none of this
patient's events were resected", which is a different claim. In the ripple band
that costs one patient (sub-08, a recurrence) from the coupled arm and nobody
from the other two. In the fast ripple band it costs two from the coupled arm
(sub-08 and sub-10) and one from the independent arm (sub-05) — which is a
second reason its numbers are not the ones reported.

### Why the fast ripple band could not answer this, and what it looked like

The pre-specified arm for the outcome study is fast ripples, and it is the arm
this screen cannot use — **the spike-coupled population there is too small to
rank channels with**:

| | coupled events per window: min · Q1 · median · max | windows with <5 | subjects whose median is <5 |
|---|---|---|---|
| fast ripple | 0 · 0 · **1** · 54 | 56/100 | 11/20 |
| ripple | 0 · 17 · **36** · 306 | 8/100 | 1/20 |

An argmax over a median of one event is a coin flip, and it duly produced the
best-looking numbers in this whole section: spike-coupled
`top_channel_resected` **0.806** (p = 0.025) and `share_in_rz` **0.826**
(p = 0.021), both apparently beating the merged rate, both on 12 patients
against 6. Neither survives Bonferroni, neither clears its own 0.88 floor, and
both are artifacts. Three things say so:

- **The tie-aware metric contradicts them.** `candidates_resected` — which
  exists precisely to catch an argmax over a set that is mostly ties — puts
  the fast-ripple spike-coupled arm at **0.479**, worst of the three and below
  chance, while it agrees with the other two arms.
- **The merged arm replicates across bands and the coupled arm does not.**
  `top_channel_resected` for merged is 0.753 (fast ripple) against 0.747
  (ripple). Only the coupled arm collapses, 0.806 → 0.487. That is the
  signature of noise in one arm rather than a property of one band.
- **The pipeline is otherwise behaving.** Merged `share_in_rz` falls
  0.714 → 0.527 from fast ripples to ripples, reproducing the standard
  finding that fast ripples localise better. The band effect that *should* be
  there, is.

**What this does and does not establish.** It does not show that the split is
worthless. Group sizes this small resolve very little: the floor is **0.85** at
13/7, **0.86** at 12/7, **0.87** at 13/6 and **0.88** at 12/6, and **not one of
the eighteen arms clears its own floor** — the merged 0.747 included. What it
shows is that *on the evidence available here there is no gain to report*, and
that the one result which looked like a gain came from a population of one
event per minute. The ripple band answer is a **secondary** analysis: it is not
comparable to the published fast-ripple AUC, and it was run only because the
pre-specified arm was unmeasurable.

One thing the split does measure reliably. The spike-coupled *share* of all
ripples is a stable patient property — within-subject SD across windows 0.013
against between-subject SD 0.063, so ICC ≈ 0.96. It is a real and repeatable
feature of a patient. It just does not predict where the surgeon cut.

Reproduce (offline, from the slices already in `artifacts/data/`, ~40 min):

```bash
ONSET_HFO_OFFLINE=1 python scripts/run_subpopulation_outcome.py
# or re-print the tables above from the committed extract, no re-analysis:
python scripts/run_subpopulation_outcome.py \
  --from-csv data/outcome/subpopulation_screen.csv
```

### What this is not

- **Discharge co-occurrence is a proxy, not a label.** No archive here marks
  which ripples were physiological. The sub-populations are named
  `spike_coupled` and `independent` for exactly that reason.
- **The morphology comparison above is one recording, one patient, 60
  seconds.** The localisation screen is the whole 20-patient cohort, but on
  five 60 s windows rather than the whole 300 s run, and in the ripple band
  rather than the pre-specified fast ripple one. Neither is the study a
  clinical claim would need.
- **The negative localisation result is "no gain to report", not "no
  effect".** At 13 seizure-free against 7 recurrences the screen could only
  have resolved an AUC of 0.85 or better, and no arm in it reaches its own
  floor. It is strong enough to retire the apparent fast-ripple gain, not to
  rule out a real one.
- **And the arms are not scored on identical patients.** A patient with an
  empty sub-population drops out of that arm; one does in the ripple band,
  three across the fast ripple one.
- **No classifier, and none is planned on this evidence.** Training one on a
  proxy and reading it as the thing is the failure mode this section exists to
  avoid.

Reproduce: `onset_hfo.populations.compare_populations` on the events of
`data/example_analysis/`. It takes a few minutes — the permutation is exact
where that is affordable and sampled at 200,000 draws where it is not.

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

---

## 6c. Is the robustness multiplier the right rule? Measured: only where the band is well sampled

§6b ends on a caveat rather than a result: on `sub-pt01` re-planning re-ordered
the top five, but the five channels were *tied*, a 5% penalty was enough to
shuffle them, and the re-ordering carried no information. That was one
recording with no surgical reference. This section answers the same question
across 20 patients who had surgery.

**What is compared.** The multiplier is measured as a *ranking rule*, not
through the ladder. Varying the planner as well would confound the answer, and
the ds003029 cohort the ladder wants is not available offline (one subject is
cached, its clinical sheet is not). So the rule is applied directly to the
ds003498 slices, holding everything else fixed — same prepared recording,
detector, survey threshold, channels, metric functions:

| rule | score | which rungs produce it |
|---|---|---|
| `plain` | `survey_rate` | S0, S1 |
| `multiplied` | `survey_rate × robustness` | S2, S3 |

`tests/test_orchestration.py` pins the premise that makes this a fair account
of S0: `FIXED_PLAN` makes exactly one `detect_hfo` call and sets no threshold,
so S0 sees one threshold per channel, the stricter loop never runs, and score
reduces to the survey rate *exactly*. The planner's robustness depends on which
stricter thresholds the model chose, so that choice is swept — 1.25×, 1.5× and
2.0× the survey threshold — rather than fixed at one arbitrary point.

Committed extract:
[`data/outcome/robustness_ablation.csv`](../data/outcome/robustness_ablation.csv).

### Ripple band: the multiplier helps, consistently

Survey 2.0 SD, 984.5 events per window (median), tied set of 5. **No patient is
lost from any arm**, so every cell below is 13 seizure-free against 7
recurrences.

| metric | plain | ×1.25 | ×1.5 | ×2.0 | best gain |
|---|---|---|---|---|---|
| `top_channel_resected` | 0.747 | **0.813** | 0.786 | 0.813 | **+0.066** |
| `tied_set_argmax_resected` | 0.736 | **0.830** | 0.813 | 0.797 | **+0.093** |
| `top3_resected` | 0.698 | 0.687 | **0.725** | 0.714 | +0.027 |
| `share_in_rz` | 0.527 | 0.527 | 0.560 | 0.615 | +0.088 |

All three re-test points beat plain on both argmax metrics, so this is not one
lucky choice of re-test threshold. The largest and most consistent gain is on
`tied_set_argmax_resected` — the metric built to isolate what the multiplier is
*for* — where permutation p falls from 0.086 to 0.014. **This is the first
evidence in this repository that the mechanism distinguishing S2/S3 from S0/S1
carries information rather than noise.**

### Fast ripple band: it hurts, on every patient

| metric | plain | ×1.25 | ×1.5 | ×2.0 |
|---|---|---|---|---|
| `top_channel_resected` | **0.753** | 0.643 | 0.643 | 0.593 |
| — patients scored | 13/7 | 13/7 | 13/7 | 13/7 |
| `tied_set_argmax_resected` | **0.687** | 0.610 | 0.621 | 0.621 |
| — patients scored | 13/7 | 13/7 | 13/7 | 13/7 |

**Every patient is now scored in every arm**, which is the difference between
this table and the one published before the rule was fixed. Previously the
stricter pass could find zero events on *every* channel, setting robustness to
0 everywhere and collapsing the score to uniform zero — no leader at all, a NaN
for the window, and at 2.0× all five windows of five patients gone, a quarter
of the cohort. `rank_channels` now refuses to fire on a re-test that measured
nothing (see `unmeasured_at`), so no patient drops out for that reason.

**What the fix changed, decomposed.** A silent re-test occurred in 84 of 600
multiplied windows (82 fast ripple, 2 ripple). Those 84 split two ways, and the
earlier write-up conflated them:

* **36 were real annihilation** — the survey found events, the stricter pass
  found none anywhere, and a working ranking was zeroed. These are rescued: the
  score is now the survey rate, robustness 1.0.
* **48 had no events at the survey threshold either.** A fast-ripple window with
  nothing in it scores zero because there is nothing there, not because the rule
  misfired. Those still read zero, correctly, and the fix deliberately does not
  rescue them: there is no rate to defend.

No window collapsed with a *non-silent* re-test, so the rule still bites exactly
where it should.

**The fast-ripple harm survives the fix, and is now cleaner.** It could
previously have been argued as survivorship. It cannot now, because nothing
drops out: on the complete 13/7 cohort plain scores 0.753 (p = 0.038) and every
multiplied arm is worse — 0.643, 0.643, 0.593 (p = 0.275, 0.275, 0.523).
`tied_set_argmax_resected`, which already kept all 13/7 for both rules, agrees:
0.687 plain against 0.610–0.621 multiplied. Two arms *rose* against the old
table (2.0× went 0.528 → 0.593) and that is not an improvement in the rule — it
is five patients rejoining the comparison. **The multiplier hurts in the fast
ripple band, measured on every patient.**

### What the rule is actually doing

| band | mean robustness (×2.0) | channels demoted per window | leader moved |
|---|---|---|---|
| ripple | 0.218 | 26.2 | 44% of windows |
| fast ripple | 0.781 | 7.7 | 42% of windows |

The two bands get wildly different treatment from the same rule, because
robustness is a ratio of rates at two thresholds and that ratio is governed by
the band's SNR distribution. So the rule is **not wrong and not right — it is
conditional on the band being well sampled**, which is a defect in a rule
applied unconditionally.

### Two conclusions that do not depend on any AUC

- **The rule should refuse to fire when the stricter pass finds nothing.** A
  robustness of 0 obtained because a band went silent means *unmeasured*, not
  *refuted*. The planner already makes exactly this distinction in the other
  direction — a channel nobody re-tested keeps robustness 1.0, documented as
  "*unchallenged*, not *verified*". The zero case is the same error mirrored,
  and it is worse, because it destroys the ranking instead of leaving it alone.
- **`candidates_resected` cannot be computed on a multiplied score at all.**
  `candidate_channels` decides which channels are tied by overlapping
  **Poisson** intervals, which needs an integer event count; `rate ×
  robustness` is not one. So this project's own tie-aware answer to "did the
  surgeon remove what the map pointed at" — added in item 10 step 3 precisely
  to stop an argmax overclaiming — is undefined for the planner's own score. It
  is reported for the plain rule and left NaN for the other, rather than
  computed on a truncated count that would look like a number.

### What this does not establish

The power floors are 0.85 at 13/7, 0.87 at 11/7 and 13/6, 0.88 at 9/6. **The
best AUC anywhere in this section is 0.830** — ripple `tied_set_argmax_resected`
at ×1.25, on 13/7 — so **not one arm clears its own floor.** The ripple gain is
*consistent with* a real effect, direction-consistent across three re-test
points and two metrics, and it is not a demonstrated one. The fast-ripple
degradation is the better-supported half, because it survives the one metric
that keeps every patient.

This is also still the scripted planner's mechanism measured without the
planner. It says the rule can carry information in a well-sampled band; it says
nothing about whether a language model would choose re-tests that exploit it.

Reproduce (offline, from the slices already in `artifacts/data/`, ~50 min):

```bash
ONSET_HFO_OFFLINE=1 python scripts/run_robustness_ablation.py
# or re-print the tables above from the committed extract, no re-analysis:
python scripts/run_robustness_ablation.py \
    --from-csv data/outcome/robustness_ablation.csv
```

## 7. Test suite

`pytest -q` — 620 tests, entirely offline. They cover the
primitives (robust scale, sliding features, threshold segmentation, bipolar
pairing), the detectors (hot channels found, events are oscillations, a flat
channel yields nothing, thresholds behave monotonically, reruns are
identical), validation (precision rises, recall survives, pops are rejected),
the report (no recommendation field, every finding carries evidence), and the
whole agent (tool schemas, argument validation, scope refusals, citation and
number verification, and the language-model path against a mock
OpenAI-compatible server that replies the way Qwen and Llama servers do —
including the two ways small models get it wrong).

Three files cover the desktop reviewer, split by what they need.
`tests/test_review_core.py` (51) and `tests/test_review_anatomy.py` (27)
import no Qt and run on any machine with the base install: that the reviewer's
rates agree with its own event list, that annotations land in the trace's time
base rather than the archive's, that the trend accounts for every event at any
bin width, that the exported review needs no optional dependency, and that
every contact position carries whether it was measured or inferred.
`tests/test_review_app.py` (46) needs a display and skips itself without the
`review` extra: that every panel builds, that each control calls the browser
method it claims to, and -- the only assumption the window's whole layout rests
on -- that MNE's Qt figure is still a `QMainWindow` and can host our dock
widgets.

`tests/test_pipeline.py` grew a preprocessing section when those steps became
reachable from the interface. It pins two things: that the measured defaults
produce byte-identical output to before the controls existed, and that every
setting which would yield numbers rather than a measurement — a low-pass under
the high-pass, a low-pass above the Nyquist left by downsampling, a
non-positive notch width or sampling rate, excluding every channel — raises
rather than running. A companion test asserts that the panel warns about
exactly what the pipeline refuses, because a warning the pipeline does not
enforce teaches a reviewer to ignore warnings, and a refusal the panel did not
predict arrives as a failure after a minute of work. It has already caught one
drift in each direction.

The split is load-bearing rather than tidy. All of it used to live in one file
under a module-level `importorskip`, and a skip there takes the whole module
with it: on a machine without Qt, which is what the main CI job is, **none** of
the numbers were checked while the suite reported green. Verified by installing
`.[dev]` without the extra -- before, 0 of 62 ran; after, 49 do.

`test_the_modules_this_file_covers_import_no_qt` keeps the boundary. It parses
this file's own imports and loads each one in a subprocess with `qtpy`,
`PySide6`, `PyQt5` and `PyQt6` blocked by a meta-path hook -- a subprocess
because by then pytest has already imported Qt in a full run, and a module
reaching for it would be handed the live one and pass. The first version of the
guard hardcoded its module list and missed the second breach entirely, which is
why it now reads the file rather than a list. It has caught two: the
preprocessing warnings, which moved to `onset_hfo.preprocess` beside the
refusals they mirror, and `patient_record`, which moved to
`onset_review.record`.

The anatomy file is almost entirely about provenance rather than arithmetic.
Neither archive ships electrode coordinates, so the 3D view's contact positions
are inferred from electrode names, and a picture of contacts on a brain is read
as an implantation plan unless it works hard not to be. The tests pin the
`source` column on every row, the wording of the caption that has to appear
under any drawing of it, that a mixed layout is reported as schematic, and that
the unexercised path -- real coordinates from a BIDS `electrodes.tsv`, in
millimetres -- is used and declared when one is present.

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
