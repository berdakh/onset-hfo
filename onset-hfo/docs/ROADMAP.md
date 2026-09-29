# Roadmap

**Start with [What is left, in order](#what-is-left-in-order).** That table is
the current answer and it is kept current; everything after it is history,
numbered in the order the items were *raised* rather than the order they should
now be done.

Each item names the files it touches and roughly what is involved, so someone
joining can pick one up without a handover meeting — and the finished ones are
kept rather than deleted, because what an item found (and what it cost) is the
most useful thing to read before starting the next one.

## What is left, in order

Eleven numbered items below carry the history — why each mattered, what it
found, and what it cost. This section is the short answer: **what is still
open, and which of it is blocked on what.** Five items are open; the other six
are done and kept for what they found.

| | Open work | State | Blocked on |
|---|---|---|---|
| **1** | [**Run the agent ladder under a real model**](#8-the-agent-more-tools-and-a-measured-evaluation-of-it) | harness done, **numbers missing** | a GPU afternoon |
| **2** | [**A second cohort**](#11-a-second-cohort) | not started | finding an archive |
| **3** | [The ladder across the 22-subject cohort](#2-outcome-as-the-reference-standard--done-twice-the-ladder-across-the-cohort-is-not) | cheap; recordings cached | deliberately gated on #1 |
| **4** | [A hand-annotated benchmark](#5-a-small-hand-annotated-benchmark--harness-built-marking-not-done) | **harness built**, unmarked | two reviewers' afternoons |
| **5** | [Electrode geometry](#3-electrode-geometry--blocked-on-this-dataset-needs-a-different-archive) | write it against the schema | an archive with coordinates |

**What just came off the board.** "Sub-populations against outcome" was item 5
and is [answered](#6-physiological-versus-epileptic-ripples--reported-and-the-localisation-question-answered),
negatively: splitting ripples by discharge coupling does **not** localise the
resection better than the merged rate. It was thought to need the archive; it
did not, because the cached 60 s slices were enough once the screen pooled five
of them per patient.

**Why #1 is first.** Every orchestration number this project has published
comes from the deterministic scripted planner, which the docs have called
"the control, not the result" throughout. `onset_agent/benchmark.py` and
`notebooks/06_agent_benchmark.ipynb` now sweep model × quantization × rung,
checkpoint each cell, and record the accelerator and library versions
automatically. The machinery is tested offline; what is missing is a GPU and
an afternoon. A free Colab T4 covers seven of the nine cells.

**Why #2 is second.** Everything measured here comes from one centre, one
annotation protocol and one surgical team. The reproducibility result — that a
whole recording is a stable unit and a minute of one is not — is the finding
most worth testing elsewhere, and the one most likely to hold.

**#3 is gated on purpose.** Running the ladder over 22 patients before a real
model has driven it would produce 22 patients' worth of *scripted-planner*
numbers, which measure the script rather than the thesis.

Not on this list, because it lives in its own document with its own ordered
plan: the expo material ([`EXPO.md`](EXPO.md), [`POSTER.md`](POSTER.md) — the
claim is written, the board is not built).

The two-repository cleanup ([`DUPLICATION.md`](DUPLICATION.md)) is **done**:
all eight items, on both sides, closed by onset-hfo#29, onset-hfo#30 and
berdakh/onset#3. Item 4 was revised rather than executed — that document
records why, and it is the more useful half of the outcome.

---

## History

---

## 1. Interictal recordings — *done*

**Status.** `ds003498` — Zurich interictal slow-wave sleep at 2000 Hz, with
expert HFO markings — is wired in, and every headline number in the project now
comes from it: the detector benchmark (`docs/EVALUATION.md` §0) and the outcome
study (`docs/OUTCOME.md`). The ictal dataset remains as the quickstart example.
Fast ripples are analysable there too, which they never were at 1000 Hz.

The original reasoning, kept because it explains why this was item 1:

**Why.** Everything in this repository was measured on ictal data, because that
is what `ds003029` publishes with signals attached. The clinical HFO
literature measures **interictal** rate, usually in slow-wave sleep. Until the
pipeline runs on interictal data, its numbers cannot be compared to that
literature at all.

**What.** Find a public interictal iEEG dataset with a high sampling rate
(candidates: other OpenNeuro iEEG datasets; the Montreal/Zurich HFO datasets
distributed with `mne-hfo`; SWEC-ETHZ; institutional data under your own
approvals). Add a loader beside `fetch_slice` — the `Recording` dataclass is
the only contract.

**Do not be misled by this archive's `task-interictal` files.** There are 25
of them and all but one are metadata with no signal attached; exactly one
interictal recording (`sub-umf002`, run 01) ships an `.eeg`. Checking this
takes one S3 listing and saves a week of building against a dataset that is
not there.

**Touches.** `onset_hfo/datasets.py`, `docs/DATA.md`.

---

## 2. Outcome as the reference standard — *done twice; the ladder across the cohort is not*

**The stronger version is now built, on the other dataset.** ds003498 ships
both halves of the question — which contacts were resected
(`sourcedata/clinical_ch_sheet_zurich.xlsx`) and whether the patient became
seizure-free (`participants.tsv`) — so the HFO map can be tested against
surgery rather than against a clinician's marker. [`OUTCOME.md`](OUTCOME.md)
has the design and the result: the expert markings reproduce the published
finding (busiest fast-ripple channel inside the resection in 12/13
seizure-free vs 2/7 recurrences, AUC 0.82, p = 0.007) and **our detector does
not reach significance on the same metric** (AUC 0.70, p = 0.13).

**That gap was mostly the analysis window, which is a more important finding
than the gap.** Re-running on whole 300-second runs instead of the first 60
seconds moved the expert arm from AUC 0.82 (p = 0.007) to 0.71 (p = 0.12) and
ours from 0.70 to 0.67 — the gap closed because the *expert* number came down,
not because ours went up, and nothing in the study survives correction. Two
patients account for all of it.

So the top open item is **not** "make the detector better", it is
**item&nbsp;10: is the channel ranking stable at all?** A `top_channel_resected`
claim is an argmax over 6–65 channels whose Poisson rate intervals overlap.
`metrics.py` already refuses to rank channels whose intervals overlap when it
reports rates; the outcome metric does not, and it should. Adding that, and
plotting AUC against window length (cheap — the recordings are cached after
the first run), comes before any detector work.

The detector still has no morphology criterion where the source study's did,
and that remains worth building — just not as the thing that would have
explained a gap the data says is mostly noise.

**The earlier, weaker version (ds003029, clinician SOZ contacts) is also
done:**

**What turned out to be true.** The archive carries more than outcome scores.
`sourcedata/clinical_data_summary.xlsx` gives **curated clinician SOZ
contacts** per patient, alongside Engel, ILAE, surgery type and clinical
centre. `onset_hfo/cohort.py` reads it, reconciles the subject ids, decodes
the `S`/`F` outcome trap, and falls back to a local CSV or the free-text
markers — recording which source it used. 32 of the 35 subjects with signals
have a row, and parsed contact names match `channels.tsv` exactly.

**The cohort run is done.** `onset_hfo/batch.py` analysed **22 of 31 planned
subjects — 1466 channels, 301 labelled SOZ** — and the table is committed at
`data/cohort/features.csv.gz`, so the modelling half runs with no download.
`docs/LOCALIZATION.md` has the results: the learned model barely beats the
rate it was built from (0.480 against 0.467 AUPRC), the within-subject ceiling
is far above both (0.709), and five clinician-labelled contacts recover 38% of
that gap.

The nine exclusions are findings in their own right, and anyone planning a
cross-site experiment on this archive needs them first: **four UMMC recordings
sample at 250 Hz**, where an 80–250 Hz ripple is not measurable at all, and
four UMF signal files are shorter than their own marked seizure time. So
"leave-one-site-out" here is really NIH (13) against JHH (6), with UMF and
UMMC contributing one and two subjects.

**What is still to do.** The S0–S3 ladder has only ever been run on one
patient. Running it across the 22 gives per-rung SOZ localization with an
interval around it instead of `sub-pt01`'s anecdote, and stratification by
`seizure_free` so the trustworthy positives (clinician named it *and* the
surgery worked) are scored apart from the ambiguous ones. It is cheap now that
the recordings are cached — but it is worth doing **after** a real model has
driven the ladder at all (item 8), because 22 patients' worth of scripted-
planner numbers measure the script, not the thesis.

**Touches.** `onset_agent/orchestrate.py` (a cohort loop over the ladder),
`docs/EVALUATION.md`. `batch.py`, `cohort.py`, `models.py`, `uncertainty.py`
and `onset_agent/scoring.py` already exist.

---

## 3. Electrode geometry — *blocked on this dataset; needs a different archive*

**Why it matters.** Bipolar pairs are formed from consecutive contact
*numbers*. On a grid, numbering wraps at the end of a row, so some pairs join
contacts that are centimetres apart. Every rate computed on such a pair is
suspect.

**Why it cannot be done here.** `ds003029` publishes **no `electrodes.tsv` for
any subject**. There are no coordinates in the archive at all. Anything that
needs geometry — distance-based pairing, distance-to-neighbour features,
source localisation — requires a different dataset.

**What to do instead.** Write the code against the BIDS `electrodes.tsv`
schema so it is ready, pair by Euclidean distance with a maximum where
coordinates exist, fall back to numbering where they do not, and state which
rule was used in the report's method section. Then validate it on an archive
that ships coordinates.

**Touches.** `onset_hfo/preprocess.py`, `onset_hfo/datasets.py`, `report.py`.

---

## 4. More than two detectors, and a proper agreement analysis — *done*

**Why.** Two detectors matching about half their events is a finding worth
taking seriously. Three or four would show whether the disagreement is
idiosyncratic or structural.

**Built.** `detect_hilbert` (the smoothed analytic-signal envelope) and
`detect_short_time_energy` (the sum of squares), both through the same
`detect_with_feature` engine, so the four differ only in the feature they
threshold. `metrics.agreement_matrix` reports every pair, and
`metrics.consensus_ranking` ranks channels by the *number of detectors* that
place them in their own top group rather than by an average of four rates on
four different scales. Both are **opt-in**: `run_pipeline` still defaults to
two, because turning them on would change every published number without
anyone deciding to.

**What it found, on the simulator** ([`EVALUATION.md`](EVALUATION.md) §1b):

* **The disagreement is structural.** Every pair of the four agrees on between
  47% and 79% of events. No two of these detectors agree on more than four
  events in five.
* **A prediction written into the code was wrong**, which is why it was
  written down first. `short_time_energy` was expected to be nearly redundant
  with `rms` — they are monotone-related, so a *fixed* threshold would select
  identical samples. It is the **least** similar pair (0.471), and the closest
  pair is RMS with the envelope (0.788).
* **The cause generalises, and is the actual result.** `median + 5 robustSD`
  sits near the 98th percentile of an RMS trace and the 96th of an energy
  trace, because squaring is not affine. **A threshold in robust SDs is not a
  portable operating point between features.** §3 said the threshold is a
  choice rather than a fact; this says the choice does not transfer.

**Swept on real data, and the answer is no.** 20 patients of ds003498, first
60 s of run-01 each, all four detectors over a common threshold grid in both
bands. At each detector's *own* best threshold, by channel-rank agreement:

| band | best | rank ρ | RMS | margin |
|---|---|---|---|---|
| ripple | short-time energy @ 4.0 SD | 0.658 | 0.655 @ 2.0 SD | **+0.003** |
| fast ripple | Hilbert envelope @ 5.0 SD | 0.613 | 0.610 @ 5.0 SD | **+0.003** |

**Two more detectors, each tuned on real data, bought no measurable
improvement in channel ranking.** Three thousandths of a Spearman correlation
across twenty patients is a tie. That is worth knowing: the ceiling here is
not the choice of feature.

**The portability claim, measured.** Optimal threshold in robust SDs —
short-time energy wants **2× the ripple threshold and 2.4× the fast-ripple
threshold** that RMS does, while the envelope wants almost exactly what RMS
wants:

| | RMS | line length | Hilbert | short-time energy |
|---|---|---|---|---|
| ripple | 2.0 | 1.5 | 1.5 | **4.0** |
| fast ripple | 5.0 | 5.0 | 5.0 | **12.0** |

Run short-time energy at RMS's 5.0 SD in the fast-ripple band and rank
agreement falls from 0.601 to 0.485. So the synthetic conclusion holds, and
one synthetic *verdict* was too harsh: "mostly the energy detector at an
untuned, lower operating point" is right about a shared threshold and wrong
about the detector. At its own it is competitive.

**Four of sixteen arms first peaked on an endpoint** — the grid running out,
not an optimum, which is the §0 lesson. Extending moved one materially:
short-time energy's fast-ripple rank agreement went 0.570 at 8.0 SD → **0.601
at 12.0**. All sixteen are now interior, and a test enforces it.

**And the block was never the network.** The board read "blocked on a
`benchmark` run with network access" for weeks. All 20 subjects were cached at
exactly this window; only `list_subjects` needs the network, and naming the
subjects skips it. The real blocker was `benchmark.py` holding a private
two-entry copy of the detector registry, so `--detectors hilbert` raised a
KeyError rather than sweeping. One registry now, and a test that every module
sweeps the same set.

Tables: [`EVALUATION.md`](EVALUATION.md) §1b. Extract:
`data/benchmark/four_detector_sweep.csv`.

**Touches.** `onset_hfo/detectors/` (done), `metrics.py` (done),
`config.py` (done), `report.py` — the report still shows one pairwise
agreement rather than the matrix.

---

## 5. A small hand-annotated benchmark — *harness built, marking not done*

**Why.** The simulator can only measure what it simulates. "Precision 0.97 on
synthetic ripples" is a statement about the generator; a few hundred windows
marked by two people would make it a statement about recordings.

**Status.** `onset_hfo/review.py`, `onset-hfo review` and
`notebooks/07_annotation.ipynb` are the half that does not need people. What
remains is two reviewers' afternoons, and it cannot be faked.

**Most of what makes such a benchmark worthless is in that built half**, which
is why it was worth building before anyone marks anything:

1. **The sample is drawn from the recording, not from the detector.** Review
   only the detector's own hits and you can measure precision and *never*
   recall, because a window it did not propose is never looked at — and the
   reference becomes a function of the thing being scored. Three strata,
   equal numbers: accepted candidates, rejected candidates, and **background
   windows with no detection in them**. The third is the only one that can
   ever reveal a false negative.
2. **The reviewer is blinded.** The manifest carries four columns — an opaque
   id, a channel, two times. The stratum and the detector's verdict go to a
   separate `.key.csv`, so handing over the wrong file is a visible mistake.
   The sampler checks its own output for ordering that would leak the stratum
   through the numbering.
3. **Agreement is chance-corrected.** Most windows are not ripples, so two
   reviewers who say "no" to everything agree ~90% of the time. Cohen's kappa
   with a bootstrap interval, the prevalence beside it, and `unsure` as a
   first-class answer that is dropped and *counted* rather than guessed.
4. **The ceiling is printed before any score.** Two reviewers agreeing at
   kappa 0.6 define a reference no detector can be measured against more
   finely than that. `ceiling_note` says so in a sentence.
5. **Scores are per stratum, never pooled.** The strata were sampled in equal
   numbers rather than in proportion, so a pooled figure would weight a
   background window as heavily as a candidate and describe a recording that
   does not exist. Both consensus rules (`both` / `either`) are reported
   because they bracket the answer, and quoting whichever is kinder is how a
   detector's precision gets published.

**A bug the harness caught on its first real run.** The shipped example is a
slice from 50–110 s. With the time origin left at its default the background
stratum was drawn from 0–60 s, so a third of the sample pointed at signal the
analysis does not contain — which shows up as a blank figure, not an error,
and would have cost a reviewer an afternoon before anyone noticed.
`sample_windows` now refuses a range its candidates do not live in.

**What is still left.** Two reviewers, a few hundred windows, and publishing
the annotations — the marks are the contribution, since a scored detector is
reproducible from them and nobody else has to spend the afternoon.

**Touches.** `onset_hfo/review.py` (new), `cli.py`,
`notebooks/07_annotation.ipynb` — 04 in the original plan, which was already
taken by the orchestration notebook.

---

## 6. Physiological versus epileptic ripples — *reported, and the localisation question answered*

**Why.** The single biggest scientific gap. A high ripple rate in healthy
occipital cortex is not a finding, and nothing in this prototype can tell the
two apart. Every rate published here merges the two.

**Status.** `onset_hfo/populations.py` splits events by whether they ride an
interictal discharge, `run_pipeline` reports the sub-populations side by side
(`populations_<detector>.csv`) and never merges them back, and the report
carries the caveat that neither sub-population is a label.

**And it measures whether the split is worth anything**, which is the part
that could have been skipped. Full tables in
[`EVALUATION.md`](EVALUATION.md) §3b; the short version:

- **RMS: nothing survives Bonferroni across five morphology features**, and
  `min_detectable_auc(95, 1454) = 0.59` against a largest observed effect
  equivalent to 0.575. *Not shown*, not *not there*.
- **Line length: peak frequency separates** — coupled events peak 26 Hz
  higher, AUC 0.585, Bonferroni p = 0.001, above that arm's 0.57 floor. Real,
  and far too weak to classify an individual event.
- **The two detectors disagree about the direction.** RMS puts coupled events
  slightly *lower* in peak frequency (AUC 0.459), line length clearly
  *higher* (0.585). Same recording, same discharges, opposite sign. Two
  detectors differing only in the feature they threshold should not disagree
  about a property of the ripples, so the most economical reading is that
  this is a fact about which events each detector selects.

**So no classifier, and none is planned on this evidence.** A split that
cannot be seen consistently in the waveform is one to carry forward, not to
resolve. Training on a proxy and reading it as the thing is the failure this
item exists to avoid.

**The interesting half, now measured: the split does not localise better.**
Whether the spike-coupled rate predicts the resection better than the merged
rate is the question that would make the split clinically meaningful. It was
listed here as needing the archive, and it did not: the cached 60 s slices are
enough once the screen uses five of them per patient and pools by subject mean.
Full tables in [`EVALUATION.md`](EVALUATION.md) §3b, extract in
[`data/outcome/subpopulation_screen.csv`](../data/outcome/subpopulation_screen.csv),
entry point `scripts/run_subpopulation_outcome.py`. The short version:

- **In the ripple band, every sub-population arm is at or below the merged
  rate** — gains of −0.011, −0.027 and −0.011 on the three outcome metrics.
  The spike-coupled arm sits at chance on all three (0.487, 0.513, 0.474), and
  on `top_channel_resected` its sign is reversed.
- **The fast ripple band — the pre-specified arm — could not answer it.** Its
  spike-coupled population has a median of **one** event per 60 s window, and
  fewer than five in 56 of 100 subject-windows.
- **And that near-empty population produced the best-looking numbers in the
  section**: 0.806 and 0.826, apparently beating the merged rate, neither
  surviving Bonferroni. Three checks retire them: the tie-aware
  `candidates_resected` metric puts the same arm at 0.479; the merged arm
  replicates across bands (0.753 → 0.747) while the coupled arm collapses
  (0.806 → 0.487); and the band effect that should be there is (merged
  `share_in_rz` falls 0.714 → 0.527 from fast ripples to ripples, as fast
  ripples localising better predicts).

**What it does not establish.** Group sizes this small resolve very little —
the power floor runs 0.85 to 0.88 depending on the arm, and **no arm clears its
own**, the merged 0.747 included. This is strong enough to retire the apparent
gain, not to rule out a real one; and the ripple-band answer is a secondary
analysis, not comparable to the published fast-ripple AUC. The arms are also
not scored on identical patients, because a patient whose sub-population is
empty in every window drops out of that arm — one patient in the ripple band,
three across the fast ripple one. **The lesson worth keeping** is that the
tie-aware metric earned its place: it was added in item 10 step 3 to stop an
argmax overclaiming, and here it is the thing that caught one.

One property of the split *is* reliable: the spike-coupled share of all ripples
is stable within a patient and varies between them (within-subject SD 0.013
against between-subject 0.063, ICC ≈ 0.96). It is a real patient feature that
does not predict the resection — which is a more useful thing to know than
either "no signal" or "promising".

**Touches.** `onset_hfo/populations.py`, `scripts/run_subpopulation_outcome.py`
(new), `data/outcome/subpopulation_screen.csv` (new), `pipeline.py`,
`report.py` via
the existing `data_quality` section — deliberately not a new report section,
because the agent's tool contract is frozen and enumerates the sections it may
read, and a stable contract is worth more than a tidier schema.

---

## 7. Scaling: whole recordings instead of one-minute slices — *done*

**Status.** `onset_hfo/streaming.py` and `onset-hfo stream` analyse a recording
longer than memory. `datasets.iter_slices` serves it in overlapping chunks
through the same byte-range fetcher the single-slice path uses, so there is no
second loader to keep in step.

**Why.** A minute is enough to demonstrate a method and not enough to measure a
patient. Rates in clinical studies come from ten-minute or hour-long
interictal windows.

**Memory, measured.** One RMS detector over a synthetic recording served from
disk one chunk at a time, 60-second chunks:

| recording | whole array | streamed |
|---|---|---|
| 2 min | 380 MB | 279 MB |
| 5 min | 663 MB | 345 MB |
| 10 min | 1,136 MB | 344 MB |
| 20 min | 2,071 MB | 349 MB |

The whole-array path grows at about 95 MB per minute of recording; the
streamed path is flat from five minutes on. An hour would be roughly 6 GB
against 350 MB.

**The hard part was never the memory.** Every detector fires at
`median + k × robustSD` *of the channel's own feature trace*. Let each chunk
measure its own median and the detector's answer starts depending on where the
boundaries fell — and it is not a rounding error: measured on a 60-second
synthetic recording in 10-second chunks, a per-chunk baseline **invented 89
events and lost 117, out of 460**. A busy chunk raises its own threshold and
hides its own events; a quiet one lowers it and invents them. This project has
already published a result that changed when the analysis window changed
(item 10); reintroducing the same defect as a scaling optimisation would have
been worse than not scaling at all.

So the module makes **two passes**: measure one median and one MAD per channel
over the whole recording, then detect with those numbers injected. Two passes
cost roughly twice the filtering, and buy an answer that does not depend on the
chunk size.

**What the tests hold it to**, in order of how much they matter:

1. Where the baseline sees every sample, streaming returns **exactly** the
   events the in-memory pipeline returns, at every chunk size tried. No
   boundary bug, no double counting, no truncation.
2. Above the subsample cap the result is still **exactly invariant to chunk
   size**, because the retained samples are chosen by absolute position in the
   recording rather than within the chunk.
3. The bounded-memory baseline costs **under 1%** of events against an
   exhaustive one (1 event in 460, measured).
4. A per-chunk baseline is asserted to be *wrong by more than 10%*, so nobody
   deletes the second pass as an optimisation.

**A bug caught in the writing.** The first sketch chose its subsample stride
from each chunk's own size, which is bounded and deterministic and still
wrong: 60-second chunks thinned four times as hard as 15-second ones and moved
three events out of 460 across the threshold. Bounded and deterministic is not
the same as chunk-independent.

**The refusals.** `plan_chunks` raises rather than warns when the overlap is
shorter than the longer of the band-pass filter's ring-in and the longest
acceptable event, and says which of the two bound. A short overlap does not
fail loudly — it quietly returns slightly wrong events near every boundary,
which is the kind of error that reaches a paper.

**What is still not here.** `run` and `benchmark` still take a slice: both
build a report and figures, which need the whole signal array by construction.
`stream` writes events and per-channel rates — what a long recording is
actually for — and says so rather than producing a partial report. Per-channel
parallelism is also absent; the detectors are embarrassingly parallel and this
runs them in a loop. Both are speed, and neither changes an answer.

**Touches.** `onset_hfo/streaming.py` (new), `datasets.py`, `detectors/base.py`
(`ChannelBaseline`), `detectors/engine.py`, `cli.py`.

---

## 8. The agent: more tools, and a measured evaluation of it

**Why.** The agent currently answers questions about one analysis. The obvious
next step — "compare this patient's ranking to their previous recording" —
needs multi-analysis tools, and a way to evaluate whether the agent's answers
are actually *useful* rather than merely verified.

**What has since been built.** The agent can now *drive* the analysis rather
than read it: a frozen JSON tool contract with run ids, nine live tools that
each re-run the real pipeline at parameters the planner chooses, an
append-only evidence store, the S0–S3 ablation ladder, three stopping rules,
and two verifiers with a measured delta between them. See
[`ORCHESTRATION.md`](ORCHESTRATION.md).

**Also built since:**

* ~~**Falsification tests.**~~ **Done.** `onset_agent/falsify.py` runs five:
  anonymised channel names, a shuffled name-to-signal mapping, the leading
  channel removed, a recording with no pathology, and run-to-run stability.
  The fourth one *failed* and produced `metrics.leader_separation` — see
  [`ORCHESTRATION.md`](ORCHESTRATION.md) §6b.
* ~~**The harness for the model and quantization ladder.**~~ **Done.**
  `onset_agent/benchmark.py` sweeps model × quantization × rung, checkpoints
  each cell so a disconnecting runtime resumes, records the accelerator and
  every library version per cell, and refuses to let wall-clock be compared
  across sessions. `notebooks/06_agent_benchmark.ipynb` is the Colab path;
  29 offline tests cover it.

**What remains, in order:**

* **The run itself.** Every ladder number published so far comes from the
  deterministic scripted planner. That is the control, not the result. The
  harness is written and tested; what is missing is a GPU and an afternoon.
  A free Colab T4 covers seven of the nine model × quantization cells — only
  8B and 14B at fp16 need a larger card.
* **A tool the rest of the system does not have**: rejected events with their
  reasons, and comparison across two recordings of the same patient.

**Touches.** `onset_agent/falsify.py` (done), `onset_agent/benchmark.py`
(done), `scripts/run_model_ladder.py` (done) — the depth run to this module's
breadth sweep.

---

## 9. Interface — *done*

**Status.** `app/` — `streamlit run app/Home.py`, installed with
`pip install -e ".[app]"`. Five tabs: ranking (with intervals and the "does
anything stand out?" verdict above the table), evidence (a channel's citable
events and the three-panel figure for the one you pick), disagreements (both
ranks, neither preferred), the agent (every citation expands to its stored
record *and* its signal window), and the report as written.

**What it is for.** Every number this project produces already carried the
signal window behind it, but only inside a JSON file — a reader could not
*look* at the window without writing code, which makes "evidence-based" a
claim rather than a property. The page closes that gap.

**The rule held.** Everything comes through `ResultStore` via `app/panels.py`,
which imports no Streamlit and is therefore tested offline. The page has no
thresholds and no analysis of its own, so it cannot disagree with the report
it displays. One documented exception: `leader_note` runs
`metrics.leader_separation` on the stored rate table, because a page that
showed a ranking without the "are they all tied?" verdict would be the most
misleading thing this project could ship.

**What building it caught.** The first version constructed the event for the
figure from times alone, so the figure printed `peak nan Hz, nan dB over
background` directly beneath a panel showing 192 Hz and 12.5 dB.
`panels.event_from_record` now copies every stored field, with a test named
after the incident. Two numbers disagreeing on one screen is the failure this
project is organised against, and an interface is where it surfaces.

**The cohort screen, added afterwards.** `5_Outcome` reported the group
tables; `6_Patients` is the same study at the grain a clinician asks about —
one row per patient, both sources side by side, and the per-patient caveats
(resection coverage, tie-set size, window and run stability) computed from
`data/outcome/` rather than written by hand. Five of the twenty patients trip
none of the four checks; the other fifteen do, and the page says which.

**What building *that* caught.** `subject_caveats` tested each stability flag
with `is False`. pandas hands back `numpy.bool_(False)`, which is not the
`False` singleton, so every unstable patient came back silently uncaveated —
on the one screen in this project where a missing caveat sits directly beneath
a single patient's answer. The fix is three lines; the test that would have
caught it (`test_every_patient_is_either_clean_or_carries_a_reason`, which
asserts that an empty caveat list means the checks *passed* rather than that a
join dropped a row) is the part worth keeping.

**The benchmark screen, added next.** `4_Detectors` now reads
`data/benchmark/` rather than restating it: the full sweep for all four arms,
a curve for any metric, and the best threshold per arm *derived* from the
table with an `at_boundary` flag beside it. A test asserts that the derived
optima are the values `config.THRESHOLDS` names, so a re-run that moves an
optimum fails in CI instead of leaving every document quoting a preset that no
longer matches.

**What that caught, which is the argument for the whole exercise.** The nine
hand-typed rows on the old page were all correct. The sentence beneath them
was not: it quoted the mean expert fast-ripple count as 228, `config.py`
quoted it as ~70 — which is this detector's own detection count at 5.0 SD, not
the expert count at all — and the true value is 278. Three copies of a number
that lived only in untracked `artifacts/`, two of them wrong, none of them
checkable from a clone. `data/benchmark/cohort.csv` now carries the
per-subject reference counts, and the page reads them.

Surfaced by the same change: the ripple line-length arm's F1 optimum sits on
the *lowest* threshold swept, so it is not an optimum — that arm was never
extended downwards the way the RMS arms were. The page labels it "not yet
measured" rather than reporting it as a detector that lost.

**Touches.** `app/`, reusing `onset_hfo/store.py`, `viz.py` and `metrics.py`.

---

## 10. Is the channel ranking stable? — *done*

**Why.** The outcome study's conclusion changed between a 60-second window and
a 300-second one (§2), moving on two patients out of twenty. Every clinical
claim this pipeline could ever make is a statement about *which channels*, so a
ranking that moves when the window moves undermines all of them — and unlike
most items here, this one is cheap and entirely offline once the recordings are
cached.

**Steps 1 and 2 are done** — `onset_hfo/stability.py`, `onset-hfo stability`,
and `docs/OUTCOME.md` has the figure and the tables. What they found:

* The growing-window curve **settles from 180 s** (identical AUC at 180, 240
  and 300 s), so the whole-run number is a settled estimate of one recording
  and the 60-second result was an excursion.
* Across five **disjoint** 60 s windows the expert AUC spans 0.566–0.819 and
  its p-value 0.007–0.613. The published 0.007 was the best of five minutes.
* **Our detector is more stable than the expert markings** — AUC spread 0.09
  against 0.25, same busiest channel in 12/20 patients against 7/20, same
  inside/outside answer in 16/20 against 9/20. Partly deflation, not entirely.
* Channel identity and the decision built on it come apart (7/20 vs 9/20 for
  the experts), because the top few channels are often all inside or all
  outside the resection.

**Step 3 is done.** `outcome.candidate_channels` applies the same rule
`metrics.leader_separation` uses -- a channel is tied with the leader when its
Poisson interval overlaps the leader's -- and `candidates_resected` reports
the share of the tied set that was removed. It reduces to the old metric
exactly when the set has one member. What it found:

* **The data picks a single channel in fewer than half the patients**: 9/20
  for the experts, 8/20 for us, median set size 2, worst case 24 tied
  channels out of 37. `top_channel_resected` was naming a winner that had not
  been chosen.
* **Honesty is nearly free at the group level**: AUC 0.709 -> 0.692 for the
  experts, unchanged at 0.670 for us.
* **Per-patient values move less** across disjoint windows (mean movement
  0.55 -> 0.30 expert, 0.20 -> 0.18 ours), **but the group statistic is not
  thereby stabilised** -- its spread halves for the experts and more than
  doubles for us, because in a 60 s window our detector sees a median of 30
  fast-ripple events, its intervals are wide, and the set size itself becomes
  noisy. Candidate sets do not rescue a short window; they make its
  uncertainty visible.

**Step 4 is done, and it is the good news of the four.** The archive holds
**385 runs** over the 20 subjects -- 1 to 39 each, not the 1-6 the `nights`
column suggests -- so `across_runs` reads the first five per subject and
prunes each slice after use. What it found:

* **A whole run is a stable unit of measurement; a minute is not.** The
  per-patient answer holds across five different nights for **18/20** patients
  (experts) and 16/20 (ours), against 9/20 and 16/20 across five minutes of a
  single run. The expert markings doubled their agreement with themselves; our
  detector was already there and gained nothing, because it had not lost
  anything.
* **Pooling a patient's runs resolves the ties.** Median candidate set 2 -> 1,
  worst case 24 channels -> 5 (expert) and 37 -> 5 (ours). This is what more
  recording was supposed to buy, and unlike the window study it delivers.
* **It does not rescue the group result.** Nothing in the pooled table reaches
  p < 0.05 (minimum 0.070), and pooling *lowers* the expert argmax arm from
  0.709 to 0.637 because more recurrence patients turn out to have had their
  busiest channel removed.

So the quantity this pipeline measures is a property of the **patient**, not
of the recording session. That was the precondition for anything clinical and
it now holds. What twenty patients cannot settle is whether the quantity
predicts outcome.

**What is left is no longer item 10.** Every question it posed has an answer.
The next one is item 11.

**Touches.** `onset_hfo/outcome.py`, `onset_hfo/metrics.py`,
`onset_hfo/stability.py`, `docs/OUTCOME.md`.

---

## 11. A second cohort

**Why.** Every measured number in this project comes from twenty patients at
one centre, marked under one annotation protocol, operated on by one surgical
team. That is the single largest threat to everything above it, and no amount
of further analysis of the same twenty patients addresses it.

**What to test first, and it is not the outcome result.** The outcome arm is
underpowered by construction — 13 against 7 detects only AUC ≥ 0.85 at 80%
power — so a second cohort of similar size would not settle it either. The
finding worth porting is the **reproducibility** one, because it is a
within-cohort comparison and does not depend on that power: does the
per-patient answer hold across whole recordings and move across minutes on
*other* patients, from other electrodes, in another hospital? If it does, "a
whole recording is a stable unit of measurement" becomes a statement about HFO
analysis rather than about Zurich.

**What it needs.** An archive with interictal recordings at ≥ 2000 Hz, several
runs per subject, and ideally resection or outcome information. The loader
contract is one dataclass (`Recording`), so a new archive is a loader and a
`DatasetSpec`, not a rewrite. Candidates worth checking before building
anything: other OpenNeuro iEEG datasets, SWEC-ETHZ, and institutional data
under your own approvals. Check that signal files actually exist before
planning around a dataset — this project lost time to `ds003029`'s 25
`task-interictal` entries, 24 of which are metadata with no `.eeg`.

**What would make it a stronger paper rather than a longer one.** Pre-register
the window analysis before touching the second cohort's outcome labels. The
first version of this study moved its own headline when the window grew, and
the credibility of the correction rests on the metric and band having been
fixed in advance. Doing that deliberately the second time is cheap and is the
difference between a replication and another exploratory run.

**Touches.** `onset_hfo/datasets.py` (a loader and a `DatasetSpec`),
`onset_hfo/stability.py` and `outcome.py` (unchanged if the loader is right),
`docs/DATA.md`.

---

## Open questions worth someone's attention

* ~~**Is the robustness rule the right rule?**~~ **Measured, and the answer is
  "only where the band is well sampled"** — see
  [`EVALUATION.md`](EVALUATION.md) §6c, extract in
  [`data/outcome/robustness_ablation.csv`](../data/outcome/robustness_ablation.csv).
  Ablated as a ranking rule across the 20 ds003498 patients against surgical
  outcome, sweeping the stricter re-test point over 1.25×, 1.5× and 2.0×:
  * **In the ripple band it helps**, at every re-test point, most on the
    tie-breaking metric it exists for (0.736 → 0.830, permutation p 0.086 →
    0.014). That is the first evidence here that the mechanism separating
    S2/S3 from S0/S1 carries information rather than noise.
  * **In the fast ripple band it hurts, and can annihilate the ranking.** When
    a sparse band goes silent at the stricter threshold, robustness is 0 on
    every channel and the score is uniformly zero — 84 of 600 multiplied
    windows, wiping out five patients entirely at 2.0×. The harm survives the
    one metric that keeps every patient, so it is not survivorship.
  * **The fix is to make the rule refuse to fire on a silent re-test.** A
    robustness of 0 from "no events at this threshold" means *unmeasured*, not
    *refuted* — the mirror image of the "unchallenged, not verified" case the
    planner already handles correctly. Not yet implemented; it changes
    `rank_channels`, which every orchestration number depends on, so it wants
    its own change.
  * And `candidates_resected` **cannot be computed on a multiplied score at
    all**: Poisson intervals need integer counts. The project's own tie-aware
    metric is undefined for the planner's own score.

  ~~Should the ranking refuse to order channels whose intervals overlap?~~
  **Answered, and the answer was yes**: `metrics.candidate_channels` and
  `leader_separation` report the set of channels that cannot be told apart
  from the leader, and [`OUTCOME.md`](OUTCOME.md) measures what reporting the
  set instead of a winner costs (0.017 AUC).
* ~~**Threshold choice.**~~ **Answered by the ds003498 benchmark, not by the
  hand-annotated one this entry used to point at.** On the simulator 3–4 robust
  SDs beat the published 5 on F1; against the expert markings the optimum is
  below 5 in every arm too, and the two bands want operating points a factor of
  2.5 apart (ripples 2.0 SD, fast ripples 5.0 SD). §0 has the tables and
  `onset_hfo.outcome.BAND_THRESHOLD_SD` maps bands to them.
* **Is the cycle-count criterion earning its place?** §2's ablation says the
  spectral check does nearly all the work and the decision recorded there is
  *keep*, as a cheap guard against events too short to have a spectrum. What is
  still missing is that the ablation is **synthetic only** — the criterion has
  never been ablated against the expert markings on real data.
* **Does the bipolar montage help or hurt for ripple *rate* specifically?**
  Easy experiment, currently unmeasured: run the whole pipeline in referential
  and bipolar montages and compare rankings.
* ~~**How stable is the ranking across windows?**~~ **Promoted to item 10**,
  because the outcome study answered part of it by accident: the conclusion
  moved between a 60 s and a 300 s window. It was described here as "the
  cheapest experiment in the list and possibly the most informative", and that
  turned out to be right.
