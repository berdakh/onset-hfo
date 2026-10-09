# Limitations

Read this before quoting any number from this repository to anyone.

## What this is

An **AI-assisted research tool** for intracranial EEG, written to be readable
and checkable. It runs classical detectors and the Epileptogenicity Index on
public recordings or your own — from one minute to a patient's whole
monitoring, through a case workflow ending in a signed report. An assistant on
a local open-weight model can read the results and is prevented from inventing
them. Its methods are tested on public archives (`EVALUATION.md`,
`OUTCOME.md`, `ICTAL.md`, `TEMPLATE_MAP.md`); on anyone else's recordings, those
numbers are what to expect, not what is guaranteed.

## What it is not

* **Not a medical device.** Not certified, not validated, not suitable for any
  clinical decision.
* **Not a diagnosis, and not a seizure-onset-zone finder.** The ictal index
  ranks channels by how early and how strongly they change at a seizure a
  person marked; whether they are the onset zone is the clinical team's
  judgement (`ICTAL.md`: AUC 0.80 against clinicians' zones, top channel in the
  zone in 15 of 28 patients). High event rate is
  a measurement. Physiological ripples occur in healthy tissue — mesial
  temporal structures and occipital cortex particularly — and a channel with a
  high rate may simply be healthy tissue that ripples. On the real recording
  here, the top-ranked channels do **not** overlap the contacts the clinician
  named more than chance predicts (`EVALUATION.md` §6).
* **Not validated on patients.** The detectors are scored against expert
  HFO markings on 20 real subjects (`docs/EVALUATION.md` §0) and against
  post-surgical seizure outcome in the same cohort (`docs/OUTCOME.md`).
  Neither is clinical validation. The first measures *agreement with another
  detector's validated output*; the second is a 20-patient retrospective
  study on 60 seconds per patient, in which **our detector does not reach
  significance on the metric where the expert markings do**.
* **Not tuned.** Thresholds are the published defaults, checked for sanity on
  synthetic data and deliberately *not* fitted to it.

## Specific limitations, and what each one means

| Limitation | Consequence |
|---|---|
| ~~**Ictal data only.**~~ **Resolved.** `ds003498` provides interictal slow-wave sleep at 2000 Hz with expert markings, and the benchmark runs on it. | The ictal quickstart still measures "where the ripple band is loudest during this seizure", which is not the interictal HFO rate. Use the interictal dataset for anything resembling a clinical claim. |
| **The reference is a detector, not a census.** ds003498's markings are the original study's Morphology detector output after human validation. | Our precision against it is a floor, not a measurement: an event we find that it never proposed counts against us whether or not it is real. Reported as *agreement*, never as accuracy. |
| **Only reviewed channels can be scored.** 6 to 65 of ~43 possible channels per subject, because the study kept the three most mesial bipolar channels in temporal-lobe cases. | Scores describe the reviewed subset. A detector could behave differently on the channels nobody read. |
| **60 seconds per subject, one run each.** The recordings are five minutes and each subject has 10–39 runs. | Rates and rankings from one minute are noisy; the benchmark is a measurement of the method, not of the patients. |
| **The shipped threshold is not the measured optimum.** The default stays at the literature's 5.0 SD; ripples prefer 1.5–2.0 SD. | Anyone running the defaults on interictal ripple data gets recall 0.12 and near-chance channel ranking unless they pass `--threshold interictal-agreement`. |
| **The two bands need different thresholds, and only `outcome` applies them automatically.** Ripples rank best at 2.0 SD, fast ripples at 5.0 SD. | Running 2.0 SD in the fast-ripple band drops precision to 0.086 and flattens the outcome signal entirely (`docs/OUTCOME.md`). `run` and `benchmark` take whatever threshold you pass, for every band. |
| **The outcome study is 13 seizure-free against 7 recurrences.** The smallest effect these group sizes can detect at 80% power is AUC 0.85. | Every p above 0.05 in `docs/OUTCOME.md` means "underpowered", not "no effect". **Nothing survives Bonferroni**; one of 36 uncorrected comparisons reaches p = 0.034, which is fewer than the ~2 that 36 tests produce by chance. |
| **A single short window is not a measurement, but a whole run is.** Across five *runs* on different nights the per-patient answer holds for 18/20 patients (experts) and 16/20 (ours); across five *minutes* of one run, 9/20 and 16/20. | Analyse whole runs, and pool a patient's runs where they exist. Never quote a result from a 60-second window. |
| **The outcome result depends on the analysis window** — now measured (`onset-hfo stability`, `docs/OUTCOME.md`). Across five disjoint 60 s windows of the same recordings the expert AUC spans 0.566–0.819 and its p-value 0.007–0.613. | The 0.007 that this project once published was the most favourable of five minutes. The growing-window curve *does* settle from 180 s, so the whole-run number is a settled estimate of one recording; what is unstable is any single short window. |
| **The busiest channel is often not the same channel.** Across those five windows the experts' top fast-ripple channel is identical in 7 of 20 patients and our detector's in 12 of 20; the inside/outside answer holds in 9 and 16. | **Addressed:** the outcome study now reports `candidates_resected` over the set of channels whose Poisson rate intervals overlap the leader's — the same rule `metrics.leader_separation` uses. On whole runs the data picks a single channel in only 9 of 20 patients (experts) and 8 of 20 (ours), median set size 2, worst case 24 tied channels of 37. Reporting the set costs 0.017 AUC. Never quote `top_channel_resected` without `n_candidates` beside it. |
| **Resected contacts that were never recorded cannot be scored.** In five temporal-lobe subjects only 4 of 16 listed resected contacts appear in the recording. | Those patients' "share inside the resection" describes a quarter of their resection. `recordings.csv` carries `rz_coverage` per subject. |
| ~~**The fast-ripple detector is event-starved in 60 s.**~~ **Resolved by using whole runs.** At 5.0 SD in 60 s, 5 of 20 subjects yielded ≤1 detection and one yielded none; over 300 s every subject produces detections. | It was a real defect and it is fixed — it was just not what was holding the detector back. The default window is now the whole run. |
| **1000 Hz sampling in `ds003029`.** Nyquist is 500 Hz. | Ripples only in that dataset; the pipeline refuses to analyse fast ripples there. `ds003498` is 2000 Hz, so fast ripples *are* analysed there (F1 0.30, ρ 0.59). |
| **One patient for the ictal demo; 20 for the benchmark.** | The quickstart's numbers describe one recording. The benchmark's describe 20 subjects from one centre, one scanner, one annotation protocol. |
| **Small counts.** A 60 s window turns a rate into a count. | 30 events/min *is* 30 events. Confidence intervals are printed for this reason; overlapping intervals mean "tied", not "ranked". |
| **Bipolar pairs by contact number.** On a grid the numbering wraps at the end of a row. | `G8-G9` may be two contacts on opposite edges of the grid. Depth electrodes and strips are fine. Fixing this needs electrode coordinates (roadmap). |
| **No electrode coordinates exist in either archive.** Neither `ds003498` nor `ds003029` ships an `electrodes.tsv` or a `coordsystem.json`; there is not one stereotactic position in either. | The desktop reviewer's 3D view is therefore a **schematic**: each shaft is drawn at the textbook location of the structure its electrode name claims, contacts in spatial order along it. It supports "these active contacts are neighbours on one right mesial temporal shaft, and they sit inside the resection" — which is a real finding. It supports **no** distance, trajectory, margin or volume. The panel says so permanently, every row of the layout carries whether its position was measured or inferred, and pointing the software at BIDS data that *does* have coordinates switches it to the real ones. |
| **No physiological-ripple discrimination.** | The pipeline cannot tell an epileptic ripple from a normal one. Nothing in it tries. The one available proxy — whether the ripple rides an interictal discharge — was screened against surgical outcome across all 20 patients and **does not localise the resection better than the merged rate** ([`EVALUATION.md`](EVALUATION.md) §3b). So the split is reported and never used as a label. |
| **Detector agreement is only moderate** (Jaccard ≈ 0.5 on real data). | Two reasonable detectors disagree about half the individual events. Any single-detector rate table is less certain than it looks. This is reported rather than hidden. |
| **Artifact rejection is not exhaustive.** | It catches filter ringing from large transients. It does not catch muscle artifact, stimulation, or the many creative ways real recordings go wrong. |
| **The agent's number check has a hole.** Bare integers ≤ 20 with no unit are allowed through. | A model could state "3 channels" when the tool said 4. Citations and unit-carrying numbers are checked; this class is not. |
| **A small model refuses a lot.** | With Qwen2.5-1.5B many answers fail verification and the agent declines. That is the system working, but it is not a pleasant demo — use a 7B model. |
| **Absolute amplitudes are only as good as the file header.** The archive declares 1 nV per stored unit; the resulting background is ~120 µV RMS, plausible but high. | Microvolt values in a report are not calibrated measurements. Detection is unaffected — all thresholds are in robust SDs of the channel itself. |
| **No security model.** No authentication, no audit log, no protection of the results directory. | Anyone who can write to `artifacts/results/` controls what the agent believes. Do not expose this to untrusted users. |
| **The archive carries no clinical history.** No seizure semiology, imaging report, scalp EEG, neuropsychology, medication or surgical history for any patient. | The reviewer's Patient panel shows those fields *empty*, with where a site would connect its own source. They are deliberately not filled with an example: an invented history in a clinical tool is indistinguishable from a real one. Age, sex and handedness are in the archive and are deliberately not published here either — see `data/outcome/README.md`. |
| **No privacy controls.** | The public data is already de-identified. If you point this at your own recordings, de-identification, ethics approval and data governance are entirely your responsibility — see `DATA.md`. |

## The evidence for HFOs themselves is contested

This software detects HFOs well enough to agree moderately with experts. That
says nothing about whether HFOs should guide surgery, and the best available
evidence is not encouraging: in [the HFO Trial](https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(22)00311-8/fulltext)
(Lancet Neurology, 2022), 78 patients randomised to intraoperative
HFO-guided versus spike-guided tailoring, seizure freedom at one year was
**67% with HFO guidance against 90% with spikes** — non-inferiority not met.

The retrospective evidence is friendlier — [Fedele et al. (2017)](https://www.nature.com/articles/s41598-017-13064-1),
whose markings this repository is scored against, found that resecting HFO-
generating tissue predicted outcome in individual patients, and `docs/OUTCOME.md`
reproduces that finding from their expert markings in 60 seconds per patient —
but a randomised trial outranks a retrospective series, including this one.
Reproducing a retrospective result is evidence that our *analysis* is sound,
not evidence that the *clinical claim* is — and on the full recordings this
repository does not reproduce it to significance either.

What this means for anyone presenting or building on this work: the
contribution here is **measurement quality and provenance**, not a claim that
HFO rate identifies epileptogenic tissue. The pipeline detects interictal
discharges as well, and the trial above suggests spikes deserve at least
equal billing.

## Things that would change the conclusions

If you are deciding whether to build on this, these are the questions that
matter most, in order:

1. **Does the ranking hold on interictal data?** Until that is checked,
   nothing here can be compared to the HFO literature.
2. **Does it hold across windows and across patients?** A ranking that moves
   when you shift the window by a minute is not a finding.
3. ~~**Does it relate to surgical outcome?**~~ **Asked, and the answer is
   instructive** (`docs/OUTCOME.md`). On whole runs, the busiest
   fast-ripple channel was inside the resection in 11/13 seizure-free
   patients and 3/7 recurrences for the expert markings (AUC 0.71, p = 0.12)
   and 10/13 vs 3/7 for our detector (0.67, p = 0.17). The direction is the
   published one; the cohort cannot establish it.
4. ~~**Is the channel ranking stable?**~~ **Answered** (`docs/OUTCOME.md`).
   Within a recording it is not: the same analysis gave AUC 0.82 on the first
   minute and 0.71 on the whole run. Across *recordings* it is: the
   per-patient answer holds over five different nights for 18/20 patients,
   and pooling a patient's runs cuts the median candidate set from 2 channels
   to 1 and the worst case from 24 to 5. So the quantity is a property of the
   patient, not of the session — a minute is simply too short a unit.
5. **Does any of it survive a second cohort?** Nothing here has been shown
   to generalise beyond one centre, one annotation protocol and one
   surgical team.

`ROADMAP.md` turns these into work items.

## A note on the agent

The guards in `onset_agent/guard.py` make it hard for the model to state a
number that no tool produced, and easy for it to refuse. They do **not** make
its answers correct: the model can still emphasise the wrong thing, miss a
caveat, or phrase a true number misleadingly. The report is the authoritative
artefact. The agent is a convenience for reading it.
