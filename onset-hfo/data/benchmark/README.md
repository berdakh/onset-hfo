# `data/benchmark/`

`agreement_sweep.csv` — cohort-mean agreement with the expert HFO markings of
ds003498, swept over the detection threshold, for both detectors and both
bands. Twenty subjects, first 60 s of `run-01` each, scored only on the
channels the annotators reviewed.

| column | meaning |
|---|---|
| `band` | `ripple` (80–250 Hz) or `fast_ripple` (250–500 Hz) |
| `detector` | `rms` or `line_length` |
| `threshold_sd` | detection threshold, in robust SD of the channel's own feature |
| `precision`, `recall`, `f1` | event-level, matched within 20 ms on the same channel |
| `detections` | mean detections per subject per 60 s |
| `rank_rho` | Spearman correlation between our channel ranking and the experts' |
| `top5_overlap` | mean shared channels between the two top-5 lists, out of 5 |

Regenerate with `python -m onset_hfo.cli benchmark`, which writes to
`artifacts/` — **not tracked** — and takes about an hour after the first
fetch. This file is the committed extract, so the figures in
[`../../paper/`](../../paper/) and the tables in
[`../../docs/EVALUATION.md`](../../docs/EVALUATION.md) can be rebuilt with no
network and no re-analysis. It is a concatenation of three sweeps: the main
2.0–6.0 SD grid, an extension downwards for ripples (the first grid put its
optimum on the boundary, which is not an optimum), and an extension upwards to
10 SD for fast ripples.

Read `precision` as **agreement**, never accuracy: the reference is the
validated output of the source study's Morphology detector, not a census of
every oscillation, so an event we find that it never proposed counts against
us whether or not it is real. See
[`../../docs/EVALUATION.md`](../../docs/EVALUATION.md) §0.

---

## `cohort.csv` — what the sweep was scored against

One row per subject: the recording the sweep used, and the expert markings in
it. 20 rows, 930 bytes.

| column | meaning |
|---|---|
| `duration_s`, `sfreq_hz` | the analysed slice — first 60 s of `run-01`, 2000 Hz |
| `n_channels`, `n_reviewed_channels` | recorded, and reviewed by the annotators |
| `n_expert_events` | marked events on the reviewed channels |
| `expert_ripples`, `expert_fast_ripples` | the same, split by band |

Totals across the cohort: **41,187** marked events — 35,620 ripples and 5,567
fast ripples — on 14 to 65 reviewed channels per subject. Per subject per 60 s
that is a mean of **1,781 ripples** and **278 fast ripples**.

**Why this file exists.** Every `recall` in `agreement_sweep.csv` is a fraction
of a number that lived only in untracked `artifacts/`, so any claim about *how
many* expert events there were had to be transcribed. It was transcribed three
times and two of them were wrong: `docs/EVALUATION.md` said 228, and
`onset_hfo/config.py` said ~70 — which is this detector's own mean detection
count at 5.0 SD, not the expert count at all. The demo's Detectors page reads both
files instead of restating either, and `tests/test_studytables.py` pins the numbers the
prose quotes against what is committed here.

---

## `four_detector_sweep.csv` — all four detectors, both bands

The same scoring as `agreement_sweep.csv`, extended to the two detectors added
later (`hilbert`, `short_time_energy`) and run over a common grid so the four
are comparable. 79 rows, same columns.

| | grid swept (robust SD) |
|---|---|
| all four, both bands | 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0 |
| `hilbert`, `line_length` — ripple | extended down: 0.5, 0.75 |
| `short_time_energy` — fast ripple | extended up: 10.0, 12.0, 15.0 |

**The extensions were not optional.** On the first pass, 4 of the 16
detector × band × criterion optima sat on an endpoint, which is the grid
running out rather than an optimum. Extending moved one materially:
short-time energy's fast-ripple rank agreement went from 0.570 at 8.0 SD to
**0.601 at 12.0 SD** (0.606 and 0.635 in the table as it stands). All sixteen are interior.

Regenerate offline, from the slices already in `artifacts/data/`:

```bash
python -m onset_hfo.cli benchmark --subjects sub-01 ... sub-20 \
  --detectors rms line_length hilbert short_time_energy \
  --bands ripple fast_ripple --thresholds 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0 8.0
```

Naming the subjects explicitly matters: only the subject *listing* needs the
network, so with the slices cached the whole sweep runs with
`ONSET_HFO_OFFLINE=1`. The 27 cells this run shares with `agreement_sweep.csv`
reproduce it **exactly**, which is how the offline path was checked against
the original networked one.

---

## `per_patient.csv.gz` — the scores the means were averaged from

One row per patient, band, detector and threshold: 20 patients × 2 bands ×
4 detectors × 17 thresholds (0.5–15 SD, the union of the grids above), 2,720
rows, 160 KB. Every row of `agreement_sweep.csv` and `four_detector_sweep.csv`
is the mean of the matching rows here, with undefined values left out,
rounded to three decimals; all 113 reproduce exactly, and
`tests/test_study_reproduction.py` keeps it so.

| column | meaning |
|---|---|
| `subject`, `run`, `band`, `detector`, `threshold_sd` | the cell |
| `n_truth` | expert events of the band on the reviewed channels |
| `n_detections`, `n_true_positives`, `n_false_positives`, `n_missed` | the detector's events and how they matched, within 20 ms on the same channel |
| `precision`, `recall`, `f1` | event-level; precision is empty when the detector found nothing |
| `n_channels` | reviewed channels the ranking is over |
| `spearman_rho`, `p_value` | rank correlation of each channel's detection count with its expert count; empty when it is undefined |
| `top5_overlap`, `top_k` | channels shared by the two top-k lists, and k (5, or fewer channels) |
| `expert_top`, `ours_top` | the two top-k lists |

**How many patients a mean is over.** Recall is defined for every patient:
each has expert events of both bands on the channels scored. Precision and
ρ are undefined in the 53 rows where the detector found nothing, so the means
of those two columns are over fewer patients at high thresholds, while F1 (0
there) is over all 20. One quoted figure is affected: the RMS detector's
fast-ripple ρ 0.643 and precision 0.761 at 5.0 SD are over 19 patients,
because it finds nothing on sub-10's reviewed channels at that threshold.

**Five patients' markings use other electrode names than their recordings.**
In sub-02, sub-05, sub-06, sub-07 and sub-08 the recording names the
hippocampal and entorhinal electrodes AH, EC and PH, while the archive's
markings (and the resection sheet, `clinical_ch_sheet_zurich.xlsx`) use
another scheme, and which letter is which differs between patients. Through
0.3.24 nothing mapped one onto the other, so those patients were scored on
the 6 of their 24 reviewed channels whose names matched — and in four of
them those six carried another electrode's markings — and 2,697 of the 41,187
marked events were never scored. The pairing is now read from the signals
and applied when the markings are loaded (`onset_hfo/archive_electrodes.csv`,
which holds the evidence), and these five patients' rows, and every mean, were
re-run with it. All 41,187 events are now scored. `cohort.csv`'s
`n_reviewed_channels` counts the channels scored.

**Re-run with the peak-contrast check.** Validation now also requires a peak
inside the band (`ValidationConfig.min_peak_contrast_db`, 6 dB;
[`../../docs/EVALUATION.md`](../../docs/EVALUATION.md) §2), which rejects some
of the events the older checks accepted, for all four detectors and in both
bands: 5.6% of the RMS detector's ripple detections at 2.0 SD and 0.9% of its
fast-ripple detections at 5.0 SD. This file and the two tables of means were re-run with it
by `scripts/rerun_benchmark_per_patient.py`. Before the re-run, that script
with the check off (`--check-off sub-01`) reproduced the previous rows of the
one patient it was tried on, all 136, to within 1e-16; the other nineteen
were not compared that way. The largest change in any
cohort mean, over both tables:

| `precision` | `recall` | `f1` | `rank_rho` | `top5_overlap` | `detections` |
|---|---|---|---|---|---|
| 0.014 | 0.026 | 0.020 | 0.019 | 0.15 | 78.1 |

Each of these is in the ripple band. In the fast-ripple band no mean moved
by more than 0.010 in precision, 0.004 in ρ, 0.003 in F1, 0.002 in recall,
0.05 in top-5 overlap or one detection per subject. No detector's best
threshold by ρ moved in either band. By F1 one did, to a tie: on the
four-detector grid RMS in the ripple band now scores 0.478 at both 1.5 and
2.0 SD, where 2.0 SD was ahead (0.494 against 0.489). The rows where the
detector found nothing went from 52 to 53. `min_peak_contrast_db=0`
reproduces the tables as they were.

Regenerated with `onset_hfo.benchmark.benchmark_subject`, patient by patient,
from the 60 s slices; the per-patient tables of the original runs were in the
untracked `artifacts/` and were not kept. Before the five patients were
re-scored, this file reproduced every committed mean exactly, and the patient
descriptions it records matched `cohort.csv`; the means and `cohort.csv` were
then rebuilt from it. Read and checked step by step in
[`notebooks/09_benchmark_per_patient.ipynb`](../../notebooks/09_benchmark_per_patient.ipynb).

---

## `peak_contrast.csv` — what the peak-contrast check costs against the markings

One row per patient × band × detector × threshold × floor: 20 patients, the
benchmark's window (first 60 s of `run-01`, reviewed channels), both
detectors at the thresholds the project reports at, and the same candidates
validated with `ValidationConfig.min_peak_contrast_db` at 0 (off), 3, 6 (the
default, and what the sweeps and `per_patient.csv.gz` above are computed
with) and 9 dB.

| column | meaning |
|---|---|
| `floor_db` | the peak-contrast floor the candidates were validated with |
| `accepted` | events accepted in the window |
| `precision`, `recall`, `f1` | against the expert markings, matched within 20 ms |
| `rank_rho`, `top5_overlap` | channel ranking against the experts' |

`peak_contrast_events.csv.gz` holds every candidate the older checks accept:
its contrast in dB and whether an expert marking on the same channel overlaps
it. Regenerate both with `python scripts/run_peak_contrast_cost.py` (needs the
archive; about fifteen minutes after the first fetch). The reading is in
[`../../docs/EVALUATION.md`](../../docs/EVALUATION.md) §2.
