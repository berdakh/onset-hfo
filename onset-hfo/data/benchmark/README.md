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
fast ripples — on 6 to 65 reviewed channels per subject. Per subject per 60 s
that is a mean of **1,781 ripples** and **278 fast ripples**.

**Why this file exists.** Every `recall` in `agreement_sweep.csv` is a fraction
of a number that lived only in untracked `artifacts/`, so any claim about *how
many* expert events there were had to be transcribed. It was transcribed three
times and two of them were wrong: `docs/EVALUATION.md` said 228, and
`onset_hfo/config.py` said ~70 — which is this detector's own mean detection
count at 5.0 SD, not the expert count at all. The Detectors page now reads both
files instead of restating either, and `tests/test_app.py` pins the numbers the
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
**0.601 at 12.0 SD**. All sixteen are now interior.

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
