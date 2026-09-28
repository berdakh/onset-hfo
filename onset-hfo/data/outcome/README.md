# `data/outcome/`

The **per-subject** tables of the outcome study on OpenNeuro
[`ds003498`](https://openneuro.org/datasets/ds003498) — 20 patients who had
epilepsy surgery, with expert HFO markings, the contacts their surgeon
removed, and whether their seizures stopped.

`data/stability/` already carried the *group* tables: one row per
band × scope × source × metric, which is what a paper reports. These four
files are the rows underneath them, committed for the same reason — so the
[Patients screen](../../app/pages/6_Patients.py) and anyone checking a number
can work from a fresh clone with no 700 MB download and no 90-minute rerun.

| file | grain | rows |
|---|---|---|
| `participants.csv` | one patient | 20 |
| `recordings.csv` | one patient's analysed run | 20 |
| `subjects.csv` | patient × band × scope × source | 160 |
| `channels.csv.gz` | patient × band × channel | 1,880 |
| `subpopulation_screen.csv` | band × patient × window × population | 600 |

## What each file carries

**`participants.csv`** — the archive's own `participants.tsv`, reduced to the
columns this project uses. `epilepsy` is `TLE` or `ETE` (temporal or
extratemporal), `ilae` is the ILAE outcome class, `outcome` is `S`
(seizure-free, ILAE 1–2) or `F` (recurrence), `nights` is how many nights the
archive holds for that patient, `lesion` is 1 when imaging found one.

**`recordings.csv`** — what the analysed run actually contained.
`n_reviewed` is the channels the annotators reviewed, which is the only set
anything is scored on. `rz_coverage` is the fraction of the contacts the
clinical sheet lists as resected that appear in the recording at all —
**the most important column in this directory**, because in five patients it
is 0.25, so their "share inside the resection" describes a quarter of their
resection. `missing` names the contacts that fell out.

**`subjects.csv`** — every metric the study computes, for each patient, in
each band (`ripple`, `fast_ripple`), each scope (`reviewed` channels only, or
`all`), from each source (`expert` markings, or our `rms` detector).
`top_channel_resected` is 1 when the busiest channel was removed;
`n_candidates` is how many channels were statistically tied for busiest, and
`candidates_resected` is the share of that tied set that was removed — the
honest version of the same question. The pre-specified arm is
`band=fast_ripple, scope=reviewed`.

**`subpopulation_screen.csv`** — the same metrics again, but with the events
split into the two sub-populations of
[`EVALUATION.md`](../../docs/EVALUATION.md) §3b, to answer whether either one
localises the resection better than the merged rate. One row per
`band` × `subject` × `window` × `population`, where `population` is `merged`,
`spike_coupled` (the event rode an interictal discharge) or `independent`.

This one is **not** the per-subject grain of the three files above: it is five
60 s windows per patient rather than the whole 300 s run, because the whole run
is not cached and fetching it needs the archive. The analysis pools the windows
by subject mean. `n_events` is the population's event count in that window and
is the column to read first — in the fast ripple band its median is **1**, which
is why the screen's answer comes from the ripple band and is reported as a
secondary analysis.

**The answer is no**, in the ripple band where the populations are large enough
to measure: every sub-population AUC is at or below the merged one. §3b has the
tables and why the flattering fast-ripple numbers in this file are artifacts of
an argmax over a near-empty set.

**`channels.csv.gz`** — one row per channel, with `zone` = `resected` (both
contacts removed), `partial` (one) or `spared` (neither), plus the event count
from each source. This is the row-level evidence under any per-patient claim;
a `partial` channel is the one to look at hardest before believing either
answer.

## What was deliberately left out

The archive's `participants.tsv` also carries **age, sex and handedness**.
They are populated for 19 of the 20 patients and no analysis in this project
uses any of them, so they are not committed. Publishing three demographic
fields per patient alongside pathology, surgical extent and outcome narrows a
cohort of twenty considerably, and nothing here needs them. If an analysis
ever does — an age-stratified comparison, say — take them from the archive at
that point and say so in the method.

No signal is in this directory. Every column is a count, a share or a label
the pipeline already writes into its own report.

## Regenerating

```bash
python -m onset_hfo.cli outcome        # whole 300 s runs, both bands
```

writes the full study to `artifacts/results/outcome_ds003498/` — **not
tracked** — from which the first four files are the committed extract. The group
tables in `../stability/` come from the same run.

`subpopulation_screen.csv` has its own entry point, which needs no network at
all once the 60 s slices are in `artifacts/data/`:

```bash
ONSET_HFO_OFFLINE=1 python scripts/run_subpopulation_outcome.py   # ~40 min
```

Naming the subjects explicitly is what makes that offline: only the subject
*listing* touches the archive. Pass `--from-csv data/outcome/subpopulation_screen.csv`
to re-print the §3b tables from this file with no re-analysis.

The source data is CC0. Anything published from it should cite the dataset and
[Fedele et al. 2017](https://www.nature.com/articles/s41598-017-13064-1); the
citation is in `provenance.json` of every analysis.
