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
| `robustness_ablation.csv` | band × patient × window × rule × re-test | 800 |
| `subpopulation_groups.csv` | band × metric × population | 18 |
| `robustness_groups.csv` | band × metric × rule × re-test | 32 |

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

**`robustness_ablation.csv`** — the ablation of
`onset_agent.planner.rank_channels`'s robustness multiplier
([`EVALUATION.md`](../../docs/EVALUATION.md) §6c). One row per
`band` × `subject` × `window` × `rule` × `stricter_x`, where `rule` is `plain`
(score = survey rate, what rungs S0/S1 produce) or `multiplied`
(score = rate × robustness, what S2/S3 produce), and `stricter_x` is the re-test
threshold as a multiple of the survey threshold.

Two columns exist because the multiplied score is **not an event count**:

| column | meaning |
|---|---|
| `n_events_survey` | the real integer count from the survey pass, identical across rules for a window |
| `score_mass`, `score_mass_{resected,partial,spared}` | the summed *score*, which for `multiplied` is not a count |

`retest_silent == 1` marks a window where the stricter pass found no events on
any channel: 84 of the 600 multiplied windows, 82 of them fast ripple. The rule
no longer fires on those (`planner.rank_channels`, `unmeasured_at`), because a
robustness of 0 from "nothing was detected here" means *unmeasured*, not
*refuted*. `n_events_stricter` is the count it is derived from.

`score_mass == 0` now marks only the 48 windows where the **survey** found
nothing either — a fast-ripple minute with nothing in it, where zero is the
right answer rather than a destroyed ranking. Those are the rows whose metrics
are NaN. Before the fix there were 84, conflating the 48 genuinely-empty
windows with 36 in which a working ranking was zeroed.

`candidates_resected` and its companions are populated for `plain` only:
`candidate_channels` needs integer counts for its Poisson intervals.
`tied_set_argmax_resected` is the comparable substitute — the tied set comes
from the shared survey counts, so both rules are judged on the same channels.

**`subpopulation_groups.csv`** and **`robustness_groups.csv`** — the reduced
group tables for the two screens above: AUC, bootstrap interval, exact
permutation `p` and a Bonferroni column, with the realised `n_SF`/`n_rec` per
arm. They exist for the same reason `../stability/` holds group tables beside
the per-subject ones: reducing the per-window rows takes a bootstrap and an
exact permutation, which is seconds of work and not something the Streamlit app
should do on a page load. Regenerate either by re-running its script and writing
`analyse(rows)` to the file; write them at full precision, since rounding to
four decimals on the way out moved two cells when it was first tried.

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

The last two files have their own entry points, which need no network at all
once the 60 s slices are in `artifacts/data/`:

```bash
ONSET_HFO_OFFLINE=1 python scripts/run_subpopulation_outcome.py   # ~40 min
ONSET_HFO_OFFLINE=1 python scripts/run_robustness_ablation.py     # ~50 min
```

Naming the subjects explicitly is what makes that offline: only the subject
*listing* touches the archive. Pass `--from-csv data/outcome/subpopulation_screen.csv`
to re-print the §3b tables from this file with no re-analysis; the ablation
script takes the same flag against `robustness_ablation.csv` for §6c.

The source data is CC0. Anything published from it should cite the dataset and
[Fedele et al. 2017](https://www.nature.com/articles/s41598-017-13064-1); the
citation is in `provenance.json` of every analysis.
