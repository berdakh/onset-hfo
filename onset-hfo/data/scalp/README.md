# The scalp EEG comparison's table

The output of

```bash
python scripts/run_scalp_comparison.py --subjects 01-10
```

on OpenNeuro `ds003555` (sleep scalp EEG from 30 children with epilepsy,
Cserpan et al.): this project's RMS ripple detector against the dataset's own
published detector, on the first 5-minute N3 interval of subjects 01–10.

- The intervals, their 52 nearest-neighbour bipolar channels and the
  published detections are the dataset's derivatives.
- `theirs_passed` counts the detections that passed the published artefact
  rejection; `theirs_all` counts them all.
- `rho_passed` and `rho_all` are Spearman's rho across the 52 channels between
  our count and theirs.
- The table contains no signal. The source data is CC0; cite the dataset and
  its paper if you publish anything derived from it.

| file | what it is |
|---|---|
| `scalp_comparison.csv` | one row per subject |
| `concurrency_sweep.csv` | the same ten subjects under each concurrency limit tried (`fraction`, 0 = no limit): rejecting events seen on more than that share of the other channels at once |

Write-up: `docs/MODALITIES.md`.
