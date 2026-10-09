# The ictal onset study's tables

The output of

```bash
python -m onset_hfo.ictal_study artifacts/results/ictal_ds003029          # the study
python -m onset_hfo.ictal_study artifacts/results/ictal_ds003029_control 40  # the control
```

on OpenNeuro `ds003029`: the Epileptogenicity Index (`onset_hfo.ictal`) of
every bipolar channel, for up to three seizures per patient, scored against
the clinicians' seizure onset zone from the archive's own
`sourcedata/clinical_data_summary.xlsx`. The control is the same seizures and
the same method with the "onset" put 40 s before the real one, so that
everything analysed is before the seizure. Committed because rebuilding them
means fetching 70 s around each of about eighty seizures. They contain no
signal. The source data is CC0; cite the dataset and its paper if you
publish anything derived from them.

| file | what it is |
|---|---|
| `per_seizure.csv` | one row per seizure tried: the onset mark and its kind, whether it was analysed and why not, the AUC of the index and of the energy ratio alone, whether the top channel is in the zone |
| `per_channel.csv` | one row per channel per seizure: index, whether a change was detected, its time from the marked onset, the energy ratio after it, and whether the channel is in the zone |
| `per_patient.csv` | one row per patient: the median index per channel over their seizures, then the AUCs |
| `summary.json` | the medians with bootstrap intervals, the tests, by site and by outcome |
| `control_*` | the same four for the control |

What the numbers say, and what they cannot, is in
[`docs/ICTAL.md`](../../docs/ICTAL.md).
