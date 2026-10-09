# The HUP replication's tables

The output of

```bash
python -m onset_hfo.hup ictal artifacts/results/hup_ictal              # the ictal index
python -m onset_hfo.hup ictal artifacts/results/hup_ictal_control40 40 # its control
python -m onset_hfo.hup interictal artifacts/results/hup_interictal    # interictal ripples
```

on OpenNeuro `ds004100` (the HUP iEEG archive): the ictal onset study and the
interictal outcome study, unchanged, on a second archive.

- The onset zone and the resected contacts come from the archive's own
  `channels.tsv`.
- Seizure onsets are its electrographic onset marks.
- Only the EDF records needed are fetched.
- The tables contain no signal. The source data is CC0; cite the dataset and
  its paper if you publish anything derived from them.

| file | what it is |
|---|---|
| `ictal/summary.json`, `ictal/per_patient.csv`, `ictal/per_seizure.csv`, `ictal/per_channel.csv` | the index at the marked onset, up to three seizures per patient |
| `ictal/control_*` | the same seizures with the "onset" put 40 s earlier |
| `interictal/summary.json`, `interictal/per_patient.csv`, `interictal/per_channel.csv` | RMS ripple and fast-ripple rates on the five-minute interictal recordings, against the onset zone, the resection and outcome; a recording sampled too slowly for a band is a row saying so |

Write-ups: `docs/ICTAL.md` and `docs/OUTCOME.md`, section "A second archive".
