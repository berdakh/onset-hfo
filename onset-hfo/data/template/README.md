# The template planner study's tables

The output of

```bash
python -m onset_hfo.template_study artifacts/results/template_ds004100
```

on OpenNeuro `ds004100` (HUP iEEG, CC0). It covers 38 SEEG patients whose contacts were
localised on their own imaging and published in fsaverage space, moved here
to MNI152. Each depth shaft is planned as a straight line from its deepest
contact towards its last, at four choices of spacing, and compared with the
real contacts: the distance, and whether the atlas gives both the same
probable structure.

| file | what it is |
|---|---|
| `per_shaft.csv` | one row per shaft: contacts, its own spacing on the template, its furthest contact from a straight line |
| `per_contact.csv` | one row per contact: distance from the tip and from the line, its atlas label, and per plan the error and whether the label matches |
| `summary.json` | the medians, percentiles and shares in [`docs/TEMPLATE_MAP.md`](../../docs/TEMPLATE_MAP.md) |

The electrode files themselves are not copied here; the command fetches them
(38 small TSVs).
