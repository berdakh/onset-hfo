# The registration study's tables

The output of `python -m onset_hfo.imaging_study artifacts/results/imaging_ds003688`
on OpenNeuro `ds003688`.

- Each patient's T1 is registered to MNI152 by the Map step's own code
  (`onset_hfo.case.imaging.register_image`).
- Their contacts, given in that T1's space, are carried to MNI.
- The result is compared with reading the contacts as MNI directly.
- The tables contain no images.

| file | what it is |
|---|---|
| `summary.json` | the totals `docs/IMAGING.md` quotes |
| `per_patient.csv` | the registration check (intensity correlation), scaling, the share of contacts near the template brain both ways, and the targeted contacts that agree |
| `per_contact.csv.gz` | every contact's position, depth relative to the template brain and, for named amygdala and hippocampus shafts, the atlas label both ways |
