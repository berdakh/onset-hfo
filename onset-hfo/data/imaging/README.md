# The registration study's tables

The output of `python -m onset_hfo.imaging_study artifacts/results/imaging_ds003688`
on OpenNeuro `ds003688`. The study registers each patient's T1 to MNI152
with the Map step's own code (`onset_hfo.case.imaging.register_image_warped`)
and carries their contacts, given in that T1's space, to MNI four ways:

- `registered`: the affine alone;
- `warped`: the affine, then the non-linear warp in full;
- `blended`: the warp faded in from the template brain's edge, as the Map step
  applies it;
- `identity`: the contacts read as MNI directly.

The tables contain no images.

| file | what it is |
|---|---|
| `summary.json` | the totals `docs/IMAGING.md` quotes |
| `per_patient.csv` | the registration check after the affine and after the warp, scaling, how far the warp moved contacts, the share of contacts near the template brain each way, and the targeted contacts that agree each way |
| `per_contact.csv.gz` | every contact's position and depth relative to the template brain each way, and, for named amygdala and hippocampus shafts, the atlas label each way |
