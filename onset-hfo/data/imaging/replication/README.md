# The registration replication's tables

The output of
`python -m onset_hfo.imaging_replication artifacts/results/imaging_replication`
on three OpenNeuro archives from other hospitals than the one the warp's fade
was chosen on:
- `ds004473` (OHSU, sEEG);
- `ds004696` (Mayo Clinic, sEEG);
- `ds005574` (NYU, ECoG).

The study registers each patient's T1 with the Map step's own code and the
fade unchanged (10 mm). It carries their contacts to MNI the same four ways as
`data/imaging/`:
- `registered`;
- `warped`;
- `blended`;
- `identity`.

The truth for a contact is the structure the patient's own FreeSurfer
segmentation gives it.

The tables contain no images.

| file | what it is |
|---|---|
| `summary.json` | the totals `docs/IMAGING.md` quotes, per archive and pooled, with the patients the frame check excluded and the split by the registration check |
| `per_patient.csv` | each patient's registration check, the median distance to the authors' own MNI positions each way, the share of contacts near the template brain each way, and the hippocampus and amygdala contacts that agree each way |
| `per_contact.csv.gz` | every contact's depth relative to the template brain and distance to the authors' MNI position each way; for the deep contacts, the patient's own structure and the atlas label each way |
