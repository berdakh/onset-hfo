# The patient's own MRI: does registering it put the contacts in the right place?

The Map step places contacts on a template brain. Without the patient's imaging
those positions are approximate: imported in MNI152 or fsaverage, or planned as
straight shafts (`docs/TEMPLATE_MAP.md`). When the patient's T1-weighted MRI is
to hand, the case can use it (`onset_hfo/case/imaging.py`):

1. **Add the patient's MRI…** copies the T1 into the case
   (`sub-*/ses-implant01/anat/`) and registers it to the MNI152 template with
   MNE's volume registration (dipy): translation, then rigid, then affine.
   - The template is MNI152NLin2009cAsym at 2 mm *with its skull*, from
     TemplateFlow, fetched once (1.8 MB). It is the same space as the atlas.
   - The transform and a check are kept in
     `derivatives/onset/imaging/t1_to_mni.json`.
2. **Import contacts in the MRI's space…** takes positions in the patient's
   own scanner or ACPC millimetres, as planning or localisation software
   exports them.
   - They are kept as they are (BIDS `space-T1w`).
   - They are also carried to MNI by the registration, marked `source = patient`.
3. **Place contacts on a CT…** is for when there is a CT with the electrodes
   in it but no positions yet.
   - The CT is rigidly aligned to the T1 and opened in MNE's iEEG contact
     locator (`mne-gui-addons`, the `imaging` extra).
   - The contacts placed there come back in the patient's scanner millimetres.
4. **Show on the MRI** draws the contacts on the patient's own T1 in three
   planes. On their own MRI the positions are as good as the localisation.
   It is only the step to MNI (and to the atlas names) that is approximate.

Every note, caption, the 3D view and the report say whether a position is
the patient's (registered) or the template's.

## What was checked

**Synthetic, in the tests** (`tests/test_case_imaging.py`):

- A head that is the template turned 8° and shifted 12 mm registers back
  within 3 mm.
- A registration 25 mm off scores far lower on the check.
- Contacts round-trip through the case's files.
- The conversion from the locator's surface RAS to scanner millimetres matches
  MNE's own in three image orientations.
- A contact placed in the real locator, opened on a synthetic CT under a
  virtual display, comes back within 0.01 mm of where it was.

**On real patients: OpenNeuro ds003688** (Berezutskaya et al. 2022, UMC
Utrecht). The archive gives each patient's T1 and their contacts in that T1's
space (ACPC millimetres, `IntendedFor` the T1). There are no MNI positions to
compare with, so the checks are anatomical.

- **Named medial temporal contacts.** In this archive's naming, a shaft
  called `AR`/`AL` is aimed at the amygdala and `AHR`/`AHL`/`PHR`/`PHL` at the
  hippocampus, on the side the last letter says. That reading of the names is
  an assumption, not a field of the dataset. The deepest two contacts of each
  such shaft should be labelled that structure on that side (in it, or within
  the atlas's 5 mm "near").
- **The brain.** The share of each patient's contacts inside the template
  brain or within 5 mm of it. Surface grids sit on the cortex, a few
  millimetres out.
- **The registration's own check**: the intensity correlation of the
  registered head with the template, inside the template's brain.

Each is compared with the shortcut of reading the patient's ACPC millimetres
as if they were MNI. Run it with
`python -m onset_hfo.imaging_study artifacts/results/imaging_ds003688`. It
downloads about 30 MB of MRI per patient and deletes each after use. The
tables are in [`data/imaging/`](../data/imaging).

## Results

51 patients, 4,665 contacts, none failed; a median of 21 s of registration
each.

| | registered | ACPC read as MNI |
|---|---|---|
| Targeted contacts in the intended structure (5 patients, 34 contacts) | **23 of 34 (68%)** | 2 of 34 (6%) |
| Contacts within 5 mm of the template brain | 87% | 77% |
| Per patient, median share near the brain | 98% | 86% |
| Patients with ≥ 90% of contacts near the brain | 34 of 51 | 23 of 51 |

The targeted contacts, patient by patient:

| patient | registered | ACPC as MNI | check |
|---|---|---|---|
| sub-01 | 8/8 | 2/8 | 0.25 |
| sub-09 | 4/6 | 0/6 | 0.51 |
| sub-21 | 2/6 | 0/6 | 0.32 |
| sub-23 | 5/8 | 0/8 | 0.42 |
| sub-42 | 4/6 | 0/6 | 0.43 |

Where registered contacts missed, they mostly landed next door: 10 of the 11
misses were in the parahippocampal gyrus, beside the hippocampus and amygdala. The
amygdala is the hardest: 2 of 6 on the right, 3 of 4 on the left. Read as MNI
without registration, the same contacts land in orbitofrontal cortex, the
temporal pole or the putamen. The two spaces are offset by centimetres.

**The check.** The correlation ran from 0.09 to 0.60, median 0.47.

- Of the 51 heads, 12 had fewer than 80% of their contacts near the template
  brain once registered.
- Below 0.25 (`CHECK_WARN`), the check flagged 6 of those 12, and 2 of the 39
  that were fine. Below 0.20 it flagged 4, all among the 12.
- Under 0.25, the Map step says so and asks for the contacts to be checked on
  the MRI.
- It does not catch everything. One patient correlated at 0.47 with only 10%
  of their contacts near the brain.

Scaling ran from 0.78 to 1.70 along an axis. Three of the five largest belong
to children aged 5 to 9, whose heads the affine stretched the most. Two of
those three are among the poor registrations.

## What this does and does not support

- **Registering the patient's MRI is far better than reading their
  coordinates as MNI.** For targeted contacts it was 68% against 6%. Do not
  import native coordinates as MNI.
- **It is an affine, not a nonlinear registration.** It aligns the head as a
  whole and does not bend one brain into another. A third of the targeted
  contacts landed a structure away. The atlas names stay "probably" for
  patient contacts as for template ones.
- **Look at it.** The correlation catches about half of the poor
  registrations. **Show on the MRI** shows where the contacts really are, and
  the 3D view shows where they landed on the template.
- Five patients and 34 contacts is what the archive's names allow for the
  anatomical check. That is enough to see the difference from the shortcut,
  not to quote a precise hit rate.
- The CT locator path was tested on a synthetic CT, not on a patient's.
  ds003688 has no CTs.
