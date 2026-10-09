# The patient's own MRI: does registering it put the contacts in the right place?

The Map step places contacts on a template brain. Without the patient's imaging
those positions are approximate: imported in MNI152 or fsaverage, or planned as
straight shafts (`docs/TEMPLATE_MAP.md`). When the patient's T1-weighted MRI is
to hand, the case can use it (`onset_hfo/case/imaging.py`):

1. **Add the patient's MRI…** copies the T1 into the case
   (`sub-*/ses-implant01/anat/`) and registers it to the MNI152 template with
   MNE's volume registration (dipy). The steps are translation, rigid and
   affine, then a non-linear warp (symmetric diffeomorphic registration, 3 mm).
   - The template is MNI152NLin2009cAsym at 2 mm *with its skull*, from
     TemplateFlow, fetched once (1.8 MB). It is the same space as the atlas.
   - The affine and a check are kept in
     `derivatives/onset/imaging/t1_to_mni.json`.
   - The warp is kept in `t1_to_mni_warp.npz` beside it (about 6.5 MB, plain
     arrays, never pickled).
2. **Import contacts in the MRI's space…** takes positions in the patient's
   own scanner or ACPC millimetres, as planning or localisation software
   exports them.
   - They are kept as they are (BIDS `space-T1w`).
   - They are also carried to MNI by the registration, marked `source = patient`.
   - The affine places every contact.
   - The warp then moves each one by a share that grows with depth:
     - outside the template brain, by none of it;
     - in the first 10 mm inside, by a share proportional to the depth;
     - deeper, by all of it.

     Results below show why.
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
- A head whose tissue near one point was pushed 6 mm by a smooth bump comes
  back:
  - affine alone: 5.5–5.9 mm off;
  - with the warp: 0.7–1.3 mm off on the TemplateFlow template, under 2 mm on
    average on the bundled one the tests use.
- The warp's direction is easy to get backwards, and the test pins it:
  - MNE's `apply_volume_registration_points` leaves points where they were;
  - dipy's forward point transform doubles the error;
  - the inverse transform, in world millimetres, is right.
- Outside the brain the fade leaves a contact alone. At the brain's edge it
  moves it by its share, and deep inside it moves it by the whole warp.
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

Each is reported four ways:
- **affine:** the affine alone;
- **warp, in full:** the affine and then the whole warp;
- **warp, faded:** the warp faded in from the brain's edge, as the Map step
  applies it;
- **ACPC as MNI:** the shortcut of reading the patient's ACPC millimetres as if
  they were MNI. Run it with
`python -m onset_hfo.imaging_study artifacts/results/imaging_ds003688`. It
downloads about 30 MB of MRI per patient and deletes each after use. The
tables are in [`data/imaging/`](../data/imaging).

## Results

51 patients, 4,665 contacts, none failed. Each took a median of 26 s,
download included. The warp moved contacts a median 3.5 mm from where the
affine put them.

| | affine | warp, in full | **warp, faded** | ACPC as MNI |
|---|---|---|---|---|
| Targeted contacts in the intended structure (5 patients, 34 contacts) | 23 of 34 | 30 of 34 | **29 of 34** | 2 of 34 |
| Contacts within 5 mm of the template brain | 87% | 84% | **87%** | 77% |
| Per patient, median share near the brain | 98% | 92% | **98%** | 86% |
| Patients with ≥ 90% of contacts near the brain | 34 of 51 | 29 of 51 | **34 of 51** | 23 of 51 |
| Intensity correlation with the template (median) | 0.47 | 0.68 | | |

The targeted contacts, patient by patient:

| patient | affine | warp, in full | warp, faded | ACPC as MNI | check |
|---|---|---|---|---|---|
| sub-01 | 8/8 | 8/8 | 8/8 | 2/8 | 0.25 |
| sub-09 | 4/6 | 5/6 | 5/6 | 0/6 | 0.51 |
| sub-21 | 2/6 | 4/6 | 4/6 | 0/6 | 0.32 |
| sub-23 | 5/8 | 8/8 | 8/8 | 0/8 | 0.42 |
| sub-42 | 4/6 | 5/6 | 4/6 | 0/6 | 0.43 |

**The warp helps deep contacts.** With the affine alone, 10 of the 11 misses
were next door, in the parahippocampal gyrus. The warp moves them into the
hippocampus:
- hippocampus: 24 of 24 with the faded warp, against 18 of 24 with the affine;
- amygdala: still the hardest, 5 of 10;
- all five misses that remain are amygdala contacts labelled the adjacent
  anterior parahippocampal gyrus.

Read as MNI without registration, the same contacts land in orbitofrontal
cortex, the temporal pole or the putamen. The two spaces are offset by
centimetres.

**The warp in full hurts surface grids, so it is faded in.** Applied to every
contact, the warp left fewer of them near the template brain in 15 of 51
patients, by more than 5 percentage points. For example, sub-06 fell from
99% to 74%. At the brain's edge the warp follows the skull and the scalp,
not the cortex.

Faded in over the first 10 mm inside the brain, the warp:
- leaves every surface contact where the affine put it, with not one patient
  changing by more than 5 points;
- keeps 29 of the full warp's 30 deep contacts.

The 10 mm fade was chosen among five (0–10, 5–15, 5–20, 10–20 and 10–25 mm)
on this same data. The deeper fades kept only 25 of 34, because the
amygdala and hippocampus lie close to the brain's lower surface. So the gain
is measured on the data that picked the rule. The next section tests it on
patients from three other hospitals.

**The check** is computed after the affine. Its correlation ran from 0.09 to
0.60, median 0.47.

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

## On patients from other hospitals

The fade was then tested, unchanged, on three archives from other hospitals
(`onset_hfo/imaging_replication.py`; tables in
[`data/imaging/replication/`](../data/imaging/replication)). The endpoints
and the exclusion rule were written into the module before the run.

| archive | hospital | patients | contacts | the truth |
|---|---|---|---|---|
| ds004473 (Rockhill et al.) | OHSU | 8, sEEG | 978 | the patient's own FreeSurfer segmentation, looked up at each contact |
| ds004696 (Ojeda Valencia et al. 2023) | Mayo Clinic | 8, sEEG | 1,076 used | the label of the patient's own FreeSurfer segmentation, given per contact; and the authors' MNI positions |
| ds005574 (Zada et al.) | NYU | 9, ECoG grids | 1,330 | the authors' MNI positions |

Here the truth for a deep contact is the structure the patient's own
segmentation puts it in. That is stricter than ds003688's shaft names: it
covers every contact on a shaft, edges included, not only the deepest two.

**Primary, deep.** Of 154 contacts in 14 patients that the patient's own
segmentation puts in the hippocampus or amygdala, the atlas labelled the
right structure:

| | affine | warp, in full | **warp, faded** | read as MNI |
|---|---|---|---|---|
| all 154 | 87 (56%) | 101 (66%) | **101 (66%)** | 79 (51%) |
| hippocampus (99) | 55 | 69 | **68** | 47 |
| amygdala (55) | 32 | 32 | **33** | 32 |
| OHSU (91) | 41 | 49 | **49** | 48 |
| Mayo (63) | 46 | 52 | **52** | 31 |

**Primary, surface.** Within 5 mm of the template brain:

| | affine | warp, in full | **warp, faded** |
|---|---|---|---|
| all 3,384 contacts | 98.8% | 98.7% | **98.8%** |
| patients whose share fell more than 5 points from the affine's | | 1 of 23 | **0 of 23** |

The one was an NYU grid patient (sub-04): 98% of contacts near the brain
with the affine, 88% with the full warp, and 98% with the fade.

**Secondary.**
- **Every deep structure.** For all 194 deep contacts, the faded warp got 119
  (61%), the full warp 119, the affine 102 and reading as MNI 102.
- **The putamen** was the poorest: 10 of 32 with the faded warp. At Mayo it
  was 0 of 21.
- **The authors' own MNI positions.** Median distance to them, as a guide
  only, because each archive used another version of MNI152:

  | | affine | full warp | faded warp | read as MNI |
  |---|---|---|---|---|
  | all | 5.7 mm | 5.4 mm | 5.0 mm | 27.8 mm |
  | NYU | 4.6 mm | 4.9 mm | 4.3 mm | |
  | Mayo | 8.1 mm | 6.8 mm | 6.9 mm | |

**The registration check.** Five of the 14 sEEG patients scored below
`CHECK_WARN` (0.25), and the Map step would have asked for their contacts
to be checked on the MRI.

| | patients | contacts | affine | faded warp |
|---|---|---|---|---|
| flagged | 5 | 74 | 27 | 39 |
| not flagged | 9 | 80 | 60 (75%) | 62 (78%) |

- The faded warp missed 53 contacts.
- 25 of them landed in the pallidum, putamen or thalamus. Every one of those
  25 came from a flagged registration: there, the head was registered a
  structure or more off.
- On the patients the check passed, 16 of the 18 misses were next door:
  - 15 in the parahippocampal gyrus;
  - 1 amygdala contact in the hippocampus.
- The other 2 were in the temporal fusiform cortex.

**Excluded, and why.** Two of Mayo's eight patients were left out by the
frame rule, their affine positions a median 28.7 and 23.8 mm from the
authors' (sub-06 and sub-07).
- Their positions are in an ACPC-aligned copy of the T1 that the archive
  does not include.
- Read as FreeSurfer surface coordinates, sub-06's positions did not fit
  either (29.2 mm).

**Frames the check corrected before the numbers.**
- **OHSU.** The archive's segmentation is in its ACPC frame, so its ACPC
  positions are used, with FreeSurfer's copy of the T1.
- **NYU.** Its "native" positions are FreeSurfer surface coordinates.
- Both are described in the module. The first run read them otherwise, and
  the frame check excluded those patients.
- Each frame was set by the frame check alone: the share of contacts inside
  the patient's own segmented brain, or the distance to the authors' MNI
  positions. The first run's per-patient lines, near-brain shares
  included, had been seen by then.

**Reading as MNI.**
- For OHSU, the positions are true ACPC coordinates, with the origin at the
  anterior commissure. Read as MNI, they did as well as registering (48
  against 49), and stayed near the brain.
- ds003688's "ACPC" was centimetres off.
- So whether the shortcut works depends on what a site's "ACPC" really is,
  and the program still asks for the registration.

**What the three archives say about the fade.**
- **Deep contacts.** It keeps the full warp's gain: 101 against 101, and
  hippocampus 68 against 55 with the affine.
- **Surface grids.** It leaves them where the affine put them, the one thing
  the full warp got wrong.
- **The amygdala.** The warp did not help it here (32 → 33 of 55). At
  ds003688 it was 5 of 10 with the affine and with the fade.

The 10 mm fade stays as it is.

## What this does and does not support

- **Registering the patient's MRI is far better than reading their
  coordinates as MNI.** For targeted contacts it was 85% against 6% (68% with
  the affine alone). Do not import native coordinates as MNI.
- **The warp helps contacts deep in the brain, and only those.** Faded in from
  the brain's edge, it took the targeted contacts from 68% to 85% in their
  structure, without moving surface contacts off the brain. On 23 patients
  from three other hospitals, it took hippocampus and amygdala contacts from
  56% to 66% (75% to 78% where the registration check passed), and moved no
  patient's contacts off the brain. It is still a
  registration to a template, not a segmentation of this patient's brain. A
  contact can land a structure away (the amygdala most of all), so the atlas
  names stay "probably" for patient contacts as for template ones.
- **Look at it.** The correlation catches about half of the poor
  registrations. On the other hospitals' patients, every contact that landed
  a structure or more away (in the basal ganglia or thalamus) came from a
  registration the check flagged. **Show on the MRI** shows where the contacts really are, and
  the 3D view shows where they landed on the template.
- Five patients and 34 contacts is what ds003688's names allow for the
  anatomical check. The other archives add 14 patients and 154 contacts
  labelled by each patient's own segmentation. Together that is enough to
  see the warp's gain and the fade's safety, but still not enough to quote a
  precise hit rate for a structure.
- The CT locator path was tested on a synthetic CT, not on a patient's.
  ds003688 has no CTs.
