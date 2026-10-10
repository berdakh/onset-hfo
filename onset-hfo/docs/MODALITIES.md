# Scalp EEG and MEG

Everything this project has validated, it validated on intracranial EEG. The
same detectors now also run on scalp EEG and on MEG. This page says how
each is read, how to open one, and what happened when they were tried on real
recordings. The short version is in the last section: the scalp detector
tracks a published scalp detector's candidates, not its final events, and
nothing here has been checked against a MEG reference.

## How each kind of recording is read

| kind | channels analysed | unit | "bipolar" means |
|---|---|---|---|
| intracranial EEG | SEEG and ECoG | µV | neighbouring contacts on one lead |
| scalp EEG | EEG | µV | the longitudinal "double banana" |
| MEG | planar gradiometers, else magnetometers | fT/cm, fT | nothing: MEG is never re-referenced |

- **Which kind.** Read from the channel types (`onset_hfo.modality`), or named
  with `PreprocessConfig.modality`. In the app, the import dialog asks.
- **Scalp EEG.** The double banana is Fp1-F7-T7-P7-O1, Fp1-F3-C3-P3-O1,
  Fp2-F4-C4-P4-O2, Fp2-F8-T8-P8-O2 and Fz-Cz-Pz. The older names T3, T4, T5
  and T6 work too. A missing electrode removes its two links rather than
  joining its neighbours across the gap.
- **MEG.** One sensor type at a time, because magnetometers and gradiometers
  are in different units. MNE types a CTF axial gradiometer as a magnetometer,
  since it reads out a field, so a CTF system is analysed in fT. Reference
  sensors are never analysed.
- **What every window says first.** On scalp EEG and MEG, the first
  preprocessing step says that the detectors were validated only on
  intracranial recordings.

## Opening one

- **A file of your own:** *File → Open a file…*, then *This recording is…* in
  the import dialog. See [`CLINICAL_GUIDE.md` §5](CLINICAL_GUIDE.md).
- **MEG formats:** MEGIN/Elekta `.fif`, CTF `.ds` folders (pick any file
  inside) and KIT/Yokogawa `.sqd` and `.con`.
- **From OpenNeuro:** the catalogue now has MEG. It lists 748 datasets: 577
  with scalp EEG, 86 with iEEG and 89 with MEG.
  - A MEG recording comes down whole, since none of its formats can be cut by
    byte range.
  - A CTF recording comes down as its folder.
  - A FIF recording split across files comes down with all its parts.
- **Head-position coils.** A MEG recording with continuous head-position
  (cHPI) coils has a preprocessing step naming their frequencies and the HFO
  band they fall in. They are not removed.

## Tried on real recordings

### Scalp EEG: ds003555

Sleep scalp EEG from 30 children with epilepsy, recorded in Zurich, with the
group's own automated ripple detections in its derivatives.

**The raw recording.** The first 60 s of sub-01 opened as scalp EEG at
1024 Hz with 23 channels named in the old 10-20 system.
- **A bug it found.** The EDF leaves its unit field blank, which MNE reads as
  volts. Every amplitude came out in tens of volts, and the quality checks set
  17 of 18 channels aside. The fetcher now reads the unit from the dataset's
  `channels.tsv` when the file states none, and the window's notes say so.
- **After the fix.** All 18 double-banana pairs were kept, with robust
  amplitudes of 22–102 µV. The most ripple detections were frontal: Fp1-F7
  had 25 in the minute, where muscle and eye movement sit.

**Against the published detector.** `scripts/run_scalp_comparison.py` runs
this project's RMS detector on the dataset's own 5-minute N3 intervals. These
are the first interval of subjects 01–10, at 2000 Hz, with 52 nearest-neighbour
bipolar channels. It compares the count on each channel with the published
detector's.

| | |
|---|---|
| channels kept by the quality checks | 52 of 52 in every subject |
| Spearman ρ across channels, with their events after artefact rejection | median 0.73, range 0.16–0.90 |
| the same, with their candidates before artefact rejection | median 0.79 |
| the same busiest channel | 5 of 10 subjects |
| events found | ours 11,174; theirs 8,369 candidates, of which they kept 2,109 |

- **Where they agree.** In sub-01 both detectors rank the channels nearly
  alike (ρ 0.90) and point at the same channel, P3-O1.
- **Where they part.** In sub-02, sub-05 and sub-07 the published artefact
  rejection removed most candidates: 1,328 to 134, 3,890 to 349, 1,177 to 175.
  Ours counted 2,526, 4,270 and 2,127. In sub-02 and sub-07 the agreement
  with their final events fell to 0.24 and 0.16, while it stayed at 0.79 and
  0.65 with their candidates.
- **What it means.** On scalp EEG this detector behaves like the published
  detector's candidate stage. It has no equivalent of the artefact rejection
  that removed three quarters of those candidates. On an artefact-heavy scalp
  recording, its rates can be several times what the published method reports.
- **What it is not.** Their detector is not ground truth, and nobody here has
  looked at the events. The intervals are their choice of clean N3 sleep,
  which flatters any detector.

The table is `data/scalp/scalp_comparison.csv`.

### MEG: ds000247 (CTF) and ds003352 (MEGIN)

**CTF empty-room recording.** `ds000247`, sub-emptyroom, 60 s at 2400 Hz. This
is the first real CTF recording through the new reader and folder fetch.
- **What was read.** 270 axial gradiometers, analysed in fT, and 26 reference
  sensors, which were left out. The median robust amplitude was 79 fT, and
  every sensor was kept.
- **Ripple detections:** none, on a recording with no head in it.
- **Discharge detections:** 28 across the 270 sensors. On noise, every one is
  a false positive.

**MEGIN recording of a person.** `ds003352`, sub-1, run 00, 60 s at 1000 Hz.
- **What was read.** 204 planar gradiometers, analysed in fT/cm, all kept,
  with a median robust amplitude of 59 fT/cm.
- **Ripple detections:** 325, with mean amplitudes of 115–750 fT/cm on the
  busiest sensors. That is far larger than published MEG HFOs, which are a few
  fT/cm. These are not credible as brain ripples.
- **Not the head-position coils.** The recording's coils run at 293–328 Hz,
  above the ripple band, and removing them with `mne.chpi.filter_chpi` changed
  325 detections to 327.
- **What else is in the band.** Besides the mains harmonics, which are
  notched, the gradiometers carry narrow spectral lines at 117 and 234 Hz.
  What the detections are has not been worked out.
- **The fast-ripple band** cannot be analysed at 1000 Hz.

**One dataset that could not be fetched.** `ds008462`, a CTF dataset with an
epilepsy task, is listed on OpenNeuro's storage, but its files are refused
(access denied by S3 itself). The fetcher now says so instead of failing with
a bare HTTP 403.

## What this adds up to

- **Reading them works.** Scalp EEG and MEG are read, in the right units, from
  every format tried: EDF, MEGIN FIF and CTF. They are analysed end to end, and
  every window says the detectors were validated only on intracranial EEG.
- **Scalp EEG.** Channel rankings track a published scalp detector's
  candidates (median ρ 0.79). Without an artefact-rejection stage, absolute
  rates on noisy recordings are not comparable to published scalp rates.
- **MEG.** Nothing here has been checked against a MEG HFO reference. The one
  recording of a person produced detections too large to be brain ripples.
  Treat MEG rates as unvalidated, and read the events before believing any of
  them.
