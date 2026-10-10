# The data

## What we use, and why

[**OpenNeuro ds003029** — *Epilepsy-iEEG-Multicenter-Dataset*](https://openneuro.org/datasets/ds003029)

| | |
|---|---|
| Licence | **CC0** (public domain dedication) — no data use agreement needed |
| Contents | intracranial EEG (ECoG and SEEG) from ~100 subjects, four clinical centres (JHH, NIH, UMMC, UMF) |
| Format | BIDS-iEEG; signals as BrainVision (`.vhdr` + `.vmrk` + `.eeg`) |
| Sampling | 1000 Hz for the recordings used here |
| Annotations | clinician markers for electrographic seizure onset/offset, and free-text notes naming contacts; per-channel `good`/`bad` status |
| Published with | *Neural Fragility as an EEG Marker of the Seizure Onset Zone* (Li et al., Nature Neuroscience, 2023) |
| DOI | `10.18112/openneuro.ds003029.v1.0.3` |

It was chosen for four reasons: it is genuinely public (CC0, no gatekeeping),
it is intracranial (scalp EEG cannot show ripples), it carries clinician
annotations we can compare against, and its files can be read **in byte
ranges over plain HTTPS**, which is what makes a ten-minute notebook possible.

The default example is `sub-pt01`, `task-ictal`, `run-01`: 98 channels,
1000 Hz, 269 s, with an electrographic seizure marked at 75.95–161.82 s.

## The second dataset: expert HFO markings, interictal sleep

[OpenNeuro **ds003498**](https://openneuro.org/datasets/ds003498) — the
**Zurich iEEG HFO dataset**, CC0, 20 subjects, 2000 Hz, five-minute segments
of slow-wave sleep chosen for HFO analysis. Recorded at University Hospital
Zurich; converted to BIDS by the author of `mne-hfo`.

It matters for one reason: **its `events.tsv` carries expert HFO markings**.

```
onset    duration  trial_type        value  sample
0.0015   0.016     fr_PHR1-2         21     3
0.0015   0.0515    ripple_AHR3-4     41     3
0.0525   0.0695    ripple_HL2-3      48     105
```

Each row is one marked oscillation: its kind, and the **bipolar channel** it
sits on, written with the shared electrode name once (`HL2-3` means
`HL2`–`HL3`, which is exactly how this pipeline names that channel).
`onset_hfo.datasets.parse_hfo_annotations` turns those into a truth table, and
`onset_hfo.benchmark` scores against it. Three properties of this reference
have to be stated, because each one changes how a score should be read.

**It is a detector's output that a human validated, not a census.** The
markings come from the Morphology detector used in the original study,
reviewed by the authors. An oscillation neither the detector proposed nor the
reviewer noticed is absent from the file, so "recall" means *recall of those
validated events*, and an event we find that they did not is counted against
us even when it may be real.

**Only some channels were reviewed.** The study kept the three most mesial
bipolar channels in temporal-lobe cases, so a typical recording has markings
on ~23 of ~43 possible channels. A detection on an unreviewed channel is
**unjudged, not wrong**, and every score in this repository is restricted to
reviewed channels for that reason. `Recording.reviewed_channels` carries the
list; forgetting to use it silently halves precision.

**A `frandr` event is one event in two bands.** A fast ripple riding on a
ripple becomes two rows, one per band, because the evidence really is in
both, and each band is scored separately.

**Mains is 50 Hz here and 60 Hz in ds003029.** The `DatasetSpec` carries it
and preprocessing takes it from the recording unless told otherwise. Notching
the wrong one leaves the interference in place *and* carves a hole where
there was none — the kind of mistake that never announces itself in a plot.

```python
from onset_hfo.datasets import fetch_slice, list_subjects

list_subjects("ds003498")                      # 20 subjects
rec = fetch_slice(dataset="ds003498", subject="sub-01", run="01", t_start=0, t_stop=60)
rec.ground_truth.head()                        # expert events, per band, per channel
rec.reviewed_channels                          # the only channels a score may use
```

Cite: Fedele T, Burnos S, Boran E, Krayenbühl N, Hilfiker P, Grunwald T,
Sarnthein J. *Resection of high frequency oscillations predicts seizure
outcome in the individual patient.* Sci Rep 7:13836 (2017).
doi:10.1038/s41598-017-13064-1

### The two sidecars that make an outcome study possible

`participants.tsv` carries **surgical outcome** for all 20 subjects: `outcome`
(`S` = seizure-free, 13 subjects; `F` = recurrence, 7), ILAE class 1–6,
follow-up 10–46 months, lesional status, and whether the epilepsy was temporal
(9) or extratemporal (11).

`sourcedata/clinical_ch_sheet_zurich.xlsx` carries the **resected zone**: one
row per subject, with `rz` as free-text contact ranges (`"ahr1-4, ar1-4,
phr1-4"`) and `excluded` listing contacts the source study dropped because
electrical stimulation of them evoked a motor or language response.
`onset_hfo/clinical.py` parses both, expands the ranges to contacts, and maps
them onto bipolar channels; `onset_hfo/outcome.py` uses them. See
[`OUTCOME.md`](OUTCOME.md).

Two things to know before relying on that sheet:

* **Recorded channels are a subset of implanted contacts.** In 15 of 20
  subjects every resected contact appears in the recording. In the other five
  — all temporal-lobe cases, where the study kept only the three most mesial
  bipolar channels — only 4 of 16 listed resected contacts were recorded.
  `Resection.coverage()` reports this per subject and the outcome study writes
  it to `recordings.csv`; do not quote a "share inside the resection" for
  those five without it.
* **The sheet has a typo.** Subject 15's `excluded` cell reads `1ll22-24`
  where every other token on the row says `tll`. `expand_contact_ranges`
  returns unparsable chunks rather than dropping them, and `resection_map`
  prints a warning, so the three contacts are visibly unclassified instead of
  silently missing.

## How only 24 MB gets downloaded

BrainVision stores samples **multiplexed**: channel 1 at time 1, channel 2 at
time 1, …, channel N at time 1, then time 2. There is no per-sample header. So
sample *k* begins at byte `k × n_channels × bytes_per_sample`, and a byte range
is a time range.

`onset_hfo.datasets.fetch_slice` therefore:

1. downloads the three small text files (`.vhdr` header, `channels.tsv`,
   `events.tsv`) in full;
2. parses the header for channel count, sampling interval and binary format;
3. issues one HTTP `Range` request for the seconds you asked for;
4. writes that slice into a cache directory next to a copy of the header, plus
   a minimal marker file;
5. reads it with MNE, attaches the annotations (shifted into slice time),
   marks the dataset's `bad` channels, and records `t_offset` so that every
   time the pipeline reports is a time in the **original** recording.

60 seconds of 98 channels at 1000 Hz in 32-bit floats is 23.5 MB. The whole
run is 105 MB; the whole dataset is hundreds of gigabytes.

```python
from onset_hfo.datasets import fetch_slice, list_runs

list_runs("sub-pt01")                       # what exists, with file sizes
rec = fetch_slice("sub-pt01", "ictal", "01", t_start=50, t_stop=110)
rec.provenance()                            # everything needed to cite it
```

Slices are cached under `artifacts/data/`; a second call is instant. Set
`ONSET_HFO_HOME` to move the cache (Colab: leave it alone).

## What the annotations mean — and do not

**Seizure onset and offset** come from the clinician's markers in the events
file. The dataset's own README warns that marker wording is not standardised
across centres, which is why `datasets.py` matches them with regular
expressions rather than exact strings.

**Contact names in the markers.** During a seizure the reviewer types notes:
`"AD1-4, ATT1,2"`, `"G16"`, `"AST1,3"`. `parse_marked_contacts` expands those
into contact names. This is a **weak textual reference**, and the distinction
matters:

* it is *not* a curated seizure-onset zone — the curated version is a
  separate file in the same archive, described below;
* it is incomplete, unordered, and written for a human reader;
* a contact may be named for reasons this pipeline does not model.

`onset_hfo.evaluate.marked_contact_check` compares the top-ranked channels
against these contacts with a permutation test. Read the result as mild
encouragement at best, and never as accuracy. See `EVALUATION.md`.

## The curated labels, which are in the archive after all

An earlier version of this document said the curated seizure-onset zone
"lives in a clinical spreadsheet in the publication, not in this archive".
That was wrong. The archive ships
**`ds003029/sourcedata/clinical_data_summary.xlsx`** — 30 KB, CC0 like
everything else, one row per patient:

| column | what it gives |
|---|---|
| `soz_contacts` | the clinician-defined seizure onset contacts, in shorthand (`"TT1-6; AST1-2, mst1-2"`) |
| `engel_score`, `ilae_score` | surgical outcome scales |
| `outcome` | see the warning below |
| `surgery_type` | `resection`, `ablation`, or blank |
| `clinical_center` | `nih`, `jhh`, `ummc`, `umf` (and `cc`, whose signals are not published) |

`onset_hfo/cohort.py` reads it. Of the 35 subjects with signal files, **32
have a row**, and on every subject spot-checked the parsed contact names
matched `channels.tsv` exactly (10/10 on `sub-pt01`, 31/31 on `sub-umf004`,
19/19 on `sub-jh103`).

```python
from onset_hfo.cohort import soz_labels, cohort_table

labels = soz_labels("sub-pt01")
labels.soz_contacts      # {'AD1', ..., 'ATT2', 'PD1', ...}
labels.engel, labels.seizure_free, labels.site   # 1, True, 'NIH'
labels.trustworthy       # True: curated label AND the surgery worked

cohort_table()           # all 100 patients, decoded, ready to group by site
```

### Two traps that silently corrupt a study

**`outcome` is not what it looks like.** `S` means *success* — seizure free,
and every `S` row carries Engel 1. `F` means *failure* (Engel 2–4). `NR`
means no resection was performed. Reading `F` as "free" inverts every label
in the cohort. `decode_outcome` is the only place this mapping is written
down, and a test pins it against the Engel scores.

**Subject ids do not match across the two files.** The S3 tree has
`sub-pt01`; the spreadsheet says `pt1`; there is *also* a separate, nearly
empty `sub-pt1` directory. `normalize_subject` strips the `sub-` prefix and
the zero padding so both sides agree.

### What the labels still do not license

A contact is a trustworthy positive only when the clinician named it **and**
the patient became seizure free — resected contacts in patients who kept
seizing are ambiguous, and treating them as positives is a common flaw in
published work. `SozLabels.trustworthy` encodes exactly that conjunction.

And these are ictal recordings. Ripple energy during a seizure spreads well
beyond the onset region, so a ranking that scores no better than chance
against these labels is the expected result, not a broken detector. See
`EVALUATION.md` §6.

### Working before the real labels arrive

A cohort that has not been curated has no labels, and neither does the
simulator — which would leave the scoring and ablation machinery untestable
until a medical centre sends a spreadsheet. Two stand-ins fix that, and
neither can be mistaken for the real thing.

**Synthetic recordings** get exact labels, because the simulator knows which
contacts it implanted ripples on:

```python
from onset_hfo.cohort import labels_from_ground_truth
labels = labels_from_ground_truth(recording)     # source="synthetic_truth"
```

A contact is labelled when it carries at least a quarter of the busiest
contact's events. The relative rule matters: an absolute floor alone sweeps in
every background contact that happened to get two events, pushing label
prevalence to a third of all channels and making any score against it
meaningless. This is the one case where a *non-null* score is informative — it
proves the scorer works. On the synthetic recording the fixed pipeline scores
3 hits at k = 5 against 1.03 expected by chance, p = 0.025.

**Real data awaiting curation** gets an arbitrary but reproducible stand-in:

```python
from onset_hfo.cohort import placeholder_labels
labels = placeholder_labels("anon-01", ch_names, n_contacts=6)   # source="placeholder"
```

These contacts are chosen **independently of the signal**, on purpose. A
stand-in that quietly correlated with what the detector finds would make every
downstream number look encouraging for no reason, which is worse than having
no labels at all.

Both carry `is_placeholder = True`, both fail `trustworthy`, and the warning
travels into every score's JSON — a stand-in that reaches a results table
unmarked makes the table worthless in a way nobody can detect afterwards.
`soz_labels` itself never invents anything: asking for a stand-in is an
explicit call, so the decision to work against made-up labels is always
visible at a call site.

From the command line:

```bash
# run the whole scoring path on stand-ins, and emit the CSV a centre fills in
python -m onset_agent.orchestrate --synthetic --score --labels placeholder \
    --write-label-template labels/our-centre.csv

# ... they send it back filled in; that is the entire swap
python -m onset_agent.orchestrate --synthetic --score --labels-csv labels/our-centre.csv
```

`--labels placeholder` never overrides real labels — it is a fallback for when
there are none, and it says so when it declines.

### Using your own labels instead

Everything downstream depends only on `SozLabels`, so a local cohort drops in
without touching the pipeline or the agent. Write one CSV:

```csv
subject,soz_contacts,engel,seizure_free,site
anon-01,"LA1-3; LH2",1,True,our-centre
```

```python
from onset_hfo.cohort import soz_labels
labels = soz_labels("anon-01", csv="our-centre/soz.csv")   # source="local"
```

`soz_labels` prefers, in order: your CSV, the archive spreadsheet, and
finally the free-text markers — and records which it used in
`labels.source`, so a number computed against weak labels can never be
reported as if it were computed against curated ones.

**`status = bad`** in `channels.tsv` marks contacts the dataset authors
excluded: white matter, ventricle, CSF, outside the brain, or noisy. The
pipeline drops them by default, and the report states how many and which.

## One thing this dataset cannot give us

It publishes **ictal** snapshots — recordings around seizures. Clinical HFO
research is usually done on **interictal** data (between seizures, often in
slow-wave sleep), where a high ripple rate is the marker of interest.

The archive does list 25 `task-interictal` files, which looks promising until
you check what they are: all but one are metadata only (`channels.tsv`,
`events.tsv`, `*_ieeg.json`) with **no signal file**. Exactly one interictal
recording in the whole dataset — `sub-umf002`, run 01 — ships an `.eeg`. So
interictal analysis is effectively unavailable here, and a second archive is
needed for it.

So what the prototype measures on this data is "where, in this seizure, the
ripple band is loudest", not the classical interictal HFO rate. The pipeline
partly compensates by comparing the pre-onset portion of the window with the
seizure itself (`rate_change`), but the pre-onset stretch is minutes from a
seizure and is not equivalent to a quiet interictal recording. This is stated
in every report's limitations and is the first item in the roadmap.

## No electrode coordinates

There is no `electrodes.tsv` anywhere in `ds003029` — not for any subject.
That rules out, on this dataset, everything that needs geometry: pairing
bipolar channels by Euclidean distance instead of contact number, distance-to-
neighbour features, and any source localisation. The bipolar caveat in
`METHODS.md` §1 therefore stands unqualified here, and fixing it properly
requires coordinates from a different archive.

## A note on amplitude units

The BrainVision headers declare a resolution of 1 nV per stored unit, and the
loader converts to microvolts on that basis. The resulting background level in
this recording is about 120 µV RMS on a bipolar channel before the seizure,
which is plausible for ECoG but on the high side of the usual 20–100 µV, so
treat absolute microvolt values in the report as "as declared by the archive's
header" rather than as calibrated measurements.

Detection itself is unaffected by a constant scale factor: every threshold in
this pipeline is expressed in robust standard deviations **of the channel
itself**, so multiplying a recording by any constant changes no decision. Only
the absolute µV numbers printed alongside events would move.

## Sampling rate and the fast-ripple question

At 1000 Hz the Nyquist frequency is 500 Hz. Ripples (80–250 Hz) are fine.
**Fast ripples (250–500 Hz) are not analysable**: they sit against the Nyquist
edge, where anti-alias filtering and noise make any measurement untrustworthy.
`config.Bands.usable()` checks this, the pipeline skips the band, and the
report says so. Studies that want fast ripples record at 2000 Hz or more — the
simulator defaults to 2000 Hz for exactly that reason.

## Any recording on OpenNeuro

The archives above are described by hand in `onset_hfo/config.py`.
`onset_hfo.openneuro` reaches the rest of OpenNeuro by dataset id, the way
`sklearn.datasets` fetches a dataset:

```python
from onset_hfo import openneuro

about = openneuro.describe_openneuro("ds004100")   # name, licence, authors, DOI, README
print(about.DESCR)
runs = openneuro.list_openneuro("ds004100")        # one row per continuous recording
bunch = openneuro.fetch_openneuro("ds004100", subject="HUP060", task="ictal",
                                  t_start=100, t_stop=130)
bunch.data, bunch.times, bunch.ch_names, bunch.sfreq  # channels x samples, volts;
                                                      # original-recording seconds
bunch.recording                                       # the package's Recording
bunch.channels, bunch.events, bunch.electrodes        # the dataset's own sidecars
bunch.line_freq, bunch.license, bunch.citation
```

**What comes down.**
- **BrainVision and EDF/BDF:** only the window, by byte range. A window of an
  EDF is cut from its data records, which are a second or so long, and then
  trimmed to the exact seconds asked for.
- **EEGLAB, FIF and other formats:** whole files, refused above `max_mb`
  (500 MB by default) rather than started.
- **NWB and MEF3:** listed but not fetched. Download them with the OpenNeuro
  client and open them with *File → Open a file*.

**The sidecars.** The window comes with the dataset's own `channels.tsv`,
`events.tsv`, `*_ieeg.json`, `electrodes.tsv` and `coordsystem.json`, so these
are the dataset's:
- channel types;
- bad channels;
- mains frequency;
- events;
- electrode positions.

They are found by BIDS inheritance, so a sidecar at the dataset's top level
counts.
- **No stated mains frequency:** 50 Hz is assumed, and the recording's notes
  say so.
- **Seizure markers:** `events.tsv` is read for them only when the task is
  ictal or a marker names a seizure. Otherwise a task block's "start" or a
  stimulus "onset" would be taken for one, as it was in the first test on
  ds003688 and ds004473.

**Where it is kept.**
- Everything is kept under a data home: the `data_home` argument, else
  `$ONSET_HFO_DATA`, else the package's data cache.
- The next fetch of the same window reads it from there with no network.
- `download_if_missing=False` refuses to download.
- `clear_data_home()` deletes it all.

**Tested formats.** It was tried on windows of seven datasets, covering
BrainVision, EDF, BDF and EEGLAB, intracranial and scalp:

| dataset | what it is | format |
|---|---|---|
| ds003029 | multicentre epilepsy iEEG | BrainVision |
| ds003688 | Utrecht film-watching iEEG | BrainVision |
| ds004100 | HUP | EDF |
| ds004473 | OHSU sEEG | EDF |
| ds002778 | scalp EEG | BDF |
| ds004504 | scalp EEG | EEGLAB |
| ds002718 | scalp EEG; refused whole above `max_mb` | EEGLAB |

**From the command line:**

```bash
python -m onset_hfo.cli openneuro describe ds004100
python -m onset_hfo.cli openneuro list ds004100 --subject HUP060
python -m onset_hfo.cli openneuro fetch ds004100 --subject HUP060 --task ictal \
       --t-start 100 --t-stop 130       # prints the window's local file
```

**In the desktop app**, *File → Open from OpenNeuro…*:
1. Type a dataset id and list its recordings.
2. Pick one and a window.
3. The window goes through the usual import confirmation, with the dataset's
   channel types and mains frequency filled in. A channel the dataset marks
   bad comes up as not analysed: HUP types its scalp and EKG leads SEEG and
   marks them bad.

The window opens as a file of your own, so its trace counts from 0. Its label
says where it starts in the original recording.

**Ripples need at least 556 Hz.** Many archives record at 500 or 512 Hz,
HUP among them. On such a recording `run_pipeline` skips the HFO detectors
and says why in `result.hfo_skipped` and the report's notes. It still runs:
- the interictal discharge detector;
- the quality checks.

The import dialog does not offer to open such a recording, because the
window has no HFO band to rank it by.

## Using your own data

**The desktop reviewer does this with a file dialog.** *File → Open a file…*
reads EDF, BDF, GDF, BrainVision, Persyst, Nihon Kohden, Nicolet, Curry,
Blackrock, Neuralynx, MEF3, EEGLAB, EGI, Neuroscan, Eximia and FIF through
`onset_hfo.io`, which dispatches on the extension to one of MNE's readers and
returns the same `Recording` the archive loader returns. It confirms the
channel types with you first; see [`CLINICAL_GUIDE.md` §5](CLINICAL_GUIDE.md).
In code that is one call:

```python
from onset_hfo.io import channel_overview, open_recording

channel_overview("/path/to/recording.edf")       # what the file claims, per channel
rec = open_recording("/path/to/recording.edf", t_start=0, t_stop=60,
                     subject="anon-01", line_freq=50.0,
                     channel_types={"A1": "seeg", "EKG": "ecg"})
```

`channel_types` is not optional in practice. A clinical export declares every
channel `eeg`, the pipeline analyses anything typed eeg, ecog or seeg, and
nothing downstream can tell a depth electrode from an EKG lead. `open_recording`
crops before it loads, so a 60 s window of a 40 GB overnight file costs what
60 s costs, and the recording carries its offset so every reported time is a
time in the original recording.

The rest of this section is what to do when that is not enough.

**1. A different recording from the same archive.**

```python
from onset_hfo.datasets import list_runs, fetch_slice
list_runs("sub-jh103")
rec = fetch_slice("sub-jh103", "ictal", "02", t_start=0, t_stop=60)
```

**2. A local BrainVision file.**

```python
from onset_hfo.datasets import load_local_brainvision
rec = load_local_brainvision("/path/to/recording.vhdr", subject="anon-01",
                             seizure=(120.0, 190.0))
```

**3. Any other format MNE can read.** Build a `Recording` yourself — it is a
plain dataclass:

```python
import mne
from onset_hfo.datasets import Recording

raw = mne.io.read_raw_edf("/path/to/recording.edf", preload=True)
rec = Recording(raw=raw, source="local:our-centre", subject="anon-01",
                task="interictal", run="01", t_offset=0.0,
                seizure=(None, None), bads=["A1", "A2"],
                citation="internal, de-identified")
```

The pipeline needs only: a preloaded `Raw` in volts, channel names that follow
the `LETTERS + NUMBER` convention (so neighbouring contacts can be paired),
and — if you want the mains notch to work — the right `line_freq` in
`PreprocessConfig` (60 Hz in the Americas, 50 Hz in most of Europe and Asia).

**Before you load real patient data**, read `LIMITATIONS.md`. This is research
code with no access controls, no audit log and no validation. De-identify
first, work under your own ethics approval, and keep clinical decisions with
the clinical team.
