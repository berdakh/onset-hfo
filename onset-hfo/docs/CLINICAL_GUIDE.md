# Onset Review: what you are looking at

For a clinician sitting down with the software for the first time. It assumes
you read intracranial EEG and have never seen this program. It does not assume
you know what a robust standard deviation is.

Thirty minutes of reading. The last section is the one that matters.

> **Research prototype — not a medical device.** Not CE-marked, not
> FDA-cleared, never validated for clinical use. It opens public research
> recordings. Nothing it shows is a diagnosis or a surgical recommendation.

---

## 1. What the software is trying to do

High-frequency oscillations — **ripples**, 80–250 Hz, and **fast ripples**,
250–500 Hz — are short bursts in intracranial EEG. The hypothesis that has
driven twenty years of work on them is that the contacts producing the most of
them mark the tissue that has to come out. If that were reliable, a night of
interictal sleep recording would tell you something it currently takes weeks of
monitoring and a captured seizure to learn.

Marking them by hand is the bottleneck: an expert reviewing one patient's
night-time recording channel by channel is days of work, and two experts
marking the same minute disagree more than anyone would like. So the field
automates the marking. This software is one such detector, plus — and this is
the part worth your time — the machinery for asking whether its answer means
anything on the patient in front of you.

**What it does, concretely.** You pick a patient and a minute. It filters the
signal, re-references it into a bipolar montage, runs a detector, throws out
what looks like artifact, and shows you: the signal with the marks on it, a
map of where the activity was, a ranked list of channels, and — on this
dataset — what the original annotators marked on the same minute.

---

## 2. The window, panel by panel

![The Onset Review window](images/onset-review.png)

```
┌───────────────────────────────────────────────┬───────────────────┐
│  TREND     rate per channel over time         │  FINDINGS         │
├───────────────────────────────────────────────┤  channels ranked  │
│  CONTROLS  amplitude · window · channels · at ├───────────────────┤
├───────────────────────────────────────────────┤  EVENTS           │
│                                               │  one row each     │
│  TRACE     the signal, with marks on it       ├───────────────────┤
│                                               │  PATIENT          │
│                                               │  · contacts (3D)  │
│                                               │  · agreement      │
│                                               │  · provenance     │
│                                               │  · preprocessing  │
│                                               │  · quality        │
│                                               │  · assistant      │
├───────────────────────────────────────────────┴───────────────────┤
│  the caveat — always on screen                                    │
└───────────────────────────────────────────────────────────────────┘
```

Every panel can be dragged, floated, closed and reopened from **View**. The
four at the bottom right share a tab stack; drag one out by its title bar to
give it a window of its own.

The window opens at the size of your screen and no larger — maximised if it
would otherwise have overflowed. The rest of **View** is about the window
itself: **Restore the default layout** puts dragged panels back, **Fit the
window to this screen** (`Ctrl+0`) is for after a second monitor or an
over-wide layout, and **Maximise window** and **Full screen** (`F11`) work
whatever your window manager does with title bars. On a short screen a panel
scrolls rather than the window overflowing; the 3D view in particular is worth
more height than a laptop's tab stack has, which is why it is floatable.

### Trend — read this first

One row per channel, one column per five seconds, brightness for how many
events fell in that cell. It is the same idea as Persyst's spike trend and it
is used the same way: **find the bright patch, click it, and the trace goes
there.** Channels are in the same order as the Findings table, so the two read
across.

The trend does a second job that the tables cannot. A window whose activity is
three bright cells in four hundred empty ones *looks* like what it is. A ranked
list of rates from the same window looks authoritative.

Switch **Trend of** to *Expert markings* to see the same window as the original
annotators marked it. Flipping between the two is the fastest way to develop a
feel for what the detector is doing.

### Controls — amplitude, window, channels, position

Directly above the trace: a gain control with a readout, how many seconds are
on screen, how many channels, and where in the window you are, with buttons to
step and scroll. Everything here also has a key on the trace — these exist
because a control you cannot see is a control most people never find.

The **amplitude** readout is the height of the scale bar drawn on the trace, so
the two cannot disagree. Rescaling is the first thing to reach for: an 80 Hz
oscillation at the wrong gain is a slightly thicker line.

**Window** is the single most useful control in the program. At 10 s a ripple
is a smudge; at 1 s it is an oscillation you can count the cycles of. Marking
and rejecting events is done by the detector at full resolution regardless —
this only changes what you can see.

### Trace — the signal

This is `mne-qt-browser`, the viewer MNE-Python ships, embedded rather than
reimplemented. Arrow keys scroll in time and through channels, `+`/`-` rescale,
`b` is butterfly view, `?` lists everything.

The signal is **bipolar**: each trace is one contact minus its neighbour
(`AR1-AR2`). HFO work is done this way because a common reference shares its
noise with every channel and produces oscillations that appear everywhere at
once. One consequence worth holding on to: a bipolar channel is a *pair*, so an
event on `AR1-AR2` localises to somewhere around those two contacts, not to one
of them.

Marks are coloured by band — ripples blue, fast ripples red, interictal
discharges grey — and the annotators' own marks, when you turn them on, are the
same colours washed out.

**By default only the selected channel's marks are drawn.** A minute of this
dataset holds around five hundred detections; drawing all of them turns the
trace into a barcode and hides the signal underneath. Pick a channel anywhere
and the marks follow it. **View → Marks on every channel** if you want the
barcode.

### Findings — channels ranked by rate

Sorted by events per minute, busiest first. The columns that matter:

| column | what it means |
|---|---|
| **Rate /min** | events per minute on that channel |
| **95% low / high** | the range the true rate plausibly lies in, given how few events a minute holds |
| **Expert looked** | whether the archive's annotators reviewed this channel at all |
| **Expert events / min** | what they marked on it |

**The tinted rows are the point of this panel.** They are the channels whose
confidence intervals overlap the busiest channel's — that is, the channels the
data *cannot tell apart from the leader*. If eight channels are tinted, this
window has not identified one busiest channel; it has identified eight
candidates, and reporting the top one would be reporting an accident of
counting.

Grey rows are channels nobody annotated. A detection there is *unjudged*, not
wrong — so the expert columns are blank rather than zero, and the agreement
table leaves them out.

### Events — one row per detection

Time-ordered. Click a row and the trace centres on it; **Navigate → Next
event** walks the window. The filters answer the three questions people
actually ask of this list: one band, one channel, or only the events riding an
interictal discharge.

**Show rejected** displays the candidates the artifact filter removed, with the
reason — useful when you can see something on the trace and want to know why it
is not marked.

`Prom dB` is the one column that is not self-explanatory: how far the
oscillation rises above the recording's own background at that frequency. It is
what separates a real oscillation from the ringing a filter produces when it
hits a sharp transient, which is the classic way HFO detectors fool themselves.

### Patient — who this recording belongs to

![The patient record](images/onset-review-patient.png)

What the archive records: the epilepsy type, whether imaging found a lesion,
how many nights were recorded, how many channels the annotators reviewed, how
many contacts the surgeon removed, and whether any were eloquent cortex.

**The resection-coverage line is the one to read.** In five of these twenty
patients only a quarter of the contacts the surgeon removed appear in the
recording at all. When that is the case this panel says so in orange, because
everything the software then says about "inside the resection" for that patient
describes a quarter of their resection — and the number looks identical either
way.

**The surgical outcome is hidden until you ask for it.** Knowing a patient
became seizure-free changes how you read the same rate table, and this software
is for forming an impression from the signal. Tick the box when you want it.

> **The clinical record is empty, deliberately.** This archive carries no
> seizure semiology, no imaging report, no scalp EEG, no neuropsychology, no
> medication list and no surgical history. Those fields are listed on the panel
> so you can see where a site's own record would appear — and they are left
> blank rather than filled with a plausible example. An invented history in a
> clinical tool is indistinguishable from a real one, and that is not a risk
> worth taking for a nicer demo.
>
> Age, sex and handedness *are* in the archive and are **not** published here.
> Three demographic fields beside pathology, surgical extent and outcome narrow
> a cohort of twenty considerably, and no analysis in this project uses any of
> them. That blank is a choice, not missing data.

### Where the contacts are — the 3D view

![The 3D view](images/onset-review-3d.png)

Each contact drawn in a head, coloured by rate, with the busiest numbered in
rank order and the shafts joined so you can see which contacts belong to the
same electrode. When the dataset records what the surgeon removed, each contact
is ringed by it: green inside the resection, orange straddling it, dark outside.

What this gives you that the table cannot is **adjacency**. "`AR1-AR2` and
`AR2-AR3` lead" reads the same in a table whether they are two neighbouring
contacts of one electrode or two leaders on opposite sides of the head, and
those are completely different findings. Here you see which it is.

Click a contact to take the trace to it; pick a channel anywhere else and the
view turns to face it. **View** gives you the standard angles.

> **Read the orange banner under it.** On this archive — and on every dataset
> this software currently ships support for — the positions are **not
> measured**. `ds003498` and `ds003029` contain no `electrodes.tsv` and no
> stereotactic coordinates for any contact. What the layout uses instead is the
> electrode *names*, which do carry real information: `AHR3` is the third
> contact of the right anterior hippocampal depth electrode, and contacts along
> a depth electrode are numbered in spatial order. So each shaft is drawn at
> the textbook location of the structure its name claims.
>
> That is genuinely useful — it tells you the active contacts are all on right
> mesial temporal shafts and sit inside the resection, which is true and is the
> finding. It is **not** this patient's implantation, and no distance,
> trajectory or margin should be read off it. If you point this software at
> your own BIDS data and that data has an `electrodes.tsv`, it uses the real
> coordinates and the banner changes to say so.
>
> On a recording whose electrode names match no structure the software knows —
> most subdural montages, including `ds003029` — the banner says **montage
> diagram** instead, and means it: the shafts are fanned apart only so you can
> tell them from each other, and their positions and sides mean nothing at all.
> Even then, which contacts share an electrode and their order along it are
> real, and that is usually the question.

### Detector vs expert

This dataset's annotators marked HFOs channel by channel, so on these
recordings you can see exactly where the detector agrees with them and where it
does not, on the minute in front of you.

**The two "only" columns are what to look at.** *Only ours* is where the
detector marked something nobody did; *only theirs* is what it missed. Open a
few of each on the trace. That is the half-hour that teaches you what this kind
of detector is and is not for, and no summary statistic substitutes for it.

The percentages on a single minute of one patient are far too noisy to be the
detector's accuracy. Section 5 has the cohort numbers.

### Assistant — ask about this window

![The assistant](images/onset-review-assistant.png)

A language model that can only quote the analysis in front of you. It chooses
which read-only queries to run against this window; every number it states is
checked against what those queries returned, and an answer citing evidence it
never retrieved is thrown away and replaced with a refusal. Ask it which
channels to resect and it refuses — before the model is even called.

**The citations are the point.** Each one is a link to a channel and a time, so
an answer leads to the signal it was measured on. An assistant you cannot check
is not useful here; one you can check in two clicks is.

It opens on **No model**, which runs the whole loop deterministically with
nothing generative in it — that is how the guards are tested, and it answers
instantly. For a real model, install Ollama, run

```bash
ollama pull qwen2.5:7b-instruct
```

and pick *Qwen2.5 via Ollama*. Nothing is downloaded on your behalf, nothing
leaves the machine, and an unreachable server is reported rather than silently
falling back.

The numbers it quotes are the numbers in the Findings table, by construction:
it reads a pipeline result rebuilt from the same band, detectors and thresholds
the window on screen was produced with.

### Preprocessing — change what is done to the signal

![The preprocessing panel](images/onset-review-preprocessing.png)

Everything here was always being done — the pipeline high-passes, notches and
re-references before any detector sees the signal — but until now it could only
be changed by editing Python, which in practice meant nobody changed it and
nobody checked whether the defaults suited their recording. These are the same
MNE operations (`filter`, `notch_filter`, `resample`, `drop_channels`), with a
mouse on them.

| control | what it does | what to watch |
|---|---|---|
| **High-pass** | removes drift | anything up to 80 Hz is free; above that it eats the ripple band |
| **Low-pass** | off by default | an HFO band runs to 500 Hz. A low-pass below that removes the signal and leaves a rate |
| **Notch** | mains and its harmonics | harmonics at 180 and 240 Hz sit *inside* the ripple band. Turn the notch off and they are detected as oscillations |
| **Notch width** | 2 Hz by default | wide notches carve visible holes in the band you are measuring |
| **Re-referencing** | bipolar, common average, or none | see below |
| **Sampling rate** | downsample | 1000 Hz cannot carry the fast-ripple band at all |
| **Channels** | tick a contact to exclude it | exclusion happens *before* the bipolar montage, so removing one contact removes both pairs it was part of |

**On re-referencing.** Bipolar is the default and is standard for HFO work: a
common reference shares its noise with every channel and produces HFOs that
appear everywhere at once. Common average is standard elsewhere in EEG and is
offered for comparison, but it re-introduces exactly that shared noise — the
step it writes into the report says so.

Three things make this safe to play with:

* **Nothing is applied silently.** Every change writes a line into *How this
  was produced* and into the exported report. Widen the notch to 6 Hz and the
  report says 6 Hz.
* **Settings that would produce numbers rather than a measurement are
  refused.** Set the low-pass to 150 Hz while reviewing ripples and the panel
  turns red and declines — the detector would run and its rates would mean
  nothing. You find out from the panel, not from a dialog thirty seconds into a
  re-analysis.
* **The defaults are the measured ones.** Open the panel, press Apply without
  touching anything, and nothing changes — the button is disabled until you do.
  **Reset** returns to exactly that.

**Apply re-runs the whole window** from the signal up, because every number on
screen depends on these settings: the rates, the intervals, the tied set, the
trend, the 3D layout, the agreement with the annotators, the assistant's
evidence. The panels you have dragged into place stay where you put them.

### Data quality — which contacts and which seconds were analysed

*(The tab is labelled **Quality**.)*

![The data quality panel](images/onset-review-quality.png)

A rate ranking is unusually easy to poison, and both ways it fails look like
findings. A contact with a noisy amplifier produces band-limited energy all the
time; the detector finds it, the artifact check cannot throw it out (its
question is "is a narrow-band oscillation present?" and the answer is yes), and
that contact tops the table with a tight interval and agrees with itself in
every other panel. A dead contact fails the other way: no events, bottom of the
table, reassuring.

So before anything is detected, each contact gets five measurements and each
second gets one. **The verdicts come in two kinds, and the difference is the
most important thing on this screen.**

| | what it means | what happens to the contact |
|---|---|---|
| **Set aside** | no physiology produces this: flat, clipped at the amplifier's rail, swamped by mains, or too little surviving time to rate | not analysed. Its rate is **blank**, not zero, and its rank is blank. It keeps its row |
| **Analysed, flagged** | a measurement is unusual in a way that is *as consistent with the finding as with a fault* | analysed, ranked and rated like any other. You are asked to look at it |

**Why the second row exists.** The band-power check is the one that catches a
noisy amplifier, and it is the most valuable check here. On **sub-13** it
flagged `TR1-TR2` and `TR2-TR3` at 10× and 6× the montage's median — and the
archive's own annotators had marked **91, 102 and 164** ripples on those three
contacts. They are among the most epileptically active in the recording. A
contact full of real ripples has high band power *because the ripples are in
the band*, and nothing in this software can tell that from an amplifier. Had it
set them aside, it would have deleted the finding and shown you a cleaner
table.

That is the general rule here: **it removes only what cannot be real, and
flags what it cannot judge.**

**Burstiness says which way a band-power flag leans.** High band power on its
own cannot tell a noisy amplifier from a contact full of ripples. *How* that
power arrives can: events are tall spikes over a quiet floor, an amplifier is
a raised carpet. The column is the 99th percentile of the ripple-band envelope
over its 10th — and **6.6 is the value a contact with no events in it takes,
whatever its amplitude**, because band-passed noise has a Rayleigh envelope and
the ratio falls out of the algebra rather than out of this cohort.

| burstiness | reads as |
|---|---|
| ≈ 6.6 | *hf noise* — no event structure at all; a noisy or poorly-coupled contact |
| well above | *hf active* — the energy arrives in bursts |

On sub-16 this splits one shaft cleanly: `TL1-TL2` scores **38.7** and carries
170 expert-marked ripples; `TL6`–`TL10` score **6.5–6.7** and carry none
between them. Both ends were flagged by band power alone.

![Why a band-power flag needs a second look](images/onset-review-burstiness.png)

Read the middle panel: `TL1-TL2` in blue is bursts over near-silence,
`TL7-TL8` in red is a thin flat line with no events in it at all.

It is a lean, not a verdict, and it has two limits worth knowing. It separates
*events* from *carpet*, not real from artifactual — an electrode popping once a
second is bursty too, and scores like a hippocampus full of ripples. And the
separation rests on 11 contacts against 11 from this one archive. Which is why
the number is shown beside the label: neither reading removes anything, and
both still say *open it on the trace*.

Here is sub-13 with the stage running. `TR3-TR4`, `TR2-TR3` and `TR1-TR2` hold
the top three places — and two of them are flagged. Their rates and ranks are
untouched; what the software is saying is "look at these before you quote
them".

![sub-13, two of the top three contacts flagged](images/onset-review-quality-window.png)

On the twenty-patient archive this stage changes nothing: the rate tables and
the leading channel are identical with it on and off for every subject
checked. It earns its place on a recording from a clinic, not on a curated
research archive.

**Blank is not zero.** In the Findings table, a contact with rate `0.00` was
analysed and no events were found — a measurement. A contact with a blank rate
and rank 0 was not analysed at all. The difference matters when you are reading
the bottom of a ranking.

**Seconds, not just contacts.** Each second of each contact is tested for a
*discontinuity* — a step no physiology produces. It is deliberately not tested
on amplitude: an interictal discharge is the largest thing in a normal second,
and an amplitude test throws away exactly the seconds that carry the pathology,
on the channels that carry it. (It did, on sub-01: six seconds of `AR2-AR3`,
the second busiest channel in that window. That is why the test changed.)
Each contact's rate is then divided by **the time that survived**, not by the
nominal minute — otherwise rejecting an artifact would make the table less
accurate than leaving it in.

**Two controls, and no more.** *Check data quality* turns the whole stage off,
which reproduces the numbers this project measured before it existed.
*Reinstate* puts a **set-aside** contact back after you have looked at it —
greyed out on a flagged row, because a flagged contact was never taken out.
To remove a contact the checks passed, use *Channels* on the Preprocessing
panel, where it is recorded as your decision.

### How this was produced

Every step applied to the signal, in order, plus the dataset's own notes and
its citation. If a number on screen ever has to be defended, it starts here.

---

## 3. Working through a patient

1. Open **sub-01**, the first minute, the ripple band. Leave the threshold at
   the measured default.
2. **Read the status bar first.** Either a channel stands out or no channel
   does, and that sentence changes what the rest of the screen is worth.
3. Look at the **trend**. Is the activity concentrated on a few channels, or
   smeared everywhere?
4. Look at the **tinted rows** in Findings. One tinted row means a leader;
   eight means a candidate set.
5. Switch the trend to **Expert markings**. Same patch? Different patch?
6. Open **Detector vs expert**, sort by *Only theirs*, and open three of them
   on the trace. What did the detector miss, and would you have marked it?
7. Open **Where the contacts are**. Are the busy contacts neighbours on one
   shaft, or scattered? Are they inside what the surgeon removed?
8. Open the **Assistant** and ask it something you already know the answer to
   from the table. Then ask it which channels to resect, and read the refusal.
9. Open **Preprocessing**, widen the notch to 4 Hz, and press Apply. Watch the
   rates move. That is how much of this number is a filter choice, and it is
   worth knowing before quoting one.
10. Open **Data quality**. Any contact flagged there needs looking at on the
    trace before you read its rank — the software is telling you it cannot
    tell a noisy amplifier from a great deal of real activity. On sub-13 the
    two flagged contacts are the ones the annotators marked most heavily.
11. Now the same patient's **second minute** (60–120 s). Does the answer hold?

Step 11 is the one most worth doing, and the one most likely to surprise you.
Across these twenty patients, the annotators' own busiest fast-ripple channel
is the same channel in only **7 of 20 patients** when you compare one minute of
a recording against another minute of the *same* recording. That is the
experts, not our detector. A single-minute answer — from anyone — is less
stable than it looks, which is why this interface puts confidence intervals and
a candidate set in front of you rather than a number.

**Review → Export review…** writes everything above to a file, with the
provenance, the caveat and the tables. Re-running the same patient, window,
band, detector and threshold reproduces it exactly.

---

## 4. The settings you can change, and what they cost

**Band.** Ripples are commoner and easier to detect; fast ripples are the more
specific marker and are far rarer. On a 1000 Hz recording the fast-ripple band
is greyed out — a 500 Hz band needs more than 1000 Hz of sampling, and
analysing it anyway produces numbers that look fine and mean nothing.

**Detector.** Four are offered. They are different ways of asking "is this
stretch unusually energetic in this band": root-mean-square energy, line
length, Hilbert envelope, short-time energy. On this cohort they rank channels
very similarly; the feature choice matters much less than the threshold does.

**Threshold.** How far above a channel's own background a burst must rise to be
marked, in robust standard deviations. Lower finds more and marks more noise;
higher is more confident and misses more. The default is the value measured for
each detector on this cohort — not a value anyone picked by eye. Section 5 says
what moving it costs.

**Interictal discharges.** Detected alongside, and marked rather than removed.
HFOs riding on a discharge are a real and clinically distinct population, and
the `On spike` column tells the two apart.

---

## 5. Opening a recording of your own

![The import dialog](images/onset-review-import.png)

**Review → Open a file…** (or the *Open a file…* button in the open dialog)
reads a recording from this machine. There is no conversion step and no
proprietary format: the file goes to one of MNE's readers and comes back as the
same recording the archive produces, so every panel, the report and the
assistant work on it unchanged.

| System | What to open |
|---|---|
| Persyst | `.lay` (the `.dat` beside it is found automatically) |
| Nihon Kohden | `.eeg` |
| Nicolet | `.data` |
| Micromed, Natus, most clinical exports | `.edf` / `.bdf` / `.gdf` |
| BrainVision | `.vhdr` — the header, not the `.eeg` |
| Blackrock | `.ns3` / `.ns5` |
| MEF3 | any file inside the `.mefd` bundle |
| Neuralynx | any `.ncs` in the recording's folder |
| Curry, EEGLAB, EGI, Neuroscan, Eximia, MNE | `.cdt`, `.set`, `.mff`, `.cnt`, `.nxe`, `.fif` |

`.eeg` is the one ambiguous extension: BrainVision writes the *signal* to
it beside a `.vhdr` header, Nihon Kohden writes the whole recording to it.
A sibling `.vhdr` settles which, so either file of a BrainVision triplet
opens the right one. Anything MNE reads that is missing from this list is
one line in `onset_hfo/io.py`; a Nicolet `.data` needs its channel type
supplied, because its header does not carry one.

### The screen that matters is the channel list

A clinical export almost always declares **every channel as scalp EEG**,
whatever is actually in it — the one real intracranial BrainVision file this
project caches declares all fifty that way. This software analyses anything
typed SEEG, ECoG or scalp EEG and drops the rest, so a file taken at its word
produces a complete review — rates, Poisson intervals, a candidate channel set
— for the EKG lead and the DC channels as readily as for a depth electrode, and
nothing anywhere says so.

So the dialog shows you what the file claims, next to what it will be analysed
as, and will not open until something is marked SEEG or ECoG. Set them all with
one of the three buttons, then change the exceptions — the EKG, the DC
channels, the trigger — individually. The count beside the buttons is the
number that will actually be analysed.

### What else it asks, and why

* **From / To.** Only this window is read. A night of intracranial EEG is tens
  of gigabytes and this software analyses a minute of it; the range is bounded
  by the file's own length. Every time it later reports is a time in the
  original recording, not in your window.
* **Band.** Offered from the file's sampling rate, so the fast-ripple band
  disappears below 1000 Hz for the same reason it does for the archive.
* **Mains.** 50 or 60 Hz, and it is *not* read from the file. Notching the
  wrong one leaves the interference in place and carves a hole where there was
  none, and it never announces itself.
* **Label.** Goes on the report. Use a study code, not a patient name.

![The window on an imported recording](images/onset-review-imported.png)

### What an imported file does not bring

No expert markings, no resected zone, no participant record. *Detector vs
expert* and the resection overlay say they are unavailable rather than showing
an empty table, and **Patient** says the record does not exist rather than that
none was found. The contacts view still places shafts from their names, with
the same warning it always carries.

**Nothing is uploaded anywhere.** Equally, nothing here checks
de-identification, ethics approval or data governance — those remain yours.

### From a script

The window route asks for the channel types; a scripted one has to state them:

```bash
onset-review --open /data/study-001.edf              --all-channels-as seeg --channel-type 'EKG=ecg'              --window 0 60 --subject study-001              --export reviews/study-001.md
```

Without `--all-channels-as` or `--channel-type` it refuses and says why. That
refusal is deliberate: a batch run that trusted the file's own answer is the
one way to produce a folder of confident reviews of the wrong channels.

---

## 6. What this has actually been measured to do

Cohort figures from [`EVALUATION.md`](EVALUATION.md), 20 patients, against the
annotators' 41,187 marked events, on reviewed channels only.

**Ripple band, individual events.** At 2.0 SD the detector agrees with about
half of what it marks (precision 0.51) and finds about a third of what the
annotators marked (recall 0.38). Raising the threshold to 5.0 SD buys precision
0.59 and costs recall — 0.12.

**Fast ripple band.** Much harder. At 2.0 SD precision is 0.086: eleven
detections for every real one. It takes 5.0 SD to reach precision 0.54.

**Ranking channels is easier than marking events,** which is the useful part:
the detector's channel ordering correlates with the annotators' at ρ ≈ 0.65
(ripples, 2.0 SD) even where individual-event agreement is poor. If what you
want is "which channels", that is a better-supported question than "which
events".

**Against surgical outcome** — the one comparison whose reference standard is
not another opinion — see [`OUTCOME.md`](OUTCOME.md). On the whole recording,
the channel with the most expert-marked fast ripples was inside the resection
in 11 of 13 patients who became seizure-free and 3 of 7 whose seizures
returned: **AUC 0.71, p = 0.12**. Our detector, same channels: 10 of 13 versus
3 of 7, AUC 0.67, p = 0.17. **Neither is significant.** An earlier version of
this analysis, on the first minute only, gave AUC 0.82 and p = 0.007, and that
number was wrong in the sense that matters: it does not survive looking at the
other four minutes of the same recordings. Two patients accounted for the whole
difference.

That history is in the documentation on purpose. It is the clearest statement
available of how much a twenty-patient study with a binary per-patient metric
can move, and it is why this interface insists on confidence intervals,
candidate sets and a second window.

---

## 7. The five things never to conclude from this screen

1. **"The busiest channel is the seizure onset zone."** It is the busiest
   channel in one minute of one recording, by one detector, at one threshold.
   Whether that tracks the onset zone is the open question the field has not
   settled; on this cohort the association did not reach significance.

2. **"No channel stands out, so there is nothing here."** The status bar says
   the *ordering* is not supported by this window. A longer window, a different
   night, or a different band may well order it. Absence of a leader is not
   absence of disease.

3. **"The detector agrees with the expert, so it is right."** The annotators
   are the available reference standard, not the truth. Two experts marking the
   same minute disagree substantially. Agreement with one of them bounds how
   well a detector *could* be doing; it does not establish it.

4. **"The 3D view shows where the electrodes are."** It shows where electrodes
   *with those names* conventionally go. No dataset this software currently
   reads contains a measured coordinate for any contact. The view is for
   adjacency and for the relationship to the resection, not for geometry.

5. **"This tells me what to resect."** It does not. No part of this software
   has been validated against a surgical decision, the cohort is twenty
   patients, five of whom have only a quarter of their resected contacts
   present in the recording at all, and the one outcome comparison it has been
   through did not separate the groups. [`LIMITATIONS.md`](LIMITATIONS.md) is
   the full list and is shorter than this guide.

---

## Where to go next

* [`INSTALL.md`](INSTALL.md) — installing it, and what to do when it will not start
* [`DATA.md`](DATA.md) — the recordings: where they come from, what the
  annotations mean, and how to point the loader at your own data
* [`METHODS.md`](METHODS.md) — every algorithm and threshold, with the paper
  each came from
* [`EVALUATION.md`](EVALUATION.md) — everything that has been measured
* [`LIMITATIONS.md`](LIMITATIONS.md) — **read before quoting any number from
  this software to anyone**
