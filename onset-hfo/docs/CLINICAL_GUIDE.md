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

```
┌──────────────────────────────────────────────┬────────────────────┐
│  TREND    rate per channel over time         │  FINDINGS          │
├──────────────────────────────────────────────┤  channels ranked   │
│                                              ├────────────────────┤
│  TRACE    the signal, with marks on it       │  EVENTS            │
│                                              │  one row each      │
│                                              ├────────────────────┤
│                                              │  DETECTOR V EXPERT │
│                                              │  HOW IT WAS MADE   │
├──────────────────────────────────────────────┴────────────────────┤
│  the caveat — always on screen                                    │
└───────────────────────────────────────────────────────────────────┘
```

Every panel can be dragged, floated, closed and reopened from **View**.

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
7. Now the same patient's **second minute** (60–120 s). Does the answer hold?

Step 7 is the one most worth doing, and the one most likely to surprise you.
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

## 5. What this has actually been measured to do

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

## 6. The four things never to conclude from this screen

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

4. **"This tells me what to resect."** It does not. No part of this software
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
