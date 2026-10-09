# Glossary

For anyone arriving from either side of this project — a clinician who wants
to know what the code does, or an engineer who has never seen an EEG.

## Clinical and recording terms

**iEEG (intracranial EEG)** — electrodes placed inside the skull, in contact
with the brain. Much cleaner and higher-bandwidth than scalp EEG, which cannot
resolve ripples at all.

**ECoG (electrocorticography)** — grids and strips of contacts on the cortical
surface.

**SEEG (stereo-EEG)** — depth electrodes inserted into tissue, each carrying
several contacts along its shaft.

**Contact** — one recording point, e.g. `ATT3`. Named as electrode label +
number.

**Channel** — what is actually analysed. In a **bipolar montage** a channel is
the difference between two neighbouring contacts (`ATT3-ATT4`).

**Montage** — how channels are derived from contacts. *Referential*: every
contact minus a common reference. *Bipolar*: neighbouring contacts subtracted.
HFO work prefers bipolar because a noisy common reference otherwise injects
the same artifact into every channel.

**Ictal / interictal / pre-ictal** — during a seizure / between seizures /
shortly before one.

**Electrographic onset** — the moment the seizure becomes visible in the EEG,
marked by a reviewer. It may precede the *clinical* onset (visible symptoms).

**SOZ (seizure onset zone)** — the tissue where seizures begin, as judged by
the clinical team. Often the target of surgery. **This software does not
identify it and does not claim to.** Its ictal index ranks channels by how
early and strongly they change at a marked seizure; the zone is the clinical
team's call (`ICTAL.md`).

**Engel / ILAE score** — outcome scales after epilepsy surgery. Engel I ≈
seizure free. Present in this dataset's `participants.tsv`, unused here so
far.

**Resection / ablation** — surgical removal or destruction of tissue. Outside
the scope of anything this software says.

## Signal terms

**HFO (high-frequency oscillation)** — a brief oscillation above the classical
EEG range. **Ripple**: 80–250 Hz. **Fast ripple**: 250–500 Hz. Typically
20–100 ms, a handful of cycles. Elevated rates are associated with
epileptogenic tissue — but ripples also occur physiologically.

**IED (interictal epileptiform discharge)** — a "spike" or sharp wave between
seizures: a brief, sharp, high-amplitude transient, 20–70 ms, often followed
by a slow wave.

**Ripple riding on a spike** — an HFO occurring at the same moment as a
discharge. Real, common in epileptic tissue, and reported separately here
rather than rejected.

**Filter ringing / "false ripple"** — a sharp transient contains energy at all
frequencies, so band-passing it to 80–250 Hz produces something that looks
like a beautiful oscillation. The central false-positive problem in HFO
detection, and the reason for `validate.py`. (Bénar et al., 2010.)

**Band-pass, high-pass, notch** — filters that keep a frequency range, remove
low frequencies, or remove one narrow frequency (mains interference at
50/60 Hz and its harmonics).

**Zero-phase filtering** — filtering forwards and backwards so the output is
not shifted in time. Essential when the question is *when* something happened.

**RMS (root mean square)** — a measure of signal energy in a window.

**Line length** — the mean absolute change between consecutive samples.
Sensitive to both amplitude and frequency.

**Robust SD / MAD** — a standard-deviation estimate built from the median
absolute deviation, barely affected by outliers. Used instead of the plain SD
so that a channel full of events does not raise its own threshold.

**PSD (power spectral density)** — how power is distributed across frequency.

**1/f background** — real brain signals have power falling roughly as 1/f.
A genuine oscillation appears as a *bump above* that line; ringing does not.

**Spectral prominence** — how far above the fitted 1/f background the in-band
peak rises, in dB. The number that separates a real oscillation from ringing.

**Nyquist frequency** — half the sampling rate; the highest frequency a
recording can represent. 500 Hz for the 1000 Hz recordings used here.

**SNR (signal-to-noise ratio)** — here, an implanted ripple's peak amplitude
divided by the RMS the ripple band already carries.

**Poisson confidence interval** — an interval around a rate estimated from a
count of events, assuming events arrive independently.

## Terms specific to this codebase

**`Recording`** — signal plus provenance (source, subject, seizure markers,
slice offset).

**`Prepared`** — the microvolt array a detector sees, plus a log of every
preprocessing step.

**`Event`** — one detection, with its channel, window, score, peak frequency,
spectral prominence, and whether it was accepted (and if not, why).

**`evidence_id`** — `subject|channel|detector|start`, the citable identifier
for one event window. What the agent must cite.

**`ResultStore`** — the read-only view of a saved analysis; the agent's entire
world.

**Scripted backend** — a deterministic keyword policy that speaks the agent's
tool protocol so everything can be tested with no model. Not a language model.

**Hot contact** (synthetic data only) — a simulated contact with a much higher
implanted event rate, standing in for tissue an epileptologist would care
about.
