# Presenting Onset at a medical expo

Audience: **clinicians** — epileptologists, neurosurgeons, neurophysiologists,
and the fellows who will actually be asked to use something like this.

The one thing to hold onto: **you are not selling a detector.** Every stand at
a medical expo has a number that sounds good. What this project has that
almost none of them do is a *checked* number, and a record of having corrected
itself when the check came back badly. Lead with that and the conversation
goes somewhere. Lead with 0.82 and you will be asked, correctly, why anyone
should believe it.

---

## The 90-second version

> Epilepsy surgery works when you remove the right tissue. High-frequency
> oscillations are one of the markers proposed for finding it, and the
> evidence for them is genuinely contested — the randomised HFO Trial in 2022
> came out *against* HFO-guided tailoring.
>
> We built the measurement layer rather than another claim. Every rate this
> system reports traces back to the exact signal window it was measured on,
> and you can click through to see it. We scored our detectors against expert
> HFO markings on twenty patients, then asked whether the map predicted who
> became seizure-free.
>
> It points the right way and it does not reach significance on twenty
> patients. What we found along the way was more useful: our first result was
> an artefact of using sixty seconds instead of the whole recording. We
> published the correction next to the original.
>
> The demo is live — and if you ask the assistant which tissue to resect, it
> refuses, and shows you why.

If they only take one sentence away, this is it:

> **We can show you the signal behind every number, and we can show you the
> two times we were wrong.**

---

## The five-minute demo

Run it in the desktop application, installed from the latest release with
`./install.sh --with-sample`. Rehearse this exact path; the application
needs no network and no data of theirs. The page names below map to the
application as **Recording** → Review › Recording, **Report** → Report and
**Assistant** → Assistant. The cohort pages (**Detectors**, **Outcome**,
**Patients**) are no longer in the application: they are on the no-install
demo at [onsetnu.streamlit.app](https://onsetnu.streamlit.app/), which needs
the network, so open it in a browser tab before the doors open. Offline, the
application's **Validation** page gives each study's result and where it
fails, on one page.

### 1 · Recording — "where does a number come from?" *(90 s)*

Open **Recording**. Point at the ranking table.

> Two detectors, side by side. They differ in one thing only — the feature
> they threshold — so when they disagree, the disagreement is about the
> feature, not about two implementations drifting apart.

Point at the **amber rows**.

> These are channels the two rank five or more places apart. We show both
> ranks and prefer neither. In most tools you would never learn this happened.

Point at the green box above the table.

> And before any of it: does *any* channel actually stand out? A ranking
> function will happily sort noise. This asks whether the leader's rate
> interval clears the middle of the pack. If it does not, it says so.

Scroll to **Evidence window**, pick an event, let the figure draw.

> That is the actual signal. Wideband on top, band-passed in the middle, and
> at the bottom the event's spectrum against the recording's own background. A
> real oscillation leaves a bump above that dashed line. Filter ringing — a
> sharp transient that an 80–250 Hz filter turns into a convincing-looking
> ripple — does not. That check is where almost all of our precision comes
> from: 0.63 before it, 0.97 after.

**Why this lands:** a clinician's first instinct is that automated detectors
are black boxes that over-call. You have just shown them the opposite.

### 2 · Assistant — "what stops the AI making things up?" *(60 s)*

Open **Assistant**. The three seeded questions are already answered.

Scroll to the third: *Which channels should we resect?*

> Refused — and look at the reason: *out of scope, checked before the model
> ran*. There is no model in that path to be talked round. Treatment questions
> never reach one.

Point at the first answer's citation line.

> Everything else it says is checked afterwards: every citation has to resolve
> to a window that was actually retrieved, and every number has to appear in a
> tool result. If it does not, the answer is thrown away and it declines.

If they look interested, expand a citation — it opens to the stored record
*and* the signal window.

**Why this lands:** every clinician has now seen an AI tool hallucinate. This
is the answer to the question they are already forming.

### 3 · Outcome — "does it actually work?" *(2 min)*

Open **Outcome**. This is the important one. Do not rush it.

Tab **The result**:

> Twenty patients, whole recordings, and the question is: was the channel
> generating the most fast ripples inside the tissue the surgeon removed?
> Eleven of thirteen who became seizure-free. Three of seven whose seizures
> came back. That is the direction the literature predicts — and with these
> numbers it does not reach significance.

Tab **Does the window matter?** — the one to spend time on:

> Our first version of this used the first sixty seconds of each recording and
> got AUC 0.82, p equals 0.007. That went on our README. Then we ran it on the
> whole five-minute recording and it became 0.71, p equals 0.12.
>
> It was not cherry-picked — sixty seconds was chosen for download size before
> we touched any outcome data. It was something more ordinary: an analysis
> window short enough to change the answer, which nobody had checked. Across
> five separate minutes of the same recordings the p-value ranges from 0.007
> to 0.61. We had the best minute of five.
>
> We corrected it everywhere and left both tables side by side.

Tab **What it cannot support**:

> And this is the page we would want you to read first. Thirteen against seven
> can only detect a very large effect. Nothing here reaches significance, even
> before correcting for multiple comparisons. It is retrospective, one centre, one surgical team.

**Why this lands:** you have just done, unprompted, the thing they spend their
professional lives wishing vendors would do.

### 4 · Close *(30 s)*

> What we are offering is not a better detector. It is a way of making these
> measurements checkable — so that when someone does have a result, you can
> see what it rests on. The code is open, the data is public, and the
> five-minute recording you just looked at is on OpenNeuro if you want to run
> it yourself.

---

## Questions they will ask, and honest answers

**"Is it better than my technician / my fellow?"**
No, and we do not claim it. We *agree* with expert markings — channel-ranking
correlation about 0.7, event-level F1 about 0.5. The reference we score
against is itself a published detector's output after human validation, so
these are agreement numbers, not accuracy. What we are better at is being
consistent: across five separate minutes of the same recording our detector
gave the same answer for 15 of 20 patients and the human markings for 10 of 20.
Reproducibility, not accuracy.

**"Would it have changed a surgical decision?"**
We cannot say, and nobody should let us. Nothing in the study reaches
significance, even before correction for multiple comparisons.

**"What about the HFO Trial?"**
[Jacobs et al., *Lancet Neurology* 2022.] Seventy-eight patients randomised to
intraoperative HFO-guided versus spike-guided tailoring; seizure freedom at one
year was 67% with HFO guidance against 90% with spikes — non-inferiority not
met. We cite it in our own limitations document, because a randomised trial
outranks the retrospective series we reproduced. If they raise it, agree with
them. It is the strongest argument against the biomarker and pretending
otherwise costs you the room.

**"Can I try it on my patients' data?"**
Not clinically and not yet. There is no de-identification pipeline, no
authentication, no audit log and no security model. It runs on public,
CC0, already-de-identified archives. Pointing it at identifiable recordings
would be your governance problem, not a feature we offer.

**"How is this different from Persyst or the other commercial spike/HFO
detectors?"**
They are validated products with regulatory clearance and we are not competing
with them. We are the layer underneath: reproducible measurement with
provenance, open code, and published failures. If the field settles on a
biomarker, this is how you would check a vendor's claim about it.

**"What is the AI actually doing?"**
Reading results and writing sentences. It cannot compute a number, cannot run
a detector, cannot open another patient's data, and cannot answer a treatment
question — that is refused before any model runs. Every number it writes must
already exist in a tool result or the answer is discarded.

**"Twenty patients is nothing."**
Correct, and that is the honest limit. Thirteen seizure-free against seven
recurrences can only detect a very large effect — AUC 0.85 or above. That is
why we report effect sizes and intervals rather than leaning on p-values, and
why a second cohort is the biggest thing on our roadmap.

**"What would convince you it works?"**
A second centre with expert markings, resection maps and outcomes; a
prospective design; and a way to tell epileptic from physiological ripples,
which nothing in our pipeline currently attempts. We would say the same to
anyone claiming otherwise.

---

## What not to say

- Never "detects the seizure onset zone". It does not. High rate is a
  measurement; physiological ripples occur in healthy tissue.
- Never quote AUC 0.82 / p = 0.007. It is the superseded number and someone
  will find the correction.
- Never "validated". Nothing here is validated on patients.
- Never imply the assistant could support a clinical decision. It is refused
  from doing so by design, and that refusal is the demo.
- Do not oversell a small p-value from any table. The outcome study's 34
  comparisons have none below 0.05 (the smallest is 0.057, where chance alone
  would give about two); the p = 0.032 an earlier release showed went when five
  patients' electrode names were corrected after 0.3.24. The sub-population
  screen's fast-ripple arm at p = 0.004 is contradicted by its own tie-aware
  metric and is not a finding.

---

## At the stand: practicalities

- **The app needs the internet.** The first evidence figure pulls 24 MB from
  OpenNeuro. Open the Recording page and draw one figure *before* the doors
  open so it is cached, and again after any network drop.
- **If the wifi dies**, everything except the evidence figure still works —
  ranking, report, outcome, all of it reads committed data. The page says why
  the figure is missing rather than breaking. Say "that panel fetches the
  original recording live; the measurements are all here."
- **Have the demo's Outcome page open in a second tab** at the *window*
  section. It is the conversation you want.
- **A laptop beats a phone.** Three-panel figures and the ranking table need
  the width.
- **Printed handout:** `site/handout.html`, also linked from the results site
  and live at <https://berdakh.github.io/onset-hfo/handout.html> — open it and
  print to PDF. It is laid out to fill exactly one A4 page, so print it before
  you travel and check the page count.

---

## Follow-up

Everything is public: the code, both archives, and every table behind every
number on the stand.

- Desktop application — <https://github.com/berdakh/onset-hfo/releases/latest>
- Results and documentation — <https://berdakh.github.io/onset-hfo/>
- Code — <https://github.com/berdakh/onset-hfo>
- The outcome study in full, including what it cannot support —
  [`docs/OUTCOME.md`](OUTCOME.md)
- What this must not be used for — [`docs/LIMITATIONS.md`](LIMITATIONS.md)

Brain–Machine Interfaces Laboratory, School of Computing and Artificial
Intelligence, Nazarbayev University.
