# The poster claim

A poster gets one claim. Everything on the board either supports it, bounds
it, or should not be on the board. This document fixes that claim before any
artwork exists, so the layout serves an argument rather than filling space.

Audience: clinicians and clinical researchers at a medical expo. Companion
documents: [`EXPO.md`](EXPO.md) for what to say at the stand,
[`OUTCOME.md`](OUTCOME.md) for every number quoted here.

---

## The claim

> **In HFO analysis the unit of analysis decides whether a result is
> reproducible. Across five minutes of one recording an expert's busiest-channel
> answer holds for 9 of 20 patients; across five whole recordings it holds for
> 18. The single significant p-value this project ever reported was the most
> favourable of five minutes.**

The two-sentence version, for a passer-by:

> We asked whether the HFO map predicts who becomes seizure-free, and then we
> asked the same question again on a different minute of the same recordings.
> The answer changed — and that turned out to be the finding worth reporting.

### Why this claim, and not the obvious one

The obvious poster claim is *"our detector predicts surgical outcome"*. We
cannot make it: on this cohort neither our detector (AUC 0.67, p = 0.17) nor
the experts' own markings (AUC 0.71, p = 0.12) reach significance, and with 13
seizure-free against 7 recurrences the smallest effect the design can detect at
80% power is AUC 0.85. A poster claiming prediction here would be claiming
something the data cannot carry.

The claim above is one the data *can* carry, because it is a within-cohort
comparison of the same patients analysed two ways, not an underpowered
between-group test. It is also the more useful claim for this audience: every
HFO paper an epileptologist has read reports a number computed on some window,
and almost none report what that number would have been on the next window.

---

## The evidence ledger

Each clause of the claim, and exactly what backs it.

| Clause | Evidence | Source |
|---|---|---|
| "expert's busiest-channel answer holds for 9 of 20 across five minutes" | 5 disjoint 60 s windows tiling the same run, analysed identically | [`OUTCOME.md` § A whole run is a stable unit](OUTCOME.md) |
| "…holds for 18 of 20 across five whole recordings" | 5 runs per subject from different nights; only `sub-08` and `sub-18` move | same |
| "the single significant p-value was the most favourable of five minutes" | expert arm p ranges 0.007–0.613 across the five minutes; 0.007 is the minimum | [`OUTCOME.md` § But which minute you pick matters](OUTCOME.md) |
| the detector is the more reproducible of the two at short windows | same busiest channel in all five minutes: 12/20 (rms) vs 7/20 (expert); same inside/outside answer 16/20 vs 9/20; AUC spread 0.09 vs 0.25 | [`OUTCOME.md` § Our detector is more stable](OUTCOME.md) |
| the outcome estimate itself is settled on a whole run | AUC identical from 180 s to 300 s — identical, not merely close | [`OUTCOME.md` § The estimate settles](OUTCOME.md) |
| the headline outcome numbers | expert 11/13 vs 3/7, AUC 0.71 (0.50–0.92), p = 0.12; rms 10/13 vs 3/7, AUC 0.67 (0.45–0.89), p = 0.17 | [`OUTCOME.md`](OUTCOME.md) primary table |
| the power floor | 13 vs 7 detects AUC ≥ 0.85 at 80% power | `onset_hfo.outcome.min_detectable_auc` |
| multiplicity | 36 comparisons; one at p = 0.034 uncorrected, Bonferroni 1.00; chance alone gives ~2 | [`OUTCOME.md`](OUTCOME.md), [`LIMITATIONS.md`](LIMITATIONS.md) |
| detector agreement with expert markings | ripple 2.0 SD: precision 0.51, recall 0.38, channel-rank ρ 0.66. fast ripple 5.0 SD: 0.54 / 0.20 / 0.61, over 41,187 marked events | [`EVALUATION.md`](EVALUATION.md) |

Two caveats that belong **on the poster**, not in a drawer:

- The across-runs comparison is restricted to the 18 subjects with more than
  one run (13 seizure-free against 5), so its absolute AUCs are not comparable
  to the 13-vs-7 numbers elsewhere.
- At 30 s the detector arm covers only 16 of 20 patients, because the rest
  produce no fast-ripple detections in that little data. Its short-window
  points are a different cohort and are not part of the same curve.

---

## The non-claims

Printed small on the board, and said out loud when asked:

- **Not** that HFOs localise the epileptogenic zone. The randomised HFO Trial
  (2022) came out against HFO-guided tailoring; this poster does not reopen
  that.
- **Not** that our detector is accurate. Against expert markings it reproduces
  about half of their validated events. That is *agreement with another
  detector*, not accuracy against ground truth.
- **Not** that reproducibility implies correctness. A detector that returns the
  same wrong channel every minute is perfectly reproducible. The experts remain
  the arm closer to outcome; ours is the arm that agrees with itself more.
- **Not** a clinical tool. Research tool, tested on public de-identified data, no
  diagnosis and no treatment recommendation anywhere in the system.

---

## Abstract

> **Which minute you analyse decides your answer: reproducibility of HFO-based
> channel ranking on a public cohort with surgical outcome.**
>
> High-frequency oscillations are among the most-studied candidate markers for
> identifying resectable tissue, and results are typically reported from a
> single analysis window. We asked what happens when the window moves. Using
> twenty patients from a public intracranial archive (ds003498; interictal
> sleep at 2000 Hz, CC0) that ships expert HFO markings, the resected contacts
> and seizure outcome, we compared the channel generating the most fast ripples
> against the tissue the surgeon removed — first on whole 300 s recordings,
> then on five disjoint one-minute windows of the same recordings, then across
> five whole recordings from different nights.
>
> On whole recordings the busiest channel lay inside the resection in 11 of 13
> seizure-free patients against 3 of 7 with recurrence (AUC 0.71, 95% CI
> 0.50–0.92, p = 0.12) using expert markings, and 10 of 13 against 3 of 7 (AUC
> 0.67, 0.45–0.89, p = 0.17) using a threshold-crossing RMS detector. Neither
> reaches significance; these group sizes detect only AUC ≥ 0.85 at 80% power.
>
> Reproducibility separated the two arms sharply. Across five minutes of one
> recording the expert markings gave the same busiest channel for 7 of 20
> patients and the same inside/outside answer for 9 of 20, with group AUC
> spanning 0.566–0.819 and p spanning 0.007–0.613; the automated detector gave
> 12 of 20 and 16 of 20, with AUC spanning 0.650–0.740. Changing the unit of
> analysis from a minute to a whole recording raised the experts' agreement
> with themselves from 9 of 20 to 18 of 20, while the detector was unchanged at
> 16 of 20. An earlier version of this analysis, computed on the first 60
> seconds, reported AUC 0.82 and p = 0.007 — the most favourable of the five
> minutes — and that figure is published alongside its correction.
>
> A whole recording is a stable unit of measurement; a minute of one is not.
> All code, both archives and every table are public.

---

## Panel plan (A0 portrait, six panels)

| # | Panel | Content | Figure |
|---|---|---|---|
| 1 | **The question** | Epilepsy surgery works when the right tissue comes out. HFOs are a proposed marker. Almost every report uses one window. What if it moves? | none — type only |
| 2 | **What one detection is** | Wideband, band-passed, and the event's spectrum against the recording's own 1/f background. A real oscillation leaves a bump above the dashed line; filter ringing does not. | `img/real_example_event.png` |
| 3 | **Against surgical outcome** | The primary table, both arms, with CI, p, and the power floor stated beneath it. | table + `img/real_rates_rms.png` |
| 4 | **Which minute you pick** | The five-window spread for both arms; the p range 0.007–0.613 called out. This is the panel the claim rests on. | `img/window_stability.png` |
| 5 | **A whole run is stable** | 9/20 → 18/20 for the experts; 16/20 → 16/20 for the detector. The growing-window curve going flat from 180 s. | `img/run_stability.png` |
| 6 | **What we corrected, and what we will not say** | The 60 s → 300 s correction published next to the original; the four non-claims; the multiplicity note. | none — type only |

Footer strip: the four addresses from [`../../site/handout.html`](../../site/handout.html),
the CC0 attributions for ds003498 and ds003029, the lab, and **not a medical
device**.

Panel 4 should be the largest. If the board runs out of room, panels 2 and 3
shrink before it does.

### Titles, in order of preference

1. *Which minute you analyse decides your answer* — reproducibility of
   HFO-based channel ranking against surgical outcome
2. *A whole recording is a stable unit of measurement. A minute of one is not.*
3. *The p-value we published was the best of five minutes* — and what we did
   about it

Avoid any title containing "predicts", "identifies" or "localises".

---

## What a reviewer will attack

| Attack | Answer |
|---|---|
| "Twenty patients proves nothing." | Agreed for the outcome comparison, and the poster says so with its power floor. The reproducibility comparison is within-patient, same cohort analysed two ways, and does not depend on that power. |
| "You only used one detector family." | Two detectors differing in exactly one thing — the feature they threshold — so a disagreement is attributable to the feature. Both are classical and cited. A learned detector is future work, and would need this same reproducibility check before anyone believed it. |
| "Expert markings are not ground truth either." | Correct, and that is the point of the comparison rather than a flaw in it. We report agreement, never accuracy. |
| "You cherry-picked the window that fails." | The opposite: the window that *succeeded* is the one we retracted. Both tables are published side by side, and the five windows tile the recording rather than being chosen. |
| "Is 5 runs per subject enough?" | The archive holds 385 runs across the 20 subjects, about 46 GB. We read the first five per subject and state that limit. The direction of the effect is large enough that more runs would have to reverse it, not merely soften it. |
| "Why is this on a poster and not in a journal?" | It is a negative-shaped methods result on public data. It should be both. |

---

## Regenerating every number here

```bash
python -m onset_hfo.cli outcome                  # the primary table
python -m onset_hfo.cli stability                # 11 windows, ~90 min after the first fetch
python -m onset_hfo.cli stability --across-runs 5
```

Nothing on the poster may be typed by hand from memory. Every figure in the
panel plan is written by the pipeline into `docs/img/`, and every table cell
traces to a file under `artifacts/`. If a number on the board cannot be
regenerated by one of the three commands above, it does not go on the board.
