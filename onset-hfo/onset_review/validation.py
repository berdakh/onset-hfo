"""What each method was measured to do: the Validation page's text. Qt-free.

A reader who sees a rate, an index or a template position should be able to
find, without leaving the window, how far that kind of number was found to
agree with what clinicians marked, on which public data, and where it fails.
This module holds that, method by method, as plain sentences.

The figures are written out here rather than read from the study tables, so
the page reads the same on an install that has no tables (the wheel carries
none). `tests/test_validation.py` holds every figure to the committed tables
in ``data/``, so the two cannot drift apart. The full studies -- every
threshold, patient and chart -- are in ``docs/`` and on the no-install demo.

:func:`this_recording` adds the one thing the page can say about the open
window: when it carries expert markings, its detections scored the way the
detector study scored them (`onset_review.thisrecording.score_window`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Method", "METHODS", "markdown", "this_recording", "HEADLINE"]

HEADLINE = ("**Research software, not validated for clinical use.** These are the "
            "measurements on public recordings behind each kind of number this program "
            "shows: what it was compared with, how well it agreed, and where it fails. "
            "None of them is a measure of how it will do on your patient.")


@dataclass(frozen=True)
class Method:
    key: str
    title: str
    question: str
    data: str
    findings: tuple[str, ...]
    limits: tuple[str, ...]
    doc: str
    #: Between the findings and the failures, in order: ("figure", key) for a
    #: picture (`onset_review.validationfigures`), ("text", paragraph).
    extras: tuple[tuple[str, str], ...] = ()


METHODS = (
    Method(
        key="detectors",
        title="Finding HFOs",
        question="Do the detectors find the oscillations experts marked?",
        data=("OpenNeuro ds003498: 20 patients, the first 60 s of each patient's first "
              "run, 41,187 expert-marked HFOs, scored on the channels the experts "
              "reviewed."),
        findings=(
            "Channel ranking against the experts': best channel-rank ρ 0.70 for ripples "
            "(RMS detector at 1.75 SD; the shipped 2.0 SD is within 0.02 of it) and 0.65 "
            "for fast ripples (RMS at 5 SD).",
            "Precision 0.53 and 0.76 at those thresholds. Read it as agreement, not "
            "accuracy: the reference is another detector's validated output, not a "
            "census of every oscillation.",
        ),
        limits=(
            "It does not beat a published detector and does not claim to.",
            "Each patient was scored on one minute of one night; rates and rankings "
            "from a single minute are noisy.",
        ),
        doc="docs/EVALUATION.md"),
    Method(
        key="outcome",
        title="Interictal HFOs and surgical outcome",
        question="Was the busiest channel inside the tissue whose removal stopped the "
                 "seizures?",
        data=("OpenNeuro ds003498: 20 patients after epilepsy surgery, 13 seizure-free and "
              "7 with recurrence, with the contacts that were resected."),
        findings=(
            "On the expert markings, the busiest fast-ripple channel was inside the "
            "resection in 85% of the seizure-free patients and 43% of the others: AUC "
            "0.71 (95% interval 0.49–0.92), permutation p = 0.12.",
            "With this program's RMS detector: AUC 0.71 (0.49–0.92), p = 0.12.",
            "That answer, inside the resection or not, was the same on every night "
            "recorded for 18 of 20 patients on the expert markings and 17 of 20 with the "
            "RMS detector: it is a property of the patient more than of the recording.",
            "On a second archive (OpenNeuro ds004100, HUP), 15 interictal recordings were "
            "sampled fast enough for ripples: the ripple rate ranked the onset zone above "
            "the other channels (median AUC 0.76, 0.55–0.82), and nothing separated the "
            "10 seizure-free patients from the 5 with recurrence.",
        ),
        limits=(
            "Neither comparison above reached significance, and none of the study's "
            "comparisons reaches p < 0.05 even before correction for multiple "
            "comparisons: read every result as hypothesis-generating.",
            "Thirteen against seven patients can detect only a very large effect: the "
            "smallest AUC it would find with 80% power is 0.85.",
            "The study's result describes groups of patients; it predicts nothing about "
            "one.",
        ),
        doc="docs/OUTCOME.md",
        extras=(
            ("figure", "outcome"),
            ("text", "**The busiest channel is often not one channel.** On the expert "
                     "markings the ranking singles out one channel in only 9 of 20 "
                     "patients, and in one patient 27 channels are tied with the busiest; "
                     "the RMS detector's leader is tied in 13 of 20. So the study also "
                     "asks what share of the tied channels was resected, which the figure "
                     "shows: AUC 0.73 (0.49–0.93), p = 0.08, on the expert markings and "
                     "0.70 (0.44–0.92), p = 0.13, with the detector."),
            ("figure", "window"),
            ("text", "**The result that did not hold up.** The first version of this "
                     "study analysed the first 60 s of each recording and found AUC 0.82, "
                     "p = 0.007. On the whole 300 s run the same code and the same metric "
                     "give 0.71, p = 0.12. Over five separate minutes of the same "
                     "recordings the AUC ranges from 0.57 to 0.82 and p from 0.007 to "
                     "0.61: the 0.007 was the best of five minutes. A window short enough "
                     "to change the conclusion is too short to report."),
        )),
    Method(
        key="ictal",
        title="Ictal onset: the Epileptogenicity Index",
        question="Does the index rank the clinicians' onset-zone channels first?",
        data=("OpenNeuro ds003029: 28 patients, up to three seizures each, against the onset "
              "zone in the archive's clinical summary; repeated on ds004100 (HUP)."),
        findings=(
            "Median per-patient AUC 0.80 (95% interval 0.70–0.87); the fast-over-slow "
            "energy ratio alone reaches 0.73.",
            "With the onset put 40 s before the seizure it falls to 0.52 over the 27 "
            "seizures where that could be tested (0.58 over their 12 patients), so the "
            "agreement comes from the seizure.",
            "On HUP, 54 patients: median AUC 0.79 (0.72–0.85), and 0.55 at 40 s before.",
        ),
        limits=(
            "Its single top channel was in the onset zone in only 15 of 28 patients.",
            "It needs a person to mark the electrographic onset, and a channel that "
            "changes first is not thereby the onset zone.",
        ),
        doc="docs/ICTAL.md",
        extras=(("figure", "ictal"),)),
    Method(
        key="template",
        title="Contacts placed on the template",
        question="How far does a planned contact land from the real one, and what is an "
                 "atlas label worth?",
        data=("OpenNeuro ds004100: 38 SEEG patients whose contacts were localised on their "
              "own imaging, moved to MNI152."),
        findings=(
            "A straight line from each shaft's deepest contact lands a median 2.5 mm off "
            "with the shaft's own spacing, 84% of contacts within 5 mm.",
            "At a 3.5 mm catalogue pitch the median is 10.1 mm.",
        ),
        limits=(
            "Even at the real positions the atlas names a structure for only 51% of "
            "contacts, so every label is a probable one.",
            "Template positions are not the patient's anatomy; the patient's own imaging "
            "is.",
        ),
        doc="docs/TEMPLATE_MAP.md",
        extras=(("figure", "template"),)),
)


def _method(method: Method) -> str:
    lines = [f"## {method.title}", "", f"*{method.question}*", "",
             f"**Data.** {method.data}", "", "**What was found**", ""]
    lines += [f"- {line}" for line in method.findings]
    lines.append("")
    for kind, value in method.extras:
        lines += (_figure(value) if kind == "figure" else [value, ""])
    lines += ["**Where it fails**", ""]
    lines += [f"- {line}" for line in method.limits]
    lines += ["", f"The full study: `{method.doc}`.", ""]
    return "\n".join(lines)


def _figure(key: str) -> list[str]:
    """A shipped figure and its caption; nothing when the picture is missing."""
    from onset_review.validationfigures import FIGURES, path_for

    path = path_for(key)
    if path is None:
        return []
    return [f"![{FIGURES[key]}]({path.as_posix()})", "", f"*{FIGURES[key]}*", ""]


def markdown(session=None) -> str:
    """The page: the headline, the open window when it can be scored, then
    every method."""
    from onset_review import HOMEPAGE

    parts = ["# Validation", "", HEADLINE, ""]
    here = this_recording(session)
    if here:
        parts += [here, ""]
    parts += [_method(m) for m in METHODS]
    parts += ["## Where to read more", "",
              "- `docs/LIMITATIONS.md`: what this program cannot do, in one place.",
              f"- [{HOMEPAGE}]({HOMEPAGE}): the results and documents, and the no-install "
              "demo with every study's tables, patients and charts.", ""]
    return "\n".join(parts)


def this_recording(session) -> str:
    """The open window scored against its expert markings, as the detector
    study scored each patient; "" when there is no window or no markings."""
    from onset_review import thisrecording
    from onset_review.studycharts import DETECTOR_LABELS

    if session is None or getattr(session, "raw", None) is None:
        return ""
    try:
        scores = thisrecording.score_window(session)
    except Exception:       # noqa: BLE001 - a section must not cost the page
        return ""
    if scores is None or not len(scores):
        return ""
    rows = ["| detector | threshold (SD) | channel-rank ρ | precision | recall | F1 |",
            "|---|---|---|---|---|---|"]
    for row in scores.itertuples():
        name = DETECTOR_LABELS.get(row.detector, row.detector)
        rows.append(f"| {name} | {row.threshold_sd:g} | {row.rank_rho:.2f} | "
                    f"{row.precision:.2f} | {row.recall:.2f} | {row.f1:.2f} |")
    first = scores.iloc[0]
    return "\n".join([
        "## This recording", "",
        f"This window carries expert markings, so its detections are scored as the "
        f"detector study scored each patient: {int(first['n_reviewed'])} reviewed "
        f"channels, {int(first['n_expert'])} marked events, {first['band']} band.", "",
        *rows, "", f"*{thisrecording.CAVEAT}*"])
