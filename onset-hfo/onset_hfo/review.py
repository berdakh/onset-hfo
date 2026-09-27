"""A hand-annotated benchmark: the sampling, the agreement, and the ceiling.

Roadmap item 5. The simulator can only measure what it simulates. "Precision
0.97 on synthetic ripples" is a statement about the generator; a few hundred
windows marked by two people on a real recording would make it a statement
about recordings.

**This module is the half that does not need people.** Marking is two
reviewers' afternoons and cannot be faked here; everything around it can be
built, and most of what makes such a benchmark worthless is in that
surrounding half rather than in the marking.

Three decisions do the work:

**1. Sample the recording, not the detector.** The obvious design -- show the
reviewers the detector's top candidates -- measures precision and *cannot
measure recall at all*, because a window the detector never proposed is never
looked at. Worse, it makes the reference a function of the thing being
scored. :func:`sample_windows` draws three strata: accepted candidates,
rejected candidates, and **windows with no detection in them**, the last
drawn uniformly from the recording. The third stratum is the only thing that
can ever reveal a false negative.

**2. Blind the reviewer.** The manifest deliberately does not carry the
detector's name, its verdict, its score, or the stratum. A reviewer who can
see that the machine said "ripple" is no longer an independent reference, and
the whole exercise is circular. :func:`sample_windows` checks its own output
for leakage before returning it.

**3. Report chance-corrected agreement.** Most windows are not ripples, so raw
percent agreement will look excellent whatever the reviewers do -- two people
who both say "no" to everything agree 90% of the time. :func:`agreement`
returns Cohen's kappa with a bootstrap interval, and the prevalence, so a
high kappa cannot be confused with a high agreement rate or vice versa.

**And the ceiling is the point.** Two reviewers who agree at kappa 0.6 have
defined a reference standard that no detector can be measured against more
finely than that. :func:`ceiling_note` says so in a sentence, because a
benchmark published without it invites a detector to be compared to a number
that is mostly noise.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.detectors.base import Event

__all__ = [
    "STRATA",
    "LABELS",
    "ReviewWindow",
    "sample_windows",
    "write_manifest",
    "read_manifest",
    "AnnotationLog",
    "agreement",
    "ceiling_note",
    "score_against_annotations",
]

#: Where each window came from. Kept in the *sampling* record, never in the
#: manifest a reviewer sees.
STRATA = ("accepted", "rejected", "background")

#: What a reviewer may say. ``unsure`` is a first-class answer: forcing a
#: binary choice on an ambiguous window manufactures agreement that is not
#: there, and the analysis can drop or keep them explicitly.
LABELS = ("hfo", "not_hfo", "unsure")


@dataclass(frozen=True)
class ReviewWindow:
    """One window to be looked at, with nothing in it that gives the game away."""

    window_id: str
    channel: str
    t_start: float
    t_stop: float
    #: Sampling stratum. Present on the object because the *analysis* needs it;
    #: :func:`write_manifest` refuses to write it to the reviewer's file.
    stratum: str = "background"
    #: The event this window was drawn around, when it was drawn around one.
    #: Also analysis-only.
    source_detector: str = ""
    source_accepted: bool | None = None


def sample_windows(events_by_detector: dict[str, list[Event]], duration_s: float,
                   channels: list[str], n_per_stratum: int = 70,
                   window_s: float = 0.4, seed: int = 0,
                   t_offset: float = 0.0) -> list[ReviewWindow]:
    """Draw a blinded, stratified sample of windows to review.

    ``n_per_stratum`` windows are drawn from each of accepted candidates,
    rejected candidates, and background -- the last uniformly over the
    recording, excluding anything within ``window_s`` of a candidate so that
    a "background" window really is one.

    The three strata are drawn in equal numbers **by design, not by
    prevalence**: the background class vastly outnumbers the others in a real
    recording, and sampling it proportionally would spend the reviewers'
    entire afternoon on empty windows. The imbalance is corrected in the
    scoring, not in the sampling, which is why the stratum is recorded.
    """
    rng = np.random.default_rng(seed)
    pooled: list[tuple[Event, str]] = []
    for detector, events in events_by_detector.items():
        for event in events:
            pooled.append((event, detector))

    # The background stratum is drawn from [t_offset, t_offset + duration_s].
    # If the candidates do not live there, that range is wrong, and the result
    # is a sample where a third of the windows point at signal the analysis
    # does not contain -- which costs a reviewer an afternoon and shows up as
    # a blank figure rather than an error. Caught exactly that way once.
    if pooled:
        starts = [e.start for e, _ in pooled]
        lo, hi = t_offset, t_offset + duration_s
        if min(starts) < lo - window_s or max(starts) > hi + window_s:
            raise ValueError(
                f"candidates span {min(starts):.1f}-{max(starts):.1f} s but the "
                f"background stratum would be drawn from {lo:.1f}-{hi:.1f} s. "
                f"Pass t_offset (the analysis's slice_start_s) so both strata "
                f"come from the same recording.")

    def _draw(candidates, stratum):
        if not candidates:
            return []
        idx = rng.choice(len(candidates), size=min(n_per_stratum, len(candidates)),
                         replace=False)
        out = []
        for i in sorted(int(j) for j in idx):
            event, detector = candidates[i]
            mid = 0.5 * (event.start + event.stop)
            out.append(ReviewWindow(
                window_id="", channel=event.channel,
                t_start=mid - window_s / 2, t_stop=mid + window_s / 2,
                stratum=stratum, source_detector=detector,
                source_accepted=bool(event.accepted)))
        return out

    windows = _draw([p for p in pooled if p[0].accepted], "accepted")
    windows += _draw([p for p in pooled if not p[0].accepted], "rejected")

    # Background: uniform in time and channel, and genuinely empty.
    taken: dict[str, list[tuple[float, float]]] = {}
    for event, _ in pooled:
        taken.setdefault(event.channel, []).append((event.start, event.stop))
    background: list[ReviewWindow] = []
    attempts = 0
    while len(background) < n_per_stratum and attempts < n_per_stratum * 200:
        attempts += 1
        channel = channels[int(rng.integers(len(channels)))]
        mid = float(rng.uniform(t_offset + window_s, t_offset + duration_s - window_s))
        if any(abs(mid - 0.5 * (a + b)) < window_s for a, b in taken.get(channel, [])):
            continue
        background.append(ReviewWindow(
            window_id="", channel=channel,
            t_start=mid - window_s / 2, t_stop=mid + window_s / 2,
            stratum="background"))
    windows += background

    # Shuffle, then number. The id encodes nothing: a reviewer who notices
    # that ids 1-70 are all detections has been unblinded by the filing.
    order = rng.permutation(len(windows))
    shuffled = [windows[int(i)] for i in order]
    numbered = [ReviewWindow(window_id=f"w{i:04d}", channel=w.channel,
                             t_start=w.t_start, t_stop=w.t_stop, stratum=w.stratum,
                             source_detector=w.source_detector,
                             source_accepted=w.source_accepted)
                for i, w in enumerate(shuffled)]
    _assert_blind(numbered)
    return numbered


def _assert_blind(windows: list[ReviewWindow]) -> None:
    """Refuse to hand over a sample whose order or numbering leaks the stratum."""
    strata = [w.stratum for w in windows]
    if len(set(strata)) < 2:
        return
    runs = sum(1 for a, b in zip(strata, strata[1:], strict=False) if a != b)
    # A perfectly sorted sample has len(set)-1 changes. Anything near that is
    # not shuffled, whatever the shuffle was supposed to do.
    if runs <= len(set(strata)):
        raise RuntimeError(
            "the sample is ordered by stratum, so its numbering tells the "
            "reviewer which windows the detector proposed")


# --------------------------------------------------------------------------
# The reviewer's files
# --------------------------------------------------------------------------

#: The only columns a reviewer's manifest may contain.
MANIFEST_COLUMNS = ("window_id", "channel", "t_start", "t_stop")


def write_manifest(windows: list[ReviewWindow], path: str | Path) -> Path:
    """Write the blinded manifest, plus a sibling key the reviewer must not open.

    Two files on purpose. ``<path>`` is what a reviewer is given and carries
    four columns and nothing else. ``<path>.key.csv`` holds the stratum and
    provenance for the analysis, and is named so that handing over the wrong
    one is a visible mistake rather than a silent one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{c: getattr(w, c) for c in MANIFEST_COLUMNS} for w in windows])
    frame.to_csv(path, index=False)

    key = pd.DataFrame([{
        "window_id": w.window_id, "stratum": w.stratum,
        "source_detector": w.source_detector, "source_accepted": w.source_accepted,
    } for w in windows])
    key.to_csv(path.with_suffix(".key.csv"), index=False)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest, refusing one that carries more than it should."""
    frame = pd.read_csv(path)
    extra = set(frame.columns) - set(MANIFEST_COLUMNS)
    if extra:
        raise ValueError(
            f"manifest carries {sorted(extra)}, which would unblind the reviewer")
    return frame


@dataclass
class AnnotationLog:
    """Append-only annotations, resumable across sessions.

    Append-only because four hundred windows is several sittings, and a tool
    that rewrites its output file loses an afternoon to one crash. Resuming
    means the reviewer is shown what they have not yet marked, in the
    manifest's order, which is already shuffled.
    """

    path: Path
    reviewer: str
    _seen: dict[str, str] = field(default_factory=dict)

    FIELDS = ("window_id", "reviewer", "label", "note")

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for row in csv.DictReader(self.path.open()):
                if row.get("reviewer") == self.reviewer:
                    self._seen[row["window_id"]] = row["label"]
        else:
            with self.path.open("w", newline="") as fh:
                csv.writer(fh).writerow(self.FIELDS)

    def record(self, window_id: str, label: str, note: str = "") -> None:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, not {label!r}")
        with self.path.open("a", newline="") as fh:
            csv.writer(fh).writerow([window_id, self.reviewer, label, note])
        self._seen[window_id] = label

    def remaining(self, manifest: pd.DataFrame) -> list[str]:
        """Window ids this reviewer has not marked, in the manifest's order."""
        return [w for w in manifest["window_id"] if w not in self._seen]

    @property
    def done(self) -> int:
        return len(self._seen)

    def frame(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(
            columns=list(self.FIELDS))


# --------------------------------------------------------------------------
# Agreement, and the ceiling it sets
# --------------------------------------------------------------------------


def agreement(annotations: pd.DataFrame, reviewers: tuple[str, str] | None = None,
              drop_unsure: bool = True, n_boot: int = 2000,
              seed: int = 0) -> dict:
    """Cohen's kappa between two reviewers, with a bootstrap interval.

    Chance-corrected, because raw percent agreement is inflated by prevalence
    and prevalence here is extreme: two reviewers who call everything
    ``not_hfo`` agree on nine windows in ten and have agreed about nothing.
    Both numbers are returned so neither can stand in for the other.

    ``drop_unsure`` removes windows either reviewer marked ``unsure``, and the
    count of those is returned: a pair who are unsure about a third of the
    sample have told you something the kappa on the remainder will not.
    """
    if reviewers is None:
        names = sorted(annotations["reviewer"].unique())
        if len(names) != 2:
            raise ValueError(f"expected exactly two reviewers, found {names}")
        reviewers = (names[0], names[1])

    wide = annotations.pivot_table(index="window_id", columns="reviewer",
                                   values="label", aggfunc="last")
    missing = [r for r in reviewers if r not in wide.columns]
    if missing:
        raise ValueError(f"no annotations from {missing}")
    pair = wide[list(reviewers)].dropna()
    n_total = len(pair)
    unsure = int((pair == "unsure").any(axis=1).sum())
    if drop_unsure:
        pair = pair[~(pair == "unsure").any(axis=1)]

    a = pair[reviewers[0]].to_numpy()
    b = pair[reviewers[1]].to_numpy()
    out = {
        "reviewers": list(reviewers),
        "n_common": int(n_total),
        "n_scored": int(len(pair)),
        "n_unsure_dropped": unsure if drop_unsure else 0,
        "raw_agreement": float(np.mean(a == b)) if len(a) else float("nan"),
        "prevalence_hfo": float(np.mean(np.concatenate([a, b]) == "hfo")) if len(a) else float("nan"),
        "kappa": float("nan"), "kappa_lo": float("nan"), "kappa_hi": float("nan"),
    }
    if len(a) < 2:
        return out

    out["kappa"] = _kappa(a, b)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        boots[i] = _kappa(a[idx], b[idx])
    finite = boots[np.isfinite(boots)]
    if len(finite):
        out["kappa_lo"] = float(np.percentile(finite, 2.5))
        out["kappa_hi"] = float(np.percentile(finite, 97.5))
    return out


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa. NaN when both reviewers used one label -- undefined, not 1."""
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return float("nan")
    observed = float(np.mean(a == b))
    expected = sum(np.mean(a == label) * np.mean(b == label) for label in labels)
    if expected >= 1.0:
        return float("nan")
    return float((observed - expected) / (1.0 - expected))


def ceiling_note(stats: dict) -> str:
    """One sentence a benchmark must not be published without."""
    kappa = stats.get("kappa", float("nan"))
    if not np.isfinite(kappa):
        return ("The two reviewers used a single label between them, so their "
                "agreement is undefined and this sample sets no ceiling at all.")
    strength = ("slight" if kappa < 0.21 else "fair" if kappa < 0.41 else
                "moderate" if kappa < 0.61 else "substantial" if kappa < 0.81
                else "almost perfect")
    # Name whichever label is actually the common one. Getting this backwards
    # -- crediting the rare class with inflating raw agreement -- was the
    # first version of this sentence.
    share_hfo = stats.get("prevalence_hfo", float("nan"))
    commoner, share = (("an HFO", share_hfo) if share_hfo >= 0.5
                       else ("not an HFO", 1.0 - share_hfo))
    return (
        f"Inter-rater agreement is kappa = {kappa:.2f} "
        f"({stats.get('kappa_lo', float('nan')):.2f}–{stats.get('kappa_hi', float('nan')):.2f}, "
        f"{strength}) on {stats.get('n_scored', 0)} windows, against a raw "
        f"agreement of {stats.get('raw_agreement', float('nan')):.0%} that the "
        f"{share:.0%} prevalence of '{commoner}' would largely have produced on "
        f"its own. **No detector can be scored against this reference more "
        f"precisely than the reviewers agree with each other**; read every "
        f"number below against that ceiling.")


def score_against_annotations(events: list[Event], manifest: pd.DataFrame,
                              annotations: pd.DataFrame, key: pd.DataFrame,
                              consensus: str = "both", detector: str | None = None) -> dict:
    """Score a detector against the marked windows, stratum by stratum.

    ``consensus`` is ``"both"`` (a window counts as an HFO only where the
    reviewers agree it is one) or ``"either"``. Both are reported in practice
    because they bracket the answer, and quoting whichever is kinder is how a
    detector's precision gets published.

    Rates are reported **per stratum and not pooled**, because the strata were
    sampled in equal numbers rather than in proportion: pooling them would
    weight a background window as heavily as a candidate and describe a
    recording that does not exist.
    """
    labelled = annotations.pivot_table(index="window_id", columns="reviewer",
                                       values="label", aggfunc="last")
    if consensus == "both":
        truth = labelled.apply(lambda r: bool((r == "hfo").all() and r.notna().all()), axis=1)
    elif consensus == "either":
        truth = labelled.apply(lambda r: bool((r == "hfo").any()), axis=1)
    else:
        raise ValueError("consensus must be 'both' or 'either'")

    picked = [e for e in events if detector is None or e.detector == detector]
    frame = manifest.merge(key, on="window_id", how="left")
    frame["is_hfo"] = frame["window_id"].map(truth).fillna(False).astype(bool)
    frame["detected"] = [
        any(e.channel == row.channel and e.accepted
            and row.t_start <= 0.5 * (e.start + e.stop) < row.t_stop
            for e in picked)
        for row in frame.itertuples()]
    frame = frame[frame["window_id"].isin(truth.index)]

    out: dict = {"consensus": consensus, "n_windows": int(len(frame)), "strata": {}}
    for stratum, part in frame.groupby("stratum"):
        tp = int((part["is_hfo"] & part["detected"]).sum())
        fp = int((~part["is_hfo"] & part["detected"]).sum())
        fn = int((part["is_hfo"] & ~part["detected"]).sum())
        tn = int((~part["is_hfo"] & ~part["detected"]).sum())
        out["strata"][str(stratum)] = {
            "n": int(len(part)), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "reviewer_hfo_rate": float(part["is_hfo"].mean()),
        }
    return out


def save_study(directory: str | Path, stats: dict, scores: dict) -> Path:
    """Write the agreement and the scores, with the ceiling sentence on top."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(
        {"ceiling": ceiling_note(stats), **stats}, indent=2))
    (out / "scores.json").write_text(json.dumps(scores, indent=2))
    return out
