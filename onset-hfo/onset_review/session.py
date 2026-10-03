"""A review session: the recording, the detections, and the findings table.

This module is the whole product minus the windows. It holds no Qt import on
purpose, so everything a reviewer is shown can be computed and tested without a
display -- which is also what lets the test suite check the numbers the
interface puts in front of a clinician.

The division of labour matches the rest of the project: this file decides what
is true and `onset_review.window` decides how it looks. A panel that wants a
number asks for it here; nothing downstream recomputes a rate.

Nothing here is new analysis. Every number comes from the same functions the
command line, the report and the Streamlit app use -- `prepare`, the detector
registry, `channel_rates`, `leader_separation`, `candidate_channels` -- so a
rate read off the screen is the rate the paper quotes. That constraint is the
reason this module is thin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from onset_hfo.config import BANDS, DATA_CACHE, PipelineConfig
from onset_hfo.datasets import Recording, fetch_slice
from onset_hfo.detectors import DETECTORS, HFO_DETECTORS
from onset_hfo.detectors.base import Event
from onset_hfo.metrics import channel_rates, leader_separation, rank_channels
from onset_hfo.outcome import candidate_channels
from onset_hfo.preprocess import prepare
from onset_hfo.validate import validate_events

__all__ = ["ReviewRequest", "ReviewSession", "load_session",
           "session_from_recording", "annotations_for",
           "expert_annotations", "cached_windows", "BAND_COLOURS",
           "DETECTOR_LABELS"]

#: Annotation colours. The two HFO bands separate at a glance and discharges
#: read as a different kind of thing rather than a third band.
#:
#: The expert entries get their own hues rather than paler versions of ours,
#: which was the first attempt. Overlaid on a trace at MNE's annotation alpha a
#: pale blue and a blue are the same mark, and the one question the overlay
#: exists to answer -- did the annotator mark this one too -- becomes
#: unanswerable by eye. Telling the *source* apart matters more here than
#: telling the band apart, because every mark already carries its band in its
#: own label text.
BAND_COLOURS = {
    "ripple": "#1f77b4",              # blue
    "fast ripple": "#d62728",         # red
    "spike": "#7f7f7f",               # grey
    "expert ripple": "#2ca02c",       # green
    "expert fast ripple": "#ff7f0e",  # orange
}

#: Detector names as a clinician should read them, with the threshold the
#: project measured for each. Used by the launcher's menu so the interface
#: never shows a bare registry key.
DETECTOR_LABELS = {
    "rms": "Root-mean-square energy",
    "line_length": "Line length",
    "hilbert": "Hilbert envelope",
    "short_time_energy": "Short-time energy",
}


@dataclass(frozen=True)
class ReviewRequest:
    """Everything the reviewer chose before any signal was read.

    Kept as a value object so a session can be reproduced from a report: every
    field here appears in the exported header, and re-running with the same
    request gives the same screen.
    """

    dataset: str = "ds003498"
    subject: str = "sub-01"
    run: str = "01"
    task: str | None = None
    t_start: float = 0.0
    t_stop: float = 60.0
    detectors: tuple[str, ...] = ("rms",)
    band: str = "ripple"
    #: Detection threshold in robust SDs, or None to use each detector's own
    #: measured default. Not a shared number: see `_detect`.
    threshold_sd: float | None = None
    with_spikes: bool = True

    def __post_init__(self) -> None:
        unknown = [d for d in self.detectors if d not in HFO_DETECTORS]
        if unknown:
            raise ValueError(
                f"unknown detector(s) {unknown}; choose from "
                f"{sorted(HFO_DETECTORS)}")
        if not self.detectors:
            raise ValueError("a review needs at least one detector")
        if not hasattr(BANDS, self.band):
            raise ValueError(f"unknown band {self.band!r}; "
                             "choose 'ripple' or 'fast_ripple'")
        if self.t_stop <= self.t_start:
            raise ValueError(f"window ends at or before it starts "
                             f"({self.t_start:g}–{self.t_stop:g} s)")

    @property
    def duration(self) -> float:
        return max(0.0, self.t_stop - self.t_start)

    @property
    def primary(self) -> str:
        """The detector the findings table and the ranking are built from."""
        return self.detectors[0]

    @property
    def band_hz(self) -> tuple[float, float]:
        return getattr(BANDS, self.band)

    def band_label(self) -> str:
        low, high = self.band_hz
        return f"{self.band.replace('_', ' ')} ({low:.0f}–{high:.0f} Hz)"

    def label(self) -> str:
        return (f"{self.dataset} · {self.subject} · run-{self.run} · "
                f"{self.t_start:g}–{self.t_stop:g} s")

    def pipeline_config(self) -> PipelineConfig:
        """The project's `PipelineConfig` that reproduces this review exactly.

        One function, so that the detection the reviewer is looking at and any
        later re-run of the same window are the same analysis rather than two
        that agree by inspection. The assistant needs this: it answers from a
        saved pipeline result, and a result built with the default band while
        the screen shows the fast-ripple one would have it quoting numbers that
        are nowhere on the display.

        Each detector keeps its own measured threshold unless the reviewer set
        one, because they are not interchangeable -- line length runs at 3.0 SD
        and the energy detectors at 2.0.
        """
        cfg = PipelineConfig()
        for name in HFO_DETECTORS:
            detector_cfg = replace(getattr(cfg, name), band=self.band_hz)
            if self.threshold_sd is not None:
                detector_cfg = replace(detector_cfg,
                                       threshold_sd=float(self.threshold_sd))
            setattr(cfg, name, detector_cfg)
        return cfg


@dataclass
class ReviewSession:
    """A loaded window, its detections, and everything a panel needs to show."""

    request: ReviewRequest
    raw: object                       #: mne.io.RawArray, volts, as MNE wants
    events: list[Event]
    findings: pd.DataFrame
    leader: dict
    candidates: list[str]
    expert: list[Event] = field(default_factory=list)
    reviewed_channels: list[str] = field(default_factory=list)
    #: The surgeon's resected zone for this subject, when the dataset ships a
    #: clinical sheet. `None` means unknown, which is not the same as "nothing
    #: was removed" and is why the 3D view greys those contacts rather than
    #: colouring them "spared".
    resection: object | None = None
    #: Measured electrode coordinates from the dataset's `electrodes.tsv`.
    #: Neither archive here has one; see `onset_review.anatomy`.
    electrodes: pd.DataFrame | None = None
    #: The fetched slice itself. Kept so the assistant can run the project's
    #: own pipeline over the very window on screen rather than answering from
    #: a differently-configured one; the signal is already in memory as `raw`,
    #: so holding it costs nothing.
    recording: object | None = None
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    citation: str = ""
    sfreq: float = 0.0
    t_offset: float = 0.0
    montage: str = "bipolar"

    @property
    def accepted(self) -> list[Event]:
        return [e for e in self.events if e.accepted]

    @property
    def hfo_events(self) -> list[Event]:
        """Accepted band events only -- discharges are a different question."""
        return [e for e in self.accepted if e.detector != "spike"]

    @property
    def spikes(self) -> list[Event]:
        return [e for e in self.accepted if e.detector == "spike"]

    @property
    def duration_min(self) -> float:
        return self.request.duration / 60.0

    @property
    def has_expert(self) -> bool:
        return bool(self.expert)

    def events_of(self, detector: str) -> list[Event]:
        return [e for e in self.accepted if e.detector == detector]

    def summary(self) -> dict:
        """The handful of numbers the title bar and the report header carry."""
        return {
            "window": self.request.label(),
            "band": self.request.band_label(),
            "channels": int(len(self.findings)),
            "montage": self.montage,
            "detected": len(self.events),
            "accepted": len(self.accepted),
            "rejected": len(self.events) - len(self.accepted),
            "spikes": len(self.spikes),
            "leader": self.leader.get("leader"),
            "leader_distinguishable": bool(self.leader.get("distinguishable", False)),
            "n_candidates": len(self.candidates),
            "expert_events": len(self.expert),
            "n_reviewed": len(self.reviewed_channels),
        }

    def caveat(self) -> str:
        """The one sentence that has to be on screen whatever else is.

        A ranked list of channels is the output a clinician will act on, and it
        is exactly the output that looks identical whether or not the data
        supports it. `leader_separation` is the project's answer to that, so it
        is promoted to a permanent element of the interface rather than a
        column someone might scroll past.
        """
        if not self.leader.get("available"):
            return "No channel ranking: nothing was detected in this window."
        if self.leader.get("distinguishable"):
            return (f"{self.leader['leader']} stands out "
                    f"({self.leader['leader_rate_per_min']:g}/min, interval "
                    f"{self.leader['leader_ci'][0]:g}–{self.leader['leader_ci'][1]:g}); "
                    f"{len(self.candidates)} channel(s) cannot be told apart from it.")
        return ("NO CHANNEL STANDS OUT in this window: the busiest channel's "
                "interval overlaps the median channel's, so this ordering is "
                "counting noise. Review a longer window before reading it as a "
                "ranking.")


def cached_windows(cache_dir: Path | None = None) -> pd.DataFrame:
    """Every window already on disk, read from the cache's own metadata.

    The launcher offers these and nothing else when running offline, which is
    the default. That is not a limitation dressed up as a feature: a reviewer
    opening the software on a hospital machine should not discover that the
    thing they clicked needs 700 MB from an archive they cannot reach, and a
    window that is present answers in seconds instead of minutes.

    One row per cached slice, with the sampling rate, so the launcher can grey
    out the fast-ripple band on a 1000 Hz recording rather than letting a
    reviewer choose a band that cannot honestly be analysed.
    """
    root = Path(cache_dir or DATA_CACHE)
    rows = []
    for meta_path in sorted(root.glob("*/*/slice.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        sfreq = float(meta.get("sfreq") or 0.0)
        rows.append({
            "dataset": str(meta.get("dataset") or meta_path.parents[1].name),
            "subject": str(meta.get("subject") or ""),
            "task": meta.get("task"),
            "run": str(meta.get("run") or "01"),
            "t_start": float(meta.get("t_start") or 0.0),
            "t_stop": float(meta.get("t_stop") or 0.0),
            "sfreq": sfreq,
            "n_channels": int(meta.get("n_channels") or 0),
            "bands": ", ".join(name for name in ("ripple", "fast_ripple")
                               if BANDS.usable(sfreq, getattr(BANDS, name))),
            "path": str(meta_path.parent),
        })
    frame = pd.DataFrame(rows, columns=["dataset", "subject", "task", "run",
                                        "t_start", "t_stop", "sfreq",
                                        "n_channels", "bands", "path"])
    if frame.empty:
        return frame
    frame["duration"] = frame["t_stop"] - frame["t_start"]
    return (frame.sort_values(["dataset", "subject", "run", "t_start"])
            .reset_index(drop=True))


def _to_raw(data_uv: np.ndarray, ch_names: list[str], sfreq: float):
    """Wrap the prepared array as an MNE Raw, converting microvolts to volts.

    The conversion is the one unit change in this file and it is explicit,
    because MNE scales its own axes from SI and a silent factor of 1e6 is how a
    trace ends up unreadable at a plausible-looking gain.
    """
    import mne

    info = mne.create_info(list(ch_names), float(sfreq), ch_types="seeg")
    return mne.io.RawArray(np.asarray(data_uv, dtype=float) / 1e6, info,
                           verbose="ERROR")


def _label_for(event: Event) -> str:
    """The annotation text a reviewer reads on the trace.

    Named for the band rather than the detector, because a clinician asks "is
    that a fast ripple" and not "did line length fire"; the detector is in the
    findings table and the report for anyone who needs it.
    """
    if event.detector == "spike":
        return "spike"
    low = float(event.band[0]) if event.band else 0.0
    return "fast ripple" if low >= 200.0 else "ripple"


def annotations_for(events: list[Event], t_offset: float = 0.0,
                    accepted_only: bool = True, prefix: str = ""):
    """Detections as MNE annotations, in the Raw's own time base.

    Event times are archive-file seconds so a window quoted in a report can be
    found again; the Raw starts at zero, so the offset is subtracted here and
    nowhere else.

    `prefix` is how the expert overlay gets its own description and therefore
    its own colour -- MNE keys annotation colours off the description string,
    so two sources sharing a band have to differ in the text.
    """
    import mne

    onsets, durations, labels, channels = [], [], [], []
    for event in events:
        if accepted_only and not event.accepted:
            continue
        onsets.append(max(0.0, float(event.start) - float(t_offset)))
        durations.append(max(0.0, float(event.stop) - float(event.start)))
        labels.append(f"{prefix}{_label_for(event)}" if prefix else _label_for(event))
        channels.append((event.channel,) if event.channel else ())
    return mne.Annotations(onset=onsets, duration=durations, description=labels,
                           ch_names=channels)


def expert_annotations(session: ReviewSession):
    """The annotators' own markings, labelled so they read as a second source.

    This is the single most useful thing the interface can show someone getting
    familiar with HFO review: the same minute of signal with the detector's
    marks and the Zurich annotators' marks side by side. Where they agree the
    reviewer learns what the detector is for; where they do not, they learn what
    it costs.
    """
    return annotations_for(session.expert, session.t_offset,
                           accepted_only=False, prefix="expert ")


def _expert_events(record: Recording, band: str) -> list[Event]:
    """Expert markings for one band, as `Event`s on the bipolar channels.

    The archive marks HFOs per bipolar channel already, which is why these can
    be compared to our detections without any re-referencing. Only the kind
    matching the reviewed band is returned, so the overlay answers the same
    question as the trace it sits on.
    """
    frame = getattr(record, "ground_truth", None)
    needed = {"kind", "channel", "start", "stop"}
    if frame is None or not len(frame) or not needed <= set(frame.columns):
        # Only markings already made per bipolar channel qualify, which in
        # practice means ds003498's. The synthetic recording's implanted truth
        # names a *contact*, and a contact sits on two bipolar channels, so
        # spreading it across both would turn one implanted ripple into two
        # expert marks and quietly halve every precision in the agreement
        # table. A recording whose truth is not channel-level simply has no
        # overlay, which the interface says plainly.
        return []
    wanted = "fast_ripple" if band == "fast_ripple" else "ripple"
    rows = frame[frame["kind"].astype(str) == wanted]
    band_hz = getattr(BANDS, band)
    return [Event(channel=str(row["channel"]),
                  start=float(row["start"]) + float(record.t_offset),
                  stop=float(row["stop"]) + float(record.t_offset),
                  detector="expert", band=band_hz, accepted=True)
            for _, row in rows.iterrows()]


def _resection_for(record: Recording):
    """The surgeon's resected contacts, when the dataset has a clinical sheet.

    Guarded completely. Only `ds003498` ships one, it is a separate (cached)
    download, and a review window that refused to open because a surgical
    sidecar was missing would be useless on every other recording. `None` means
    "not known", which the interface says rather than silently drawing every
    contact as spared.
    """
    subject = str(getattr(record, "subject", "") or "")
    dataset = str(getattr(record, "dataset_id", "") or "")
    if not subject or not dataset:
        return None
    try:
        from onset_hfo.clinical import resection_map

        return resection_map(dataset).get(subject)
    except Exception:
        return None


def _electrodes_for(record: Recording):
    """Measured contact coordinates, if the recording carries any.

    Written against the attribute rather than the archive because no dataset
    this project reads has an `electrodes.tsv`; this is the seam a site pointing
    the software at its own BIDS data would come in through, and it is tested
    with a constructed table so the path does not rot untried.
    """
    frame = getattr(record, "electrodes", None)
    if frame is None or not len(frame):
        return None
    return frame


def _detect(prep, cfg: PipelineConfig, name: str) -> list[Event]:
    """Run one detector, reading its settings off the request's own config.

    Nothing is decided here: `ReviewRequest.pipeline_config` already wrote the
    band and any threshold override onto each detector's entry, which is what
    makes this review and a `run_pipeline` over the same request the same
    analysis rather than two implementations of it.
    """
    found = DETECTORS[name](prep, getattr(cfg, name))
    validate_events(found, prep, cfg.validation)
    return list(found)


def _findings_table(events: list[Event], channels: list[str], duration_s: float,
                    detector: str, reviewed: list[str],
                    expert: list[Event]) -> pd.DataFrame:
    """Per-channel rates, ranked, with the two columns a reviewer needs beside.

    `channel_rates` already returns counts, rates and the Poisson interval that
    says whether to believe them; this adds the rank, whether the channel is one
    the archive's annotators actually looked at, and their own rate on it. The
    `reviewed` column matters more than it looks: in `ds003498` only a subset of
    channels was annotated, so a detection elsewhere is unjudged rather than
    wrong, and a reviewer comparing the two sources has to be able to see which
    rows the comparison is even defined on.
    """
    table = channel_rates([e for e in events if e.detector == detector],
                          duration_s, channels=channels)
    if table.empty:
        return table
    table = rank_channels(table)
    reviewed_set = set(reviewed)
    table["reviewed"] = [ch in reviewed_set for ch in table["channel"]]
    minutes = max(duration_s / 60.0, 1e-9)
    expert_counts: dict[str, int] = {}
    for event in expert:
        expert_counts[event.channel] = expert_counts.get(event.channel, 0) + 1
    if "n_with_spike" in table.columns:
        table["n_with_spike"] = table["n_with_spike"].astype(int)
    table["expert_n"] = [expert_counts.get(ch, 0) for ch in table["channel"]]
    table["expert_rate_per_min"] = table["expert_n"] / minutes
    if not expert:
        table = table.drop(columns=["expert_n", "expert_rate_per_min"])
    return table.reset_index(drop=True)


def load_session(request: ReviewRequest, cache_dir: Path | None = None,
                 progress=None) -> ReviewSession:
    """Fetch the window the request names, then analyse it.

    `progress` is an optional callable taking (fraction, message); the launcher
    passes one so the reviewer sees which stage is running rather than a frozen
    window. It is called from whatever thread this runs on, so a Qt caller must
    marshal it rather than touching widgets directly.
    """
    def say(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    say(0.05, f"Fetching {request.subject} {request.t_start:g}–{request.t_stop:g} s")
    record = fetch_slice(dataset=request.dataset, subject=request.subject,
                         run=request.run, task=request.task,
                         t_start=request.t_start, t_stop=request.t_stop,
                         cache_dir=cache_dir, verbose=False)
    return session_from_recording(record, request, progress=progress)


def session_from_recording(record: Recording, request: ReviewRequest,
                           progress=None) -> ReviewSession:
    """Everything after the signal is in hand: preprocess, detect, measure.

    Split from `load_session` so the product can be exercised end to end on a
    recording that did not come from an archive -- the synthetic one the test
    suite builds, or a file a site loaded itself. The alternative, a test that
    needs 60 s of `ds003498` on disk, would mean the interface's numbers are
    only ever checked on a developer's machine.
    """
    def say(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    say(0.3, "Preprocessing: high-pass, notch, bipolar montage")
    cfg = request.pipeline_config()
    prep = prepare(record, cfg.preprocess, verbose=False)

    band = request.band_hz
    if not BANDS.usable(prep.sfreq, band):
        raise ValueError(
            f"{request.band.replace('_', ' ')}s need a sampling rate above "
            f"{2 * band[1]:.0f} Hz; this recording is {prep.sfreq:.0f} Hz. "
            f"Choose the ripple band, or a recording sampled higher.")

    events: list[Event] = []
    span = 0.3 / len(request.detectors)
    for index, name in enumerate(request.detectors):
        say(0.35 + span * index,
            f"Detecting with {DETECTOR_LABELS.get(name, name).lower()}")
        events.extend(_detect(prep, cfg, name))

    if request.with_spikes:
        say(0.68, "Detecting interictal discharges")
        events.extend(DETECTORS["spike"](prep, cfg.spikes))

    say(0.78, "Reading the expert markings")
    expert = _expert_events(record, request.band)
    resection = _resection_for(record)
    electrodes = _electrodes_for(record)
    reviewed = [c for c in getattr(record, "reviewed_channels", [])
                if c in set(prep.ch_names)]

    say(0.85, "Measuring rates and confidence intervals")
    findings = _findings_table(events, list(prep.ch_names), prep.duration,
                               request.primary, reviewed, expert)
    leader = leader_separation(findings) if not findings.empty else {}
    counts = (findings.set_index("channel")["n_events"]
              if not findings.empty else pd.Series(dtype=float))
    tied = (candidate_channels(counts, max(prep.duration / 60.0, 1e-9))
            if len(counts) else [])

    say(0.95, "Building the trace")
    raw = _to_raw(prep.data, list(prep.ch_names), prep.sfreq)
    raw.set_annotations(annotations_for(events, prep.t_offset))

    say(1.0, "Ready")
    return ReviewSession(
        request=request, raw=raw, events=events, findings=findings,
        leader=leader, candidates=list(tied), expert=expert,
        reviewed_channels=reviewed, resection=resection, electrodes=electrodes,
        steps=list(prep.steps),
        notes=list(getattr(record, "notes", [])),
        citation=str(getattr(record, "citation", "")),
        sfreq=float(prep.sfreq), t_offset=float(prep.t_offset),
        montage=prep.montage, recording=record,
    )
