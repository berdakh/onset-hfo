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

from onset_hfo.config import BANDS, DATA_CACHE, PipelineConfig, PreprocessConfig
from onset_hfo.datasets import Recording, fetch_slice
from onset_hfo.detectors import DETECTORS, HFO_DETECTORS
from onset_hfo.detectors.base import Event
from onset_hfo.metrics import channel_rates, leader_separation, rank_channels
from onset_hfo.outcome import candidate_channels
from onset_hfo.preprocess import prepare
from onset_hfo.quality import (
    analysable_seconds,
    channel_quality,
    quality_summary,
    reject_unusable_events,
    segment_quality,
)
from onset_hfo.validate import validate_events
from onset_review.adjudication import Adjudication

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
    #: What is done to the signal before any detector sees it. Part of the
    #: request because the request is everything the reviewer chose before the
    #: signal was read, and a filter choice is exactly that -- it changes every
    #: number downstream, so a session reproduced from a report has to carry it.
    #: `None` means the project's measured defaults.
    preprocess: PreprocessConfig | None = None
    #: A local file to read instead of the archive. When set, `dataset` is
    #: ignored and `onset_hfo.io` picks the reader from the extension.
    path: Path | None = None
    #: The reviewer's correction to the file's channel types, which on most
    #: clinical exports is not optional: an EDF typically declares every
    #: channel `eeg`, and the pipeline would analyse a scalp montage as
    #: intracranial. Only consulted for an imported file.
    channel_types: tuple[tuple[str, str], ...] = ()
    #: Mains frequency for an imported file. The archive carries its own; a
    #: bare EDF does not, and the wrong one leaves the interference in place
    #: *and* carves a hole where there was none.
    line_freq: float = 50.0
    #: Run the data-quality stage: which contacts and which seconds are fit to
    #: analyse. On by default. Off reproduces the numbers this project
    #: measured before the stage existed, which is why it is a switch.
    check_quality: bool = True
    #: Channels the reviewer reinstated after looking at them, overruling the
    #: automatic verdict. One-directional on purpose: excluding a channel the
    #: checks passed is `PreprocessConfig.exclude`, where it is recorded as
    #: the reviewer's own choice rather than as an override of a verdict.
    keep_channels: tuple[str, ...] = ()
    #: What to *analyse*, when that is more than the trace shows, in
    #: original-recording seconds. `None` means the two are the same, which is
    #: how this software worked until a span could be longer than memory.
    #: Above `onset_hfo.longrun.STREAM_ABOVE_S` the analysis is streamed in
    #: chunks and only `t_start`–`t_stop` is held as signal, so a reviewer can
    #: rank ten minutes of contacts and look at one of them.
    #:
    #: Absolute rather than a length from `t_start`, because the trace window
    #: moves inside the span -- clicking an event eight minutes in loads that
    #: minute -- and a span defined relative to the trace would slide along
    #: with it, which would mean every rate quietly changed when the reviewer
    #: scrolled.
    span_start: float | None = None
    span_stop: float | None = None
    #: A coordinate file the reviewer pointed the software at, when the
    #: recording itself carries none. Carried on the request so that it
    #: survives a re-analysis: a filter change does not move an electrode.
    electrodes_path: Path | None = None

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

    def span(self) -> tuple[float, float]:
        """What is analysed, which is at least what is shown."""
        if self.span_start is None or self.span_stop is None:
            return float(self.t_start), float(self.t_stop)
        start, stop = float(self.span_start), float(self.span_stop)
        if stop - start <= self.duration:
            return float(self.t_start), float(self.t_stop)
        return start, stop

    @property
    def span_duration(self) -> float:
        start, stop = self.span()
        return stop - start

    @property
    def streamed(self) -> bool:
        """Whether this request needs the chunked path to be analysable."""
        from onset_hfo.longrun import STREAM_ABOVE_S

        return self.span_duration > STREAM_ABOVE_S

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

    def preprocess_label(self) -> str:
        """What will be done to the signal, in one line, for a header.

        Built from the config rather than from the steps the pipeline logs,
        because a header has to say what was *asked for* before a recording is
        in hand; `Prepared.steps` says what was actually done, and both appear
        in the report.
        """
        cfg = self.preprocess or PreprocessConfig()
        parts = []
        if cfg.highpass:
            parts.append(f"high-pass {cfg.highpass:g} Hz")
        if cfg.lowpass:
            parts.append(f"low-pass {cfg.lowpass:g} Hz")
        if cfg.notch:
            mains = f"{cfg.line_freq:g} Hz" if cfg.line_freq else "mains"
            parts.append(f"notch {mains}"
                         + (" + harmonics" if cfg.notch_harmonics else "")
                         + f" ({cfg.notch_width:g} Hz wide)")
        if cfg.resample:
            parts.append(f"resample to {cfg.resample:g} Hz")
        parts.append("bipolar montage" if cfg.bipolar
                     else "common average reference" if cfg.average_reference
                     else "no re-referencing")
        if cfg.exclude:
            parts.append(f"{len(cfg.exclude)} channel(s) excluded by the reviewer")
        if cfg.regress_channels:
            parts.append(f"{', '.join(cfg.regress_channels)} regressed out")
        if cfg.ica:
            parts.append("ICA" + (f" (components {', '.join(str(i) for i in cfg.ica_exclude)} "
                                  "removed)" if cfg.ica_exclude else " (nothing removed)"))
        return ", ".join(parts) if parts else "none"

    def band_label(self) -> str:
        low, high = self.band_hz
        return f"{self.band.replace('_', ' ')} ({low:.0f}–{high:.0f} Hz)"

    @property
    def imported(self) -> bool:
        """True when this window comes from a file rather than the archive."""
        return self.path is not None

    def label(self) -> str:
        where = self.path.name if self.imported else self.dataset
        return (f"{where} · {self.subject} · run-{self.run} · "
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
        if self.preprocess is not None:
            cfg.preprocess = self.preprocess
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
    #: One row per channel: the quality measurements and the verdict. Empty
    #: when the stage was switched off.
    quality: pd.DataFrame | None = None
    #: One row per channel per segment: which seconds were usable.
    segments: pd.DataFrame | None = None
    #: Channel -> seconds actually analysed. This, not the window length, is
    #: what every rate in `findings` was divided by.
    clean_seconds: dict[str, float] = field(default_factory=dict)
    #: The fetched slice itself. Kept so the assistant can run the project's
    #: own pipeline over the very window on screen rather than answering from
    #: a differently-configured one; the signal is already in memory as `raw`,
    #: so holding it costs nothing.
    recording: object | None = None
    #: What the *reader* decided, as opposed to what the detector decided.
    #: Empty until someone gives a verdict; loaded from disk and reconciled
    #: against these events by the launcher. See `onset_review.adjudication`.
    read: Adjudication = field(default_factory=Adjudication)
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    citation: str = ""
    sfreq: float = 0.0
    t_offset: float = 0.0
    montage: str = "bipolar"
    #: What was analysed, in original-recording seconds. Equal to the trace
    #: window unless the request asked for a longer span, in which case every
    #: rate, every rank and the trend cover this and the trace covers a slice
    #: of it. Zero-zero means "the same as the trace window" and is what every
    #: session built before spans existed carries.
    span_start: float = 0.0
    span_stop: float = 0.0
    #: The ICA stage's record when it ran (see `onset_hfo.preprocess`), for
    #: the Components panel. None when the stage is off.
    ica: dict | None = None

    @property
    def span(self) -> tuple[float, float]:
        if self.span_stop > self.span_start:
            return self.span_start, self.span_stop
        return float(self.request.t_start), float(self.request.t_stop)

    @property
    def span_duration(self) -> float:
        start, stop = self.span
        return stop - start

    @property
    def streamed(self) -> bool:
        return self.span_duration > self.request.duration + 1e-9

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
        """Minutes **analysed**, which is what every rate is divided by."""
        return self.span_duration / 60.0

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
            "analysed_s": self.span_duration,
            "trace_s": self.request.duration,
        }

    def scope(self) -> str:
        """What the numbers are over, when that is not what the trace shows.

        Empty for an ordinary window, where the two are the same thing and a
        sentence saying so would be noise. Not empty the moment they differ,
        because every rate on screen is then over ten minutes while the signal
        under it is one, and a reviewer who assumed otherwise would be reading
        the table wrong in the direction that matters.
        """
        if not self.streamed:
            return ""
        start, stop = self.span
        return (f"Rates and ranks are over {start:g}–{stop:g} s "
                f"({self.span_duration / 60:.1f} min). The trace holds "
                f"{self.request.t_start:g}–{self.request.t_stop:g} s; click any "
                f"event to load the minute it is in.")

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
                    accepted_only: bool = True, prefix: str = "",
                    window_s: float | None = None):
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
        onset = float(event.start) - float(t_offset)
        # An event outside the loaded trace is not drawn at its edge. When the
        # analysed span is longer than the trace, most events are outside it,
        # and clamping them to zero would pile ten minutes of marks onto the
        # first sample of the minute on screen.
        if onset < 0 or (window_s is not None and onset >= float(window_s)):
            continue
        onsets.append(onset)
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


def _electrodes_from_file(path, session_like):
    """Coordinates a reviewer pointed the software at, or None.

    Failures are silent here and loud in the interface: this runs on every
    re-analysis, and a file that has since been moved should not stop the
    window rebuilding -- the view falls back to the schematic layout it had
    before anyone supplied coordinates, which is the honest thing to show.
    """
    if not path:
        return None
    from onset_review.coordinates import read_coordinates

    try:
        read = read_coordinates(path, session_like)
    except Exception:                       # noqa: BLE001
        return None
    return read.frame if read.usable else None


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


def _findings_table(events: list[Event], channels: list[str],
                    duration_s, detector: str, reviewed: list[str],
                    expert: list[Event], window_s: float | None = None
                    ) -> pd.DataFrame:
    """Per-channel rates, ranked, with the two columns a reviewer needs beside.

    `channel_rates` already returns counts, rates and the Poisson interval that
    says whether to believe them; this adds the rank, whether the channel is one
    the archive's annotators actually looked at, and their own rate on it. The
    `reviewed` column matters more than it looks: in `ds003498` only a subset of
    channels was annotated, so a detection elsewhere is unjudged rather than
    wrong, and a reviewer comparing the two sources has to be able to see which
    rows the comparison is even defined on.

    `duration_s` may be one number or a per-channel mapping of analysed
    seconds. `window_s` is the window's own length, used for the annotators'
    rate: they marked the whole window, and dividing their count by *our*
    clean time would credit them with a rate they never claimed.
    """
    table = channel_rates([e for e in events if e.detector == detector],
                          duration_s, channels=channels)
    if table.empty:
        return table
    table = rank_channels(table)
    reviewed_set = set(reviewed)
    table["reviewed"] = [ch in reviewed_set for ch in table["channel"]]
    minutes = max(float(window_s if window_s is not None else duration_s) / 60.0,
                  1e-9)
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

    if request.streamed:
        return streamed_session(request, cache_dir, progress)

    if request.imported:
        from onset_hfo.io import open_recording

        say(0.05, f"Reading {request.path.name} "
                  f"{request.t_start:g}–{request.t_stop:g} s")
        record = open_recording(
            request.path, t_start=request.t_start, t_stop=request.t_stop,
            subject=request.subject, line_freq=request.line_freq,
            channel_types=dict(request.channel_types), run=request.run)
    else:
        say(0.05, f"Fetching {request.subject} "
                  f"{request.t_start:g}–{request.t_stop:g} s")
        record = fetch_slice(dataset=request.dataset, subject=request.subject,
                             run=request.run, task=request.task,
                             t_start=request.t_start, t_stop=request.t_stop,
                             cache_dir=cache_dir, verbose=False)
    return session_from_recording(record, request, progress=progress)


def source_for(request: ReviewRequest, cache_dir: Path | None = None):
    """A callable that produces any stretch of this request's recording.

    One seam for both halves of the streamed path: the chunks the analysis
    reads, and the window the trace holds. They must come from the same place
    or the trace would show a different recording from the one that was
    ranked.
    """
    if request.imported:
        from onset_hfo.io import open_recording

        def read_file(t_start: float, t_stop: float) -> Recording:
            return open_recording(
                request.path, t_start=t_start, t_stop=t_stop,
                subject=request.subject, line_freq=request.line_freq,
                channel_types=dict(request.channel_types), run=request.run)

        return read_file

    def read_archive(t_start: float, t_stop: float) -> Recording:
        return fetch_slice(dataset=request.dataset, subject=request.subject,
                           run=request.run, task=request.task,
                           t_start=t_start, t_stop=t_stop,
                           cache_dir=cache_dir, verbose=False)

    return read_archive


def streamed_session(request: ReviewRequest, cache_dir: Path | None = None,
                     progress=None) -> ReviewSession:
    """Analyse a span longer than memory; hold one window of it as signal.

    The division of labour is the whole feature. Rates, ranks, the trend, the
    event list and the quality stage cover `request.span()`; `raw` covers
    `t_start`–`t_stop` and nothing else. A reviewer ranks ten minutes of
    contacts and looks at one of them, which is how the work is actually done
    and what a one-minute tool could never support.
    """
    from onset_hfo.longrun import analyse_span

    def say(fraction: float, message: str) -> None:
        if progress is not None:
            progress(fraction, message)

    cfg = request.pipeline_config()
    # ICA is fitted on a whole stretch of signal; a span this long is analysed
    # chunk by chunk, and a decomposition fitted per chunk would remove
    # different things from each. So it is left off here and the notes say so.
    ica_note: list[str] = []
    if cfg.preprocess.ica:
        import dataclasses as _dc

        cfg = _dc.replace(cfg, preprocess=_dc.replace(cfg.preprocess, ica=False,
                                                       ica_exclude=()))
        ica_note = ["ICA was not applied: a span this long is analysed chunk by chunk, "
                    "and a decomposition fitted per chunk would remove different things "
                    "from each. Analyse a minute at a time to use it."]
    span_start, span_stop = request.span()
    source = source_for(request, cache_dir)

    # Expert markings live on the `Recording`, not in the signal, so they are
    # harvested chunk by chunk as the analysis goes past. Comparing a span's
    # detections against one window's annotations would make every detection
    # outside that window look like a false positive.
    experts: list[Event] = []
    seen_marks: set = set()
    reviewed: list[str] = []

    def harvest(record, plan) -> None:
        for event in _expert_events(record, request.band):
            key = (event.channel, round(float(event.start), 4))
            if key not in seen_marks:
                seen_marks.add(key)
                experts.append(event)
        for channel in getattr(record, "reviewed_channels", []):
            if channel not in reviewed:
                reviewed.append(channel)

    analysis = analyse_span(
        source, span_start, span_stop, cfg,
        detectors=tuple(request.detectors), with_spikes=request.with_spikes,
        check_quality=request.check_quality,
        progress=lambda fraction, message: say(0.03 + 0.80 * fraction, message),
        collect=harvest)

    say(0.86, f"Loading {request.t_start:g}–{request.t_stop:g} s to look at")
    record = source(request.t_start, request.t_stop)
    prep = prepare(record, cfg.preprocess, verbose=False)
    raw = _to_raw(prep.data, list(prep.ch_names), prep.sfreq)

    events = analysis.events
    say(0.92, "Measuring rates over the whole span")
    if request.check_quality:
        reject_unusable_events(events, analysis.segments, analysis.quality,
                               keep=request.keep_channels)
    clean = analysable_seconds(analysis.segments, analysis.quality,
                               list(analysis.ch_names),
                               analysis.duration, keep=request.keep_channels)
    reviewed = [c for c in reviewed if c in set(analysis.ch_names)]
    findings = _findings_table(events, list(analysis.ch_names),
                               clean or analysis.duration, request.primary,
                               reviewed, experts, window_s=analysis.duration)
    leader = leader_separation(findings) if not findings.empty else {}
    counts = (findings.set_index("channel")["n_events"]
              if not findings.empty else pd.Series(dtype=float))
    tied = (candidate_channels(counts, max(analysis.duration / 60.0, 1e-9))
            if len(counts) else [])

    # Only the events the trace actually holds become annotations; the rest
    # are in the tables, where they belong.
    raw.set_annotations(annotations_for(events, prep.t_offset,
                                        window_s=prep.duration))

    notes = list(getattr(record, "notes", []))
    notes.append(
        f"analysed {analysis.duration:g} s ({span_start:g}–{span_stop:g} s) in "
        f"{analysis.chunks} chunks; the trace holds "
        f"{request.t_start:g}–{request.t_stop:g} s of it, and every rate above "
        f"is over the whole span")
    if request.check_quality and analysis.quality is not None:
        notes.append(quality_summary(analysis.quality, analysis.segments,
                                     kept=request.keep_channels))

    say(1.0, "Ready")
    session = ReviewSession(
        request=request, raw=raw, events=events, findings=findings,
        leader=leader, candidates=list(tied), expert=experts,
        reviewed_channels=reviewed, resection=_resection_for(record)
        if not request.imported else None,
        electrodes=_electrodes_for(record),
        quality=analysis.quality, segments=analysis.segments,
        clean_seconds=dict(clean or {}), steps=list(analysis.steps),
        notes=notes + ica_note, citation=str(getattr(record, "citation", "")),
        sfreq=float(prep.sfreq), t_offset=float(prep.t_offset),
        montage=prep.montage, recording=record,
        span_start=float(span_start), span_stop=float(span_stop))
    supplied = _electrodes_from_file(request.electrodes_path, session)
    if supplied is not None:
        session.electrodes = supplied
    return session


def reload_trace(session: ReviewSession, t_start: float, t_stop: float,
                 cache_dir: Path | None = None) -> ReviewSession:
    """The same analysis, a different stretch of signal underneath it.

    This is what makes a long span navigable. The detections, the ranking, the
    quality verdicts and the reviewer's own read all belong to the span and
    must not move; only the minute of signal the trace holds does. Re-running
    the analysis to look at a different minute of it would cost minutes and
    change the numbers under the reviewer, which is the opposite of what
    clicking an event should do.

    A new `ReviewSession` rather than a mutated one, because the window is
    rebuilt around it and a half-swapped session seen by a panel mid-rebuild
    is the kind of bug that only shows up on someone else's machine.
    """
    import dataclasses

    span_start, span_stop = session.span
    start = max(span_start, float(t_start))
    stop = min(span_stop, float(t_stop))
    if stop <= start:
        return session

    request = dataclasses.replace(session.request, t_start=start, t_stop=stop,
                                  span_start=span_start, span_stop=span_stop)
    record = source_for(request, cache_dir)(start, stop)
    prep = prepare(record, request.pipeline_config().preprocess, verbose=False)
    raw = _to_raw(prep.data, list(prep.ch_names), prep.sfreq)
    raw.set_annotations(annotations_for(session.events, prep.t_offset,
                                        window_s=prep.duration))
    return dataclasses.replace(
        session, request=request, raw=raw, recording=record,
        t_offset=float(prep.t_offset), sfreq=float(prep.sfreq))


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
        resampled = (cfg.preprocess.resample
                     and abs(float(cfg.preprocess.resample) - prep.sfreq) < 1.0)
        raise ValueError(
            f"{request.band.replace('_', ' ')}s need a sampling rate above "
            f"{2 * band[1]:.0f} Hz; this recording is {prep.sfreq:.0f} Hz"
            + (f", because preprocessing resampled it to "
               f"{float(cfg.preprocess.resample):g} Hz. Raise or remove the "
               f"resampling, or choose the ripple band." if resampled else
               ". Choose the ripple band, or a recording sampled higher."))
    low_pass = cfg.preprocess.lowpass
    if low_pass and low_pass < band[1]:
        raise ValueError(
            f"the low-pass filter is set to {low_pass:g} Hz, which removes most "
            f"of the {request.band.replace('_', ' ')} band "
            f"({band[0]:.0f}–{band[1]:.0f} Hz). Raise it above {band[1]:.0f} Hz "
            f"or turn it off.")

    # Which contacts and which seconds are fit to analyse, before anything is
    # detected: the answer changes the denominator every rate is divided by,
    # and a channel that is set aside should not have events attributed to it.
    quality = segments = None
    if cfg.check_quality:
        say(0.33, "Checking the contacts and the seconds")
        segments = segment_quality(prep, cfg.quality)
        quality = channel_quality(prep, cfg.quality, segments=segments)

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
    resection = None if request.imported else _resection_for(record)
    electrodes = _electrodes_for(record)
    reviewed = [c for c in getattr(record, "reviewed_channels", [])
                if c in set(prep.ch_names)]

    say(0.85, "Measuring rates and confidence intervals")
    if cfg.check_quality:
        reject_unusable_events(events, segments, quality,
                               keep=request.keep_channels)
    clean = analysable_seconds(segments, quality, list(prep.ch_names),
                               prep.duration, keep=request.keep_channels)
    findings = _findings_table(events, list(prep.ch_names), clean or prep.duration,
                               request.primary, reviewed, expert,
                               window_s=prep.duration)
    leader = leader_separation(findings) if not findings.empty else {}
    counts = (findings.set_index("channel")["n_events"]
              if not findings.empty else pd.Series(dtype=float))
    tied = (candidate_channels(counts, max(prep.duration / 60.0, 1e-9))
            if len(counts) else [])

    say(0.95, "Building the trace")
    raw = _to_raw(prep.data, list(prep.ch_names), prep.sfreq)
    raw.set_annotations(annotations_for(events, prep.t_offset))

    say(1.0, "Ready")
    session = ReviewSession(
        request=request, raw=raw, events=events, findings=findings,
        leader=leader, candidates=list(tied), expert=expert,
        reviewed_channels=reviewed, resection=resection, electrodes=electrodes,
        quality=quality, segments=segments, clean_seconds=dict(clean or {}),
        steps=list(prep.steps),
        notes=(list(getattr(record, "notes", []))
               + ([quality_summary(quality, segments,
                                   kept=request.keep_channels)]
                  if cfg.check_quality else [])),
        citation=str(getattr(record, "citation", "")),
        sfreq=float(prep.sfreq), t_offset=float(prep.t_offset),
        montage=prep.montage, recording=record, ica=getattr(prep, "ica", None),
    )
    # A coordinate file the reviewer supplied outranks whatever the archive
    # shipped, which for every dataset here is nothing. Applied after the
    # session exists because matching the names needs the channel list.
    supplied = _electrodes_from_file(request.electrodes_path, session)
    if supplied is not None:
        session.electrodes = supplied
    return session
