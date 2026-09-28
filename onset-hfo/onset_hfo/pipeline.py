"""The end-to-end pipeline: recording in, evidence tables and report out.

    recording -> preprocess -> detect (x2 HFO + 1 spike) -> validate
              -> rates & ranks -> detector agreement -> report

One function, :func:`run_pipeline`, does all of it and returns a
:class:`PipelineResult` that holds every intermediate table. Nothing is
hidden: the events another step aggregated are still there, with the reason
any of them was rejected.

The result can be saved to a directory of CSV/JSON files. The agent
(:mod:`onset_agent`) reads *only* that saved form, which is why the agent
cannot invent numbers: it has no access to the signal, only to what the
pipeline wrote down.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from onset_hfo.config import PIPELINE_VERSION, RESULTS_DIR, PipelineConfig
from onset_hfo.datasets import Recording
from onset_hfo.detectors import HFO_DETECTORS as _HFO_DETECTORS
from onset_hfo.detectors import (
    detect_spikes,
)
from onset_hfo.detectors.base import Event, bandpass, events_to_frame
from onset_hfo.metrics import channel_rates, compare_rankings, detector_agreement, rate_change
from onset_hfo.populations import population_notes, population_rates
from onset_hfo.preprocess import Prepared, prepare
from onset_hfo.report import Report, build_report
from onset_hfo.validate import flag_spike_cooccurrence, rejection_summary, validate_events

__all__ = ["PipelineResult", "run_pipeline"]

#: Every HFO detector that can be asked for by name. The *default* is still
#: two -- see ``run_pipeline(detectors=...)`` -- because the two added later
#: carry inherited thresholds, and turning them on by default would change
#: every published number without anyone deciding to.
# Re-exported so callers that import it from here keep working.
HFO_DETECTORS = _HFO_DETECTORS


@dataclass
class PipelineResult:
    """Everything one run produced, in memory."""

    recording: Recording
    prepared: Prepared
    config: PipelineConfig
    events: dict[str, list[Event]]      #: detector name -> events (accepted and rejected)
    spikes: list[Event]
    rates: dict[str, pd.DataFrame]      #: detector name -> per-channel rates
    spike_rates: pd.DataFrame
    comparison: pd.DataFrame
    agreement: dict
    report: Report
    rate_change: pd.DataFrame | None = None
    timings: dict[str, float] = field(default_factory=dict)
    #: detector name -> per-channel rates split into the two sub-populations
    #: of roadmap item 6. Never merged back into one number by this class.
    populations: dict[str, pd.DataFrame] = field(default_factory=dict)

    # -- access -----------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return self.prepared.duration

    def events_frame(self, accepted_only: bool = False) -> pd.DataFrame:
        """All events from all detectors as one table."""
        frames = [events_to_frame(v) for v in list(self.events.values()) + [self.spikes]]
        out = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
        if accepted_only and len(out):
            out = out[out["accepted"]].reset_index(drop=True)
        return out

    def top_channels(self, detector: str = "rms", k: int = 5) -> pd.DataFrame:
        return self.rates[detector].head(k)

    def evidence(self, channel: str, detector: str | None = None, k: int = 3) -> list[Event]:
        """The strongest accepted events on a channel -- what a reviewer looks at."""
        pool = [e for name, evs in self.events.items() if detector in (None, name)
                for e in evs if e.channel == channel and e.accepted]
        return sorted(pool, key=lambda e: -e.score)[:k]

    # -- persistence ------------------------------------------------------
    def save(self, directory: str | Path | None = None, name: str | None = None) -> Path:
        """Write the result as CSV/JSON. Returns the directory it wrote to."""
        rec = self.recording
        name = name or f"{rec.subject}_{rec.task}_run-{rec.run}"
        out = Path(directory or RESULTS_DIR) / name
        out.mkdir(parents=True, exist_ok=True)
        self.events_frame().to_csv(out / "events.csv", index=False)
        for det, table in self.rates.items():
            table.to_csv(out / f"rates_{det}.csv", index=False)
        self.spike_rates.to_csv(out / "rates_spike.csv", index=False)
        for det, table in self.populations.items():
            table.to_csv(out / f"populations_{det}.csv", index=False)
        self.comparison.to_csv(out / "comparison.csv", index=False)
        if self.rate_change is not None:
            self.rate_change.to_csv(out / "rate_change.csv", index=False)
        (out / "agreement.json").write_text(json.dumps(self.agreement, indent=2, default=str))
        (out / "report.json").write_text(self.report.to_json())
        (out / "report.md").write_text(self.report.to_markdown())
        (out / "provenance.json").write_text(json.dumps(rec.provenance(), indent=2, default=str))
        (out / "config.json").write_text(json.dumps(
            {"pipeline_version": PIPELINE_VERSION, "config": self.config.as_dict(),
             "preprocessing": self.prepared.steps, "timings_s": self.timings},
            indent=2, default=str))
        print(f"[onset-hfo] results written to {out}")
        return out


def run_pipeline(recording: Recording, config: PipelineConfig | None = None,
                 detectors: tuple[str, ...] = ("rms", "line_length"),
                 with_spikes: bool = True, save_to: str | Path | None = None,
                 verbose: bool = True) -> PipelineResult:
    """Run the whole analysis on one recording.

    Parameters
    ----------
    recording:
        From :func:`onset_hfo.datasets.fetch_slice` (public data) or
        :func:`onset_hfo.synthetic.make_synthetic_recording` (offline, labelled).
    config:
        All thresholds; see :class:`onset_hfo.config.PipelineConfig`.
    detectors:
        Which HFO detectors to run. Two is the default *by design*: a single
        detector cannot disagree with anything.
    with_spikes:
        Also run the interictal discharge detector, and flag HFOs that
        coincide with a discharge.
    save_to:
        Directory to write CSV/JSON results into; ``None`` to skip saving.
    """
    cfg = config or PipelineConfig()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    prep = prepare(recording, cfg.preprocess, verbose=verbose)
    timings["preprocess"] = time.perf_counter() - t0

    # Both HFO detectors share one band-pass: filtering twice would be slower
    # and, worse, would make a difference between them possible for a reason
    # that has nothing to do with the detectors.
    t0 = time.perf_counter()
    filtered = bandpass(prep.data, prep.sfreq, cfg.rms.band)
    timings["bandpass"] = time.perf_counter() - t0

    events: dict[str, list[Event]] = {}
    for name in detectors:
        if name not in HFO_DETECTORS:
            raise ValueError(f"Unknown detector {name!r}; available: {sorted(HFO_DETECTORS)}")
        t0 = time.perf_counter()
        det_cfg = getattr(cfg, name)
        found = HFO_DETECTORS[name](prep, det_cfg, filtered=filtered)
        validate_events(found, prep, cfg.validation)
        events[name] = found
        timings[f"detect_{name}"] = time.perf_counter() - t0
        if verbose:
            print(f"[onset-hfo] {name}: {len(found)} candidates, "
                  f"{sum(e.accepted for e in found)} accepted after artifact validation "
                  f"({timings[f'detect_{name}']:.1f} s)")

    spikes: list[Event] = []
    if with_spikes:
        t0 = time.perf_counter()
        spikes = detect_spikes(prep, cfg.spikes)
        for name in events:
            flag_spike_cooccurrence(events[name], spikes)
        timings["detect_spikes"] = time.perf_counter() - t0
        if verbose:
            print(f"[onset-hfo] spikes: {len(spikes)} interictal discharges "
                  f"({timings['detect_spikes']:.1f} s)")

    duration = prep.duration
    rates = {name: channel_rates(evs, duration, prep.ch_names) for name, evs in events.items()}
    spike_rates = channel_rates(spikes, duration, prep.ch_names) if spikes else \
        channel_rates([], duration, prep.ch_names)

    names = list(events)
    if len(names) >= 2:
        comparison = compare_rankings(rates[names[0]], rates[names[1]], names[0], names[1],
                                      cfg.disagreement_ranks, cfg.top_k)
        agreement = detector_agreement(events[names[0]], events[names[1]], names[0], names[1])
    else:
        comparison = pd.DataFrame(columns=["channel", "disagrees", "rank_gap", "best_rank"])
        agreement = {"note": "only one detector was run; no agreement can be computed"}

    change_frame = None
    change_rows: list[dict] = []
    onset, offset = recording.seizure
    start_abs = prep.t_offset
    stop_abs = prep.t_offset + duration
    if onset is not None and start_abs < onset < stop_abs:
        before = (start_abs, onset)
        during = (onset, min(offset or stop_abs, stop_abs))
        if before[1] - before[0] >= 5 and during[1] - during[0] >= 5:
            change_frame = rate_change(events[names[0]], before, during, prep.ch_names)
            change_rows = change_frame.head(cfg.top_k).to_dict("records")
            if verbose:
                print(f"[onset-hfo] rate change computed: before {before[0]:.0f}-{before[1]:.0f} s "
                      f"vs during {during[0]:.0f}-{during[1]:.0f} s")

    # Roadmap item 6: a rate that merges physiological and epileptic ripples
    # is the project's largest silent assumption. It cannot be resolved here,
    # so it is at least reported -- two rates, and the caveat that neither
    # sub-population is a label.
    populations = {name: population_rates(evs, duration, prep.ch_names)
                   for name, evs in events.items()}
    notes = list(recording.notes or [])
    notes += population_notes(events, duration)

    rejections = {name: rejection_summary(evs) for name, evs in events.items()}
    report = build_report(
        provenance=recording.provenance(), rates=rates, comparison=comparison, events=events,
        spike_rates=spike_rates, duration_s=duration, preprocessing_steps=prep.steps,
        config=cfg.as_dict(), rejections=rejections, rate_change_rows=change_rows,
        top_k=cfg.top_k, notes=notes, citation=recording.citation)

    result = PipelineResult(recording=recording, prepared=prep, config=cfg, events=events,
                            spikes=spikes, rates=rates, spike_rates=spike_rates,
                            comparison=comparison, agreement=agreement, report=report,
                            rate_change=change_frame, timings=timings,
                            populations=populations)
    if save_to is not None:
        result.save(save_to)
    if verbose:
        print(f"[onset-hfo] done in {sum(timings.values()):.1f} s")
    return result
