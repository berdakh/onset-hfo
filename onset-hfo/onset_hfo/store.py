"""A read-only view of a saved pipeline result.

This is the boundary between the two halves of the project. The pipeline
writes CSV and JSON files; the agent reads them **through this class and
nothing else**. It cannot touch the signal, cannot re-run a detector, and
cannot change a number.

Every method returns plain Python types (dicts, lists, floats) that serialise
straight to JSON, because they are the return values of the agent's tools.

Each returned record carries an ``evidence_id`` of the form
``subject|channel|detector|start``. That is what the agent has to cite, and
:meth:`ResultStore.resolve` is what verifies a citation was actually retrieved
rather than invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["ResultStore"]


def _clean(value):
    """Make a value JSON-safe (NaN -> None, numpy scalar -> Python scalar)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return round(value, 4) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, str) and value == "nan"):
        return None
    return value


def _record(row: pd.Series, keys: list[str]) -> dict:
    return {k: _clean(row[k]) for k in keys if k in row}


def _table_path(directory: Path, name: str) -> Path | None:
    """``name`` or its gzipped twin, whichever exists.

    A minute of real recording produces an ``events.csv`` of roughly a
    megabyte, which is fine on disk and wasteful in a repository. Accepting
    ``.csv.gz`` lets a worked example ship with the source -- pandas reads it
    transparently -- without the pipeline having to write a second format.
    """
    for candidate in (directory / name, directory / f"{name}.gz"):
        if candidate.exists():
            return candidate
    return None


def _read_table(directory: Path, name: str) -> pd.DataFrame:
    """An optional table, empty when it was never written."""
    path = _table_path(directory, name)
    return pd.read_csv(path) if path is not None else pd.DataFrame()


class ResultStore:
    """Load and query one saved analysis."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        events = _table_path(self.dir, "events.csv")
        if events is None:
            raise FileNotFoundError(
                f"{self.dir} does not look like a saved analysis "
                "(no events.csv). Run the pipeline with save_to=... first.")
        self.events = pd.read_csv(events)
        self.provenance = json.loads((self.dir / "provenance.json").read_text())
        self.report = json.loads((self.dir / "report.json").read_text())
        self.config = json.loads((self.dir / "config.json").read_text())
        self.agreement = json.loads((self.dir / "agreement.json").read_text()) \
            if (self.dir / "agreement.json").exists() else {}
        self.comparison = _read_table(self.dir, "comparison.csv")
        self.rate_change = _read_table(self.dir, "rate_change.csv")
        #: Where each channel sits, when whoever saved the analysis knew:
        #: shaft, side, region, position and its provenance. Written by the
        #: desktop window beside its analysis; absent from a plain pipeline
        #: run, and then `contact_map` says so.
        self.contacts = _read_table(self.dir, "contacts.csv")
        self.rates = {}
        for path in sorted(self.dir.glob("rates_*.csv*")):
            name = path.name.removeprefix("rates_").removesuffix(".gz").removesuffix(".csv")
            self.rates[name] = pd.read_csv(path)
        self.events["evidence_id"] = [
            self.make_evidence_id(r.channel, r.detector, r.start)
            for r in self.events.itertuples()]

    # -- identity ---------------------------------------------------------
    @classmethod
    def from_result(cls, result, directory: str | Path | None = None) -> ResultStore:
        """Save an in-memory :class:`~onset_hfo.pipeline.PipelineResult`, then load it."""
        return cls(result.save(directory))

    @property
    def subject(self) -> str:
        return str(self.provenance.get("subject", "unknown"))

    def make_evidence_id(self, channel: str, detector: str, start: float) -> str:
        return f"{self.subject}|{channel}|{detector}|{float(start):.3f}"

    # -- the queries the agent is allowed to make -------------------------
    def metadata(self) -> dict:
        """What was analysed, how, and with which settings."""
        duration = None
        if self.provenance.get("slice_stop_s") is not None:
            duration = float(self.provenance["slice_stop_s"]) - float(self.provenance["slice_start_s"])
        hfo_detectors = [d for d in self.rates if d != "spike"]
        return {
            "subject": self.subject,
            "source": self.provenance.get("source"),
            "task": self.provenance.get("task"),
            "run": self.provenance.get("run"),
            "sampling_rate_hz": _clean(self.provenance.get("sfreq_hz")),
            "channels_in_recording": _clean(self.provenance.get("n_channels")),
            "channels_analysed": int(self.events["channel"].nunique()) if len(self.events) else 0,
            "analysed_window_s": [_clean(self.provenance.get("slice_start_s")),
                                  _clean(self.provenance.get("slice_stop_s"))],
            "duration_s": _clean(duration),
            "seizure_onset_s": _clean(self.provenance.get("seizure_onset_s")),
            "seizure_offset_s": _clean(self.provenance.get("seizure_offset_s")),
            "hfo_detectors": hfo_detectors,
            "bad_channels_excluded": self.provenance.get("bad_channels", []),
            "pipeline_version": self.config.get("pipeline_version"),
            "band_hz": self.config.get("config", {}).get("rms", {}).get("band"),
            "citation": self.report.get("citation", ""),
            "notes": self.provenance.get("notes", []),
        }

    def list_channels(self, detector: str | None = None) -> list[dict]:
        """Every analysed channel with its event count and rate."""
        name = detector or self._default_detector()
        table = self.rates.get(name)
        if table is None:
            return []
        table = table.sort_values("rate_per_min", ascending=False).reset_index(drop=True)
        return [{"channel": str(r["channel"]), "detector": name,
                 "rank": i + 1, "n_events": _clean(r["n_events"]),
                 "rate_per_min": _clean(r["rate_per_min"])}
                for i, r in table.iterrows()]

    def top_channels(self, detector: str | None = None, k: int = 5) -> list[dict]:
        """The ``k`` channels with the highest event rate, with intervals."""
        name = detector or self._default_detector()
        table = self.rates.get(name)
        if table is None:
            return []
        rows = table.sort_values("rate_per_min", ascending=False).head(int(k))
        out = []
        for i, (_, r) in enumerate(rows.iterrows()):
            out.append({
                "channel": str(r["channel"]), "detector": name, "rank": i + 1,
                "rate_per_min": _clean(r["rate_per_min"]),
                "n_events": _clean(r["n_events"]),
                "rate_ci_95": [_clean(r.get("rate_ci_low")), _clean(r.get("rate_ci_high"))],
                "mean_frequency_hz": _clean(r.get("mean_frequency_hz")),
                "mean_duration_ms": _clean(r.get("mean_duration_ms")),
            })
        return out

    def channel_summary(self, channel: str) -> dict:
        """Everything known about one channel, from every detector."""
        known = self.channels()
        if channel not in known:
            return {"error": f"channel {channel!r} was not analysed",
                    "analysed_channels": known[:40]}
        out: dict = {"channel": channel, "detectors": {}}
        for name, table in self.rates.items():
            ordered = table.sort_values("rate_per_min", ascending=False).reset_index(drop=True)
            match = ordered.index[ordered["channel"] == channel]
            if not len(match):
                continue
            row = ordered.loc[match[0]]
            out["detectors"][name] = {
                "rank": int(match[0]) + 1,
                "rate_per_min": _clean(row["rate_per_min"]),
                "n_events": _clean(row["n_events"]),
                "rate_ci_95": [_clean(row.get("rate_ci_low")), _clean(row.get("rate_ci_high"))],
                "mean_frequency_hz": _clean(row.get("mean_frequency_hz")),
                "mean_amplitude_uv": _clean(row.get("mean_amplitude_uv")),
                "n_with_spike": _clean(row.get("n_with_spike")),
            }
        ch_events = self.events[(self.events["channel"] == channel) & self.events["accepted"]]
        out["n_accepted_events_all_detectors"] = int(len(ch_events))
        out["contacts"] = sorted({c for s in self.events.loc[self.events["channel"] == channel,
                                                             "contacts"].fillna("")
                                  for c in str(s).split("|") if c})
        return out

    def evidence(self, channel: str, detector: str | None = None, k: int = 3,
                 include_spikes: bool = False) -> list[dict]:
        """The strongest accepted events on a channel: the citable windows.

        HFO events only by default. Interictal discharges are a different
        measurement on a different band and mixing them into one "evidence"
        list would let a reader take a spike for a ripple; ask for them
        explicitly with ``detector="spike"``.
        """
        sel = self.events[(self.events["channel"] == channel) & self.events["accepted"]]
        if detector:
            sel = sel[sel["detector"] == detector]
        elif not include_spikes:
            sel = sel[sel["detector"] != "spike"]
        sel = sel.sort_values("score", ascending=False).head(int(k))
        keys = ["evidence_id", "channel", "detector", "start", "stop", "duration_ms",
                "peak_frequency_hz", "spectral_prominence_db", "peak_amplitude_uv",
                "n_peaks", "co_occurs_with_spike"]
        return [_record(r, keys) for _, r in sel.iterrows()]

    def disagreements(self) -> list[dict]:
        """Channels the two detectors rank very differently (as the report defines it)."""
        if not len(self.comparison):
            return []
        rows = self.comparison[self.comparison["disagrees"]]
        out = []
        for _, r in rows.iterrows():
            record = {"channel": str(r["channel"]), "rank_gap": _clean(r["rank_gap"])}
            for col in self.comparison.columns:
                if col.startswith("rank_") and col != "rank_gap":
                    record[col] = _clean(r[col])
                if col.startswith("rate_per_min_"):
                    record[col] = _clean(r[col])
            out.append(record)
        return out

    def detector_agreement(self) -> dict:
        """Event-by-event agreement between the HFO detectors that were run."""
        return {k: _clean(v) for k, v in self.agreement.items()}

    def rate_change_table(self) -> list[dict]:
        """Per-channel rate before versus during the marked seizure, if computed."""
        if not len(self.rate_change):
            return []
        return [{k: _clean(v) for k, v in row.items()}
                for row in self.rate_change.head(10).to_dict("records")]

    def contact_map(self, detector: str | None = None, top: int = 5) -> dict:
        """Where the activity sits: each channel's shaft, side and region
        with its rate for `detector`, and how the leading channels
        distribute over shafts and sides. Carries no resection, by design:
        see ``onset_review.contactmap``."""
        if self.contacts is None or self.contacts.empty:
            return {"available": False,
                    "note": "no contact positions were saved with this analysis"}
        name = detector or self._default_detector()
        frame = self.contacts.copy()
        rates = self.rates.get(name)
        if rates is not None and name != self._default_detector():
            by_channel = rates.set_index("channel")["rate_per_min"]
            frame["rate_per_min"] = [_clean(by_channel.get(c, 0.0)) for c in frame["channel"]]
            order = frame["rate_per_min"].rank(ascending=False, method="first")
            frame["rank"] = order.astype(int)
        frame = frame.sort_values("rank")
        rows = [{"channel": str(r.channel), "shaft": str(r.shaft), "side": str(r.hemisphere),
                 "region": str(r.region), "rank": int(r.rank),
                 "rate_per_min": _clean(r.rate_per_min)}
                for r in frame.itertuples()]
        out = summarise_contacts(frame, top=top)
        out.update({"available": True, "detector": name, "placed": rows})
        return out

    def report_section(self, section: str) -> dict | list | str:
        """One section of the structured report.

        Valid sections: ``summary``, ``findings``, ``disagreements``,
        ``data_quality``, ``methods``, ``limitations``, ``recording``,
        ``rate_change``. There is no ``recommendation`` section, by design.
        """
        valid = ["summary", "findings", "disagreements", "data_quality", "methods",
                 "limitations", "recording", "rate_change"]
        if section not in valid:
            return {"error": f"unknown section {section!r}", "available_sections": valid}
        return self.report.get(section, [])

    # -- helpers ----------------------------------------------------------
    def channels(self) -> list[str]:
        table = self.rates.get(self._default_detector())
        if table is not None:
            return [str(c) for c in table["channel"]]
        return sorted({str(c) for c in self.events["channel"]})

    def detectors(self) -> list[str]:
        return sorted(self.rates)

    def resolve(self, evidence_id: str) -> dict | None:
        """Look up an evidence id. Returns ``None`` if it does not exist.

        The agent's answer is checked against this: a citation that does not
        resolve means the model made it up, and the answer is refused.
        """
        hit = self.events[self.events["evidence_id"] == evidence_id]
        if not len(hit):
            return None
        keys = ["evidence_id", "channel", "detector", "start", "stop", "duration_ms",
                "peak_frequency_hz", "spectral_prominence_db", "peak_amplitude_uv", "accepted"]
        return _record(hit.iloc[0], keys)

    def _default_detector(self) -> str:
        for name in ("rms", "line_length"):
            if name in self.rates:
                return name
        return next(iter(self.rates), "rms")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ResultStore({self.dir.name}: {self.subject}, "
                f"{len(self.events)} events, detectors={self.detectors()})")


#: Region labels that say nothing, left out of the sentence.
_NO_REGION = {"", "unmapped", "unknown", "nan", "none"}


def summarise_contacts(contacts: pd.DataFrame, top: int = 5) -> dict:
    """How the leading channels distribute over shafts and sides, as counts
    and one sentence. Five leaders on one shaft is one finding; five leaders
    on five shafts is a different one, and this is what says which."""
    if contacts is None or contacts.empty:
        return {"positions": "none", "summary": "No channels to place.", "shafts": []}
    frame = contacts.sort_values("rank").head(int(top))
    positions = ("measured" if set(contacts["source"]) == {"archive"}
                 else "schematic (inferred from electrode names)")
    groups = []
    for shaft, rows in frame.groupby("shaft", sort=False):
        side = str(rows["hemisphere"].iloc[0])
        region = str(rows["region"].iloc[0])
        groups.append({"shaft": str(shaft),
                       "side": side if side in ("left", "right") else "",
                       "region": region if region.lower() not in _NO_REGION else "",
                       "n_of_top": int(len(rows)),
                       "channels": [str(c) for c in rows["channel"]]})
    groups.sort(key=lambda g: -g["n_of_top"])
    n = int(len(frame))
    parts = []
    for g in groups:
        known = ", ".join(x for x in (g["side"], g["region"]) if x)
        parts.append(f"{g['n_of_top']} of the top {n} channels on shaft {g['shaft']}"
                     + (f" ({known})" if known else ""))
    sides = sorted({g["side"] for g in groups if g["side"]})
    summary = "; ".join(parts) + "."
    if len(groups) == 1:
        summary += " The leading channels are neighbours on one electrode."
    elif len(sides) == 1 and sides[0] in ("left", "right"):
        summary += f" All on the {sides[0]} side."
    return {"positions": positions, "n_channels": int(len(contacts)),
            "n_shafts": int(contacts["shaft"].nunique()), "top": n,
            "shafts": groups, "summary": summary}
