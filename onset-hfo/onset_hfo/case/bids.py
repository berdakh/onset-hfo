"""A recording written as BIDS-iEEG, and read back: no Qt, no extra packages.

BIDS-iEEG is the open layout intracranial recordings are shared in; MNE reads
it, and so do other groups' tools, so a clinic's data converted here is not
locked into this software. The pieces, for one run of one patient:

    sub-P017/ses-implant01/ieeg/
      sub-P017_ses-implant01_task-monitoring_run-01_ieeg.vhdr  -- BrainVision header
      ..._ieeg.eeg, ..._ieeg.vmrk                               -- the samples, a marker file
      ..._ieeg.json      -- sampling rate, mains, reference, duration
      ..._channels.tsv   -- name, type, units, sampling rate, good or bad
      ..._events.tsv     -- seizures, sleep, artefacts, the system's own marks

The samples are written as BrainVision, 32-bit float, in microvolts, a chunk
at a time: a day of 200 contacts at 2 kHz is over a hundred gigabytes and is
never in memory at once. The native sampling rate is kept.

The writer writes what it is given and nothing else. No name, birth date or
recording date goes into any of these files (`onset_hfo.case.bridges` decides
what is given); the events keep times from the start of the recording.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["BidsRun", "write_run", "read_run", "label", "BIDS_TYPES", "MNE_TYPES",
           "EVENT_COLUMNS", "CHANNEL_COLUMNS", "write_dataset_description"]

#: MNE channel type -> BIDS channel type.
BIDS_TYPES = {"seeg": "SEEG", "ecog": "ECOG", "eeg": "EEG", "ecg": "ECG", "emg": "EMG",
              "eog": "EOG", "stim": "TRIG", "misc": "MISC", "resp": "RESP", "bio": "MISC",
              "dbs": "DBS", "temperature": "MISC", "gsr": "MISC"}
#: BIDS channel type -> MNE channel type.
MNE_TYPES = {"SEEG": "seeg", "ECOG": "ecog", "EEG": "eeg", "ECG": "ecg", "EMG": "emg",
             "EOG": "eog", "TRIG": "stim", "MISC": "misc", "RESP": "resp", "DBS": "dbs",
             "VEOG": "eog", "HEOG": "eog", "REF": "misc", "PD": "misc", "SYSCLOCK": "misc"}

CHANNEL_COLUMNS = ("name", "type", "units", "sampling_frequency", "low_cutoff",
                   "high_cutoff", "status", "status_description")
#: onset and duration in seconds from the recording's first sample; trial_type
#: from `onset_hfo.case.annotations.KINDS`; channels "n/a" for all; value the
#: original text; source "file" (the clinical system) or "reader"; by and at
#: who added it and when.
EVENT_COLUMNS = ("onset", "duration", "trial_type", "channels", "value", "source", "by", "at")

#: Samples per chunk when writing, at most: about 64 MB of float32 at 200
#: channels.
_CHUNK_SAMPLES = 80_000


def label(text: str) -> str:
    """A BIDS label: letters and digits only."""
    out = re.sub(r"[^A-Za-z0-9]", "", str(text))
    if not out:
        raise ValueError(f"{text!r} has no letters or digits to make a BIDS label from")
    return out


@dataclass(frozen=True)
class BidsRun:
    """Where one run lives in a BIDS folder."""

    root: Path
    subject: str
    session: str = "implant01"
    task: str = "monitoring"
    run: str = "01"

    @property
    def folder(self) -> Path:
        return Path(self.root) / f"sub-{self.subject}" / f"ses-{self.session}" / "ieeg"

    @property
    def stem(self) -> str:
        return (f"sub-{self.subject}_ses-{self.session}_task-{self.task}_run-{self.run}")

    def path(self, suffix: str) -> Path:
        """`suffix` is "ieeg.vhdr", "channels.tsv", "events.tsv" and so on."""
        return self.folder / f"{self.stem}_{suffix}"

    @property
    def header(self) -> Path:
        return self.path("ieeg.vhdr")


def _escape(name: str) -> str:
    # BrainVision separates fields with commas; a comma in a name is "\1".
    return str(name).replace(",", r"\1")


def _volts(raw) -> np.ndarray:
    """Which channels MNE holds in volts (written in µV); the rest as they are."""
    from mne.io.constants import FIFF

    return np.array([ch["unit"] == FIFF.FIFF_UNIT_V for ch in raw.info["chs"]])


def write_run(where: BidsRun, raw, *, channels: pd.DataFrame | None = None,
              events: pd.DataFrame | None = None, line_freq: float = 50.0,
              reference: str = "n/a", manufacturer: str = "", extra_json: dict | None = None,
              progress=None, should_stop=None) -> BidsRun:
    """Write `raw` (an MNE Raw, preloaded or not) as one BIDS-iEEG run.

    `channels` overrides the type and status of each channel (columns name,
    type in MNE's spelling, status good/bad, status_description); `events`
    is written as the run's events.tsv. `progress(fraction)` is told how far
    the samples are; `should_stop()` stops between chunks, leaving nothing
    half-written under the final names.
    """
    where.folder.mkdir(parents=True, exist_ok=True)
    sfreq = float(raw.info["sfreq"])
    names = list(raw.ch_names)
    volts = _volts(raw)
    header, data_path, marker = where.header, where.path("ieeg.eeg"), where.path("ieeg.vmrk")
    partial = data_path.with_suffix(".eeg.partial")
    n_times = int(raw.n_times)
    with open(partial, "wb") as out:
        for start in range(0, n_times, _CHUNK_SAMPLES):
            if should_stop is not None and should_stop():
                out.close()
                partial.unlink(missing_ok=True)
                raise InterruptedError("stopped before the recording was written")
            stop = min(n_times, start + _CHUNK_SAMPLES)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                block = raw.get_data(start=start, stop=stop)
            block[volts] *= 1e6
            np.ascontiguousarray(block.T, dtype="<f4").tofile(out)
            if progress is not None:
                progress(stop / max(n_times, 1))
    partial.replace(data_path)

    units = ["µV" if v else "n/a" for v in volts]
    lines = ["Brain Vision Data Exchange Header File Version 1.0",
             "; Written by onset-hfo: 32-bit float, microvolts, multiplexed", "",
             "[Common Infos]", "Codepage=UTF-8", f"DataFile={data_path.name}",
             f"MarkerFile={marker.name}", "DataFormat=BINARY",
             "DataOrientation=MULTIPLEXED", f"NumberOfChannels={len(names)}",
             f"SamplingInterval={1e6 / sfreq!r}", "",
             "[Binary Infos]", "BinaryFormat=IEEE_FLOAT_32", "",
             "[Channel Infos]"]
    lines += [f"Ch{i + 1}={_escape(n)},,1,{u}" for i, (n, u) in enumerate(zip(names, units,
                                                                            strict=True))]
    header.write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n\n"
                      "[Common Infos]\nCodepage=UTF-8\n"
                      f"DataFile={data_path.name}\n\n[Marker Infos]\n"
                      "Mk1=New Segment,,1,1,0\n", encoding="utf-8")

    table = _channel_table(raw, channels)
    table.to_csv(where.path("channels.tsv"), sep="\t", index=False, na_rep="n/a")
    write_events(where, events)
    types = set(table["type"])
    sidecar = {
        "TaskName": where.task,
        "SamplingFrequency": sfreq,
        "PowerLineFrequency": float(line_freq),
        "iEEGReference": reference or "n/a",
        "SoftwareFilters": "n/a",
        "RecordingDuration": round(n_times / sfreq, 6),
        "RecordingType": "continuous",
        "SEEGChannelCount": int(sum(t == "SEEG" for t in table["type"])),
        "ECOGChannelCount": int(sum(t == "ECOG" for t in table["type"])),
        "EEGChannelCount": int(sum(t == "EEG" for t in table["type"])),
        "ECGChannelCount": int(sum(t == "ECG" for t in table["type"])),
        "EMGChannelCount": int(sum(t == "EMG" for t in table["type"])),
        "MiscChannelCount": int(sum(t not in {"SEEG", "ECOG", "EEG", "ECG", "EMG"}
                                    for t in table["type"])),
        "HardwareFilters": "n/a",
    }
    if manufacturer:
        sidecar["Manufacturer"] = manufacturer
    sidecar.update(extra_json or {})
    if not types:
        raise ValueError("a recording with no channels")
    where.path("ieeg.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return where


def _channel_table(raw, channels: pd.DataFrame | None) -> pd.DataFrame:
    sfreq = float(raw.info["sfreq"])
    low = raw.info.get("highpass") or 0.0
    high = raw.info.get("lowpass") or sfreq / 2
    given = {}
    if channels is not None and len(channels):
        given = {str(r["name"]): r for r in channels.to_dict("records")}
    bads = set(raw.info["bads"])
    rows = []
    volts = _volts(raw)
    for name, kind, volt in zip(raw.ch_names, raw.get_channel_types(), volts, strict=True):
        row = given.get(name, {})
        mne_type = str(row.get("type") or kind)
        status = str(row.get("status") or ("bad" if name in bads else "good"))
        rows.append({"name": name, "type": BIDS_TYPES.get(mne_type, "MISC"),
                     "units": "µV" if volt else "n/a", "sampling_frequency": sfreq,
                     "low_cutoff": float(low), "high_cutoff": float(high),
                     "status": status if status in ("good", "bad") else "good",
                     "status_description": str(row.get("status_description") or "n/a")})
    return pd.DataFrame(rows, columns=list(CHANNEL_COLUMNS))


def write_events(where: BidsRun, events: pd.DataFrame | None) -> Path:
    """The run's events.tsv, sorted by onset; an empty table is still written."""
    frame = pd.DataFrame(columns=list(EVENT_COLUMNS)) if events is None else events.copy()
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = "n/a" if column not in ("onset", "duration") else 0.0
    frame = frame[list(EVENT_COLUMNS)].copy()
    frame["onset"] = frame["onset"].astype(float).round(6)
    frame["duration"] = frame["duration"].astype(float).round(6)
    frame = frame.sort_values(["onset", "trial_type"], kind="stable")
    path = where.path("events.tsv")
    where.folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="n/a")
    return path


def read_events(where: BidsRun) -> pd.DataFrame:
    path = where.path("events.tsv")
    if not path.exists():
        return pd.DataFrame(columns=list(EVENT_COLUMNS))
    frame = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = "n/a"
    frame["onset"] = pd.to_numeric(frame["onset"], errors="coerce")
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce").fillna(0.0)
    return frame[list(EVENT_COLUMNS)].dropna(subset=["onset"]).reset_index(drop=True)


def read_channels(where: BidsRun) -> pd.DataFrame:
    return pd.read_csv(where.path("channels.tsv"), sep="\t", keep_default_na=False)


def read_run(where: BidsRun, preload: bool = False):
    """The run as an MNE Raw: types and bad channels from channels.tsv, the
    mains from the sidecar, the events as annotations (channel-specific where
    the events say so). Lazy unless asked."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_brainvision(where.header, preload=preload, verbose="ERROR")
    channels = read_channels(where)
    types = {str(r.name): MNE_TYPES.get(str(r.type).upper(), "misc")
             for r in channels.itertuples() if str(r.name) in raw.ch_names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_channel_types(types, verbose="ERROR")
    raw.info["bads"] = [str(r.name) for r in channels.itertuples()
                        if str(r.status).lower() == "bad" and str(r.name) in raw.ch_names]
    sidecar = json.loads(where.path("ieeg.json").read_text())
    line = sidecar.get("PowerLineFrequency")
    if isinstance(line, (int, float)) and math.isfinite(line):
        with raw.info._unlock():
            raw.info["line_freq"] = float(line)
    events = read_events(where)
    if len(events):
        ch_names = [[c for c in str(v).split(",") if c in raw.ch_names]
                    if str(v) not in ("n/a", "") else [] for v in events["channels"]]
        raw.set_annotations(mne.Annotations(
            onset=events["onset"].to_numpy(float), duration=events["duration"].to_numpy(float),
            description=events["trial_type"].astype(str).to_list(),
            ch_names=ch_names, orig_time=raw.annotations.orig_time))
    return raw


def write_dataset_description(root: Path, name: str) -> Path:
    path = Path(root) / "dataset_description.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"Name": name, "BIDSVersion": "1.9.0",
                                    "DatasetType": "raw",
                                    "GeneratedBy": [{"Name": "onset-hfo"}]}, indent=1) + "\n")
    ignore = Path(root) / ".bidsignore"
    if not ignore.exists():
        ignore.write_text("case.json\nderivatives/\n")
    return path
