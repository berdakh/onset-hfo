"""Hours of recording at a glance: a sampled envelope. Qt-free.

A day of 200 contacts at 2 kHz is over a hundred gigabytes; reading all of
it to draw one line would take as long as the analysis. So the overview
*samples*: every `block_s` seconds it reads `sample_s` seconds and measures
them, across the intracranial contacts not marked bad --

* `rms_uv`, the median contact's RMS (µV): where the recording is loud,
  quiet or flat;
* `line_length`, the line length per second (µV/s) of the busiest tenth of
  the contacts (their 90th percentile): rises with fast activity, seizures
  among it. Not the median: a seizure that starts on four contacts of a
  hundred does not move the median at all;
* `flat_share`, the share of contacts whose signal barely moves (< 0.5 µV
  RMS): a disconnected headbox or a recording gap.

It says that it samples (`coverage`), and it is cached beside the case's
other results, so it is computed once per recording.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["envelope", "cached_envelope", "COLUMNS", "FLAT_UV"]

COLUMNS = ("t", "rms_uv", "line_length", "flat_share")
FLAT_UV = 0.5


def envelope(raw, block_s: float = 30.0, sample_s: float = 2.0, progress=None,
             should_stop=None) -> pd.DataFrame:
    """The sampled overview of `raw` (an MNE Raw, lazy is best). One row per
    block: its start (s from the first sample) and the three measures."""
    sfreq = float(raw.info["sfreq"])
    types = raw.get_channel_types()
    picks = [i for i, (name, kind) in enumerate(zip(raw.ch_names, types, strict=True))
             if kind in ("seeg", "ecog") and name not in raw.info["bads"]]
    if not picks:
        picks = [i for i, kind in enumerate(types) if kind in ("eeg", "seeg", "ecog")]
    if not picks:
        raise ValueError("no intracranial or EEG channel to draw an overview from")
    duration = raw.n_times / sfreq
    starts = np.arange(0.0, duration, float(block_s))
    rows = []
    for index, start in enumerate(starts):
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped before the overview was finished")
        a = int(round(start * sfreq))
        b = min(raw.n_times, a + max(2, int(round(sample_s * sfreq))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = raw.get_data(picks=picks, start=a, stop=b) * 1e6
        data = data - data.mean(axis=1, keepdims=True)
        rms = np.sqrt((data ** 2).mean(axis=1))
        seconds = max((b - a) / sfreq, 1e-9)
        length = np.abs(np.diff(data, axis=1)).sum(axis=1) / seconds
        rows.append({"t": float(start), "rms_uv": float(np.median(rms)),
                     "line_length": float(np.percentile(length, 90)),
                     "flat_share": float((rms < FLAT_UV).mean())})
        if progress is not None:
            progress((index + 1) / len(starts))
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame.attrs.update(block_s=float(block_s), sample_s=float(sample_s),
                       coverage=min(1.0, float(sample_s) / float(block_s)),
                       n_contacts=len(picks), duration_s=float(duration))
    return frame


def cached_envelope(case, rec, block_s: float = 30.0, sample_s: float = 2.0, progress=None,
                    should_stop=None, compute: bool = True) -> pd.DataFrame | None:
    """The overview of a case's recording, from the cache when it was drawn
    with the same settings and the same channels; else computed (unless
    `compute` is False) and cached."""
    from onset_hfo.case import bids

    where = case.where(rec)
    folder = Path(case.derivatives) / "overview"
    table, meta_path = folder / f"{where.stem}_overview.tsv", \
        folder / f"{where.stem}_overview.json"
    channels = bids.read_channels(where)
    signature = {"version": 2, "block_s": float(block_s), "sample_s": float(sample_s),
                 "bad": sorted(channels.loc[channels["status"] == "bad", "name"].astype(str)),
                 "types": sorted(f"{n}:{t}" for n, t in zip(channels["name"], channels["type"],
                                                            strict=True))}
    if table.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("signature") == signature:
                frame = pd.read_csv(table, sep="\t")
                frame.attrs.update(meta.get("attrs", {}))
                return frame
        except (OSError, ValueError):
            pass
    if not compute:
        return None
    raw = bids.read_run(where, preload=False)
    frame = envelope(raw, block_s, sample_s, progress, should_stop)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table, sep="\t", index=False)
    meta_path.write_text(json.dumps({"signature": signature, "attrs": dict(frame.attrs)},
                                    indent=1))
    return frame
