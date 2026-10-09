"""A second archive: the HUP iEEG dataset (OpenNeuro ds004100), for replication.

The ictal study (`onset_hfo.ictal_study`, ds003029) and the outcome study
(`onset_hfo.outcome`, ds003498) each rest on one archive. HUP is a third,
from a different centre, with what both questions need for every patient in
the archive's own files:

* each contact's **onset-zone** and **resected** status
  (``channels.tsv``: ``status_description`` holds ``soz`` and ``resect``);
* **outcome** after surgery (``participants.tsv``: seizure-free ``S``,
  recurrence ``F``, ``NR`` no resection) and the Engel class;
* **seizures** with their onset marked (``task-ictal`` runs, ``events.tsv``);
* five minutes of **interictal** recording (``task-interictal``).

Its signals are EDF. A window of one is read without downloading the file:
EDF stores fixed-length data records after a header that says how long each
is, so `edf_window` fetches the header and only the records the window
needs, and writes them as a short EDF of their own. Nothing here changes
the methods: the same preprocessing, the same index, the same detector and
operating point, the same statistics, so a difference from the first
archive is a difference in the data.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["DATASET", "participants", "runs", "channel_labels", "edf_window", "read_window",
           "run_ictal", "run_interictal", "summarise_interictal", "bipolar_label"]

DATASET = "ds004100"
_BUCKET = "https://s3.amazonaws.com/openneuro.org"
LINE_FREQ = 60.0


def _get(path: str, byte_range=None) -> bytes:
    from onset_hfo.datasets import _http_get

    return _http_get(f"{_BUCKET}/{DATASET}/{path}", byte_range=byte_range)


def _list(prefix: str) -> list[str]:
    import urllib.parse

    from onset_hfo.datasets import _http_get

    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": f"{DATASET}/{prefix}", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        text = _http_get(f"{_BUCKET}?{urllib.parse.urlencode(query)}").decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", text)
        found = re.search(r"<NextContinuationToken>([^<]*)<", text)
        if not found:
            return keys
        token = found.group(1)


def _tsv(text: str) -> pd.DataFrame:
    import io

    return pd.read_csv(io.StringIO(text.lstrip("﻿")), sep="\t", dtype=str,
                       keep_default_na=False)


def participants() -> pd.DataFrame:
    frame = _tsv(_get("participants.tsv").decode("utf-8", "replace"))
    frame["subject"] = frame["participant_id"]
    frame["seizure_free"] = frame["outcome"].map({"S": True, "F": False})
    return frame


def runs(subject: str) -> pd.DataFrame:
    """Every run of a subject: task, acquisition, run, and its file stem."""
    rows = []
    for key in _list(f"{subject}/"):
        name = key.rsplit("/", 1)[-1]
        if not name.endswith("_ieeg.edf"):
            continue
        task = re.search(r"task-([^_]+)", name)
        acq = re.search(r"acq-([^_]+)", name)
        run = re.search(r"run-([^_]+)", name)
        rows.append({"task": task.group(1) if task else "", "acq": acq.group(1) if acq else "",
                     "run": run.group(1) if run else "",
                     "stem": key.split(f"{DATASET}/", 1)[1][:-len("_ieeg.edf")]})
    return pd.DataFrame(rows, columns=["task", "acq", "run", "stem"]).sort_values(
        ["task", "run"]).reset_index(drop=True)


def clean_name(name: str) -> str:
    """The contact's name without the recording system's decoration: older HUP
    files label channels ``EEG LG 29-Ref``, newer ones ``LG29``; both become
    ``LG29``, so the montage can pair contacts on a lead and the labels match."""
    text = re.sub(r"(?i)^eeg\s+", "", str(name).strip())
    text = re.sub(r"(?i)-ref$", "", text)
    return re.sub(r"\s+", "", text)


def channel_labels(stem: str) -> pd.DataFrame:
    """name (clean), file_name (as recorded), type, bad, soz, resected -- from
    the run's channels.tsv."""
    frame = _tsv(_get(f"{stem}_channels.tsv").decode("utf-8", "replace"))
    words = frame.get("status_description", pd.Series([""] * len(frame))).str.lower()
    return pd.DataFrame({
        "name": frame["name"].map(clean_name), "file_name": frame["name"].str.strip(),
        "type": frame["type"].str.upper(),
        "bad": frame.get("status", "").str.lower().eq("bad"),
        "soz": words.str.contains("soz"), "resected": words.str.contains("resect")})


def events(stem: str) -> pd.DataFrame | None:
    try:
        return _tsv(_get(f"{stem}_events.tsv").decode("utf-8", "replace"))
    except RuntimeError:
        return None


def seizure_onsets(frame: pd.DataFrame | None) -> list[float]:
    """Electrographic onsets marked in a run, in seconds."""
    if frame is None or not len(frame):
        return []
    kinds = frame["trial_type"].str.lower()
    marked = frame[kinds.str.contains("onset") & ~kinds.str.contains("clinical")]
    return sorted(float(v) for v in marked["onset"])


# -- reading a window of an EDF on the archive ---------------------------------------------
def edf_window(stem: str, t_start: float, t_stop: float, folder: Path) -> tuple[Path, float]:
    """The records of a remote EDF that cover [t_start, t_stop), as a short
    EDF in `folder`. Returns its path and the time of its first sample in
    the original file."""
    url_path = f"{stem}_ieeg.edf"
    head = _get(url_path, (0, 255))
    n_signals = int(head[252:256].decode().strip())
    header_bytes = int(head[184:192].decode().strip())
    n_records = int(head[236:244].decode().strip())
    record_s = float(head[244:252].decode().strip())
    header = _get(url_path, (0, header_bytes - 1))
    offset = 256 + n_signals * 216
    samples = [int(header[offset + 8 * i: offset + 8 * i + 8].decode().strip())
               for i in range(n_signals)]
    record_bytes = 2 * sum(samples)
    first = max(0, int(np.floor(t_start / record_s)))
    last = min(n_records, int(np.ceil(t_stop / record_s)))
    if last <= first:
        raise ValueError(f"the window {t_start:g}–{t_stop:g} s is outside the recording "
                         f"({n_records * record_s:g} s)")
    data = _get(url_path, (header_bytes + first * record_bytes,
                           header_bytes + last * record_bytes - 1))
    count = len(data) // record_bytes
    header = bytearray(header)
    header[236:244] = f"{count:<8d}".encode()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{Path(stem).name}_{first}-{first + count}.edf"
    path.write_bytes(bytes(header) + data[:count * record_bytes])
    return path, first * record_s


def read_window(stem: str, t_start: float, t_stop: float, labels: pd.DataFrame,
                folder: Path):
    """A `Recording` of [t_start, t_stop) with the archive's channel types;
    times in the result are original-file seconds minus the returned offset."""
    from onset_hfo.io import open_recording

    path, offset = edf_window(stem, t_start, t_stop, folder)
    types = {row.file_name: ("seeg" if row.type == "SEEG" else "ecog" if row.type == "ECOG"
                             else "misc") for row in labels.itertuples()}
    record = open_recording(path, subject=stem.split("/")[0], line_freq=LINE_FREQ,
                            channel_types=types)
    rename = {n: clean_name(n) for n in record.raw.ch_names if clean_name(n) != n}
    if rename:
        record.raw.rename_channels(rename)
    return record, offset


def bipolar_label(channel: str, contacts: set[str]) -> str:
    parts = [p.strip().upper() for p in str(channel).split("-")]
    inside = sum(p in contacts for p in parts)
    return "resected" if inside == len(parts) else "partial" if inside else "spared"


def _in(channel: str, contacts: set[str]) -> bool:
    return any(p.strip().upper() in contacts for p in str(channel).split("-"))


def _config(labels: pd.DataFrame):
    import dataclasses

    from onset_hfo.config import PreprocessConfig

    bad = tuple(labels.loc[labels["bad"], "name"])
    return dataclasses.replace(PreprocessConfig(), exclude=bad)


# -- the ictal replication ---------------------------------------------------------------------
def run_ictal(out, max_seizures: int = 3, settings=None, progress=print,
              shift_s: float = 0.0) -> dict:
    """The Epileptogenicity Index against HUP's onset-zone contacts, written
    as `onset_hfo.ictal_study` writes ds003029's (same tables, same summary).
    `shift_s` is that study's control: the "onset" put that many seconds
    before the real one, so all that is analysed precedes the seizure."""
    from onset_hfo.ictal import IctalSettings, epileptogenicity
    from onset_hfo.ictal_study import (
        WINDOW_AFTER_S,
        WINDOW_BEFORE_S,
        auc,
        per_patient,
        summarise,
    )
    from onset_hfo.preprocess import prepare

    if shift_s and shift_s < WINDOW_AFTER_S:
        raise ValueError(f"a control shift under {WINDOW_AFTER_S:g} s reaches the seizure")
    settings = settings or IctalSettings()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    people = participants().set_index("subject")
    seizure_rows, channel_rows = [], []
    scratch = Path(tempfile.mkdtemp(prefix="onset-hup-"))
    try:
        for subject in people.index:
            listed = runs(subject)
            ictal = listed[listed["task"] == "ictal"]
            if not len(ictal):
                continue
            done = 0
            for row in ictal.itertuples():
                if done >= max_seizures:
                    break
                labels = channel_labels(row.stem)
                soz = set(labels.loc[labels["soz"], "name"].str.upper())
                base = {"subject": subject, "run": row.run, "acq": row.acq,
                        "site": row.acq.upper(),
                        "seizure_free": people.at[subject, "seizure_free"],
                        "engel": people.at[subject, "engel"]}
                if not soz:
                    seizure_rows.append({**base, "status": "no onset-zone contacts marked"})
                    continue
                onsets = seizure_onsets(events(row.stem))
                if not onsets or onsets[0] < WINDOW_BEFORE_S:
                    seizure_rows.append({**base, "onset_s": onsets[0] if onsets else None,
                                         "status": "no usable onset mark"})
                    continue
                onset = onsets[0]
                if shift_s:
                    if onset - shift_s < WINDOW_BEFORE_S:
                        seizure_rows.append({**base, "onset_s": onset, "status": "too little "
                                             "recording before the seizure for the control"})
                        continue
                    onset = onset - shift_s
                    base["analysed_onset_s"] = onset
                started = time.monotonic()
                try:
                    record, offset = read_window(row.stem, onset - WINDOW_BEFORE_S,
                                                 onset + WINDOW_AFTER_S, labels, scratch)
                    prep = prepare(record, _config(labels), verbose=False)
                    table = epileptogenicity(prep.data, list(prep.ch_names), prep.sfreq,
                                             onset_s=onset - offset, start_s=prep.t_offset,
                                             settings=settings)
                except Exception as error:      # noqa: BLE001 - a failure is a row
                    seizure_rows.append({**base, "onset_s": onset,
                                         "status": f"{type(error).__name__}: {error}"[:200]})
                    continue
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)
                    scratch.mkdir(exist_ok=True)
                zone = [_in(ch, soz) for ch in table["channel"]]
                table = table.assign(subject=subject, run=row.run, soz=zone)
                channel_rows += table.to_dict("records")
                done += 1
                top = table.iloc[0]
                seizure_rows.append({**base, "onset_s": onset, "marker": "sz onset",
                                     "status": "ok", "sfreq": prep.sfreq,
                                     "n_channels": len(table), "n_soz": int(sum(zone)),
                                     "n_detected": int(table["detected"].sum()),
                                     "auc_ei": auc(table["ei"], zone),
                                     "auc_er": auc(table["er_after"].fillna(0.0), zone),
                                     "top_in_zone": bool(top["soz"]),
                                     "seconds": round(time.monotonic() - started, 1)})
                progress(f"{subject} run {row.run}: AUC {seizure_rows[-1]['auc_ei']:.2f}, top "
                         f"{top['channel']} {'in' if top['soz'] else 'outside'} the zone")
                pd.DataFrame(seizure_rows).to_csv(out / "per_seizure.csv", index=False)
                pd.DataFrame(channel_rows).to_csv(out / "per_channel.csv", index=False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    seizures = pd.DataFrame(seizure_rows)
    seizures.to_csv(out / "per_seizure.csv", index=False)
    patients = per_patient(pd.DataFrame(channel_rows), seizures)
    patients.to_csv(out / "per_patient.csv", index=False)
    summary = summarise(patients, seizures, settings)
    summary["dataset"] = DATASET
    if shift_s:
        summary["control_shift_s"] = shift_s
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


# -- the interictal replication ------------------------------------------------------------
def run_interictal(out, detector: str = "rms", progress=print) -> dict:
    """Interictal HFO rates on HUP's five-minute recordings, against the
    resection and outcome (as `onset_hfo.outcome` asks of ds003498) and
    against the onset zone. Ripples where the sampling allows (above 500 Hz),
    fast ripples above 1000 Hz, each at the outcome study's own operating
    point."""
    from onset_hfo.config import BANDS, DetectorConfig
    from onset_hfo.detectors import HFO_DETECTORS
    from onset_hfo.ictal_study import auc
    from onset_hfo.outcome import (
        BAND_THRESHOLD_SD,
        _share_in_resection,
        _top_channel_resected,
        _top_k_resected,
    )
    from onset_hfo.preprocess import prepare

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    people = participants().set_index("subject")
    rows, channel_rows = [], []
    scratch = Path(tempfile.mkdtemp(prefix="onset-hup-"))
    try:
        for subject in people.index:
            listed = runs(subject)
            inter = listed[listed["task"] == "interictal"]
            if not len(inter):
                continue
            row = next(inter.itertuples())
            base = {"subject": subject, "implant": row.acq.upper(),
                    "outcome": people.at[subject, "outcome"],
                    "seizure_free": people.at[subject, "seizure_free"],
                    "engel": people.at[subject, "engel"],
                    "therapy": people.at[subject, "therapy"]}
            try:
                labels = channel_labels(row.stem)
                record, _offset = read_window(row.stem, 0.0, 1e9, labels, scratch)
                prep = prepare(record, _config(labels), verbose=False)
            except Exception as error:      # noqa: BLE001
                rows.append({**base, "band": "", "status": f"{type(error).__name__}: "
                                                         f"{error}"[:200]})
                continue
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
                scratch.mkdir(exist_ok=True)
            resected = set(labels.loc[labels["resected"], "name"].str.upper())
            soz = set(labels.loc[labels["soz"], "name"].str.upper())
            channels = list(prep.ch_names)
            zones = pd.Series([bipolar_label(c, resected) for c in channels], index=channels)
            in_soz = np.array([_in(c, soz) for c in channels])
            for band_name in ("ripple", "fast_ripple"):
                band = getattr(BANDS, band_name)
                if not BANDS.usable(prep.sfreq, band):
                    rows.append({**base, "band": band_name, "sfreq": prep.sfreq,
                                 "status": f"not analysable at {prep.sfreq:g} Hz"})
                    continue
                cut = BAND_THRESHOLD_SD[band_name]
                found = HFO_DETECTORS[detector](prep, DetectorConfig(band=band,
                                                                     threshold_sd=cut))
                counts = pd.Series(0.0, index=channels)
                for event in found:
                    if event.accepted and event.channel in counts.index:
                        counts[event.channel] += 1.0
                minutes = prep.duration / 60.0
                rates = counts / minutes
                rows.append({**base, "band": band_name, "status": "ok", "sfreq": prep.sfreq,
                             "minutes": round(minutes, 2), "threshold_sd": cut,
                             "n_channels": len(channels), "n_soz": int(in_soz.sum()),
                             "n_resected_channels": int((zones == "resected").sum()),
                             **_share_in_resection(rates, zones),
                             "top_channel_resected": _top_channel_resected(rates, zones),
                             "top3_resected": _top_k_resected(rates, zones, k=3),
                             "auc_soz": auc(rates.to_numpy(), in_soz) if in_soz.any()
                             else float("nan"),
                             "auc_resected": auc(rates.to_numpy(), (zones == "resected").to_numpy())
                             if (zones == "resected").any() else float("nan")})
                for channel in channels:
                    channel_rows.append({"subject": subject, "band": band_name,
                                         "channel": channel, "events": counts[channel],
                                         "rate_per_min": rates[channel],
                                         "zone": zones[channel],
                                         "soz": bool(_in(channel, soz))})
                progress(f"{subject} {band_name}: AUC vs onset zone "
                         f"{rows[-1]['auc_soz']:.2f}, top channel "
                         f"{'resected' if rows[-1]['top_channel_resected'] == 1 else 'not resected'}")
            pd.DataFrame(rows).to_csv(out / "per_patient.csv", index=False)
            pd.DataFrame(channel_rows).to_csv(out / "per_channel.csv", index=False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "per_patient.csv", index=False)
    summary = summarise_interictal(table)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def summarise_interictal(table: pd.DataFrame) -> dict:
    """Per band: rate against the onset zone across patients, and the outcome
    comparison of the resection metrics (seizure-free against recurrence)."""
    from onset_hfo.ictal_study import bootstrap_median
    from onset_hfo.outcome import rank_comparison

    out = {"when": time.strftime("%Y-%m-%d"), "dataset": DATASET, "bands": {}}
    ok = table[table.get("status", "") == "ok"] if len(table) else table
    for band, part in ok.groupby("band"):
        values = part["auc_soz"].dropna().to_numpy(float)
        median, low, high = bootstrap_median(values)
        block = {"patients": int(len(part)),
                 "auc_soz": {"n": int(len(values)), "median": median, "ci_low": low,
                             "ci_high": high,
                             "share_above_half": float((values > 0.5).mean())
                             if len(values) else float("nan")},
                 "outcome": {}}
        if len(values) >= 6:
            from scipy.stats import wilcoxon

            block["auc_soz"]["wilcoxon_p"] = float(wilcoxon(values - 0.5).pvalue)
        for metric in ("share_in_rz", "top_channel_resected", "top3_resected"):
            free = part.loc[part["outcome"] == "S", metric].to_numpy(float)
            recur = part.loc[part["outcome"] == "F", metric].to_numpy(float)
            block["outcome"][metric] = rank_comparison(free, recur)
        out["bands"][band] = block
    return out


if __name__ == "__main__":      # pragma: no cover - a network run
    import os
    import sys

    os.environ.pop("ONSET_HFO_OFFLINE", None)
    which = sys.argv[1] if len(sys.argv) > 1 else "ictal"
    target = Path(sys.argv[2] if len(sys.argv) > 2 else f"artifacts/results/hup_{which}")
    shift = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    result = (run_ictal(target, shift_s=shift) if which == "ictal"
              else run_interictal(target))
    print(json.dumps(result, indent=1, default=str))
