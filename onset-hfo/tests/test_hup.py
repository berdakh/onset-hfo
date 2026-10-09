"""The HUP replication's plumbing, offline.

What has to hold: a window of a remote EDF, read as header plus only the
records it needs, gives exactly the samples of the whole file at those
times; channel names decorated by the recording system (``EEG LG 29-Ref``)
become the plain names the montage pairs and the labels match; a bipolar
channel is resected only when both contacts were; seizure onsets are the
electrographic marks, not the offsets or clinical onsets; and the
interictal summary compares seizure-free with recurrence per band.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import pytest

from onset_hfo import hup


@pytest.fixture
def remote_edf(tmp_path, monkeypatch):
    """A local EDF served as if it were the archive's, by byte range."""
    rng = np.random.default_rng(0)
    names = ["EEG LA 01-Ref", "EEG LA 02-Ref", "EEG LA 03-Ref", "EEG EKG1-Ref"]
    raw = mne.io.RawArray(rng.normal(0, 50e-6, (4, 256 * 20)),
                          mne.create_info(names, 256.0, "eeg"), verbose="ERROR")
    path = tmp_path / "whole.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    blob = path.read_bytes()

    def fake_get(url_path, byte_range=None):
        if byte_range is None:
            return blob
        return blob[byte_range[0]:byte_range[1] + 1]

    monkeypatch.setattr(hup, "_get", fake_get)
    return path


def test_a_window_is_the_whole_file_at_those_times(remote_edf, tmp_path):
    whole = mne.io.read_raw_edf(remote_edf, preload=True, verbose="ERROR")
    path, offset = hup.edf_window("sub-X/ses-1/ieeg/sub-X_run-01", 5.2, 9.7, tmp_path / "w")
    part = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    record_s = 1.0
    assert offset == pytest.approx(5.0) and offset % record_s == 0
    assert part.n_times / part.info["sfreq"] == pytest.approx(5.0), "records 5 to 10"
    start = int(offset * whole.info["sfreq"])
    assert np.array_equal(part.get_data(), whole.get_data()[:, start:start + part.n_times])
    assert path.stat().st_size < remote_edf.stat().st_size / 2, "only what was needed"
    with pytest.raises(ValueError, match="outside the recording"):
        hup.edf_window("sub-X/ses-1/ieeg/sub-X_run-01", 50.0, 60.0, tmp_path / "w")


def test_decorated_channel_names_become_plain_contacts():
    assert hup.clean_name("EEG LG 29-Ref") == "LG29"
    assert hup.clean_name("EEG EKG1-Ref") == "EKG1"
    assert hup.clean_name("LA01") == "LA01"
    assert hup.clean_name("  eeg RPF a3-REF ") == "RPFa3"


def test_zones_and_onsets():
    resected = {"LA01", "LA02"}
    assert hup.bipolar_label("LA01-LA02", resected) == "resected"
    assert hup.bipolar_label("LA02-LA03", resected) == "partial"
    assert hup.bipolar_label("LB01-LB02", resected) == "spared"
    events = pd.DataFrame({"onset": ["120.0", "150.0", "200.0", "10"],
                           "trial_type": ["sz onset", "clinical onset", "sz offset",
                                          "eeg sz onset"]})
    assert hup.seizure_onsets(events) == [10.0, 120.0]
    assert hup.seizure_onsets(None) == []


def test_the_interictal_summary_compares_outcomes_per_band():
    rows = []
    for i in range(12):
        free = i < 8
        rows.append({"subject": f"s{i}", "band": "ripple", "status": "ok",
                     "outcome": "S" if free else "F", "auc_soz": 0.6 + 0.02 * i,
                     "share_in_rz": 0.8 if free else 0.2,
                     "top_channel_resected": 1.0 if free else 0.0,
                     "top3_resected": 0.67 if free else 0.33})
    rows.append({"subject": "s99", "band": "ripple", "status": "not analysable"})
    summary = hup.summarise_interictal(pd.DataFrame(rows))
    block = summary["bands"]["ripple"]
    assert block["patients"] == 12 and block["auc_soz"]["n"] == 12
    assert block["outcome"]["top_channel_resected"]["auc"] == pytest.approx(1.0)
    assert block["outcome"]["share_in_rz"]["n_seizure_free"] == 8
