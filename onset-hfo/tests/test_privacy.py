"""A patient's name in a file name must not become the label on everything."""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.privacy import (
    is_loopback,
    looks_identifying,
    safe_label,
    shown_name,
    study_code,
)


@pytest.mark.parametrize("stem", [
    "SMITH_JOHN_19800101", "Mueller, Anna", "doe^jane", "smith_j_12-03-1975",
    "patient_4471829", "Garcia Lopez EEG",
])
def test_a_hospital_exports_name_is_recognised(stem):
    assert looks_identifying(stem)


@pytest.mark.parametrize("stem", [
    "study-007", "sub-01_ses-01_task-rest_run-01_ieeg", "decorated_raw", "P17",
    "case12_night2", "sub-pt01_ictal_run-01", "recording", "sleep_export_3",
])
def test_an_ordinary_study_file_name_is_left_alone(stem):
    assert looks_identifying(stem) == ""


def test_the_label_and_the_shown_name_never_carry_the_file_name(tmp_path):
    named = tmp_path / "SMITH_JOHN_19800101.edf"
    named.write_bytes(b"0" * 10)
    code = study_code(named)
    assert code.startswith("study-") and len(code) == 12
    assert study_code(named) == code, "the same file gets the same code"
    assert safe_label(named) == code
    assert shown_name(named) == f"{code}.edf"
    plain = tmp_path / "study-007.edf"
    plain.write_bytes(b"0")
    assert safe_label(plain) == "study-007" and shown_name(plain) == "study-007.edf"


def test_an_imported_recording_carries_a_code_not_the_patients_name(tmp_path):
    import mne

    from onset_hfo.io import open_recording

    rng = np.random.default_rng(0)
    raw = mne.io.RawArray(rng.normal(0, 8e-6, size=(2, 4000)),
                          mne.create_info(["A1", "A2"], 2000.0, ch_types="seeg"),
                          verbose="ERROR")
    path = tmp_path / "SMITH_JOHN_19800101_raw.fif"
    raw.save(path, verbose="ERROR")
    record = open_recording(path)
    everything = " ".join([record.subject, record.source, *record.notes,
                           str(record.provenance())])
    assert "SMITH" not in everything and "19800101" not in everything
    assert record.subject == study_code(path)
    assert any("not shown" in note for note in record.notes)
    # A label somebody typed is theirs to choose.
    assert open_recording(path, subject="P17").subject == "P17"


@pytest.mark.parametrize("url,local", [
    ("http://127.0.0.1:11434/v1", True), ("http://localhost:8000/v1", True),
    ("http://[::1]:8000/v1", True), ("http://10.0.0.5:8000/v1", False),
    ("https://models.example.org/v1", False), ("not a url", False), ("", False),
])
def test_only_this_machine_counts_as_local(url, local):
    assert is_loopback(url) is local
