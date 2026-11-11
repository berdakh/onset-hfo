"""Archive electrode names that are not the recording's (ds003498, five patients)."""

from __future__ import annotations

import pandas as pd
import pytest

from onset_hfo import clinical
from onset_hfo.datasets import (
    _archive_electrodes,
    archive_electrode_names,
    parse_hfo_annotations,
    rename_contacts,
)

FIVE = ("sub-02", "sub-05", "sub-06", "sub-07", "sub-08")


def test_a_swap_renames_each_contact_once():
    names = {"AL": "AHL", "EL": "AL"}
    assert rename_contacts("AL1-AL2", names) == "AHL1-AHL2"
    assert rename_contacts("EL3-EL4", names) == "AL3-AL4"
    assert rename_contacts("al2", names) == "AHL2"
    assert rename_contacts("IPR1-IPR2", names) == "IPR1-IPR2"
    assert rename_contacts("AL1-AL2", {}) == "AL1-AL2"


def test_the_table_names_the_five_patients_one_to_one():
    table = _archive_electrodes()
    assert set(table["subject"]) == set(FIVE)
    for subject in FIVE:
        names = archive_electrode_names("ds003498", subject)
        assert len(names) == 8
        assert len(set(names.values())) == 8, f"{subject}: two archive electrodes on one"
        assert set(names.values()) == {f"{e}{s}" for e in ("A", "AH", "EC", "PH") for s in "LR"}
        # The same letter pairing on both sides.
        assert {k[:-1]: v[:-1] for k, v in names.items() if k.endswith("L")} == \
               {k[:-1]: v[:-1] for k, v in names.items() if k.endswith("R")}
    assert archive_electrode_names("ds003498", "sub-01") == {}
    assert archive_electrode_names("ds003029", "sub-pt01") == {}


def test_where_there_were_marks_the_assigned_electrode_beats_chance_and_the_rest():
    table = _archive_electrodes()
    marked = table[table["n_marks"] >= 5]
    assert len(marked) >= 30
    assert (marked["recall"] > 3 * marked["chance"]).all()
    assert (marked["recall"] >= marked["best_other_recall"]).mean() >= 0.9


def test_markings_and_the_resection_sheet_are_read_in_the_recording_s_names(monkeypatch):
    events = pd.DataFrame({"onset": [1.0, 2.0], "duration": [0.05, 0.05],
                           "trial_type": ["ripple_AR1-2", "fr_ER2-3"]})
    truth, reviewed = parse_hfo_annotations(events)
    assert reviewed == ["AR1-AR2", "ER2-ER3"]       # the parser keeps the archive's names

    sheet = pd.DataFrame({"subject": ["sub-02", "sub-01"],
                          "rz": ["ar1-4, er1-2", "ar1-2"], "excluded": [None, None]})
    monkeypatch.setattr(clinical, "fetch_clinical_sheet", lambda *a, **k: sheet)
    resections = clinical.resection_map("ds003498", verbose=False)
    assert resections["sub-02"].resected == ("AHR1", "AHR2", "AHR3", "AHR4", "AR1", "AR2")
    assert resections["sub-02"].rz_text == "ar1-4, er1-2"
    assert resections["sub-01"].resected == ("AR1", "AR2")


@pytest.mark.parametrize("subject", ["sub-02"])
def test_a_cached_slice_reads_every_reviewed_channel(subject):
    from onset_hfo.datasets import OfflineError, fetch_slice

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ONSET_HFO_OFFLINE", "1")
            rec = fetch_slice(dataset="ds003498", subject=subject, run="01",
                              t_start=0, t_stop=60, verbose=False)
    except OfflineError:
        pytest.skip(f"{subject}'s first minute is not cached")
    missing = [c for c in rec.reviewed_channels
               if any(p not in rec.raw.ch_names for p in c.split("-"))]
    assert missing == []
    assert any("read as AL=AHL" in note for note in rec.notes)
