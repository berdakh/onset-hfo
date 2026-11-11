"""Saved study summaries must reproduce from their committed intermediate data."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import rederive_outcome_tables


def test_committed_outcome_tables_reproduce_without_being_rewritten(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    outcome = tmp_path / "outcome"
    outcome.mkdir()
    for name in ("channels.csv.gz", "subjects.csv", "recordings.csv", "participants.csv"):
        shutil.copyfile(root / "data/outcome" / name, outcome / name)
    groups = tmp_path / "outcome_groups_300s.csv"
    shutil.copyfile(root / "data/stability/outcome_groups_300s.csv", groups)
    files = [*outcome.iterdir(), groups]
    before = {path: path.read_bytes() for path in files}
    monkeypatch.setattr(rederive_outcome_tables, "OUTCOME", outcome)
    monkeypatch.setattr(rederive_outcome_tables, "GROUPS", groups)
    monkeypatch.setattr(sys, "argv", ["rederive_outcome_tables.py", "--check"])

    status = rederive_outcome_tables.main()

    assert {path: path.read_bytes() for path in files} == before
    assert status == 0, capsys.readouterr().out


@pytest.mark.parametrize("folder", ["ictal", "hup/ictal"])
@pytest.mark.parametrize("prefix", ["", "control_"])
def test_saved_ictal_patient_scores_reproduce(folder, prefix):
    from onset_hfo.ictal_study import per_patient

    root = Path(__file__).resolve().parents[1] / "data" / folder
    channels = pd.read_csv(root / f"{prefix}per_channel.csv")
    seizures = pd.read_csv(root / f"{prefix}per_seizure.csv")
    stored = pd.read_csv(root / f"{prefix}per_patient.csv").set_index("subject").sort_index()

    rebuilt = per_patient(channels, seizures).set_index("subject").sort_index()

    # The control includes a highest-EI tie between an SOZ and a non-SOZ
    # channel. An unstable sort flips the flag without changing any AUC.
    columns = ["auc_ei", "auc_er", "n_channels", "n_soz", "n_seizures", "top_in_zone"]
    pd.testing.assert_frame_equal(rebuilt[columns], stored[columns], check_exact=False,
                                  rtol=1e-12, atol=1e-12)


def test_benchmark_means_reproduce_from_the_per_patient_scores():
    """Every committed cohort mean is the mean of `per_patient.csv.gz`'s rows,
    undefined values left out, to three decimals: exactly, not approximately."""
    bench = Path(__file__).resolve().parents[1] / "data" / "benchmark"
    patients = pd.read_csv(bench / "per_patient.csv.gz")
    keys = ["band", "detector", "threshold_sd"]
    rebuilt = patients.groupby(keys).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        detections=("n_detections", "mean"), rank_rho=("spearman_rho", "mean"),
        top5_overlap=("top5_overlap", "mean")).round(3).reset_index()
    for name in ("agreement_sweep.csv", "four_detector_sweep.csv"):
        committed = pd.read_csv(bench / name)
        both = committed.merge(rebuilt, on=keys, how="left", suffixes=("", "_rebuilt"))
        assert len(both) == len(committed)
        for column in ("precision", "recall", "f1", "detections", "rank_rho", "top5_overlap"):
            pd.testing.assert_series_equal(both[column], both[f"{column}_rebuilt"],
                                           check_names=False, obj=f"{name} {column}")
    reviewed = patients.groupby("subject")["n_channels"].first()
    cohort = pd.read_csv(bench / "cohort.csv").set_index("subject")
    assert reviewed.equals(cohort["n_reviewed_channels"].rename("n_channels"))
