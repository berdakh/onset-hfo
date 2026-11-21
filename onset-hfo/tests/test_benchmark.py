"""The agent benchmark harness, exercised offline.

Everything here runs with the scripted planner on a synthetic recording: no
model, no GPU, no network. That is deliberate. A sweep harness whose only test
is the GPU run it was written for is a harness nobody can trust before they
spend the GPU hour.
"""

from __future__ import annotations

import json

import pytest

from onset_agent.benchmark import (
    QUANTIZATIONS,
    RUNGS,
    Cell,
    _overlap,
    _rank_agreement,
    _ranks,
    backend_provenance,
    environment,
    expand_matrix,
    fits,
    load_cells,
    run_cell,
    run_matrix,
    score_result,
    summarise,
    timing_warnings,
)
from onset_agent.planner import ScriptedPlanner, run_rung

MODEL = "test/tiny"


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


def test_cell_rejects_an_unknown_quantization():
    with pytest.raises(ValueError, match="quantization"):
        Cell(model=MODEL, quantization="3bit", rung="S2", subject="sub-01")


def test_cell_rejects_an_unknown_rung():
    with pytest.raises(ValueError):
        Cell(model=MODEL, quantization="fp16", rung="S9", subject="sub-01")


def test_cell_id_is_filesystem_safe():
    cell = Cell(model="Qwen/Qwen2.5-7B-Instruct", quantization="4bit",
                rung="S3", subject="sub-pt01")
    assert "/" not in cell.cell_id
    assert cell.cell_id.endswith("__S3__sub-pt01")


def test_s0_appears_once_per_subject_not_once_per_quantization():
    """S0 runs no model, so three precisions would be three identical rows."""
    cells = expand_matrix([MODEL], QUANTIZATIONS, RUNGS, ["sub-01"])
    s0 = [c for c in cells if c.rung == "S0"]
    assert len(s0) == 1
    # every other rung gets one cell per quantization
    assert len(cells) == 1 + 3 * (len(RUNGS) - 1)


def test_matrix_is_deterministic():
    a = expand_matrix([MODEL, "other/model"], subjects=["s1", "s2"])
    b = expand_matrix([MODEL, "other/model"], subjects=["s1", "s2"])
    assert [c.cell_id for c in a] == [c.cell_id for c in b]


def test_fits_leaves_headroom_and_is_permissive_for_unknown_sizes():
    assert fits("4B", "4bit", 16.0)
    assert not fits("14B", "fp16", 16.0)
    assert fits("8B", "fp16", 24.0)
    assert fits("70B", "4bit", 4.0)  # unknown size: do not block the user


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_environment_records_what_could_change_a_number():
    env = environment()
    for key in ("python", "platform", "git_commit", "transformers", "torch", "gpu_name"):
        assert key in env
    assert isinstance(env["gpu_total_gb"], float)


def test_backend_provenance_survives_a_backend_that_knows_nothing():
    class Bare:
        pass

    info = backend_provenance(Bare())
    assert info["backend"] == "Bare"
    assert info["model_id"] == ""


def test_backend_provenance_picks_up_what_a_backend_does_expose():
    planner = ScriptedPlanner()
    info = backend_provenance(planner)
    assert info["backend"]
    assert "describe" in info


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_overlap_counts_shared_top_k():
    assert _overlap(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0
    assert _overlap(["a", "b", "c"], ["c", "b", "a"], k=3) == 1.0  # a set, not an order
    assert _overlap(["a", "b"], ["x", "y"], k=2) == 0.0


def test_rank_agreement_declines_to_measure_two_points():
    assert _rank_agreement(["a", "b"], ["b", "a"]) is None


def test_rank_agreement_is_one_for_an_identical_order():
    order = ["a", "b", "c", "d", "e"]
    assert _rank_agreement(order, order) == 1.0


def test_rank_agreement_is_minus_one_for_a_reversed_order():
    assert _rank_agreement(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == -1.0


def test_midranks_share_a_rank_for_ties():
    assert _ranks([10, 10, 20]) == [0.5, 0.5, 2.0]


def test_score_result_reports_validity_and_unsupported_claims(session):
    session.reset_memo()
    result = run_rung(session, rung="S0")
    row = score_result(result)
    assert row["rung"] == "S0"
    assert row["n_tool_calls"] > 0
    assert 0.0 <= row["tool_call_validity"] <= 1.0
    assert 0.0 <= row["unsupported_claim_rate"] <= 1.0
    # provenance coverage is the complement, by construction
    assert row["provenance_coverage"] == pytest.approx(
        1.0 - row["unsupported_claim_rate"], abs=1e-4)


def test_score_result_compares_against_the_reference_when_given_one(session):
    session.reset_memo()
    result = run_rung(session, rung="S0")
    row = score_result(result, reference_top=result.top_channels)
    assert row["top5_overlap_vs_s0"] == 1.0


def test_unsupported_claim_rate_is_measured_even_on_rungs_that_do_not_verify(session):
    """S2 does not verify in production; the benchmark still measures it."""
    session.reset_memo()
    result = run_rung(session, rung="S2", backend=ScriptedPlanner())
    row = score_result(result)
    assert "unsupported_claim_rate" in row
    assert row["n_sentences"] > 0


# --------------------------------------------------------------------------
# Running, checkpointing, resuming
# --------------------------------------------------------------------------


def _factories(session):
    return (lambda _subject: session), (lambda _cell: ScriptedPlanner())


def test_run_cell_records_a_failure_instead_of_raising(session):
    def explode(_cell):
        raise RuntimeError("out of memory")

    cell = Cell(model=MODEL, quantization="4bit", rung="S2", subject="sub-01")
    record = run_cell(cell, session=session, backend_factory=explode)
    assert record["ok"] is False
    assert record["error"]["type"] == "RuntimeError"
    assert "out of memory" in record["error"]["message"]
    assert record["wall_clock_s"] >= 0


def test_run_matrix_writes_one_checkpoint_per_cell(session, tmp_path):
    session_factory, backend_factory = _factories(session)
    cells = [Cell(model=MODEL, quantization="fp16", rung=r, subject="sub-01")
             for r in ("S0", "S2")]
    out = run_matrix(cells, session_factory=session_factory,
                     backend_factory=backend_factory, out=tmp_path / "sweep",
                     verbose=False)
    files = sorted((out / "cells").glob("*.json"))
    assert len(files) == 2
    record = json.loads(files[0].read_text())
    assert record["ok"] is True
    assert record["cell"]["subject"] == "sub-01"
    assert record["environment"]["python"]


def test_run_matrix_resumes_by_skipping_finished_cells(session, tmp_path):
    session_factory, backend_factory = _factories(session)
    cells = [Cell(model=MODEL, quantization="fp16", rung="S0", subject="sub-01")]
    out = run_matrix(cells, session_factory=session_factory,
                     backend_factory=backend_factory, out=tmp_path / "sweep",
                     verbose=False)
    written = next((out / "cells").glob("*.json"))
    stamp = written.stat().st_mtime_ns

    calls = []

    def counting_factory(cell):
        calls.append(cell)
        return ScriptedPlanner()

    run_matrix(cells, session_factory=session_factory, backend_factory=counting_factory,
               out=tmp_path / "sweep", verbose=False)
    assert calls == []                           # the cell was skipped
    assert written.stat().st_mtime_ns == stamp   # and not rewritten


def test_overwrite_reruns_a_finished_cell(session, tmp_path):
    session_factory, backend_factory = _factories(session)
    cells = [Cell(model=MODEL, quantization="fp16", rung="S2", subject="sub-01")]
    kwargs = {"session_factory": session_factory, "out": tmp_path / "sweep",
              "verbose": False}
    run_matrix(cells, backend_factory=backend_factory, **kwargs)

    calls = []

    def counting_factory(cell):
        calls.append(cell)
        return ScriptedPlanner()

    run_matrix(cells, backend_factory=counting_factory, overwrite=True, **kwargs)
    assert len(calls) == 1


def test_s0_never_loads_a_model(session, tmp_path):
    """The fixed pipeline must not touch the backend factory at all."""
    def explode(_cell):
        raise AssertionError("S0 must not construct a backend")

    cells = [Cell(model=MODEL, quantization="fp16", rung="S0", subject="sub-01")]
    out = run_matrix(cells, session_factory=lambda _s: session, backend_factory=explode,
                     out=tmp_path / "sweep", verbose=False)
    record = json.loads(next((out / "cells").glob("*.json")).read_text())
    assert record["ok"] is True
    assert record["backend_provenance"]["backend"].startswith("none")


# --------------------------------------------------------------------------
# Reading a finished sweep
# --------------------------------------------------------------------------


def test_summarise_keeps_failed_cells_as_rows(session, tmp_path):
    def sometimes_explode(cell):
        if cell.rung == "S3":
            raise RuntimeError("boom")
        return ScriptedPlanner()

    cells = [Cell(model=MODEL, quantization="fp16", rung=r, subject="sub-01")
             for r in ("S0", "S2", "S3")]
    out = run_matrix(cells, session_factory=lambda _s: session,
                     backend_factory=sometimes_explode, out=tmp_path / "sweep",
                     verbose=False)
    frame = summarise(out)
    assert len(frame) == 3
    assert frame["ok"].sum() == 2
    assert (~frame["ok"]).sum() == 1
    assert "error" in frame.columns


def test_load_cells_is_ordered(session, tmp_path):
    cells = [Cell(model=MODEL, quantization="fp16", rung=r, subject="sub-01")
             for r in ("S0", "S2")]
    out = run_matrix(cells, session_factory=lambda _s: session,
                     backend_factory=lambda _c: ScriptedPlanner(),
                     out=tmp_path / "sweep", verbose=False)
    ids = [r["cell"]["rung"] for r in load_cells(out)]
    assert ids == sorted(ids)


def test_timing_warnings_flags_a_cpu_only_sweep(session, tmp_path):
    cells = [Cell(model=MODEL, quantization="fp16", rung="S0", subject="sub-01")]
    out = run_matrix(cells, session_factory=lambda _s: session,
                     backend_factory=lambda _c: ScriptedPlanner(),
                     out=tmp_path / "sweep", verbose=False)
    warnings = timing_warnings(summarise(out))
    assert any("CPU" in w for w in warnings)


def test_timing_warnings_flags_mixed_accelerators():
    import pandas as pd

    frame = pd.DataFrame([
        {"ok": True, "gpu_name": "Tesla T4", "run_id": "a", "quantization": "4bit"},
        {"ok": True, "gpu_name": "NVIDIA L4", "run_id": "a", "quantization": "4bit"},
    ])
    warnings = timing_warnings(frame)
    assert any("different accelerators" in w for w in warnings)


def test_timing_warnings_flags_int8_on_a_pre_ampere_card():
    import pandas as pd

    frame = pd.DataFrame([
        {"ok": True, "gpu_name": "Tesla T4", "run_id": "a", "quantization": "8bit"},
    ])
    assert any("int8" in w for w in timing_warnings(frame))


def test_timing_warnings_is_quiet_for_one_card_one_sweep():
    import pandas as pd

    frame = pd.DataFrame([
        {"ok": True, "gpu_name": "NVIDIA A100-SXM4-40GB", "run_id": "a",
         "quantization": "fp16"},
        {"ok": True, "gpu_name": "NVIDIA A100-SXM4-40GB", "run_id": "a",
         "quantization": "4bit"},
    ])
    assert timing_warnings(frame) == []


def test_timing_warnings_says_so_when_nothing_succeeded():
    import pandas as pd

    frame = pd.DataFrame([{"ok": False, "gpu_name": "none", "run_id": "a",
                           "quantization": "fp16"}])
    assert timing_warnings(frame) == ["no successful cells"]


# -- the peak-contrast check, against the markings ----------------------------


def test_the_document_quotes_the_peak_contrast_table():
    """What the check costs on real recordings is a measurement with a file
    behind it, and the document's table is that file's cohort means."""
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    table = pd.read_csv(root / "data" / "benchmark" / "peak_contrast.csv")
    assert table["subject"].nunique() == 20
    assert sorted(table["floor_db"].unique()) == [0.0, 3.0, 6.0, 9.0]
    means = table.groupby(["band", "detector", "threshold_sd", "floor_db"])[
        ["accepted", "precision", "recall", "rank_rho"]].mean()
    text = (root / "docs" / "EVALUATION.md").read_text(encoding="utf-8")
    labels = {"rms": "rms", "line_length": "line length"}
    for (band, detector, threshold), check in (
            (("ripple", "rms", 2.0), {0.0: "off", 6.0: "6 dB"}),
            (("ripple", "rms", 5.0), {0.0: "off", 6.0: "6 dB"}),
            (("ripple", "line_length", 2.0), {0.0: "off", 6.0: "6 dB"}),
            (("fast_ripple", "rms", 5.0), {0.0: "off", 6.0: "6 dB"})):
        for floor, word in check.items():
            row = means.loc[(band, detector, threshold, floor)]
            line = (f"| {band.replace('_', ' ')} | {labels[detector]} | {threshold:.1f} SD | "
                    f"{word} | {row['accepted']:.0f} | {row['precision']:.3f} | "
                    f"{row['recall']:.3f} | {row['rank_rho']:.3f} |")
            assert line in text, line

    # The check never accepts more, and at the default it does not buy agreement.
    off = means.xs(0.0, level="floor_db")
    on = means.xs(6.0, level="floor_db")
    assert (on["accepted"] <= off["accepted"]).all()
    assert (on["recall"] <= off["recall"] + 1e-9).all()
    assert (on["precision"] - off["precision"]).abs().max() < 0.02

    events = pd.read_csv(root / "data" / "benchmark" / "peak_contrast_events.csv.gz")
    at = events[(events["band"] == "ripple") & (events["detector"] == "rms")
                & (events["threshold_sd"] == 2.0)]
    matched = (at[at["matched"]]["contrast_db"] < 6).mean()
    unmatched = (at[~at["matched"]]["contrast_db"] < 6).mean()
    assert f"{matched:.1%}" == "5.2%" and f"{unmatched:.1%}" == "6.2%"
    assert "5.2% of those an expert" in text and "and 6.2% of those none matches" in text
