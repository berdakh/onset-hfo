"""The study pages' own figures: drawn from the tables beside them, embedded
in the page, and never the reason a page fails to build. No Qt."""

from __future__ import annotations

from pathlib import Path

import pytest

from onset_review import studies, studycharts


@pytest.fixture(scope="module")
def panels():
    module = studies._panels()
    assert module is not None, "the committed tables are part of the checkout"
    return module


def test_the_sweep_chart_draws_one_line_per_detector_with_the_chosen_point(panels, tmp_path):
    frame = panels.sweep()
    best = panels.operating_points("rank_rho")
    out = studycharts.sweep_chart(frame, "ripple", "rank_rho", best, tmp_path / "sweep.png")
    assert out.exists() and out.stat().st_size > 10_000
    header = out.read_bytes()[:8]
    assert header.startswith(b"\x89PNG")


def test_the_outcome_chart_draws_every_patient(panels, tmp_path):
    import pandas as pd

    cohort = panels.cohort_overview()
    groups = pd.read_csv(Path(panels.STUDIES) / "outcome_groups_300s.csv").query(
        "scope == 'reviewed' and band == 'fast_ripple'")
    out = studycharts.outcome_chart(cohort, groups, tmp_path / "outcome.png")
    assert out.exists() and out.stat().st_size > 10_000
    # Without the groups table the chart still draws, just without the AUC.
    out = studycharts.outcome_chart(cohort, None, tmp_path / "bare.png")
    assert out.exists()


def test_the_pages_embed_their_figures():
    detectors = studies.build("detectors", band="ripple", metric="rank_rho")
    images = [line for line in detectors.splitlines() if line.startswith("![")]
    assert any("threshold sweep" in line for line in images)
    assert all(Path(line.split("](", 1)[1].rstrip(")")).exists() for line in images)
    outcome = studies.build("outcome")
    images = [line for line in outcome.splitlines() if line.startswith("![")]
    assert any("Every patient" in line for line in images)
    assert "Each dot is one patient" in outcome
    # The sweep's figure follows the page's choices.
    other = studies.build("detectors", band="fast_ripple", metric="f1")
    assert "sweep-fast_ripple-f1" in other and "sweep-ripple-rank_rho" in detectors


def test_a_figure_that_cannot_be_drawn_leaves_the_page_whole(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("no renderer")

    monkeypatch.setattr(studycharts, "sweep_chart", broken)
    text = studies.build("detectors")
    assert text.startswith("# Detectors") and "threshold (SD)" in text
    assert "threshold sweep" not in text, "no image line for a figure that failed"


def test_the_series_hues_are_the_validated_pair():
    assert studycharts.SERIES == ("#0B6BCB", "#C2410C")
    assert studycharts.SERIES[0].lower() in open(
        Path(__file__).resolve().parents[1] / "onset_review" / "theme.py").read().lower(), \
        "the first hue is the window's accent"
