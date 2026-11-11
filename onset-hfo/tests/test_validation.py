"""The Validation page: what each method was measured to do.

What has to hold:

* every figure the page states is the committed tables' own, rounded as
  written, so the page cannot drift from the studies behind it;
* the page reads the same with no tables at all (the wheel carries none);
* the open window is scored exactly as the detector study scored a patient
  (markings equal to its own detections score 1.0), caveated as one window,
  and absent when the window has no markings;
* the sidebar has Validation at its foot, and the quick start guide is under
  Help.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pandas as pd
import pytest

from onset_review import thisrecording, validation
from onset_review.session import ReviewRequest, session_from_recording

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
needs_tables = pytest.mark.skipif(not (DATA / "stability").is_dir(),
                                  reason="the study tables are not in this checkout")


def _text(key: str) -> str:
    method = next(m for m in validation.METHODS if m.key == key)
    extras = tuple(value for kind, value in method.extras if kind == "text")
    return " ".join((method.data,) + method.findings + extras + method.limits)


# -- the figures are the tables' ---------------------------------------------------------------
@needs_tables
def test_the_detector_figures_are_the_benchmark_s():
    from onset_review import studytables

    best = studytables.operating_points("rank_rho").set_index(["band", "detector"])
    text = _text("detectors")
    for band, name in (("ripple", "ripples"), ("fast_ripple", "fast ripples")):
        rows = best.loc[band].sort_values("rank_rho", ascending=False)
        top = rows.iloc[0]
        assert rows.index[0] == "rms"
        assert f"{top['rank_rho']:.2f} for {name}" in text
        assert f"at {top['threshold_sd']:g} SD" in text
        assert f"{top['precision']:.2f}" in text
    events = int(studytables.benchmark_cohort()["n_expert_events"].sum())
    assert f"{events:,} expert-marked HFOs" in text
    assert f"{len(studytables.benchmark_cohort())} patients" in text


@needs_tables
def test_the_outcome_figures_are_the_outcome_study_s():
    from onset_hfo.outcome import min_detectable_auc

    groups = pd.read_csv(DATA / "stability" / "outcome_groups_300s.csv")
    arm = groups.query("scope == 'reviewed' and band == 'fast_ripple' "
                       "and metric == 'top_channel_resected'").set_index("source")
    text = _text("outcome")
    expert, rms = arm.loc["expert"], arm.loc["rms"]
    assert (f"{expert['mean_seizure_free']:.0%} of the seizure-free patients and "
            f"{expert['mean_recurrence']:.0%} of the others") in text
    assert (f"AUC {expert['auc']:.2f} (95% interval {expert['auc_lo']:.2f}–"
            f"{expert['auc_hi']:.2f}), permutation p = {expert['p_permutation']:.2f}") in text
    assert (f"AUC {rms['auc']:.2f} ({rms['auc_lo']:.2f}–{rms['auc_hi']:.2f}), "
            f"p = {rms['p_permutation']:.2f}") in text
    n_sf, n_rec = int(expert["n_seizure_free"]), int(expert["n_recurrence"])
    assert f"{n_sf} seizure-free and {n_rec} with recurrence" in text
    assert (groups["p_permutation"] < 0.05).sum() == 0, "none, even before correction"
    assert "none of the study's comparisons reaches p < 0.05 even before correction" in text
    assert (groups["p_bonferroni"] >= 0.05).all(), "none survives correction"
    assert f"is {min_detectable_auc(n_sf, n_rec):.2f}" in text
    runs = pd.read_csv(DATA / "stability" / "decision_stability_across_runs.csv")
    runs = runs.query("band == 'fast_ripple' and metric == 'top_channel_resected'")
    stable = runs.groupby("source")["stable"].agg(["sum", "count"])
    assert (f"{stable.loc['expert', 'sum']} of {stable.loc['expert', 'count']} patients on "
            f"the expert markings and {stable.loc['rms', 'sum']} of "
            f"{stable.loc['rms', 'count']} with the RMS") in text
    hup = json.loads((DATA / "hup" / "interictal" / "summary.json").read_text())["bands"]
    ripple = hup["ripple"]
    soz = ripple["auc_soz"]
    assert f"{ripple['patients']} interictal recordings" in text
    assert f"median AUC {soz['median']:.2f}, {soz['ci_low']:.2f}–{soz['ci_high']:.2f}" in text
    share = ripple["outcome"]["share_in_rz"]
    assert (f"{share['n_seizure_free']} seizure-free patients from the "
            f"{share['n_recurrence']} with recurrence") in text


@needs_tables
def test_the_ictal_figures_are_the_ictal_study_s():
    from onset_review.casereport import VALIDATION

    ictal = json.loads((DATA / "ictal" / "summary.json").read_text())
    control = json.loads((DATA / "ictal" / "control_summary.json").read_text())
    hup = json.loads((DATA / "hup" / "ictal" / "summary.json").read_text())
    hup_control = json.loads((DATA / "hup" / "ictal" / "control_summary.json").read_text())
    text = _text("ictal")
    block = ictal["ei_all"]
    assert f"{ictal['patients']} patients" in text
    assert (f"AUC {block['median']:.2f} (95% interval {block['ci_low']:.2f}–"
            f"{block['ci_high']:.2f})") in text
    assert f"energy ratio alone reaches {ictal['er_all']['median']:.2f}" in text
    per_seizure, per_patient = control["ei_per_seizure"], control["ei_all"]
    assert (f"falls to {per_seizure['median']:.2f} over the {per_seizure['n']} seizures "
            f"where that could be tested ({per_patient['median']:.2f} over their "
            f"{per_patient['n']} patients)") in text
    assert f"in only {round(ictal['top_in_zone_patients'] * 28)} of 28" in text
    assert (f"{hup['ei_all']['n']} patients: median AUC {hup['ei_all']['median']:.2f} "
            f"({hup['ei_all']['ci_low']:.2f}–{hup['ei_all']['ci_high']:.2f}), and "
            f"{hup_control['ei_all']['median']:.2f} at 40 s before") in text
    # The case report says the same in one line.
    assert f"AUC {block['median']:.2f}" in VALIDATION["ictal"]


@needs_tables
def test_the_template_figures_are_the_template_study_s():
    template = json.loads((DATA / "template" / "summary.json").read_text())
    text = _text("template")
    own, pitch = template["variants"]["own"], template["variants"]["3.5mm"]
    assert f"{template['patients']} SEEG patients" in text
    assert f"median {own['median_mm']:.1f} mm off" in text
    assert f"{own['within_5mm']:.0%} of contacts within 5 mm" in text
    assert f"the median is {pitch['median_mm']:.1f} mm" in text
    assert f"only {template['real_labels']['in'] * 100:.0f}% of" in text


@needs_tables
def test_the_ties_and_the_short_window_are_the_outcome_study_s():
    from onset_review import studytables

    text = _text("outcome")
    cohort = studytables.cohort_overview(band="fast_ripple", scope="reviewed")
    expert, rms = cohort["n_candidates_expert"], cohort["n_candidates_rms"]
    assert f"one channel in only {int((expert == 1).sum())} of {len(cohort)}" in text
    assert f"{int(expert.max())} channels are tied" in text
    assert f"tied in {int((rms > 1).sum())} of {len(cohort)}" in text
    groups = pd.read_csv(DATA / "stability" / "outcome_groups_300s.csv").query(
        "scope == 'reviewed' and band == 'fast_ripple' and metric == 'candidates_resected'"
    ).set_index("source")
    for source in ("expert", "rms"):
        row = groups.loc[source]
        assert (f"{row['auc']:.2f} ({row['auc_lo']:.2f}–{row['auc_hi']:.2f}), "
                f"p = {row['p_permutation']:.2f}") in text, source
    curve = pd.read_csv(DATA / "stability" / "curve.csv").query("source == 'expert'")
    first = curve[curve["window_s"] == 60].iloc[0]
    whole = curve[curve["window_s"] == 300].iloc[0]
    assert f"AUC {first['auc']:.2f}, p = {first['p_permutation']:.3f}" in text
    assert f"give {whole['auc']:.2f}, p = {whole['p_permutation']:.2f}" in text
    spread = pd.read_csv(DATA / "stability" / "spread.csv").set_index("source").loc["expert"]
    assert int(spread["n_windows"]) == 5
    assert (f"AUC ranges from {spread['auc_min']:.2f} to {spread['auc_max']:.2f} and p "
            f"from {spread['p_min']:.3f} to {spread['p_max']:.2f}") in text


@needs_tables
def test_every_figure_shows_the_numbers_the_tables_hold_now():
    """A figure ships as a picture; the digest beside it is of the numbers it
    plots. Recomputed here from the tables, so a table that changes without
    `python -m onset_review.validationfigures` being run fails here."""
    from onset_review import validationfigures as figures

    shipped = figures.shipped_digests()
    assert set(shipped) == set(figures.FIGURES)
    for key in figures.FIGURES:
        assert figures.digest(figures.figure_data(key, DATA)) == shipped[key], (
            f"the {key} figure is out of date: run python -m onset_review.validationfigures")
        path = figures.path_for(key)
        assert path is not None and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_figure_draws_from_the_tables(tmp_path):
    pytest.importorskip("matplotlib")
    from onset_review import validationfigures as figures

    if not (DATA / "ictal").is_dir():
        pytest.skip("the study tables are not in this checkout")
    path = figures.draw("ictal", tmp_path / "ictal.png", DATA)
    assert path.stat().st_size > 10_000


def test_the_page_names_every_method_its_data_its_failures_and_its_study():
    text = validation.markdown()
    assert text.startswith("# Validation")
    assert "not validated for clinical use" in text
    for method in validation.METHODS:
        assert f"## {method.title}" in text
        assert method.limits and method.findings
        assert f"`{method.doc}`" in text
        assert (ROOT / method.doc).is_file(), method.doc
        assert "OpenNeuro ds00" in method.data
    assert "## This recording" not in text, "nothing open, nothing scored"
    from onset_review.validationfigures import FIGURES, path_for

    for key, caption in FIGURES.items():
        assert f"![{caption}]({path_for(key).as_posix()})" in text, key


# -- this recording ----------------------------------------------------------------------------
@pytest.fixture
def window(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration),
                                 detectors=("rms", "line_length")))


def _marked(session):
    """The window with its own RMS detections as its expert markings."""
    accepted = [e for e in session.events if e.detector == "rms" and e.accepted]
    truth = pd.DataFrame({"channel": [e.channel for e in accepted],
                          "start": [e.start for e in accepted],
                          "stop": [e.stop for e in accepted], "kind": "ripple"})
    session.recording = types.SimpleNamespace(**{**vars(session.recording),
                                                 "ground_truth": truth})
    session.reviewed_channels = list(session.raw.ch_names)
    return session


def test_a_window_is_scored_the_way_the_study_scored_a_patient(window):
    assert validation.this_recording(window) == "", "no markings, no score"
    session = _marked(window)
    scores = thisrecording.score_window(session).set_index("detector")
    rms = scores.loc["rms"]
    assert (rms.precision, rms.recall, rms.f1, rms.rank_rho) == (1.0, 1.0, 1.0, 1.0), \
        "markings equal to the detections score perfectly"
    assert rms.threshold_sd == thisrecording.threshold_for(session, "rms")
    assert 0 < scores.loc["line_length"].f1 < 1
    text = validation.markdown(session)
    assert "## This recording" in text and thisrecording.CAVEAT in text
    assert text.index("## This recording") < text.index("## Finding HFOs")
    assert "| RMS energy |" in text and "| Line length |" in text


# -- the window --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_validation_is_one_place_at_the_foot_and_quick_start_is_under_help(qapp, tmp_path,
                                                                             monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    from onset_review import guide
    from onset_review.pages import PageWindow, sections

    assert [key for key, _label, _pages in sections()][-2:] == ["validation", "code"]
    window = PageWindow()
    try:
        assert window.show_page("validation")
        assert window.current_page() == "validation"
        text = window._pages["validation"].view.toPlainText()
        assert "Validation" in text and "Finding HFOs" in text
        assert not window.show_page("quickstart"), "the guide is not a page any more"
        assert guide.run_target(window, "page:quickstart")
        dialog = window._quickstart_dialog
        assert dialog.isVisible() and "Validation" in dialog.page.browser.toPlainText()
        assert guide.show_quickstart(window) is dialog, "a second F1 brings it forward"
        dialog.close()
    finally:
        window.close()
