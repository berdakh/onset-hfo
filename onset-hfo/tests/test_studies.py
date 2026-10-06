"""The study pages' text, built from the committed tables. No Qt.

What the site shows and what the window shows come from the same loaders;
these tests pin that the builders run on the committed extracts, that the
numbers in the text are the extracts' numbers, and that a missing site is
explained rather than crashed on.
"""

from __future__ import annotations

import os

import pytest

from onset_review import studies


@pytest.mark.parametrize("key", [k for k, _, _ in studies.STUDIES])
def test_every_study_page_builds_from_the_committed_tables(key):
    assert studies.available(), "the committed tables are part of the checkout"
    text = studies.build(key, subject="sub-01")
    assert text.startswith("# "), "a page starts with its title"
    assert "|---" in text, "every page carries at least one table"
    assert "could not be built" not in text


def test_the_detectors_page_reads_the_sweep_rather_than_restating_it():
    from app import panels

    text = studies.build("detectors", band="ripple", metric="rank_rho")
    sweep = panels.sweep()
    rms = sweep[(sweep.band == "ripple") & (sweep.detector == "rms")]
    at_two = float(rms[rms.threshold_sd == 2.0].rank_rho.iloc[0])
    assert f"ρ = {at_two:.2f}" in text, "the tuned ρ in the prose is the table's"
    assert "The two bands want different operating points" in text


def test_the_patients_page_derives_its_counts_from_the_cohort():
    rows = studies.patient_rows()
    hit = int((rows["top_resected_expert"] == 1).sum())
    text = studies.build("patients")
    assert f"| expert markings | {hit}/{len(rows)} |" in text
    text = studies.build("patients", subject=str(rows.subject.iloc[0]))
    assert f"## One patient: {rows.subject.iloc[0]}" in text
    assert "What each source said about this patient" in text


def test_a_non_primary_arm_is_flagged_as_such():
    assert "not the arm the study pre-specified" in studies.build(
        "patients", band="ripple", scope="all")
    assert "not the arm the study pre-specified" not in studies.build("patients")


def test_the_research_page_quotes_the_robustness_and_montage_extracts():
    from app import panels

    text = studies.build("research")
    montage = panels.montage_groups()
    rho = montage[(montage.kind == "agreement") & (montage.metric == "spearman_rho")
                  & (montage.band == "ripple") & (montage.montage == "bipolar")].value
    assert f"{float(rho.iloc[0]):.3f} bipolar" in text
    assert "0.753 plain" in text


def test_cached_window_for_picks_the_earliest_of_the_subject():
    import pandas as pd

    windows = pd.DataFrame([
        {"dataset": "ds003498", "subject": "sub-04", "t_start": 120.0, "t_stop": 180.0},
        {"dataset": "ds003498", "subject": "sub-04", "t_start": 0.0, "t_stop": 60.0},
        {"dataset": "ds003029", "subject": "sub-04", "t_start": 0.0, "t_stop": 60.0},
    ])
    assert studies.cached_window_for("sub-04", windows)["t_start"] == 0.0
    assert studies.cached_window_for("sub-05", windows) is None
    assert studies.cached_window_for("sub-04", None) is None


def test_a_missing_site_is_explained_not_raised(monkeypatch):
    monkeypatch.setattr(studies, "_panels", lambda: None)
    assert studies.available() is False
    text = studies.build("outcome")
    assert "not on this machine" in text and studies.SITE in text
    assert studies.build("data").startswith("# The data"), "static pages need no tables"


def test_an_unknown_page_is_an_error():
    with pytest.raises(KeyError):
        studies.build("methods")


def test_the_markdown_table_helper_escapes_and_rounds():
    import pandas as pd

    frame = pd.DataFrame({"name": ["a|b"], "value": [0.123456], "count": [3.0], "flag": [True]})
    text = studies.table(frame)
    assert "a\\|b" in text and "0.123" in text and "| 3 |" in text
    assert studies.table(frame.iloc[0:0]) == "*(no rows)*\n"


def test_the_loaders_are_read_from_their_file_not_through_the_sites_package():
    """`app/__init__` imports Streamlit, which the desktop does not ship; the
    loaders must not need it."""
    studies._loaded.clear()
    panels = studies._panels()
    assert panels is not None and panels.__name__ == "onset_site_panels"
    assert panels.__file__.endswith("app/panels.py")


def test_an_installed_site_is_found_by_the_environment_and_sets_the_data_root(
        tmp_path, monkeypatch):
    """A release bundle puts the loaders and the tables under site/; the
    reviewer must read both from there, not from a checkout it does not have."""
    import shutil

    from onset_hfo.config import PROJECT_ROOT

    site = tmp_path / "site"
    (site / "app").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "app" / "panels.py", site / "app" / "panels.py")
    (site / "data").mkdir()
    shutil.copytree(PROJECT_ROOT / "data" / "outcome", site / "data" / "outcome")
    monkeypatch.setenv("ONSET_REVIEW_SITE_DIR", str(site))
    monkeypatch.delenv("ONSET_HFO_DATA_ROOT", raising=False)
    studies._loaded.clear()
    try:
        assert studies.site_root() == site
        panels = studies._panels()
        assert panels is not None
        assert str(panels.COHORT).startswith(str(site)), "the tables come from the site"
        assert studies.available()
        assert "## One patient: sub-01" in studies.build("patients", subject="sub-01")
        assert "not on this machine" not in studies.build("data")
    finally:
        # `_panels` set this itself, so it is cleared the same way: through
        # monkeypatch it would be *restored* to the site's value at teardown.
        studies._loaded.clear()
        os.environ.pop("ONSET_HFO_DATA_ROOT", None)


def test_no_site_anywhere_means_the_pages_say_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_SITE_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setattr(studies, "INSTALLED_SITE", tmp_path / "nothing")
    import onset_hfo.config as config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "bare")
    studies._loaded.clear()
    try:
        assert studies.site_root() is None
        assert "not on this machine" in studies.build("outcome")
    finally:
        studies._loaded.clear()
