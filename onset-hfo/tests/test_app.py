"""The reading interface's view model.

Offline, and deliberately free of Streamlit: everything the page decides
lives in :mod:`app.panels`, so all of it can be tested without a browser.
What the page itself does — arranging these results — is verified by driving
a real browser against a real server, which is not something to run in CI.

The rule under test throughout is the one in that module's docstring: the
interface shows what the pipeline wrote and does not compute a second answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app import panels


@pytest.fixture(scope="module")
def study_groups():
    """The committed *group* table, for checking the per-patient view against."""
    import pandas as pd

    return pd.read_csv(panels.STUDIES / "outcome_groups_300s.csv")


# -- finding analyses ------------------------------------------------------

def test_a_directory_without_events_is_not_an_analysis(tmp_path, store):
    """The cohort studies live in the same results folder and are not analyses.

    Offering `outcome_ds003498` in the picker and then failing to load it is
    worse than not offering it.
    """
    (tmp_path / "outcome_ds003498").mkdir()
    (tmp_path / "outcome_ds003498" / "groups.csv").write_text("a,b\n1,2\n")
    real = tmp_path / "sub-01_run-01"
    real.mkdir()
    for name in ("events.csv", "provenance.json", "report.json", "config.json"):
        (real / name).write_text("{}" if name.endswith(".json") else "channel\n")
    found = panels.find_results(tmp_path)
    assert [p.name for p in found] == ["sub-01_run-01"]


def test_a_missing_folder_is_empty_not_an_error(tmp_path):
    assert panels.find_results(tmp_path / "nope") == []


# -- the header ------------------------------------------------------------

def test_metadata_rows_name_the_window_that_was_analysed(store):
    rows = dict(panels.metadata_rows(store))
    assert rows["Subject"] == store.subject
    assert "s" in rows["Window analysed"]
    assert rows["Pipeline version"] != "?"


# -- the ranking, and the null hypothesis attached to it -------------------

def test_ranking_carries_the_intervals_not_just_the_rate(store):
    """Two channels whose intervals overlap are tied, so the page must show them."""
    table = panels.ranking_table(store)
    assert {"channel", "rank", "rate_per_min", "ci_low", "ci_high"} <= set(table.columns)
    assert table["rank"].tolist() == sorted(table["rank"].tolist())


def test_ranking_of_an_unknown_detector_is_empty_not_an_error(store):
    assert panels.ranking_table(store, "no_such_detector").empty


def test_the_leader_note_is_available_and_says_which_way(store):
    """The one computation the page is allowed, and the reason it is allowed."""
    note = panels.leader_note(store)
    assert note["available"] is True
    assert isinstance(note["distinguishable"], bool)
    assert note["statement"]
    assert note["n_tied_with_leader"] <= note["n_channels"]


def test_the_leader_note_degrades_rather_than_raising(store):
    assert panels.leader_note(store, "no_such_detector")["available"] is False


# -- evidence, and rebuilding the event behind it --------------------------

def test_evidence_rows_carry_a_citable_id(store):
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=3)
    if not len(table):
        pytest.skip("no accepted events on the leading channel of this fixture")
    assert table["evidence_id"].map(store.resolve).notna().all()


def test_the_rebuilt_event_carries_the_stored_measurements(store):
    """The figure prints these in its subtitle; losing them prints 'nan'.

    This is not hypothetical — the first version of the page built the event
    from times alone and drew ``peak nan Hz, nan dB over background`` directly
    beneath a panel showing 192 Hz and 12.5 dB.
    """
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events on the leading channel of this fixture")
    record = table.iloc[0].to_dict()
    event = panels.event_from_record(store, record)
    assert event.channel == record["channel"]
    assert event.start == pytest.approx(record["start"])
    assert event.peak_frequency_hz == pytest.approx(record["peak_frequency_hz"])
    assert event.spectral_prominence_db == pytest.approx(record["spectral_prominence_db"])
    assert not np.isnan(event.peak_frequency_hz)


def test_the_rebuilt_event_uses_the_band_the_detector_used(store):
    """Guessing the band would filter the trace differently from the measurement."""
    detector = store.detectors()[0]
    band = panels.detector_band(store, detector)
    assert band[0] < band[1]
    channel = store.list_channels(detector)[0]["channel"]
    table = panels.evidence_table(store, channel, detector=detector, k=1)
    if not len(table):
        pytest.skip("no accepted events for this detector in the fixture")
    assert panels.event_from_record(store, table.iloc[0].to_dict()).band == band


def test_a_null_in_the_record_does_not_become_a_crash(store):
    """Stored CSVs carry NaN, and the store turns those into None."""
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events in the fixture")
    record = {**table.iloc[0].to_dict(), "peak_frequency_hz": None, "n_peaks": None}
    event = panels.event_from_record(store, record)
    assert np.isnan(event.peak_frequency_hz) and event.n_peaks == 0


# -- citations -------------------------------------------------------------

def test_a_real_citation_resolves(store):
    channel = store.list_channels()[0]["channel"]
    table = panels.evidence_table(store, channel, k=1)
    if not len(table):
        pytest.skip("no accepted events in the fixture")
    rows = panels.citations(store, [table.iloc[0]["evidence_id"]])
    assert rows[0]["resolved"] is True and rows[0]["channel"] == channel


def test_an_invented_citation_is_flagged_rather_than_dropped(store):
    """A row that does not resolve is a guard failure worth seeing on the page."""
    rows = panels.citations(store, ["sub-xx|MADE-UP|rms|0.000"])
    assert rows[0]["resolved"] is False


# -- what ships with the source -------------------------------------------

def test_panels_never_imports_streamlit():
    """The whole reason `panels.py` exists, asserted rather than assumed.

    CI installs the `dev` extra, which has no Streamlit. A path constant put
    in `common.py` (which does import it) failed there while passing locally,
    because the local environment happened to have Streamlit installed. This
    is the guard that stops that recurring.
    """
    import ast

    tree = ast.parse(Path(panels.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported, \
        "app/panels.py must import no Streamlit: the tests run where it is absent"


def test_no_test_in_this_file_imports_the_streamlit_half_of_the_app():
    """The other direction of the same rule, and the one that bit twice.

    `test_panels_never_imports_streamlit` stops a constant drifting into
    `common.py`. It does not stop a *test* importing `common.py` to reach one,
    which passes locally -- where Streamlit happens to be installed -- and
    fails in CI, where the `dev` extra does not install it. Three tests did
    exactly that; the constants they wanted now live in `panels.py`.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app"):
            names = {alias.name for alias in node.names}
            assert "common" not in names and node.module != "app.common", \
                "this file must not import app.common: CI has no Streamlit"
        elif isinstance(node, ast.Import):
            assert not any(alias.name.startswith("app.common")
                           for alias in node.names), \
                "this file must not import app.common: CI has no Streamlit"

def test_the_example_analysis_ships_and_loads():
    """Every page must work on a fresh clone, with no download.

    The Onset prototype builds a synthetic cohort at startup; this one cannot
    generate real recordings, so a real analysis is committed instead.
    """
    from app.panels import EXAMPLE
    from onset_hfo.store import ResultStore

    assert EXAMPLE.exists(), "the shipped example analysis is missing"
    shipped = ResultStore(EXAMPLE)
    assert len(shipped.events) and shipped.detectors()
    assert str(shipped.provenance.get("source", "")).startswith("openneuro:"), \
        "the shipped example must be real data, not a simulation"


def test_the_committed_studies_are_readable():
    """The Outcome page reads these rather than rerunning a 90-minute sweep."""
    import pandas as pd

    from app.panels import STUDIES

    groups = pd.read_csv(STUDIES / "outcome_groups_300s.csv")
    assert {"source", "metric", "auc", "p_permutation", "p_bonferroni"} <= set(groups.columns)
    # Nothing survives correction. This assertion exists because the docs said
    # "nothing reaches p < 0.05" for two PRs after a metric was added that made
    # one row do so uncorrected; the claim the pages actually rest on is this
    # one, so it is the one under test.
    assert not (groups["p_bonferroni"] < 0.05).any(), \
        "a result now survives Bonferroni; every page claiming otherwise must change"
    uncorrected = int((groups["p_permutation"] < 0.05).sum())
    assert uncorrected <= len(groups) * 0.05 + 2, \
        "more uncorrected hits than chance explains; the pages must say so"


def test_a_gzipped_analysis_loads_like_a_plain_one(tmp_path, store):
    """The example ships gzipped, so the store has to read both."""
    import gzip
    import shutil

    from onset_hfo.store import ResultStore

    target = tmp_path / "gzipped"
    target.mkdir()
    for path in Path(store.dir).iterdir():
        if path.suffix == ".csv":
            with path.open("rb") as src, gzip.open(target / f"{path.name}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
        elif path.is_file():
            shutil.copy(path, target / path.name)
    zipped = ResultStore(target)
    assert len(zipped.events) == len(store.events)
    assert sorted(zipped.rates) == sorted(store.rates)


# -- re-loading the signal -------------------------------------------------

def test_a_synthetic_analysis_offers_no_reload(store):
    """There is no archive behind simulated data, so the page must not offer one."""
    spec = panels.reload_spec(store)
    if str(store.provenance.get("source", "")).startswith("openneuro:"):
        assert spec is not None and spec["t_start"] < spec["t_stop"]
    else:
        assert spec is None


def test_reload_spec_names_everything_fetch_slice_needs(store, monkeypatch):
    monkeypatch.setitem(store.provenance, "source", "openneuro:ds003029")
    monkeypatch.setitem(store.provenance, "slice_start_s", 50.0)
    monkeypatch.setitem(store.provenance, "slice_stop_s", 110.0)
    spec = panels.reload_spec(store)
    assert set(spec) == {"dataset", "subject", "run", "session", "task", "acq",
                         "t_start", "t_stop"}
    assert spec["dataset"] == "ds003029"


# -- the cohort, per patient -----------------------------------------------
#
# The Patients page shows one row per patient rather than one row per group,
# which is the first place in this project where a *single patient's* number
# is on screen. These tests exist because that is exactly where a number is
# easiest to over-read: they pin the join, and they pin the caveats that sit
# next to it.


def test_every_committed_cohort_table_is_readable_on_a_fresh_clone():
    """The page must work with no download, like every other page."""
    for name, expected in (("participants.csv", 20), ("recordings.csv", 20),
                           ("subjects.csv", 160), ("channels.csv", 1880)):
        table = panels.cohort_table(name)
        assert len(table) == expected, f"{name} has {len(table)} rows, expected {expected}"


def test_a_missing_cohort_table_is_empty_not_an_error():
    assert panels.cohort_table("no_such_table.csv").empty


def test_the_committed_tables_carry_no_demographics():
    """Age, sex and handedness are in the archive and deliberately not here.

    Twenty patients with pathology, surgical extent and outcome is already a
    small cohort; three demographic fields per patient narrow it considerably
    and no analysis in the project uses them. See `data/outcome/README.md`.
    """
    for name in ("participants.csv", "recordings.csv", "subjects.csv", "channels.csv"):
        columns = set(panels.cohort_table(name).columns)
        assert not columns & {"age", "sex", "hand", "handedness"}, \
            f"{name} carries a demographic column that was meant to be dropped"


def test_the_overview_has_one_row_per_patient_and_both_arms_on_it():
    overview = panels.cohort_overview()
    assert len(overview) == 20
    assert overview["subject"].is_unique
    for source in ("expert", "rms"):
        assert {f"top_resected_{source}", f"n_candidates_{source}",
                f"share_in_rz_{source}"} <= set(overview.columns)


def test_the_overview_reproduces_the_published_outcome_split():
    """The per-patient view must add up to the group number, or one of them is wrong.

    13 seizure-free against 7 recurrences is the cohort the whole study rests
    on; a join that silently dropped or duplicated a patient would still look
    like a table.
    """
    overview = panels.cohort_overview()
    assert int(overview["seizure_free"].sum()) == 13
    assert int((~overview["seizure_free"]).sum()) == 7


def test_the_overview_matches_the_group_table_it_sits_beside(study_groups):
    """Counting patients by hand must give the group table's own means.

    Two numbers disagreeing on one screen is the failure this project is
    organised against, and this page puts the rows and the means one click
    apart.
    """
    overview = panels.cohort_overview()
    for source in ("expert", "rms"):
        row = study_groups.query(
            "source == @source and band == 'fast_ripple' and scope == 'reviewed' "
            "and metric == 'top_channel_resected'")
        if row.empty:
            pytest.skip(f"no group row for {source}")
        hit = overview[f"top_resected_{source}"] == 1
        for column, mask in (("mean_seizure_free", overview["seizure_free"]),
                             ("mean_recurrence", ~overview["seizure_free"])):
            assert float(row.iloc[0][column]) == pytest.approx(
                float(hit[mask].mean()), abs=5e-3), \
                f"{source} {column} disagrees with the patients it is a mean of"


def test_the_overview_carries_resection_coverage_and_five_patients_are_short_of_it():
    """The column a view of this data must not hide.

    In five temporal-lobe patients only a quarter of the resected contacts were
    recorded, so their "share inside the resection" describes a quarter of
    their resection.
    """
    overview = panels.cohort_overview()
    thin = overview[overview["rz_coverage"] < 1.0]
    assert len(thin) == 5
    assert thin["rz_coverage"].tolist() == pytest.approx([0.25] * 5)


def test_a_band_scope_arm_that_does_not_exist_is_empty_not_a_wrong_answer():
    assert panels.cohort_overview(band="theta", scope="reviewed").empty or \
        panels.cohort_overview(band="theta", scope="reviewed")["top_resected_expert"].isna().all()


def test_subject_metrics_covers_both_bands_both_scopes_both_sources():
    rows = panels.subject_metrics("sub-01")
    assert len(rows) == 8
    assert set(rows["band"]) == {"ripple", "fast_ripple"}
    assert set(rows["scope"]) == {"reviewed", "all"}
    assert set(rows["source"]) == {"expert", "rms"}


def test_subject_metrics_for_someone_not_in_the_cohort_is_empty():
    assert panels.subject_metrics("sub-99").empty


def test_subject_channels_are_reviewed_zoned_and_busiest_first():
    rows = panels.subject_channels("sub-01")
    assert len(rows)
    assert rows["reviewed"].all(), "the page must only show channels that were scored"
    assert set(rows["zone"]) <= {"resected", "partial", "spared"}
    assert rows["expert_events"].is_monotonic_decreasing


def test_subject_channels_can_be_asked_for_the_unreviewed_ones_too():
    reviewed = panels.subject_channels("sub-01", reviewed_only=True)
    every = panels.subject_channels("sub-01", reviewed_only=False)
    assert len(every) > len(reviewed)


def test_a_thin_resection_produces_a_caveat_naming_the_coverage():
    """sub-02 had 4 of 16 resected contacts recorded; the page must say so."""
    notes = panels.subject_caveats("sub-02")
    assert any("25%" in note for note in notes), notes


def test_the_worst_tie_in_the_cohort_produces_a_caveat_naming_its_size():
    """sub-12's expert arm had 24 tied channels — the documented worst case."""
    notes = panels.subject_caveats("sub-12")
    assert any("24" in note and "expert" in note for note in notes), notes
    assert any("RMS arm" in note for note in notes), "the RMS arm must be named as RMS"


def test_a_patient_whose_answer_changed_between_minutes_gets_told_on():
    """Window instability is the study's own headline correction, per patient."""
    overview = panels.cohort_overview()
    unstable = overview[overview["stable_across_windows_expert"] == False]  # noqa: E712
    assert len(unstable), "no unstable patients: the stability tables did not join"
    notes = panels.subject_caveats(unstable.iloc[0]["subject"])
    assert any("changed between the five" in note for note in notes), notes


def test_every_patient_is_either_clean_or_carries_a_reason():
    """No patient may be silently uncaveated because a lookup missed.

    An empty list has to mean "the four checks passed", never "the join
    dropped this row" -- so every patient with a known problem must produce a
    note, and the clean ones must be clean for a checkable reason.
    """
    overview = panels.cohort_overview().set_index("subject")
    clean = 0
    for subject, row in overview.iterrows():
        notes = panels.subject_caveats(subject)
        problems = (row["rz_coverage"] < 1.0
                    or row["n_candidates_expert"] > 1 or row["n_candidates_rms"] > 1
                    or row["stable_across_windows_expert"] is False
                    or row["stable_across_windows_rms"] is False
                    or row["stable_across_runs_expert"] is False
                    or row["stable_across_runs_rms"] is False)
        assert bool(notes) == bool(problems), f"{subject}: {notes} vs {problems}"
        clean += not notes
    assert clean == 5, f"{clean} patients trip none of the four checks, expected 5"


def test_an_unknown_subject_gets_a_caveat_rather_than_an_empty_all_clear():
    """Silence must never be the answer for a patient we know nothing about."""
    assert panels.subject_caveats("sub-99") == [
        "No committed cohort tables for this subject."]


def test_the_page_defaults_to_the_arm_the_study_pre_specified():
    assert panels.PRIMARY == {"band": "fast_ripple", "scope": "reviewed"}


# -- the detector sweep ----------------------------------------------------
#
# The Detectors page used to carry the sweep as nine hand-typed rows. Every
# one of them was right; the sentence beneath them quoted a mean expert
# fast-ripple count that was wrong in three places at two different values.
# These tests pin the numbers the page and the docs quote against the files
# they are quoting, so the next drift fails here instead of on screen.


def test_the_committed_sweep_covers_both_bands_and_both_detectors():
    frame = panels.sweep()
    assert not frame.empty
    assert set(frame["band"]) == {"ripple", "fast_ripple"}
    assert set(frame["detector"]) == {"rms", "line_length"}
    assert {"precision", "recall", "f1", "detections", "rank_rho",
            "top5_overlap"} <= set(frame.columns)


def test_every_swept_rate_is_a_rate():
    frame = panels.sweep()
    for column in ("precision", "recall", "f1"):
        assert frame[column].between(0.0, 1.0).all(), f"{column} is out of range"
    assert (frame["detections"] > 0).all()
    assert frame["top5_overlap"].between(0.0, 5.0).all()


def test_detections_fall_as_the_threshold_rises():
    """A threshold that admitted more events as it tightened would be a bug."""
    frame = panels.sweep()
    for _, arm in frame.groupby(["band", "detector"]):
        ordered = arm.sort_values("threshold_sd")["detections"]
        assert ordered.is_monotonic_decreasing, "detections rose with the threshold"


def test_the_named_thresholds_are_the_ones_the_sweep_actually_picks():
    """`config.THRESHOLDS` and the committed sweep must not drift apart.

    The presets are the page's and the CLI's shared vocabulary. If a re-run
    moves an optimum, the name has to move with it or every document quoting
    "the measured one" is quoting something else.
    """
    from onset_hfo.config import THRESHOLDS

    by_rho = panels.operating_points("rank_rho").set_index(["band", "detector"])
    assert by_rho.loc[("ripple", "rms"), "threshold_sd"] == \
        THRESHOLDS["interictal-agreement"]
    assert by_rho.loc[("fast_ripple", "rms"), "threshold_sd"] == \
        THRESHOLDS["interictal-agreement-fast-ripple"]

    by_f1 = panels.operating_points("f1").set_index(["band", "detector"])
    assert by_f1.loc[("ripple", "rms"), "threshold_sd"] == \
        THRESHOLDS["interictal-recall"]


def test_the_rms_optima_are_interior_because_the_grid_was_extended_to_make_them_so():
    """The standard this project already applied to itself, as an assertion."""
    for criterion in ("rank_rho", "f1"):
        points = panels.operating_points(criterion).set_index(["band", "detector"])
        for band in ("ripple", "fast_ripple"):
            assert not points.loc[(band, "rms"), "at_boundary"], \
                f"{band} rms peaks on the edge of its grid by {criterion}"


def test_a_boundary_optimum_is_reported_rather_than_hidden():
    """Line length was never extended downwards, and the page has to say so.

    This is not a defect to fix by editing the table -- it is an arm that was
    not swept far enough, and the page labels it "not yet measured". The test
    exists so that labelling cannot silently disappear.
    """
    points = panels.operating_points("f1").set_index(["band", "detector"])
    assert points.loc[("ripple", "line_length"), "at_boundary"], \
        "the one known boundary optimum has moved; the page's warning must follow it"


def test_the_sweep_grid_records_that_the_arms_do_not_share_one():
    grid = panels.sweep_grid().set_index(["band", "detector"])
    assert grid.loc[("ripple", "rms"), "lowest"] == 1.0
    assert grid.loc[("fast_ripple", "rms"), "highest"] == 10.0
    assert grid.loc[("ripple", "line_length"), "lowest"] == 2.0, \
        "line length was never extended downwards; the page says so"


def test_the_reference_counts_are_committed_and_match_the_published_totals():
    """The number that was wrong in three places, now with a file behind it."""
    cohort = panels.benchmark_cohort()
    assert len(cohort) == 20
    assert int(cohort["n_expert_events"].sum()) == 41_187
    assert int(cohort["expert_ripples"].sum()) == 35_620
    assert int(cohort["expert_fast_ripples"].sum()) == 5_567
    assert (cohort["expert_ripples"] + cohort["expert_fast_ripples"]
            == cohort["n_expert_events"]).all(), "the bands do not sum to the total"


def test_the_expert_fast_ripple_mean_the_docs_quote_is_the_one_on_disk():
    """`docs/EVALUATION.md` said 228 and `config.py` said ~70. It is 278."""
    cohort = panels.benchmark_cohort()
    assert round(float(cohort["expert_fast_ripples"].mean())) == 278
    assert round(float(cohort["expert_ripples"].mean())) == 1781

    evaluation = (Path(__file__).resolve().parents[1] / "docs" / "EVALUATION.md").read_text()
    assert "mean of 278 expert-marked fast ripples" in evaluation
    assert "228 expert-marked" not in evaluation


def test_the_reviewed_channel_range_the_page_prints_is_the_one_on_disk():
    cohort = panels.benchmark_cohort()
    assert int(cohort["n_reviewed_channels"].min()) == 6
    assert int(cohort["n_reviewed_channels"].max()) == 65
    assert (cohort["n_reviewed_channels"] <= cohort["n_channels"]).all()


def test_the_sweep_is_empty_rather_than_raising_when_the_extract_is_gone(monkeypatch,
                                                                        tmp_path):
    monkeypatch.setattr(panels, "BENCHMARK", tmp_path)
    assert panels.sweep().empty
    assert panels.benchmark_cohort().empty
    assert panels.operating_points().empty
    assert panels.sweep_grid().empty


# -- the four things that must stay in step with berdakh/onset --------------
#
# DUPLICATION.md item 5 and item 8. Nothing can enforce a convention across two
# repositories, but the half that lives in this one can be enforced here: the
# banner must be built from the canonical constants rather than from a fourth
# copy of the sentence, and the wording the plan quotes must be the wording the
# code ships. A checklist the code already contradicts is worse than none.


def _duplication_plan() -> str:
    return (Path(__file__).resolve().parents[1] / "docs" / "DUPLICATION.md").read_text()


def test_the_disclaimer_is_assembled_from_the_canonical_constants():
    """No page may carry a second copy of the sentence that matters most.

    Read as text rather than imported: `app/common.py` imports Streamlit and
    CI does not install it. That is also why the constants live in
    `panels.py`.
    """
    source = (Path(__file__).resolve().parents[1] / "app" / "common.py").read_text()
    assert panels.DISCLAIMER_LEAD not in source, \
        "the banner restates the disclaimer instead of using DISCLAIMER_LEAD"
    assert "{DISCLAIMER_LEAD}" in source and "{DATA_SENTENCE}" in source, \
        "the banner does not interpolate the canonical constants"


def test_the_disclaimer_still_says_there_is_no_recommendation():
    """The one sentence this whole product is organised around."""
    whole = (f"{panels.DISCLAIMER_LEAD} {panels.DATA_SENTENCE} "
             f"{panels.DISCLAIMER_TAIL}")
    assert "not a medical device" in whole
    assert "no recommendation anywhere in this product" in whole
    assert "the clinician decides" in whole


def test_the_duplication_plan_quotes_the_wording_the_code_ships():
    """The plan is the other repository's only source for this text."""
    plan = _duplication_plan()
    assert panels.DATA_SENTENCE in plan, \
        "DUPLICATION.md quotes a DATA_SENTENCE the app no longer uses"
    assert "There is no " in panels.DISCLAIMER_TAIL


def test_the_plan_is_closed_out_rather_than_left_half_done():
    """Items 5-8 had two halves each; both sides have now shipped.

    This asserted `count("**half done**") == 4` while only this repository was
    reachable. The other half shipped in berdakh/onset#3, so the assertion is
    the opposite one: nothing in the table may still be waiting.
    """
    plan = _duplication_plan()
    assert "**half done**" not in plan and "| todo |" not in plan, \
        "an item is still open; the table must say which half"
    assert plan.count("**done**") >= 8, "the eight planned actions must all be recorded"
    assert "berdakh/onset#3" in plan, "the table must name the PR that closed items 4-8"


def test_the_theme_file_names_its_twin():
    theme = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
    assert theme.exists()
    assert "TWIN FILE" in theme.read_text()


def test_the_contributing_notes_carry_the_four_item_checklist():
    notes = (Path(__file__).resolve().parents[1] / "docs" / "CONTRIBUTING.md").read_text()
    assert "stay in step with" in notes
    for item in ("DATA_SENTENCE", ".streamlit/config.toml", "Architecture"):
        assert item in notes, f"the checklist does not name {item}"


def test_the_sidebar_opens_with_the_four_shared_destinations_in_order():
    """Checklist item 3, which item 3 of the plan says is done."""
    source = (Path(__file__).resolve().parents[1] / "app" / "common.py").read_text()
    names = ["Clinical guide", "Implementation walkthrough", "Results & docs",
             "Onset project"]
    positions = [source.index(f"[{name}]") for name in names]
    assert positions == sorted(positions), \
        "the shared link row is out of order; the two apps must match"
