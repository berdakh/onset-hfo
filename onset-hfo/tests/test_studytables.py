"""The study pages' view model (:mod:`onset_review.studytables`).

Offline and free of any interface toolkit: everything a study page decides
lives in that module, so all of it can be tested without a display.

The rule under test throughout is the one in that module's docstring: the
interface shows what the pipeline wrote and does not compute a second answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onset_review import studytables as panels


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

def test_the_study_tables_import_no_interface_toolkit():
    """The view model stays testable without a display.

    CI's main job installs the `dev` extra, which has no Qt; a loader that
    imported a toolkit would pass locally and fail there.
    """
    import ast

    tree = ast.parse(Path(panels.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"streamlit", "qtpy", "PySide6", "PyQt5", "PyQt6"}, \
        "studytables.py must import no interface toolkit: the tests run where it is absent"



def test_the_example_analysis_ships_and_loads():
    """Every page must work on a fresh clone, with no download.

    The Onset prototype builds a synthetic cohort at startup; this one cannot
    generate real recordings, so a real analysis is committed instead.
    """
    from onset_hfo.store import ResultStore
    from onset_review.studytables import EXAMPLE

    assert EXAMPLE.exists(), "the shipped example analysis is missing"
    shipped = ResultStore(EXAMPLE)
    assert len(shipped.events) and shipped.detectors()
    assert str(shipped.provenance.get("source", "")).startswith("openneuro:"), \
        "the shipped example must be real data, not a simulation"


def test_the_committed_studies_are_readable():
    """The Outcome page reads these rather than rerunning a 90-minute sweep."""
    import pandas as pd

    from onset_review.studytables import STUDIES

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


def test_the_overview_carries_resection_coverage_and_every_resection_is_recorded():
    """The column a view of this data must not hide.

    Through 0.3.24 it read 0.25 for five temporal-lobe patients: their
    resected hippocampal and entorhinal contacts were recorded under other
    names than the clinical sheet's, and nothing mapped one onto the other
    (`onset_hfo/archive_electrodes.csv`). Every resected contact is now found.
    """
    overview = panels.cohort_overview()
    assert "rz_coverage" in overview.columns
    assert overview["rz_coverage"].tolist() == pytest.approx([1.0] * len(overview))


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


def test_a_thin_resection_produces_a_caveat_naming_the_coverage(monkeypatch):
    """A patient with part of the resection unrecorded; the page must say so.

    No patient in the committed tables is one any more, so the overview is
    given one: sub-02 as it read through 0.3.24, 4 of 16 contacts.
    """
    overview = panels.cohort_overview()
    thin = overview.assign(rz_coverage=overview["rz_coverage"].where(
        overview["subject"] != "sub-02", 0.25))
    monkeypatch.setattr(panels, "cohort_overview", lambda *a, **k: thin)
    notes = panels.subject_caveats("sub-02")
    assert any("25%" in note for note in notes), notes
    assert not any("%" in note and "resected were" in note
                   for note in panels.subject_caveats("sub-01"))


def test_the_worst_tie_in_the_cohort_produces_a_caveat_naming_its_size():
    """sub-12's expert arm has 27 tied channels — the documented worst case
    (24 under the approximate interval used before 0.3.17)."""
    notes = panels.subject_caveats("sub-12")
    assert any("27" in note and "expert" in note for note in notes), notes
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

    The ripple presets were measured before five patients' electrodes were
    read correctly; since then the optimum of both criteria is 1.75 SD. The
    presets stay, because the outcome and HUP studies run at them as operating
    points fixed beforehand, and the documents say so. What this pins is that
    they remain within 0.02 of the optimum they are named after.
    """
    from onset_hfo.config import THRESHOLDS

    sweep = panels.sweep().set_index(["band", "detector", "threshold_sd"])
    for criterion, band, name in (("rank_rho", "ripple", "interictal-agreement"),
                                  ("rank_rho", "fast_ripple", "interictal-agreement-fast-ripple"),
                                  ("f1", "ripple", "interictal-recall")):
        best = panels.operating_points(criterion).set_index(["band", "detector"])
        top = best.loc[(band, "rms"), criterion]
        at_preset = sweep.loc[(band, "rms", THRESHOLDS[name]), criterion]
        assert top - at_preset <= 0.02, f"{name} is {top - at_preset:.3f} below the optimum"


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
    assert int(cohort["n_reviewed_channels"].min()) == 14
    assert int(cohort["n_reviewed_channels"].max()) == 65
    assert (cohort["n_reviewed_channels"] <= cohort["n_channels"]).all()


def test_the_sweep_is_empty_rather_than_raising_when_the_extract_is_gone(monkeypatch,
                                                                        tmp_path):
    monkeypatch.setattr(panels, "BENCHMARK", tmp_path)
    assert panels.sweep().empty
    assert panels.benchmark_cohort().empty
    assert panels.operating_points().empty
    assert panels.sweep_grid().empty


# -- the disclaimer ------------------------------------------------------------






def test_the_disclaimer_still_says_there_is_no_recommendation():
    """The one sentence this whole product is organised around."""
    whole = (f"{panels.DISCLAIMER_LEAD} {panels.DATA_SENTENCE} "
             f"{panels.DISCLAIMER_TAIL}")
    assert "not a medical device" in whole
    assert "no recommendation anywhere in this product" in whole
    assert "the clinician decides" in whole












# The four-detector family and the two studies that followed it
# --------------------------------------------------------------------------
# The pages quote these numbers in prose. The point of reading them from a
# committed file is that the file is the owner; these tests pin the agreement
# between the two, so a sweep that changes the answer fails here rather than
# leaving a stale sentence on a deployed page.


def test_the_family_sweep_covers_four_detectors_in_both_bands():
    best = panels.family_operating_points()
    assert set(best["detector"]) == {"rms", "line_length", "hilbert",
                                     "short_time_energy"}
    assert set(best["band"]) == {"ripple", "fast_ripple"}
    assert len(best) == 8
    # Each detector at its own threshold, so the thresholds must not be uniform.
    assert best["threshold_sd"].nunique() > 1


def test_adding_two_detectors_bought_nothing_and_the_page_says_so():
    """The Detectors page says the margins are within what twenty patients
    resolve. This is the size half of that claim; the per-patient bootstrap
    half is in tests/test_detector_ensemble.py."""
    margin = panels.family_margin()
    assert set(margin) == {"ripple", "fast_ripple"}
    for band, m in margin.items():
        assert 0 <= m["margin"] < 0.03, (
            f"{band} margin is now {m['margin']:+.3f}; the Detectors page says "
            f"the added features bought nothing and would need rewriting")




def test_the_fast_ripple_coupled_population_is_the_unmeasurable_one():
    """Why the Outcome page reports the split in the ripple band."""
    meas = panels.subpopulation_measurability()
    coupled = meas[meas["population"] == "spike_coupled"].set_index("band")
    assert coupled.loc["fast_ripple", "median"] <= 10
    assert coupled.loc["ripple", "median"] > 5 * max(
        coupled.loc["fast_ripple", "median"], 1)
    assert coupled.loc["fast_ripple", "windows_under_5"] > 30


def test_the_subpopulation_screen_keeps_every_patient_in_the_ripple_band():
    rows = panels.subpopulation_outcome("ripple")
    assert rows["subject"].nunique() == 20
    assert set(rows["population"]) == {"merged", "spike_coupled", "independent"}
    assert (rows["band"] == "ripple").all()


def test_an_empty_ranking_is_now_only_an_empty_window():
    """What is left after the rule stopped firing on a silent re-test.

    Before the rule was fixed a silent re-test conflated two things: windows
    where a working ranking was zeroed (14 of 600) and windows with no events
    at the survey threshold either (15). Only the second kind remains, and it
    is purely a fast-ripple sparsity statement. (84, 36 and 48 before five
    patients' 24 reviewed channels were read instead of 6.)
    """
    collapse = panels.robustness_collapse()
    assert collapse["windows"] == 600
    assert collapse["collapsed"] == 15
    assert collapse["by_band"] == {"fast_ripple": 15}, \
        "an empty ripple window would be new and worth looking at"


def test_a_silent_re_test_no_longer_destroys_a_ranking():
    """The regression that matters, read off the published extract: every
    window whose stricter pass found nothing keeps the survey's ranking."""
    rows = panels.robustness_ablation()
    silent = rows[(rows["rule"] == "multiplied") & (rows["retest_silent"] == 1)]
    assert len(silent) == 29
    rescued = silent[silent["n_events_survey"] > 0]
    assert len(rescued) == 14
    assert (rescued["mean_robustness"] == 1.0).all()
    assert (rescued["score_mass"] > 0).all()






def test_the_two_sweeps_agree_wherever_they_overlap():
    """The Detectors page tells the reader this, so it must stay true.

    The page shows two tables whose line-length optima differ (2.5 SD against
    1.5 SD), and explains the difference as a grid that stopped too high rather
    than a disagreement. That explanation is only honest while the shared cells
    actually match.
    """
    import pandas as pd

    key = ["band", "detector", "threshold_sd"]
    shared = panels.sweep().merge(panels.family_sweep(), on=key,
                                  suffixes=("_a", "_f"))
    assert len(shared) >= 27, f"only {len(shared)} shared cells"
    for column in ("rank_rho", "f1", "precision", "recall"):
        delta = (shared[f"{column}_a"] - shared[f"{column}_f"]).abs().max()
        assert delta == pytest.approx(0.0, abs=1e-9), \
            f"the two sweeps now differ by {delta} on {column}"
    assert isinstance(shared, pd.DataFrame)




def test_the_committed_group_tables_match_what_the_scripts_compute():
    """The extracts must be regenerable, not just present."""
    groups = panels.subpopulation_groups("ripple").set_index(
        ["metric", "population"])["auc"].round(3)
    assert groups[("top_channel_resected", "merged")] == pytest.approx(0.747)
    assert groups[("top_channel_resected", "spike_coupled")] == pytest.approx(0.533)
    assert groups[("share_in_rz", "merged")] == pytest.approx(0.516)

    rob = panels.robustness_groups()
    ripple = rob[(rob.band == "ripple")
                 & (rob.metric == "tied_set_argmax_resected")]
    plain = float(ripple[ripple.rule == "plain"].auc.iloc[0])
    multiplied = ripple[ripple.rule == "multiplied"].auc
    assert round(plain, 3) == pytest.approx(0.736)
    assert (multiplied > plain).all(), \
        "the Research page says the rule helps in the ripple band at every point"






def test_the_montage_extract_keeps_every_patient_in_every_arm():
    """§3c's premise: the three arms are scored on the same 20 patients, so a
    difference between them is a difference of montage and nothing else."""
    rows = panels.montage_screen()
    assert not rows.empty
    kept = rows.groupby(["band", "montage"]).subject.nunique()
    assert (kept == 20).all(), kept.to_dict()
    groups = panels.montage_groups()
    outcome = groups[groups["kind"] == "outcome"]
    assert (outcome["n_SF"] == 13).all() and (outcome["n_rec"] == 7).all()


