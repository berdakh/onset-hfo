"""The desktop reviewer's numbers, checked without a display.

Every figure a clinician reads off the screen is computed in
`onset_review.session`, `trends` and `report`, none of which import Qt. So they
are tested here, in a file that runs on any machine with the base install --
including the CI job that does not have the `review` extra.

That separation is the reason this file exists at all. These tests used to live
beside the Qt ones, under a module-level `importorskip`; a skip there takes the
whole module with it, so on a machine without Qt the numbers were not checked
at all while the suite reported green. `tests/test_review_app.py` now holds only
what genuinely needs a window.

Everything runs on the synthetic recording, so none of it needs a cached slice
of `ds003498` or a network. That is why `session_from_recording` exists as a
function separate from `load_session`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_review import report, trends
from onset_review.session import (
    DETECTOR_LABELS,
    ReviewRequest,
    ReviewSession,
    annotations_for,
    cached_windows,
    session_from_recording,
)


@pytest.fixture(scope="module")
def review(recording) -> ReviewSession:
    """One analysed window of the synthetic recording, reused by every test."""
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


# -- the request -----------------------------------------------------------

def test_request_refuses_an_unknown_detector():
    with pytest.raises(ValueError, match="unknown detector"):
        ReviewRequest(detectors=("wavelet",))


def test_request_refuses_an_empty_window():
    with pytest.raises(ValueError, match="before it starts"):
        ReviewRequest(t_start=60.0, t_stop=60.0)


def test_request_names_every_detector_it_offers():
    """The launcher shows `DETECTOR_LABELS`; a missing one would show a key."""
    from onset_hfo.detectors import HFO_DETECTORS

    assert set(DETECTOR_LABELS) == set(HFO_DETECTORS)


# -- the session's own numbers ---------------------------------------------

def test_every_channel_appears_in_the_findings(review, prepared):
    """Zero is a result. A channel with no events must still have a row."""
    assert set(review.findings["channel"]) == set(prepared.ch_names)


def test_rates_agree_with_the_event_list(review):
    """The table and the list are two views of one set of events, not two counts."""
    listed = trends.event_table(review)
    primary = listed[listed["detector"] == review.request.primary]
    counted = primary.groupby("channel").size()
    for _, row in review.findings.iterrows():
        assert int(row["n_events"]) == int(counted.get(row["channel"], 0))


def test_rate_is_the_count_over_the_window(review):
    minutes = review.request.duration / 60.0
    expected = review.findings["n_events"] / minutes
    assert np.allclose(review.findings["rate_per_min"], expected)


def test_the_interval_contains_the_rate(review):
    low, rate, high = (review.findings[c] for c in
                       ("rate_ci_low", "rate_per_min", "rate_ci_high"))
    assert ((low <= rate + 1e-9) & (rate <= high + 1e-9)).all()


def test_the_caveat_says_which_of_the_two_things_it_is(review):
    """The status bar's sentence has to follow `leader_separation`, not drift."""
    caveat = review.caveat()
    if review.leader.get("distinguishable"):
        assert "stands out" in caveat
    else:
        assert "NO CHANNEL STANDS OUT" in caveat


def test_the_leader_is_in_the_candidate_set(review):
    if review.candidates:
        assert review.leader["leader"] in review.candidates


def test_rejected_events_are_kept_but_not_counted(review):
    """Rejections are the filter's working; losing them loses the explanation."""
    assert len(review.events) >= len(review.accepted)
    rejected = [e for e in review.events if not e.accepted]
    if rejected:
        shown = trends.event_table(review)
        assert len(shown) == len(review.accepted)
        assert len(trends.event_table(review, include_rejected=True)) == len(review.events)


# -- annotations -----------------------------------------------------------

def test_annotations_are_in_the_trace_time_base(review):
    """Events carry archive time; the Raw starts at zero. Mix them and marks slide."""
    annotations = annotations_for(review.accepted, review.t_offset)
    assert len(annotations) == len(review.accepted)
    assert (np.asarray(annotations.onset) >= 0).all()
    # Compared as sorted sequences: MNE orders annotations by onset, while the
    # event list comes out grouped by detector, so pairing them positionally
    # would compare two different events and pass or fail by accident.
    expected = sorted(e.start - review.t_offset for e in review.accepted)
    assert np.allclose(sorted(annotations.onset), expected, atol=1e-9)


def test_annotations_are_named_for_the_band_not_the_detector(review):
    """A clinician asks "is that a fast ripple", never "did line length fire"."""
    descriptions = set(annotations_for(review.accepted, review.t_offset).description)
    assert descriptions <= {"ripple", "fast ripple", "spike"}


def test_the_raw_is_in_volts(review, prepared):
    """`prepare` returns microvolts and MNE scales from SI; the factor is 1e6."""
    assert review.raw.get_data().shape == prepared.data.shape
    assert np.allclose(review.raw.get_data() * 1e6, prepared.data, atol=1e-6)


# -- the trend -------------------------------------------------------------

def test_the_trend_holds_every_event_of_its_detector(review):
    matrix = trends.rate_matrix(review, bin_s=5.0)
    assert int(matrix.to_numpy().sum()) == len(review.events_of(review.request.primary))


def test_the_trend_covers_the_whole_window(review):
    """A trend that stops short of the trace hides whatever is in the tail."""
    for bin_s in (1.0, 5.0, 7.0):          # 7 does not divide 30 or 60
        matrix = trends.rate_matrix(review, bin_s=bin_s)
        assert matrix.shape[1] * bin_s >= review.request.duration
        assert int(matrix.to_numpy().sum()) == len(
            review.events_of(review.request.primary))


def test_trend_rows_follow_the_findings_order(review):
    """The two panels are read across, so row n has to be the same channel."""
    matrix = trends.rate_matrix(review)
    assert list(matrix.index) == list(review.findings["channel"])


def test_the_curve_is_a_rate_not_a_count(review):
    curve = trends.rate_curve(review, bin_s=5.0)
    widths = curve["t_stop"] - curve["t_start"]
    assert np.allclose(curve["rate_per_min"], curve["n_events"] / (widths / 60.0))


# -- agreement -------------------------------------------------------------

def test_agreement_is_absent_rather_than_zero_without_expert_marks(review):
    """A recording nobody annotated has no agreement; 0% would be a lie."""
    if not review.has_expert:
        summary = trends.agreement_summary(review)
        assert summary["available"] is False
        assert trends.agreement(review).empty


def test_agreement_restricts_itself_to_reviewed_channels():
    """A detection on an unreviewed channel is unjudged, not a false positive."""
    import inspect

    source = inspect.getsource(trends.agreement)
    assert "reviewed_channels" in source


# -- the exported review ---------------------------------------------------

def test_the_report_leads_with_the_disclaimer(review):
    text = report.review_markdown(review)
    assert text.index(report.DISCLAIMER) < text.index("## What was reviewed")


def test_the_report_carries_the_caveat_above_the_table(review):
    text = report.review_markdown(review)
    assert review.caveat() in text
    assert text.index(review.caveat()) < text.index("## Per-channel findings")


def test_the_report_states_the_provenance(review):
    text = report.review_markdown(review)
    for step in review.steps:
        assert step in text
    assert review.request.subject in text
    assert f"{review.sfreq:g} Hz" in text


def test_the_report_needs_no_optional_dependency(review, monkeypatch):
    """An export is the last step of a review; it must not need `tabulate`."""
    import builtins

    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("tabulate is not installed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert "| rank |" in report.review_markdown(review)


def test_the_report_formats_what_pandas_actually_hands_it():
    """NumPy scalars, not Python ones, are what come out of a DataFrame row.

    `np.bool_` is not a subclass of `bool`, so a `reviewed` column printed
    without checking for it reads "True" and "False" in a clinical document
    that says "yes" and "no" everywhere else.
    """
    from onset_review.report import _cell

    assert _cell(np.bool_(True)) == "yes"
    assert _cell(np.bool_(False)) == "no"
    assert _cell(np.int64(3)) == "3"
    assert _cell(np.float64(2.5)) == "2.50"
    for missing in (None, np.nan, pd.NA, pd.NaT):
        assert _cell(missing) == "—"
    assert _cell("AR1-AR2") == "AR1-AR2"


def test_the_exported_tables_hold_no_raw_python_repr(review):
    """Nothing in the document should read `True`, `nan` or `None`."""
    text = report.review_markdown(review)
    for row in [line for line in text.splitlines() if line.startswith("| ")]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert not ({"True", "False", "nan", "None", "NaN", "<NA>"} & set(cells))


def test_write_review_round_trips(review, tmp_path):
    written = report.write_review(review, tmp_path / "r.md", reviewer="Dr Test")
    assert "Dr Test" in written.read_text()
    page = report.write_review(review, tmp_path / "r.html")
    assert page.read_text().lstrip().startswith("<!doctype html>")


# -- the cache catalogue ---------------------------------------------------

def test_cached_windows_is_empty_rather_than_failing_on_an_empty_cache(tmp_path):
    frame = cached_windows(tmp_path)
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty


def test_cached_windows_reads_the_slice_metadata(tmp_path):
    """The launcher offers only what is here, so a bad read means a bad list."""
    import json

    slice_dir = tmp_path / "ds003498" / "sub-09_run-01_0-60s"
    slice_dir.mkdir(parents=True)
    (slice_dir / "slice.json").write_text(json.dumps({
        "dataset": "ds003498", "subject": "sub-09", "task": None, "run": "01",
        "t_start": 0, "t_stop": 60, "sfreq": 2000.0, "n_channels": 50}))
    low = tmp_path / "ds003029" / "sub-pt01_task-ictal_run-01_0-10s"
    low.mkdir(parents=True)
    (low / "slice.json").write_text(json.dumps({
        "dataset": "ds003029", "subject": "sub-pt01", "task": "ictal",
        "run": "01", "t_start": 0, "t_stop": 10, "sfreq": 1000.0,
        "n_channels": 98}))

    frame = cached_windows(tmp_path).set_index("subject")
    assert frame.loc["sub-09", "bands"] == "ripple, fast_ripple"
    # 1000 Hz cannot carry a 500 Hz band, and the dialog must not offer it.
    assert frame.loc["sub-pt01", "bands"] == "ripple"


def test_cached_windows_survives_a_damaged_entry(tmp_path):
    slice_dir = tmp_path / "ds003498" / "sub-01_run-01_0-60s"
    slice_dir.mkdir(parents=True)
    (slice_dir / "slice.json").write_text("{ not json")
    assert cached_windows(tmp_path).empty


# -- what the preprocessing panel promises, before it is clicked -----------
#
# `describe` is the sentence and the red text a reviewer reads under the
# controls. It is pure, so it is tested here; its job is to mirror the refusals
# in `onset_hfo.preprocess.prepare` exactly, because a reviewer should learn
# that a 150 Hz low-pass is wrong from the panel, not from a dialog thirty
# seconds into a re-analysis.

def _describe(**changes):
    from dataclasses import replace as _replace

    from onset_hfo.config import PreprocessConfig
    from onset_hfo.preprocess import describe

    return describe(_replace(PreprocessConfig(), **changes), (80.0, 250.0), 2000.0)


def test_the_defaults_describe_themselves_without_warning():
    summary, warnings = _describe()
    assert "high-pass at 1 Hz" in summary
    assert "bipolar" in summary
    assert warnings == []


@pytest.mark.parametrize("changes,phrase", [
    ({"lowpass": 150.0}, "cuts into the band"),
    ({"lowpass": 0.5}, "passes nothing"),
    ({"resample": 1000.0, "lowpass": 600.0}, "Nyquist"),
    ({"notch": False}, "mains harmonics sit inside"),
    ({"notch_width": 6.0}, "is wide"),
    ({"highpass": 120.0}, "inside the band"),
    ({"bipolar": False}, "same noise on every channel"),
])
def test_a_questionable_setting_is_called_out_before_it_is_applied(changes, phrase):
    _, warnings = _describe(**changes)
    assert any(phrase in w for w in warnings), warnings


def test_the_panel_warns_about_exactly_what_the_pipeline_refuses(recording):
    """The two must not drift: a warning the pipeline does not enforce teaches
    a reviewer to ignore warnings, and a refusal the panel did not predict
    arrives as a failure after a minute of work."""
    from dataclasses import replace as _replace

    from onset_hfo.config import PreprocessConfig
    from onset_hfo.preprocess import prepare

    for changes in ({"lowpass": 0.5}, {"resample": 500.0, "lowpass": 400.0},
                    {"notch_width": 0.0}, {"resample": 0.0},
                    {"resample": -250.0}):
        cfg = _replace(PreprocessConfig(), **changes)
        with pytest.raises(ValueError):
            prepare(recording, cfg, verbose=False)
        from onset_hfo.preprocess import describe

        assert describe(cfg, (80.0, 250.0), 2000.0)[1], changes


def test_the_request_describes_its_own_preprocessing():
    from onset_hfo.config import PreprocessConfig

    plain = ReviewRequest(t_start=0, t_stop=60)
    assert "bipolar montage" in plain.preprocess_label()
    assert "high-pass 1 Hz" in plain.preprocess_label()

    custom = ReviewRequest(t_start=0, t_stop=60, preprocess=PreprocessConfig(
        bipolar=False, average_reference=True, resample=1000.0,
        notch_harmonics=False, exclude=("AR1",)))
    label = custom.preprocess_label()
    assert "common average reference" in label
    assert "resample to 1000 Hz" in label
    assert "harmonics" not in label
    assert "1 channel(s) excluded" in label


def test_preprocessing_choices_travel_into_the_pipeline_config():
    """The request is what a session is reproduced from, so it has to carry
    the filtering as well as the band and the detector."""
    from onset_hfo.config import PreprocessConfig

    chosen = PreprocessConfig(highpass=2.0, notch_width=4.0, bipolar=False,
                              average_reference=True)
    request = ReviewRequest(t_start=0, t_stop=60, preprocess=chosen)
    assert request.pipeline_config().preprocess == chosen
    # And the default is still the project's measured one, untouched.
    assert ReviewRequest(t_start=0, t_stop=60).pipeline_config().preprocess \
        == PreprocessConfig()


def test_a_band_destroying_resample_is_refused_with_the_reason(recording):
    from onset_hfo.config import PreprocessConfig

    with pytest.raises(ValueError, match="resampled it to 1000 Hz"):
        session_from_recording(recording, ReviewRequest(
            t_start=0.0, t_stop=float(recording.duration), band="fast_ripple",
            preprocess=PreprocessConfig(resample=1000.0)))


def test_a_band_destroying_lowpass_is_refused_with_the_reason(recording):
    from onset_hfo.config import PreprocessConfig

    with pytest.raises(ValueError, match="removes most of the ripple band"):
        session_from_recording(recording, ReviewRequest(
            t_start=0.0, t_stop=float(recording.duration),
            preprocess=PreprocessConfig(lowpass=150.0)))


def test_changed_preprocessing_reaches_the_report(recording):
    from onset_hfo.config import PreprocessConfig

    session = session_from_recording(recording, ReviewRequest(
        t_start=0.0, t_stop=float(recording.duration),
        preprocess=PreprocessConfig(notch_width=4.0, bipolar=False,
                                    average_reference=True)))
    text = report.review_markdown(session)
    assert "4 Hz wide" in text
    assert "common average" in text.lower()
    assert session.montage == "average"


# -- the boundary this file exists to defend -------------------------------

def test_the_modules_this_file_covers_import_no_qt():
    """Every module this file imports must load with no Qt installed at all.

    Not a hypothetical. It has now happened twice: the preprocessing warnings
    went into `onset_review.preprocessing`, and `patient_record` into
    `onset_review.patient`, both of which import Qt at module level, and both
    were tested from here -- so the main CI job, which installs no `review`
    extra, went red. Each time the fix was to move the logic to a Qt-free
    module, because the test was right.

    The first version of this guard hardcoded the module list and so missed the
    second case entirely. It now reads the imports out of this file, which
    means it covers whatever the file actually uses rather than whatever was
    true when it was written.
    """
    import ast
    import pathlib as _pathlib
    import subprocess
    import sys

    source = _pathlib.Path(__file__)
    tree = ast.parse(source.read_text())
    wanted = {"onset_review", "onset_hfo"}
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in wanted:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in wanted:
                    modules.add(alias.name)
    assert modules, "found no imports to check; has this file changed shape?"

    # A subprocess with the Qt packages blocked, rather than poking
    # `sys.modules` in-process: by the time this runs in a full suite pytest
    # has already imported Qt, and a module reaching for it would be handed the
    # live one and pass.
    script = (
        "import sys\n"
        "class Blocked:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'qtpy', 'PySide6', 'PyQt5', 'PyQt6'}:\n"
        "            raise ImportError(f'{name} is blocked for this check')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocked())\n"
        "import importlib\n"
        f"for name in {sorted(modules)!r}:\n"
        "    importlib.import_module(name)\n"
        "print('clean')\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, (
        f"a module this file imports needs Qt:\n{done.stderr[-1500:]}")
    assert "clean" in done.stdout


# -- the patient record ----------------------------------------------------
#
# The panel that shows this is Qt; the record it shows is not, and the record
# is the part that matters. These tests are mostly about what it must *not*
# contain.

def _record(subject, **kwargs):
    from onset_review.record import patient_record

    return patient_record(subject, **kwargs)


def test_the_patient_record_reads_the_committed_tables():
    """`data/outcome/` exists so a number can be checked from a fresh clone."""
    record = _record("sub-01")
    assert record["available"]
    assert record["epilepsy"] == "TLE"
    assert record["n_channels"] == 43
    assert record["n_reviewed"] == 23
    assert record["rz_coverage"] == pytest.approx(1.0)


def test_a_patient_with_partial_resection_coverage_carries_the_missing_contacts():
    """The single most important caveat for five of these twenty patients.

    In sub-02 only a quarter of the contacts the surgeon removed appear in the
    recording, so anything said about "inside the resection" describes that
    quarter. A reviewer who is not told reads it as describing the resection.
    """
    record = _record("sub-02")
    assert record["rz_coverage"] == pytest.approx(0.25)
    assert record["missing"]
    assert "ER1" in record["missing"]


def test_an_unknown_subject_is_unavailable_rather_than_invented():
    record = _record("sub-99")
    assert record["available"] is False
    assert set(record) == {"subject", "available"}


def test_a_missing_data_directory_does_not_raise(tmp_path):
    assert _record("sub-01", data_dir=tmp_path)["available"] is False


def test_the_record_holds_no_demographics():
    """Age, sex and handedness are in the archive and deliberately not here.

    Three demographic fields published beside pathology, surgical extent and
    outcome narrow a cohort of twenty considerably, and no analysis in this
    project uses any of them. `data/outcome/README.md` records the decision;
    this stops it being undone by accident.
    """
    record = _record("sub-01")
    for field in ("age", "sex", "handedness", "dob", "name"):
        assert field not in record


def test_the_chart_fields_are_named_but_never_filled():
    """The one thing this record must not do.

    A clinical-looking panel populated with plausible semiology, imaging or
    medication is a fabricated medical record, and in software a clinician is
    asked to trust it is indistinguishable from a real one. The fields exist so
    a site can see where its own data lands; they carry no content.
    """
    from onset_review.record import CHART_FIELDS

    names = [name for name, _ in CHART_FIELDS]
    assert "MRI" in names and "Medication" in names and "Seizure semiology" in names
    for name, why in CHART_FIELDS:
        assert why and why[0].islower()      # a description, not a value
        assert not any(ch.isdigit() for ch in why), f"{name} looks like data"


def test_every_ilae_class_the_cohort_uses_has_a_description():
    """A bare "ILAE 5" means nothing to anyone outside epilepsy surgery."""
    import pathlib as _pathlib

    from onset_review.record import ILAE_CLASSES

    participants = pd.read_csv(
        _pathlib.Path(__file__).resolve().parent.parent / "data" / "outcome"
        / "participants.csv")
    for value in sorted(participants["ilae"].unique()):
        assert int(value) in ILAE_CLASSES, value


# -- opening a file from the command line ----------------------------------
#
# `onset_review.app` imports Qt lazily, so the routing, the flag parsing and
# the one refusal it makes are all checkable here, on a machine with no
# display. Which is the point: the refusal below is what stands between a
# scripted batch and a folder full of confident reviews of scalp EEG.

def test_a_file_on_the_command_line_builds_an_imported_request(tmp_path):
    from onset_review.app import _request_from, build_parser

    recording = tmp_path / "study-001.edf"
    args = build_parser().parse_args(
        ["--open", str(recording), "--all-channels-as", "seeg",
         "--window", "10", "40", "--line-freq", "60", "--band", "fast_ripple"])
    assert args.open_path == recording

    # `--all-channels-as` reads the file's channel list, which does not exist
    # here; the named exceptions alone do not.
    args.all_channels_as = None
    args.channel_types = ["AR1=seeg", "EKG=ecg"]
    request = _request_from(args)

    assert request.imported is True
    assert request.path == recording
    assert request.subject == "study-001"            # from the filename
    assert (request.t_start, request.t_stop) == (10.0, 40.0)
    assert request.line_freq == 60.0
    assert dict(request.channel_types) == {"AR1": "seeg", "EKG": "ecg"}
    assert request.dataset == ""


def test_a_malformed_channel_type_says_what_the_flag_wants(tmp_path):
    from onset_review.app import _channel_types, build_parser

    args = build_parser().parse_args(
        ["--open", str(tmp_path / "a.edf"), "--channel-type", "AR1"])
    with pytest.raises(SystemExit, match="NAME=TYPE"):
        _channel_types(args)


def test_a_headless_import_with_nothing_said_about_the_channels_is_refused(
        tmp_path, capsys):
    """The one refusal in the entry point, and the reason it exists.

    A clinical export declares every channel `eeg`; this pipeline analyses
    anything typed eeg, ecog or seeg. A scripted `--export` that said nothing
    about the channels would therefore write a complete review -- rates,
    Poisson intervals, a candidate channel set -- for whatever happened to be
    in the file, with no step anywhere having been wrong. Stating the types is
    one flag. Being wrong silently has no flag at all.
    """
    from onset_review.app import EXIT_FAILED, main

    recording = tmp_path / "study-001.edf"
    recording.write_bytes(b"")
    assert main(["--open", str(recording), "--export",
                 str(tmp_path / "out.md")]) == EXIT_FAILED
    assert "--all-channels-as" in capsys.readouterr().err

    # And the same refusal for a window, where `--no-confirm` is what skips
    # the dialog that would otherwise have asked.
    assert main(["--open", str(recording), "--no-confirm"]) == EXIT_FAILED


def test_export_still_needs_a_subject_or_a_file(tmp_path, capsys):
    from onset_review.app import EXIT_FAILED, main

    assert main(["--export", str(tmp_path / "out.md")]) == EXIT_FAILED
    assert "--subject or --open" in capsys.readouterr().err


def test_the_report_names_the_imported_file_where_an_accession_would_go(
        recording, tmp_path):
    """An empty Dataset cell reads as a missing value, not an absent one."""
    path = tmp_path / "study-001_raw.fif"
    request = ReviewRequest(dataset="", subject="study-001", path=path,
                            t_start=0.0, t_stop=float(recording.duration))
    review = session_from_recording(recording, request)
    text = report.review_markdown(review)

    assert f"local file `{path.name}`" in text
    assert "not a public archive recording" in text
    assert path.name in request.label()


def test_a_bare_filename_means_the_same_as_open(tmp_path, capsys):
    """`onset-review study-001.edf`, and what a file manager passes in.

    The same refusal applies: a file arriving by double-click has had nothing
    said about its channels either.
    """
    from onset_review.app import EXIT_FAILED, main

    recording = tmp_path / "study-001.edf"
    recording.write_bytes(b"")
    assert main([str(recording), "--no-confirm"]) == EXIT_FAILED
    assert "--all-channels-as" in capsys.readouterr().err
