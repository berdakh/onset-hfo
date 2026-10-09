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

def _judge_everything(session, verdict="agree", reader="Dr Smith"):
    from onset_review.adjudication import Adjudication, event_key

    session.read = Adjudication(reader=reader)
    for event in session.events:
        if event.accepted:
            session.read.judge_event(
                event_key(event.channel, event.start, event.detector), verdict)
    return session.read


def test_an_unread_window_says_so_rather_than_saying_nothing(review):
    """A report with no reader section would read as a complete review."""
    text = report.review_markdown(review)
    assert "## The reader's own read" in text
    assert "Nobody has recorded a verdict" in text


def test_a_partial_read_is_labelled_as_one(review):
    from onset_review.adjudication import Adjudication, event_key

    previous = review.read
    try:
        review.read = Adjudication(reader="Dr Smith")
        first = next(e for e in review.events if e.accepted)
        review.read.judge_event(
            event_key(first.channel, first.start, first.detector), "agree")
        text = report.review_markdown(review)
        assert "**This is a partial read.**" in text
        assert "must not be read as one" in text
    finally:
        review.read = previous


def test_the_confirmed_rate_reproduces_the_detectors_rate_when_all_agreed(review):
    """The arithmetic test for the whole feature.

    If a reader agrees with every event the detector found on a contact, their
    rate for that contact must come out identical to the detector's -- same
    events, same denominator. Any mismatch means the confirmed rate is
    measuring something other than what the column above it measures, which
    would be worse than not reporting it.
    """
    previous = review.read
    try:
        _judge_everything(review)
        confirmed = report._confirmed_rates(review)
        assert confirmed, "no channel came out complete"
        for _, row in review.findings.iterrows():
            channel = str(row["channel"])
            if channel in confirmed:
                assert float(confirmed[channel]) == pytest.approx(
                    float(row["rate_per_min"]), abs=0.01)
    finally:
        review.read = previous


def test_disagreeing_with_everything_zeroes_the_confirmed_rate(review):
    previous = review.read
    try:
        _judge_everything(review, verdict="disagree")
        confirmed = report._confirmed_rates(review)
        assert confirmed
        assert set(confirmed.values()) == {"0.00"}
    finally:
        review.read = previous


def test_a_half_judged_contact_gets_no_confirmed_rate(review):
    """It would be a confirmed count divided by the whole window."""
    from onset_review.adjudication import event_key

    previous = review.read
    try:
        read = _judge_everything(review)
        channel = next(iter(report._confirmed_rates(review)))
        victim = next(e for e in review.events
                      if e.accepted and e.channel == channel
                      and e.detector == review.request.primary)
        read.clear_event(event_key(victim.channel, victim.start, victim.detector))
        assert channel not in report._confirmed_rates(review)
    finally:
        review.read = previous


def test_the_report_names_the_reader_and_lists_what_they_rejected(review):
    from onset_review.adjudication import Adjudication, event_key

    previous = review.read
    try:
        review.read = Adjudication(reader="Dr Smith")
        first = next(e for e in review.events if e.accepted)
        review.read.judge_event(
            event_key(first.channel, first.start, first.detector),
            "disagree", note="ringing on a sharp transient")
        text = report.review_markdown(review)
        assert "| Reviewer | Dr Smith |" in text
        assert "### Events the reader rejected" in text
        assert "ringing on a sharp transient" in text
    finally:
        review.read = previous


def test_the_report_owns_up_to_orphaned_verdicts(review):
    """A verdict the analysis no longer matches is named, not quietly dropped."""
    from onset_review.adjudication import Adjudication, Judgement

    previous = review.read
    try:
        review.read = Adjudication(
            reader="Dr Smith",
            orphaned={"ZZ1-ZZ2|9.999|rms": Judgement(
                "agree", "Dr Smith", "2026-01-01T00:00:00Z")})
        text = report.review_markdown(review)
        assert "1 earlier verdict(s) no longer match" in text
        assert "not deleted" in text
    finally:
        review.read = previous


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


def test_every_cli_command_this_project_tells_people_to_run_exists():
    """The installer, the docs and the GUI all name `onset_hfo.cli` commands.

    `fetch` was named in five places and implemented in none, so
    `install-ubuntu.sh --with-sample` had never once worked: it printed
    argparse's "invalid choice" and then blamed the network, and the reviewer
    opened with an empty cache and advice to re-run the flag that had just
    failed. Nobody noticed because nothing executed those strings — a
    documented command is only as real as something that runs it.

    So this reads the strings back out of the files that print them and checks
    each against the parser. It is the same shape as the no-Qt guard above and
    exists for the same reason: the check has to come from the files rather
    than from a list someone remembers to update.
    """
    import pathlib as _pathlib
    import re

    from onset_hfo.cli import build_parser

    root = _pathlib.Path(__file__).resolve().parent.parent
    sources = [root / "packaging" / "install-ubuntu.sh",
               root / "docs" / "INSTALL.md",
               root / "onset_review" / "launcher.py",
               root / "onset_review" / "app.py",
               root / "onset_hfo" / "cli.py"]

    known = set(build_parser()._subparsers._group_actions[0].choices)
    assert "fetch" in known, "the command the installer runs has to exist"

    named: dict[str, list[str]] = {}
    for path in sources:
        if not path.exists():
            continue
        for command in re.findall(r"onset_hfo\.cli\s+([a-z_]+)", path.read_text()):
            named.setdefault(command, []).append(path.name)

    assert named, "found no commands to check; have these files changed shape?"
    unknown = {c: sorted(set(w)) for c, w in named.items() if c not in known}
    assert not unknown, (
        "these are told to users but are not commands: "
        + "; ".join(f"{c} (in {', '.join(w)})" for c, w in unknown.items()))


# -- the flat contact map (Qt-free) ------------------------------------------


def test_the_map_frame_carries_a_value_for_every_measure(review):
    from onset_review import contactmap

    primary = contactmap.map_frame(review)
    assert not primary.empty and "value" in primary.columns
    assert (primary["value"] == primary["rate_per_min"]).all()
    rank = contactmap.map_frame(review, measure="rank")
    assert (rank["value"] == rank["rank"]).all()
    # A detector counted from the window's own events, per minute.
    minutes = (float(review.span[1]) - float(review.span[0])) / 60.0
    rms = contactmap.map_frame(review, measure="rms")
    counted = {}
    for event in review.events:
        if event.accepted and event.detector == "rms":
            counted[event.channel] = counted.get(event.channel, 0) + 1
    for row in rms.itertuples():
        assert row.value == pytest.approx(counted.get(row.channel, 0) / minutes)
    assert (contactmap.map_frame(review, measure="expert")["value"] >= 0).all()


def test_every_view_projects_and_mirrors_as_seen_from_that_side(review):
    from onset_review import contactmap

    frame = contactmap.map_frame(review)
    for view in contactmap.VIEWS:
        h, v = contactmap.project(frame, view)
        assert h.shape == v.shape == (len(frame),)
    right_h, _ = contactmap.project(frame, "Right")
    left_h, _ = contactmap.project(frame, "Left")
    assert np.allclose(left_h, -right_h), "left is right mirrored"
    top_h, top_v = contactmap.project(frame, "Top")
    assert np.allclose(top_h, frame["x"]) and np.allclose(top_v, frame["y"])


def test_the_head_outline_is_a_closed_convex_curve_round_the_contacts(review):
    from onset_review import contactmap

    frame = contactmap.map_frame(review)
    for view in contactmap.VIEWS:
        h, v = contactmap.head_outline(view)
        assert h[0] == h[-1] and v[0] == v[-1], "closed"
        assert len(h) > 8
        # Convex: every turn goes the same way.
        dx, dy = np.diff(h), np.diff(v)
        cross = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
        assert (cross >= -1e-12).all() or (cross <= 1e-12).all()
        ph, pv = contactmap.project(frame, view)
        assert ph.min() > h.min() - 0.05 and ph.max() < h.max() + 0.05


def test_the_assistants_contact_table_never_carries_the_resection(review):
    """The reviewer sees the resection as rings; the model is never told what
    the surgeon removed, so nothing it says can read as an opinion on it."""
    from onset_review import contactmap

    table = contactmap.contacts_table(review)
    assert list(table.columns) == contactmap.CONTACT_COLUMNS
    assert "zone" not in table.columns and "resect" not in " ".join(table.columns)
    assert set(table["hemisphere"]) <= {"left", "right", "unknown"}
    assert len(table) == len(review.findings)


def test_where_summary_counts_the_leaders_per_shaft_and_says_nothing_it_does_not_know():
    import pandas as pd

    from onset_review import contactmap

    rows = [
        ("AR1-AR2", "AR", "right", "amygdala", 1), ("AR2-AR3", "AR", "right", "amygdala", 2),
        ("PHR1-PHR2", "PHR", "right", "parahippocampal", 3), ("AR3-AR4", "AR", "right", "amygdala", 4),
        ("HL1-HL2", "HL", "left", "hippocampus", 5), ("HL2-HL3", "HL", "left", "hippocampus", 6),
    ]
    frame = pd.DataFrame([{"channel": c, "shaft": s, "hemisphere": h, "region": r, "rank": k,
                           "rate_per_min": 10.0 - k, "n_events": 10 - k, "source": "archive",
                           "x": 0.0, "y": 0.0, "z": 0.0} for c, s, h, r, k in rows])
    out = contactmap.where_summary(frame, top=5)
    assert out["positions"] == "measured" and out["n_shafts"] == 3
    assert out["shafts"][0] == {"shaft": "AR", "side": "right", "region": "amygdala",
                                "n_of_top": 3, "channels": ["AR1-AR2", "AR2-AR3", "AR3-AR4"]}
    assert out["summary"].startswith("3 of the top 5 channels on shaft AR (right, amygdala)")
    # Unknown side and an unmapped region are left unsaid, not said as "unknown".
    frame["hemisphere"], frame["region"], frame["source"] = "unknown", "unmapped", "inferred"
    out = contactmap.where_summary(frame, top=5)
    assert "unknown" not in out["summary"] and "unmapped" not in out["summary"]
    assert out["positions"].startswith("schematic")
    assert out["summary"].startswith("3 of the top 5 channels on shaft AR;")



# -- the spectrum (Qt-free) ----------------------------------------------------


def test_the_spectrum_is_of_the_preprocessed_signal_in_microvolts(review):
    from onset_review import spectrum

    spec = spectrum.compute(review)
    assert spec.available and spec.power.shape == (len(spec.channels), spec.freqs.size)
    assert spec.channels == list(review.raw.ch_names)
    assert spec.freqs[0] >= 0.5 and spec.freqs[-1] <= spec.sfreq / 2.0
    assert (spec.power >= 0).all()
    # Microvolts squared per hertz: the integral is the variance in µV².
    i = 0
    variance = float(np.var(review.raw.get_data()[i] * 1e6))
    integral = float(np.trapezoid(spec.power[i], spec.freqs)) if hasattr(np, "trapezoid") \
        else float(np.trapz(spec.power[i], spec.freqs))
    assert 0.2 * variance < integral < 5.0 * variance


def test_the_slope_tells_a_steep_background_from_a_flat_carpet():
    from onset_review import spectrum

    freqs = np.linspace(0.5, 500.0, 1000)
    steep = freqs ** -2.0
    flat = np.full_like(freqs, 1e-3)
    spec = spectrum.Spectrum(freqs, np.vstack([steep, flat]), ["steep", "flat"],
                             sfreq=1000.0, line_freq=50.0, band=(80.0, 250.0))
    slope, _ = spectrum.slope_fit(spec, "steep")
    assert slope == pytest.approx(-2.0, abs=0.05)
    assert spectrum.slope_fit(spec, "flat")[0] == pytest.approx(0.0, abs=0.05)
    assert spectrum.band_share(spec, "flat") > spectrum.band_share(spec, "steep")
    table = spectrum.summarise(spec)
    assert list(table["reading"]) == ["steep", "flat"]
    assert spectrum.mains_lines(spec) == [50.0 * k for k in range(1, 10)]
    # A comb of mains peaks reads as mains.
    comb = steep.copy()
    for line in spectrum.mains_lines(spec):
        comb[np.abs(freqs - line) <= 1.0] += 10.0
    spec = spectrum.Spectrum(freqs, comb[None, :], ["comb"], 1000.0, 50.0, (80.0, 250.0))
    assert spectrum.summarise(spec)["reading"].iloc[0] == "mains"
    assert "mains lines" in spectrum.describe(spec, "comb")


# --------------------------------------------------------------------------
# The template surface: a real cortex, only where real coordinates can sit on it
# --------------------------------------------------------------------------

def _write_stand_in_template(subjects_dir):
    """A tiny closed surface written the way FreeSurfer writes a pial one, in
    both hemispheres, so the reader and the decimation run on the real path
    without the real 300 MB download."""
    from onset_review.anatomy import TEMPLATE_SUBJECT, TEMPLATE_SURFACE

    surf = subjects_dir / TEMPLATE_SUBJECT / "surf"
    surf.mkdir(parents=True)
    # An octahedron per hemisphere, in millimetres, 60 mm across, offset left
    # and right the way the hemispheres are.
    base = np.array([[30, 0, 0], [-30, 0, 0], [0, 30, 0], [0, -30, 0],
                     [0, 0, 30], [0, 0, -30]], dtype=float)
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for hemi, shift in (("lh", -40.0), ("rh", 40.0)):
        vertices = base + np.array([shift, 0.0, 0.0])
        _write_freesurfer_triangles(surf / f"{hemi}.{TEMPLATE_SURFACE}", vertices, faces)
    return subjects_dir


def _write_freesurfer_triangles(path, vertices, faces):
    """FreeSurfer's triangle format, as nibabel writes it: the format
    ``lh.pial`` is in. Written here so the test does not need nibabel."""
    with open(path, "wb") as handle:
        handle.write(b"\xff\xff\xfe")
        handle.write(b"created by a test\n\n")
        np.array([len(vertices), len(faces)], dtype=">i4").tofile(handle)
        np.asarray(vertices, dtype=">f4").reshape(-1).tofile(handle)
        np.asarray(faces, dtype=">i4").reshape(-1).tofile(handle)


def test_a_freesurfer_surface_is_read_without_nibabel(tmp_path):
    from onset_review.anatomy import read_surface

    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    faces = np.array([[0, 1, 2]])
    _write_freesurfer_triangles(tmp_path / "lh.pial", vertices, faces)
    got_v, got_f = read_surface(tmp_path / "lh.pial")
    assert np.allclose(got_v, vertices) and np.array_equal(got_f, faces)
    (tmp_path / "lh.bad").write_bytes(b"nope")
    with pytest.raises((ValueError, OSError)):
        read_surface(tmp_path / "lh.bad")


def test_vertex_clustering_merges_a_cell_and_drops_collapsed_faces():
    from onset_review.anatomy import cluster_decimate

    # Four vertices, two of them within one cell: a triangle on each pair.
    vertices = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]])
    merged, kept = cluster_decimate(vertices, faces, cell=4.0)
    assert merged.shape[0] == 3                       # 0 and 1 became one
    assert kept.shape[0] == 1                         # the two big triangles became one
    assert (kept[:, 0] != kept[:, 1]).all() and (kept[:, 1] != kept[:, 2]).all()
    # The merged vertex sits at the pair's mean.
    assert np.allclose(sorted(merged[:, 0]), [0.0, 0.25, 10.0])
    # No decimation asked for: everything comes back as it went in.
    same_v, same_f = cluster_decimate(vertices, faces, cell=0)
    assert same_v.shape == vertices.shape and same_f.shape == faces.shape


def test_the_template_is_found_only_where_it_has_been_fetched(tmp_path):
    from onset_review.anatomy import template_dir

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert template_dir(empty) is None or template_dir(empty) != empty
    root = _write_stand_in_template(tmp_path / "subjects")
    assert template_dir(root) == root


def test_the_template_surface_is_both_hemispheres_in_metres_and_cached(tmp_path):
    from onset_review.anatomy import TEMPLATE_SUBJECT, template_surface

    root = _write_stand_in_template(tmp_path / "subjects")
    cache = tmp_path / "cache"
    vertices, faces = template_surface(root, cell_mm=1.0, cache_dir=cache)
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    assert vertices.shape[0] == 12                    # six a hemisphere, none merged at 1 mm
    assert faces.shape[0] == 16
    assert faces.max() < vertices.shape[0]
    # Millimetres became metres: the hemispheres sit 40 mm either side of the midline.
    assert abs(vertices[:, 0].min() + 0.070) < 1e-6
    assert abs(vertices[:, 0].max() - 0.070) < 1e-6
    assert set(np.sign(vertices[:, 0]).astype(int)) == {-1, 1}
    cached = list(cache.glob(f"template-{TEMPLATE_SUBJECT}-*.npz"))
    assert len(cached) == 1
    # The second read comes from the cache, so it works with the surfaces gone.
    import shutil

    shutil.rmtree(root / TEMPLATE_SUBJECT / "surf")
    again_v, again_f = template_surface(root, cell_mm=1.0, cache_dir=cache)
    assert np.allclose(again_v, vertices) and np.array_equal(again_f, faces)


def test_a_missing_template_says_how_to_fetch_it(tmp_path, monkeypatch):
    from onset_review import anatomy

    # Whatever this machine has configured in MNE, the template is not found,
    # and nilearn's bundled one is not installed.
    monkeypatch.setattr(anatomy, "template_dir", lambda _subjects_dir=None: None)
    monkeypatch.setattr(anatomy, "nilearn_surface", lambda: None)
    with pytest.raises(FileNotFoundError, match="fetch_fsaverage"):
        anatomy.template_surface(tmp_path, cache_dir=tmp_path)


def test_without_a_fetch_nilearns_bundled_template_is_drawn(tmp_path, monkeypatch):
    pytest.importorskip("nilearn")
    from onset_review import anatomy

    monkeypatch.setattr(anatomy, "template_dir", lambda _subjects_dir=None: None)
    vertices, faces = anatomy.template_surface(tmp_path, cache_dir=tmp_path)
    assert vertices.shape == (2 * 10242, 3) and faces.max() == len(vertices) - 1
    assert vertices[:, 0].min() < -0.06 and vertices[:, 0].max() > 0.06, "both hemispheres"
    assert np.abs(vertices).max() < 0.12, "metres"


def test_the_coordinate_space_is_read_from_the_sidecar_and_never_guessed(tmp_path):
    import json

    from onset_review.coordinates import coordinate_space

    path = tmp_path / "sub-01_electrodes.tsv"
    path.write_text("name\tx\ty\tz\nA1\t0\t0\t0\n", encoding="utf-8")
    assert coordinate_space(path) == "unknown"
    sidecar = tmp_path / "sub-01_coordsystem.json"
    sidecar.write_text(json.dumps({"iEEGCoordinateSystem": "MNI152NLin2009aSym"}),
                       encoding="utf-8")
    assert coordinate_space(path) == "template"
    sidecar.write_text(json.dumps({"iEEGCoordinateSystem": "Other",
                                   "iEEGCoordinateSystemDescription": "scanner RAS"}),
                       encoding="utf-8")
    assert coordinate_space(path) == "patient"
    sidecar.write_text("not json", encoding="utf-8")
    assert coordinate_space(path) == "unknown"


# --------------------------------------------------------------------------
# The average event and the threshold re-test
# --------------------------------------------------------------------------

def test_the_average_event_aligns_at_the_peak_and_reads_the_mean(review):
    from onset_review import average

    channels = average.channels_with_events(review)
    assert channels, "the synthetic window has accepted events"
    leader, count = channels[0]
    assert leader == str(review.findings.sort_values("rank")["channel"].iloc[0])
    avg = average.compute(review, leader)
    assert avg.available and avg.n_available == count and 0 < avg.n <= count
    assert avg.times.shape == avg.mean_wideband.shape == avg.mean_band.shape
    assert abs(avg.times[0] + average.HALF_S) < 1e-6 and abs(avg.times[-1] - average.HALF_S) < 1e-6
    # Aligned at the band-passed peak: the mean band-passed trace peaks at zero.
    assert abs(float(avg.times[np.argmax(avg.mean_band)])) <= 0.001
    assert avg.sd_band.min() >= 0 and avg.mean_power_db.shape == (avg.freqs.size,
                                                                   avg.tfr_times.size)
    assert avg.reading in ("island", "column", "unclear") and avg.why
    assert np.isfinite(avg.mean_duration_ms) and np.isfinite(avg.mean_frequency_hz)
    text = average.describe(avg)
    assert leader in text and "averaged" in text and avg.reading in text
    # The synthetic bursts are oscillations by construction.
    assert avg.band_contrast_db is not None and avg.band_contrast_db > 0


def test_the_average_event_is_empty_without_events_or_signal(review):
    from onset_review import average

    nothing = average.compute(review, "NOT-A-CHANNEL")
    assert not nothing.available and nothing.n == 0 and average.describe(nothing) == ""

    class Bare:
        request = review.request
        events = review.events
        findings = review.findings
        raw = None

    assert not average.compute(Bare(), average.channels_with_events(review)[0][0]).available


def test_the_subsample_is_spread_through_the_window_not_the_first_n():
    from onset_review.average import _subsample

    events = list(range(100))
    chosen = _subsample(events, 10)
    assert len(chosen) == 10 and chosen[0] == 0 and chosen[-1] == 99
    assert _subsample(events[:5], 10) == events[:5]


def test_the_threshold_re_test_follows_the_leaders_and_says_who_survives(review):
    from onset_review import sensitivity

    sens = sensitivity.compute(review, factors=(1.0, 2.0), top=3)
    assert sens.available and sens.runtime_s >= 0
    assert sens.detector == review.request.primary
    assert sens.thresholds == [sens.base_threshold, round(sens.base_threshold * 2, 3)]
    assert sens.channels == [str(c) for c in review.findings.sort_values("rank")["channel"]][:3]
    assert list(sens.rates.columns) == sens.thresholds
    assert set(sens.rates.index) == set(review.findings["channel"].astype(str))
    # A stricter threshold never finds more events.
    assert (sens.counts[sens.thresholds[1]] <= sens.counts[sens.thresholds[0]]).all()
    assert len(sens.leaders) == len(sens.tied) == len(sens.stands_out) == 2
    survives = sens.survives_to()
    assert survives in (None, 1.0, 2.0)
    text = sensitivity.describe(sens)
    assert "the window's own" in text and ("leads" in text or "nothing detected" in text)
    handed = sensitivity.as_dict(sens)
    assert handed["available"] and handed["window_leader_survives_to_factor"] == survives
    assert set(handed["leading_channels_rate_per_min"]) == set(sens.channels)


def test_the_threshold_re_test_says_why_when_it_cannot_run(review):
    from onset_review import sensitivity

    class Bare:
        request = review.request
        findings = review.findings
        recording = None

    sens = sensitivity.compute(Bare())
    assert not sens.available and "no recording" in sens.reason
    assert "Not re-tested" in sensitivity.describe(sens)
    assert sensitivity.as_dict(sens) == {"available": False, "reason": sens.reason}


# --------------------------------------------------------------------------
# The findings draft, and the other windows of the same recording
# --------------------------------------------------------------------------

def test_the_findings_draft_is_filed_with_who_wrote_it_and_survives_the_round_trip():
    from onset_review.adjudication import Adjudication

    read = Adjudication(window="w", reader="Dr A")
    assert not read.draft and not read.draft_is_assistants
    read.set_draft("  AR1-AR2 led at 52/min.  ", "assistant (scripted (no language model))")
    assert read.draft == "AR1-AR2 led at 52/min." and read.draft_is_assistants
    assert read.draft_at.endswith("Z")
    again = Adjudication.from_json(read.to_json())
    assert (again.draft, again.draft_by, again.draft_at) == (read.draft, read.draft_by, read.draft_at)
    read.set_draft("AR1-AR2 led at 52/min, which I confirmed on the trace.", "Dr A")
    assert not read.draft_is_assistants and read.draft_by == "Dr A"
    read.set_draft("", "Dr A")
    assert read.draft == "" and read.draft_by == "" and read.draft_at == ""


def test_the_report_prints_the_draft_as_the_assistants_until_the_reader_edits_it(review):
    from onset_review import report

    read = review.read
    before = (read.draft, read.draft_by, read.draft_at)
    try:
        read.set_draft("", "")
        assert "## Findings" not in report.review_markdown(review)
        read.set_draft("AR1-AR2 led at 52/min.", "assistant (scripted (no language model))")
        text = report.review_markdown(review)
        assert "## Findings" in text and "AR1-AR2 led at 52/min." in text
        assert "Drafted by the assistant (scripted (no language model))" in text
        assert "has not been edited by the reader" in text
        assert text.index("## Findings") < text.index("## Per-channel findings")
        read.set_draft("AR1-AR2 led at 52/min; I agree.", "Dr A")
        text = report.review_markdown(review)
        assert "_Written by Dr A on" in text and "Drafted by" not in text
    finally:
        read.draft, read.draft_by, read.draft_at = before


def _other_window_fixture(review, recording):
    """A cache listing with this window and one more, and a loader that
    analyses the synthetic recording instead of fetching anything."""
    import pandas as pd

    from onset_review.session import session_from_recording

    request = review.request
    span = request.span()
    rows = [{"dataset": request.dataset, "subject": request.subject, "task": None,
             "run": request.run, "t_start": span[0], "t_stop": span[1], "sfreq": 2000.0,
             "n_channels": 50, "bands": "ripple", "path": "x"},
            {"dataset": request.dataset, "subject": request.subject, "task": None,
             "run": request.run, "t_start": span[1], "t_stop": span[1] + (span[1] - span[0]),
             "sfreq": 2000.0, "n_channels": 50, "bands": "ripple", "path": "y"},
            {"dataset": request.dataset, "subject": "sub-99", "task": None, "run": "01",
             "t_start": 0.0, "t_stop": 60.0, "sfreq": 2000.0, "n_channels": 50,
             "bands": "ripple", "path": "z"}]
    frame = pd.DataFrame(rows)
    loads = []

    def loader(request, _cache_dir=None):
        loads.append((request.t_start, request.t_stop))
        return session_from_recording(recording, request)

    return (lambda _cache_dir=None: frame), loader, loads


def test_the_other_windows_are_this_recordings_and_not_this_one(review, recording, monkeypatch):
    from onset_review import windows

    windows._CACHE.clear()
    monkeypatch.setattr(windows, "_cache_dir", lambda: None)
    lister, loader, loads = _other_window_fixture(review, recording)
    others = windows.other_windows(review, lister=lister)
    span = review.request.span()
    assert len(others) == 1 and others[0]["t_start"] == span[1]
    assert others[0]["analysed"] is False and others[0]["duration_s"] == round(span[1] - span[0], 1)
    assert windows.other_windows(review, lister=lambda _c=None: None) == []


def test_another_window_is_analysed_with_this_windows_settings_once(review, recording,
                                                                     monkeypatch, tmp_path):
    from onset_review import windows

    windows._CACHE.clear()
    monkeypatch.setattr(windows, "_cache_dir", lambda: tmp_path / "windows")
    lister, loader, loads = _other_window_fixture(review, recording)
    other = windows.other_windows(review, lister=lister)[0]
    there = windows.analyse_window(review, other["t_start"], other["t_stop"], loader=loader)
    assert loads == [(other["t_start"], other["t_stop"])] and there["cached"] is False
    assert there["leader"] and there["leading"] and there["window_s"] == [other["t_start"], other["t_stop"]]
    again = windows.analyse_window(review, other["t_start"], other["t_stop"], loader=loader)
    assert len(loads) == 1 and again["cached"] is True
    assert windows.other_windows(review, lister=lister)[0]["analysed"] is True
    # The disk copy serves a fresh process too.
    windows._CACHE.clear()
    third = windows.analyse_window(review, other["t_start"], other["t_stop"], loader=loader)
    assert len(loads) == 1 and third["cached"] is True
    here = windows.summarise(review)
    comparison = windows.compare(here, there)
    assert comparison["leader_this_window"] == here["leader"]
    assert comparison["leader_changed"] == (here["leader"] != there["leader"])
    assert {m["channel"] for m in comparison["leading_channels"]} >= {here["leader"]}
    assert "Compare the tied sets" in comparison["note"]
    # A different band is a different analysis, filed apart.
    import dataclasses

    other_band = dataclasses.replace(review.request, band="fast_ripple")
    assert windows.window_key(other_band) != windows.window_key(review.request)
