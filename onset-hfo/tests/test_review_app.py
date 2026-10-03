"""The desktop reviewer: the numbers it shows, and that the window builds.

Two halves, deliberately uneven. The Qt-free half is thorough, because every
figure a clinician reads off the screen is computed in `onset_review.session`
and `onset_review.trends` and can be checked here without a display. The Qt
half is a smoke test: it builds the real window on the real session and asserts
the structure the layout depends on, which is the part that breaks silently
when a dependency changes underneath it.

Everything runs on the synthetic recording, so none of it needs a cached slice
of `ds003498` or a network. That is why `session_from_recording` exists as a
function separate from `load_session`.
"""

from __future__ import annotations

import dataclasses

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


# -- the window ------------------------------------------------------------
#
# From here on a display is needed. These are skipped rather than failed when
# Qt is absent, because the review extra is optional and the whole Qt-free half
# above is what guards the numbers. What they do guard is the structural
# assumption the layout rests on -- that MNE's figure is still a QMainWindow
# and can host our docks -- which is exactly the thing that would otherwise
# break silently on a dependency bump.

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MNE_BROWSER_BACKEND", "qt")
    app = qt.QApplication.instance() or qt.QApplication([])
    yield app


@pytest.fixture(scope="module")
def built(qapp, review):
    from onset_review import window

    figure = window.open_trace(review, show=False)
    parts = window.decorate(figure, review)
    yield parts
    try:
        figure.close()
    except Exception:
        pass


def test_mne_figure_can_still_host_our_docks(built):
    """The one architectural assumption of the whole window.

    `mne-qt-browser` returns a QMainWindow today, which is why the panels can be
    docked onto the trace instead of living in a second window. If a release
    ever changes that, this is where it should be noticed.
    """
    from onset_review import window

    assert window.has_dock_host(built.figure)
    assert built.docked


def test_every_panel_is_docked(built):
    assert set(built.docks) == {"trends", "controls", "findings", "events",
                                "brain", "agreement", "provenance", "assistant"}
    assert all(dock.widget() is not None for dock in built.docks.values())


def test_the_caveat_is_on_the_status_bar(built, review):
    """It is not allowed to be somewhere a reviewer might not look."""
    label = built.host.statusBar().findChild(qt.QLabel, "onset_caveat")
    assert label is not None
    assert label.text() == review.caveat()


def test_the_findings_panel_shows_every_channel(built, review):
    assert built.panels["findings"].model.rowCount() == len(review.findings)


def test_the_events_panel_filters_down_to_discharges(built, review):
    panel = built.panels["events"]
    panel.kind.setCurrentIndex(3)           # Discharges only
    panel.refilter()
    assert panel.model.rowCount() == len(review.spikes)
    panel.kind.setCurrentIndex(0)
    panel.refilter()
    assert panel.model.rowCount() == len(review.accepted)


def test_stepping_the_event_list_moves_one_row(built):
    panel = built.panels["events"]
    panel.step(+1)
    first = panel.view.selectionModel().selectedRows()[0].row()
    panel.step(+1)
    assert panel.view.selectionModel().selectedRows()[0].row() == first + 1


def test_marks_default_to_one_channel(built, review):
    """The default that makes the trace readable at all.

    Marking every detection draws five hundred full-height spans over forty
    channels of signal; the reviewer sees a barcode. Scoping to the channel
    under review is the default, and this is the test that keeps it one.
    """
    from onset_review import window

    everything = window.marks_for(review, scope="all")
    one = window.marks_for(review, scope="selected",
                           channel=review.leader.get("leader"))
    assert len(one) < len(everything)
    assert len(window.marks_for(review, scope="none")) == 0
    assert built.display.scope == "selected"


def test_the_expert_overlay_is_labelled_apart_from_ours(review):
    """Same band, two sources: MNE colours by description, so they must differ.

    The expert marks are grafted on rather than taken from the recording,
    because the synthetic one carries none and this path -- the comparison that
    makes the product worth sitting with -- should not go untested for want of
    a cached slice of `ds003498`.
    """
    import dataclasses

    from onset_hfo.detectors.base import Event
    from onset_review import window
    from onset_review.session import BAND_COLOURS

    channel = review.findings["channel"].iloc[0]
    expert = [Event(channel=channel, start=t, stop=t + 0.05, detector="expert",
                    band=(80.0, 250.0), accepted=True) for t in (1.0, 2.0, 3.0)]
    annotated = dataclasses.replace(review, expert=expert,
                                    reviewed_channels=[channel])

    overlay = window.marks_for(annotated, scope="all", expert=True)
    plain = window.marks_for(annotated, scope="all", expert=False)
    assert len(overlay) == len(plain) + len(expert)
    assert "expert ripple" in set(overlay.description)
    assert "expert ripple" not in set(plain.description)
    # Every description the trace can carry must have a colour, or MNE invents
    # one and the two sources stop being told apart by eye.
    assert set(overlay.description) <= set(BAND_COLOURS)

    # And the agreement table has to appear now that there is something to
    # agree with, restricted to the one channel the annotator looked at.
    summary = trends.agreement_summary(annotated)
    assert summary["available"] is True
    assert summary["n_expert"] == len(expert)
    assert list(trends.agreement(annotated)["channel"]) == [channel]


def test_a_trend_cell_maps_back_to_a_channel(built, review):
    panel = built.panels["trends"]
    panel.redraw()
    matrix = trends.rate_matrix(review, bin_s=float(panel.bin_s.currentData()))
    assert list(matrix.index) == list(review.findings["channel"])


# -- the open dialog -------------------------------------------------------
#
# The first screen a reviewer sees, and the one place a mistake is invisible in
# a headless test of everything else: the dialog opened with no detector
# selected for a while, and `request()` quietly fell back to the right one, so
# nothing downstream noticed.

@pytest.fixture
def cache(tmp_path):
    """Two cached windows: one 2000 Hz, one 1000 Hz, written as the loader does."""
    import json

    for meta in (
        {"dataset": "ds003498", "subject": "sub-01", "task": None, "run": "01",
         "t_start": 0, "t_stop": 60, "sfreq": 2000.0, "n_channels": 50},
        {"dataset": "ds003029", "subject": "sub-pt01", "task": "ictal",
         "run": "01", "t_start": 50, "t_stop": 60, "sfreq": 1000.0,
         "n_channels": 98},
    ):
        name = (f"{meta['subject']}_"
                + (f"task-{meta['task']}_" if meta["task"] else "")
                + f"run-{meta['run']}_{meta['t_start']:g}-{meta['t_stop']:g}s")
        directory = tmp_path / meta["dataset"] / name
        directory.mkdir(parents=True)
        (directory / "slice.json").write_text(json.dumps(meta))
    return tmp_path


def test_the_dialog_opens_with_a_detector_selected(qapp, cache):
    from onset_review.launcher import DEFAULT_DETECTOR, LauncherDialog

    dialog = LauncherDialog(cache)
    from qtpy.QtCore import Qt

    chosen = [item.data(Qt.UserRole) for item in dialog.detectors.selectedItems()]
    assert chosen == [DEFAULT_DETECTOR]
    assert dialog.request().detectors == (DEFAULT_DETECTOR,)


def test_the_dialog_offers_only_the_bands_the_rate_supports(qapp, cache):
    """A 500 Hz band on a 500 Hz Nyquist is meaningless, not conservative."""
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    bands = {}
    for index in range(dialog.subject.count()):
        dialog.subject.setCurrentIndex(index)
        _, subject = dialog.subject.itemData(index)
        bands[subject] = [dialog.band.itemData(i) for i in range(dialog.band.count())]
    assert bands["sub-01"] == ["ripple", "fast_ripple"]
    assert bands["sub-pt01"] == ["ripple"]


def test_the_dialog_says_why_a_band_is_missing(qapp, cache):
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    for index in range(dialog.subject.count()):
        if dialog.subject.itemData(index)[1] == "sub-pt01":
            dialog.subject.setCurrentIndex(index)
    assert "1000 Hz" in dialog.status.text()
    assert "interictal sleep" in dialog.status.text()      # and that it is ictal


def test_the_dialog_builds_the_request_it_is_showing(qapp, cache):
    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(cache)
    for index in range(dialog.subject.count()):
        if dialog.subject.itemData(index)[1] == "sub-pt01":
            dialog.subject.setCurrentIndex(index)
    request = dialog.request()
    assert (request.dataset, request.subject, request.task) == (
        "ds003029", "sub-pt01", "ictal")
    assert (request.t_start, request.t_stop) == (50.0, 60.0)
    # The spin box sits at its minimum until a reviewer moves it, and that
    # means "each detector's own measured threshold", not "1.0 SD".
    assert request.threshold_sd is None
    dialog.threshold.setValue(3.5)
    assert dialog.request().threshold_sd == 3.5


def test_an_empty_cache_disables_opening_and_says_what_to_do(qapp, tmp_path):
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(tmp_path)
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    assert "fetch" in dialog.status.text().lower()


# -- the trace controls ----------------------------------------------------
#
# The bar does not decide anything: it calls MNE's own view methods and reads
# its readouts back out of MNE. These tests pin exactly that, because the
# failure mode of a control bar that keeps its own copy of the state is that it
# silently disagrees with the trace it sits above.

def test_the_controls_read_their_values_from_the_browser(built):
    controls = built.panels["controls"]
    state = built.figure.mne
    assert controls.seconds.value() == pytest.approx(float(state.duration))
    assert controls.channels.value() == int(state.n_channels)
    assert controls.position.value() == pytest.approx(float(state.t_start))


def test_the_gain_readout_is_mnes_own_scalebar(built):
    """A readout that disagrees with the scalebar drawn on the trace is worse
    than no readout."""
    controls = built.panels["controls"]
    assert controls.gain.text() in set(built.figure._get_scale_bar_texts())


def test_scaling_changes_the_trace_and_the_readout(built):
    from onset_review.controls import AMPLITUDE_STEP

    controls = built.panels["controls"]
    before = float(built.figure.mne.scale_factor)
    controls._scale(AMPLITUDE_STEP)
    assert float(built.figure.mne.scale_factor) == pytest.approx(
        before * AMPLITUDE_STEP)
    assert controls.gain.text() in set(built.figure._get_scale_bar_texts())
    controls._scale(1 / AMPLITUDE_STEP)        # put it back for other tests
    assert float(built.figure.mne.scale_factor) == pytest.approx(before)


def test_changing_the_window_length_goes_through_the_browser(built):
    """And lands on the seconds asked for.

    `change_duration` takes a fraction of the current duration, not seconds, so
    a control that passes the difference in seconds overshoots badly -- 10 s to
    "12 s" becomes 30 s. That is the bug this asserts against.
    """
    controls = built.panels["controls"]
    before = float(built.figure.mne.duration)
    controls.seconds.setValue(before + 2.0)
    assert float(built.figure.mne.duration) == pytest.approx(before + 2.0, abs=0.1)
    controls.seconds.setValue(before)
    assert float(built.figure.mne.duration) == pytest.approx(before, abs=0.1)


def test_changing_the_channel_count_goes_through_the_browser(built):
    controls = built.panels["controls"]
    before = int(built.figure.mne.n_channels)
    controls.channels.setValue(before + 1)
    assert int(built.figure.mne.n_channels) == before + 1
    controls.channels.setValue(before)


def test_scrolling_moves_the_window_and_stays_inside_it(built):
    controls = built.panels["controls"]
    # A window as long as the recording has nowhere to scroll, and an earlier
    # test could have left it that way; make the state this test needs.
    controls.seconds.setValue(min(5.0, float(built.figure.mne.xmax) / 3))
    controls.go_home()
    assert controls.position.value() == pytest.approx(0.0)
    controls._hscroll("right")
    assert float(built.figure.mne.t_start) > 0.0
    for _ in range(50):                 # far past the end
        controls._hscroll("+full")
    state = built.figure.mne
    assert float(state.t_start) <= float(state.xmax) - float(state.duration) + 1e-6
    controls.go_home()


def test_a_control_whose_method_is_missing_leaves_the_window_standing(built):
    """The reviewer still has the keyboard; a dead button must not raise."""
    controls = built.panels["controls"]
    controls._call("no_such_method_on_the_browser", step=1)


# -- the 3D view -----------------------------------------------------------

def test_the_brain_panel_places_every_channel(built, review):
    panel = built.panels["brain"]
    assert len(panel.layout_frame) == len(review.findings)
    assert list(panel.layout_frame["channel"]) == list(
        review.findings.sort_values("rank")["channel"])


def test_the_brain_panel_says_it_is_schematic(built):
    """The caption is the honesty guard and is not allowed to go missing."""
    assert "SCHEMATIC" in built.panels["brain"].caption.text()
    assert "Schematic" in built.panels["brain"].headline.text()


def test_the_brain_panel_survives_every_option(built):
    panel = built.panels["brain"]
    for index in range(panel.colour_by.count()):
        panel.colour_by.setCurrentIndex(index)
        for labels in (True, False):
            panel.labels.setChecked(labels)
            panel.shafts.setChecked(not labels)
            panel.redraw()
    panel.colour_by.setCurrentIndex(0)
    panel.labels.setChecked(True)
    panel.shafts.setChecked(True)


def test_every_preset_view_is_a_real_angle(built):
    from onset_review.brainview import VIEWS

    panel = built.panels["brain"]
    for name in VIEWS:
        panel.set_view(name)
        assert (panel._elev, panel._azim) == VIEWS[name]


def test_highlighting_turns_to_the_right_hemisphere(built):
    panel = built.panels["brain"]
    frame = panel.layout_frame
    for sign, expected in ((1, 0), (-1, 180)):
        rows = frame[np.sign(frame["x"]) == sign]
        if rows.empty:
            continue
        panel.highlight(str(rows.iloc[0]["channel"]))
        assert panel._azim == expected


# -- the assistant ---------------------------------------------------------

def test_an_evidence_id_resolves_to_a_channel_and_a_time():
    from onset_review.assistant import parse_evidence_id

    assert parse_evidence_id("sub-01|AR1-AR2|rms|3.505") == ("AR1-AR2", 3.505)
    assert parse_evidence_id("not an id") is None
    assert parse_evidence_id("sub-01|AR1-AR2|rms|not-a-number") is None


def test_a_clicked_citation_survives_qts_url_encoding(qapp, review):
    """Qt percent-encodes the `|` in an evidence id on the way into a QUrl.

    Without unquoting, every citation link silently does nothing when clicked
    -- and the clickable citation is the entire argument that the assistant's
    answers can be checked rather than trusted.
    """
    from qtpy.QtCore import QUrl

    from onset_review.assistant import AssistantPanel

    panel = AssistantPanel(review)
    seen = []
    panel.evidencePicked.connect(lambda channel, t: seen.append((channel, t)))
    panel._citation_clicked(QUrl("evidence:sub-01|AR1-AR2|rms|3.505"))
    assert seen == [("AR1-AR2", 3.505)]
    assert "%7C" in QUrl("evidence:sub-01|AR1-AR2").toString()   # the hazard
    panel.deleteLater()


def test_the_assistant_offers_a_question_it_will_refuse():
    """A reviewer should meet the boundary from the interface, early."""
    from onset_review.assistant import SUGGESTIONS

    assert any("resect" in question.lower() for question, _ in SUGGESTIONS)
    assert all(label and len(label) < 24 for _, label in SUGGESTIONS)


def test_the_default_backend_runs_no_model():
    from onset_review.assistant import BACKENDS

    assert BACKENDS[0][1] == "scripted"
    assert any(kind == "ollama" for _, kind in BACKENDS)


def test_the_request_rebuilds_the_config_that_produced_the_review(review):
    """The assistant answers from a re-run; it has to be the *same* run.

    `pipeline_config` is what makes the review on screen and a `run_pipeline`
    over the same request one analysis rather than two that happen to agree.
    """
    from onset_hfo.detectors import HFO_DETECTORS

    config = review.request.pipeline_config()
    for name in HFO_DETECTORS:
        assert getattr(config, name).band == review.request.band_hz

    strict = dataclasses.replace(review.request, threshold_sd=4.25)
    for name in HFO_DETECTORS:
        assert getattr(strict.pipeline_config(), name).threshold_sd == 4.25


def test_the_assistant_answers_with_the_numbers_on_screen(qapp, review):
    """The point of the panel, and the thing that would rot silently.

    An assistant quoting rates from a differently-configured run would be worse
    than none: the numbers would look like the table's and not be them. This
    builds the store the agent reads and checks it against the session.
    """
    from onset_review.assistant import AssistantPanel

    panel = AssistantPanel(review)
    try:
        assert panel._ensure_store()
        rates = panel._store.rates[review.request.primary]
        mine = review.findings.set_index("channel")["n_events"]
        theirs = rates.set_index("channel")["n_events"]
        for channel, count in mine.items():
            assert int(theirs.get(channel, 0)) == int(count), channel
    finally:
        panel.deleteLater()


def test_the_assistant_keeps_the_window_painting_while_it_works(qapp, review):
    """It must not block the GUI thread while a model is thinking.

    `QThread.start()` followed by `wait()` on the GUI thread stops the window
    repainting entirely: the "Thinking…" line never appears, the panel does not
    visibly disable, and behind a local model that is half a minute of an
    application the desktop reports as not responding. The fix is a nested
    event loop, and this is what keeps it.
    """
    import inspect

    from onset_review import assistant

    source = inspect.getsource(assistant)
    assert "QEventLoop" in source
    # Exactly one place waits, and it is after the loop has already drained.
    waits = [line.strip() for line in source.splitlines()
             if ".wait()" in line and not line.strip().startswith("#")]
    assert waits == ["worker.wait()"], waits

    panel = assistant.AssistantPanel(review)
    try:
        painted = []
        panel.ask("Which channels have the highest ripple rate?")
        qapp.processEvents()
        painted.append(panel.transcript.toPlainText())
        assert "Thinking" in painted[0]
        assert panel.isEnabled()          # re-enabled when the answer landed
    finally:
        panel.deleteLater()
