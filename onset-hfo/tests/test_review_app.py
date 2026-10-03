"""The review window: that it builds, and that its controls drive the trace.

A display is needed from the first line, so the import guard is at the top and
takes the whole file with it when the `review` extra is absent. The numbers are
not in here -- they are in `tests/test_review_core.py`, which has no Qt import
and runs everywhere. What is in here is the part that breaks silently when a
dependency changes underneath it: that MNE's figure is still a `QMainWindow`
and can host our docks, that every panel builds against a real session, and
that each control calls the browser method it claims to.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from onset_review import trends
from onset_review.session import ReviewRequest, ReviewSession, session_from_recording

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")

from qtpy.QtCore import Qt  # noqa: E402  (after the import guard, on purpose)


@pytest.fixture(scope="module")
def review(recording) -> ReviewSession:
    """One analysed window of the synthetic recording, reused by every test."""
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))

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
                                "brain", "agreement", "provenance", "assistant",
                                "preprocess"}
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


def test_the_brain_panel_says_it_is_not_anatomy(built):
    """The caption is the honesty guard and is not allowed to go missing.

    Either wording may apply -- a schematic layout when the electrode names map
    to structures, a montage diagram when none of them do -- and the headline
    has to agree with whichever the caption chose, or one of the two reads as
    boilerplate.
    """
    from onset_review.anatomy import NOT_ANATOMY

    caption = built.panels["brain"].caption.text()
    headline = built.panels["brain"].headline.text()
    assert NOT_ANATOMY in caption
    verdict = caption.split("—")[0].strip()
    assert verdict in ("SCHEMATIC LAYOUT", "MONTAGE DIAGRAM")
    assert headline.lower().startswith(verdict.lower())


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


# -- the preprocessing panel ----------------------------------------------

def test_the_preprocessing_panel_opens_on_the_measured_defaults(built):
    """Opening it and pressing Apply without touching it must change nothing,
    so Apply starts disabled and there is nothing to warn about."""
    from onset_hfo.config import PreprocessConfig

    panel = built.panels["preprocess"]
    assert panel.config() == PreprocessConfig()
    assert not panel.apply.isEnabled()
    assert panel.warnings.text() == ""


def test_every_control_reaches_the_config(built):
    panel = built.panels["preprocess"]
    try:
        panel.highpass.setValue(2.0)
        panel.notch_width.setValue(3.0)
        panel.harmonics.setChecked(False)
        panel.average.setChecked(True)
        cfg = panel.config()
        assert cfg.highpass == 2.0
        assert cfg.notch_width == 3.0
        assert cfg.notch_harmonics is False
        assert (cfg.bipolar, cfg.average_reference) == (False, True)
        assert panel.apply.isEnabled()          # something changed
    finally:
        panel.reset_to_defaults()


def test_reset_returns_to_the_measured_defaults(built):
    from onset_hfo.config import PreprocessConfig

    panel = built.panels["preprocess"]
    panel.highpass.setValue(5.0)
    panel.monopolar.setChecked(True)
    panel.channels.item(0).setCheckState(Qt.Checked)
    assert panel.config() != PreprocessConfig()
    panel.reset_to_defaults()
    assert panel.config() == PreprocessConfig()
    assert not panel.apply.isEnabled()


def test_marking_a_contact_excludes_it(built):
    panel = built.panels["preprocess"]
    try:
        name = panel.channels.item(0).text()
        panel.channels.item(0).setCheckState(Qt.Checked)
        assert panel.config().exclude == (name,)
        # Contacts, not channels: exclusion happens before the bipolar montage
        # is built, which is what makes it a preprocessing choice.
        assert "-" not in name
    finally:
        panel.reset_to_defaults()


def test_a_band_destroying_setting_is_refused_by_the_panel(built):
    """Not merely warned about: Apply declines, so the reviewer finds out from
    the red text rather than from a dialog after a re-analysis."""
    panel = built.panels["preprocess"]
    emitted = []
    panel.applied.connect(emitted.append)
    try:
        panel.lowpass.setValue(150.0)
        # `isVisible` is False for anything inside a window that was never
        # shown, which is every widget here; what the reviewer reads is the
        # text, and that is what this is about.
        assert "cuts into the band" in panel.warnings.text()
        panel._apply()
        assert emitted == []
        assert "Fix this before applying" in panel.warnings.text()
    finally:
        panel.reset_to_defaults()


def test_a_sound_change_is_emitted(built):
    panel = built.panels["preprocess"]
    emitted = []
    panel.applied.connect(emitted.append)
    try:
        panel.notch_width.setValue(3.0)
        panel._apply()
        assert len(emitted) == 1
        assert emitted[0].notch_width == 3.0
    finally:
        panel.reset_to_defaults()


def test_apply_is_disabled_when_nothing_can_act_on_it(review):
    """`decorate` without a reload callback must not offer a button that does
    nothing when pressed."""
    from onset_review import window as window_module

    figure = window_module.open_trace(review, show=False)
    parts = window_module.decorate(figure, review)       # no on_preprocess
    try:
        assert not parts.panels["preprocess"].apply.isEnabled()
    finally:
        figure.close()
