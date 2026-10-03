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
                                "preprocess", "patient"}
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


# -- the patient panel and the theme ---------------------------------------

def test_the_patient_panel_shows_the_record_and_hides_the_outcome(built, review):
    """The outcome is hidden behind a deliberate click.

    Knowing the patient became seizure-free changes how the same rate table
    reads, and this software is for forming an impression from the signal.
    """
    panel = built.panels["patient"]
    if not panel.record.get("available"):
        pytest.skip("the synthetic recording has no participant record")
    assert not panel.outcome.isVisibleTo(panel)
    panel.outcome_shown.setChecked(True)
    assert panel.outcome.isVisibleTo(panel)
    assert "ILAE" in panel.outcome.text()


def test_the_patient_panel_opens_on_a_subject_with_no_record(review):
    """A recording from outside the cohort must not take the panel down."""
    import dataclasses

    from onset_review.patient import PatientPanel

    stranger = dataclasses.replace(
        review, request=dataclasses.replace(review.request, subject="sub-99"))
    panel = PatientPanel(stranger)
    try:
        assert panel.record["available"] is False
    finally:
        panel.deleteLater()


def test_the_theme_sets_a_palette_and_a_stylesheet(qapp):
    """Forcing a theme with CSS alone leaves platform-drawn controls -- check
    indicators, spin-box arrows -- rendered for the other one."""
    from onset_review import theme

    for palette in (theme.LIGHT, theme.DARK):
        chosen = theme.apply_theme(qapp, palette)
        assert chosen is palette
        assert theme.current() is palette
        assert palette.accent in qapp.styleSheet()
        assert qapp.palette().base().color().name().lower() == \
            palette.surface.lower()
    theme.apply_theme(qapp, theme.LIGHT)


def test_panels_read_the_active_palette_rather_than_importing_one(review):
    """The dark theme's first version rendered tinted rows as light text on a
    light tint, because every panel had imported the light palette by name."""
    from onset_review import theme
    from onset_review.panels import FindingsPanel

    try:
        theme.set_current(theme.DARK)
        dark = FindingsPanel(review)
        tinted = [theme.DARK.highlight, theme.DARK.surface_alt, None]
        row = dark.model.frame.iloc[0]
        assert dark.model._highlight(row) in tinted
        dark.deleteLater()
    finally:
        theme.set_current(theme.LIGHT)


def test_every_warning_in_the_interface_is_the_same_warning_colour():
    """Before `theme.card` there were four hand-written hex values for this and
    two of them disagreed."""
    from onset_review import theme

    for kind in ("info", "warn", "bad", "plain"):
        assert "border-radius" in theme.card(kind)
    assert theme.LIGHT.bad in theme.card("bad", theme.LIGHT)
    assert theme.DARK.bad in theme.card("bad", theme.DARK)


def test_no_panel_hardcodes_a_colour():
    """Colours come from the palette so the two themes cannot diverge.

    Three files are exempt, each for its own reason. `theme.py` is where the
    colours live. `session.BAND_COLOURS` and `brainview.ZONE_EDGES` are data,
    not styling -- what a ripple is drawn as, what "inside the resection" is
    drawn as -- and they mean the same thing on either background. And
    `report.py` styles an exported HTML document that is read in a browser or
    printed, which should not inherit whatever theme the application happened
    to be in when it was written.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "onset_review"
    allowed = {"theme.py", "session.py", "brainview.py", "report.py"}
    for path in sorted(root.glob("*.py")):
        if path.name in allowed:
            continue
        hits = re.findall(r"#[0-9a-fA-F]{6}\b", path.read_text())
        assert not hits, f"{path.name} hardcodes {hits}"


# -- importing a file from this machine -----------------------------------
#
# The one screen in this software whose job is to stop something, rather than
# to show something. A clinical EDF or BrainVision export declares every
# channel `eeg` -- including the one real iEEG file this project's own cache
# holds -- and the pipeline analyses anything typed eeg, ecog or seeg. So an
# import that trusts the file produces a full review, with rates, intervals and
# a candidate channel set, for a scalp montage, and says nothing. These tests
# pin the parts of the dialog that exist to prevent exactly that.

@pytest.fixture(scope="module")
def recording_file(tmp_path_factory):
    """A file on disk that declares every channel `eeg`, as exports do."""
    import mne

    rng = np.random.default_rng(11)
    names = ["AR1", "AR2", "AR3", "HL1", "EKG", "DC1"]
    data = rng.normal(0.0, 2e-5, size=(len(names), 20 * 2000))
    info = mne.create_info(names, 2000.0, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path_factory.mktemp("import") / "study-001_raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    return path


def test_the_import_dialog_shows_the_file_s_own_answer_beside_the_choice(
        qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    assert dialog.error == ""
    assert dialog.table.rowCount() == 6
    declared = [dialog.table.item(row, 1).text() for row in range(6)]
    assert declared == ["eeg"] * 6                 # what the file claims
    assert "6 × eeg" in dialog._declared_summary()
    assert set(dialog.channel_types().values()) == {"seeg"}


def test_the_import_dialog_refuses_to_open_with_nothing_to_analyse(
        qapp, recording_file):
    """The one disabled button in this software that is an argument.

    Everything typed ecg or misc is dropped in preprocessing, so a file where
    nothing is marked SEEG or ECoG would load, preprocess down to no channels
    and fail somewhere unhelpful. Refusing here, with the reason on the button,
    is the same refusal made legible.
    """
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    button = dialog.buttons.button(QDialogButtonBox.Open)
    assert button.isEnabled()
    assert dialog.counts.text() == "6 of 6 will be analysed"

    dialog.set_all("ecg")
    assert not button.isEnabled()
    assert dialog.counts.text() == "0 of 6 will be analysed"
    assert "nothing to analyse" in button.toolTip()

    dialog.set_all("ecog")
    assert button.isEnabled()
    assert dialog.counts.text() == "6 of 6 will be analysed"


def test_one_channel_changed_by_hand_updates_the_count(qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    ekg = [row for row in range(dialog.table.rowCount())
           if dialog.table.item(row, 0).text() == "EKG"][0]
    chooser = dialog.table.cellWidget(ekg, 2)
    chooser.setCurrentIndex(
        [i for i in range(chooser.count()) if chooser.itemData(i) == "ecg"][0])
    assert dialog.counts.text() == "5 of 6 will be analysed"
    assert dialog.channel_types()["EKG"] == "ecg"


def test_the_import_dialog_builds_the_request_it_is_showing(qapp, recording_file):
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    dialog.set_all("seeg")
    dialog.subject.setText("study-001")
    dialog.t_stop.setValue(15.0)
    dialog.line_freq.setCurrentIndex(1)              # 60 Hz
    request = dialog.request()

    assert request.imported is True
    assert request.path == recording_file
    assert request.subject == "study-001"
    assert (request.t_start, request.t_stop) == (0.0, 15.0)
    assert request.line_freq == 60.0
    assert dict(request.channel_types) == {name: "seeg" for name in
                                           ("AR1", "AR2", "AR3", "HL1",
                                            "EKG", "DC1")}


def test_the_window_cannot_be_set_past_the_end_of_the_file(qapp, recording_file):
    """A reviewer asking for 0–60 s of a 20 s file is corrected by the spin box.

    Not by an exception after the reader has run: by then they have waited, and
    the message is about a window rather than about the file being short.
    """
    from onset_review.importer import ImportDialog

    dialog = ImportDialog(recording_file)
    dialog.t_stop.setValue(600.0)
    assert dialog.t_stop.value() == pytest.approx(20.0, abs=0.01)
    assert dialog.t_stop.value() <= dialog.info["duration"] + 0.01


def test_the_import_dialog_offers_only_the_bands_the_file_can_support(
        qapp, tmp_path_factory):
    """The same refusal the launcher makes, made from the file's own header."""
    import mne

    from onset_review.importer import ImportDialog

    info = mne.create_info(["A1", "A2"], 1000.0, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 5000)), info, verbose="ERROR")
    slow = tmp_path_factory.mktemp("slow") / "slow_raw.fif"
    raw.save(slow, overwrite=True, verbose="ERROR")

    dialog = ImportDialog(slow)
    assert [dialog.band.itemData(i) for i in range(dialog.band.count())] == ["ripple"]
    assert "needs more than 1000 Hz" in _form_text(dialog)


def _form_text(dialog) -> str:
    from qtpy.QtWidgets import QLabel

    return " ".join(label.text() for label in dialog.findChildren(QLabel))


def test_an_unreadable_file_is_a_message_rather_than_a_traceback(
        qapp, tmp_path):
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    broken = tmp_path / "truncated_raw.fif"
    broken.write_bytes(b"not a fif file")
    dialog = ImportDialog(broken)
    assert dialog.table is None
    assert dialog.error
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()


def test_the_dialog_says_what_an_imported_file_does_not_bring(qapp,
                                                              recording_file):
    """Said before the reviewer finds the panels empty, not after."""
    from onset_review.importer import ImportDialog

    text = _form_text(ImportDialog(recording_file))
    assert "no expert markings" in text
    assert "not a medical device" in text
    assert "declare every channel as scalp EEG" in text


def test_the_launcher_offers_a_file_even_with_an_empty_cache(qapp, tmp_path):
    """An empty archive cache is exactly when someone has their own recording."""
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.launcher import LauncherDialog

    dialog = LauncherDialog(tmp_path)
    assert not dialog.buttons.button(QDialogButtonBox.Open).isEnabled()
    assert dialog.import_button.isEnabled()
    assert "open a file" in dialog.status.text().lower()


def test_an_imported_request_keeps_the_launcher_s_analysis_choices(
        qapp, cache, recording_file, monkeypatch):
    """The band comes from the file's rate; the detectors come from the dialog.

    Two screens, each answering what it is in a position to answer. Asking for
    the detector again on the import screen would be a third page nobody needs,
    and asking for the band on the launcher would offer one the file cannot
    support.
    """
    from onset_review import importer
    from onset_review.importer import ImportDialog
    from onset_review.launcher import LauncherDialog

    def fake_choose(parent=None, start=None):
        dialog = ImportDialog(recording_file)
        dialog.set_all("seeg")
        return dialog.request()

    monkeypatch.setattr(importer, "choose_file", fake_choose)

    dialog = LauncherDialog(cache)
    dialog.threshold.setValue(3.5)
    dialog.spikes.setChecked(False)
    dialog.expert.setChecked(True)
    dialog.open_file()

    request = dialog.request()
    assert request.imported is True
    assert request.threshold_sd == 3.5
    assert request.with_spikes is False
    assert request.band == "ripple"
    # The expert overlay belongs to the archive; an imported file has none, so
    # a checkbox left ticked from a previous choice must not travel with it.
    assert dialog.overlay_expert() is False


def test_a_cancelled_import_leaves_the_launcher_as_it_was(qapp, cache,
                                                          monkeypatch):
    from onset_review import importer
    from onset_review.launcher import LauncherDialog

    monkeypatch.setattr(importer, "choose_file", lambda parent=None, start=None: None)
    dialog = LauncherDialog(cache)
    dialog.open_file()
    assert dialog.request().imported is False


def test_the_review_menu_can_open_another_file(qapp, built):
    """And says so even when it cannot, rather than offering a dead entry."""
    actions = {action.text(): action
               for action in built.host.menuBar().actions()[0].menu().actions()}
    opener = [text for text in actions if "Open a file" in text]
    assert opener, list(actions)
    # `built` is decorated without an import callback, so the entry is there to
    # be found and explicitly disabled.
    assert not actions[opener[0]].isEnabled()
    assert "not available" in actions[opener[0]].toolTip()


def test_the_patient_panel_says_an_imported_window_has_no_record(qapp, recording):
    """Not "no record found": no record *exists*, and the difference matters."""
    import pathlib

    from onset_review.patient import PatientPanel

    request = ReviewRequest(dataset="", subject="study-001",
                            path=pathlib.Path("study-001_raw.fif"),
                            t_start=0.0, t_stop=float(recording.duration))
    review = session_from_recording(recording, request)
    panel = PatientPanel(review)
    text = " ".join(label.text() for label in panel.findChildren(qt.QLabel))
    assert "imported from study-001_raw.fif" in text
    assert "no participant record, no resected zone" in text
    # The archive's withheld-demographics note is about a cohort this window is
    # not part of, so it is not shown.
    assert "age, sex and handedness" not in text


def test_the_command_line_flags_open_the_dialog_on_their_answer(qapp,
                                                                recording_file):
    """`--window 0 30 --all-channels-as seeg` is an answer, not a suggestion.

    Discarding it and asking again would make the flags useless for anyone who
    wants to check what they typed before it runs.
    """
    from onset_review.app import _prefill, build_parser
    from onset_review.importer import ImportDialog

    args = build_parser().parse_args(
        ["--open", str(recording_file), "--window", "2", "12",
         "--line-freq", "60", "--subject", "study-001",
         "--all-channels-as", "seeg", "--channel-type", "EKG=ecg",
         "--channel-type", "NOT-A-CHANNEL=seeg"])

    dialog = ImportDialog(recording_file)
    _prefill(dialog, args)

    request = dialog.request()
    assert (request.t_start, request.t_stop) == (2.0, 12.0)
    assert request.line_freq == 60.0
    assert request.subject == "study-001"
    assert dict(request.channel_types)["EKG"] == "ecg"
    assert dict(request.channel_types)["AR1"] == "seeg"
    assert dialog.counts.text() == "5 of 6 will be analysed"
