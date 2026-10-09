"""Two analyses side by side.

Asked for: after changing a setting, see what moved. What has to hold: the
same analysis compared with itself moves nothing; two montages that name
channels differently are compared by contact, and say so; the events found
by both and by one add up; the verdicts that differ are listed; and the
window shows it, saves it and opens the other analysis.
"""

from __future__ import annotations

import dataclasses
import os
import types

import numpy as np
import pytest

from onset_hfo.config import PreprocessConfig
from onset_review import comparison
from onset_review.session import ReviewRequest, session_from_recording


@pytest.fixture(scope="module")
def sessions(recording):
    base = ReviewRequest(t_start=0.0, t_stop=float(recording.duration),
                         detectors=("rms", "line_length"))
    bipolar = session_from_recording(recording, base)
    laplacian = session_from_recording(recording, dataclasses.replace(
        base, preprocess=PreprocessConfig(reference="laplacian", grid_columns=(("SA", 3),))))
    return bipolar, laplacian


def test_an_analysis_compared_with_itself_moves_nothing(sessions):
    a, _ = sessions
    c = comparison.compare(a, a)
    assert c.unit == "channel" and c.settings == []
    assert c.rho == pytest.approx(1.0) and c.top_shared == comparison.TOP
    assert (c.ranking["moved"].dropna() == 0).all()
    assert (c.events["only_a"] == 0).all() and (c.events["only_b"] == 0).all()
    assert (c.events["both"] == c.events["in_a"]).all()
    assert c.sentences[0] == "The two were analysed with the same settings."
    assert "the same place" in " ".join(c.sentences)


def test_two_montages_are_compared_by_contact_and_add_up(sessions):
    a, b = sessions
    c = comparison.compare(a, b, "bipolar", "Laplacian")
    assert c.unit == "contact"
    assert ("reference", "bipolar", "laplacian") in c.settings
    assert ("grid columns", "–", "SA:3") in c.settings
    assert set(c.ranking["contact"]) == {n.upper() for n in b.findings["channel"]}
    # A contact's rate is that of the busiest channel built from it.
    rates = dict(zip(a.findings["channel"], a.findings["rate_per_min"], strict=True))
    sa2 = max(rates["SA1-SA2"], rates["SA2-SA3"])
    assert c.ranking.set_index("contact").loc["SA2", "rate_a"] == pytest.approx(sa2)
    for row in c.events.itertuples():
        assert row.both + row.only_a == row.in_a and row.both + row.only_b == row.in_b
        assert row.both > 0
    text = " ".join(c.sentences)
    assert "compared by contact" in text and "reference (bipolar → laplacian)" in text
    assert np.isfinite(c.rho)


def test_events_match_on_a_shared_contact_within_the_tolerance():
    def event(channel, start, stop=None):
        return types.SimpleNamespace(channel=channel, start=start,
                                     stop=start + 0.05 if stop is None else stop)

    a = [event("A1-A2", 1.0), event("B1-B2", 2.0), event("A1-A2", 5.0)]
    b = [event("A2", 1.03), event("B3", 2.0), event("A1", 5.06)]
    # A2 shares a contact and overlaps; B3 shares none; A1 starts 10 ms after
    # A1-A2 ends, inside the 20 ms tolerance.
    assert comparison._match(a, b, by_contact=True) == 2
    assert comparison._match(a, b, by_contact=False) == 0
    assert comparison.contacts_of_channel("sa1-sa2") == ["SA1", "SA2"]


def test_verdicts_that_differ_are_listed(sessions):
    a, _ = sessions
    b = dataclasses.replace(a, read=dataclasses.replace(a.read, channels={}))
    a.read.judge_channel("SA1-SA2", "accept", reader="test")
    try:
        c = comparison.compare(a, b)
        assert c.verdicts.to_dict("records") == [
            {"channel": "SA1-SA2", "verdict_a": "accept", "verdict_b": "–"}]
        assert "1 channel verdict(s) differ" in " ".join(c.sentences)
        page = comparison.to_markdown(c, a.request, b.request)
        assert "## Channel verdicts that differ" in page and "| SA1-SA2 | accept | – |" in page
    finally:
        a.read.channels.clear()


def test_the_markdown_has_what_differs_and_every_contact(sessions):
    a, b = sessions
    c = comparison.compare(a, b)
    page = comparison.to_markdown(c, a.request, b.request)
    assert page.startswith("# Comparison — A and B")
    assert "| reference | bipolar | laplacian |" in page
    assert page.count("\n| SA") + page.count("\n| SB") + page.count("\n| SC") == len(c.ranking)
    assert "## Accepted events" in page


# -- the window ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_the_window_shows_saves_and_opens_the_other(qapp, sessions, tmp_path):
    from onset_review.comparewindow import CompareWindow

    a, b = sessions
    window = CompareWindow(a, b)
    try:
        assert window.ranking.rowCount() == len(window.result.ranking)
        assert window.events.rowCount() == 2
        assert window.tabs.tabText(0) == "Ranking, by contact"
        assert "reference (bipolar → laplacian)" in window.summary.text()
        saved = window.save(tmp_path / "c.md")
        assert saved.read_text().startswith("# Comparison")
        emitted = []
        window.openRequested.connect(emitted.append)
        window.open_button.click()
        assert emitted == [b.request]
    finally:
        window.close()


def test_other_settings_are_this_window_s_with_what_was_changed(qapp, sessions):
    from qtpy.QtWidgets import QDialog

    from onset_review.comparewindow import OtherSettingsDialog

    a, _ = sessions
    dialog = OtherSettingsDialog(a.request)
    assert dialog.reference.currentData() == "bipolar"
    dialog.reference.setCurrentIndex(dialog.reference.findData("laplacian"))
    dialog.grid.setText("SA:3")
    dialog.band.setCurrentIndex(dialog.band.findData("fast_ripple"))
    dialog.threshold.setValue(4.0)
    other = dialog.other()
    assert other.preprocess.reference == "laplacian" and not other.preprocess.bipolar
    assert other.preprocess.grid_columns == (("SA", 3),)
    assert (other.band, other.threshold_sd) == ("fast_ripple", 4.0)
    assert (other.t_start, other.t_stop, other.detectors) == \
        (a.request.t_start, a.request.t_stop, a.request.detectors)
    dialog.grid.setText("SA three")
    dialog._accept()
    assert dialog.result() != QDialog.Accepted and "SA three" in dialog.problem.text()


def test_the_app_analyses_the_other_and_shows_both(qapp, sessions, monkeypatch):
    from onset_review import launcher
    from onset_review.app import _Review

    a, b = sessions
    asked = []
    monkeypatch.setattr(launcher, "load_with_progress",
                        lambda request, cache_dir=None, parent=None: asked.append(request) or b)
    fake = types.SimpleNamespace(parts=types.SimpleNamespace(session=a),
                                 args=types.SimpleNamespace(cache_dir=None),
                                 _host=lambda: None, _open_request=lambda request: None)
    fake._show_comparison = lambda other, b_label="B": _Review._show_comparison(
        fake, other, b_label)
    window = _Review.compare_settings(fake, b.request)
    try:
        assert asked == [b.request] and window.result.unit == "contact"
        assert window.isVisible()
    finally:
        window.close()
