"""A recording sampled too slowly for ripples: said, not crashed.

What has to hold: at 500 and 512 Hz (common in archives, HUP among them) the
pipeline skips the HFO detectors and says why, in the result and in the
saved report, while the interictal discharges and the quality checks still
run; at 1000 Hz nothing changes; the import dialog opens such a recording,
saying what will be skipped; the window opened on it has its discharges and a
note first saying why there are no HFOs; and the band is chosen after opening,
on the Signal page, which offers only what the rate can carry.
"""

from __future__ import annotations

import os

import mne
import numpy as np
import pytest

from onset_hfo.pipeline import run_pipeline
from onset_hfo.synthetic import make_synthetic_recording


def _at(sfreq: float):
    record = make_synthetic_recording(verbose=False, duration_s=30)
    record.raw.resample(sfreq, verbose="ERROR")
    return record


@pytest.mark.parametrize("sfreq", [500.0, 512.0])
def test_too_slow_for_ripples_skips_the_hfo_detectors_and_says_why(sfreq, tmp_path):
    result = run_pipeline(_at(sfreq), verbose=False, save_to=tmp_path)
    assert result.hfo_skipped.startswith("HFO detection skipped")
    assert "at least 556 Hz" in result.hfo_skipped and f"{sfreq:g} Hz" in result.hfo_skipped
    assert all(len(found) == 0 for found in result.events.values())
    assert all(result.rates[name]["rate_per_min"].sum() == 0 for name in result.rates)
    assert len(result.spikes) > 0, "the discharges are still detected"
    assert len(result.quality) == len(result.prepared.ch_names)
    assert np.isfinite(result.quality["burstiness"]).all()
    saved = " ".join(p.read_text(errors="ignore") for p in tmp_path.rglob("*")
                     if p.suffix in (".md", ".json"))
    assert "HFO detection skipped" in saved


def test_fast_enough_is_unchanged():
    result = run_pipeline(_at(1000.0), verbose=False)
    assert result.hfo_skipped == ""
    assert sum(len(found) for found in result.events.values()) > 0


def test_the_import_dialog_does_not_offer_what_it_cannot_analyse(tmp_path):
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets.QApplication.instance() or widgets.QApplication([])
    from qtpy.QtWidgets import QDialogButtonBox

    from onset_review.importer import ImportDialog

    def saved(sfreq: float):
        info = mne.create_info(["A1", "A2"], sfreq, "seeg")
        raw = mne.io.RawArray(np.random.default_rng(0).normal(size=(2, int(20 * sfreq))) * 1e-5,
                              info, verbose="ERROR")
        path = tmp_path / f"r{int(sfreq)}_ieeg.fif"
        raw.save(path, verbose="ERROR")
        return path

    slow = ImportDialog(saved(500.0))
    button = slow.buttons.button(QDialogButtonBox.Open)
    assert button.isEnabled(), "a slow recording still has discharges to read"
    text = " ".join(label.text() for label in slow.findChildren(widgets.QLabel))
    assert "Too slow for HFOs (ripples need at least 556 Hz)" in text
    fast = ImportDialog(saved(1000.0))
    assert fast.buttons.button(QDialogButtonBox.Open).isEnabled()


def _session(sfreq: float, band: str = "ripple"):
    from onset_review.session import ReviewRequest, session_from_recording

    record = _at(sfreq)
    return session_from_recording(record, ReviewRequest(
        t_start=0.0, t_stop=float(record.duration), band=band))


def test_a_window_too_slow_for_hfos_opens_with_its_discharges_and_says_why():
    session = _session(500.0)
    assert session.notes[0].startswith("HFO detection skipped")
    assert "at least 556 Hz" in session.notes[0]
    assert {event.detector for event in session.events} == {"spike"}
    assert len(session.findings) == len(session.raw.ch_names)
    # Ranked by the discharges, not by an all-zero HFO count, and said so.
    assert session.notes[1].startswith("Channels are ranked by interictal discharges")
    assert session.findings["n_events"].sum() == len(session.events) > 0
    assert session.findings["n_events"].iloc[0] == session.findings["n_events"].max()


def test_a_band_the_rate_cannot_carry_is_still_refused_when_another_fits():
    with pytest.raises(ValueError, match="Choose the ripple band"):
        _session(1000.0, band="fast_ripple")


@pytest.mark.parametrize("sfreq, offered", [(2000.0, ["ripple", "fast_ripple"]),
                                            (1000.0, ["ripple"]), (500.0, [])])
def test_the_signal_page_offers_the_bands_the_rate_carries(sfreq, offered):
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets.QApplication.instance() or widgets.QApplication([])
    from onset_review.preprocessing import PreprocessPanel

    panel = PreprocessPanel(_session(sfreq))
    model = panel.band.model()
    enabled = [panel.band.itemData(i) for i in range(panel.band.count())
               if model.item(i).isEnabled() and panel.band.isEnabled()]
    assert enabled == offered
    assert not panel.apply.isEnabled(), "nothing changed yet"
    emitted = []
    panel.applied.connect(emitted.append)
    if "fast_ripple" in offered:
        panel.band.setCurrentIndex(panel.band.findData("fast_ripple"))
        assert panel.apply.isEnabled() and panel.band_name() == "fast_ripple"
    else:
        # Every other setting still applies on a recording no band fits.
        panel.notch_width.setValue(3.0)
        assert "cannot carry" not in panel.warnings.text()
    panel._apply()
    assert len(emitted) == 1
