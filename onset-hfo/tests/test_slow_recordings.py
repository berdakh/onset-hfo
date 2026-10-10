"""A recording sampled too slowly for ripples: said, not crashed.

What has to hold: at 500 and 512 Hz (common in archives, HUP among them) the
pipeline skips the HFO detectors and says why, in the result and in the
saved report, while the interictal discharges and the quality checks still
run; at 1000 Hz nothing changes; and the import dialog does not offer to
open a recording no HFO band fits, saying why.
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
    assert not slow.band.isEnabled() and not button.isEnabled()
    assert "too slowly for ripples" in button.toolTip()
    fast = ImportDialog(saved(1000.0))
    assert fast.buttons.button(QDialogButtonBox.Open).isEnabled()
