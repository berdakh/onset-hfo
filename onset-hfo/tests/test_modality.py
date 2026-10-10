"""Scalp EEG and MEG beside intracranial EEG.

What has to hold: a recording's kind is read from its channel types
(intracranial first, then MEG gradiometers, magnetometers, then scalp EEG)
and can be named instead; scalp EEG is read in µV with the double banana as
its bipolar montage (old 10-20 names accepted, a missing electrode removing
its links rather than bridging them); MEG is read one sensor type at a time
in fT/cm or fT and never re-referenced; on the same synthetic signal dressed
as each kind, the implanted focus leads; and intracranial recordings are
analysed exactly as before.
"""

from __future__ import annotations

import pytest

from onset_hfo.config import PipelineConfig, PreprocessConfig
from onset_hfo.modality import MODALITIES, detect, double_banana_pairs, resolve
from onset_hfo.pipeline import run_pipeline
from onset_hfo.preprocess import prepare
from onset_hfo.synthetic import as_modality, make_synthetic_recording


@pytest.fixture(scope="module")
def base():
    return make_synthetic_recording(verbose=False, duration_s=60)


def test_the_kind_is_read_from_the_channel_types():
    assert detect(["seeg", "ecog", "eeg", "ecg"]) == "ieeg"
    assert detect(["eeg", "grad", "mag", "eog"]) == "meg_grad"
    assert detect(["mag", "eeg"]) == "meg_mag"
    assert detect(["eeg", "eog", "ecg"]) == "eeg"
    assert resolve("meg", ["mag", "grad"]).key == "meg_grad"
    assert resolve("meg", ["mag"]).key == "meg_mag"
    assert resolve("auto", ["eeg"]).unit == "µV"
    assert resolve("eeg", ["seeg", "eeg"]).key == "eeg", "named, not detected"
    with pytest.raises(ValueError, match="modality must be"):
        resolve("fnirs", ["eeg"])
    assert {m.unit for m in MODALITIES.values()} == {"µV", "fT/cm", "fT"}


def test_the_double_banana_takes_old_names_and_does_not_bridge_a_gap():
    names = ["FP1", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz"]
    assert double_banana_pairs(names) == [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"),
                                          ("T5", "O1"), ("Fz", "Cz"), ("Cz", "Pz")]
    without_t3 = [n for n in names if n != "T3"]
    pairs = double_banana_pairs(without_t3)
    assert ("F7", "T5") not in pairs and len(pairs) == 4


def test_scalp_eeg_is_read_in_microvolts_on_the_double_banana(base):
    scalp = as_modality(base, "eeg")
    prepared = prepare(scalp, verbose=False)
    assert (prepared.modality, prepared.unit, prepared.montage) == ("eeg", "µV", "bipolar")
    assert "Fp1-F7" in prepared.ch_names and "F7-T7" in prepared.ch_names
    assert prepared.steps[0].startswith("analysed as scalp EEG, in µV")
    assert "validated only there" in prepared.steps[0]
    assert any("double banana" in s for s in prepared.steps)
    shaft = prepare(scalp, PreprocessConfig(reference="shaft"), verbose=False)
    assert shaft.montage == "average"
    assert any("scalp EEG uses the common average" in s for s in shaft.steps)


@pytest.mark.parametrize("kind, unit", [("meg_grad", "fT/cm"), ("meg_mag", "fT")])
def test_meg_is_read_in_its_own_unit_and_never_re_referenced(base, kind, unit):
    meg = as_modality(base, kind)
    prepared = prepare(meg, verbose=False)
    assert (prepared.modality, prepared.unit, prepared.montage) == (kind, unit, "monopolar")
    assert any("not re-referenced" in s for s in prepared.steps)
    assert 100 < float(prepared.data.std()) < 2000, "tens to hundreds of fT, not tesla"


@pytest.mark.parametrize("kind", ["eeg", "meg_grad", "meg_mag"])
def test_the_implanted_focus_leads_whatever_the_recording_is_dressed_as(base, kind):
    result = run_pipeline(as_modality(base, kind), detectors=("rms",), verbose=False)
    hot = {as_modality(base, kind).raw.ch_names[base.raw.ch_names.index(c)]
           for c in base.marked_contacts}
    leader = result.rates["rms"].iloc[0]["channel"]
    assert set(leader.split("-")) & hot, (kind, leader, hot)
    assert not result.quality.empty and result.quality["good"].all()


def test_intracranial_recordings_are_analysed_as_before(base):
    prepared = prepare(base, verbose=False)
    assert (prepared.modality, prepared.unit, prepared.montage) == ("ieeg", "µV", "bipolar")
    assert prepared.steps[0] == ("kept 18 intracranial channels; dropped 0 non-brain "
                                 "channels (DC/trigger/ECG/misc)")
    assert not any(s.startswith("analysed as") for s in prepared.steps)


def test_a_kind_the_recording_does_not_have_is_refused(base):
    meg = as_modality(base, "meg_grad")
    with pytest.raises(ValueError, match="No scalp EEG channels"):
        prepare(meg, PreprocessConfig(modality="eeg"), verbose=False)
    config = PipelineConfig()
    assert config.preprocess.modality == "auto"
