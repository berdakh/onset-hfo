"""The Epileptogenicity Index, on seizures where the answer is known.

A synthetic seizure: background of slow rhythm and noise on every contact;
fast activity (40 Hz) starts on two contacts 1 s after the marked onset,
spreads to two more 6 s later and weaker, and never reaches the rest. What
has to hold: the energy ratio rises where the fast activity is; the change
test fires on the four contacts and only them, at the right times; the index
ranks the early pair first, the late pair next, the rest at zero; across
seizures, a contact that leads in every one outranks one that led once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.ictal import (
    EI_CUTOFF,
    IctalSettings,
    combine,
    energy_ratio,
    epileptogenicity,
    page_hinkley,
)

SF = 512.0
NAMES = [f"C{i}" for i in range(1, 9)]


def _seizure(onset=40.0, early=(0, 1), late=(2, 3), seed=0, length=80.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(length * SF)) / SF
    data = rng.normal(0, 1.0, (len(NAMES), len(t)))
    data += 4.0 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 6, (len(NAMES), 1)))
    for c in early:
        on = t >= onset + 1.0
        data[c, on] += 6.0 * np.sin(2 * np.pi * 40.0 * t[on])
    for c in late:
        on = t >= onset + 7.0
        data[c, on] += 3.0 * np.sin(2 * np.pi * 40.0 * t[on])
    return data


def test_the_energy_ratio_rises_where_the_fast_activity_is():
    data = _seizure()
    centres, ratios = energy_ratio(data, SF)
    assert ratios.shape == (8, len(centres)) and centres[0] == pytest.approx(0.5)
    before, after = centres < 35, centres > 50
    assert ratios[0, after].mean() > 20 * ratios[0, before].mean()
    assert ratios[7, after].mean() == pytest.approx(ratios[7, before].mean(), rel=0.5)
    with pytest.raises(ValueError):
        energy_ratio(data[:, :100], SF)


def test_page_hinkley_dates_a_step_and_ignores_noise():
    rng = np.random.default_rng(1)
    flat = 1 + rng.normal(0, 0.1, 200)
    assert page_hinkley(flat, 0.5, 15.0) == (None, None)
    step = flat.copy()
    step[120:] += 5.0
    alarm, change = page_hinkley(step, 0.5, 15.0)
    assert 118 <= change <= 121 and change < alarm <= 125


def test_the_index_ranks_early_then_late_then_none():
    table = epileptogenicity(_seizure(), NAMES, SF, onset_s=40.0)
    by = table.set_index("channel")
    assert set(table.loc[table["detected"], "channel"]) == {"C1", "C2", "C3", "C4"}
    assert set(table["channel"].iloc[:2]) == {"C1", "C2"}
    assert set(table["channel"].iloc[2:4]) == {"C3", "C4"}
    assert by.loc["C1", "ei"] == pytest.approx(1.0) or by.loc["C2", "ei"] == pytest.approx(1.0)
    assert (by.loc[["C5", "C6", "C7", "C8"], "ei"] == 0).all()
    assert by.loc["C1", "change_s"] == pytest.approx(1.0, abs=1.0)
    assert by.loc["C3", "change_s"] == pytest.approx(7.0, abs=1.0)
    assert by.loc["C3", "ei"] < by.loc["C1", "ei"]


def test_a_start_offset_puts_times_in_the_recording_s_base():
    data = _seizure(onset=40.0)
    shifted = epileptogenicity(data, NAMES, SF, onset_s=1040.0, start_s=1000.0)
    plain = epileptogenicity(data, NAMES, SF, onset_s=40.0)
    assert np.allclose(shifted.set_index("channel")["ei"], plain.set_index("channel")["ei"])
    with pytest.raises(ValueError, match="baseline"):
        epileptogenicity(data, NAMES, SF, onset_s=10.0)


def test_across_seizures_the_consistent_leader_wins():
    first = epileptogenicity(_seizure(early=(0, 1), late=(2, 3), seed=1), NAMES, SF, 40.0)
    second = epileptogenicity(_seizure(early=(0, 4), late=(2,), seed=2), NAMES, SF, 40.0)
    third = epileptogenicity(_seizure(early=(0, 5), late=(), seed=3), NAMES, SF, 40.0)
    table = combine({"sz1": first, "sz2": second, "sz3": third})
    top = table.iloc[0]
    assert top["channel"] == "C1" and top["seizures_high"] == 3 and top["seizures"] == 3
    by = table.set_index("channel")
    assert by.loc["C2", "seizures_high"] == 1 and by.loc["C8", "median_ei"] == 0
    assert by.loc["C1", "median_change_s"] == pytest.approx(1.0, abs=1.0)
    assert EI_CUTOFF == 0.3 and "Page–Hinkley" in IctalSettings().describe()


def test_the_study_s_auc_and_interval():
    from onset_hfo.ictal_study import auc, bootstrap_median

    assert auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert auc([0, 1, 2, 3], [True, True, False, False]) == 0.0
    assert auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert np.isnan(auc([1, 2], [False, False]))
    median, low, high = bootstrap_median([0.6, 0.7, 0.8, 0.9, 1.0])
    assert median == 0.8 and 0.6 <= low <= median <= high <= 1.0


def test_patients_pool_their_seizures_by_median():
    from onset_hfo.ictal_study import per_patient

    channels = pd.DataFrame([
        {"subject": "p1", "run": "01", "channel": "A1-A2", "ei": 1.0, "soz": True,
         "er_after": 5.0},
        {"subject": "p1", "run": "01", "channel": "B1-B2", "ei": 0.2, "soz": False,
         "er_after": 1.0},
        {"subject": "p1", "run": "02", "channel": "A1-A2", "ei": 0.4, "soz": True,
         "er_after": 3.0},
        {"subject": "p1", "run": "02", "channel": "B1-B2", "ei": 0.6, "soz": False,
         "er_after": 2.0}])
    seizures = pd.DataFrame([{"subject": "p1", "run": r, "status": "ok", "site": "X",
                              "seizure_free": True, "engel": 1} for r in ("01", "02")])
    row = per_patient(channels, seizures).iloc[0]
    assert row["n_seizures"] == 2 and row["auc_ei"] == 1.0 and row["top_in_zone"]


def test_a_control_shift_must_clear_the_seizure(tmp_path):
    from onset_hfo.ictal_study import WINDOW_AFTER_S, run_study

    with pytest.raises(ValueError, match="reaches the seizure"):
        run_study(tmp_path, subjects=[], shift_s=WINDOW_AFTER_S - 1)
