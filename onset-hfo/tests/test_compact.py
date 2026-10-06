"""The compact views of the tables: what a reader sees first (Qt-free)."""

from __future__ import annotations

import pandas as pd

from onset_review import compact


def test_the_rate_cell_puts_the_interval_beside_the_rate_in_whole_numbers():
    assert compact.rate_cell(52.0, 38.79, 67.13) == "52 (39–67)"
    assert compact.rate_cell(0.0, 0.0, 4.15) == "0 (0–4)"
    assert compact.rate_cell(12.4) == "12"
    assert compact.rate_cell(float("nan")) == "—"
    assert compact.rate_cell(3.0, float("nan"), 5.0) == "3", "no half an interval"


def test_compact_findings_keeps_the_rank_and_says_who_was_reviewed():
    frame = pd.DataFrame([
        {"rank": 1, "channel": "AR1-AR2", "rate_per_min": 52.0, "rate_ci_low": 38.79,
         "rate_ci_high": 67.13, "reviewed": True, "expert_n": 62},
        {"rank": 2, "channel": "AHR6-AHR7", "rate_per_min": 17.0, "rate_ci_low": 9.8,
         "rate_ci_high": 26.1, "reviewed": False, "expert_n": 0},
    ])
    out = compact.compact_findings(frame)
    assert list(out["rate"]) == ["52 (39–67)", "17 (10–26)"]
    assert list(out["annotators"]) == ["62 marked", "not reviewed"]
    assert [c for c in compact.COMPACT_COLUMNS["findings"] if c in out.columns] \
        == ["rank", "channel", "rate", "annotators"]
    assert compact.compact_findings(frame.iloc[0:0]).empty


def test_the_compact_column_sets_are_subsets_of_the_full_ones():
    for table, columns in compact.COMPACT_COLUMNS.items():
        full = set(compact.FULL_COLUMNS[table]) | {"rate", "annotators"}
        assert set(columns) <= full, table
        assert len(columns) <= 7 < len(compact.FULL_COLUMNS[table])


def test_agreement_bars_are_the_two_counts_a_reader_compares():
    frame = pd.DataFrame([
        {"channel": "HL3-HL4", "n_expert": 107, "n_detector": 3, "matched": 2},
        {"channel": "AR1-AR2", "n_expert": 62, "n_detector": 52, "matched": 40},
        {"channel": "AL3-AL4", "n_expert": 0, "n_detector": 0, "matched": 0},
    ])
    bars = compact.agreement_bars(frame, top=2)
    assert [b["channel"] for b in bars] == ["HL3-HL4", "AR1-AR2"]
    assert bars[1] == {"channel": "AR1-AR2", "expert": 62, "detector": 52, "matched": 40}
    assert compact.agreement_bars(pd.DataFrame()) == []


def test_quality_chips_put_the_contacts_worth_a_look_first():
    quality = pd.DataFrame([
        {"channel": "AR1-AR2", "good": True, "flagged": False, "reason": ""},
        {"channel": "TR1-TR2", "good": True, "flagged": True, "reason": "hf_outlier"},
        {"channel": "DC1-DC2", "good": False, "flagged": False, "reason": "dead"},
        {"channel": "EK1-EK2", "good": False, "flagged": False, "reason": "clipped"},
    ])
    chips = compact.quality_chips(quality, kept=("EK1-EK2",))
    assert [c["kind"] for c in chips] == ["set_aside", "flagged", "reinstated", "analysed"]
    assert chips[0]["channel"] == "DC1-DC2" and chips[0]["reason"] == "dead"
    assert chips[2]["channel"] == "EK1-EK2"
    assert compact.counts_by_kind(chips) == {"set_aside": 1, "flagged": 1,
                                             "reinstated": 1, "analysed": 1}
    assert compact.quality_chips(None) == [] and compact.quality_chips(pd.DataFrame()) == []
