"""`scripts/run_montage_comparison.py`: the projections between units, and
the committed extract's invariants.

The bipolar arm ranks pairs and the two referential arms rank contacts, so
every comparison between them rests on three small projections. Each is
pinned here on names chosen by hand, because a projection that silently
drops a contact or double-counts a marking would make the whole table lie
without a single number looking wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from onset_hfo.config import PROJECT_ROOT
from onset_hfo.detectors.base import Event

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_montage_comparison as montage  # noqa: E402

PAIRS = ["AH1-AH2", "AH2-AH3", "PH1-PH2"]


def _event(channel, start, stop, accepted=True, score=1.5):
    return Event(channel=channel, start=start, stop=stop, detector="rms",
                 band=(80.0, 250.0), score=score, peak_amplitude_uv=1.0,
                 accepted=accepted, contacts=[channel])


# -- the three arms are built from one config each --------------------------


def test_each_arm_changes_only_the_reference():
    from dataclasses import asdict

    base = asdict(montage.montage_config("bipolar"))
    for name in ("referential", "average"):
        other = asdict(montage.montage_config(name))
        differing = {k for k in base if base[k] != other[k]}
        assert differing <= {"bipolar", "average_reference"}, (name, differing)
    assert montage.montage_config("referential").bipolar is False
    assert montage.montage_config("referential").average_reference is False
    assert montage.montage_config("average").average_reference is True
    with pytest.raises(ValueError):
        montage.montage_config("laplacian")


# -- projections between pairs and contacts ---------------------------------


def test_contacts_behind_keeps_order_and_drops_duplicates():
    assert montage.contacts_behind(PAIRS) == ["AH1", "AH2", "AH3", "PH1", "PH2"]


def test_expert_counts_project_onto_every_contact_of_a_pair():
    expert = pd.Series({"AH1-AH2": 4.0, "AH2-AH3": 1.0, "PH1-PH2": 0.0})
    on_contacts = montage.expert_on_contacts(expert, ["AH1", "AH2", "AH3", "PH1", "PH2"])
    # AH2 sits in both pairs and is credited with both markings.
    assert on_contacts.to_dict() == {"AH1": 4.0, "AH2": 5.0, "AH3": 1.0,
                                     "PH1": 0.0, "PH2": 0.0}


def test_expert_projection_ignores_a_contact_the_arm_did_not_keep():
    expert = pd.Series({"AH1-AH2": 4.0})
    out = montage.expert_on_contacts(expert, ["AH2"])
    assert out.to_dict() == {"AH2": 4.0}


def test_contact_counts_project_onto_pairs_as_the_mean_of_both_ends():
    counts = pd.Series({"AH1": 2.0, "AH2": 6.0, "AH3": 0.0, "PH1": 1.0})
    on_pairs = montage.contacts_on_pairs(counts, PAIRS)
    assert on_pairs.to_dict() == {"AH1-AH2": 4.0, "AH2-AH3": 3.0}
    assert "PH1-PH2" not in on_pairs.index, "PH2 was not ranked, so the pair is left out"


# -- event projection for the referential arms ------------------------------


def test_events_on_either_contact_land_on_the_pair_and_overlaps_merge():
    events = [_event("AH1", 1.00, 1.05), _event("AH2", 1.03, 1.10),
              _event("AH2", 5.00, 5.04), _event("AH3", 9.00, 9.02),
              _event("PH1", 2.00, 2.02, accepted=False)]
    projected = montage.events_as_pairs(events, PAIRS)
    by_pair = {}
    for event in projected:
        by_pair.setdefault(event.channel, []).append((event.start, event.stop))
    # AH1 at 1.00-1.05 and AH2 at 1.03-1.10 overlap: one event on AH1-AH2.
    assert by_pair["AH1-AH2"] == [(1.00, 1.10), (5.00, 5.04)]
    # AH2 is shared, so both of its events are also what a reader of AH2-AH3
    # sees -- there is nothing on AH3 at 1.03 for the first one to merge with.
    assert by_pair["AH2-AH3"] == [(1.03, 1.10), (5.00, 5.04), (9.00, 9.02)]
    # Rejected events are not projected.
    assert "PH1-PH2" not in by_pair
    assert all(event.contacts == list(montage.pair_contacts(event.channel))
               for event in projected)


def test_event_projection_leaves_the_originals_alone():
    events = [_event("AH1", 1.00, 1.05)]
    montage.events_as_pairs(events, PAIRS)
    assert events[0].channel == "AH1" and events[0].contacts == ["AH1"]


def test_spearman_is_nan_where_nothing_can_be_ranked():
    flat = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0})
    ranked = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    assert pd.isna(montage.spearman(flat, ranked))
    assert montage.spearman(ranked, ranked) == pytest.approx(1.0)
    assert pd.isna(montage.spearman(ranked.iloc[:2], ranked))


# -- the committed extract ---------------------------------------------------


@pytest.fixture(scope="module")
def extract():
    path = PROJECT_ROOT / "data" / "outcome" / "montage_screen.csv"
    if not path.exists():
        pytest.skip("montage_screen.csv is not committed")
    return pd.read_csv(path)


def test_the_extract_covers_every_patient_window_and_arm(extract):
    assert set(extract.montage) == set(montage.MONTAGES)
    assert set(extract.band) == {"ripple", "fast_ripple"}
    assert extract.subject.nunique() == 20
    assert extract.window.nunique() == 5
    counts = extract.groupby(["band", "montage"]).size()
    assert (counts == 100).all(), counts.to_dict()


def test_the_extract_scores_the_same_physical_contacts_in_every_arm(extract):
    """The referential arms rank the contacts behind the reviewed pairs, so
    they never have fewer channels than there are pairs plus one lead."""
    wide = extract.pivot_table(index=["band", "subject", "window"],
                               columns="montage", values="n_channels")
    assert (wide["referential"] == wide["average"]).all()
    assert (wide["referential"] >= 2).all()


def test_the_extract_labels_how_each_arm_was_event_scored(extract):
    scoring = extract.groupby("montage").event_scoring.unique()
    assert list(scoring["bipolar"]) == ["direct"]
    assert list(scoring["referential"]) == ["projected"]
    assert list(scoring["average"]) == ["projected"]
    assert (extract[extract.montage == "bipolar"].rho_vs_bipolar == 1.0).all()


def test_the_group_table_matches_the_extract(extract):
    groups = PROJECT_ROOT / "data" / "outcome" / "montage_groups.csv"
    if not groups.exists():
        pytest.skip("montage_groups.csv is not committed")
    committed = pd.read_csv(groups)
    fresh = montage.analyse(extract)
    key = ["band", "montage", "kind", "metric"]
    merged = committed.merge(fresh, on=key, suffixes=("_committed", "_fresh"))
    assert len(merged) == len(committed) == len(fresh)
    for column in ("auc", "value", "n_SF", "n_rec"):
        a, b = merged[f"{column}_committed"], merged[f"{column}_fresh"]
        assert ((a - b).abs().fillna(0) < 1e-6).all(), column
