"""The hand-annotated benchmark's harness: sampling, blinding, agreement.

Roadmap item 5. Marking a few hundred windows is two reviewers' afternoons
and cannot be faked here. Everything *around* the marking can be built, and
most of what makes such a benchmark worthless lives in that surrounding half:
a sample drawn from the detector's own hits can never measure recall, a
manifest that leaks the detector's verdict makes the reference circular, and
raw percent agreement on a rare class looks excellent no matter what the
reviewers did.

These tests are the guards against each of those three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_hfo.detectors.base import Event
from onset_hfo.review import (
    LABELS,
    MANIFEST_COLUMNS,
    AnnotationLog,
    agreement,
    ceiling_note,
    read_manifest,
    sample_windows,
    score_against_annotations,
    write_manifest,
)

CHANNELS = [f"A{i}-A{i + 1}" for i in range(1, 9)]


def _events(n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        start = float(rng.uniform(1.0, 55.0))
        out.append(Event(channel=CHANNELS[i % len(CHANNELS)], start=start,
                         stop=start + 0.03, detector="rms", band=(80.0, 250.0),
                         accepted=bool(i % 3)))
    return {"rms": out}


@pytest.fixture
def sample():
    return sample_windows(_events(), duration_s=60.0, channels=CHANNELS,
                          n_per_stratum=20, seed=1)


# --------------------------------------------------------------------------
# 1. The sample must be able to find a false negative
# --------------------------------------------------------------------------


def test_the_sample_includes_windows_the_detector_never_proposed(sample):
    """Without this stratum recall is unmeasurable, whoever does the marking."""
    strata = pd.Series([w.stratum for w in sample]).value_counts()
    assert strata.get("background", 0) > 0, \
        "a sample drawn only from detections can measure precision and nothing else"
    assert set(strata.index) == {"accepted", "rejected", "background"}


def test_the_three_strata_are_drawn_in_equal_numbers_not_by_prevalence(sample):
    counts = pd.Series([w.stratum for w in sample]).value_counts()
    assert counts.nunique() == 1, "equal by design; the imbalance is corrected in scoring"


def test_a_background_window_really_is_empty():
    """"Background" that overlapped a candidate would corrupt the recall estimate."""
    events = _events(seed=3)
    windows = sample_windows(events, duration_s=60.0, channels=CHANNELS,
                             n_per_stratum=20, seed=3)
    by_channel: dict[str, list] = {}
    for event in events["rms"]:
        by_channel.setdefault(event.channel, []).append(event)
    for window in windows:
        if window.stratum != "background":
            continue
        mid = 0.5 * (window.t_start + window.t_stop)
        for event in by_channel.get(window.channel, []):
            assert abs(mid - 0.5 * (event.start + event.stop)) >= 0.4, \
                "a background window sits on top of a candidate"


def test_a_recording_with_no_rejections_still_yields_the_other_two_strata():
    events = {"rms": [Event(channel=CHANNELS[0], start=float(i), stop=float(i) + 0.03,
                            detector="rms", band=(80.0, 250.0), accepted=True)
                      for i in range(1, 40)]}
    windows = sample_windows(events, duration_s=60.0, channels=CHANNELS,
                             n_per_stratum=10, seed=0)
    strata = {w.stratum for w in windows}
    assert "accepted" in strata and "background" in strata
    assert "rejected" not in strata, "an empty stratum must be absent, not fabricated"


# --------------------------------------------------------------------------
# 2. The reviewer must not be able to see the answer
# --------------------------------------------------------------------------


def test_the_manifest_carries_nothing_that_unblinds_the_reviewer(sample, tmp_path):
    path = write_manifest(sample, tmp_path / "manifest.csv")
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(MANIFEST_COLUMNS)
    for leak in ("stratum", "detector", "accepted", "score", "source_detector"):
        assert leak not in frame.columns


def test_reading_a_manifest_that_leaked_is_refused(sample, tmp_path):
    path = tmp_path / "leaky.csv"
    frame = pd.DataFrame([{**{c: getattr(w, c) for c in MANIFEST_COLUMNS},
                           "stratum": w.stratum} for w in sample])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unblind"):
        read_manifest(path)


def test_the_window_numbering_does_not_track_the_stratum(sample):
    """Ids 1-20 all being detections would unblind by filing alone."""
    strata = [w.stratum for w in sample]
    changes = sum(1 for a, b in zip(strata, strata[1:], strict=False) if a != b)
    assert changes > len(set(strata)), "the sample is ordered by stratum"


def test_the_key_is_written_beside_the_manifest_under_its_own_name(sample, tmp_path):
    """Two files, so handing over the wrong one is a visible mistake."""
    path = write_manifest(sample, tmp_path / "manifest.csv")
    key = path.with_suffix(".key.csv")
    assert key.exists()
    assert set(pd.read_csv(key).columns) == {
        "window_id", "stratum", "source_detector", "source_accepted"}


# --------------------------------------------------------------------------
# 3. The log has to survive a closed laptop
# --------------------------------------------------------------------------


def test_annotation_is_append_only_and_resumes_where_it_stopped(sample, tmp_path):
    manifest = pd.read_csv(write_manifest(sample, tmp_path / "manifest.csv"))
    log = AnnotationLog(tmp_path / "ann.csv", reviewer="A")
    assert len(log.remaining(manifest)) == len(manifest)
    for wid in list(manifest["window_id"])[:5]:
        log.record(wid, "hfo")

    resumed = AnnotationLog(tmp_path / "ann.csv", reviewer="A")
    assert resumed.done == 5
    assert len(resumed.remaining(manifest)) == len(manifest) - 5
    assert list(manifest["window_id"])[:5] not in resumed.remaining(manifest)


def test_two_reviewers_share_a_file_without_seeing_each_other(sample, tmp_path):
    manifest = pd.read_csv(write_manifest(sample, tmp_path / "manifest.csv"))
    a = AnnotationLog(tmp_path / "ann.csv", reviewer="A")
    for wid in list(manifest["window_id"])[:4]:
        a.record(wid, "hfo")
    b = AnnotationLog(tmp_path / "ann.csv", reviewer="B")
    assert b.done == 0, "B must not inherit A's marks"
    assert len(b.remaining(manifest)) == len(manifest)


def test_an_invalid_label_is_refused_at_the_point_of_writing(tmp_path):
    log = AnnotationLog(tmp_path / "ann.csv", reviewer="A")
    with pytest.raises(ValueError, match="label must be"):
        log.record("w0001", "probably?")
    assert "unsure" in LABELS, "forcing a binary choice manufactures agreement"


# --------------------------------------------------------------------------
# 4. Agreement must be chance-corrected, and say so
# --------------------------------------------------------------------------


def _annotations(pairs):
    rows = []
    for i, (a, b) in enumerate(pairs):
        rows.append({"window_id": f"w{i:04d}", "reviewer": "A", "label": a, "note": ""})
        rows.append({"window_id": f"w{i:04d}", "reviewer": "B", "label": b, "note": ""})
    return pd.DataFrame(rows)


def test_perfect_agreement_on_a_mixed_sample_is_kappa_one():
    pairs = [("hfo", "hfo")] * 20 + [("not_hfo", "not_hfo")] * 20
    assert agreement(_annotations(pairs), n_boot=50)["kappa"] == pytest.approx(1.0)


def test_two_reviewers_who_both_say_no_to_everything_have_agreed_about_nothing():
    """The failure raw percent agreement hides, and the reason for kappa."""
    stats = agreement(_annotations([("not_hfo", "not_hfo")] * 40), n_boot=50)
    assert stats["raw_agreement"] == 1.0
    assert np.isnan(stats["kappa"]), "one label between them defines no agreement"
    assert "no ceiling at all" in ceiling_note(stats)


def test_a_rare_positive_class_inflates_raw_agreement_but_not_kappa():
    """36 easy negatives and a coin-flip on the rest."""
    pairs = [("not_hfo", "not_hfo")] * 36 + [("hfo", "not_hfo"), ("not_hfo", "hfo"),
                                             ("hfo", "hfo"), ("not_hfo", "hfo")]
    stats = agreement(_annotations(pairs), n_boot=200)
    assert stats["raw_agreement"] > 0.9
    assert stats["kappa"] < 0.5, "kappa must not follow raw agreement upwards"


def test_unsure_windows_are_dropped_and_counted_rather_than_guessed():
    pairs = [("hfo", "hfo")] * 10 + [("unsure", "hfo")] * 5 + [("not_hfo", "not_hfo")] * 10
    stats = agreement(_annotations(pairs), n_boot=50)
    assert stats["n_unsure_dropped"] == 5
    assert stats["n_scored"] == 20 and stats["n_common"] == 25

    kept = agreement(_annotations(pairs), drop_unsure=False, n_boot=50)
    assert kept["n_scored"] == 25


def test_agreement_needs_exactly_two_reviewers():
    rows = _annotations([("hfo", "hfo")] * 5)
    rows = pd.concat([rows, pd.DataFrame([{"window_id": "w0000", "reviewer": "C",
                                           "label": "hfo", "note": ""}])])
    with pytest.raises(ValueError, match="exactly two"):
        agreement(rows)


def test_the_ceiling_sentence_names_whichever_label_is_the_common_one():
    """Crediting the rare class with inflating raw agreement was the first bug here."""
    mostly_negative = ceiling_note({"kappa": 0.7, "kappa_lo": 0.6, "kappa_hi": 0.8,
                                    "n_scored": 400, "raw_agreement": 0.94,
                                    "prevalence_hfo": 0.08})
    assert "92% prevalence of 'not an HFO'" in mostly_negative

    mostly_positive = ceiling_note({"kappa": 0.34, "kappa_lo": 0.19, "kappa_hi": 0.48,
                                    "n_scored": 180, "raw_agreement": 0.68,
                                    "prevalence_hfo": 0.61})
    assert "61% prevalence of 'an HFO'" in mostly_positive


def test_the_ceiling_sentence_refuses_to_be_omitted_quietly():
    note = ceiling_note({"kappa": 0.55, "kappa_lo": 0.4, "kappa_hi": 0.7,
                         "n_scored": 300, "raw_agreement": 0.8, "prevalence_hfo": 0.3})
    assert "No detector can be scored against this reference more" in note
    assert "moderate" in note


# --------------------------------------------------------------------------
# 5. Scoring
# --------------------------------------------------------------------------


def test_scores_are_reported_per_stratum_and_never_pooled(sample, tmp_path):
    """Pooling equal-sized strata would describe a recording that does not exist."""
    path = write_manifest(sample, tmp_path / "manifest.csv")
    manifest = pd.read_csv(path)
    key = pd.read_csv(path.with_suffix(".key.csv"))
    marks = pd.DataFrame([
        {"window_id": w, "reviewer": r, "label": "hfo", "note": ""}
        for w in manifest["window_id"] for r in ("A", "B")])

    scores = score_against_annotations(_events()["rms"], manifest, marks, key)
    assert set(scores["strata"]) <= {"accepted", "rejected", "background"}
    assert "precision" not in scores, "a pooled headline is the number not to publish"
    for stratum in scores["strata"].values():
        assert stratum["n"] > 0


def test_the_two_consensus_rules_bracket_the_answer(sample, tmp_path):
    """Quoting whichever is kinder is how a detector's precision gets published."""
    path = write_manifest(sample, tmp_path / "manifest.csv")
    manifest = pd.read_csv(path)
    key = pd.read_csv(path.with_suffix(".key.csv"))
    ids = list(manifest["window_id"])
    marks = pd.DataFrame(
        [{"window_id": w, "reviewer": "A", "label": "hfo", "note": ""} for w in ids] +
        [{"window_id": w, "reviewer": "B", "label": "not_hfo", "note": ""} for w in ids])

    both = score_against_annotations(_events()["rms"], manifest, marks, key, consensus="both")
    either = score_against_annotations(_events()["rms"], manifest, marks, key, consensus="either")
    n_both = sum(s["true_positive"] + s["false_negative"] for s in both["strata"].values())
    n_either = sum(s["true_positive"] + s["false_negative"] for s in either["strata"].values())
    assert n_both == 0 and n_either > 0, "the rules must genuinely bracket"


def test_an_unknown_consensus_rule_is_refused(sample, tmp_path):
    path = write_manifest(sample, tmp_path / "manifest.csv")
    manifest = pd.read_csv(path)
    key = pd.read_csv(path.with_suffix(".key.csv"))
    marks = pd.DataFrame([{"window_id": manifest["window_id"].iloc[0], "reviewer": "A",
                           "label": "hfo", "note": ""}])
    with pytest.raises(ValueError, match="consensus must be"):
        score_against_annotations([], manifest, marks, key, consensus="majority")


# --------------------------------------------------------------------------
# 6. The window must point at signal that exists
# --------------------------------------------------------------------------


def test_background_windows_come_from_the_same_recording_as_the_candidates():
    """The bug that would have cost a reviewer an afternoon.

    The shipped example is a slice from 50-110 s. With `t_offset` left at its
    default the background stratum was drawn from 0-60 s, so a third of the
    sample pointed at signal the analysis does not contain -- which shows up
    as a blank figure, not an error.
    """
    events = {"rms": [Event(channel=CHANNELS[0], start=50.0 + i, stop=50.0 + i + 0.03,
                            detector="rms", band=(80.0, 250.0), accepted=True)
                      for i in range(50)]}
    windows = sample_windows(events, duration_s=60.0, channels=CHANNELS,
                             n_per_stratum=20, seed=0, t_offset=50.0)
    # A window centred on an event at the very first sample reaches half a
    # window before it; `plot_event` clamps to the array, so that overhang is
    # context the reviewer does not get rather than a window that is wrong.
    # The bug this guards against was a 50-second error, not a 0.2-second one.
    for window in windows:
        assert 49.8 <= window.t_start < 110.0, \
            f"{window.window_id} at {window.t_start:.1f}s is outside the recording"
    assert min(w.t_start for w in windows) > 49.0, "a whole stratum is misplaced"


def test_a_time_range_that_does_not_contain_the_candidates_is_refused():
    """Silence here is the expensive failure, so it raises."""
    events = {"rms": [Event(channel=CHANNELS[0], start=50.0 + i, stop=50.0 + i + 0.03,
                            detector="rms", band=(80.0, 250.0), accepted=True)
                      for i in range(20)]}
    with pytest.raises(ValueError, match="same recording"):
        sample_windows(events, duration_s=60.0, channels=CHANNELS,
                       n_per_stratum=5, seed=0)          # t_offset left at 0


def test_the_cli_passes_the_analysis_offset_through(tmp_path):
    """End to end on the shipped 50-110 s example."""
    from onset_hfo.cli import build_parser

    args = build_parser().parse_args(
        ["review", "sample", "--analysis", "data/example_analysis",
         "--out", str(tmp_path), "--per-stratum", "12"])
    assert args.func(args) == 0
    manifest = pd.read_csv(tmp_path / "manifest.csv")
    assert len(manifest) == 36
    assert manifest["t_start"].min() >= 50.0
    assert manifest["t_stop"].max() <= 110.0


# --------------------------------------------------------------------------
# 7. The notebooks are written by the builder, with one recorded exception
# --------------------------------------------------------------------------


def test_every_committed_notebook_is_produced_by_the_builder():
    """`scripts/build_notebooks.py` says notebooks are written there. One is not.

    `06_agent_benchmark.ipynb` was committed directly and the builder never
    learned about it, which nothing caught. This test does not pretend that is
    fixed -- it pins the exception so a second one cannot appear quietly.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    builder = (root / "scripts" / "build_notebooks.py").read_text()
    committed = {p.name for p in (root / "notebooks").glob("*.ipynb")}
    # Look for the constant assignment, not the bare name: the builder
    # *mentions* 06 in a comment explaining why it is absent, and a substring
    # check counted that as registration.
    unregistered = {n for n in committed if f'= "{n}"' not in builder}
    assert unregistered == {"06_agent_benchmark.ipynb"}, (
        f"notebooks not written by the builder: {sorted(unregistered)}. Add them "
        f"to scripts/build_notebooks.py, or extend this exception deliberately.")


def test_the_annotation_notebook_exists_and_carries_its_colab_badge():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks" / "07_annotation.ipynb"
    assert path.exists()
    text = path.read_text()
    assert "colab-badge" in text
    for idea in ("kappa", "blind", "recall"):
        assert idea in text.lower(), f"the notebook does not mention {idea}"
