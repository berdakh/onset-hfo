"""The two opt-in detectors, and the two ensemble metrics built on them.

Roadmap item 4. The point of a third and fourth detector is not a better
detector -- it is to find out whether the disagreement between the first two
is *structural* or an accident of that pair, and the agreement matrix is what
answers that. These tests pin the machinery; the measured answers live in
``docs/EVALUATION.md`` §1b.
"""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.config import DetectorConfig, PipelineConfig
from onset_hfo.detectors import DETECTORS, detect_hilbert, detect_short_time_energy
from onset_hfo.detectors.base import (
    robust_scale,
    sliding_energy,
    sliding_hilbert_envelope,
    sliding_rms,
)
from onset_hfo.metrics import agreement_matrix, consensus_ranking
from onset_hfo.pipeline import HFO_DETECTORS, run_pipeline

HFO_NAMES = ("rms", "line_length", "hilbert", "short_time_energy")


# --------------------------------------------------------------------------
# The features
# --------------------------------------------------------------------------


def test_both_new_features_keep_the_signal_length():
    x = np.random.default_rng(0).normal(size=512)
    for feature in (sliding_hilbert_envelope, sliding_energy):
        assert feature(x, 7).shape == x.shape


def test_the_envelope_follows_a_burst_and_is_flat_elsewhere():
    fs, n = 2000.0, 4000
    t = np.arange(n) / fs
    x = np.zeros(n)
    burst = slice(1800, 2200)
    x[burst] = np.sin(2 * np.pi * 150 * t[burst])
    envelope = sliding_hilbert_envelope(x, 6)
    assert envelope[burst].mean() > 20 * envelope[:1000].mean()


def test_the_envelope_is_non_negative():
    """It is a magnitude; a negative sample would mean the transform is wrong."""
    x = np.random.default_rng(1).normal(size=1024)
    assert (sliding_hilbert_envelope(x, 5) >= 0).all()


def test_energy_is_the_square_of_rms_times_the_window():
    """The identity that makes the two features monotone transforms."""
    x = np.random.default_rng(2).normal(size=600)
    window = 8
    assert np.allclose(sliding_energy(x, window),
                       sliding_rms(x, window) ** 2 * window)


def test_the_robust_threshold_is_not_portable_between_those_two_features():
    """The measured reason short-time energy is not a duplicate of RMS.

    The features are monotone-related, so a *fixed* threshold would select
    identical samples. ``median + k * robustSD`` does not, because squaring is
    not affine -- and the direction matters: the same k is more permissive on
    the squared feature.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    x[::500] *= 8  # a few bursts, as a real channel has

    quantiles = []
    for feature in (sliding_rms, sliding_energy):
        trace = feature(x, 6)
        centre, scale = robust_scale(trace)
        threshold = float(np.squeeze(centre)) + 5.0 * float(np.squeeze(scale))
        quantiles.append(float((trace < threshold).mean()))

    rms_q, energy_q = quantiles
    assert energy_q < rms_q, "energy at 5 SD should sit lower in its own distribution"


# --------------------------------------------------------------------------
# The detectors
# --------------------------------------------------------------------------


def test_both_new_detectors_are_registered_and_addressable():
    for name in ("hilbert", "short_time_energy"):
        assert name in DETECTORS
        assert name in HFO_DETECTORS


def test_the_default_pipeline_still_runs_exactly_two_detectors(recording):
    """Adding detectors must not silently change every published number."""
    result = run_pipeline(recording, verbose=False)
    assert set(result.events) == {"rms", "line_length"}


def test_the_new_detectors_are_opt_in_by_name(recording):
    result = run_pipeline(recording, detectors=HFO_NAMES, verbose=False)
    assert set(result.events) == set(HFO_NAMES)
    for name in HFO_NAMES:
        assert result.events[name], f"{name} found nothing at all"


@pytest.mark.parametrize("detector", [detect_hilbert, detect_short_time_energy])
def test_new_detectors_find_events_that_are_oscillations(prepared, detector):
    events = detector(prepared, DetectorConfig())
    assert events
    assert all(e.n_peaks >= 6 for e in events)      # the oscillation criterion
    assert all(e.stop > e.start for e in events)


@pytest.mark.parametrize("detector", [detect_hilbert, detect_short_time_energy])
def test_new_detectors_stamp_their_own_name(prepared, detector):
    events = detector(prepared, DetectorConfig())
    assert {e.detector for e in events} == {
        "hilbert" if detector is detect_hilbert else "short_time_energy"}


@pytest.mark.parametrize("detector", [detect_hilbert, detect_short_time_energy])
def test_a_flat_channel_yields_nothing(prepared, detector):
    """Zero signal must produce zero detections, not a divide-by-zero storm."""
    import copy

    flat = copy.copy(prepared)
    flat.data = np.zeros_like(prepared.data[:2])
    flat.ch_names = prepared.ch_names[:2]
    flat.pairs = prepared.pairs[:2]
    assert detector(flat, DetectorConfig()) == []


def test_short_time_energy_still_fires_where_rms_is_silenced(prepared):
    """The portability finding, as an executable statement.

    At a threshold high enough to silence the energy detector completely, the
    squared feature is still detecting -- because ``median + k * robustSD``
    means something different on a distribution that squaring has stretched.
    Anyone tempted to reuse a threshold across features should read this as
    the counter-example.
    """
    from onset_hfo.detectors import detect_rms

    cfg = DetectorConfig(threshold_sd=50.0)
    assert detect_rms(prepared, cfg) == []
    assert detect_short_time_energy(prepared, cfg) != []


@pytest.mark.parametrize("detector", [detect_hilbert, detect_short_time_energy])
def test_new_detectors_are_monotone_in_threshold(prepared, detector):
    loose = detector(prepared, DetectorConfig(threshold_sd=3.0))
    tight = detector(prepared, DetectorConfig(threshold_sd=8.0))
    assert len(tight) <= len(loose)


def test_new_detectors_have_a_config_slot_that_round_trips():
    cfg = PipelineConfig()
    assert cfg.hilbert.threshold_sd == 5.0
    assert cfg.short_time_energy.threshold_sd == 5.0
    as_dict = cfg.as_dict()
    assert "hilbert" in as_dict and "short_time_energy" in as_dict


# --------------------------------------------------------------------------
# The agreement matrix
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def four(recording):
    """One pipeline run with all four HFO detectors."""
    result = run_pipeline(recording, detectors=HFO_NAMES, verbose=False)
    return {name: result.events[name] for name in HFO_NAMES}, result


def test_agreement_matrix_is_square_symmetric_and_unit_diagonal(four):
    events, _ = four
    matrix = agreement_matrix(events)
    assert list(matrix.index) == list(matrix.columns) == list(HFO_NAMES)
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T, equal_nan=True)


def test_agreement_with_itself_is_one():
    from onset_hfo.detectors.base import Event

    event = Event(channel="A1-A2", start=1.0, stop=1.05, detector="x", band=(80.0, 250.0))
    matrix = agreement_matrix({"a": [event], "b": [event]})
    assert matrix.loc["a", "b"] == pytest.approx(1.0)


def test_agreement_between_disjoint_detectors_is_zero():
    from onset_hfo.detectors.base import Event

    def ev(start):
        return Event(channel="A1-A2", start=start, stop=start + 0.02,
                     detector="x", band=(80.0, 250.0))

    matrix = agreement_matrix({"a": [ev(1.0)], "b": [ev(9.0)]})
    assert matrix.loc["a", "b"] == pytest.approx(0.0)


def test_agreement_of_two_empty_detectors_is_undefined_not_perfect():
    """Two detectors that found nothing have not agreed about anything."""
    matrix = agreement_matrix({"a": [], "b": []})
    assert np.isnan(matrix.loc["a", "b"])


# --------------------------------------------------------------------------
# The consensus ranking
# --------------------------------------------------------------------------


def test_consensus_counts_detectors_not_events(four):
    events, result = four
    votes = consensus_ranking(events, duration_s=result.duration_s)
    assert votes["n_detectors"].max() <= len(HFO_NAMES)
    assert (votes["n_detectors"] >= 0).all()


def test_consensus_puts_the_implanted_channels_on_top(four, recording):
    """The vote must recover what the simulator implanted."""
    events, result = four
    votes = consensus_ranking(events, duration_s=result.duration_s)
    hot = set(recording.marked_contacts)
    top = votes.head(3)["channel"]
    assert all(set(ch.split("-")) & hot for ch in top), \
        f"consensus top-3 missed the implanted contacts: {list(top)}"


def test_consensus_is_ordered_by_votes_then_mean_rank(four):
    events, result = four
    votes = consensus_ranking(events, duration_s=result.duration_s)
    keys = list(zip(-votes["n_detectors"], votes["mean_rank"], strict=True))
    assert keys == sorted(keys)


def test_a_channel_no_detector_saw_ranks_below_everything_seen(four):
    events, result = four
    votes = consensus_ranking(events, duration_s=result.duration_s)
    assert votes["n_detectors"].iloc[-1] <= votes["n_detectors"].iloc[0]


def test_consensus_of_nothing_is_an_empty_frame_with_columns():
    frame = consensus_ranking({}, duration_s=60.0)
    assert frame.empty
    assert {"channel", "n_detectors", "mean_rank"} <= set(frame.columns)


def test_consensus_keeps_the_per_detector_rates_it_voted_on(four):
    """A vote a reader cannot audit is worse than no vote."""
    events, result = four
    votes = consensus_ranking(events, duration_s=result.duration_s)
    for name in HFO_NAMES:
        assert f"rate_{name}" in votes.columns
        assert f"rank_{name}" in votes.columns


# --------------------------------------------------------------------------
# One registry, because there were four
# --------------------------------------------------------------------------


def test_every_module_sweeps_the_same_set_of_detectors():
    """The drift that kept roadmap item 4 blocked without anyone noticing.

    `benchmark.py` and `outcome.py` each kept a private two-entry copy of the
    detector registry. Two detectors were added to the package months before,
    and `--detectors hilbert` failed with a KeyError rather than sweeping — so
    "the new detectors have only been measured on synthetic data" was a fact
    about a dict literal, not about the data.
    """
    from onset_hfo.benchmark import DETECTORS as BENCH
    from onset_hfo.detectors import HFO_DETECTORS
    from onset_hfo.outcome import DETECTORS as OUTCOME
    from onset_hfo.pipeline import HFO_DETECTORS as PIPELINE

    assert set(HFO_DETECTORS) == set(HFO_NAMES)
    for name, registry in (("benchmark", BENCH), ("outcome", OUTCOME),
                           ("pipeline", PIPELINE)):
        assert set(registry) == set(HFO_DETECTORS), \
            f"{name} sweeps a different set of detectors than the package defines"


def test_the_shared_registry_holds_no_spike_detector():
    """A discharge is not an HFO, and the two are not interchangeable."""
    from onset_hfo.detectors import DETECTORS, HFO_DETECTORS

    assert "spike" not in HFO_DETECTORS
    assert "spike" in DETECTORS
    assert set(DETECTORS) == set(HFO_DETECTORS) | {"spike"}


def test_the_agents_registry_stays_pinned_to_its_frozen_contract():
    """The one copy that must NOT follow the package.

    `onset_agent/tools.py` publishes `enum: [rms, line_length, spike]` as part
    of a deliberately frozen JSON tool contract. Widening the agent's registry
    without widening the contract would let the planner call a tool the schema
    says does not exist; widening the contract is a decision, not a tidy-up.
    """
    from onset_agent.analysis import _HFO_DETECTORS
    from onset_agent.tools import tool_schemas

    assert set(_HFO_DETECTORS) == {"rms", "line_length"}
    enums = []
    for schema in tool_schemas():
        # OpenAI-style: {"type": "function", "function": {"parameters": {...}}}
        parameters = schema.get("function", schema).get("parameters", {})
        detector = parameters.get("properties", {}).get("detector", {})
        if "enum" in detector:
            enums.append(detector["enum"])
    assert enums, "no tool exposes a detector argument; this test is looking at nothing"
    for enum in enums:
        assert set(enum) == set(_HFO_DETECTORS) | {"spike"}, \
            "the agent's registry and its frozen contract have diverged"


# --------------------------------------------------------------------------
# The real-data sweep, pinned
# --------------------------------------------------------------------------


def _sweep():
    import pandas as pd

    from onset_hfo.config import PROJECT_ROOT

    return pd.read_csv(PROJECT_ROOT / "data" / "benchmark" / "four_detector_sweep.csv")


def test_the_four_detector_sweep_covers_every_arm():
    frame = _sweep()
    assert set(frame["detector"]) == set(HFO_NAMES)
    assert set(frame["band"]) == {"ripple", "fast_ripple"}
    for column in ("precision", "recall", "f1", "rank_rho"):
        assert frame[column].between(-1.0, 1.0).all()


def test_no_arms_optimum_sits_on_the_edge_of_the_swept_grid():
    """The standard §0 set for itself, applied to all sixteen arms.

    Four of them failed this on the first pass. Extending the grid moved
    short-time energy's fast-ripple optimum from 0.570 at 8.0 SD to 0.601 at
    12.0 -- so the check is not ceremony, it changed a published number.
    """
    frame = _sweep()
    for (band, detector), part in frame.groupby(["band", "detector"]):
        lo, hi = part["threshold_sd"].min(), part["threshold_sd"].max()
        for criterion in ("rank_rho", "f1"):
            best = float(part.loc[part[criterion].idxmax(), "threshold_sd"])
            assert best not in (lo, hi), (
                f"{detector}/{band} peaks at {best:g} SD by {criterion}, an "
                f"endpoint of the {lo:g}-{hi:g} grid: that is the grid running "
                f"out, not an optimum")


def test_the_two_added_detectors_did_not_improve_the_ranking():
    """The measured answer to 'was adding them worth it'. It was not.

    Pinned because it is the kind of null that quietly becomes a positive
    claim when someone re-runs with a different grid and quotes the winner.
    """
    frame = _sweep()
    best = (frame.loc[frame.groupby(["band", "detector"])["rank_rho"].idxmax()]
            .set_index(["band", "detector"])["rank_rho"])

    ripple_margin = best[("ripple", "short_time_energy")] - best[("ripple", "rms")]
    fast_margin = best[("fast_ripple", "hilbert")] - best[("fast_ripple", "rms")]
    assert 0 <= ripple_margin < 0.01, f"ripple margin over RMS is now {ripple_margin:.3f}"
    assert 0 <= fast_margin < 0.01, f"fast-ripple margin over RMS is now {fast_margin:.3f}"


def test_short_time_energy_needs_its_own_threshold_on_real_data():
    """A threshold in robust SDs is not portable between features, measured."""
    frame = _sweep()
    best = (frame.loc[frame.groupby(["band", "detector"])["rank_rho"].idxmax()]
            .set_index(["band", "detector"])["threshold_sd"])

    assert best[("ripple", "short_time_energy")] == 2 * best[("ripple", "rms")]
    assert best[("fast_ripple", "short_time_energy")] > 2 * best[("fast_ripple", "rms")]
    # The envelope, by contrast, wants what RMS wants.
    assert abs(best[("fast_ripple", "hilbert")] - best[("fast_ripple", "rms")]) < 0.01


def test_the_shared_cells_reproduce_the_original_networked_sweep():
    """The offline run against cached slices must match the downloaded one.

    27 cells overlap with `agreement_sweep.csv`, which was produced by a run
    that fetched from OpenNeuro. They agree exactly; that is what licenses
    regenerating this table without the network.
    """
    import pandas as pd

    from onset_hfo.config import PROJECT_ROOT

    old = pd.read_csv(PROJECT_ROOT / "data" / "benchmark" / "agreement_sweep.csv")
    both = old.merge(_sweep(), on=["band", "detector", "threshold_sd"],
                     suffixes=("_old", "_new"))
    assert len(both) >= 27, f"only {len(both)} overlapping cells to check"
    for column in ("precision", "recall", "f1", "detections", "rank_rho"):
        assert (both[f"{column}_old"] - both[f"{column}_new"]).abs().max() == 0.0, \
            f"{column} differs between the networked and offline runs"
