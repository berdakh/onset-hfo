"""The Laplacian reference, for ECoG grids and strips.

Each contact minus the mean of its neighbours on its own lead. What has to
hold: a grid's neighbours are its four sides (from positions when there are
any, else from the columns given), never a diagonal and never across a row's
end; a strip or shaft takes the contacts either side; a long lead with
neither positions nor columns is left as recorded rather than guessed at; and
the signal is what the arithmetic says.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onset_hfo.config import PreprocessConfig
from onset_hfo.preprocess import (
    describe,
    effective_reference,
    laplacian_neighbours,
    parse_grid_columns,
    prepare,
)

GRID = [f"G{i}" for i in range(1, 33)]


def test_a_grid_s_neighbours_are_its_four_sides_and_never_across_a_row_end():
    near, how = laplacian_neighbours(GRID + ["AD1", "AD2", "AD3"], grid_columns=(("G", 8),))
    assert sorted(near["G10"]) == ["G11", "G18", "G2", "G9"]
    assert sorted(near["G1"]) == ["G2", "G9"]          # a corner
    assert sorted(near["G8"]) == ["G16", "G7"]         # G9 starts the next row
    assert "G8" not in near["G9"]
    assert sorted(near["G32"]) == ["G24", "G31"]
    assert near["AD2"] == ["AD1", "AD3"] and near["AD1"] == ["AD2"]
    assert any("4×8 grid" in line for line in how)


def test_a_long_lead_with_nothing_to_go_on_is_left_as_recorded():
    near, how = laplacian_neighbours(GRID)
    assert near == {}
    assert any("left as recorded" in line and "G:8" in line for line in how)
    # A missing contact is simply not a neighbour.
    near, _ = laplacian_neighbours([n for n in GRID if n != "G11"], grid_columns=(("g", 8),))
    assert sorted(near["G10"]) == ["G18", "G2", "G9"]


def test_positions_give_the_sides_and_leave_out_the_diagonals():
    rng = np.random.default_rng(0)
    positions = {}
    for number in range(1, 17):           # a 4x4 grid, 10 mm apart, a little jitter
        row, col = divmod(number - 1, 4)
        positions[f"H{number}"] = np.array([col * 0.01, row * 0.01, 0.0]) \
            + rng.normal(0, 0.0003, 3)
    near, how = laplacian_neighbours(list(positions), positions)
    assert sorted(near["H6"]) == ["H10", "H2", "H5", "H7"]
    assert sorted(near["H1"]) == ["H2", "H5"]
    assert any("from their positions" in line for line in how)


def test_grid_columns_are_read_as_typed_and_refused_when_not():
    assert parse_grid_columns("G:8, lt:4") == (("G", 8), ("LT", 4))
    assert parse_grid_columns("") == ()
    for bad in ("G8", "G:", ":8", "G:0", "G:x"):
        with pytest.raises(ValueError):
            parse_grid_columns(bad)
    with pytest.raises(ValueError):
        from onset_hfo.preprocess import _check

        _check(PreprocessConfig(reference="laplacian", grid_columns=(("G", 0),)), 2000.0)


def test_the_signal_is_each_contact_minus_its_neighbours_mean(recording):
    plain = prepare(recording, PreprocessConfig(reference="none"), verbose=False)
    lap = prepare(recording, PreprocessConfig(reference="laplacian"), verbose=False)
    assert effective_reference(PreprocessConfig(reference="laplacian")) == "laplacian"
    assert lap.montage == "laplacian" and lap.ch_names == plain.ch_names
    near, _ = laplacian_neighbours(plain.ch_names)
    assert near, "the test recording's shafts are strips"
    index = {n: i for i, n in enumerate(plain.ch_names)}
    for name in plain.ch_names:
        expected = plain.data[index[name]]
        if name in near:
            expected = expected - plain.data[[index[n] for n in near[name]]].mean(axis=0)
        assert np.allclose(lap.data[index[name]], expected), name
    assert any(step.startswith("Laplacian reference") for step in lap.steps)
    summary, warnings = describe(PreprocessConfig(reference="laplacian",
                                                  grid_columns=(("G", 8),)),
                                 (80.0, 250.0), 2000.0)
    assert "neighbours (Laplacian) on grid G (8 columns)" in summary
    assert any("Laplacian" in w for w in warnings)


CACHED_GRID = (Path(__file__).resolve().parents[1] / "artifacts" / "data" / "ds003029"
               / "sub-pt01_task-ictal_run-01_50-60s")


@pytest.mark.skipif(not CACHED_GRID.is_dir(), reason="the cached ECoG recording is not here")
def test_on_a_real_ecog_grid(monkeypatch):
    from onset_hfo.datasets import fetch_slice

    rec = fetch_slice("sub-pt01", task="ictal", run="01", t_start=50, t_stop=60,
                      session="ses-presurgery", acq="ecog", dataset="ds003029",
                      cache_dir=CACHED_GRID.parents[1], verbose=False)
    plain = prepare(rec, PreprocessConfig(reference="none"), verbose=False)
    lap = prepare(rec, PreprocessConfig(reference="laplacian", grid_columns=(("G", 8),)),
                  verbose=False)
    index = {n: i for i, n in enumerate(plain.ch_names)}
    sides = ["G9", "G11", "G2", "G18"]
    expected = plain.data[index["G10"]] - plain.data[[index[n] for n in sides]].mean(axis=0)
    assert np.allclose(lap.data[lap.ch_names.index("G10")], expected)
    assert any("G: 4×8 grid" in step for step in lap.steps)


def _grid_file(tmp_path) -> Path:
    """SA as a 2×3 grid, 10 mm apart, names in lower case; SB and SC absent."""
    lines = ["name\tx\ty\tz"]
    for number in range(1, 7):
        row, col = divmod(number - 1, 3)
        lines.append(f"sa{number}\t{col * 10.0}\t{row * 10.0}\t40.0")
    path = tmp_path / "grid.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_positions_from_a_reader_s_file_set_a_lead_s_neighbours(recording, tmp_path):
    from onset_review.session import ReviewRequest, laplacian_positions

    request = ReviewRequest(preprocess=PreprocessConfig(reference="laplacian"),
                            electrodes_path=_grid_file(tmp_path))
    positions, name = laplacian_positions(request)
    assert name == "grid.tsv" and set(positions) == {f"SA{i}" for i in range(1, 7)}
    assert laplacian_positions(ReviewRequest(electrodes_path=_grid_file(tmp_path))) == \
        (None, ""), "only the Laplacian reads it"

    plain = prepare(recording, PreprocessConfig(reference="none"), verbose=False)
    lap = prepare(recording, PreprocessConfig(reference="laplacian"), verbose=False,
                  positions=positions, positions_from=name)
    index = {n: i for i, n in enumerate(plain.ch_names)}

    def expect(name, near):
        return plain.data[index[name]] - plain.data[[index[n] for n in near]].mean(axis=0)

    # SA2 sits mid-row: SA1, SA3 beside it and SA5 below; never the diagonals.
    assert np.allclose(lap.data[index["SA2"]], expect("SA2", ["SA1", "SA3", "SA5"]))
    assert np.allclose(lap.data[index["SA4"]], expect("SA4", ["SA1", "SA5"]))
    # SB is not in the file: the contacts either side, as before.
    assert np.allclose(lap.data[index["SB2"]], expect("SB2", ["SB1", "SB3"]))
    step = next(s for s in lap.steps if s.startswith("Laplacian"))
    assert "SA: 6 contacts, neighbours from their positions in grid.tsv" in step
    assert "SB: 6 in a line" in step


def test_the_review_session_uses_the_file_it_was_given(recording, tmp_path):
    from onset_review.session import ReviewRequest, session_from_recording

    request = ReviewRequest(t_start=0.0, t_stop=float(recording.duration),
                            preprocess=PreprocessConfig(reference="laplacian"),
                            electrodes_path=_grid_file(tmp_path))
    session = session_from_recording(recording, request)
    assert session.montage == "laplacian"
    assert any("from their positions in grid.tsv" in step for step in session.steps)


def test_a_file_given_under_the_laplacian_offers_a_re_analysis(tmp_path):
    import types

    from onset_review.app import _Review
    from onset_review.session import ReviewRequest

    for reference, answer, expected in (("laplacian", True, 1), ("laplacian", False, 0),
                                        ("bipolar", True, 0)):
        reruns = []
        fake = types.SimpleNamespace(
            request=ReviewRequest(preprocess=PreprocessConfig(reference=reference)),
            parts=object(), _ask_reanalyse_for_positions=lambda answer=answer: answer,
            _rerun=lambda reruns=reruns, **changes: reruns.append(changes))
        _Review.use_coordinates(fake, _grid_file(tmp_path))
        assert fake.request.electrodes_path == _grid_file(tmp_path)
        assert len(reruns) == expected, (reference, answer)
