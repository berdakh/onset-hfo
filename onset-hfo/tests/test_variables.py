"""The workspace list, Qt-free: what the session holds, named and sized."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from onset_review import variables as V
from onset_review.session import ReviewRequest, session_from_recording


@pytest.fixture(scope="module")
def review(recording):
    return session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration)))


def test_every_session_field_is_listed_with_a_kind_a_size_and_a_group(review):
    listed = V.variables(review)
    names = [v.name for v in listed]
    for expected in ("raw", "signal", "times", "channels", "events", "findings",
                     "leader", "candidates", "quality", "segments", "read", "request",
                     "steps", "resection", "electrodes"):
        assert expected in names
    assert len(names) == len(set(names)), "one row per name"
    groups = [g for g, _ in V.GROUPS]
    assert all(v.group in groups for v in listed)
    by = {v.name: v for v in listed}
    assert by["raw"].kind == "Raw (MNE)" and "ch ×" in by["raw"].size
    assert by["findings"].kind == "DataFrame" and by["findings"].size.startswith(
        f"{len(review.findings):,} ×")
    assert by["events"].kind == "list[Event]" and "accepted" in by["events"].summary
    assert by["resection"].kind == "None" and by["resection"].missing
    assert "not available" in by["resection"].summary
    assert by["request"].kind == "ReviewRequest" and "subject=" in by["request"].summary


def test_the_signal_is_described_without_being_fetched_and_fetched_on_demand(review):
    by = {v.name: v for v in V.variables(review)}
    signal = by["signal"]
    assert signal.value is None and signal.load is not None and not signal.missing
    assert signal.kind == "ndarray float64" and "fetched when opened" in signal.summary
    n_ch, n_times = len(review.raw.ch_names), int(review.raw.n_times)
    assert signal.size == f"{n_ch:,} × {n_times:,}"
    array = signal.get()
    assert isinstance(array, np.ndarray) and array.shape == (n_ch, n_times)
    times = by["times"].get()
    assert times.shape == (n_times,) and times[0] == pytest.approx(review.t_offset)


def test_describe_covers_the_shapes_a_session_holds():
    kind, size, summary = V.describe(pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))
    assert (kind, size) == ("DataFrame", "2 × 2") and "a, b" in summary
    kind, size, summary = V.describe(np.arange(6.0).reshape(2, 3))
    assert kind == "ndarray float64" and size == "2 × 3" and "min 0" in summary and "max 5" in summary
    assert V.describe(None)[0] == "None"
    assert V.describe(["x", "y"])[:2] == ("list[str]", "2")
    assert V.describe({"k": 1})[:2] == ("dict", "1")
    assert V.describe(2.5)[2] == "2.5" and V.describe(True)[0] == "bool"
    kind, size, summary = V.describe("x" * 200)
    assert kind == "str" and size == "200" and summary.endswith("…") and len(summary) <= 90


def test_the_views_follow_the_object(review):
    assert V.as_table(review.findings) is review.findings
    events = V.as_table(review.events)
    assert isinstance(events, pd.DataFrame) and len(events) == len(review.events)
    assert "channel" in events.columns
    assert V.as_table(review.raw) is None and V.as_array(review.raw).ndim == 2
    assert V.as_table({"a": 1, "b": 2}).shape == (2, 2)
    assert V.as_table(["x", "y"]).shape == (2, 1)
    tree = V.as_tree(review.request)
    assert tree and tree[0][0] == "dataset"
    nested = V.as_tree({"outer": {"inner": [1, 2]}})
    assert nested[0][0] == "outer" and nested[0][2][0][0] == "inner"
    assert V.as_tree("plain text") == [] and V.as_tree(3.0) == []


def test_export_writes_the_type_the_object_calls_for(review, tmp_path):
    by = {v.name: v for v in V.variables(review)}
    out = V.export(review.findings, tmp_path / "findings")
    assert out.suffix == ".csv" and len(pd.read_csv(out)) == len(review.findings)
    out = V.export(by["signal"].get(), tmp_path / "signal.whatever")
    assert out.suffix == ".npy" and np.load(out).shape == by["signal"].get().shape
    out = V.export(review.request, tmp_path / "request")
    assert out.suffix == ".json"
    assert json.loads(out.read_text())["subject"] == review.request.subject
    out = V.export(review.events, tmp_path / "events")
    assert out.suffix == ".csv" and "channel" in pd.read_csv(out).columns
    out = V.export(review.citation, tmp_path / "citation")
    assert out.suffix == ".txt" and out.read_text() == review.citation
    out = V.export(review.read, tmp_path / "read")
    assert out.suffix == ".json" and "events" in json.loads(out.read_text())
