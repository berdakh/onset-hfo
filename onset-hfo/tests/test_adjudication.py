"""The reader's verdicts: attribution, persistence, and what happens to a
judgement when the analysis under it changes.

These are the tests that matter most in this module, because the failure they
guard against is silent. A dropped verdict does not raise; it just means a
clinician's recorded opinion is gone and nobody finds out until they look for
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from onset_review import adjudication as adj


@dataclass
class _Event:
    """Just enough of `onset_hfo.detectors.base.Event` to key on."""

    channel: str
    start: float
    detector: str = "rms"


@pytest.fixture
def request_(tmp_path, monkeypatch):
    """A window identity whose read lands in a temporary directory."""
    monkeypatch.setenv("ONSET_REVIEW_READS", str(tmp_path / "reads"))

    @dataclass
    class _Request:
        dataset: str = "ds003498"
        subject: str = "sub-01"
        run: str = "01"
        t_start: float = 0.0
        t_stop: float = 60.0
        path: object = None

    return _Request()


# -- identity ---------------------------------------------------------------


def test_the_window_id_ignores_everything_about_the_analysis(request_):
    """A verdict is about the signal, so it must not be filed under settings.

    This is the test that keeps someone from "fixing" the id by adding the
    band or the threshold to it, which would silently file a second read every
    time a reviewer changed a filter.
    """
    identity = adj.window_id(request_)
    assert "sub-01" in identity and "0-60s" in identity
    for noise in ("ripple", "rms", "threshold", "notch", "bipolar"):
        assert noise not in identity


def test_two_files_with_the_same_name_are_different_recordings(tmp_path):
    @dataclass
    class _Imported:
        dataset: str = "import"
        subject: str = "local"
        run: str = "01"
        t_start: float = 0.0
        t_stop: float = 60.0
        path: object = None

    one, two = tmp_path / "a", tmp_path / "b"
    one.mkdir()
    two.mkdir()
    (one / "export.edf").write_bytes(b"")
    (two / "export.edf").write_bytes(b"")
    first = adj.window_id(_Imported(path=one / "export.edf"))
    second = adj.window_id(_Imported(path=two / "export.edf"))
    assert first != second
    assert "export" in first and "export" in second


def test_the_reader_name_never_invents_an_identity(monkeypatch):
    """A login name is a prefill, and `root` is not a clinician."""
    for name in ("ONSET_REVIEWER", "USER", "USERNAME"):
        monkeypatch.delenv(name, raising=False)
    assert adj.reader_name() == ""
    monkeypatch.setenv("USER", "root")
    assert adj.reader_name() == ""
    monkeypatch.setenv("USER", "jdoe")
    assert adj.reader_name() == "jdoe"
    assert adj.reader_name("Dr Smith") == "Dr Smith"


# -- recording a verdict ----------------------------------------------------


def test_a_verdict_carries_who_and_when():
    read = adj.Adjudication()
    key = adj.event_key("AD1-AD2", 12.345, "rms")
    judged = read.judge_event(key, "agree", reader="Dr Smith")
    assert judged.verdict == "agree"
    assert judged.reader == "Dr Smith"
    assert judged.at.endswith("Z") and judged.at.startswith("20")
    assert read.verdict_of(key) == "agree"


def test_an_unknown_verdict_is_refused():
    read = adj.Adjudication()
    with pytest.raises(ValueError):
        read.judge_event(adj.event_key("A1-A2", 1.0), "probably")
    with pytest.raises(ValueError):
        read.judge_channel("A1-A2", "agree")   # an event verdict, not a channel one


def test_changing_your_mind_keeps_the_note_that_explained_the_first_one():
    read = adj.Adjudication(reader="Dr Smith")
    key = adj.event_key("AD1-AD2", 12.345, "rms")
    read.judge_event(key, "disagree", note="ringing on a sharp transient")
    read.judge_event(key, "agree")
    assert read.events[key].verdict == "agree"
    assert read.events[key].note == "ringing on a sharp transient"


def test_a_second_reader_is_recorded_rather_than_collapsed():
    read = adj.Adjudication()
    read.judge_event(adj.event_key("A1-A2", 1.0), "agree", reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 2.0), "disagree", reader="Dr Jones")
    assert read.readers == ["Dr Smith", "Dr Jones"]


# -- progress ---------------------------------------------------------------


def test_a_channel_is_complete_only_when_every_event_on_it_was_judged():
    """Why it matters: a confirmed *rate* may only be quoted for a complete
    channel. Half a channel judged is a count, not a rate."""
    events = [_Event("A1-A2", 1.0), _Event("A1-A2", 2.0), _Event("B1-B2", 3.0)]
    read = adj.Adjudication(reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 1.0, "rms"), "agree")
    progress = read.progress(events)
    assert progress["A1-A2"]["judged"] == 1
    assert progress["A1-A2"]["total"] == 2
    assert progress["A1-A2"]["complete"] is False
    assert progress["B1-B2"]["judged"] == 0

    read.judge_event(adj.event_key("A1-A2", 2.0, "rms"), "disagree")
    progress = read.progress(events)
    assert progress["A1-A2"]["complete"] is True
    assert progress["A1-A2"]["agree"] == 1
    assert progress["A1-A2"]["disagree"] == 1


# -- files ------------------------------------------------------------------


def test_a_read_survives_a_round_trip_through_disk(request_):
    read = adj.load(request_)
    assert read.empty
    key = adj.event_key("AD1-AD2", 12.345, "rms")
    read.judge_event(key, "agree", reader="Dr Smith", note="clear ripple")
    read.judge_channel("AD1-AD2", "accept", reader="Dr Smith")
    read.note = "Reviewed the first minute only."
    path = adj.save(request_, read)
    assert path.exists()

    again = adj.load(request_)
    assert again.verdict_of(key) == "agree"
    assert again.events[key].note == "clear ripple"
    assert again.events[key].reader == "Dr Smith"
    assert again.channel_verdict("AD1-AD2") == "accept"
    assert again.note == "Reviewed the first minute only."


def test_a_corrupt_file_costs_the_verdicts_but_not_the_window(request_):
    """Opening the recording has to work even when the read cannot be parsed,
    and the unreadable file is left alone rather than overwritten."""
    read = adj.Adjudication()
    read.judge_event(adj.event_key("A1-A2", 1.0), "agree", reader="Dr Smith")
    path = adj.save(request_, read)
    path.write_text("{not json at all", encoding="utf-8")

    recovered = adj.load(request_)
    assert recovered.empty
    assert path.read_text(encoding="utf-8") == "{not json at all"


def test_the_write_is_atomic_in_the_sense_that_matters(request_):
    """No `.tmp` left behind, so a directory listing shows one read per window."""
    read = adj.Adjudication()
    read.judge_event(adj.event_key("A1-A2", 1.0), "agree", reader="Dr Smith")
    adj.save(request_, read)
    adj.save(request_, read)
    files = sorted(p.name for p in adj.store_dir().iterdir())
    assert files == [f"{adj.window_id(request_)}.json"]


def test_the_file_is_readable_by_something_that_is_not_this_module(request_):
    """A clinical record a person cannot open without the software that wrote
    it is not a record. Plain JSON, with the vocabulary spelled out."""
    read = adj.Adjudication()
    read.judge_event(adj.event_key("AD1-AD2", 12.345, "rms"), "disagree",
                     reader="Dr Smith", note="filter ringing")
    data = json.loads(adj.save(request_, read).read_text(encoding="utf-8"))
    assert data["schema"] == adj.SCHEMA
    entry = data["events"]["AD1-AD2|12.345|rms"]
    assert entry == {"verdict": "disagree", "reader": "Dr Smith",
                     "at": entry["at"], "note": "filter ringing"}


# -- reconciliation: the whole point ----------------------------------------


def test_a_verdict_follows_its_event_through_a_small_onset_shift():
    """Re-running with a different threshold moves an onset by milliseconds.
    That is the same oscillation, and the reader should not be asked again."""
    read = adj.Adjudication(reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 12.340, "rms"), "agree")
    moved = [_Event("A1-A2", 12.355)]          # 15 ms later

    out = adj.reconcile(read, moved)
    key = adj.event_key("A1-A2", 12.355, "rms")
    assert out.verdict_of(key) == "agree"
    assert out.events[key].match == "near"
    assert out.orphaned == {}


def test_an_exact_match_is_reported_as_one():
    read = adj.Adjudication(reader="Dr Smith")
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree")
    out = adj.reconcile(read, [_Event("A1-A2", 12.340)])
    assert out.events[key].match == "exact"


def test_a_verdict_whose_event_is_gone_is_kept_not_deleted():
    """The one behaviour this module must not have."""
    read = adj.Adjudication(reader="Dr Smith")
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree", note="clear ripple")

    out = adj.reconcile(read, [_Event("B9-B10", 40.0)])
    assert out.events == {}
    assert key in out.orphaned
    assert out.orphaned[key].note == "clear ripple"
    assert out.counts()["orphans"] == 1


def test_an_orphan_is_written_back_and_found_again_by_the_old_settings(request_):
    """Orphaning must be reversible: undo the filter change and the verdict is
    there. That is only true if the orphan was saved under its original key."""
    read = adj.Adjudication(reader="Dr Smith")
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree")
    adj.save(request_, adj.reconcile(read, [_Event("B9-B10", 40.0)]))

    restored = adj.reconcile(adj.load(request_), [_Event("A1-A2", 12.340)])
    assert restored.verdict_of(key) == "agree"
    assert restored.orphaned == {}


def test_two_verdicts_never_land_on_the_same_event():
    """Without one-event-per-verdict, two neighbouring verdicts both claim the
    nearer event and one reader's opinion silently overwrites the other's."""
    read = adj.Adjudication(reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 12.300, "rms"), "agree")
    read.judge_event(adj.event_key("A1-A2", 12.320, "rms"), "disagree")

    out = adj.reconcile(read, [_Event("A1-A2", 12.310)])
    assert len(out.events) == 1
    assert len(out.orphaned) == 1
    assert set(out.counts().values()) >= {1}


def test_a_verdict_does_not_jump_to_a_different_oscillation():
    """Half a second away is a different event, whatever the channel."""
    read = adj.Adjudication(reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 12.000, "rms"), "agree")
    out = adj.reconcile(read, [_Event("A1-A2", 12.500)])
    assert out.events == {}
    assert len(out.orphaned) == 1


def test_a_verdict_does_not_cross_channels():
    read = adj.Adjudication(reader="Dr Smith")
    read.judge_event(adj.event_key("A1-A2", 12.000, "rms"), "agree")
    out = adj.reconcile(read, [_Event("A2-A3", 12.000)])
    assert out.events == {}
    assert len(out.orphaned) == 1


def test_channel_verdicts_and_the_window_note_survive_reconciliation():
    """They are about the contact and the window, not about any one event, so
    nothing the detector does can orphan them."""
    read = adj.Adjudication(reader="Dr Smith", note="first minute only")
    read.judge_channel("A1-A2", "ignore", note="popping electrode")
    out = adj.reconcile(read, [])
    assert out.channel_verdict("A1-A2") == "ignore"
    assert out.channels["A1-A2"].note == "popping electrode"
    assert out.note == "first minute only"
    assert out.reader == "Dr Smith"
