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
from pathlib import Path

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
    and the unreadable file is kept rather than overwritten -- beside where
    it was, since left in place the next save would replace it."""
    read = adj.Adjudication()
    read.judge_event(adj.event_key("A1-A2", 1.0), "agree", reader="Dr Smith")
    path = adj.save(request_, read)
    path.write_text("{not json at all", encoding="utf-8")

    recovered = adj.load(request_)
    assert recovered.empty
    assert Path(recovered.set_aside).read_text(encoding="utf-8") == "{not json at all"


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


def test_a_verdict_on_an_oscillation_does_not_land_on_a_discharge():
    """10 ms apart on the same channel, and not the same thing: "real" said of
    a ripple would otherwise count the spike as confirmed by a reader who
    never judged it."""
    import dataclasses

    read = adj.Adjudication(reader="Dr Smith")
    ripple = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(ripple, "agree")
    spike = dataclasses.replace(_Event("A1-A2", 12.330), detector="spike")

    out = adj.reconcile(read, [spike])
    assert out.events == {}
    assert list(out.orphaned) == [ripple]

    # Another oscillation detector finding it 10 ms off is the same event.
    other = dataclasses.replace(_Event("A1-A2", 12.330), detector="line_length")
    out = adj.reconcile(read, [other])
    assert out.verdict_of(adj.event_key("A1-A2", 12.330, "line_length")) == "agree"


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


def test_an_unreadable_read_is_moved_aside_not_written_over(request_):
    """A truncated file loaded as an empty read and stayed where it was, so
    the next verdict's save replaced it: a crash during one write cost every
    verdict before it."""
    read = adj.Adjudication()
    read.judge_event(adj.event_key("A1-A2", 1.0, "rms"), "agree", reader="Dr Smith")
    path = adj.save(request_, read)
    damaged = path.read_text(encoding="utf-8")[:40]
    path.write_text(damaged, encoding="utf-8")

    loaded = adj.load(request_)
    assert loaded.events == {} and loaded.set_aside
    assert not path.exists()
    aside = [p for p in path.parent.iterdir() if ".unreadable-" in p.name]
    assert [str(p) for p in aside] == [loaded.set_aside]
    assert aside[0].read_text(encoding="utf-8") == damaged

    loaded.judge_event(adj.event_key("A1-A2", 2.0, "rms"), "disagree", reader="Dr B")
    adj.save(request_, loaded)
    assert aside[0].read_text(encoding="utf-8") == damaged
    assert "set_aside" not in path.read_text(encoding="utf-8")
    assert adj.reconcile(loaded, []).set_aside == loaded.set_aside


def test_a_read_of_the_wrong_shape_does_not_raise(request_):
    path = adj.save(request_, adj.Adjudication())
    path.write_text('{"events": ["not", "a", "mapping"]}', encoding="utf-8")
    assert adj.load(request_).set_aside


def test_a_near_match_does_not_take_another_verdicts_own_event():
    """Two verdicts 30 ms apart; a re-run keeps only the later event. The
    earlier verdict used to claim it as a near match, and the verdict that
    was actually given to it was orphaned."""
    read = adj.Adjudication(reader="Dr Smith")
    gone = adj.event_key("A1-A2", 12.340, "rms")
    kept = adj.event_key("A1-A2", 12.370, "rms")
    read.judge_event(gone, "disagree")
    read.judge_event(kept, "agree")

    out = adj.reconcile(read, [_Event("A1-A2", 12.370)])
    assert out.verdict_of(kept) == "agree" and out.events[kept].match == "exact"
    assert list(out.orphaned) == [gone]


# -- a second reader ----------------------------------------------------------


def test_a_second_readers_verdict_does_not_erase_the_first(request_):
    """Dr B's click replaced Dr A's verdict and left no sign there had been
    one. The later verdict still counts; the earlier one is kept."""
    key = adj.event_key("A1-A2", 3.0, "rms")
    other = adj.event_key("A1-A2", 7.0, "rms")
    read = adj.Adjudication()
    read.judge_event(key, "agree", reader="Dr A", note="clear island")
    read.judge_event(other, "agree", reader="Dr A")
    read.judge_event(key, "disagree", reader="Dr B")
    read.judge_event(other, "agree", reader="Dr B")

    assert read.verdict_of(key) == "disagree" and read.events[key].reader == "Dr B"
    assert [(j.reader, j.verdict, j.note) for j in read.superseded[key]] == [
        ("Dr A", "agree", "clear island")]
    assert read.readers == ["Dr A", "Dr B"]
    second = read.second_opinions()
    assert second["n"] == 2 and second["agreed"] == 1
    assert second["disagreed"] == [(key, "Dr A", "agree", "Dr B", "disagree")]

    # Changing your own mind is still a replacement, not a second opinion.
    read.judge_event(key, "unsure", reader="Dr B")
    assert len(read.superseded[key]) == 1

    adj.save(request_, read)
    back = adj.load(request_)
    assert back.superseded[key][0].reader == "Dr A"
    assert back.second_opinions()["n"] == 2


def test_an_earlier_verdict_follows_its_event_through_a_re_analysis():
    read = adj.Adjudication()
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree", reader="Dr A")
    read.judge_event(key, "disagree", reader="Dr B")
    out = adj.reconcile(read, [_Event("A1-A2", 12.355)])
    moved = adj.event_key("A1-A2", 12.355, "rms")
    assert out.verdict_of(moved) == "disagree"
    assert out.superseded[moved][0].reader == "Dr A"
    assert out.second_opinions()["disagreed"][0][0] == moved


def test_a_read_remembers_which_file_it_was_given_on(request_):
    """A read is found by the file's path. The fingerprint is how it is known
    to be about the file now there and not one that replaced it."""
    read = adj.Adjudication(fingerprint="size:10:aaaa")
    read.judge_event(adj.event_key("A1-A2", 1.0, "rms"), "agree", reader="Dr A")
    adj.save(request_, read)
    back = adj.load(request_)
    assert back.fingerprint == "size:10:aaaa"
    assert adj.reconcile(back, []).fingerprint == "size:10:aaaa"
    # An archive window has none, and its file does not grow a field.
    plain = adj.Adjudication()
    assert "fingerprint" not in plain.to_json()


# -- what the window was recorded under ---------------------------------------


def test_recording_conditions_are_kept_with_the_read_and_never_guessed(request_):
    read = adj.Adjudication(reader="Dr A")
    assert dict(read.condition_rows()) == {
        "Vigilance state": adj.NOT_RECORDED,
        "Since the last seizure": adj.NOT_RECORDED,
        "Anti-seizure medication": adj.NOT_RECORDED}
    assert read.empty

    read.set_conditions({"state": "N2/N3 sleep", "medication": "being tapered",
                         "since_seizure": ""})
    assert not read.empty
    rows = dict(read.condition_rows())
    assert rows["Vigilance state"] == "N2/N3 sleep"
    assert rows["Since the last seizure"] == adj.NOT_RECORDED
    assert read.conditions["by"] == "Dr A" and read.conditions["at"]

    adj.save(request_, read)
    back = adj.reconcile(adj.load(request_), [])
    assert back.conditions["medication"] == "being tapered"
    back.set_conditions({})
    assert back.conditions == {}


# -- a verdict is about one band ----------------------------------------------


@dataclass
class _Banded:
    channel: str
    start: float
    detector: str = "rms"
    band: tuple = (80.0, 250.0)


def test_a_ripple_verdict_does_not_carry_onto_a_fast_ripple_at_the_same_moment():
    """The key is channel, onset and detector. Re-run in the fast-ripple band,
    the same detector finds a different event there, and "a real ripple" was
    carried onto it."""
    ripple = _Banded("A1-A2", 12.340)
    read = adj.reconcile(adj.Adjudication(reader="Dr A"), [ripple])
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree")
    assert read.events[key].band == "80-250"
    assert read.to_json()["events"][key]["band"] == "80-250"

    fast = _Banded("A1-A2", 12.340, band=(250.0, 500.0))
    near = _Banded("A1-A2", 12.352, band=(250.0, 500.0))
    for events in ([fast], [near]):
        out = adj.reconcile(read, events)
        assert out.events == {} and list(out.orphaned) == [key]

    # Back in the ripple band it is found again, and it is still Dr A's.
    # (Through the file, as the application does: orphans are saved among the
    # events they came from.)
    stored = adj.Adjudication.from_json(adj.reconcile(read, [fast]).to_json())
    back = adj.reconcile(stored, [ripple])
    assert back.verdict_of(key) == "agree" and back.events[key].match == "exact"


def test_a_verdict_from_before_bands_were_kept_matches_as_it_always_did():
    read = adj.Adjudication(reader="Dr A")
    key = adj.event_key("A1-A2", 12.340, "rms")
    read.judge_event(key, "agree")                 # no analysis attached: no band
    assert read.events[key].band == ""
    out = adj.reconcile(read, [_Banded("A1-A2", 12.340, band=(250.0, 500.0))])
    assert out.verdict_of(key) == "agree"
    assert adj.band_label((80.0, 250.0)) == "80-250" and adj.band_label(None) == ""
