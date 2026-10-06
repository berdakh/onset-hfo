"""The model box on the Assistant panel, against a stand-in Ollama.

The chooser, the status call, the streamed pull, the defaults file and the
hand-over to the panel are all the shipped code; only the server and the
machine are stood in for. A real pull is several gigabytes and needs a daemon
this test host does not have, which is exactly why the box exists.
"""

from __future__ import annotations

import os

import pytest

qt = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")

from _fake_ollama import FakeOllama  # noqa: E402

from onset_agent import hardware  # noqa: E402
from onset_agent.hardware import Machine  # noqa: E402
from onset_review.assistant_config import config_path, load_defaults  # noqa: E402
from onset_review.modelsetup import ModelSetupBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield qt.QApplication.instance() or qt.QApplication([])


def _laptop() -> Machine:
    return Machine(accelerator="cpu", ram_gb=16.0, free_disk_gb=60.0, cores=8)


@pytest.fixture
def described(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "probe", lambda machine=None: _laptop())
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path))
    return hardware.served_options(_laptop())[1]


def test_the_box_pulls_the_choosers_pick_and_hands_it_over(qapp, described, monkeypatch):
    expected = described
    with FakeOllama(tag="something:else") as fake:        # holds the wrong model
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        box = ModelSetupBox()
        ready = []
        box.modelReady.connect(lambda tag, url: ready.append((tag, url)))

        box.look()
        box.wait()
        assert expected in box.status.text()
        assert "not pulled" in box.status.text()
        assert box.action.text() == "Download and use" and box.action.isEnabled()

        box.action.click()
        box.wait()
        assert fake.pulled == [expected], "the pull went to the server, once"
        assert ready == [(expected, fake.base.rstrip("/") + "/v1")]
        assert box.action.text() == "Use this model"
        assert "ready" in box.status.text()

    saved = load_defaults(env={}, path=config_path(), probe=False)
    assert saved.kind == "ollama" and saved.model == expected


def test_a_model_already_there_is_used_without_a_pull(qapp, described, monkeypatch):
    with FakeOllama(tag=described) as fake:
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        box = ModelSetupBox()
        ready = []
        box.modelReady.connect(lambda tag, url: ready.append(tag))
        box.look()
        box.wait()
        assert "already holds" in box.status.text()
        assert box.action.text() == "Use this model"
        box.action.click()
        box.wait()
        assert fake.pulled == [] and ready == [described]


def test_no_server_means_the_address_and_a_disabled_button(qapp, described, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")      # nothing listens
    box = ModelSetupBox()
    box.look()
    box.wait()
    text = box.status.text()
    assert "No Ollama server" in text and "ollama.com/download" in text
    assert described in text, "it still says what it would pull"
    assert not box.action.isEnabled() and box.again.isEnabled()


def test_the_window_opens_on_the_fast_size_and_offers_the_others(qapp, described, monkeypatch):
    """A 7B on a CPU took minutes per question. The box opens on the 3B --
    the policy is the window's, the CLI keeps the reference size -- and lists
    every size that fits, smallest first, so the reviewer can go either way."""
    assert described == "qwen2.5:3b-instruct"
    assert hardware.choose(_laptop(), route="ollama").ollama_tag == "qwen2.5:7b-instruct"
    with FakeOllama(tag="qwen2.5:7b-instruct") as fake:        # the big one is there
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        box = ModelSetupBox()
        box.look()
        box.wait()
        tags = [box.size.itemData(i) for i in range(box.size.count())]
        assert tags[:3] == ["qwen3:0.6b", "qwen2.5:1.5b-instruct", "qwen3:1.7b"]
        assert "qwen2.5:7b-instruct" in tags and tags == sorted(
            tags, key=lambda t: [c.ollama_tag for c in hardware.served_options(_laptop())[0]].index(t))
        assert box.size.currentData() == described
        assert "not pulled" in box.status.text() and box.action.text() == "Download and use"
        # Pick the size that is already there: no pull, straight to use.
        box.size.setCurrentIndex(tags.index("qwen2.5:7b-instruct"))
        assert "already holds" in box.status.text() and "your pick" in box.status.text()
        assert box.action.text() == "Use this model"
        ready = []
        box.modelReady.connect(lambda tag, url: ready.append(tag))
        box.action.click()
        box.wait()
        assert fake.pulled == [] and ready == ["qwen2.5:7b-instruct"]
        # And back to a size that is not: the pull goes for that one.
        box.size.setCurrentIndex(tags.index("qwen2.5:1.5b-instruct"))
        assert box.action.text() == "Download and use"
        box.action.click()
        box.wait()
        assert fake.pulled == ["qwen2.5:1.5b-instruct"]


def test_the_panel_switches_to_the_model_the_box_made_ready(qapp, monkeypatch):
    from onset_review.assistant import AssistantPanel

    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    panel = AssistantPanel(session=object())
    panel.show()
    qapp.processEvents()
    assert panel._looked is True and "Not checked" in panel.setup.status.text(), \
        "no probe when the environment forbids it, and it says so"
    panel.setup.modelReady.emit("qwen3:4b", "http://127.0.0.1:11434/v1")
    assert panel.backend.currentData() == "ollama"
    assert panel.model.text() == "qwen3:4b"
    assert "qwen3:4b" in panel.transcript.toPlainText()
    panel.close()


# -- stopping, and saying why ----------------------------------------------------


def test_the_stop_button_is_only_live_while_a_question_runs(qapp, monkeypatch):
    from onset_review.assistant import AssistantPanel

    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    panel = AssistantPanel(session=object())
    assert not panel.stop_button.isEnabled() and panel.send.isEnabled()
    panel._set_busy(True)
    assert panel.stop_button.isEnabled() and not panel.send.isEnabled()
    assert panel.busy.text().startswith("thinking")
    panel._set_busy(False)
    assert not panel.stop_button.isEnabled() and panel.send.isEnabled()
    assert panel.busy.text() == ""
    panel.stop()        # nothing running: a no-op, not an error
    panel.close()


def test_a_refusal_says_what_the_model_wrote_and_which_check_failed(qapp, monkeypatch):
    from onset_agent.agent import AgentAnswer
    from onset_review.assistant import AssistantPanel, explain_refusal

    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    answer = AgentAnswer(
        question="q", text="I could not produce an answer I can stand behind.",
        refused=True, reason="verification failed", trace=[
            {"type": "tool_call", "tool": "top_channels", "ok": True},
            {"type": "answer", "step": 1, "verified": False,
             "problems": ["the value '99.5 events/min' does not appear in any tool result"],
             "text": "AR1-AR2 had 99.5 events/min.", "evidence_ids": []},
            {"type": "format_error", "step": 2, "content": "Sure! Here is"},
            {"type": "gave_up", "steps": 6, "retries": 2}])
    lines = explain_refusal(answer)
    assert any("99.5 events/min" in line and "AR1-AR2 had" in line for line in lines)
    assert any("required JSON" in line for line in lines)
    assert any("Gave up after 6 steps" in line for line in lines)
    panel = AssistantPanel(session=object())
    panel._say_answer(answer)
    shown = panel.transcript.toPlainText()
    assert "Refused." in shown and "99.5 events/min" in shown and "Gave up" in shown
    stopped = AgentAnswer(question="q", text="Stopped before an answer was produced.",
                          refused=True, reason="stopped by the reviewer",
                          trace=[{"type": "stopped", "step": 0}])
    assert explain_refusal(stopped) == ["Stopped by you."]
    panel.close()


# -- seeing the work, and the larger type --------------------------------------


def test_the_transcript_shows_which_data_the_model_saw_and_what_the_checks_made_of_it(
        qapp, monkeypatch):
    """Asked for twice: a wait with nothing on screen is a hang as far as the
    person at the window is concerned, and a refusal with no visible cause is
    a bug as far as they are concerned. Every trace entry is a line, as it
    happens."""
    from onset_review.assistant import AssistantPanel, describe_step

    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    panel = AssistantPanel(session=object())
    assert panel.transcript.font().pointSizeF() >= 11, "the transcript was asked to be larger"
    lines = [describe_step(e) for e in [
        {"type": "tool_call", "tool": "top_channels", "arguments": {"k": 5}, "ok": True,
         "briefing": True, "digest": "rms: AR1-AR2 22.0/min, AHR1-AHR2 19.0/min"},
        {"type": "tool_call", "tool": "get_evidence", "arguments": {"channel": "AR1-AR2"},
         "ok": True, "digest": "AR1-AR2: 2 window(s), 3.5–3.6 s (120 Hz)"},
        {"type": "model", "step": 0, "asked_for": ["get_evidence(channel=AR1-AR2)"],
         "content": ""},
        {"type": "prose", "step": 1, "content": "AR1-AR2 leads."},
        {"type": "citations", "attached": ["sub-01|AR1-AR2|rms|3.505"],
         "dropped": ["evidence_id_1"]},
        {"type": "answer", "verified": True, "evidence_ids": ["sub-01|AR1-AR2|rms|3.505"]},
        {"type": "answer", "verified": False, "problems": ["the value '99 /min' does not appear"]},
    ]]
    assert "Retrieved for the model" in lines[0] and "22.0/min" in lines[0]
    assert "The model asked for" in lines[1] and "get_evidence" in lines[1]
    assert lines[2] is None, "the tool line says it"
    assert "sentence" in lines[3]
    assert "Attached 1" in lines[4] and "Dropped 1" in lines[4] and "evidence_id_1" in lines[4]
    assert lines[5].startswith("Checks passed") and "1 citation" in lines[5]
    assert lines[6].startswith("Check failed") and "99 /min" in lines[6]
    for line in lines:
        if line:
            panel._progress({"type": "x"})      # unknown entries are ignored
    panel._progress({"type": "tool_call", "tool": "top_channels", "arguments": {}, "ok": True,
                     "briefing": True, "digest": "rms: AR1-AR2 22.0/min"})
    assert "Retrieved for the model top_channels: rms: AR1-AR2 22.0/min" \
        in panel.transcript.toPlainText().replace("· ", "")
    panel.close()


def test_what_can_you_do_is_answered_at_once_without_evidence_or_a_model(qapp, monkeypatch):
    """The first thing a reviewer types, often before any recording is open;
    it must not build a pipeline result or reach for a server to answer."""
    from onset_review.assistant import AssistantPanel

    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    panel = AssistantPanel(session=object())          # no recording at all
    panel.ask("What can you do?")
    text = panel.transcript.toPlainText()
    assert "highest ripple" in text and "treatment" in text and "this window" in text
    assert "no recording attached" not in text and panel._store is None
    assert "no model needed" in text
    panel.close()
