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
    return hardware.choose(_laptop(), route="ollama").ollama_tag


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
