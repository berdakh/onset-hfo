"""The assistant answering *correctly* through a served model.

Every earlier run against a real server used random weights and could only show
the refusal path. These tests put a protocol-faithful fake Ollama on a free
port -- an oracle that does what the system prompt asks: survey, fetch
evidence for the leader, answer citing it -- and drive the shipped code through
it: `auto_backend` discovering the server, the agent loop, the guards, and the
CLI. A lying variant proves the guards still bite over the same wire.

No weights, no network beyond localhost, nothing mocked inside the package.
"""

from __future__ import annotations

import pytest
from _fake_ollama import FakeOllama

from onset_agent.agent import OnsetAgent
from onset_agent.backends import OllamaBackend
from onset_agent.hardware import Machine, auto_backend, ollama_status
from onset_agent.tools import dispatch

QUESTION = "Which channel had the highest ripple rate?"


def _laptop() -> Machine:
    return Machine(accelerator="cpu", ram_gb=16.0, free_disk_gb=100.0, cores=8,
                   torch_available=False, transformers_available=False)


@pytest.fixture
def served(monkeypatch):
    with FakeOllama() as fake:
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        yield fake


def test_auto_finds_the_served_model_without_pulling(served):
    """OLLAMA_HOST moved the server; the probe followed; the tag was already
    there, so nothing was pulled."""
    up, names = ollama_status()
    assert up and served.tag in names
    backend, choice = auto_backend(machine=_laptop())
    assert isinstance(backend, OllamaBackend)
    assert backend.model == served.tag
    assert choice.route == "ollama"
    assert served.pulled == []


def test_the_agent_answers_with_the_right_number_and_a_real_citation(served, store):
    """The success path, end to end: the answer names the real leader, quotes
    the rate the tool returned, and cites an evidence id the tools handed out.
    None of that is hard-coded here -- it is read back from the same store."""
    leader = dispatch(store, "top_channels", {"k": 1})["channels"][0]
    expected_ids = {e["evidence_id"] for e in
                    dispatch(store, "get_evidence", {"channel": leader["channel"]})["evidence"]}

    backend, _ = auto_backend(machine=_laptop())
    answer = OnsetAgent(store, backend).ask(QUESTION)

    assert not answer.refused, answer.text
    assert leader["channel"] in answer.text
    assert str(leader["rate_per_min"]) in answer.text
    assert answer.evidence_ids and set(answer.evidence_ids) <= expected_ids
    # The oracle did what the prompt asks: survey, then evidence, then answer.
    assert [c[2] for c in served.calls] == [0, 1, 2]


def test_a_model_that_invents_a_number_is_refused_over_the_same_wire(store, monkeypatch):
    """Without this, the test above proves only that the oracle is polite."""
    with FakeOllama(mode="liar") as fake:
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        backend, _ = auto_backend(machine=_laptop())
        answer = OnsetAgent(store, backend).ask(QUESTION)
    assert answer.refused
    assert "99.5" not in (answer.text or "") or "stand behind" in answer.text


def test_the_cli_answers_through_auto(served, result, tmp_path, capsys):
    from onset_agent import cli

    saved = result.save(tmp_path / "analysis")
    code = cli.main(["--results", str(saved), "--backend", "auto", "--question", QUESTION])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "[answer]" in out
    assert "[refused]" not in out
    assert f"via Ollama as {served.tag}" in out


def test_the_installer_style_pull_happens_when_the_tag_is_missing(monkeypatch):
    with FakeOllama(tag="qwen3:14b") as fake:          # not what the chooser wants
        monkeypatch.setenv("OLLAMA_HOST", fake.base)
        backend, choice = auto_backend(machine=_laptop())
    assert fake.pulled == [choice.ollama_tag]
    assert backend.model == choice.ollama_tag
