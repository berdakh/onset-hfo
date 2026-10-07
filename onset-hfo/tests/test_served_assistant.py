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
    # The briefing had already surveyed and fetched the leader's evidence, so
    # the oracle answered on its first call, with those results in front of it.
    assert len(served.calls) == 1 and served.calls[0][2] >= 4


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


# -- stopping a served model ----------------------------------------------------


def test_abort_closes_the_connection_and_the_call_returns_interrupted():
    """A model that is taking its time is stopped by closing the request; the
    call in flight must come back promptly, as `Interrupted`, not after the
    server's own timeout."""
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from onset_agent.backends import Interrupted, OpenAICompatBackend

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_POST(self):       # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            time.sleep(8)        # a model "thinking"
            body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                        "content": "late"}}]}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

    # Threading, and daemon handler threads: a plain HTTPServer's shutdown
    # waits for the sleeping handler, which would be measured as the client
    # taking its time.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    backend = OpenAICompatBackend(model="slow", base_url=f"http://127.0.0.1:{server.server_port}/v1",
                                  timeout=30)
    outcome = {}

    def ask():
        try:
            backend.chat([{"role": "user", "content": "hi"}], [])
            outcome["result"] = "answered"
        except Interrupted:
            outcome["result"] = "interrupted"
        except Exception as error:      # noqa: BLE001
            outcome["result"] = f"other: {error!r}"

    worker = threading.Thread(target=ask)
    started = time.monotonic()
    worker.start()
    time.sleep(0.5)
    backend.abort()
    worker.join(timeout=5)
    elapsed = time.monotonic() - started
    server.shutdown()
    assert not worker.is_alive(), "the call did not come back after abort"
    assert outcome["result"] == "interrupted"
    assert elapsed < 5, "it came back promptly, not on timeout"
    with pytest.raises(Interrupted):
        backend.chat([{"role": "user", "content": "again"}], [])


def test_a_server_address_without_v1_still_reaches_the_chat_endpoint():
    """A reviewer types http://host:11434 into the box; Ollama's OpenAI
    endpoint is under /v1, and the backend adds it rather than 404ing."""
    assert OllamaBackend(model="m", base_url="http://127.0.0.1:11434").base_url \
        == "http://127.0.0.1:11434/v1"
    assert OllamaBackend(model="m", base_url="http://127.0.0.1:11434/v1/").base_url \
        == "http://127.0.0.1:11434/v1"
