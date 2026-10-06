"""A stand-in Ollama that speaks the real wire protocol and answers like a
careful model -- or, on request, like a careless one.

Not a language model: an oracle. It reads whatever tool results are in the
conversation -- the agent's briefing puts the survey and the leaders' evidence
there before the first call -- and calls ``top_channels`` only if no survey is
there, ``get_evidence`` only if the leader's windows are not, exactly as the
system prompt instructs; then it writes the answer the contract demands and
cites the evidence id it was handed. Every layer between the
reviewer and it -- the panel, the worker thread, OllamaBackend, the agent loop,
the citation and number guards -- is the shipped code. Only the brain is
scripted, and scripted to be right, so the success path is exercised rather
than only the refusal path a random model produces.

``mode="liar"`` states a rate it never retrieved; the number guard must catch
that through the same wire, or the success path would be proving nothing.

Serves  GET /api/tags  POST /api/pull  GET /v1/models  POST /v1/chat/completions
on a free port, in a daemon thread. Set ``OLLAMA_HOST`` to ``base`` and the
shipped code finds it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

__all__ = ["FakeOllama"]


def _leader(payload: dict) -> dict | None:
    rows = payload.get("channels") or []
    return rows[0] if rows else None


def _answer(evidence_payload: dict, leader: dict | None, mode: str) -> dict:
    if leader is None:
        return {"refusal": "The tools returned no channels, so there is nothing to report."}
    items = evidence_payload.get("evidence") or []
    ids = [items[0]["evidence_id"]] if items else []
    rate = leader.get("rate_per_min")
    if mode == "liar":
        rate = 99.5                                # never retrieved by any tool
    text = (f"{leader.get('channel')} had the highest rate at {rate} events/min. "
            "A high rate is a measurement, not a seizure-onset zone.")
    return {"answer": text, "evidence_ids": ids}


class FakeOllama:
    def __init__(self, tag: str = "qwen2.5:7b-instruct", mode: str = "honest"):
        self.tag, self.mode = tag, mode
        self.calls: list[tuple] = []
        self.pulled: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - http.server API
                if self.path.startswith("/api/tags"):
                    return self._send(200, {"models": [{"name": fake.tag, "model": fake.tag}]})
                if self.path.startswith("/v1/models"):
                    return self._send(200, {"object": "list",
                                            "data": [{"id": fake.tag, "object": "model"}]})
                self._send(404, {"error": {"message": "not here"}})

            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
                if self.path.startswith("/api/pull"):
                    fake.pulled.append(str(request.get("name")))
                    return self._send(200, {"status": "success"})
                if not self.path.startswith("/v1/chat/completions"):
                    return self._send(404, {"error": {"message": "not here"}})
                messages = request.get("messages", [])
                names = {t["function"]["name"] for t in request.get("tools") or []}
                tool_results = []
                for m in messages:
                    if m.get("role") == "tool":
                        try:
                            tool_results.append(json.loads(m.get("content") or "{}"))
                        except ValueError:
                            tool_results.append({})
                fake.calls.append(("chat", len(messages), len(tool_results)))

                survey = next((r for r in tool_results if r.get("channels")), None)
                leader = _leader(survey) if survey else None
                evidence = next((r for r in tool_results
                                 if r.get("evidence") and leader
                                 and r.get("channel") == leader.get("channel")), None)
                if survey is None and "top_channels" in names:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "top_channels",
                                     "arguments": json.dumps({"k": 3})}}]}
                elif leader and evidence is None and "get_evidence" in names:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_2", "type": "function",
                        "function": {"name": "get_evidence",
                                     "arguments": json.dumps(
                                         {"channel": leader.get("channel", "")})}}]}
                else:
                    message = {"role": "assistant",
                               "content": json.dumps(_answer(evidence or {}, leader, fake.mode))}
                self._send(200, {"id": "chatcmpl-fake", "object": "chat.completion",
                                 "model": fake.tag,
                                 "choices": [{"index": 0, "message": message,
                                              "finish_reason": "stop"}]})

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_port
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        return False
