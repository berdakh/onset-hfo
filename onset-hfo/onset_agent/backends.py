"""Model backends: where the open-weight model actually runs.

Four ways to run the same agent, all speaking the same small protocol
(messages in, assistant message with optional tool calls out):

===================  ===================================================
``scripted``         No model at all. A deterministic policy that emits
                     the same tool-call protocol, so the agent, the tools
                     and the guards can be tested and demonstrated
                     offline. It is NOT a language model and never claims
                     to be one.
``ollama``           A local Ollama server (``ollama serve``), default
                     model ``qwen2.5:7b-instruct``. Easiest local setup:
                     ``ollama pull qwen2.5:7b-instruct``.
``openai_compat``    Any OpenAI-compatible endpoint: vLLM, llama.cpp
                     ``--server``, LM Studio, Together, or a colleague's
                     GPU box. Open weights, your choice of host.
``transformers``     Hugging Face ``transformers`` in-process -- the one
                     that works in a Colab notebook with a GPU and no
                     server to start.
===================  ===================================================

All of them are open-weight models by default (Qwen2.5-Instruct). Nothing in
this project requires a hosted proprietary model; the API shape is
OpenAI-compatible only because that is what open-weight servers speak.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
from dataclasses import dataclass, field

__all__ = ["ToolCall", "AssistantMessage", "Backend", "ScriptedBackend", "OllamaBackend",
           "OpenAICompatBackend", "TransformersBackend", "make_backend"]

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = "call-1"


@dataclass
class AssistantMessage:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: str = ""


class Interrupted(RuntimeError):
    """The request in flight was abandoned on purpose, by `Backend.abort`."""


class Backend:
    """Interface every backend implements."""

    name = "backend"
    is_language_model = True

    def abort(self) -> None:
        """Stop whatever request is in flight, from another thread.

        The base backend has nothing to interrupt; a served one closes its
        connection, which is also how the server learns to stop generating.
        """

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


# --------------------------------------------------------------------------
# Parsing helpers (small models put tool calls in odd places)
# --------------------------------------------------------------------------

_TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Recover tool calls a model wrote into its message content.

    Open-weight models frequently emit ``<tool_call>{...}</tool_call>`` inside
    the content instead of using the structured field. Recovering them here
    turns a common failure into a non-event.
    """
    calls: list[ToolCall] = []
    for i, blob in enumerate(_TOOL_TAG.findall(text or "")):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        if isinstance(name, str):
            args = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {},
                                  id=f"text-call-{i + 1}"))
    return calls


def extract_json_object(text: str) -> dict | None:
    """Find the JSON object in a model's final message, tolerating chatter."""
    if not text:
        return None
    for candidate in (_FENCE.findall(text) or []):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


# --------------------------------------------------------------------------
# HTTP backends
# --------------------------------------------------------------------------


class OpenAICompatBackend(Backend):
    """Any server that speaks the OpenAI chat-completions API.

    Tested against Ollama and vLLM serving Qwen2.5-Instruct. ``api_key`` is
    optional: local servers ignore it.
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000/v1",
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 512, timeout: float = 180.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = f"{type(self).__name__.replace('Backend', '').lower()}:{model}"
        #: The open HTTP response while `chat` is reading one, so that
        #: `abort` can close it from the GUI thread.
        self._inflight = None
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True
        inflight = self._inflight
        if inflight is None:
            return
        # Closing the response is not enough: a `recv` blocked on another
        # thread keeps blocking on Linux until data or the timeout arrives.
        # Shutting the socket down is what makes it return now -- and what
        # the server sees as a disconnect, so it stops generating.
        import socket

        try:
            sock = inflight.sock                 # http.client's connection socket
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except Exception:       # noqa: BLE001 - best effort, any transport
            pass
        try:
            inflight.close()
        except Exception:       # noqa: BLE001
            pass

    def describe(self) -> str:
        return f"{self.name} at {self.base_url}"

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.aborted:
            raise Interrupted("the request was stopped before it was sent")
        # http.client rather than urlopen: the connection object exists from
        # before the request goes out, so `abort` has a socket to shut while a
        # slow model is still thinking about its first byte. urlopen hands
        # back nothing until the status line arrives, which is exactly the
        # stretch a reviewer wants to be able to end.
        import http.client
        import io
        import socket

        parts = urllib.parse.urlsplit(f"{self.base_url}/chat/completions")
        maker = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = maker(parts.hostname, parts.port, timeout=self.timeout)
        self._inflight = connection
        try:
            connection.request("POST", parts.path or "/", body=json.dumps(body).encode(),
                               headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 400:
                error = urllib.error.HTTPError(f"{self.base_url}/chat/completions",
                                               response.status, response.reason,
                                               response.headers, io.BytesIO(raw))
                # The server was reached and said no. Telling the person to
                # start one would be wrong; the reason is in the body it sent.
                raise RuntimeError(_describe_rejection(self.base_url, error)) from error
            payload = json.loads(raw.decode("utf-8", "replace"))
        except Interrupted:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            # A connection cut by `abort` surfaces here as a socket, protocol
            # or JSON error; either way it was asked for.
            if self.aborted:
                raise Interrupted("the request was stopped") from exc
            if isinstance(exc, (TimeoutError, socket.timeout)) or "timed out" in str(exc):
                # The server was there; the model had not written a byte in
                # the time allowed. On a CPU that is the prompt being read.
                raise RuntimeError(
                    f"The model at {self.base_url} did not answer within "
                    f"{self.timeout:.0f} s. On a CPU the model reads the whole prompt "
                    "before it writes, so a smaller size answers sooner; or Stop and ask "
                    "again, as the second question reuses what the first one read."
                ) from exc
            if isinstance(exc, (ConnectionError, OSError)) and not isinstance(exc, ValueError):
                raise RuntimeError(
                    f"Could not reach the model server at {self.base_url} ({exc}). "
                    "Start one (`ollama serve`, or vLLM), or use --backend scripted."
                ) from exc
            raise
        finally:
            self._inflight = None
            try:
                connection.close()
            except Exception:       # noqa: BLE001
                pass
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        calls = []
        for i, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args or {},
                                  id=call.get("id", f"call-{i + 1}")))
        if not calls:
            calls = parse_tool_calls_from_text(content)
        return AssistantMessage(content=content, tool_calls=calls, raw=json.dumps(message))



def _describe_rejection(base_url: str, exc) -> str:
    """Turn an HTTP error from an OpenAI-compatible server into a sentence.

    Found by serving a GGUF with llama.cpp: the agent's opening turn -- the
    system prompt plus eight tool schemas -- overran a 1024-token context, the
    server answered ``400 context_length_exceeded`` with a clear message, and
    the backend reported "could not reach the model server, start one". It had
    been reached. The body is read and quoted, and the one rejection a served
    model is likely to produce is given its remedy.
    """
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:
        raw = ""
    message, code = raw.strip()[:400], ""
    try:
        error = json.loads(raw).get("error", {})
        if isinstance(error, dict):
            message = str(error.get("message") or message)
            code = str(error.get("code") or error.get("type") or "")
    except (ValueError, AttributeError):
        pass
    text = (f"The model server at {base_url} rejected the request "
            f"(HTTP {exc.code}{', ' + code if code else ''}): {message}")
    if "context" in (code + message).lower():
        text += (" The agent's first turn is a few thousand tokens -- the system "
                 "prompt plus the tool schemas -- so the served model needs a "
                 "context window of at least 4096. Ollama: set "
                 "OLLAMA_CONTEXT_LENGTH=8192 before `ollama serve` (its default is "
                 "2048), or `PARAMETER num_ctx 8192` in a Modelfile. llama.cpp: "
                 "`-c 8192`. vLLM: --max-model-len.")
    return text

class OllamaBackend(OpenAICompatBackend):
    """A local Ollama server. ``ollama pull qwen2.5:7b-instruct`` and go.

    The default URL follows ``OLLAMA_HOST`` the way Ollama itself does, read
    when the backend is built rather than when this module was imported.
    Found by the desktop panel's own test: it *discovered* a server on a moved
    port through the same variable, opened on its model, and then tried to
    talk to 127.0.0.1:11434 -- finding the model and failing to reach it.
    """

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL,
                 base_url: str | None = None, **kwargs):
        if base_url is None:
            from onset_agent.hardware import ollama_url
            base_url = ollama_url().rstrip("/") + "/v1"
        super().__init__(model=model, base_url=base_url, **kwargs)


# --------------------------------------------------------------------------
# In-process backend (Colab)
# --------------------------------------------------------------------------


class TransformersBackend(Backend):
    """Run an open-weight model in this process with Hugging Face transformers.

    Notes for a notebook:

    * ``Qwen/Qwen2.5-7B-Instruct`` calls tools reliably and needs a GPU
      (about 16 GB, or ~6 GB in 4-bit).
    * ``Qwen/Qwen2.5-1.5B-Instruct`` (the default) runs on a free Colab
      instance and is noticeably worse at tool use. That is not a flaw to hide:
      the guards in :mod:`onset_agent.guard` exist because small models fail,
      and watching a 1.5B model get caught is the most useful thing a reader
      can see.
    * ``quantization`` is ``"fp16"``, ``"8bit"`` or ``"4bit"`` and is the
      axis :mod:`onset_agent.benchmark` sweeps. On a card without bfloat16 ---
      a T4, for instance --- pass ``dtype="float16"`` explicitly: most recent
      model configs declare bfloat16, and ``dtype="auto"`` will honour that on
      hardware that cannot run it.
    * ``revision`` pins the model to a commit. Leaving it at ``"main"`` means
      a re-run months later may not load the same weights, which is the kind
      of unrecorded difference this project exists to avoid.
    """

    #: What ``quantization=`` accepts. ``fp16`` loads the weights as the
    #: backend would anyway; the other two need ``bitsandbytes``.
    QUANTIZATIONS = ("fp16", "8bit", "4bit")

    def __init__(self, model_id: str = DEFAULT_HF_MODEL, max_new_tokens: int = 384,
                 device: str | None = None, dtype: str = "auto", load_in_4bit: bool = False,
                 load_in_8bit: bool = False, quantization: str | None = None,
                 revision: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if quantization is not None:
            if quantization not in self.QUANTIZATIONS:
                raise ValueError(f"quantization must be one of {self.QUANTIZATIONS}, "
                                 f"not {quantization!r}")
            load_in_4bit = quantization == "4bit"
            load_in_8bit = quantization == "8bit"
        if load_in_4bit and load_in_8bit:
            raise ValueError("choose one of 4-bit or 8-bit, not both")

        self.model_id = model_id
        self.revision = revision or "main"
        self.quantization = quantization or (
            "4bit" if load_in_4bit else "8bit" if load_in_8bit else "fp16")
        self.dtype = dtype
        self.name = f"transformers:{model_id}"
        self.max_new_tokens = max_new_tokens
        load_kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        kwargs: dict = {"dtype": dtype} if dtype != "auto" else {}
        if load_in_4bit or load_in_8bit:  # pragma: no cover - needs bitsandbytes + GPU
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device or "auto", **kwargs, **load_kwargs)

    def describe(self) -> str:
        return f"{self.name} (in-process)"

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        prompt = self.tokenizer.apply_chat_template(
            _to_template_messages(messages), tools=tools or None,
            tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        return AssistantMessage(content=text, tool_calls=parse_tool_calls_from_text(text),
                                raw=text)


def _to_template_messages(messages: list[dict]) -> list[dict]:
    """Convert our OpenAI-shaped history into what a chat template expects."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [{"type": "function",
                                        "function": {"name": c["function"]["name"],
                                                     "arguments": json.loads(c["function"]["arguments"])
                                                     if isinstance(c["function"]["arguments"], str)
                                                     else c["function"]["arguments"]}}
                                       for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "name": m.get("name", "tool"), "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


# --------------------------------------------------------------------------
# Offline scripted backend
# --------------------------------------------------------------------------


class ScriptedBackend(Backend):
    """A deterministic stand-in for a language model.

    It reads the question with a handful of keyword rules, calls one or two
    tools, and formats the result. It exists so that the agent, the tool
    dispatcher, the citation checks and the refusal rules can be exercised
    with no model, no GPU and no network -- in tests, in CI, and for anyone
    who wants to see the machinery before installing a 7B model.

    It is not a language model, it cannot generalise, and the CLI says so
    every time it runs.
    """

    name = "scripted (no language model)"
    is_language_model = False

    def describe(self) -> str:
        return "scripted policy - deterministic keyword rules, not a language model"

    def chat(self, messages: list[dict], tools: list[dict]) -> AssistantMessage:
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        observations = [m for m in messages if m["role"] == "tool"]
        q = (question or "").lower()

        if not observations:
            return AssistantMessage(tool_calls=[self._first_call(q)])
        if len(observations) == 1 and re.search(r"evidence|window|show me|why", q):
            data = json.loads(observations[0]["content"])
            channel = _first_channel(data)
            if channel:
                return AssistantMessage(tool_calls=[
                    ToolCall("get_evidence", {"channel": channel, "k": 2}, "call-2")])
        return AssistantMessage(content=json.dumps(self._answer(q, observations)))

    def _first_call(self, q: str) -> ToolCall:
        if re.search(r"disagree|differ|conflict", q):
            return ToolCall("detector_disagreements", {}, "call-1")
        if re.search(r"seizure|ictal|before|during|change over time", q):
            return ToolCall("rate_change", {}, "call-1")
        if re.search(r"limitation|caveat|weakness|trust", q):
            return ToolCall("report_section", {"section": "limitations"}, "call-1")
        if re.search(r"method|how did you|algorithm|detector work", q):
            return ToolCall("report_section", {"section": "methods"}, "call-1")
        if re.search(r"quality|artifact|rejected|excluded", q):
            return ToolCall("report_section", {"section": "data_quality"}, "call-1")
        if re.search(r"what (was|did you) analys|recording|sampling|metadata|how long", q):
            return ToolCall("get_recording_metadata", {}, "call-1")
        match = re.search(r"\b([A-Za-z]{1,6}\d{1,3}(?:-[A-Za-z]{1,6}\d{1,3})?)\b", q.upper())
        if match and re.search(r"channel|contact|about", q):
            return ToolCall("channel_summary", {"channel": match.group(1)}, "call-1")
        return ToolCall("top_channels", {"k": 5}, "call-1")

    def _answer(self, q: str, observations: list[dict]) -> dict:
        payloads = [json.loads(o["content"]) for o in observations]
        ids = []
        for payload in payloads:
            ids += _evidence_ids(payload)
        text = _render(payloads)
        return {"answer": text, "evidence_ids": ids[:4]}


def _first_channel(payload) -> str | None:
    if isinstance(payload, dict):
        if isinstance(payload.get("channels"), list) and payload["channels"]:
            return payload["channels"][0].get("channel")
        if isinstance(payload.get("channel"), str):
            return payload["channel"]
        if isinstance(payload.get("disagreements"), list) and payload["disagreements"]:
            return payload["disagreements"][0].get("channel")
        if isinstance(payload.get("rows"), list) and payload["rows"]:
            return payload["rows"][0].get("channel")
    return None


def _evidence_ids(payload) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "evidence_id" and isinstance(value, str):
                out.append(value)
            else:
                out += _evidence_ids(value)
    elif isinstance(payload, list):
        for item in payload:
            out += _evidence_ids(item)
    return out


def _render(payloads: list) -> str:
    """Format tool results as sentences. Every number is copied, never computed."""
    parts: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get("channels") and isinstance(payload["channels"], list):
            rows = payload["channels"][:3]
            det = payload.get("detector", "the detector")
            parts.append("Highest rates (" + det + "): " + "; ".join(
                f"{r['channel']} {r['rate_per_min']}/min (n={r['n_events']})" for r in rows) + ".")
        if payload.get("evidence"):
            rows = payload["evidence"][:2]
            parts.append("Evidence on " + str(payload.get("channel", "")) + ": " + "; ".join(
                f"{r['start']}-{r['stop']} s, peak {r.get('peak_frequency_hz')} Hz"
                for r in rows) + ".")
        if "disagreements" in payload:
            rows = payload["disagreements"]
            parts.append(f"The two detectors rank {len(rows)} leading channel(s) very "
                         "differently" + (": " + ", ".join(r["channel"] for r in rows[:3]) + "."
                                          if rows else "."))
        if payload.get("rows") and "before" in json.dumps(payload)[:2000]:
            rows = payload["rows"][:2]
            parts.append("Rate before vs during the marked seizure: " + "; ".join(
                f"{r['channel']} {r.get('rate_before')}/min -> {r.get('rate_during')}/min"
                for r in rows) + ".")
        if payload.get("section"):
            content = payload["content"]
            if isinstance(content, list):
                items = [c if isinstance(c, str) else str(c.get("channel", "")) for c in content[:3]]
                parts.append(f"{payload['section']}: " + "; ".join(items) + ".")
            else:
                parts.append(f"{payload['section']}: {content}")
        if payload.get("subject") and payload.get("sampling_rate_hz"):
            parts.append(f"Analysis of {payload['subject']} ({payload.get('source')}), "
                         f"{payload.get('duration_s')} s at {payload['sampling_rate_hz']} Hz, "
                         f"{payload.get('channels_analysed')} channels.")
    if not parts:
        return "The tools returned nothing that answers that."
    parts.append("Rates are measurements, not a seizure-onset zone.")
    return " ".join(parts)


def make_backend(kind: str = "scripted", model: str | None = None, base_url: str | None = None,
                 **kwargs) -> Backend:
    """Build a backend by name: scripted, ollama, openai_compat, transformers."""
    kind = (kind or "scripted").lower()
    if kind == "scripted":
        return ScriptedBackend()
    if kind == "ollama":
        return OllamaBackend(model=model or DEFAULT_OLLAMA_MODEL,
                             **({"base_url": base_url} if base_url else {}), **kwargs)
    if kind in ("openai_compat", "openai", "vllm", "llamacpp"):
        if not model:
            raise ValueError("--model is required for an OpenAI-compatible server")
        return OpenAICompatBackend(model=model,
                                   base_url=base_url or "http://127.0.0.1:8000/v1", **kwargs)
    if kind in ("transformers", "hf"):
        return TransformersBackend(model_id=model or DEFAULT_HF_MODEL, **kwargs)
    raise ValueError(f"Unknown backend {kind!r}: choose scripted, ollama, openai_compat "
                     "or transformers")
