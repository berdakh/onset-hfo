"""The tools the agent may call -- and nothing else.

Each tool is a small, read-only query against a saved analysis
(:class:`onset_hfo.store.ResultStore`). There is no tool that runs a detector,
changes a threshold, writes a file, opens a URL or executes code. The agent's
entire universe is these eight functions over one patient's results.

Three rules are enforced here rather than in the prompt, because a prompt is a
request and a dispatcher is a rule:

* **Application-owned scope.** The patient is chosen by the application when
  the store is opened. No tool takes a subject argument, so no model output
  can change which patient is being discussed.
* **Strict arguments.** Unknown argument names, wrong types and out-of-range
  values are rejected before anything runs.
* **Data, not instructions.** Tool results are returned as JSON to the model.
  Text inside them (channel names, report sentences) is content to quote, and
  the system prompt tells the model to treat it that way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from onset_hfo.store import ResultStore

__all__ = ["Tool", "TOOLS", "tool_schemas", "dispatch", "ToolError"]


class ToolError(ValueError):
    """Raised when a tool call is malformed. Reported to the model as an error."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]

    def schema(self) -> dict:
        """OpenAI-compatible function schema (understood by Qwen, Llama, Mistral...)."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False}


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def _metadata(store: ResultStore) -> dict:
    return store.metadata()


def _list_channels(store: ResultStore, detector: str | None = None) -> dict:
    rows = store.list_channels(detector)
    return {"n_channels": len(rows), "channels": rows}


def _top_channels(store: ResultStore, detector: str | None = None, k: int = 5) -> dict:
    return {"detector": detector or store.detectors()[0],
            "channels": store.top_channels(detector, k)}


def _channel_summary(store: ResultStore, channel: str) -> dict:
    return store.channel_summary(channel)


def _get_evidence(store: ResultStore, channel: str, detector: str | None = None,
                  k: int = 3) -> dict:
    windows = store.evidence(channel, detector, k)
    return {"channel": channel, "n_windows": len(windows), "evidence": windows,
            "note": "times are seconds in the original recording"}


def _disagreements(store: ResultStore) -> dict:
    return {"disagreements": store.disagreements(),
            "agreement": store.detector_agreement(),
            "definition": "a channel is listed when the two detectors' ranks differ by at "
                          "least the configured number of places and at least one detector "
                          "puts it in the top group"}


def _rate_change(store: ResultStore) -> dict:
    rows = store.rate_change_table()
    return {"available": bool(rows), "rows": rows,
            "note": "event rate per channel before versus during the clinician-marked seizure"}


def _contact_map(store: ResultStore, detector: str | None = None) -> dict:
    return store.contact_map(detector)


def _report_section(store: ResultStore, section: str) -> dict:
    return {"section": section, "content": store.report_section(section)}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("get_recording_metadata",
         "What was analysed: subject, source dataset, sampling rate, analysed time window, "
         "seizure markers, detectors used, excluded channels. Call this first when the "
         "question is about the recording itself.",
         _obj({}), _metadata),
    Tool("list_channels",
         "Every analysed channel with its event count and rate for one detector, ordered by "
         "rate. Use when asked what was analysed or whether a channel exists.",
         _obj({"detector": {"type": "string", "enum": ["rms", "line_length", "spike"],
                            "description": "which detector's counts to list"}}),
         _list_channels),
    Tool("top_channels",
         "The channels with the highest event rate, with 95% confidence intervals. Use for "
         "'which channels stand out', 'highest rate', 'strongest evidence'.",
         _obj({"detector": {"type": "string", "enum": ["rms", "line_length", "spike"]},
               "k": {"type": "integer", "minimum": 1, "maximum": 20,
                     "description": "how many channels to return (default 5)"}}),
         _top_channels),
    Tool("channel_summary",
         "Everything known about one channel: each detector's rate, rank, confidence "
         "interval, mean frequency and amplitude, and how many of its HFOs coincide with an "
         "interictal discharge.",
         _obj({"channel": {"type": "string",
                           "description": "channel name exactly as analysed, e.g. 'AD1-AD2'"}},
              ["channel"]),
         _channel_summary),
    Tool("get_evidence",
         "The strongest detected events on a channel: time windows, peak frequency, spectral "
         "prominence and amplitude, each with an evidence_id. Every factual claim in an "
         "answer must cite one of these ids.",
         _obj({"channel": {"type": "string"},
               "detector": {"type": "string", "enum": ["rms", "line_length", "spike"]},
               "k": {"type": "integer", "minimum": 1, "maximum": 10}},
              ["channel"]),
         _get_evidence),
    Tool("detector_disagreements",
         "Channels the two HFO detectors rank very differently, plus their event-by-event "
         "agreement. Use for 'where do the detectors disagree' and to qualify any claim "
         "about a leading channel.",
         _obj({}), _disagreements),
    Tool("rate_change",
         "Event rate per channel before versus during the clinician-marked seizure, when the "
         "analysed window contains one.",
         _obj({}), _rate_change),
    Tool("contact_map",
         "Where the activity sits: each channel's electrode shaft, side and region with "
         "its rate, and how the leading channels distribute over shafts and sides. Use "
         "for 'where', 'which side', 'how many electrodes', 'neighbours', 'spread'. Says "
         "whether positions are measured or schematic; never mentions surgery.",
         _obj({"detector": {"type": "string", "enum": ["rms", "line_length", "spike"]}}),
         _contact_map),
    Tool("report_section",
         "One section of the structured report: summary, findings, disagreements, "
         "data_quality, methods, limitations, recording, rate_change. There is no "
         "recommendation section and there will never be one.",
         _obj({"section": {"type": "string",
                           "enum": ["summary", "findings", "disagreements", "data_quality",
                                    "methods", "limitations", "recording", "rate_change"]}},
              ["section"]),
         _report_section),
]}


def tool_schemas() -> list[dict]:
    """All tool schemas, for the model's ``tools`` parameter."""
    return [t.schema() for t in TOOLS.values()]


def dispatch(store: ResultStore, name: str, arguments: dict | None) -> Any:
    """Validate and run one tool call. Never evaluates model text as code."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(TOOLS)}")
    args = dict(arguments or {})
    schema = tool.parameters
    allowed = set(schema["properties"])
    unexpected = set(args) - allowed
    if unexpected:
        raise ToolError(f"{name} does not accept {', '.join(sorted(unexpected))}; "
                        f"allowed: {', '.join(sorted(allowed)) or 'no arguments'}")
    for key in schema.get("required", []):
        if key not in args:
            raise ToolError(f"{name} requires {key!r}")
    for key, value in list(args.items()):
        spec = schema["properties"][key]
        if spec["type"] == "string":
            if not isinstance(value, str):
                raise ToolError(f"{key} must be a string")
            if "enum" in spec and value not in spec["enum"]:
                raise ToolError(f"{key} must be one of {', '.join(spec['enum'])}")
        elif spec["type"] == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                raise ToolError(f"{key} must be an integer") from None
            value = max(spec.get("minimum", value), min(spec.get("maximum", value), value))
            args[key] = value
    return tool.handler(store, **args)
