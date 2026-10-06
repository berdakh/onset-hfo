"""What the window adds to the assistant's tool set, bound to its session.

The agent's own tools read a saved table. The window has more: the recording
itself, so a detector can be re-run at another threshold; and the picture
behind the selected event, so "is this one real" can be answered from the
measurements the picture is made of. Both are offered here as tools in the
same shape as the agent's, under the same validation and the same number
check, and the analyses only when the reviewer has ticked the box that
allows them, because each one costs seconds on a CPU.

Nothing here changes the window's analysis. The analyses run on a copy of
the recording through :class:`onset_agent.analysis.AnalysisSession`; the
window's own numbers are the saved table's, as before.
"""

from __future__ import annotations

from onset_agent.tools import Tool, ToolError

__all__ = ["analysis_tools", "explain_tools", "ANALYSES", "COST"]

#: The analyses a reviewer may let the assistant run, and what each costs.
#: The research registry has more; its seizure-onset probability estimate is
#: left out on purpose, as the guard refuses that question before any model.
ANALYSES = ("detect_hfo", "detect_spikes", "spectral_power", "compare_detectors",
            "propagation_lead", "channel_qc")
COST = {
    "detect_hfo": "re-runs the detector on the window: a few seconds the first time, "
                  "under a second at another threshold",
    "detect_spikes": "re-runs the discharge detector: a few seconds",
    "spectral_power": "a spectrum per named channel: under a second",
    "compare_detectors": "runs both HFO detectors: several seconds",
    "propagation_lead": "re-runs the detector and times the first event per channel: "
                        "a few seconds",
    "channel_qc": "the quality measures per contact: under a second",
}


def analysis_tools(session) -> dict[str, Tool]:
    """The analysis registry, as agent tools over this window's recording.

    The analysis session is built on first use, so a reviewer who ticks the
    box and never asks for a run pays nothing.
    """
    recording = getattr(session, "recording", None)
    if recording is None:
        return {}
    from onset_agent.analysis import ANALYSIS_TOOLS, AnalysisSession, build_registry

    holder: dict = {}

    def registry():
        if "registry" not in holder:
            analysis = AnalysisSession(recording, session.request.pipeline_config(),
                                       verbose=False)
            holder["registry"] = build_registry(analysis)
        return holder["registry"]

    tools: dict[str, Tool] = {}
    for spec in ANALYSIS_TOOLS:
        if spec.name not in ANALYSES:
            continue

        def handler(_store, _name=spec.name, **arguments):
            run = registry().run(_name, arguments)
            if not run.ok:
                raise ToolError(run.error or "the analysis failed")
            output = dict(run.output) if isinstance(run.output, dict) else {"result": run.output}
            output["run_id"] = run.run_id
            output["runtime_s"] = round(float(run.runtime_s), 2)
            return output

        tools[spec.name] = Tool(spec.name,
                                f"{spec.description} Runs on a copy of this window's recording "
                                f"({COST.get(spec.name, 'seconds')}).",
                                spec.parameters, handler)
    return tools


def nearest_event(session, channel: str, start: float, within: float = 0.05):
    """The event on `channel` whose onset is nearest `start` (file seconds)."""
    best, best_distance = None, float(within)
    for event in getattr(session, "events", []) or []:
        if str(event.channel) != str(channel):
            continue
        distance = abs(float(event.start) - float(start))
        if distance <= best_distance:
            best, best_distance = event, distance
    return best


def explain_tools(session) -> dict[str, Tool]:
    """One tool: the selected event's own measurements and what they read as."""
    def handler(_store, channel: str, start: float):
        from onset_review import detail

        event = nearest_event(session, channel, start)
        if event is None:
            raise ToolError(f"no event on {channel} at {float(start):.3f} s in this window")
        return detail.read_event(detail.snapshot(session, event))

    return {"explain_event": Tool(
        "explain_event",
        "Why the event at (channel, start seconds) reads as a real oscillation or as "
        "filter ringing: its duration in cycles, peak frequency, how much more energy "
        "it has inside the band than outside it at the same moment, and the reading "
        "those give (island, column or unclear). Use for 'is this event real', 'why was "
        "this marked', 'is this ringing'.",
        {"type": "object",
         "properties": {"channel": {"type": "string"},
                        "start": {"type": "number",
                                  "description": "onset in seconds of the original recording"}},
         "required": ["channel", "start"], "additionalProperties": False},
        handler)}
