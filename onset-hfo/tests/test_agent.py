"""The agent half: tools, guards, backends and the loop.

No model is downloaded here. The language-model path is exercised against a
local mock server that replies exactly as an OpenAI-compatible open-weight
server does -- including the two ways small models get it wrong (tool calls
buried in the message content, and confident invented numbers).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from onset_agent.agent import OnsetAgent
from onset_agent.backends import (
    AssistantMessage,
    Backend,
    OpenAICompatBackend,
    ScriptedBackend,
    ToolCall,
    extract_json_object,
    parse_tool_calls_from_text,
)
from onset_agent.guard import check_question, verify_answer
from onset_agent.tools import TOOLS, ToolError, dispatch, tool_schemas

# -- tools -----------------------------------------------------------------

def test_every_tool_has_a_strict_schema():
    for name, tool in TOOLS.items():
        schema = tool.schema()["function"]
        assert schema["name"] == name
        assert schema["description"].strip()
        assert schema["parameters"]["additionalProperties"] is False


def test_no_tool_can_change_the_patient():
    """Patient scope is owned by the application, not by model output."""
    for tool in TOOLS.values():
        assert "subject" not in tool.parameters["properties"]
        assert "patient" not in tool.parameters["properties"]


def test_dispatch_rejects_bad_calls(store):
    with pytest.raises(ToolError):
        dispatch(store, "delete_everything", {})
    with pytest.raises(ToolError):
        dispatch(store, "top_channels", {"subject": "sub-pt99"})
    with pytest.raises(ToolError):
        dispatch(store, "channel_summary", {})
    with pytest.raises(ToolError):
        dispatch(store, "top_channels", {"detector": "magic"})


def test_dispatch_clamps_numbers(store):
    out = dispatch(store, "top_channels", {"k": 500})
    assert len(out["channels"]) <= 20


def test_unknown_channel_returns_an_error_not_an_exception(store):
    out = dispatch(store, "channel_summary", {"channel": "NOPE1-NOPE2"})
    assert "error" in out and "analysed_channels" in out


def test_evidence_ids_resolve(store):
    channel = store.top_channels(k=1)[0]["channel"]
    windows = dispatch(store, "get_evidence", {"channel": channel, "k": 2})["evidence"]
    assert windows
    for window in windows:
        assert store.resolve(window["evidence_id"]) is not None
    assert store.resolve("sub-xx|FAKE|rms|1.000") is None


def test_evidence_excludes_spikes_unless_asked(store):
    channel = store.top_channels(k=1)[0]["channel"]
    assert all(w["detector"] != "spike" for w in store.evidence(channel, k=5))
    spikes = store.evidence(channel, detector="spike", k=5)
    assert all(w["detector"] == "spike" for w in spikes)


def test_report_has_no_recommendation_section(store):
    out = dispatch(store, "report_section", {"section": "limitations"})
    assert out["content"]
    with pytest.raises(ToolError):
        dispatch(store, "report_section", {"section": "recommendation"})


# -- guards ----------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "Which region should we resect?",
    "What treatment do you recommend?",
    "Should we ablate the temporal contacts?",
    "What would you do in this case?",
])
def test_treatment_questions_are_refused_before_the_model_runs(question):
    assert not check_question(question, "sub-pt01").ok


@pytest.mark.parametrize("question", [
    "Does this patient have epilepsy?",
    "Where do the seizures start?",
    "Is this the seizure onset zone?",
    "Will the patient be seizure free?",
])
def test_diagnostic_questions_are_refused(question):
    assert not check_question(question, "sub-pt01").ok


def test_other_patients_are_refused():
    assert not check_question("What about patient sub-pt02?", "sub-pt01").ok
    assert check_question("What about ATT1-ATT2?", "sub-pt01").ok


def test_invented_numbers_are_caught():
    seen = {46.0, 31.0}
    assert verify_answer("The rate is 46.0/min.", [], set(), seen).ok
    assert not verify_answer("The rate is 91.5/min.", [], set(), seen).ok


def test_a_time_range_is_not_read_as_a_negative_number():
    seen = {45.898, 45.93}
    assert verify_answer("Window 45.898-45.93 s.", [], set(), seen).ok


def test_uncited_evidence_is_caught():
    assert not verify_answer("see evidence", ["made|up|id|1.0"], set(), {1.0}).ok
    assert verify_answer("see evidence", ["real|id|rms|1.0"], {"real|id|rms|1.0"}, {1.0}).ok


# -- parsing helpers --------------------------------------------------------

def test_tool_calls_hidden_in_content_are_recovered():
    text = 'Sure.\n<tool_call>\n{"name": "top_channels", "arguments": {"k": 3}}\n</tool_call>'
    calls = parse_tool_calls_from_text(text)
    assert calls[0].name == "top_channels" and calls[0].arguments == {"k": 3}


def test_json_is_extracted_from_chatty_output():
    assert extract_json_object('Here you go: {"answer": "x", "evidence_ids": []} thanks') \
        == {"answer": "x", "evidence_ids": []}
    assert extract_json_object("no json here") is None


# -- the loop with the scripted policy --------------------------------------

def test_scripted_agent_answers_with_tools(store):
    agent = OnsetAgent(store, ScriptedBackend())
    answer = agent.ask("Which channels have the highest ripple rate?")
    assert not answer.refused
    assert answer.tools_called == ["top_channels"]
    assert answer.verified


def test_scripted_agent_cites_evidence(store):
    agent = OnsetAgent(store, ScriptedBackend())
    answer = agent.ask("What is the evidence for the top channel?")
    assert not answer.refused
    assert answer.evidence_ids
    assert all(store.resolve(eid) for eid in answer.evidence_ids)


def test_agent_refuses_treatment_without_calling_a_tool(store):
    agent = OnsetAgent(store, ScriptedBackend())
    answer = agent.ask("Which contacts should we resect?")
    assert answer.refused and not answer.tools_called
    assert "treatment" in answer.text.lower() or "should be done" in answer.text.lower()


# -- the language-model path, against a mock open-weight server -------------

class _MockHandler(BaseHTTPRequestHandler):
    replies: list[dict] = []
    index = 0

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        message = type(self).replies[min(type(self).index, len(type(self).replies) - 1)]
        status = 200
        if isinstance(message, dict) and "__status__" in message:
            message = dict(message)
            status = int(message.pop("__status__"))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(message).encode())
            return
        type(self).index += 1
        body = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the test output
        return


@pytest.fixture
def mock_server():
    def start(replies):
        _MockHandler.replies = replies
        _MockHandler.index = 0
        server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_port}/v1"
    servers = []
    yield lambda replies: (lambda pair: (servers.append(pair[0]), pair[1])[1])(start(replies))
    for server in servers:
        server.shutdown()


def test_openai_compatible_tool_calling(store, mock_server):
    """The path a real Qwen/Llama server takes: structured tool call, then JSON."""
    url = mock_server([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "top_channels", "arguments": json.dumps({"k": 2})}}]},
        {"role": "assistant",
         "content": json.dumps({"answer": "The leading channel is listed above.",
                                "evidence_ids": []})},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Which channels stand out?")
    assert not answer.refused
    assert answer.tools_called == ["top_channels"]


def test_tool_call_written_into_content_still_works(store, mock_server):
    url = mock_server([
        {"role": "assistant",
         "content": '<tool_call>{"name": "report_section", '
                    '"arguments": {"section": "summary"}}</tool_call>'},
        {"role": "assistant", "content": json.dumps({"answer": "Summarised.",
                                                     "evidence_ids": []})},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Summarise the analysis.")
    assert answer.tools_called == ["report_section"]


def test_a_hallucinated_number_is_refused(store, mock_server):
    """The point of the whole guard layer, tested end to end."""
    url = mock_server([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "top_channels", "arguments": "{}"}}]},
        {"role": "assistant", "content": json.dumps(
            {"answer": "The top channel fires at 999.9 events/min.", "evidence_ids": []})},
        {"role": "assistant", "content": json.dumps(
            {"answer": "The top channel fires at 999.9 events/min.", "evidence_ids": []})},
        {"role": "assistant", "content": json.dumps(
            {"answer": "The top channel fires at 999.9 events/min.", "evidence_ids": []})},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("How often does the top channel fire?")
    assert answer.refused and not answer.verified
    assert "999.9" not in answer.text


def test_a_fabricated_citation_with_nothing_real_behind_it_is_refused(store, mock_server):
    """The answer names no channel, so there is no retrieved window to put in
    place of the invented id -- and an answer whose only citation is to
    nothing is refused for it."""
    url = mock_server([
        {"role": "assistant", "content": json.dumps(
            {"answer": "See the evidence.", "evidence_ids": ["sub-zz|FAKE|rms|0.000"]})},
    ] * 4)
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Show me the evidence for the top channel.")
    assert answer.refused
    assert any("never returned" in p for e in answer.trace
               if e.get("type") == "answer" for p in e["problems"])


def test_a_placeholder_citation_is_replaced_by_the_real_evidence(store, mock_server):
    """Seen with a 7B on a CPU: the model copied the contract's "<id>"
    placeholder as `evidence_id_1`. Its numbers were right. The placeholder
    is dropped, the retrieved windows of the channel it named are attached,
    and the trace says which was which."""
    top = store.top_channels(None, 1)[0]
    url = mock_server([
        {"role": "assistant", "content": json.dumps(
            {"answer": f"{top['channel']} stands out with {top['rate_per_min']} events "
                       f"per minute.",
             "evidence_ids": ["evidence_id_1", "evidence_id_2"]})},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Does any channel actually stand out?")
    assert not answer.refused, answer.trace
    assert answer.evidence_ids and all(top["channel"] in i for i in answer.evidence_ids)
    assert answer.attached_citations == answer.evidence_ids
    cites = [e for e in answer.trace if e["type"] == "citations"][0]
    assert cites["dropped"] == ["evidence_id_1", "evidence_id_2"]


def test_a_prose_answer_is_checked_as_written(store, mock_server):
    """A small model writes the sentence instead of the JSON. The sentence is
    the answer; it goes through the same number check, and passes when the
    numbers are the tools' -- with no retry round-trip spent on the format."""
    top = store.top_channels(None, 1)[0]
    url = mock_server([
        {"role": "assistant",
         "content": f"The channel {top['channel']} has the highest rate at "
                    f"{top['rate_per_min']} events per minute (95% CI: "
                    f"{top['rate_ci_95'][0]} to {top['rate_ci_95'][1]}). "
                    "证据_id：[\"evidence-001\"]"},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Which channels have the highest ripple rate?")
    assert not answer.refused, answer.trace
    assert answer.text.startswith("The channel") and "证据" not in answer.text
    assert [e["type"] for e in answer.trace if e["type"] in ("prose", "format_error")] == ["prose"]
    assert len(answer.tools_called) == 0 and answer.looked_at, "the briefing did the fetching"


def test_a_prose_answer_with_an_invented_number_is_still_refused(store, mock_server):
    url = mock_server([
        {"role": "assistant", "content": "The top channel fires at 999.9 events per minute."},
    ] * 4)
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("How often does the top channel fire?")
    assert answer.refused and "999.9" not in answer.text


def test_the_confidence_level_is_a_number_the_tools_returned(store):
    """"95 %" was flagged as invented: the tool says it as the key
    `rate_ci_95`, and keys are part of the result."""
    from onset_agent.guard import collect_numbers

    seen = collect_numbers(dispatch(store, "top_channels", {"k": 1}))
    assert 95.0 in seen
    assert verify_answer("The 95 % interval is wide.", [], set(), seen).ok


def test_the_briefing_runs_before_the_model_and_is_marked(store, mock_server):
    seen_by_model = []

    class Recorder(BaseHTTPRequestHandler):
        pass

    url = mock_server([
        {"role": "assistant", "content": json.dumps({"answer": "Read above.", "evidence_ids": []})},
    ])
    events = []
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("What was analysed?", on_event=events.append)
    briefed = [e for e in answer.trace if e.get("briefing")]
    assert [e["tool"] for e in briefed][:2] == ["get_recording_metadata", "top_channels"]
    assert any(e["tool"] == "get_evidence" for e in briefed)
    assert all("digest" in e for e in briefed)
    assert answer.tools_called == [] and answer.looked_at
    # Every trace entry was reported as it happened, in order.
    assert [e["type"] for e in events] == [e["type"] for e in answer.trace]
    del seen_by_model, Recorder


def test_the_briefing_fetches_a_channel_named_in_the_question(store):
    from onset_agent.agent import briefing

    channel = store.channels()[0]
    calls = briefing(store, f"Tell me about {channel.lower()} and its seizure rate")
    assert ("channel_summary", {"channel": channel}) in calls
    assert ("get_evidence", {"channel": channel, "k": 2}) in calls
    assert ("rate_change", {}) in calls
    assert ("report_section", {"section": "limitations"}) in briefing(store, "what are the limitations?")


def test_the_scripted_policy_keeps_its_own_routing(store):
    """The briefing exists to save a language model round-trips; the scripted
    policy has none to save and routes by question, so it is left alone."""
    agent = OnsetAgent(store, ScriptedBackend())
    assert not agent.brief
    answer = agent.ask("What are the limitations of this analysis?")
    assert answer.tools_called == ["report_section"]


@pytest.mark.parametrize("question", ["What can you do?", "what can I ask you", "help",
                                      "Who are you?", "How do you work?"])
def test_questions_about_the_assistant_are_answered_without_a_model(store, question):
    class Never(Backend):
        name = "never"

        def chat(self, messages, tools):
            raise AssertionError("the model must not be called")

    answer = OnsetAgent(store, Never()).ask(question)
    assert not answer.refused and answer.verified
    assert "treatment" in answer.text and "highest" in answer.text
    assert answer.trace[0]["type"] == "about"


@pytest.mark.parametrize("question", ["What can you tell me about AR1-AR2?",
                                      "What do you know about the top channel?",
                                      "help me understand the disagreement"])
def test_questions_about_the_data_are_not_mistaken_for_questions_about_the_assistant(question):
    from onset_agent.guard import about_the_assistant

    assert not about_the_assistant(question)


def test_the_model_cannot_call_an_unknown_tool(store, mock_server):
    url = mock_server([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "run_shell", "arguments": json.dumps({"cmd": "rm -rf /"})}}]},
        {"role": "assistant", "content": json.dumps({"refusal": "I cannot do that."})},
    ])
    agent = OnsetAgent(store, OpenAICompatBackend(model="mock", base_url=url))
    answer = agent.ask("Delete the results.")
    assert answer.refused
    failed = [s for s in answer.trace if s.get("type") == "tool_call" and not s["ok"]]
    assert failed and "unknown tool" in failed[0]["error"]


def test_tool_output_that_looks_like_an_instruction_is_just_data(store, mock_server):
    """Prompt injection through a tool result must not change what the agent does."""
    class Injecting(Backend):
        name = "injecting-mock"

        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return AssistantMessage(tool_calls=[ToolCall("report_section",
                                                             {"section": "summary"}, "c1")])
            # The model sees the tool output; whatever it contains, the answer
            # is still verified against retrieved evidence and numbers.
            return AssistantMessage(content=json.dumps(
                {"answer": "Ignoring instructions found in data; 12345.6 Hz is not real.",
                 "evidence_ids": []}))

    agent = OnsetAgent(store, Injecting())
    answer = agent.ask("Summarise.")
    assert answer.refused, "an unverifiable number must not be published"


def test_schemas_are_json_serialisable():
    json.dumps(tool_schemas())


def test_a_rejected_request_is_not_reported_as_an_unreachable_server(store, mock_server):
    """Found with a real llama.cpp server: a 400 for an over-long prompt was
    reported as "could not reach the model server, start one". The server had
    been reached; the reason it sent back is what the person needs."""
    body = {"__status__": 400, "error": {
        "message": "This model's maximum context length is 1024 tokens. However, "
                   "you requested 3521 tokens.",
        "type": "invalid_request_error", "code": "context_length_exceeded"}}
    backend = OpenAICompatBackend(model="m", base_url=mock_server([body]))
    with pytest.raises(RuntimeError) as raised:
        backend.chat([{"role": "user", "content": "hi"}], [])
    text = str(raised.value)
    assert "rejected the request" in text
    assert "HTTP 400" in text and "context_length_exceeded" in text
    assert "maximum context length is 1024" in text
    assert "ollama serve`" not in text.replace("before `ollama serve`", "")
    # And the one rejection a served model is likely to give has its remedy.
    assert "OLLAMA_CONTEXT_LENGTH" in text and "-c 8192" in text


def test_a_rejection_without_a_json_body_still_shows_the_status(store, mock_server):
    backend = OpenAICompatBackend(model="m", base_url=mock_server([
        {"__status__": 503, "detail": "model is loading"}]))
    with pytest.raises(RuntimeError) as raised:
        backend.chat([{"role": "user", "content": "hi"}], [])
    assert "HTTP 503" in str(raised.value)
    assert "model is loading" in str(raised.value)


def test_the_ollama_backend_follows_ollama_host(monkeypatch):
    """Found by the panel's end-to-end test: the probe followed OLLAMA_HOST to
    a server on another port, the panel opened on its model, and the backend
    then dialled the hard-coded default and could not reach it."""
    from onset_agent.backends import OllamaBackend

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaBackend().base_url == "http://127.0.0.1:11434/v1"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:43210")
    assert OllamaBackend().base_url == "http://127.0.0.1:43210/v1"
    # An explicit URL still wins.
    assert OllamaBackend(base_url="http://box:1/v1").base_url == "http://box:1/v1"


# -- stopping ------------------------------------------------------------------


def test_a_stop_between_steps_ends_the_loop_with_a_refusal_that_says_so(store):
    agent = OnsetAgent(store, ScriptedBackend())
    answer = agent.ask("Which channel had the highest ripple rate?",
                       should_stop=lambda: True)
    assert answer.refused and answer.reason == "stopped by the reviewer"
    assert answer.trace and answer.trace[-1]["type"] == "stopped"
    assert answer.tools_called == [], "nothing ran after the stop"


def test_a_stop_after_the_first_step_keeps_what_ran_in_the_trace(store):
    polls = []

    def should_stop():
        polls.append(1)
        return len(polls) > 1       # let one step run, then stop

    answer = OnsetAgent(store, ScriptedBackend()).ask(
        "Which channel had the highest ripple rate?", should_stop=should_stop)
    assert answer.refused and answer.reason == "stopped by the reviewer"
    assert answer.tools_called, "the first step's tool call is on record"


def test_an_unverified_answer_keeps_what_the_model_wrote_in_the_trace(store):
    """The window explains a refusal from the trace, so the trace must carry
    the model's words and the checks they failed."""
    from onset_agent.agent import AgentAnswer

    trace = [{"type": "tool_call", "tool": "top_channels", "ok": True},
             {"type": "answer", "step": 1, "verified": False,
              "problems": ["the value '99.5 events/min' does not appear in any tool result"],
              "text": "AR1-AR2 had 99.5 events/min.", "evidence_ids": []},
             {"type": "gave_up", "steps": 6, "retries": 2}]
    answer = AgentAnswer(question="q", text="refused", refused=True,
                         reason="verification failed", trace=trace)
    assert [e for e in answer.trace if e["type"] == "answer"][0]["text"].startswith("AR1-AR2")


def test_the_briefed_first_call_is_small_and_cache_friendly(store):
    """On a CPU the model reads every prompt token before it writes one, so
    the first call carries no tool schemas, the constant part (system prompt,
    base briefing) comes before the question, and the question-specific
    retrievals come after it. Tools are offered from the second step on."""
    seen = []

    class Spy(Backend):
        name = "spy"

        def chat(self, messages, tools):
            seen.append((list(messages), list(tools)))
            if len(seen) == 1:
                return AssistantMessage(content="The top channel fires at 999.9 events per minute.")
            return AssistantMessage(content=json.dumps({"refusal": "enough"}))

    agent = OnsetAgent(store, Spy(), max_retries=1)
    channel = store.channels()[0]
    agent.ask(f"What are the limitations for {channel}?")
    first, second = seen
    assert first[1] == [] and second[1], "no schemas on the first call, schemas on the second"
    roles = [m["role"] for m in first[0]]
    question_at = roles.index("user")
    assert roles[0] == "system" and roles[1] == "assistant" and roles[2] == "tool"
    assert set(roles[1:question_at]) == {"assistant", "tool"}, "the base briefing precedes the question"
    names_before = [m["name"] for m in first[0][:question_at] if m["role"] == "tool"]
    names_after = [m["name"] for m in first[0][question_at:] if m["role"] == "tool"]
    assert names_before[:2] == ["get_recording_metadata", "top_channels"]
    assert names_before.count("get_evidence") == 2 and "detector_disagreements" in names_before
    assert "channel_summary" in names_after and "report_section" in names_after
    assert "Answer from them" in first[0][0]["content"]
    assert "Call one of" in second[0][0]["content"]
    # The whole first request of a plain question, system prompt and all, is
    # small enough to read on a CPU: it was about 3,000 tokens before.
    seen.clear()
    agent.ask("Which channels have the highest ripple rate?")
    plain = len(json.dumps(seen[0][0])) // 4
    assert plain < 1700, plain


# -- tools the caller adds, memory, and named briefing queries ------------------


def test_extra_tools_are_offered_at_once_for_a_question_that_needs_a_run(store):
    """The window adds analysis tools; a question with "stricter threshold" in
    it gets the schemas on the first call, so the model can ask for the run
    without a round-trip; a plain question still gets none."""
    from onset_agent.tools import Tool

    calls = []

    def echo(_store, threshold_sd: float = 3.0):
        calls.append(threshold_sd)
        return {"threshold_sd": threshold_sd, "rate_per_min": 7.0}

    extra = {"rerun": Tool("rerun", "re-run the detector", {
        "type": "object", "properties": {"threshold_sd": {"type": "number", "minimum": 1.0,
                                                          "maximum": 12.0}},
        "required": [], "additionalProperties": False}, echo)}
    seen = []

    class Spy(Backend):
        name = "spy"

        def chat(self, messages, tools):
            seen.append([t["function"]["name"] for t in tools])
            if len(seen) == 1:
                return AssistantMessage(tool_calls=[ToolCall("rerun", {"threshold_sd": 5}, "c1")])
            return AssistantMessage(content=json.dumps(
                {"answer": "At 5 SD the rate is 7.0 events per minute.", "evidence_ids": []}))

    agent = OnsetAgent(store, Spy(), extra_tools=extra)
    assert "rerun" in agent.tools and "rerun" in agent._system(tools_offered=True)
    answer = agent.ask("Does the leader survive a stricter threshold?")
    assert not answer.refused and calls == [5.0], "the extra tool ran, with a validated number"
    assert "rerun" in seen[0], "schemas on the first call for an analysis question"
    ran = [e for e in answer.trace if e["type"] == "tool_call" and e["tool"] == "rerun"][0]
    assert ran["analysis"] is True and "seconds" in ran
    seen.clear()
    OnsetAgent(store, Spy(), extra_tools=extra).ask("Which channels have the highest rate?")
    assert seen[0] == [], "a plain question still reads the briefing first"


def test_the_conversation_sits_between_the_briefing_and_the_question(store):
    seen = []

    class Spy(Backend):
        name = "spy"

        def chat(self, messages, tools):
            seen.append(list(messages))
            return AssistantMessage(content=json.dumps({"answer": "ok", "evidence_ids": []}))

    OnsetAgent(store, Spy()).ask("And the second one?",
                                 history=[("Which channel leads?", "AR1-AR2 leads.")])
    roles = [m["role"] for m in seen[0]]
    first_user = roles.index("user")
    assert roles[first_user:first_user + 3] == ["user", "assistant", "user"]
    assert seen[0][first_user]["content"] == "Which channel leads?"
    assert seen[0][first_user + 1]["content"] == "AR1-AR2 leads."
    assert seen[0][first_user + 2]["content"] == "And the second one?"
    assert all(r in ("assistant", "tool") for r in roles[1:first_user]), "briefing before it"


def test_the_scripted_policy_answers_the_latest_question_not_the_first(store):
    answer = OnsetAgent(store, ScriptedBackend()).ask(
        "What are the limitations?",
        history=[("Which channels have the highest ripple rate?", "AR1-AR2.")])
    assert answer.tools_called == ["report_section"]


def test_a_named_briefing_query_runs_whatever_the_backend(store):
    answer = OnsetAgent(store, ScriptedBackend()).ask(
        "Tell me about it", extra_briefing=[("get_recording_metadata", {})])
    briefed = [e for e in answer.trace if e.get("briefing")]
    assert [e["tool"] for e in briefed] == ["get_recording_metadata"]


# --------------------------------------------------------------------------
# Only the relevant schemas on the first call
# --------------------------------------------------------------------------

def _named_tools(*names):
    from onset_agent.tools import TOOLS, Tool

    out = dict(TOOLS)
    for name in names:
        out[name] = Tool(name, f"{name} description", {"type": "object", "properties": {}},
                         lambda _store, **_a: {"ok": True})
    return out


def test_the_first_call_offers_only_the_tools_the_question_is_about():
    from onset_agent.agent import _CORE_TOOLS, relevant_tools

    tools = _named_tools("threshold_sensitivity", "detect_hfo", "spectral_power",
                         "other_windows", "compare_window", "explain_event", "channel_qc")
    offered = relevant_tools("Does AR1-AR2 survive a stricter threshold?", tools)
    assert set(offered) == set(_CORE_TOOLS) | {"threshold_sensitivity", "detect_hfo"}
    offered = relevant_tools("Did the leader change between the two minutes?", tools)
    assert {"other_windows", "compare_window"} <= set(offered)
    assert "spectral_power" not in offered
    assert set(relevant_tools("Is AHR6-AHR7 oscillating or just noisy?", tools)) \
        == set(_CORE_TOOLS) | {"spectral_power", "channel_qc"}
    # About nothing in particular: everything.
    assert relevant_tools("Tell me about this recording.", tools) == tools
    # About a tool that is not on offer: the core alone, the briefing answers.
    fewer = _named_tools("explain_event")
    assert set(relevant_tools("Does it survive a stricter threshold?", fewer)) == set(_CORE_TOOLS)


def test_a_threshold_question_sends_a_short_schema_list_first_and_everything_after(store):
    import json

    from onset_agent.agent import OnsetAgent
    from onset_agent.backends import AssistantMessage, Backend, ToolCall

    class Recorder(Backend):
        name = "recorder"
        is_language_model = True

        def __init__(self):
            self.offered = []

        def describe(self):
            return "recorder"

        def chat(self, messages, tools):
            self.offered.append([t["function"]["name"] for t in (tools or [])])
            if len(self.offered) == 1:
                return AssistantMessage(tool_calls=[ToolCall("top_channels", {"k": 3}, "c1")])
            return AssistantMessage(content=json.dumps(
                {"answer": "Rates are measurements.", "evidence_ids": []}))

    backend = Recorder()
    tools = {k: v for k, v in _named_tools("threshold_sensitivity", "detect_hfo",
                                           "spectral_power", "channel_qc").items()}
    extra = {k: v for k, v in tools.items()
             if k in ("threshold_sensitivity", "detect_hfo", "spectral_power", "channel_qc")}
    OnsetAgent(store, backend=backend, extra_tools=extra).ask(
        "Does the busiest channel survive a stricter threshold?")
    assert len(backend.offered) >= 2
    first, second = backend.offered[0], backend.offered[1]
    assert "threshold_sensitivity" in first and "spectral_power" not in first
    assert len(first) <= 5 and set(second) >= set(tools), "every tool from the second step on"
