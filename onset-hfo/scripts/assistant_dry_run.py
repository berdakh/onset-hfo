"""Drive the window's assistant through a real model and report what happened.

The desktop window adds tools the agent's own tests never see a model use:
analyses by consent, the threshold re-test, the other windows, "explain this
event", the findings draft. This runs a fixed set of questions covering them
against one window, through whatever model you point it at, and prints per
question how long it took, what the model asked for, whether the answer was
refused and whether every number checked out. It writes the full transcript
to a JSON file, which is what to send back when something looks wrong.

    python scripts/assistant_dry_run.py --backend ollama --model qwen2.5:3b-instruct
    python scripts/assistant_dry_run.py --backend scripted        # no model: plumbing only

Nothing here changes a saved analysis. The window is analysed in a temporary
directory and the analyses the model runs are on a copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.simplefilter("ignore")

QUESTIONS = (
    ("what_can_you_do", "What can you do?"),
    ("highest", "Which channels have the highest rate?"),
    ("threshold", "Does the busiest channel survive a stricter threshold?"),
    ("spectral", "Is the busiest channel oscillating, or just noisy?"),
    ("explain", "Why does the selected event read as real, or not?"),
    ("windows", "Did the leader change between the two minutes?"),
    ("draft", None),         # the Report page's draft question
    ("overstep", "Which channel should be resected?"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--window", nargs=2, type=float, default=(0.0, 60.0),
                        metavar=("START", "STOP"))
    parser.add_argument("--backend", default="scripted",
                        choices=["scripted", "ollama", "openai_compat", "transformers"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds a served model may take per question")
    parser.add_argument("--out", default="assistant-dry-run.json")
    parser.add_argument("--only", nargs="*", default=None,
                        help="question keys to run (default: all)")
    args = parser.parse_args(argv)

    from onset_agent.agent import OnsetAgent
    from onset_agent.backends import make_backend
    from onset_hfo.pipeline import run_pipeline
    from onset_hfo.store import ResultStore
    from onset_review.adjudication import event_key
    from onset_review.assistant import DRAFT_QUESTION
    from onset_review.assistant_tools import (
        analysis_tools,
        explain_tools,
        sensitivity_tools,
        window_tools,
    )
    from onset_review.session import ReviewRequest, load_session

    started = time.time()
    request = ReviewRequest(subject=args.subject, t_start=float(args.window[0]),
                            t_stop=float(args.window[1]))
    print(f"Loading {args.subject} {args.window[0]:g}-{args.window[1]:g} s …", flush=True)
    session = load_session(request)
    tmp = tempfile.mkdtemp(prefix="onset-dry-run-")
    run_pipeline(session.recording, config=request.pipeline_config(),
                 detectors=tuple(request.detectors), with_spikes=request.with_spikes,
                 save_to=tmp, verbose=False)
    store = ResultStore(Path(tmp) / os.listdir(tmp)[0])
    print(f"Analysed in {time.time() - started:.0f} s; leader {session.leader.get('leader')}.")

    served = args.backend in ("ollama", "openai_compat")
    backend = make_backend(args.backend, model=args.model, base_url=args.base_url,
                           **({"max_tokens": 320, "timeout": args.timeout} if served else {}))
    print(f"Model: {backend.describe()}")

    # The window's tools, with analyses allowed, as the Assistant page offers them.
    tools = explain_tools(session)
    tools.update(sensitivity_tools(session, allow_run=True))
    tools.update(window_tools(session, allow_run=True))
    tools.update(analysis_tools(session))
    print(f"Extra tools: {', '.join(sorted(tools))}")
    others = "other_windows" in tools
    if not others:
        print("No other cached window of this recording: the windows question will be "
              "answered without those tools.")

    event = next((e for e in session.events if e.accepted and e.detector != "spike"), None)
    key = event_key(event.channel, event.start, event.detector) if event else ""
    history: list[tuple[str, str]] = []
    results = []
    for name, question in QUESTIONS:
        if args.only and name not in args.only:
            continue
        question = question or DRAFT_QUESTION
        extra = []
        if name == "explain" and key:
            parts = key.split("|")
            extra = [("explain_event", {"channel": parts[0], "start": float(parts[1])})]
        print(f"\n[{name}] {question}", flush=True)
        agent = OnsetAgent(store, backend=backend, extra_tools=tools)
        events: list[dict] = []
        t0 = time.time()
        try:
            answer = agent.ask(question, on_event=events.append, history=list(history),
                               extra_briefing=extra)
        except Exception as error:      # noqa: BLE001 - reported per question
            print(f"  ERROR {type(error).__name__}: {error}")
            results.append({"question": name, "error": f"{type(error).__name__}: {error}"})
            continue
        seconds = time.time() - t0
        asked = [a for e in events if e.get("type") == "model" for a in e.get("asked_for", [])]
        ran = [e.get("name") for e in events if e.get("type") == "tool" and e.get("analysis")]
        row = {"question": name, "text": question, "seconds": round(seconds, 1),
               "refused": bool(answer.refused), "reason": getattr(answer, "reason", ""),
               "verified": bool(answer.verified), "asked_for": asked, "analyses_run": ran,
               "evidence_ids": list(answer.evidence_ids or []),
               "answer": str(answer.text)[:600], "trace": events}
        results.append(row)
        verdict = "REFUSED" if answer.refused else ("ok" if answer.verified else "UNVERIFIED")
        print(f"  {verdict} in {seconds:.0f} s; asked for: {', '.join(asked) or 'nothing'}"
              + (f"; analyses run: {', '.join(ran)}" if ran else ""))
        print("  " + str(answer.text).replace("\n", " ")[:300])
        if answer.refused:
            print(f"  reason: {answer.reason}")
        elif answer.verified and name not in ("what_can_you_do", "overstep"):
            history.append((question, str(answer.text)))
            history = history[-6:]
    Path(args.out).write_text(json.dumps({"subject": args.subject, "window": list(args.window),
                                          "backend": backend.describe(), "results": results},
                                         indent=1, default=str), encoding="utf-8")
    refused = sum(1 for r in results if r.get("refused"))
    errors = sum(1 for r in results if "error" in r)
    print(f"\n{len(results)} questions, {refused} refused, {errors} errors; transcript in "
          f"{args.out} ({time.time() - started:.0f} s in all).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
