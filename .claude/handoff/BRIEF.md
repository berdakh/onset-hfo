# Brief: <slug>

Written by the architect. Copy this file to `.claude/handoff/<slug>-brief.md`.

## Base
Base commit: `<sha of main when the brief was written>`. If main has moved, say what changed.

## Goal
One or two sentences. What must be true when this is done.

## Subtasks
Ordered, each with the files it touches. Two subtasks may run in parallel (one worktree each) only if their file lists are disjoint; otherwise they run one after another.
1. `<subtask>`: files `<paths>`

## In scope
- Files and symbols the developer may touch (paths relative to `onset-hfo/`, the folder with `pyproject.toml`).
- Files and symbols that are read-only context.

## Out of scope
What must not change (published numbers, `onset_agent/tools.py`, the contract version).

## Constraints
- `onset_hfo` should not import `onset_review` or Qt (add no new imports; three exist: `cli.py` 695 and 706, `case/imaging.py` 452).
- No recommendation field; the model never produces a number.
- Any other rule that applies to this change.

## Interface agreements
Signatures, file formats, JSON shapes the other layers rely on. Write them down here; a test should flag any deviation.

## Acceptance checks
Commands a machine can run, each with the expected result.
```
ruff check .
pytest -q tests/<files>
<a layering or content check>
```

## Escalate if
The change needs a contract change, touches a published number, or crosses a layer this brief did not name.

## Branch
Branch name: `almubdi/<topic>`, from the base commit above. Merge the base branch in before review.
