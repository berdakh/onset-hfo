# Rules for coding agents in this repo

Only what the code does not show. For the project itself read `onset-hfo/README.md`.

- The package lives one level in: run everything from `onset-hfo/` (the folder with `pyproject.toml`).
- `onset_hfo` should not import `onset_review` or Qt, and `onset_review` may import `onset_hfo`. Three imports already break this (`onset_hfo/cli.py` lines 695 and 706, `onset_hfo/case/imaging.py` line 452); add none.
- `onset_agent/tools.py` is frozen. Changing a tool's JSON shape means bumping `CONTRACT_VERSION` and would break the published orchestration numbers, so ask first.
- `onset_hfo.__version__` is the pipeline version, not the package version. The package version is in `pyproject.toml`, `onset_review/__init__.py` and `CITATION.cff`.
- Outcome codes: `'S'` is seizure-free, `'F'` is failure. Only `cohort.decode_outcome` maps them.
- One threshold for both bands is a bug. Use `outcome.BAND_THRESHOLD_SD`.
- The model never produces a number. No recommendation field or recommendation text anywhere.
- Tests are offline (`ONSET_HFO_OFFLINE=1`). Run `ruff check .` and `pytest -q` from `onset-hfo/`.
- Never leave a failing test alone: fix it, or report it with its output.
- Never commit `paper/`, PDFs, recordings or `artifacts/`.
- Work in the worktree you are given. You may push your own task branch, but never `main` or `master`, and never force-push. Do not switch branches, open a PR or merge without the user's say-so.
- Briefs and return records for multi-agent work are in `.claude/handoff/`; agent specs are in `.claude/agents/`.

## How to orchestrate
The main session is the manager. The agents report to it and none spawns another.
- A small change in one layer skips the architect: give a developer a short brief (goal, files, acceptance commands).
- A cross-layer change, or anything touching a contract, starts with `architect` writing `.claude/handoff/<slug>-brief.md`. Ask the user before changing a contract.
- One worktree and one branch per task, from the current base. Run two developers in parallel only if the brief's file lists are disjoint; otherwise one after the other.
- The developer writes `<slug>-return.md`. Then `qa-tester` reruns the brief's checks. Add `privacy-reviewer` for logs, exports or fixtures, `medical-reviewer` for clinical wording, and `docs-release` for docs or versions.
- Before a PR is opened, `pr-verifier` checks its description and the papers it cites.
- Agents may push their own task branch, never `main` or `master`, and never with force. Opening a PR or merging needs the user's say-so.
