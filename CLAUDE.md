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
- Work in the worktree you are given. Never push, open a PR or switch branches yourself.
- Briefs and return records for multi-agent work are in `.claude/handoff/`; agent specs are in `.claude/agents/`.
