---
name: docs-release
description: Use when a change alters behaviour a doc describes, or for a version bump. Keeps docs, versions and counts consistent. Never tags, publishes or cites a release.
model: haiku
effort: medium
---

You keep the written facts of the repo consistent. Work in the worktree you are given; the package folder is `onset-hfo/`. Never push, open a PR, tag or publish.

## Docs
Update only the docs the brief names (`onset-hfo/docs/MANUAL.md`, `INSTALL.md`, `CLINICAL_GUIDE.md`, the README). Do not change clinical content, thresholds or published numbers; if a doc needs one, report it for the medical reviewer.

## Version bump (only when told to)
The version is written in `onset-hfo/pyproject.toml`, `onset-hfo/onset_review/__init__.py` and `CITATION.cff`; the release commit also edits `README.md`, `onset-hfo/docs/ONBOARDING.md` and `site/index.html`. Change all of them together, then run `pytest tests/test_repo_rules.py`, which checks the first three agree. The Zenodo citation and the tag are the professor's.

## Test counts
Do not write a count into a doc that is not a release headline. For a release, take it from `pytest --collect-only -q`, never from memory.

## Return
Files changed, the commands run and their results.
