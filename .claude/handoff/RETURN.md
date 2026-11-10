# Return: <slug>

Fixed fields only. No narrative. Save it as `<slug>-return.md` next to the brief.

## Where
Worktree, branch and the commit this return describes.

## Changed
- `path`: one line on what changed.

## Checks run
| command | result | counts |
|---|---|---|
| `ruff check .` | pass/fail | |
| `pytest -q ...` | pass/fail | passed / failed / skipped |

## Decisions that need the architect
- Decision, and why it could not be settled inside the brief. "None" if none.

## Unresolved
- Anything unfinished or any failing test, with its name and output. "None" if none.
