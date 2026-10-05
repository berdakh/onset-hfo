#!/usr/bin/env python
"""Pin every model in `onset_agent.hardware.CATALOGUE` to a Hub commit.

The catalogue ships with ``revision="main"`` because the container this
project was built in cannot reach huggingface.co, and a commit hash that was
never looked up is worse than an honest moving reference. ``main`` is still a
moving target, though: a re-run months later can load different weights
without anything in the project recording that it did. This script closes
that gap on a machine that can see the Hub.

    # what would change, without touching anything
    python scripts/pin_catalogue.py --dry-run

    # resolve each model's current commit and write it into hardware.py
    python scripts/pin_catalogue.py

    # exit non-zero if any entry is unpinned or no longer matches the Hub
    python scripts/pin_catalogue.py --check

It edits the source text of `onset_agent/hardware.py` in place -- the
revisions belong in the file a reader opens, not in a sidecar they would have
to know about -- and inserts or replaces one ``revision="..."`` line per
entry. Nothing else in the file is touched. Needs the ``llm`` extra for
``huggingface_hub``.

What this does not pin: the Ollama tags. Ollama's registry keys models by its
own digests, not Hub commits, and ``ollama pull qwen3:8b`` fetches whatever
the tag points at that day. Record the digest ``ollama show`` prints when a
served-route result needs to be reproducible.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    # Runnable from a checkout without installing the package.
    sys.path.insert(0, str(REPO))

HARDWARE_PY = REPO / "onset_agent" / "hardware.py"

#: A full-length git commit hash. The Hub's ``sha`` is always one; anything
#: shorter in the catalogue is a hand edit this script should refuse to
#: mistake for a pin.
_SHA = re.compile(r"^[0-9a-f]{40}$")


class CatalogueEditError(RuntimeError):
    """The source did not look the way the rewrite expects."""


def _entry_span(lines: list[str], model_id: str) -> tuple[int, int]:
    """Line indices ``[start, end)`` of the ``ModelSpec("<model_id>", ...)`` call.

    The call ends at the first line whose trailing ``),`` closes it. Every
    entry in the catalogue is written that way, and a call that is not would
    be rewritten wrongly in silence, so this refuses instead.
    """
    opener = f'ModelSpec("{model_id}",'
    starts = [i for i, line in enumerate(lines) if opener in line]
    if len(starts) != 1:
        raise CatalogueEditError(
            f"expected exactly one catalogue entry for {model_id}, found {len(starts)}")
    start = starts[0]
    for end in range(start, len(lines)):
        if lines[end].rstrip().endswith("),"):
            return start, end + 1
    raise CatalogueEditError(f"could not find the end of the entry for {model_id}")


def rewrite(source: str, revisions: dict[str, str]) -> str:
    """Return ``source`` with each model's ``revision=`` set to its hash.

    Pure: takes the file's text and the mapping, returns the new text. An
    entry that already carries a ``revision="..."`` line has its value
    replaced; one that does not gets the line inserted directly under the
    ``ModelSpec(`` line, aligned with the keyword arguments that follow it.
    """
    for model_id, sha in revisions.items():
        if not _SHA.match(sha):
            raise CatalogueEditError(f"{model_id}: {sha!r} is not a full commit hash")

    lines = source.splitlines(keepends=True)
    for model_id, sha in revisions.items():
        start, end = _entry_span(lines, model_id)
        replaced = False
        for i in range(start, end):
            new, n = re.subn(r'revision="[^"]*"', f'revision="{sha}"', lines[i])
            if n:
                if n > 1:
                    raise CatalogueEditError(f"{model_id}: two revisions on one line")
                lines[i] = new
                replaced = True
                break
        if not replaced:
            head = lines[start]
            indent = " " * (head.index("ModelSpec(") + len("ModelSpec("))
            newline = "\r\n" if head.endswith("\r\n") else "\n"
            lines.insert(start + 1, f'{indent}revision="{sha}",{newline}')
    return "".join(lines)


def current(catalogue) -> dict[str, str]:
    return {spec.model_id: spec.revision for spec in catalogue}


def resolve(catalogue) -> dict[str, str]:
    """Ask the Hub for each entry's current commit."""
    from onset_agent.hardware import pinned

    return current(pinned(catalogue))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="compare the catalogue with the Hub; write nothing")
    mode.add_argument("--dry-run", action="store_true",
                      help="show what would change; write nothing")
    parser.add_argument("--file", type=Path, default=HARDWARE_PY,
                        help="the module to edit (default: the installed one)")
    args = parser.parse_args(argv)

    from onset_agent.hardware import CATALOGUE

    before = current(CATALOGUE)
    try:
        after = resolve(CATALOGUE)
    except ImportError:
        print("[pin-catalogue] huggingface_hub is not installed: "
              "pip install -e '.[llm]'", file=sys.stderr)
        return 2
    except Exception as error:  # the Hub client raises several unrelated types
        print(f"[pin-catalogue] could not reach the Hub: {error}\n"
              "If this machine is behind a proxy or an allowlist, run this "
              "on one that can see huggingface.co.", file=sys.stderr)
        return 2

    changed = {m: (before[m], after[m]) for m in after if before[m] != after[m]}
    for model_id, (old, new) in changed.items():
        print(f"{model_id}: {old} -> {new}")
    if not changed:
        print(f"[pin-catalogue] all {len(after)} entries already match the Hub")
        return 0
    if args.check:
        print(f"[pin-catalogue] {len(changed)} of {len(after)} entries are "
              "unpinned or stale", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    source = args.file.read_text(encoding="utf-8")
    args.file.write_text(rewrite(source, after), encoding="utf-8")
    print(f"[pin-catalogue] wrote {len(changed)} revision(s) to {args.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
