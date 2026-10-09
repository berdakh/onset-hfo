"""The template library: a pre-written script for each analysis this window does. Qt-free.

For a reader who would rather work in code than through the pages. Each
template is the same analysis a page does, written out against the names the
console starts with (`raw`, `signal`, `findings`, `events`, `quality`,
`request`, `recording`, `session` …), in ``# %%`` cells so it can be run a
step at a time. Where a page's numbers can be recomputed from the recording,
the template recomputes them with the same library calls and says whether
they match. Every template is run against an analysed session by the tests,
so none is left behind when the code under it changes.

The scripts live in ``onset_review/templates/`` as plain files, each opening
with four comment lines the library reads::

    # Template: Reproduce the ranking, step by step
    # Group: Detection and ranking
    # Mirrors: the Recording page
    # About: one sentence on what it does
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Template", "templates", "find", "grouped", "GROUPS", "FOLDER"]

FOLDER = Path(__file__).with_name("templates")
#: The order groups are offered in: the order a reader meets the pages.
GROUPS = ("Start here", "Detection and ranking", "Signal", "Events", "Seizure onset",
          "Anatomy", "Results")
_FIELDS = {"template": "title", "group": "group", "mirrors": "mirrors", "about": "about"}


@dataclass(frozen=True)
class Template:
    key: str            #: the file's stem, e.g. "b10_reproduce_ranking"
    title: str
    group: str
    mirrors: str        #: the page or view it does in code
    about: str
    text: str           #: the whole script, header included

    @property
    def path(self) -> Path:
        return FOLDER / f"{self.key}.py"


def _read(path: Path) -> Template:
    text = path.read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        name, _, value = line.lstrip("# ").partition(":")
        field = _FIELDS.get(name.strip().lower())
        if field:
            found[field] = value.strip()
    missing = [f for f in _FIELDS.values() if f not in found]
    if missing:
        raise ValueError(f"{path.name}: the header lacks {', '.join(missing)}")
    return Template(key=path.stem, text=text, **found)


def templates(folder: Path | None = None) -> list[Template]:
    """Every template, in the order of `GROUPS` and then of their file names."""
    folder = Path(folder or FOLDER)
    found = [_read(p) for p in sorted(folder.glob("*.py")) if not p.name.startswith("_")]
    rank = {g: i for i, g in enumerate(GROUPS)}
    return sorted(found, key=lambda t: (rank.get(t.group, len(GROUPS)), t.key))


def grouped(folder: Path | None = None) -> list[tuple[str, list[Template]]]:
    out: dict[str, list[Template]] = {}
    for template in templates(folder):
        out.setdefault(template.group, []).append(template)
    return list(out.items())


def find(key: str, folder: Path | None = None) -> Template | None:
    return next((t for t in templates(folder) if t.key == key or t.title == key), None)
