"""Cells in a script, and notebooks as scripts: Qt-free.

The Analysis page's editor runs a script a cell at a time, as Spyder and VS
Code do: a line starting ``# %%`` begins a cell, and Ctrl+Enter runs the one
the cursor is in. The same marking is how a Jupyter notebook is opened there
(the "percent" format jupytext uses): each code cell becomes a ``# %%``
cell, each Markdown cell a ``# %% [markdown]`` cell of comment lines. Saving
back writes the cells as a notebook again, without outputs -- the outputs
are whatever the console prints when the cells are run, here.

No package is needed: a notebook is JSON, version 4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Cell", "split_cells", "cell_at", "read_notebook", "notebook_to_script",
           "script_to_notebook", "write_notebook"]

#: A line that starts a cell, with what follows the marker.
MARKER = re.compile(r"^\s*#\s*%%(.*)$")
MARKDOWN = re.compile(r"^\s*\[markdown\]\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Cell:
    """Lines `start` (the marker, when there is one) to `stop` (exclusive)."""

    start: int
    stop: int
    title: str = ""
    markdown: bool = False

    def source(self, lines: list[str]) -> str:
        body = lines[self.start:self.stop]
        if body and MARKER.match(body[0]):
            body = body[1:]
        return "\n".join(body).strip("\n")


def split_cells(text: str) -> list[Cell]:
    """The cells of `text`. Lines before the first marker are a cell of
    their own, if there are any; a script without markers is one cell."""
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if MARKER.match(line)]
    if not starts or starts[0] != 0:
        starts = [0] + starts
    cells = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(lines)
        match = MARKER.match(lines[start])
        title, markdown = "", False
        if match:
            rest = match.group(1).strip()
            md = MARKDOWN.match(rest)
            markdown = md is not None
            title = (md.group(1) if md else rest).strip()
        cells.append(Cell(start, stop, title, markdown))
    return cells


def cell_at(text: str, line: int) -> Cell:
    """The cell holding 0-based `line`."""
    for cell in split_cells(text):
        if cell.start <= line < cell.stop:
            return cell
    return split_cells(text)[-1]


# -- notebooks ------------------------------------------------------------------
def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def read_notebook(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def notebook_to_script(notebook: dict) -> str:
    """A notebook as a percent-format script: outputs left out, Markdown
    kept as comments so the reading survives the round trip."""
    parts = []
    for cell in notebook.get("cells", []):
        kind = cell.get("cell_type")
        source = _source(cell).rstrip("\n")
        if kind == "code":
            parts.append("# %%\n" + source)
        elif kind == "markdown":
            commented = "\n".join(("# " + line) if line else "#"
                                  for line in source.split("\n"))
            parts.append("# %% [markdown]\n" + commented)
        elif kind == "raw":
            parts.append("# %% [raw]\n" + "\n".join("# " + line for line in source.split("\n")))
    return "\n\n".join(parts) + "\n"


def _uncomment(text: str) -> str:
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("# "):
            out.append(stripped[2:])
        elif stripped.startswith("#"):
            out.append(stripped[1:])
        else:
            out.append(line)
    return "\n".join(out)


def _lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def script_to_notebook(text: str, template: dict | None = None) -> dict:
    """The cells of `text` as a version-4 notebook. `template` (the notebook
    the script was opened from) lends its metadata, so a kernel choice or a
    Colab badge survives a save."""
    lines = text.split("\n")
    cells = []
    for cell in split_cells(text):
        source = cell.source(lines)
        if not source.strip() and cell.start == 0 and not MARKER.match(lines[0]):
            continue        # nothing before the first marker
        if cell.markdown:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": _lines(_uncomment(source))})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": _lines(source)})
    metadata = dict((template or {}).get("metadata") or {})
    metadata.setdefault("kernelspec", {"display_name": "Python 3", "language": "python",
                                       "name": "python3"})
    metadata.setdefault("language_info", {"name": "python"})
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 4}


def write_notebook(text: str, path: str | Path, template: dict | None = None) -> Path:
    path = Path(path)
    path.write_text(json.dumps(script_to_notebook(text, template), indent=1,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path
