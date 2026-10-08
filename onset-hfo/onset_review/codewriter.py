"""Python written by the local model, for the Analysis page: Qt-free.

The reader says what they want in words; the model the Assistant page
chose drafts the code; the code goes into the editor and nothing runs it
but the reader. What the model is told is the inventory, never the data:
each Workspace name with its type and size, a DataFrame's columns, a dict's
keys, the fields of an event -- enough to write `findings["rate"]` rather
than guess at it -- and the script open in the editor, so "fix this" and
"now plot it per channel" have something to work on.

A draft is checked for what can be checked without running it, and the
editor shows the result with the draft:

* it must parse (`ast`);
* every name it reads must exist -- made in the draft, in the console, a
  builtin, or imported -- or it is listed as unknown, the commonest way a
  small model's code fails;
* calls that delete files, start processes, or reach the network are named,
  because a reader skimming twenty lines should not have to find them.

None of that makes the code right. The draft says at its top that a model
wrote it and nothing checked what it computes.
"""

from __future__ import annotations

import ast
import builtins
import re
import types
from dataclasses import dataclass, field, fields, is_dataclass

__all__ = ["CODE_PROMPT", "CODE_TOKENS", "inventory", "messages", "extract_code",
           "check_code", "Draft", "header"]

#: Room for a script of a few dozen lines with its comments.
CODE_TOKENS = 1400

CODE_PROMPT = """You write Python for a researcher's in-process console beside an \
intracranial EEG review tool. The console already holds the names listed under WORKSPACE; \
use them directly and never load the recording again. np, pd, mne, plt, onset_hfo and Path \
are imported. `raw` is an MNE Raw object in volts; `signal` is raw's samples \
(channels x time, volts, read-only: copy with np.array(signal) before changing it); \
`times` is in seconds; `findings` is a DataFrame with one row per channel; `events` is a \
list of Event objects.

Rules:
- Answer with ONE ```python code block and at most two short sentences after it.
- Use only names from WORKSPACE, the standard library, numpy, scipy, pandas, matplotlib and \
mne. Import anything else you use at the top.
- Never change session, raw, events or findings in place: work on copies.
- Draw with matplotlib (plt.figure(), plt.plot...); figures open on their own, so do not \
call plt.savefig unless asked.
- Never delete files, start processes or use the network.
- Comment each step in one short line. Print the numbers you compute, with units.
- If the request is a medical decision (what to resect, a diagnosis), write no code and say \
that the clinician decides."""


def _columns(value) -> str:
    try:
        import pandas as pd
    except ImportError:      # pragma: no cover - pandas is a dependency
        return ""
    if isinstance(value, pd.DataFrame):
        return "columns: " + ", ".join(f"{c} ({value[c].dtype})" for c in value.columns[:40])
    if isinstance(value, dict) and value:
        keys = list(value)[:20]
        return "keys: " + ", ".join(repr(k) for k in keys) + (" …" if len(value) > 20 else "")
    if isinstance(value, (list, tuple)) and value and is_dataclass(value[0]):
        return f"items are {type(value[0]).__name__}: " + ", ".join(
            f.name for f in fields(value[0]))
    if is_dataclass(value) and not isinstance(value, type):
        return "fields: " + ", ".join(f.name for f in fields(value))
    return ""


def inventory(names: dict, limit: int = 60) -> str:
    """The WORKSPACE block: one line per name -- type, size and the shape of
    what is inside -- and no values. Modules and functions are left out."""
    from onset_review.variables import describe

    lines = []
    for name, value in list(names.items())[:limit]:
        if name.startswith("_") or isinstance(value, types.ModuleType) or (
                callable(value) and not hasattr(value, "ch_names")):
            continue
        kind, size, _summary = describe(value)
        inner = _columns(value)
        lines.append(f"- {name}: {kind}" + (f", {size}" if size else "")
                     + (f"; {inner}" if inner else ""))
    return "\n".join(lines) if lines else "(empty)"


def messages(request: str, names: dict, script: str = "",
             history: list[tuple[str, str]] | None = None) -> list[dict]:
    """The conversation for one draft: the rules, then earlier requests and
    drafts (so "now per channel" follows on), then this request with the
    inventory and the script open in the editor."""
    out = [{"role": "system", "content": CODE_PROMPT}]
    for asked, answered in (history or [])[-4:]:
        out.append({"role": "user", "content": asked})
        out.append({"role": "assistant", "content": answered})
    parts = [f"WORKSPACE\n{inventory(names)}"]
    if script.strip():
        parts.append("THE SCRIPT IN THE EDITOR\n```python\n" + script.strip()[-6000:] + "\n```")
    parts.append(f"REQUEST\n{request.strip()}")
    out.append({"role": "user", "content": "\n\n".join(parts)})
    return out


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(reply: str) -> str:
    """The code in a reply: the first fenced block (the longest, if several
    are python), or the whole reply when it parses as Python by itself."""
    blocks = _FENCE.findall(reply or "")
    if blocks:
        return max(blocks, key=len).strip("\n") + "\n"
    text = (reply or "").strip()
    if not text:
        return ""
    try:
        ast.parse(text)
    except SyntaxError:
        return ""
    return text + "\n"


#: Calls named in a draft's notes: (dotted name or suffix, what it does).
RISKY = (
    ("os.remove", "deletes a file"), ("os.unlink", "deletes a file"),
    ("os.rmdir", "deletes a folder"), ("shutil.rmtree", "deletes a folder and everything in it"),
    ("shutil.move", "moves files"), ("unlink", "deletes a file"),
    ("subprocess", "starts another program"), ("os.system", "starts another program"),
    ("os.popen", "starts another program"),
    ("requests", "uses the network"), ("urllib", "uses the network"),
    ("socket", "uses the network"), ("http.client", "uses the network"),
    ("eval", "runs text as code"), ("exec", "runs text as code"),
)


@dataclass
class Draft:
    """A draft and what could be checked about it without running it."""

    code: str
    parses: bool = True
    error: str = ""
    unknown: list[str] = field(default_factory=list)
    risky: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        out = []
        if not self.code.strip():
            return ["The model gave no code."]
        if not self.parses:
            out.append(f"Does not parse: {self.error}")
        if self.unknown:
            out.append("Uses names that do not exist yet: " + ", ".join(self.unknown))
        for what in self.risky:
            out.append(f"Read before running: {what}")
        return out


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else node.attr
    return ""


def check_code(code: str, names: dict | None = None) -> Draft:
    """What can be said of `code` before it runs: whether it parses, which
    names it reads that nothing defines, and which calls touch files,
    processes or the network."""
    draft = Draft(code)
    try:
        tree = ast.parse(code or "")
    except SyntaxError as error:
        draft.parses = False
        draft.error = f"line {error.lineno}: {error.msg}"
        return draft
    known = set(dir(builtins)) | set(names or {}) | {"__file__", "__name__", "display",
                                                     "get_ipython"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                known.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(node.name)
            for arg in ast.walk(node.args) if hasattr(node, "args") else ():
                if isinstance(arg, ast.arg):
                    known.add(arg.arg)
        elif isinstance(node, ast.Lambda):
            for arg in ast.walk(node.args):
                if isinstance(arg, ast.arg):
                    known.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            known.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            known.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            known.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            known.add(node.name)
    unknown = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id not in known and node.id not in unknown:
            unknown.append(node.id)
    draft.unknown = unknown
    risky = []
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _dotted(node.func)
            for name, what in RISKY:
                if called == name or called.endswith("." + name) or called.startswith(name + "."):
                    line = f"line {node.lineno} calls {called}() — {what}"
                    if line not in risky:
                        risky.append(line)
    for module in sorted(imported):
        for name, what in RISKY:
            if module == name or module.startswith(name + "."):
                line = f"imports {module} — {what}"
                if line not in risky:
                    risky.append(line)
    draft.risky = risky
    return draft


def header(model: str, request: str) -> str:
    """The comment a draft starts with: who wrote it, from what, and that
    nothing checked it."""
    asked = " ".join(request.split())
    if len(asked) > 160:
        asked = asked[:159] + "…"
    return (f"# Drafted by the local model ({model}) from: {asked}\n"
            "# Not checked: read it before you run it. The numbers it prints are "
            "the code's, not the analysis's.\n")
