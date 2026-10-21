"""Token usage of the Claude Code sessions that worked on this repo, from the first to now.

Reads the session transcripts Claude Code keeps under ``~/.claude/projects`` and writes
``LOG.md`` and a few PNG charts into ``.claude/handoff/`` (both ignored by git). Only counts
are read and written: no message text, titles or file contents.

    python3 .claude/tools/token_report.py            # every session of this repo, worktrees included
    python3 .claude/tools/token_report.py --no-charts

What the numbers are, and are not
---------------------------------
* A transcript logs one API response once per content block, all with the same usage, so
  responses are counted once, by message id. Adding up the rows would overcount about 2.4x.
* "New" tokens are output + cache writes + fresh input. Cache reads are the conversation being
  read again on every turn; they are cheap and huge, so they are shown apart, never added in.
* A subagent's transcript records output tokens from the start of the stream, so they are a
  lower bound (a 16,000-character reply can be logged as 26 tokens). Its input and cache
  figures are sound.
* Dollar cost is Claude Code's own cumulative snapshot per session, not a price table of ours.
  Token totals are reconciled against it in the last table.
* "Task" is the git branch that was checked out where the session ran. A session that works in
  a worktree through ``cd`` is still filed under the branch it started on.
* Responses are de-duplicated within one transcript, not across transcripts: a resumed or forked
  session that copies its history into a new file would count that history twice.
* Sessions are found by the working directory they recorded, inside the main checkout or any
  worktree of it. Output goes to the *main checkout's* ``.claude/handoff`` even when run from a
  worktree, because that is where briefs live; pass ``--out`` to put it elsewhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

KINDS = ("output_tokens", "cache_creation_input_tokens", "input_tokens", "cache_read_input_tokens")
LABEL = {"output_tokens": "output", "cache_creation_input_tokens": "cache write",
         "input_tokens": "fresh input", "cache_read_input_tokens": "cache read"}
NEW = KINDS[:3]

#: What a response was doing, from the first tool it called. A response with no tool is answering.
WORK = (
    ("reading code", ("Read", "Grep", "Glob", "mcp__codegraph")),
    ("running commands", ("Bash",)),
    ("editing", ("Edit", "Write", "NotebookEdit")),
    ("web research", ("WebSearch", "WebFetch")),
    ("delegating", ("Agent",)),
    ("artifacts", ("Artifact",)),
)


@dataclass
class Response:
    """One API response: its usage counted once, however many rows logged it."""

    model: str
    when: datetime
    branch: str
    agent: str
    session: str
    work: str
    tokens: dict[str, int]
    lower_bound: bool = False  # output tokens are under-recorded (subagent transcripts)


@dataclass
class Session:
    path: Path
    responses: list[Response] = field(default_factory=list)
    snapshot: dict | None = None  # Claude Code's last cost-state row


def work_of(content) -> str:
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "")
            for label, prefixes in WORK:
                if name.startswith(prefixes):
                    return label
            return "other tools"
    return "answering"


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone()


def read_transcript(path: Path, agent: str, subagent: bool) -> Session:
    """Responses of one transcript, deduplicated by message id."""
    session = Session(path)
    seen: dict[str, Response] = {}
    blocks: dict[str, list] = {}
    branch = "(unknown)"
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("gitBranch"):
            branch = row["gitBranch"]
        if row.get("type") == "cost-state":
            session.snapshot = row
        message = row.get("message")
        if (row.get("type") != "assistant" or not isinstance(message, dict) or not message.get("usage")
                or not row.get("timestamp")):
            continue
        usage = message["usage"]
        tokens = {k: int(usage.get(k) or 0) for k in KINDS}
        key = message.get("id") or row.get("uuid")
        blocks.setdefault(key, []).extend(message.get("content") or [])  # one row per content block
        old = seen.get(key)
        if old is not None:  # same response logged again: keep the larger count of each kind
            old.tokens = {k: max(old.tokens[k], tokens[k]) for k in KINDS}
            old.work = work_of(blocks[key])
            continue
        seen[key] = Response(
            model=message.get("model", "unknown"), when=parse_time(row["timestamp"]), branch=branch,
            agent=agent, session=path.stem[:8], work=work_of(blocks[key]),
            tokens=tokens, lower_bound=subagent)
    session.responses = sorted(seen.values(), key=lambda r: r.when)
    return session


def read_project(directory: Path) -> list[Session]:
    sessions = []
    for path in sorted(directory.glob("*.jsonl")):
        sessions.append(read_transcript(path, "main", False))
        for sub in sorted((directory / path.stem / "subagents").glob("agent-*.jsonl")):
            meta = sub.with_suffix(".meta.json")
            kind = "subagent"
            try:
                kind = json.loads(meta.read_text(encoding="utf-8")).get("agentType", kind)
            except (OSError, ValueError, AttributeError):
                pass  # no or unreadable meta: still a subagent, just unnamed
            sessions.append(read_transcript(sub, kind, True))
    return sessions


def main_checkout() -> Path:
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return Path(out).parent


def repo_roots() -> list[Path]:
    """The main checkout and every worktree made from it."""
    out = subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True,
                         check=True).stdout
    return [Path(line.split(" ", 1)[1]) for line in out.splitlines() if line.startswith("worktree ")]


def session_cwd(directory: Path) -> Path | None:
    """The folder a project's sessions ran in, from the first transcript that records one."""
    for path in sorted(directory.glob("*.jsonl"))[:1]:
        with path.open(encoding="utf-8") as handle:
            for _, line in zip(range(200), handle):
                try:
                    cwd = json.loads(line).get("cwd")
                except ValueError:
                    continue
                if cwd:
                    return Path(cwd)
    return None


def project_dirs(base: Path | None = None, roots: list[Path] | None = None) -> list[Path]:
    """Project folders whose sessions ran inside the repo or one of its worktrees.

    Matched on the working directory each transcript records, not on the folder's name, so a
    sibling project that happens to start with the same path is not picked up.
    """
    base = base or Path.home() / ".claude" / "projects"
    roots = [r.resolve() for r in (roots or repo_roots())]
    found = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        cwd = session_cwd(directory)
        if cwd is not None and any(cwd.resolve() == r or r in cwd.resolve().parents for r in roots):
            found.append(directory)
    return found


# -- aggregation ---------------------------------------------------------------------------------
def total(responses, kinds=KINDS) -> int:
    return sum(r.tokens[k] for r in responses for k in kinds)


def group(responses, key) -> dict:
    out = defaultdict(list)
    for r in responses:
        out[key(r)].append(r)
    return out


def n(value: int) -> str:
    return f"{value:,}"


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def token_rows(groups: dict, order=None) -> list[list]:
    rows = []
    for name in order or sorted(groups, key=lambda g: -total(groups[g], NEW)):
        rs = groups[name]
        rows.append([name, len(rs)] + [n(total(rs, (k,))) for k in KINDS])
    return rows


def reconcile(sessions: list[Session]) -> list[list]:
    """Our output and cache-read totals against Claude Code's own snapshot, per main session."""
    rows = []
    for s in sessions:
        if s.snapshot is None:
            continue
        theirs = Counter()
        for usage in s.snapshot.get("modelUsage", {}).values():
            theirs["output"] += usage.get("outputTokens", 0)
            theirs["cache read"] += usage.get("cacheReadInputTokens", 0)
        ours = Counter({"output": total(s.responses, ("output_tokens",)),
                        "cache read": total(s.responses, ("cache_read_input_tokens",))})
        rows.append([s.path.stem[:8], n(ours["output"]), n(theirs["output"]),
                     n(ours["cache read"]), n(theirs["cache read"]),
                     f"${s.snapshot.get('totalCostUSD', 0):,.2f}"])
    return rows


def build_log(sessions: list[Session]) -> str:
    responses = [r for s in sessions for r in s.responses]
    if not responses:
        return "# Token log\n\nNo transcripts found.\n"
    first, last = min(r.when for r in responses), max(r.when for r in responses)
    heads = ["", "responses"] + [LABEL[k] for k in KINDS]
    main_only = [r for r in responses if not r.lower_bound]
    cost = sum(s.snapshot.get("totalCostUSD", 0) for s in sessions if s.snapshot)
    parts = [
        "# Token log",
        f"{first:%Y-%m-%d} to {last:%Y-%m-%d}, {len(responses):,} responses in {len(sessions)} transcripts. "
        f"Generated by `.claude/tools/token_report.py`; counts only, no message text.",
        "How to read it: *new* tokens are output + cache write + fresh input. *Cache read* is the "
        "conversation read again each turn, so it is shown apart. Subagent output is a lower bound "
        "(the transcript records it at the start of the stream). Cost is Claude Code's own snapshot.",
        "## Total", table(heads, token_rows({"all": responses})),
        f"Claude Code's reported cost across sessions with a snapshot: ${cost:,.2f}.",
        "## By model", table(["model"] + heads[1:], token_rows(group(responses, lambda r: r.model))),
        "## By day", table(["day"] + heads[1:],
                           token_rows(group(responses, lambda r: f"{r.when:%Y-%m-%d}"),
                                      sorted({f"{r.when:%Y-%m-%d}" for r in responses}))),
        "## By task (git branch where the session ran)",
        table(["branch"] + heads[1:], token_rows(group(responses, lambda r: r.branch))),
        "## By agent", table(["agent"] + heads[1:], token_rows(group(responses, lambda r: r.agent))),
        "## By kind of work (main sessions; output tokens)",
        table(["work", "responses", "output"],
              [[w, len(rs), n(total(rs, ("output_tokens",)))]
               for w, rs in sorted(group(main_only, lambda r: r.work).items(),
                                   key=lambda kv: -total(kv[1], ("output_tokens",)))]),
        "Each response is filed under the first tool it called; one with no tool is *answering*.",
        "## Check against Claude Code's own snapshot",
        table(["session", "output (ours)", "output (Claude Code)", "cache read (ours)",
               "cache read (Claude Code)", "reported cost"], reconcile(sessions)),
    ]
    return "\n\n".join(parts) + "\n"


# -- charts --------------------------------------------------------------------------------------
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua: the validated first three slots
NEUTRAL = "#8a8984"


def charts(responses: list[Response], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def finish(fig, name, title, sub):
        height = fig.get_figheight()  # margins in inches, so short charts keep their title apart
        fig.subplots_adjust(top=1 - 0.95 / height)
        fig.text(0.01, 1 - 0.08 / height, title, ha="left", va="top", fontsize=13, fontweight="bold")
        fig.text(0.01, 1 - 0.42 / height, sub, ha="left", va="top", fontsize=9, color=MUTED)
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    def clean(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def millions(ax, axis):
        fmt = lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v else "0"  # noqa: E731
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(plt.FuncFormatter(fmt))

    # 1. over time: new tokens by kind, and cache reads on their own scale
    days = sorted({f"{r.when:%Y-%m-%d}" for r in responses})
    by_day = group(responses, lambda r: f"{r.when:%Y-%m-%d}")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"hspace": 0.18})
    base = [0] * len(days)
    for color, kind in zip(SERIES, NEW):
        vals = [total(by_day[d], (kind,)) for d in days]
        top.bar(days, vals, bottom=base, color=color, edgecolor=SURFACE, linewidth=1.5, width=0.62,
                label=LABEL[kind])
        base = [b + v for b, v in zip(base, vals)]
    for i, v in enumerate(base):
        top.text(i, v, f" {v / 1e3:,.0f}k", ha="center", va="bottom", fontsize=8, color=INK)
    top.set_ylabel("new tokens")
    top.set_ylim(0, max(base) * 1.3)
    millions(top, "y")
    top.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    reads = [total(by_day[d], ("cache_read_input_tokens",)) for d in days]
    bottom.bar(days, reads, color=NEUTRAL, width=0.62)
    for i, v in enumerate(reads):
        bottom.text(i, v, f" {v / 1e6:,.1f}M", ha="center", va="bottom", fontsize=8, color=INK)
    bottom.set_ylabel("cache read")
    millions(bottom, "y")
    for ax in (top, bottom):
        clean(ax)
    finish(fig, "token-days.png", "Tokens per day",
           "Top: new tokens by kind. Bottom: cache reads, the conversation re-read each turn (a different scale).")

    def hbars(name, title, sub, groups, stack_by=None):
        names = sorted(groups, key=lambda g: total(groups[g], NEW))
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.9))
        keys = sorted({stack_by(r) for g in groups.values() for r in g},
                      key=lambda k: -total([r for g in groups.values() for r in g if stack_by(r) == k], NEW)) \
            if stack_by else [None]
        base = [0] * len(names)
        for i, key in enumerate(keys):
            vals = [total([r for r in groups[g] if stack_by is None or stack_by(r) == key], NEW) for g in names]
            ax.barh(names, vals, left=base, color=SERIES[min(i, 2)] if stack_by else SERIES[0],
                    edgecolor=SURFACE, linewidth=1.5, height=0.6, label=key)
            base = [b + v for b, v in zip(base, vals)]
        for y, v in enumerate(base):
            ax.text(v, y, f" {v / 1e3:,.0f}k", va="center", fontsize=8, color=INK)
        ax.set_xlabel("new tokens")
        millions(ax, "x")
        ax.set_xlim(0, max(base) * 1.15 if base else 1)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if stack_by and len(keys) > 1:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        elif stack_by:
            sub += f" Model: {keys[0]}."
        finish(fig, name, title, sub)

    hbars("token-tasks.png", "Tokens per task, by model",
          "New tokens (output + cache write + fresh input) by git branch where the session ran.",
          group(responses, lambda r: r.branch), stack_by=lambda r: r.model)
    hbars("token-agents.png", "Tokens per agent",
          "New tokens. Subagent output is a lower bound: its transcript records it at the start of the stream.",
          group(responses, lambda r: r.agent))
    main_only = [r for r in responses if not r.lower_bound]
    work = group(main_only, lambda r: r.work)
    names = sorted(work, key=lambda w: total(work[w], ("output_tokens",)))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.9))
    vals = [total(work[w], ("output_tokens",)) for w in names]
    ax.barh(names, vals, color=SERIES[0], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v / 1e3:,.0f}k", va="center", fontsize=8, color=INK)
    ax.set_xlabel("output tokens")
    millions(ax, "x")
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    finish(fig, "token-work.png", "What the output tokens went to",
           "Main sessions. Each response is filed under the first tool it called; none means answering.")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--projects", nargs="*", type=Path, help="project folders (default: this repo's)")
    parser.add_argument("--out", type=Path, help="output folder (default: .claude/handoff)")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args(argv)
    dirs = args.projects or project_dirs()
    if not dirs:
        print("No Claude Code project folders found for this repo.", file=sys.stderr)
        return 1
    sessions = [s for d in dirs for s in read_project(d)]
    out = args.out or main_checkout() / ".claude" / "handoff"
    out.mkdir(parents=True, exist_ok=True)
    (out / "LOG.md").write_text(build_log(sessions), encoding="utf-8")
    print(f"wrote {out / 'LOG.md'}")
    if not args.no_charts:
        try:
            for path in charts([r for s in sessions for r in s.responses], out / "token-charts"):
                print(f"wrote {path}")
        except ImportError:
            print("matplotlib is not installed; skipped the charts.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
