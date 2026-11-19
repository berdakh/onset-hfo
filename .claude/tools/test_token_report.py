"""Checks for token_report.py on a made-up project folder. Run: python3 -m unittest discover .claude/tools"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import token_report as tr  # noqa: E402


def row(kind, **extra):
    return {"type": kind, "timestamp": "2026-10-20T10:00:00.000Z", **extra}


def reply(mid, out, tool=None, branch="main"):
    """The rows a transcript holds for one response: one per content block, the same usage on each."""
    usage = {"input_tokens": 2, "output_tokens": out, "cache_creation_input_tokens": 100,
             "cache_read_input_tokens": 1000}
    last = {"type": "tool_use", "name": tool, "input": {"path": "SECRET-INPUT"}} if tool \
        else {"type": "text", "text": "SECRET-TEXT"}
    return [row("assistant", gitBranch=branch,
                message={"id": mid, "model": "m1", "usage": usage, "content": [block]})
            for block in ({"type": "thinking", "thinking": "SECRET-THOUGHT"}, last)]


class TokenReportTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        main = self.dir / "aaaa1111.jsonl"
        rows = [*reply("msg1", 50, "Read"), *reply("msg1", 50, "Read")[1:], *reply("msg2", 70),
                row("ai-title", title="SECRET-TITLE"), row("user", cwd="/x", message={"content": "SECRET-USER"}),
                {"type": "assistant", "message": {"id": "no-time", "usage": {"output_tokens": 9}}},
                row("cost-state", totalCostUSD=1.5,
                    modelUsage={"m1": {"outputTokens": 120, "cacheReadInputTokens": 2000}})]
        main.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        sub = self.dir / "aaaa1111" / "subagents"
        sub.mkdir(parents=True)
        (sub / "agent-x.jsonl").write_text("\n".join(json.dumps(r) for r in reply("msg3", 5, "Bash")),
                                           encoding="utf-8")
        (sub / "agent-x.meta.json").write_text(json.dumps({"agentType": "general-purpose"}), encoding="utf-8")
        self.sessions = tr.read_project(self.dir)
        self.responses = [r for s in self.sessions for r in s.responses]

    def test_a_response_logged_three_times_counts_once(self):
        main = [r for r in self.responses if r.agent == "main"]
        self.assertEqual(len(main), 2)
        self.assertEqual(tr.total(main, ("output_tokens",)), 120)

    def test_subagent_is_named_and_its_output_marked_a_lower_bound(self):
        sub = [r for r in self.responses if r.agent == "general-purpose"]
        self.assertEqual(len(sub), 1)
        self.assertTrue(sub[0].lower_bound)

    def test_work_is_the_first_tool_or_answering(self):
        """The tool call is on a later row than the thinking, and must still be found."""
        self.assertEqual(sorted(r.work for r in self.responses),
                         ["answering", "reading code", "running commands"])

    def test_log_has_the_tables_and_reconciles_with_the_snapshot(self):
        log = tr.build_log(self.sessions)
        for heading in ("## Total", "## By model", "## By task", "## By agent", "## Check against"):
            self.assertIn(heading, log)
        self.assertIn("| aaaa1111 | 120 | 120 | 2,000 | 2,000 | $1.50 |", log)

    def test_log_holds_counts_only(self):
        log = tr.build_log(self.sessions)
        for secret in ("SECRET-TEXT", "SECRET-THOUGHT", "SECRET-INPUT", "SECRET-TITLE", "SECRET-USER"):
            self.assertNotIn(secret, log)

    def test_a_row_without_a_timestamp_is_skipped_not_fatal(self):
        self.assertEqual(len([r for r in self.responses if r.agent == "main"]), 2)

    def test_unreadable_subagent_meta_leaves_it_unnamed(self):
        (self.dir / "aaaa1111" / "subagents" / "agent-x.meta.json").write_text("{not json", encoding="utf-8")
        agents = {r.agent for s in tr.read_project(self.dir) for r in s.responses}
        self.assertEqual(agents, {"main", "subagent"})

    def test_project_folders_are_matched_on_the_recorded_working_directory(self):
        base, root = self.dir / "projects", self.dir / "repo"
        for name, cwd in (("inside", root / "pkg"), ("worktree", self.dir / "wt" / "a"),
                          ("sibling", self.dir / "repo-other")):
            (base / name).mkdir(parents=True)
            (base / name / "s.jsonl").write_text(json.dumps({"cwd": str(cwd)}), encoding="utf-8")
        found = tr.project_dirs(base, roots=[root, self.dir / "wt" / "a"])
        self.assertEqual([p.name for p in found], ["inside", "worktree"])

    def test_charts_are_written(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        paths = tr.charts(self.responses, self.dir / "out")
        self.assertEqual(len(paths), 4)
        self.assertTrue(all(p.stat().st_size > 1000 for p in paths))


if __name__ == "__main__":
    unittest.main()
