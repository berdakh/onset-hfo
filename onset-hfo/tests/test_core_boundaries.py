"""The pipeline and command line work without the optional desktop package."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_has_no_desktop_imports():
    violations = []
    for path in (ROOT / "onset_hfo").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split(".")[0] == "onset_review" for name in modules):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "Desktop imports in core: " + ", ".join(violations)


def test_cli_report_never_imports_desktop_or_qt(tmp_path):
    script = textwrap.dedent('''
        import importlib.abc
        import sys

        blocked = {"onset_review", "qtpy", "PySide6", "PySide2", "PyQt6", "PyQt5"}

        class NoDesktop(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise AssertionError(f"Forbidden desktop import: {fullname}")

        sys.meta_path.insert(0, NoDesktop())
        from onset_hfo.cli import main
        assert main(["report-problem", "--note", "core-only report", "--yes",
                     "--out", sys.argv[1]]) == 0
        assert not any(name.split(".")[0] in blocked for name in sys.modules)
    ''')
    out = tmp_path / "report.zip"
    done = subprocess.run([sys.executable, "-c", script, str(out)], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    with zipfile.ZipFile(out) as zipped:
        assert zipped.namelist() == ["report.txt"]
        report = zipped.read("report.txt").decode("utf-8")
    assert "core-only report" in report
    assert "Versions and machine" in report
    assert f"onset-hfo: {version('onset-hfo')}" in report.splitlines()
    assert report in done.stdout


def test_report_without_package_metadata_keeps_the_pipeline_version(monkeypatch):
    from onset_hfo import problemreport
    from onset_hfo.config import PIPELINE_VERSION

    def missing_version(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(problemreport, "version", missing_version)
    report = problemreport.report_text("source checkout")
    assert "onset-hfo: unknown (package metadata unavailable)" in report.splitlines()
    assert f"pipeline: {PIPELINE_VERSION}" in report.splitlines()
