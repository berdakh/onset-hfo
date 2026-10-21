"""Rules about the repository itself, which used to live only in prose.

* ``onset_hfo`` and ``onset_agent`` do not depend on the desktop app or on Qt. Three imports of
  ``onset_review`` from ``onset_hfo`` already exist (two in ``cli.py``, one in
  ``case/imaging.py``) and are listed below with how many times each appears. One more, even a
  repeat in the same file, fails the test; so does fixing one without taking it off the list, so
  the list can only shrink.
* Nothing that must stay local is tracked: the preprint folder, PDFs at the top, run artifacts.
* The package version is the same in the three places it is written.
* The tests are importing this checkout, not another one. With a worktree and an editable
  install made elsewhere, ``pytest`` can pass against code that is not the code under review.
"""

from __future__ import annotations

import ast
import re
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import pytest

import onset_hfo

ROOT = Path(__file__).resolve().parents[1]  # the package folder, with pyproject.toml
GIT_ROOT = ROOT.parent

#: Imports of the desktop app from the pipeline that exist today: (file, module) -> how many
#: import statements. Lower or remove an entry when an import goes; do not add one.
KNOWN_REVIEW_IMPORTS = Counter({
    ("onset_hfo/cli.py", "onset_review.problemreport"): 2,
    ("onset_hfo/case/imaging.py", "onset_review.studycharts"): 1,
})
QT = ("PySide6", "PyQt5", "PyQt6", "qtpy")


def imported_modules(package: str) -> Counter:
    """(file, module) -> number of import statements, lazy ones inside functions included."""
    found: Counter = Counter()
    for path in sorted((ROOT / package).rglob("*.py")):
        with warnings.catch_warnings():  # a file with an old escape sequence warns when parsed
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found[(path.relative_to(ROOT).as_posix(), name)] += 1
    return found


def importing(package: str, top_level: tuple[str, ...]) -> Counter:
    return Counter({key: n for key, n in imported_modules(package).items() if key[1].split(".")[0] in top_level})


def test_the_pipeline_imports_the_desktop_app_only_where_it_already_did():
    found = importing("onset_hfo", ("onset_review",))
    assert found == KNOWN_REVIEW_IMPORTS, (
        "onset_hfo must not import onset_review. More than before: "
        f"{dict(found - KNOWN_REVIEW_IMPORTS)}; fewer (lower the count in KNOWN_REVIEW_IMPORTS): "
        f"{dict(KNOWN_REVIEW_IMPORTS - found)}")


@pytest.mark.parametrize("package", ["onset_hfo", "onset_agent"])
def test_no_qt_below_the_desktop_app(package):
    qt = importing(package, QT)
    assert not qt, f"{package} must not import Qt: {dict(qt)}"


def test_the_agent_does_not_import_the_desktop_app():
    found = importing("onset_agent", ("onset_review",))
    assert not found, f"onset_agent must not import onset_review: {dict(found)}"


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(GIT_ROOT), "ls-files"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.splitlines()


def test_nothing_that_stays_local_is_tracked():
    bad = []
    for name in tracked_files():
        parts = name.split("/")
        if "paper" in parts[:2] or parts[:2] == ["onset-hfo", "artifacts"]:
            bad.append(name)
        elif len(parts) == 1 and name.lower().endswith(".pdf"):
            bad.append(name)
    assert not bad, f"these must stay untracked (see .gitignore): {bad}"


def test_the_version_is_the_same_everywhere_it_is_written():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    review = re.search(r'^__version__ = "([^"]+)"', (ROOT / "onset_review" / "__init__.py").read_text(), re.M)
    versions = {"pyproject.toml": pyproject.group(1), "onset_review/__init__.py": review.group(1)}
    cff = GIT_ROOT / "CITATION.cff"
    if cff.exists():
        versions["CITATION.cff"] = re.search(r'^version: "?([^"\n]+)"?', cff.read_text(), re.M).group(1)
    assert len(set(versions.values())) == 1, f"versions differ: {versions}"


def test_the_tests_import_this_checkout():
    here = Path(onset_hfo.__file__).resolve()
    assert here.is_relative_to(ROOT), (
        f"onset_hfo was imported from {here}, not from {ROOT}. Install this checkout "
        "(pip install -e . from the package folder) before testing it.")
