"""`scripts/pin_catalogue.py`: the rewrite is exercised on the real source.

The Hub itself is not reachable from CI, so the resolution step is stood in
for by a fake. What is tested for real is the part that can corrupt a file:
the rewrite runs on the shipped `hardware.py`, the result is imported as a
module, and the catalogue it defines is checked entry by entry.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from onset_agent import hardware
from onset_agent.hardware import CATALOGUE, ModelSpec

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pin_catalogue  # noqa: E402

SOURCE = Path(hardware.__file__).read_text(encoding="utf-8")


def fake_shas(seed: str = "a") -> dict[str, str]:
    return {spec.model_id: (seed * 40)[:36] + f"{i:04d}"
            for i, spec in enumerate(CATALOGUE)}


def load(source: str, tmp_path: Path, name: str = "hardware_rewritten"):
    """Import `source` as a module and return its CATALOGUE."""
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"onset_agent.{name}", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module.CATALOGUE


# -- the rewrite itself ----------------------------------------------------


def test_the_shipped_catalogue_is_still_unpinned():
    """Documents the state this repository is in. When the script has been run
    on a machine that sees the Hub, this test is the one to delete."""
    assert all(spec.revision == "main" for spec in CATALOGUE)


def test_rewrite_pins_every_entry_of_the_real_source(tmp_path):
    shas = fake_shas()
    pinned = load(pin_catalogue.rewrite(SOURCE, shas), tmp_path)
    assert {s.model_id: s.revision for s in pinned} == shas
    # Nothing else about the entries moved.
    for before, after in zip(CATALOGUE, pinned, strict=True):
        # Field by field: the rewritten module defines its own ModelSpec class,
        # and dataclass equality is class-bound.
        assert asdict(replace(after, revision="main")) == asdict(before)


def test_rewrite_touches_only_the_revision_lines():
    out = pin_catalogue.rewrite(SOURCE, fake_shas())
    added = [line for line in out.splitlines() if line not in SOURCE.splitlines()]
    assert len(added) == len(CATALOGUE)
    assert all(line.strip().startswith('revision="') for line in added)
    assert all(len(line) <= 110 for line in added), "would fail ruff's line length"


def test_rewrite_is_a_replacement_the_second_time(tmp_path):
    once = pin_catalogue.rewrite(SOURCE, fake_shas("a"))
    twice = pin_catalogue.rewrite(once, fake_shas("b"))
    assert twice.count('revision="') == once.count('revision="')
    assert {s.model_id: s.revision for s in load(twice, tmp_path)} == fake_shas("b")


def test_rewrite_leaves_an_unlisted_entry_alone(tmp_path):
    shas = fake_shas()
    only_one = {CATALOGUE[0].model_id: shas[CATALOGUE[0].model_id]}
    pinned = load(pin_catalogue.rewrite(SOURCE, only_one), tmp_path)
    assert pinned[0].revision == only_one[CATALOGUE[0].model_id]
    assert all(s.revision == "main" for s in pinned[1:])


def test_a_short_or_invented_hash_is_refused():
    with pytest.raises(pin_catalogue.CatalogueEditError):
        pin_catalogue.rewrite(SOURCE, {CATALOGUE[0].model_id: "abc123"})
    with pytest.raises(pin_catalogue.CatalogueEditError):
        pin_catalogue.rewrite(SOURCE, {CATALOGUE[0].model_id: "main"})


def test_an_entry_the_source_does_not_have_is_an_error_not_a_silent_skip():
    with pytest.raises(pin_catalogue.CatalogueEditError, match="exactly one"):
        pin_catalogue.rewrite(SOURCE, {"Qwen/Qwen9-999B": "f" * 40})


# -- the command -----------------------------------------------------------


@pytest.fixture
def hub(monkeypatch):
    """Stand in for the Hub: `pinned` returns the catalogue with these SHAs."""
    shas = fake_shas("c")

    def pinned(catalogue=CATALOGUE):
        return tuple(replace(s, revision=shas[s.model_id]) for s in catalogue)

    monkeypatch.setattr(hardware, "pinned", pinned)
    return shas


def test_check_fails_while_the_catalogue_is_unpinned(hub, capsys):
    assert pin_catalogue.main(["--check"]) == 1
    err = capsys.readouterr().err
    assert "unpinned or stale" in err


def test_dry_run_reports_and_writes_nothing(hub, tmp_path, capsys):
    copy = tmp_path / "hardware.py"
    copy.write_text(SOURCE, encoding="utf-8")
    assert pin_catalogue.main(["--dry-run", "--file", str(copy)]) == 0
    assert copy.read_text(encoding="utf-8") == SOURCE
    out = capsys.readouterr().out
    assert all(model_id in out for model_id in hub)


def test_the_default_run_writes_the_pins_into_the_file(hub, tmp_path):
    copy = tmp_path / "hardware.py"
    copy.write_text(SOURCE, encoding="utf-8")
    assert pin_catalogue.main(["--file", str(copy)]) == 0
    pinned = load(copy.read_text(encoding="utf-8"), tmp_path, "hardware_written")
    assert {s.model_id: s.revision for s in pinned} == hub


def test_an_unreachable_hub_is_reported_not_raised(monkeypatch, capsys):
    def pinned(catalogue=CATALOGUE):
        raise OSError("403 Forbidden: CONNECT huggingface.co")

    monkeypatch.setattr(hardware, "pinned", pinned)
    assert pin_catalogue.main(["--check"]) == 2
    err = capsys.readouterr().err
    assert "could not reach the Hub" in err and "huggingface.co" in err


def test_a_missing_hub_client_names_the_extra(monkeypatch, capsys):
    def pinned(catalogue=CATALOGUE):
        raise ImportError("No module named 'huggingface_hub'")

    monkeypatch.setattr(hardware, "pinned", pinned)
    assert pin_catalogue.main(["--check"]) == 2
    assert "[llm]" in capsys.readouterr().err


def test_pinned_passes_each_sha_through_unchanged(monkeypatch):
    """`hardware.pinned` is what the script trusts; a fake HfApi checks that it
    asks about every model and keeps the answer."""
    calls = []

    class Info:
        def __init__(self, sha):
            self.sha = sha

    class Api:
        def model_info(self, model_id):
            calls.append(model_id)
            return Info(f"{len(calls):040x}")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].HfApi = Api
    out = hardware.pinned((ModelSpec("x/one", 1.0, 2.0), ModelSpec("x/two", 1.0, 2.0)))
    assert calls == ["x/one", "x/two"]
    assert [s.revision for s in out] == [f"{1:040x}", f"{2:040x}"]
