"""The Linux installer's assistant stage, without touching the machine.

Two ways in. ``--dry-run`` prints every command instead of running it, so the
plan can be read as text; and a real run against a fake ``ollama`` and a fake
``systemctl`` on PATH exercises the branches that write files -- the config the
assistant panel reads, the launcher -- with ``--skip-install`` so no venv is
built and nothing is downloaded.

Nothing here needs sudo, a network, or Ollama.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
INSTALLER = REPO / "packaging" / "install-ubuntu.sh"

pytestmark = pytest.mark.skipif(not INSTALLER.exists(), reason="installer not present")


def _run(args, *, env_extra=None, cwd=REPO, timeout=120):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(["bash", str(INSTALLER), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _home(tmp_path: Path) -> dict:
    home = tmp_path / "home"
    home.mkdir()
    return {"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"),
            "ONSET_REVIEW_CONFIG_DIR": str(tmp_path / "cfg")}


# -- the plan, as text ---------------------------------------------------------


def test_help_mentions_the_assistant():
    done = _run(["--help"])
    assert done.returncode == 0
    assert "--with-assistant" in done.stdout
    assert "--dry-run" in done.stdout


def test_a_dry_run_plans_the_assistant_and_changes_nothing(tmp_path):
    env = _home(tmp_path)
    prefix = tmp_path / "prefix"
    done = _run(["--with-assistant", "--no-apt", "--dry-run", "--prefix", str(prefix)],
                env_extra=env)
    assert done.returncode == 0, done.stderr
    plan = done.stdout
    assert "ollama pull qwen2.5:7b-instruct" in plan
    assert "OLLAMA_CONTEXT_LENGTH=8192" in plan
    assert "write_defaults('ollama', 'qwen2.5:7b-instruct'" in plan
    assert "would be qwen2.5:7b-instruct via Ollama (dry run)" in plan
    # Nothing was created: no venv, no launcher, no config.
    assert not prefix.exists()
    assert not (Path(env["HOME"]) / ".local").exists()
    assert not Path(env["ONSET_REVIEW_CONFIG_DIR"]).exists()


def test_a_named_model_is_what_gets_pulled(tmp_path):
    done = _run(["--assistant-model", "qwen3:8b", "--no-apt", "--dry-run",
                 "--prefix", str(tmp_path / "p")], env_extra=_home(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "ollama pull qwen3:8b" in done.stdout
    assert "(as asked)" in done.stdout
    assert "qwen2.5:7b-instruct" not in done.stdout


def test_no_pull_defers_the_download_and_says_so(tmp_path):
    done = _run(["--with-assistant", "--no-pull", "--no-apt", "--dry-run",
                 "--prefix", str(tmp_path / "p")], env_extra=_home(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "ollama pull" not in [line.strip().lstrip("$ ") for line in done.stdout.splitlines()
                                 if line.strip().startswith("$")]
    assert "Not pulling (--no-pull)" in done.stdout


def test_without_the_flag_no_model_is_touched(tmp_path):
    done = _run(["--no-apt", "--dry-run", "--prefix", str(tmp_path / "p")],
                env_extra=_home(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "ollama" not in done.stdout.lower()


def test_an_unknown_option_is_refused():
    assert _run(["--with-assistants"]).returncode != 0


# -- a real run against fakes --------------------------------------------------


def _shims(tmp_path: Path) -> tuple[Path, Path]:
    """A fake ollama that records its calls, and a systemctl with no unit."""
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "ollama.log"
    ollama = shim / "ollama"
    ollama.write_text("#!/usr/bin/env bash\n"
                      f"echo \"$@\" >> '{log}'\n"
                      "case \"$1\" in serve) exit 0 ;; pull) exit 0 ;; *) exit 0 ;; esac\n")
    systemctl = shim / "systemctl"
    systemctl.write_text("#!/usr/bin/env bash\n# no ollama.service here\nexit 0\n")
    for path in (ollama, systemctl):
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return shim, log


def test_the_assistant_stage_writes_what_the_panel_reads(tmp_path):
    """--skip-install against an existing 'environment' (this interpreter),
    a fake ollama and no systemd unit: the branch a container or a minimal
    install takes. The config file is the contract with the assistant panel."""
    env = _home(tmp_path)
    shim, log = _shims(tmp_path)
    prefix = tmp_path / "prefix"
    (prefix / "venv" / "bin").mkdir(parents=True)
    (prefix / "venv" / "bin" / "python").symlink_to(sys.executable)
    env["PATH"] = f"{shim}:{os.environ['PATH']}"

    done = _run(["--with-assistant", "--skip-install", "--no-apt",
                 "--assistant-model", "qwen3:4b", "--prefix", str(prefix)],
                env_extra=env, timeout=180)
    assert done.returncode == 0, done.stderr + done.stdout

    # 1. the fake ollama was asked to serve and to pull exactly that tag
    calls = log.read_text().splitlines()
    assert any(call.startswith("serve") for call in calls)
    assert "pull qwen3:4b" in calls

    # 2. the config the assistant panel opens on
    config = Path(env["ONSET_REVIEW_CONFIG_DIR"]) / "assistant.json"
    assert config.exists(), done.stdout
    payload = json.loads(config.read_text())
    assert payload["kind"] == "ollama"
    assert payload["model"] == "qwen3:4b"
    assert payload["base_url"].endswith("/v1")

    # 3. the launcher and menu entry were installed under this HOME
    home = Path(env["HOME"])
    assert (home / ".local" / "bin" / "onset-review").exists()
    assert (home / ".local" / "share" / "applications" / "onset-review.desktop").exists()

    # 4. no server answered, and the summary says so instead of claiming success
    assert "is not reported by Ollama" in done.stderr + done.stdout
    assert "the panel opens on it" not in done.stdout


def test_uninstall_removes_the_config_and_leaves_ollama_alone(tmp_path):
    env = _home(tmp_path)
    shim, _ = _shims(tmp_path)
    env["PATH"] = f"{shim}:{os.environ['PATH']}"
    cfg = Path(env["ONSET_REVIEW_CONFIG_DIR"])
    cfg.mkdir()
    (cfg / "assistant.json").write_text("{}")
    done = _run(["--uninstall", "--prefix", str(tmp_path / "nothing")], env_extra=env)
    assert done.returncode == 0, done.stderr
    assert not (cfg / "assistant.json").exists()
    assert "left in place" in done.stdout + done.stderr


def test_bundle_mode_installs_the_wheel_beside_the_script(tmp_path):
    """make-release.sh puts install.sh at the bundle root next to the wheel,
    with no packaging/ and no pyproject.toml. The script has to notice that it
    *is* the project directory there, rather than looking one level up."""
    bundle = tmp_path / "onset-hfo-9.9.9-linux"
    bundle.mkdir()
    script = bundle / "install.sh"
    script.write_text(INSTALLER.read_text())
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (bundle / "onset_hfo-9.9.9-py3-none-any.whl").write_bytes(b"not really a wheel")
    for asset in ("onset-review.desktop", "onset-review.svg"):
        (bundle / asset).write_text((REPO / "packaging" / asset).read_text())
    env = _home(tmp_path)
    done = subprocess.run(["bash", str(script), "--no-apt", "--dry-run",
                           "--prefix", str(tmp_path / "p")],
                          cwd=bundle, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr + done.stdout
    assert "Installing the release wheel" in done.stdout
    assert "onset_hfo-9.9.9-py3-none-any.whl[review]" in done.stdout
    assert "pip install --quiet -e" not in done.stdout
