"""Which model the assistant panel should open on, decided without Qt.

Until now the panel always opened on "No model" and left a reviewer to find the
Ollama option, type a tag, and discover whether a server was running. That is
the right default for a machine with nothing installed and the wrong one for a
machine the installer has just set up: a working model that nobody notices is
not working.

So the default is decided here, in order, and the panel only applies it:

1. environment -- ``ONSET_ASSISTANT_BACKEND`` / ``_MODEL`` / ``_BASE_URL``,
   for tests and for people who script their desktop;
2. the config file the installer writes (``assistant.json`` under
   ``$XDG_CONFIG_HOME/onset-review``, or ``ONSET_REVIEW_CONFIG_DIR``);
3. a live Ollama on localhost that has a catalogue model pulled -- probed with
   a short timeout, because a refused localhost connection is instant and a
   window should not wait on anything slower;
4. otherwise "No model", exactly as before.

Nothing here downloads anything or starts a server. It only looks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["AssistantDefaults", "CONFIG_NAME", "config_path", "load_defaults",
           "write_defaults"]

CONFIG_NAME = "assistant.json"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
KINDS = ("scripted", "ollama", "openai_compat")

#: Short on purpose: a refused connection on localhost returns immediately, so
#: this only ever elapses when something is listening and slow, and a window
#: opening should not wait on it.
PROBE_TIMEOUT_S = 0.4


@dataclass(frozen=True)
class AssistantDefaults:
    kind: str = "scripted"
    model: str = ""
    base_url: str = ""
    #: Where the decision came from, so the panel can say so.
    source: str = "nothing configured"

    def as_dict(self) -> dict:
        return asdict(self)


def config_path(env: dict | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("ONSET_REVIEW_CONFIG_DIR"):
        return Path(env["ONSET_REVIEW_CONFIG_DIR"]) / CONFIG_NAME
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "onset-review" / CONFIG_NAME


def write_defaults(kind: str, model: str = "", base_url: str = "",
                   path: Path | None = None) -> Path:
    """Record the installer's choice. Atomic, like the other files this
    project writes next to a reviewer's work."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "model": model, "base_url": base_url, "schema": 1}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _from_env(env: dict) -> AssistantDefaults | None:
    kind = (env.get("ONSET_ASSISTANT_BACKEND") or "").strip().lower()
    if not kind:
        return None
    if kind not in KINDS:
        return None
    return AssistantDefaults(kind=kind, model=env.get("ONSET_ASSISTANT_MODEL", "").strip(),
                             base_url=env.get("ONSET_ASSISTANT_BASE_URL", "").strip(),
                             source="environment")


def _from_file(path: Path) -> AssistantDefaults | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind", "")).strip().lower()
    if kind not in KINDS:
        return None
    return AssistantDefaults(kind=kind, model=str(payload.get("model", "")).strip(),
                             base_url=str(payload.get("base_url", "")).strip(),
                             source=f"config file {path}")


def _default_probe() -> tuple[bool, list[str]]:
    from onset_agent.hardware import ollama_status
    return ollama_status(timeout=PROBE_TIMEOUT_S)


def _from_live_ollama(probe) -> AssistantDefaults | None:
    """A running Ollama with a catalogue model pulled is as good as a config
    file: the reviewer already did the work, the panel should notice."""
    try:
        up, names = probe()
    except Exception:
        return None
    if not up or not names:
        return None
    from onset_agent.hardware import CATALOGUE

    known = [spec.ollama_tag for spec in CATALOGUE if spec.ollama_tag]
    lowered = [name.lower() for name in names]
    # The catalogue's own order is small-to-large; prefer the largest pulled,
    # since whoever pulled a 14B meant to use it.
    for tag in reversed(known):
        if tag in lowered or f"{tag}:latest" in lowered:
            return AssistantDefaults(kind="ollama", model=tag,
                                     base_url=DEFAULT_BASE_URL,
                                     source=f"Ollama on localhost has {tag}")
    return None


def load_defaults(env: dict | None = None, path: Path | None = None,
                  probe=None) -> AssistantDefaults:
    """The backend the panel should open on. See the module docstring for the
    order; every step is injectable so the decision is testable."""
    env = os.environ if env is None else env
    found = _from_env(env)
    if found is not None:
        return found
    found = _from_file(path or config_path(env))
    if found is not None:
        return found
    if env.get("ONSET_ASSISTANT_NO_PROBE"):
        return AssistantDefaults()
    found = _from_live_ollama(probe or _default_probe)
    if found is not None:
        return found
    return AssistantDefaults()
