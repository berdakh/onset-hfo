"""Which model the assistant opens on, and where that decision comes from.

Qt-free on purpose: the decision is a function of the environment, a file and
a probe, and all three are injected here so no window and no server is needed
to test every branch.
"""

from __future__ import annotations

import json

import pytest

from onset_review import assistant_config as ac


def test_nothing_configured_means_no_model(tmp_path):
    chosen = ac.load_defaults(env={}, path=tmp_path / "none.json",
                              probe=lambda: (False, []))
    assert chosen.kind == "scripted"
    assert chosen.model == ""
    assert "nothing" in chosen.source


def test_the_installers_file_wins_over_a_live_probe(tmp_path):
    path = ac.write_defaults("ollama", "qwen3:8b", "http://127.0.0.1:11434/v1",
                             path=tmp_path / "assistant.json")
    chosen = ac.load_defaults(env={}, path=path,
                              probe=lambda: pytest.fail("the file should settle it"))
    assert chosen.kind == "ollama" and chosen.model == "qwen3:8b"
    assert "config file" in chosen.source


def test_the_environment_wins_over_the_file(tmp_path):
    path = ac.write_defaults("ollama", "qwen3:8b", path=tmp_path / "assistant.json")
    env = {"ONSET_ASSISTANT_BACKEND": "openai_compat",
           "ONSET_ASSISTANT_MODEL": "my-model",
           "ONSET_ASSISTANT_BASE_URL": "http://box:8000/v1"}
    chosen = ac.load_defaults(env=env, path=path, probe=lambda: (True, ["qwen3:8b"]))
    assert chosen.kind == "openai_compat"
    assert chosen.model == "my-model" and chosen.base_url == "http://box:8000/v1"
    assert chosen.source == "environment"


def test_a_live_ollama_with_a_catalogue_model_is_noticed(tmp_path):
    """The reviewer already pulled a model; the panel should not open on
    'No model' and make them find the option."""
    chosen = ac.load_defaults(env={}, path=tmp_path / "none.json",
                              probe=lambda: (True, ["llama3:8b", "qwen2.5:7b-instruct"]))
    assert chosen.kind == "ollama"
    assert chosen.model == "qwen2.5:7b-instruct"
    assert chosen.base_url == ac.DEFAULT_BASE_URL
    assert "localhost" in chosen.source


def test_the_largest_pulled_catalogue_model_is_preferred(tmp_path):
    chosen = ac.load_defaults(env={}, path=tmp_path / "none.json",
                              probe=lambda: (True, ["qwen3:1.7b", "qwen3:14b", "qwen3:4b"]))
    assert chosen.model == "qwen3:14b"


def test_a_live_ollama_with_only_unknown_models_is_left_alone(tmp_path):
    """A pulled llama is not a reason to guess at a tag the agent was never
    tried with."""
    chosen = ac.load_defaults(env={}, path=tmp_path / "none.json",
                              probe=lambda: (True, ["llama3:8b", "mistral:7b"]))
    assert chosen.kind == "scripted"


def test_the_probe_can_be_switched_off(tmp_path):
    chosen = ac.load_defaults(env={"ONSET_ASSISTANT_NO_PROBE": "1"},
                              path=tmp_path / "none.json",
                              probe=lambda: pytest.fail("must not probe"))
    assert chosen.kind == "scripted"


def test_a_failing_probe_is_not_a_crash(tmp_path):
    def broken():
        raise OSError("no network stack")
    assert ac.load_defaults(env={}, path=tmp_path / "x.json", probe=broken).kind == "scripted"


def test_a_malformed_or_foreign_file_is_ignored(tmp_path):
    bad = tmp_path / "assistant.json"
    bad.write_text("{not json", encoding="utf-8")
    assert ac.load_defaults(env={}, path=bad, probe=lambda: (False, [])).kind == "scripted"
    bad.write_text(json.dumps({"kind": "telepathy"}), encoding="utf-8")
    assert ac.load_defaults(env={}, path=bad, probe=lambda: (False, [])).kind == "scripted"


def test_an_unknown_kind_is_refused_on_write(tmp_path):
    with pytest.raises(ValueError):
        ac.write_defaults("telepathy", path=tmp_path / "assistant.json")


def test_the_config_path_honours_the_override_and_xdg(tmp_path):
    assert ac.config_path({"ONSET_REVIEW_CONFIG_DIR": str(tmp_path)}) == tmp_path / "assistant.json"
    xdg = ac.config_path({"XDG_CONFIG_HOME": str(tmp_path / "cfg")})
    assert xdg == tmp_path / "cfg" / "onset-review" / "assistant.json"


def test_writing_is_atomic_and_round_trips(tmp_path):
    path = ac.write_defaults("ollama", "qwen3:4b", path=tmp_path / "a" / "assistant.json")
    assert path.exists() and not path.with_suffix(".json.tmp").exists()
    payload = json.loads(path.read_text())
    assert payload == {"kind": "ollama", "model": "qwen3:4b", "base_url": "", "schema": 1}
