from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RUNTIME_FILES = [
    "__init__.py",
    "plugin.yaml",
    "provider.py",
    "config.py",
    "cli.py",
]

pytestmark = pytest.mark.contract


def _install_flat_plugin(hermes_home: Path) -> Path:
    target = hermes_home / "plugins" / "sibyl"
    target.mkdir(parents=True)
    for name in RUNTIME_FILES:
        shutil.copy2(ROOT / name, target / name)
    return target


def _clear_loaded_plugin() -> None:
    for name in list(sys.modules):
        if name == "_hermes_user_memory.sibyl" or name.startswith("_hermes_user_memory.sibyl."):
            sys.modules.pop(name)


def test_real_0182_flat_discovery_and_direct_setup(tmp_path, monkeypatch, capsys):
    pytest.importorskip("plugins.memory")
    assert importlib.metadata.version("hermes-agent") == "0.18.2"

    hermes_home = tmp_path / "hermes"
    _install_flat_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SIBYL_API_KEY", raising=False)
    _clear_loaded_plugin()

    from hermes_cli import config as hermes_config
    from hermes_cli import memory_setup, secret_prompt
    from plugins import memory

    saved_config: dict = {}
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: None)
    monkeypatch.setattr(hermes_config, "load_config", lambda: {"memory": {}})
    monkeypatch.setattr(hermes_config, "save_config", saved_config.update)
    monkeypatch.setattr(secret_prompt, "masked_secret_prompt", lambda prompt: "secret-value")
    answers = iter(["", "project_test", "space_test"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    provider = memory.load_memory_provider("sibyl")
    assert provider is not None
    assert provider.name == "sibyl"
    assert provider.get_tool_schemas() == []

    memory_setup.cmd_setup_provider("sibyl")

    assert saved_config["memory"]["provider"] == "sibyl"
    assert (hermes_home / "sibyl.json").exists()
    assert "secret-value" not in (hermes_home / "sibyl.json").read_text(encoding="utf-8")
    assert "secret-value" in (hermes_home / ".env").read_text(encoding="utf-8")

    monkeypatch.setattr(hermes_config, "load_config", lambda: saved_config)
    memory_setup.cmd_status(type("Args", (), {})())
    output = capsys.readouterr().out
    assert "sibyl" in output.lower()
    assert "available" in output.lower()

    commands = memory.discover_plugin_cli_commands()
    assert len(commands) == 1
    assert commands[0]["name"] == "sibyl"
    parser = argparse.ArgumentParser()
    commands[0]["setup_fn"](parser)
    assert parser.parse_args(["status"]).sibyl_command == "status"
