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
    "capture.py",
    "client.py",
    "outbox.py",
    "recall.py",
    "runtime.py",
    "schemas.py",
    "sessions.py",
    "provider_tools.py",
    "trust.py",
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
    plugin_config = sys.modules["_hermes_user_memory.sibyl.config"]
    monkeypatch.setattr(plugin_config, "_validate_setup_credential", lambda *args, **kwargs: None)

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


def test_real_0182_memory_manager_routes_lifecycle_and_tools(tmp_path, monkeypatch):
    pytest.importorskip("plugins.memory")
    hermes_home = tmp_path / "hermes"
    _install_flat_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SIBYL_API_KEY", "secret-value")
    (hermes_home / "sibyl.json").write_text(
        '{"base_url":"https://sibyl.example/api","project_id":"project_test",'
        '"memory_space_id":"space_test"}',
        encoding="utf-8",
    )
    _clear_loaded_plugin()

    from agent.memory_manager import MemoryManager
    from plugins import memory

    events = []

    class RecordingRuntime:
        def initialize(self, context):
            events.append(("initialize", context.session_id, context.agent_id))

        def on_turn_start(self, turn_number, message, **kwargs):
            events.append(("turn_start", turn_number, message))

        def prefetch(self, query, *, session_id=""):
            events.append(("prefetch", query, session_id))
            return "# Untrusted evidence"

        def queue_prefetch(self, query, *, session_id=""):
            events.append(("queue_prefetch", query, session_id))

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            events.append(("sync", user_content, assistant_content, session_id, messages))

        def on_session_switch(self, new_session_id, **kwargs):
            events.append(("switch", new_session_id, kwargs))

        def get_tool_schemas(self):
            return [{"name": "sibyl_recall", "description": "Recall", "parameters": {}}]

        def handle_tool_call(self, tool_name, args, **kwargs):
            return '{"handled":true}'

        def shutdown(self):
            events.append(("shutdown",))

    provider = memory.load_memory_provider("sibyl")
    assert provider is not None
    recording = RecordingRuntime()
    provider._runtime_factory = lambda context: recording
    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(
        "session-1",
        hermes_home=str(hermes_home),
        platform="signal",
        agent_workspace="home",
        agent_identity="nova",
    )
    manager.on_turn_start(1, "current")
    assert manager.prefetch_all("current", session_id="session-1") == "# Untrusted evidence"
    manager.sync_all(
        "current",
        "answer",
        session_id="session-1",
        messages=[{"role": "user", "content": "current"}],
    )
    assert manager.flush_pending(timeout=2.0)
    assert manager.has_tool("sibyl_recall")
    assert manager.handle_tool_call("sibyl_recall", {"query": "memory"}) == '{"handled":true}'
    manager.shutdown_all()

    assert events[0] == ("initialize", "session-1", "hermes:home:nova")
    assert events[1] == ("turn_start", 1, "current")
    assert events[2] == ("prefetch", "current", "session-1")
    assert events[3][0] == "sync"
    assert events[-1] == ("shutdown",)


def test_real_0182_skill_scaffolding_normalizes_to_user_instruction(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _install_flat_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _clear_loaded_plugin()

    from plugins import memory

    assert memory.load_memory_provider("sibyl") is not None

    runtime = __import__(
        "_hermes_user_memory.sibyl.runtime",
        fromlist=["normalize_hermes_user_message"],
    )
    expanded = (
        "[IMPORTANT: The user has invoked the example skill. "
        "The full skill content is loaded below.]\nsecret scaffolding\n"
        "The user has provided the following instruction alongside the skill invocation: "
        "remember only this request\n\n[Runtime note: hidden]"
    )

    assert runtime.normalize_hermes_user_message(expanded) == "remember only this request"
