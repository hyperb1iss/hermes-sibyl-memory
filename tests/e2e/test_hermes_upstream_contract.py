from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RUNTIME_FILES = (
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
)

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_SIBYL_UPSTREAM_CONTRACT") != "1",
    reason="upstream-main compatibility runs only in its allowed-failure tracker",
)


def _install_flat_plugin(hermes_home: Path) -> None:
    target = hermes_home / "plugins" / "sibyl"
    target.mkdir(parents=True)
    for name in RUNTIME_FILES:
        shutil.copy2(ROOT / name, target / name)


def _clear_plugin_modules() -> None:
    for name in tuple(sys.modules):
        if name == "_hermes_user_memory.sibyl" or name.startswith("_hermes_user_memory.sibyl."):
            sys.modules.pop(name)


def test_upstream_main_flat_discovery_and_three_tool_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("plugins.memory")
    hermes_home = tmp_path / "hermes"
    _install_flat_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SIBYL_API_KEY", "contract-secret")
    (hermes_home / "sibyl.json").write_text(
        '{"base_url":"https://sibyl.example/api","project_id":"project_contract",'
        '"memory_space_id":"space_contract"}',
        encoding="utf-8",
    )
    _clear_plugin_modules()

    from agent.memory_manager import MemoryManager
    from plugins import memory

    handled: list[str] = []

    class Runtime:
        def initialize(self, context):
            return None

        def on_turn_start(self, turn_number, message, **kwargs):
            return None

        def prefetch(self, query, *, session_id=""):
            return ""

        def queue_prefetch(self, query, *, session_id=""):
            return None

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            return None

        def on_session_switch(self, new_session_id, **kwargs):
            return None

        def get_tool_schemas(self):
            return [
                {"name": name, "description": name, "parameters": {}}
                for name in ("sibyl_recall", "sibyl_remember", "sibyl_correct")
            ]

        def handle_tool_call(self, tool_name, args, **kwargs):
            handled.append(tool_name)
            return '{"handled":true}'

        def shutdown(self):
            return None

    provider = memory.load_memory_provider("sibyl")
    assert provider is not None
    assert provider.name == "sibyl"
    provider._runtime_factory = lambda context: Runtime()

    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all("upstream-contract", hermes_home=str(hermes_home))
    for tool_name in ("sibyl_recall", "sibyl_remember", "sibyl_correct"):
        assert manager.has_tool(tool_name)
        assert manager.handle_tool_call(tool_name, {}) == '{"handled":true}'
    manager.shutdown_all()

    assert handled == ["sibyl_recall", "sibyl_remember", "sibyl_correct"]
