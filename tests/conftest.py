from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def _install_memory_provider_stub() -> None:
    try:
        importlib.import_module("agent.memory_provider")
    except ImportError:
        agent_module = types.ModuleType("agent")
        agent_module.__path__ = []
        memory_provider = types.ModuleType("agent.memory_provider")

        class MemoryProvider:
            pass

        memory_provider.__dict__["MemoryProvider"] = MemoryProvider
        sys.modules["agent"] = agent_module
        sys.modules["agent.memory_provider"] = memory_provider


@pytest.fixture
def plugin_module():
    _install_memory_provider_stub()
    module_name = "_hermes_test_sibyl"
    for loaded_name in list(sys.modules):
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
            sys.modules.pop(loaded_name)

    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
