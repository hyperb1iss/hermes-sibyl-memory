from __future__ import annotations

from typing import Any

from config import ProviderConfig, save_api_key, save_provider_config


class RecordingRuntime:
    def __init__(self) -> None:
        self.context: Any | None = None
        self.turns: list[tuple[int, str]] = []
        self.closed = False

    def initialize(self, context) -> None:
        self.context = context

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        self.turns.append((turn_number, message))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return f"{session_id}:{query}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        return None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        return None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        return "{}"

    def shutdown(self) -> None:
        self.closed = True


def test_provider_delegates_through_frozen_runtime_seam(tmp_path, plugin_module):
    provider_module = __import__(f"{plugin_module.__name__}.provider", fromlist=["provider"])
    save_provider_config(
        ProviderConfig.from_mapping(
            {
                "base_url": "https://sibyl.example/api",
                "project_id": "project_test",
                "memory_space_id": "space_test",
            }
        ),
        tmp_path,
    )
    save_api_key("secret-value", tmp_path)
    runtime = RecordingRuntime()
    provider = provider_module.SibylMemoryProvider(runtime_factory=lambda context: runtime)

    provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        platform="signal",
        agent_workspace="home",
        agent_identity="nova",
    )
    provider.on_turn_start(1, "hello")

    assert runtime.context is not None
    assert runtime.context.agent_id == "hermes:home:nova"
    assert runtime.turns == [(1, "hello")]
    assert provider.prefetch("question", session_id="session-1") == "session-1:question"
    assert "untrusted evidence" in provider.system_prompt_block()
    provider.shutdown()
    assert runtime.closed is True
