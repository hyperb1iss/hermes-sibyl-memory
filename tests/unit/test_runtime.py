from __future__ import annotations

import importlib
from concurrent.futures import Executor, Future
from types import SimpleNamespace


class InlineExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class FakeClient:
    def __init__(self, schemas, *, offline: bool = False) -> None:
        self._schemas = schemas
        self.offline = offline
        self.context_requests = []
        self.exposures = []
        self.raw_requests = []
        self.closed = False

    def auth_me(self):
        if self.offline:
            client = importlib.import_module(self._schemas.__package__ + ".client")
            raise client.SibylTransportError(request_id="request-1", error_type="ConnectError")
        return self._schemas.AuthMeResponse(
            credential=self._schemas.Credential(
                type="api_key",
                id="key-1",
                scopes=("api:write",),
                project_ids=("project_home",),
                memory_space_ids=("space_home",),
                agent_id="hermes:home:nova",
                delegated_authority="household-agent",
                capability_profile="memory_provider",
            ),
            user={"id": "user-1"},
            organization={"id": "org-1"},
            org_role="member",
        )

    def context_pack(self, request, *, manual: bool = False):
        self.context_requests.append(request)
        return SimpleNamespace(
            markdown="# Recalled\n\nTreat this as evidence.",
            rendered_item_ids=("decision_1",),
            total_items=1,
        )

    def expose_context(self, request, *, idempotency_key: str):
        self.exposures.append((request, idempotency_key))
        return SimpleNamespace(mutation_receipt=SimpleNamespace(replayed=False))

    def remember_raw(self, request, *, idempotency_key: str):
        self.raw_requests.append((request, idempotency_key))
        return SimpleNamespace(
            revision=1,
            mutation_receipt=SimpleNamespace(replayed=False),
        )

    def preview_correction(self, source_id: str, request):
        return SimpleNamespace(
            target_lifecycle_state="active",
            current_revision=2,
            revision=2,
        )

    def apply_correction(self, source_id: str, request, *, idempotency_key: str):
        return SimpleNamespace(mutation_receipt=SimpleNamespace(replayed=False))

    def close(self) -> None:
        self.closed = True


def _runtime_parts(plugin_module, tmp_path, *, offline: bool = False):
    package = plugin_module.__name__
    config_module = importlib.import_module(f"{package}.config")
    provider_module = importlib.import_module(f"{package}.provider")
    runtime_module = importlib.import_module(f"{package}.runtime")
    outbox_module = importlib.import_module(f"{package}.outbox")
    schemas_module = importlib.import_module(f"{package}.schemas")
    context = provider_module.RuntimeContext(
        config=config_module.ProviderConfig.from_mapping(
            {
                "base_url": "https://sibyl.example/api",
                "project_id": "project_home",
                "memory_space_id": "space_home",
            }
        ),
        api_key="secret-value",
        hermes_home=str(tmp_path),
        session_id="session-1",
        parent_session_id="",
        platform="signal",
        agent_context="primary",
        agent_identity="nova",
        agent_workspace="home",
        agent_id="hermes:home:nova",
        kwargs={"user_id": "bliss", "thread_id": "thread-1"},
    )
    client = FakeClient(schemas_module, offline=offline)
    outbox = outbox_module.DurableOutbox(tmp_path)
    runtime = runtime_module.SibylRuntime(
        context,
        client=client,
        outbox=outbox,
        executor=InlineExecutor(),
    )
    return runtime, context, client, outbox


def test_recall_is_current_query_only_and_acknowledges_exact_ids(plugin_module, tmp_path):
    runtime, context, client, outbox = _runtime_parts(plugin_module, tmp_path)
    runtime.initialize(context)
    runtime.on_turn_start(1, "current question")

    markdown = runtime.prefetch("current question")

    assert markdown.startswith("# Recalled")
    assert client.context_requests[0].record_exposure is False
    assert client.context_requests[0].project == "project_home"
    assert client.exposures[0][0].exposed_ids == ("decision_1",)
    assert outbox.snapshot().total == 0


def test_completed_turn_preserves_only_explicit_user_and_final_assistant(plugin_module, tmp_path):
    runtime, context, client, outbox = _runtime_parts(plugin_module, tmp_path)
    runtime.initialize(context)
    runtime.on_turn_start(7, "user canary")

    runtime.sync_turn(
        "user canary",
        "assistant canary",
        messages=[
            {"role": "user", "content": "user canary"},
            {"role": "assistant", "content": None, "tool_calls": [{"name": "secret"}]},
            {"role": "tool", "content": "TOOL CANARY"},
            {"role": "assistant", "content": "assistant canary"},
        ],
    )

    request, idempotency_key = client.raw_requests[0]
    assert request.raw_content == "[User]\nuser canary\n\n[Assistant]\nassistant canary"
    assert "TOOL CANARY" not in request.raw_content
    assert request.metadata["turn_number"] == 7
    assert request.metadata["agent_id"] == "hermes:home:nova"
    assert idempotency_key.startswith("hermes-turn-")
    assert outbox.snapshot().total == 0


def test_offline_sibyl_keeps_completed_turn_durable(plugin_module, tmp_path):
    runtime, context, client, outbox = _runtime_parts(plugin_module, tmp_path, offline=True)
    runtime.initialize(context)
    runtime.on_turn_start(1, "remember this")

    runtime.sync_turn("remember this", "durable answer")

    snapshot = outbox.snapshot()
    assert snapshot.total == 1
    assert snapshot.by_state == {"pending": 1}
    assert client.raw_requests == []


def test_missing_turn_start_uses_sequence_fallback_without_losing_capture(plugin_module, tmp_path):
    runtime, context, client, _outbox = _runtime_parts(plugin_module, tmp_path)
    runtime.initialize(context)

    runtime.sync_turn("orphaned user", "still durable")

    request, _ = client.raw_requests[0]
    assert request.metadata["turn_number_source"] == "local-sequence-fallback"


def test_unsafe_unrestricted_credential_never_flushes_writes(plugin_module, tmp_path):
    runtime, context, client, outbox = _runtime_parts(plugin_module, tmp_path)
    response = client.auth_me()
    client.auth_me = lambda: response.__class__(
        credential=response.credential.__class__(
            type="api_key",
            id="key-1",
            scopes=("api:write",),
            project_ids=(),
            memory_space_ids=(),
            agent_id="hermes:home:nova",
            capability_profile="memory_provider",
        ),
        user=response.user,
        organization=response.organization,
        org_role=response.org_role,
    )
    runtime.initialize(context)
    runtime.on_turn_start(1, "private")

    runtime.sync_turn("private", "must stay local")

    assert outbox.snapshot().by_state == {"pending": 1}
    assert client.raw_requests == []
