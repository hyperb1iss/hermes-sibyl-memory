from __future__ import annotations

import importlib
import json
from typing import Any

import httpx
import pytest


def _modules(plugin_module):
    client = importlib.import_module(f"{plugin_module.__name__}.client")
    schemas = importlib.import_module(f"{plugin_module.__name__}.schemas")
    return client, schemas


def _client(client_module, handler):
    return client_module.SibylClient(
        base_url="https://sibyl.example/api",
        api_key="sk_test_secret",
        transport=httpx.MockTransport(handler),
        request_id_factory=lambda: "req_hermes_test",
    )


def _receipt(key: str, *, replayed: bool = False) -> dict[str, Any]:
    return {
        "operation_id": key,
        "applied": True,
        "revision": 1,
        "affected_records": ["raw_captures:abc"],
        "idempotency_key": key,
        "replayed": replayed,
    }


def _context_response() -> dict[str, Any]:
    return {
        "goal": "remember the lantern",
        "intent": "general",
        "layer": "recall",
        "query": "remember the lantern",
        "domain": None,
        "project": "project_home",
        "sections": [],
        "total_items": 1,
        "usage_metadata": {"estimated_tokens": 42},
        "usage_hint": "Use memory as evidence.",
        "markdown": "# Memory\nLantern",
        "rendered_item_ids": ["raw_memory:abc"],
        "additive_future_field": True,
    }


def _raw_response(key: str) -> dict[str, Any]:
    return {
        "id": "abc",
        "organization_id": "org-1",
        "source_id": "hermes:turn:abc",
        "principal_id": "user-1",
        "memory_scope": "project",
        "scope_key": "project_home",
        "title": "Hermes turn session/1",
        "raw_content": "[User]\nhello\n\n[Assistant]\nhi",
        "tags": ["hermes", "conversation", "completed-turn"],
        "metadata": {"agent_id": "hermes:home:nova"},
        "provenance": {"adapter": "hermes-sibyl-memory"},
        "capture_surface": "hermes_memory_provider",
        "revision": 1,
        "mutation_receipt": _receipt(key),
    }


def _correction_response(
    *,
    action: str = "mark_stale",
    key: str | None = None,
) -> dict[str, Any]:
    return {
        "allowed": True,
        "applied": key is not None,
        "source_id": "abc",
        "action": action,
        "reason": "allowed",
        "target_lifecycle_state": "stale",
        "target_lifecycle_flags": ["stale"],
        "updated_review_state": None,
        "lifecycle": {},
        "reflection_finding": None,
        "affected_source_ids": ["abc"],
        "affected_derived_ids": [],
        "reversible": True,
        "recall_impact": {},
        "synthesis_impact": {},
        "audit_action": "memory.mark_stale",
        "policy_reasons": [],
        "metadata": {},
        "current_revision": 1,
        "revision": 2 if key is not None else None,
        "mutation_receipt": _receipt(key) if key is not None else None,
    }


def test_auth_me_sends_bearer_auth_request_id_and_probe_budget(plugin_module):
    client_module, _ = _modules(plugin_module)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://sibyl.example/api/auth/me"
        assert request.headers["Authorization"] == "Bearer sk_test_secret"
        assert request.headers["X-Request-ID"] == "req_hermes_test"
        assert request.extensions["timeout"]["read"] == 10.0
        return httpx.Response(
            200,
            json={
                "user": {"id": "user-1"},
                "organization": {"id": "org-1"},
                "org_role": "owner",
                "credential": {
                    "type": "api_key",
                    "id": "key-1",
                    "scopes": ["api:write"],
                    "project_ids": ["project_home"],
                    "memory_space_ids": ["space_home"],
                    "agent_id": "hermes:home:nova",
                    "delegated_authority": "household-agent",
                    "capability_profile": "memory_provider",
                    "future": "accepted",
                },
                "future": "accepted",
            },
        )

    with _client(client_module, handler) as client:
        response = client.auth_me()

    assert response.credential.type == "api_key"
    assert response.credential.agent_id == "hermes:home:nova"
    assert response.credential.project_ids == ("project_home",)


@pytest.mark.parametrize(("manual", "expected_timeout"), [(False, 5.0), (True, 10.0)])
def test_context_pack_uses_fixed_scope_and_recall_budget(
    plugin_module,
    manual: bool,
    expected_timeout: float,
):
    client_module, schemas = _modules(plugin_module)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload == {
            "goal": "remember the lantern",
            "intent": "general",
            "layer": "recall",
            "project": "project_home",
            "agent_id": "hermes:home:nova",
            "limit": 8,
            "include_related": True,
            "related_limit": 2,
            "audit": False,
            "record_exposure": False,
            "markdown_token_budget": 900,
        }
        assert request.extensions["timeout"]["read"] == expected_timeout
        return httpx.Response(
            200,
            json=_context_response(),
            headers={"X-Request-ID": "req_server_context"},
        )

    request = schemas.ContextPackRequest(
        goal="remember the lantern",
        project="project_home",
        agent_id="hermes:home:nova",
    )
    with _client(client_module, handler) as client:
        response = client.context_pack(request, manual=manual)

    assert response.markdown == "# Memory\nLantern"
    assert response.rendered_item_ids == ("raw_memory:abc",)
    assert response.request_id == "req_server_context"


def test_success_response_body_is_bounded(plugin_module):
    client_module, _ = _modules(plugin_module)
    oversized = b"{" + b"x" * client_module.MAX_SUCCESS_BODY_BYTES + b"}"

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(200, content=oversized),
        ) as client,
        pytest.raises(client_module.SibylProtocolError, match="safe body limit"),
    ):
        client.auth_me()


@pytest.mark.parametrize(
    ("estimated_tokens", "markdown"),
    [
        (901, "# Memory\nLantern"),
        (42, "x" * (900 * 16 + 1)),
    ],
)
def test_context_pack_rejects_server_budget_overruns(
    plugin_module,
    estimated_tokens: int,
    markdown: str,
):
    client_module, schemas = _modules(plugin_module)
    response = _context_response()
    response["usage_metadata"] = {"estimated_tokens": estimated_tokens}
    response["markdown"] = markdown
    request = schemas.ContextPackRequest(
        goal="remember the lantern",
        project="project_home",
        agent_id="hermes:home:nova",
    )

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(200, json=response),
        ) as client,
        pytest.raises(client_module.SibylProtocolError),
    ):
        client.context_pack(request)


def test_mutation_methods_send_keys_and_require_matching_receipts(plugin_module):
    client_module, schemas = _modules(plugin_module)
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.raw_path.decode())
        assert request.extensions["timeout"]["read"] == 10.0
        if request.url.path.endswith("/memory/expose"):
            assert request.headers["Idempotency-Key"] == "hermes-context-op1"
            return httpx.Response(
                200,
                json={
                    "recorded_ids": ["raw_memory:abc"],
                    "excluded_ids": [],
                    "denied_ids": [],
                    "mutation_receipt": _receipt("hermes-context-op1"),
                },
            )
        if request.url.path.endswith("/memory/raw"):
            assert request.headers["Idempotency-Key"] == "hermes-turn-op2"
            payload = json.loads(request.content)
            assert payload["memory_scope"] == "project"
            assert payload["scope_key"] == "project_home"
            assert payload["capture_surface"] == "hermes_memory_provider"
            return httpx.Response(200, json=_raw_response("hermes-turn-op2"))
        if request.url.path.endswith("/preview"):
            assert "Idempotency-Key" not in request.headers
            return httpx.Response(200, json=_correction_response())
        assert request.headers["Idempotency-Key"] == "hermes-correction-op3"
        return httpx.Response(
            200,
            json=_correction_response(key="hermes-correction-op3"),
        )

    exposure = schemas.ContextExposureRequest(
        exposed_ids=("raw_memory:abc",),
        project_id="project_home",
        automatic=True,
    )
    raw = schemas.RawMemoryRequest(
        title="Hermes turn session/1",
        raw_content="[User]\nhello\n\n[Assistant]\nhi",
        source_id="hermes:turn:abc",
        project_id="project_home",
        metadata={"agent_id": "hermes:home:nova"},
        provenance={"adapter": "hermes-sibyl-memory"},
    )
    correction = schemas.CorrectionRequest(action="mark_stale", reason="outdated")

    with _client(client_module, handler) as client:
        client.expose_context(exposure, idempotency_key="hermes-context-op1")
        client.remember_raw(raw, idempotency_key="hermes-turn-op2")
        preview = client.preview_correction("raw_memory:abc", correction)
        applied = client.apply_correction(
            "raw_memory:abc",
            correction,
            idempotency_key="hermes-correction-op3",
        )

    assert preview.current_revision == 1
    assert applied.revision == 2
    assert seen_paths == [
        "/api/memory/expose",
        "/api/memory/raw",
        "/api/memory/inspect/raw_memory%3Aabc/corrections/preview",
        "/api/memory/inspect/raw_memory%3Aabc/corrections",
    ]


def test_receipt_key_mismatch_is_a_protocol_error(plugin_module):
    client_module, schemas = _modules(plugin_module)
    raw = schemas.RawMemoryRequest(
        title="turn",
        raw_content="content",
        source_id="hermes:turn:abc",
        project_id="project_home",
        metadata={},
        provenance={},
    )

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(200, json=_raw_response("different-key")),
        ) as client,
        pytest.raises(client_module.SibylProtocolError) as error,
    ):
        client.remember_raw(raw, idempotency_key="hermes-turn-op1")

    assert error.value.request_id == "req_hermes_test"


@pytest.mark.parametrize(
    "code",
    [
        "idempotency_lock_held",
        "idempotency_payload_mismatch",
        "idempotency_interrupted_pending",
    ],
)
def test_structured_idempotency_conflicts_are_classified(plugin_module, code: str):
    client_module, _ = _modules(plugin_module)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": code,
                "message": "operation conflict",
                "request_id": "req_server_conflict",
            },
        )

    with (
        _client(client_module, handler) as client,
        pytest.raises(client_module.SibylConflictError) as error,
    ):
        client.auth_me()

    assert error.value.idempotency_kind.value == code
    assert error.value.request_id == "req_server_conflict"
    assert error.value.revision_conflict is False


def test_revision_conflict_is_distinct_from_idempotency(plugin_module):
    client_module, _ = _modules(plugin_module)

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(
                409,
                json={"error": "revision_conflict", "message": "revision moved"},
            ),
        ) as client,
        pytest.raises(client_module.SibylConflictError) as error,
    ):
        client.auth_me()

    assert error.value.revision_conflict is True
    assert error.value.idempotency_kind is None


def test_transport_error_is_single_attempt_and_not_an_http_error(plugin_module):
    client_module, _ = _modules(plugin_module)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("network unavailable", request=request)

    with (
        _client(client_module, handler) as client,
        pytest.raises(client_module.SibylTransportError) as error,
    ):
        client.auth_me()

    assert attempts == 1
    assert error.value.error_type == "ConnectError"
    assert not isinstance(error.value, client_module.SibylHTTPError)
    assert "network unavailable" not in str(error.value)


def test_http_error_uses_contract_envelope_without_body_echo(plugin_module):
    client_module, _ = _modules(plugin_module)

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(
                403,
                json={
                    "error": "capability_profile_forbidden",
                    "message": "Credential cannot call this endpoint",
                    "request_id": "req_server_forbidden",
                    "details": {"method": "GET"},
                },
            ),
        ) as client,
        pytest.raises(client_module.SibylHTTPError) as error,
    ):
        client.auth_me()

    assert error.value.status_code == 403
    assert error.value.error == "capability_profile_forbidden"
    assert error.value.request_id == "req_server_forbidden"
    assert "Credential cannot call this endpoint" not in str(error.value)


@pytest.mark.parametrize("body", [b"<html>upstream exploded</html>", b'[{"error":"bad"}]'])
def test_non_contract_error_bodies_are_rejected(plugin_module, body: bytes):
    client_module, _ = _modules(plugin_module)

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(502, content=body),
        ) as client,
        pytest.raises(client_module.SibylProtocolError) as error,
    ):
        client.auth_me()

    assert error.value.status_code == 502
    assert len(error.value.body_preview) <= 513


def test_error_body_is_bounded_before_parsing(plugin_module):
    client_module, _ = _modules(plugin_module)
    oversized = b"x" * (client_module.MAX_ERROR_BODY_BYTES * 4)

    with (
        _client(
            client_module,
            lambda _request: httpx.Response(500, content=oversized),
        ) as client,
        pytest.raises(client_module.SibylProtocolError) as error,
    ):
        client.auth_me()

    assert "safe body limit" in str(error.value)
    assert len(error.value.body_preview) <= 513
    assert len(str(error.value)) < 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("rendered_item_ids"),
        lambda payload: payload.__setitem__("total_items", True),
        lambda payload: payload.__setitem__("sections", {}),
    ],
)
def test_success_responses_reject_missing_or_incompatible_fields(plugin_module, mutate):
    client_module, schemas = _modules(plugin_module)
    payload = _context_response()
    mutate(payload)

    request = schemas.ContextPackRequest(
        goal="remember the lantern",
        project="project_home",
        agent_id="hermes:home:nova",
    )
    with (
        _client(
            client_module,
            lambda _request: httpx.Response(200, json=payload),
        ) as client,
        pytest.raises(client_module.SibylProtocolError),
    ):
        client.context_pack(request)
