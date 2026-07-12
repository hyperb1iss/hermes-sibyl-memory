from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from websockets.sync import client as websocket_client

from scripts import real_stack_driver as driver
from scripts import release_gates

if TYPE_CHECKING:
    from collections.abc import Mapping

DASHBOARD_VALUE = "dashboard-secret"
PROVIDER_VALUE = "provider-secret"


def request_for(scenario_slug: str) -> driver.DriverRequest:
    scenario = next(item for item in release_gates.SCENARIOS if item.slug == scenario_slug)
    return driver.DriverRequest.parse(
        {
            "schema_version": release_gates.SCHEMA_VERSION,
            "scenario": {
                "slug": scenario.slug,
                "required_checks": list(scenario.required_checks),
                "mutates_stack": scenario.mutates_stack,
            },
            "stack": {
                "hermes_url": "https://hermes.example",
                "sibyl_url": "https://sibyl.example/api",
                "profile": "release-e2e",
                "expected_hermes_version": "0.18.2",
            },
        },
        scenario_slug,
    )


class FakeRPC:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((method, dict(params)))
        return {"home": "/disposable/hermes"}


def stack_transports(
    calls: list[tuple[str, str]],
    *,
    hermes_version: str = "0.18.2",
    active_provider: str = "sibyl",
) -> tuple[httpx.MockTransport, httpx.MockTransport]:
    def hermes_handler(request: httpx.Request) -> httpx.Response:
        calls.append(("hermes", request.url.path))
        assert request.headers["X-Hermes-Session-Token"] == DASHBOARD_VALUE
        assert DASHBOARD_VALUE.encode() not in request.content
        if request.url.path.endswith("/api/status"):
            return httpx.Response(200, json={"version": hermes_version})
        if request.url.path.endswith("/api/memory"):
            return httpx.Response(200, json={"active": active_provider})
        return httpx.Response(404, json={"detail": "not found"})

    def sibyl_handler(request: httpx.Request) -> httpx.Response:
        calls.append(("sibyl", request.url.path))
        assert request.headers["Authorization"] == f"Bearer {PROVIDER_VALUE}"
        assert PROVIDER_VALUE.encode() not in request.content
        if request.url.path.endswith("/api/auth/me"):
            return httpx.Response(
                200,
                json={
                    "credential": {
                        "type": "api_key",
                        "capability_profile": "memory_provider",
                        "credential": "never-retained-provider-secret",
                    }
                },
            )
        return httpx.Response(403, json={"error": "capability_profile_forbidden"})

    return httpx.MockTransport(hermes_handler), httpx.MockTransport(sibyl_handler)


def test_capability_confinement_uses_released_authenticated_surfaces():
    calls: list[tuple[str, str]] = []
    hermes_transport, sibyl_transport = stack_transports(calls)
    rpc = FakeRPC()
    stack_driver = driver.PublicStackDriver(
        request_for("capability-confinement"),
        provider_api_key=PROVIDER_VALUE,
        hermes_session_token=DASHBOARD_VALUE,
        hermes_transport=hermes_transport,
        sibyl_transport=sibyl_transport,
        hermes_rpc=rpc,
    )

    try:
        receipt = stack_driver.run()
    finally:
        stack_driver.close()

    assert receipt["passed"] is True
    assert receipt["checks"] == {
        "memory_provider_profile_enforced_server_side": True,
        "all_disallowed_endpoints_return_403": True,
    }
    assert receipt["metrics"] == {"disallowed_probe_count": 7}
    assert calls[:3] == [
        ("hermes", "/api/status"),
        ("hermes", "/api/memory"),
        ("sibyl", "/api/auth/me"),
    ]
    assert rpc.calls == [("config.get", {"key": "profile"})]
    encoded = json.dumps(receipt)
    assert "provider-secret" not in encoded
    assert "dashboard-secret" not in encoded
    assert "never-retained" not in encoded


@pytest.mark.parametrize(
    ("scenario", "missing_hook"),
    [
        ("offline-durability", "sibyl_network_partition_and_hermes_process_restart"),
        ("auth-failure", "provider_key_revocation"),
        ("mutation-crash-recovery", "raw_effect_before_receipt_crash"),
        ("pre-outbox-crash", "sync_turn_pre_outbox_crash"),
        ("server-log-privacy", "hermes_and_sibyl_log_snapshot"),
    ],
)
def test_unreleased_fault_and_log_hooks_fail_closed_after_public_preflight(
    scenario: str,
    missing_hook: str,
):
    calls: list[tuple[str, str]] = []
    hermes_transport, sibyl_transport = stack_transports(calls)
    stack_driver = driver.PublicStackDriver(
        request_for(scenario),
        provider_api_key=PROVIDER_VALUE,
        hermes_session_token=DASHBOARD_VALUE,
        hermes_transport=hermes_transport,
        sibyl_transport=sibyl_transport,
        hermes_rpc=FakeRPC(),
    )

    with pytest.raises(driver.MissingHookError) as caught:
        stack_driver.run()
    stack_driver.close()

    assert missing_hook in caught.value.hooks
    assert calls == [
        ("hermes", "/api/status"),
        ("hermes", "/api/memory"),
        ("sibyl", "/api/auth/me"),
    ]


def test_all_fifteen_scenarios_are_explicit_and_only_public_lane_is_unblocked():
    scenario_ids = {scenario.slug for scenario in release_gates.SCENARIOS}

    assert set(driver.MISSING_HOOKS) == scenario_ids
    assert {scenario for scenario, hooks in driver.MISSING_HOOKS.items() if not hooks} == {
        "capability-confinement"
    }


def test_public_http_client_rejects_redirects():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            307,
            headers={"Location": "https://attacker.example"},
        )
    )
    client = driver.JSONHTTPClient(
        base_url="https://hermes.example",
        headers={"X-Hermes-Session-Token": DASHBOARD_VALUE},
        timeout_seconds=1,
        transport=transport,
    )

    with pytest.raises(driver.DriverError, match="returned a redirect"):
        client.request("GET", "api/status")
    client.close()


def test_public_http_client_rejects_oversized_success_bodies():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=b"x" * (driver.MAX_RESPONSE_BYTES + 1))
    )
    client = driver.JSONHTTPClient(
        base_url="https://sibyl.example/api",
        headers={"Authorization": f"Bearer {PROVIDER_VALUE}"},
        timeout_seconds=1,
        transport=transport,
    )

    with pytest.raises(driver.DriverError, match="exceeded size limit"):
        client.request("GET", "auth/me")
    client.close()


def test_wrong_hermes_version_fails_before_sibyl_mutation_probes():
    calls: list[tuple[str, str]] = []
    hermes_transport, sibyl_transport = stack_transports(calls, hermes_version="0.19.0")
    stack_driver = driver.PublicStackDriver(
        request_for("capability-confinement"),
        provider_api_key=PROVIDER_VALUE,
        hermes_session_token=DASHBOARD_VALUE,
        hermes_transport=hermes_transport,
        sibyl_transport=sibyl_transport,
        hermes_rpc=FakeRPC(),
    )

    with pytest.raises(driver.DriverError, match="does not match"):
        stack_driver.run()
    stack_driver.close()

    assert calls == [("hermes", "/api/status")]


def test_main_rejects_mutation_without_exact_consent_before_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    parsed = request_for("capability-confinement")
    payload = {
        "schema_version": release_gates.SCHEMA_VERSION,
        "scenario": {
            "slug": parsed.scenario,
            "required_checks": list(parsed.required_checks),
            "mutates_stack": parsed.mutates_stack,
        },
        "stack": {
            "hermes_url": parsed.hermes_url,
            "sibyl_url": parsed.sibyl_url,
            "profile": parsed.profile,
            "expected_hermes_version": parsed.expected_hermes_version,
        },
    }
    monkeypatch.setattr(driver, "_read_request", lambda: payload)
    monkeypatch.setenv("HERMES_SIBYL_E2E_API_KEY", PROVIDER_VALUE)
    monkeypatch.setenv(
        "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN",
        DASHBOARD_VALUE,
    )
    monkeypatch.delenv("HERMES_SIBYL_E2E_MUTATION_SCENARIO", raising=False)

    assert driver.main(["capability-confinement"]) == 2
    assert "lacks exact per-scenario consent" in capsys.readouterr().err


def test_websocket_auth_uses_the_separate_dashboard_credential():
    url = driver._websocket_url(
        "https://hermes.example/prefix",
        "dashboard secret",
    )

    assert url == "wss://hermes.example/prefix/api/ws?token=dashboard+secret"
    assert "provider-secret" not in url


def test_websocket_rpc_uses_released_json_rpc_and_bounded_authenticated_socket(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def send(self, payload: str) -> None:
            captured["request"] = json.loads(payload)

        def recv(self, *, timeout: float) -> str:
            captured["timeout"] = timeout
            request = captured["request"]
            return json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}})

    def fake_connect(url: str, **kwargs: Any) -> FakeSocket:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeSocket()

    monkeypatch.setattr(websocket_client, "connect", fake_connect)
    rpc = driver.WebSocketHermesRPC(
        base_url="https://hermes.example",
        session_token=DASHBOARD_VALUE,
        timeout_seconds=3,
    )

    assert rpc.call("config.get", {"key": "profile"}) == {"ok": True}
    assert captured["url"].endswith(f"/api/ws?token={DASHBOARD_VALUE}")
    assert captured["kwargs"]["max_size"] == driver.MAX_RESPONSE_BYTES
    assert captured["kwargs"]["proxy"] is None
    assert captured["request"]["method"] == "config.get"
    assert captured["request"]["params"] == {"key": "profile"}


def test_missing_hook_receipt_is_machine_readable_and_fails_every_required_check():
    request = request_for("offline-durability")
    error = driver.MissingHookError(request.scenario, driver.MISSING_HOOKS[request.scenario])

    receipt = driver._blocked_receipt(request, error)

    assert receipt["passed"] is False
    assert set(receipt["checks"]) == set(request.required_checks)
    assert set(receipt["checks"].values()) == {False}
    assert receipt["evidence"]["release_blocker"] == {
        "kind": "missing_disposable_stack_hooks",
        "missing_hooks": ["sibyl_network_partition_and_hermes_process_restart"],
    }
