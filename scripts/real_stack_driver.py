#!/usr/bin/env python3
"""Run one release scenario through released Hermes and Sibyl surfaces."""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PROTOCOL_VERSION = "hermes-sibyl-public-e2e/v1"
MAX_RESPONSE_BYTES = 1_048_576
HERMES_STATUS_PATH = "api/status"
HERMES_MEMORY_PATH = "api/memory"
SIBYL_AUTH_PATH = "auth/me"


class DriverError(RuntimeError):
    """A fail-closed driver, transport, or observation failure."""


class MissingHookError(DriverError):
    def __init__(self, scenario: str, hooks: Sequence[str]) -> None:
        self.scenario = scenario
        self.hooks = tuple(hooks)
        super().__init__(
            f"scenario {scenario!r} requires missing disposable-stack hooks: "
            + ", ".join(self.hooks)
        )


MISSING_HOOKS: Mapping[str, tuple[str, ...]] = {
    "cross-session-recall": ("isolated_profile_project_and_agent_key",),
    "scope-isolation": ("two_isolated_profiles_projects_and_agent_keys",),
    "offline-durability": ("sibyl_network_partition_and_hermes_process_restart",),
    "prompt-injection": ("isolated_memory_fixture_and_tool_invocation_observer",),
    "tool-privacy": ("deterministic_tool_output_canary",),
    "correction": ("isolated_recallable_source_fixture",),
    "branch-lineage": ("isolated_branchable_hermes_session",),
    "rewind": ("isolated_rewindable_hermes_session",),
    "auth-failure": ("provider_key_revocation",),
    "recovery": ("provider_key_rotation",),
    "capability-confinement": (),
    "mutation-crash-recovery": (
        "raw_effect_before_receipt_crash",
        "correction_effect_before_receipt_crash",
        "exposure_effect_before_receipt_crash",
    ),
    "revision-conflict": ("provider_partition_and_concurrent_source_mutation",),
    "pre-outbox-crash": ("sync_turn_pre_outbox_crash",),
    "server-log-privacy": ("hermes_and_sibyl_log_snapshot",),
}


@dataclass(frozen=True, slots=True)
class DriverRequest:
    scenario: str
    required_checks: tuple[str, ...]
    mutates_stack: bool
    hermes_url: str
    sibyl_url: str
    profile: str
    expected_hermes_version: str

    @classmethod
    def parse(cls, payload: Any, scenario_argument: str) -> DriverRequest:
        if not isinstance(payload, dict):
            raise DriverError("driver request must be a JSON object")
        if payload.get("schema_version") != "hermes-sibyl-release-gates/v1":
            raise DriverError("driver request has an unsupported schema_version")
        scenario = payload.get("scenario")
        stack = payload.get("stack")
        if not isinstance(scenario, dict) or not isinstance(stack, dict):
            raise DriverError("driver request is missing scenario or stack")
        slug = scenario.get("slug")
        required = scenario.get("required_checks")
        mutates = scenario.get("mutates_stack")
        if slug != scenario_argument or slug not in MISSING_HOOKS:
            raise DriverError("driver request names an unknown or mismatched scenario")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise DriverError("driver request required_checks must be a string array")
        if not isinstance(mutates, bool):
            raise DriverError("driver request mutates_stack must be boolean")
        values = {
            name: stack.get(name)
            for name in (
                "hermes_url",
                "sibyl_url",
                "profile",
                "expected_hermes_version",
            )
        }
        if not all(isinstance(value, str) and value for value in values.values()):
            raise DriverError("driver request stack fields must be non-empty strings")
        return cls(
            scenario=slug,
            required_checks=tuple(required),
            mutates_stack=mutates,
            hermes_url=cast("str", values["hermes_url"]),
            sibyl_url=cast("str", values["sibyl_url"]),
            profile=cast("str", values["profile"]),
            expected_hermes_version=cast("str", values["expected_hermes_version"]),
        )


class HermesRPC(Protocol):
    def call(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]: ...


class WebSocketHermesRPC:
    """Authenticated client for Hermes 0.18.2's released JSON-RPC WebSocket."""

    def __init__(
        self,
        *,
        base_url: str,
        session_token: str,
        timeout_seconds: float,
    ) -> None:
        self.url = _websocket_url(base_url, session_token)
        self.timeout_seconds = timeout_seconds

    def call(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            from websockets.sync.client import connect

            request_id = f"e2e_{uuid.uuid4().hex}"
            with connect(
                self.url,
                open_timeout=self.timeout_seconds,
                close_timeout=min(self.timeout_seconds, 5.0),
                max_size=MAX_RESPONSE_BYTES,
                proxy=None,
            ) as websocket:
                websocket.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": method,
                            "params": dict(params),
                        },
                        separators=(",", ":"),
                    )
                )
                while True:
                    frame = websocket.recv(timeout=self.timeout_seconds)
                    if not isinstance(frame, str):
                        raise DriverError("Hermes RPC returned a binary frame")
                    try:
                        payload = json.loads(frame)
                    except json.JSONDecodeError as exc:
                        raise DriverError("Hermes RPC returned malformed JSON") from exc
                    if not isinstance(payload, dict) or payload.get("id") != request_id:
                        continue
                    if "error" in payload:
                        raise DriverError(f"Hermes RPC method {method!r} failed")
                    result = payload.get("result")
                    if not isinstance(result, dict):
                        raise DriverError(f"Hermes RPC method {method!r} returned no object")
                    return result
        except DriverError:
            raise
        except Exception as exc:
            raise DriverError(f"Hermes RPC method {method!r} was unavailable") from exc


def _websocket_url(base_url: str, session_token: str) -> str:
    parsed = urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    base_path = parsed.path.rstrip("/")
    query = urlencode({"token": session_token})
    return urlunsplit((scheme, parsed.netloc, f"{base_path}/api/ws", query, ""))


class JSONHTTPClient:
    def __init__(
        self,
        *,
        base_url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Accept": "application/json", **headers},
            follow_redirects=False,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        payload: Mapping[str, Any] | None = None,
        expected_status: int = 200,
    ) -> Mapping[str, Any]:
        try:
            with self.client.stream(
                method,
                path,
                params=params,
                json=dict(payload) if payload is not None else None,
            ) as response:
                if response.is_redirect:
                    raise DriverError(f"{method} {path} returned a redirect")
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise DriverError(f"{method} {path} response exceeded size limit")
                    body.extend(chunk)
                if response.status_code != expected_status:
                    raise DriverError(
                        f"{method} {path} returned HTTP {response.status_code}, "
                        f"expected {expected_status}"
                    )
        except DriverError:
            raise
        except httpx.HTTPError as exc:
            raise DriverError(f"{method} {path} request failed") from exc
        try:
            decoded = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DriverError(f"{method} {path} returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise DriverError(f"{method} {path} returned a non-object response")
        return decoded

    def status(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        try:
            with self.client.stream(
                method,
                path,
                json=dict(payload) if payload is not None else None,
            ) as response:
                if response.is_redirect:
                    raise DriverError(f"{method} {path} returned a redirect")
                consumed = 0
                for chunk in response.iter_bytes():
                    consumed += len(chunk)
                    if consumed > MAX_RESPONSE_BYTES:
                        raise DriverError(f"{method} {path} response exceeded size limit")
                return response.status_code
        except DriverError:
            raise
        except httpx.HTTPError as exc:
            raise DriverError(f"{method} {path} request failed") from exc


class PublicStackDriver:
    def __init__(
        self,
        request: DriverRequest,
        *,
        provider_api_key: str,
        hermes_session_token: str,
        hermes_transport: httpx.BaseTransport | None = None,
        sibyl_transport: httpx.BaseTransport | None = None,
        hermes_rpc: HermesRPC | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.request = request
        self.hermes = JSONHTTPClient(
            base_url=request.hermes_url,
            headers={"X-Hermes-Session-Token": hermes_session_token},
            timeout_seconds=timeout_seconds,
            transport=hermes_transport,
        )
        self.sibyl = JSONHTTPClient(
            base_url=request.sibyl_url,
            headers={"Authorization": f"Bearer {provider_api_key}"},
            timeout_seconds=timeout_seconds,
            transport=sibyl_transport,
        )
        self.hermes_rpc = hermes_rpc or WebSocketHermesRPC(
            base_url=request.hermes_url,
            session_token=hermes_session_token,
            timeout_seconds=timeout_seconds,
        )

    def close(self) -> None:
        self.hermes.close()
        self.sibyl.close()

    def run(self) -> dict[str, Any]:
        status = self.hermes.request(
            "GET",
            HERMES_STATUS_PATH,
            params={"profile": self.request.profile},
        )
        version = status.get("version")
        if version != self.request.expected_hermes_version:
            raise DriverError(
                f"Hermes runtime version {version!r} does not match "
                f"{self.request.expected_hermes_version!r}"
            )
        memory_status = self.hermes.request("GET", HERMES_MEMORY_PATH)
        if memory_status.get("active") != "sibyl":
            raise DriverError("Hermes does not report the Sibyl memory provider as active")
        self.hermes_rpc.call("config.get", {"key": "profile"})
        auth = self.sibyl.request("GET", SIBYL_AUTH_PATH)

        hooks = MISSING_HOOKS[self.request.scenario]
        if hooks:
            raise MissingHookError(self.request.scenario, hooks)
        checks, metrics, evidence = self._capability_confinement(auth)
        required = set(self.request.required_checks)
        if set(checks) != required:
            raise DriverError("driver and release manifest disagree on required checks")
        return {
            "schema_version": PROTOCOL_VERSION,
            "scenario": self.request.scenario,
            "runtime": {"hermes_version": version},
            "passed": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
            "evidence": evidence,
        }

    def _capability_confinement(
        self,
        auth: Mapping[str, Any],
    ) -> tuple[dict[str, bool], dict[str, int], dict[str, Any]]:
        claims = auth.get("credential")
        if not isinstance(claims, dict):
            claims = {}
        profile_enforced = (
            claims.get("type") == "api_key"
            and claims.get("capability_profile") == "memory_provider"
        )
        probes: tuple[tuple[str, str, str, Mapping[str, Any] | None], ...] = (
            ("tasks", "GET", "tasks", None),
            ("entities", "GET", "entities", None),
            ("reflection", "POST", "context/reflect", {"notes": "e2e confinement probe"}),
            ("sharing", "POST", "memory/share/preview", {}),
            ("promotion", "POST", "memory/promote/preview", {}),
            (
                "delete",
                "POST",
                f"memory/inspect/{quote('e2e-nonexistent', safe='')}/corrections/preview",
                {"action": "delete", "reason": "e2e confinement probe"},
            ),
            (
                "redact",
                "POST",
                f"memory/inspect/{quote('e2e-nonexistent', safe='')}/corrections/preview",
                {"action": "redact", "reason": "e2e confinement probe"},
            ),
        )
        statuses = {
            name: self.sibyl.status(method, path, payload=payload)
            for name, method, path, payload in probes
        }
        all_forbidden = all(status == 403 for status in statuses.values())
        return (
            {
                "memory_provider_profile_enforced_server_side": profile_enforced,
                "all_disallowed_endpoints_return_403": all_forbidden,
            },
            {"disallowed_probe_count": len(statuses)},
            {"disallowed_statuses": statuses},
        )


def _read_request() -> Any:
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise DriverError("driver stdin did not contain one JSON object") from exc


def _blocked_receipt(request: DriverRequest, error: MissingHookError) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_VERSION,
        "scenario": request.scenario,
        "runtime": {"hermes_version": request.expected_hermes_version},
        "passed": False,
        "checks": dict.fromkeys(request.required_checks, False),
        "metrics": {},
        "evidence": {
            "release_blocker": {
                "kind": "missing_disposable_stack_hooks",
                "missing_hooks": list(error.hooks),
            }
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if len(arguments) != 1:
        print("usage: real_stack_driver.py <scenario-slug>", file=sys.stderr)
        return 2
    try:
        request = DriverRequest.parse(_read_request(), arguments[0])
        provider_api_key = os.environ.get("HERMES_SIBYL_E2E_API_KEY", "").strip()
        hermes_session_token = os.environ.get(
            "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN",
            "",
        ).strip()
        if not provider_api_key:
            raise DriverError("HERMES_SIBYL_E2E_API_KEY is required")
        if not hermes_session_token:
            raise DriverError("HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN is required")
        consent = os.environ.get("HERMES_SIBYL_E2E_MUTATION_SCENARIO", "").strip()
        if request.mutates_stack and consent != request.scenario:
            raise DriverError("mutating scenario lacks exact per-scenario consent")
        stack_driver = PublicStackDriver(
            request,
            provider_api_key=provider_api_key,
            hermes_session_token=hermes_session_token,
        )
        try:
            receipt = stack_driver.run()
        except MissingHookError as exc:
            receipt = _blocked_receipt(request, exc)
        finally:
            stack_driver.close()
    except DriverError as exc:
        print(f"real-stack driver error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
