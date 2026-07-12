"""Operator commands for the active Sibyl memory provider."""

from __future__ import annotations

import importlib.metadata
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from .capture import build_turn_capture
from .client import SibylClient, SibylClientError
from .config import config_path, load_api_key, load_provider_config, resolve_hermes_home
from .outbox import DurableOutbox, FlushReport, OutboxSnapshot
from .provider import RuntimeContext
from .runtime import ADAPTER_VERSION, HERMES_VERSION, SibylRuntime
from .schemas import AuthMeResponse, ContextPackRequest, CorrectionRequest, RawMemoryRequest

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from .config import ProviderConfig


class _Client(Protocol):
    def auth_me(self) -> AuthMeResponse: ...

    def context_pack(self, request: ContextPackRequest, *, manual: bool = False) -> Any: ...

    def remember_raw(self, request: RawMemoryRequest, *, idempotency_key: str) -> Any: ...

    def apply_correction(
        self,
        source_id: str,
        request: CorrectionRequest,
        *,
        idempotency_key: str,
    ) -> Any: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _OperatorContext:
    home: Path
    config: ProviderConfig
    api_key: str
    agent_id: str


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    state: str
    detail: str


class _WriteProbeCleanupError(RuntimeError):
    def __init__(self, source_id: str, error_type: str) -> None:
        self.source_id = source_id
        self.error_type = error_type
        super().__init__("diagnostic memory cleanup failed")


def _client_factory(context: _OperatorContext) -> _Client:
    return SibylClient(
        base_url=context.config.base_url,
        api_key=context.api_key,
    )


def _outbox_factory(home: Path) -> DurableOutbox:
    return DurableOutbox(home)


def _installed_hermes_version() -> str:
    return importlib.metadata.version("hermes-agent")


def _active_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return str(get_active_profile_name() or "default")
    except (ImportError, OSError, ValueError):
        return "default"


def _operator_context() -> _OperatorContext:
    home = resolve_hermes_home()
    return _OperatorContext(
        home=home,
        config=load_provider_config(home),
        api_key=load_api_key(home),
        agent_id=f"hermes:hermes:{_active_profile_name()}",
    )


def _format_counts(snapshot: OutboxSnapshot) -> str:
    states = (
        "pending",
        "pending_recovery",
        "blocked_auth",
        "reconcile_revision",
        "dead_letter",
    )
    return ", ".join(f"{state}={snapshot.by_state.get(state, 0)}" for state in states)


def _last_successful_write(path: Path) -> str:
    if not path.exists():
        return "none"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                """
                SELECT MAX(completed_at)
                FROM operation_history
                WHERE outcome IN ('succeeded', 'replayed')
                """
            ).fetchone()
    except sqlite3.Error:
        return "unavailable"
    return str(row[0]) if row and row[0] else "none"


def _context_probe(client: _Client, context: _OperatorContext) -> float:
    started = time.monotonic()
    client.context_pack(
        ContextPackRequest(
            goal="Hermes Sibyl provider diagnostic read probe",
            project=context.config.project_id,
            agent_id=context.agent_id,
            limit=1,
            include_related=False,
            related_limit=0,
            record_exposure=False,
            markdown_token_budget=min(context.config.context_token_budget, 128),
        ),
        manual=True,
    )
    return (time.monotonic() - started) * 1_000


def _print_credential(response: AuthMeResponse) -> None:
    credential = response.credential
    print(f"Authenticated identity: {credential.agent_id or 'unbound'}")
    print(f"Capability profile: {credential.capability_profile or 'none'}")
    print(f"Scopes: {', '.join(credential.scopes) or 'none'}")
    print(f"Project restrictions: {', '.join(credential.project_ids) or 'none'}")
    print(f"Memory-space restrictions: {', '.join(credential.memory_space_ids) or 'none'}")


def _status() -> None:
    try:
        context = _operator_context()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration: invalid ({type(exc).__name__})")
        print(f"Config path: {config_path()}")
        return

    print(f"Plugin version: {ADAPTER_VERSION}")
    print(f"Hermes identity: {context.agent_id}")
    print(f"Config path: {config_path(context.home)}")
    print(f"Configured: {'yes' if context.config.is_complete else 'no'}")
    print(f"API key: {'set' if context.api_key else 'missing'}")
    print(f"Base URL: {context.config.base_url}")
    print(f"Project: {context.config.project_id or 'missing'}")
    print(f"Memory space: {context.config.memory_space_id or 'missing'}")

    outbox: DurableOutbox | None = None
    try:
        outbox = _outbox_factory(context.home)
        snapshot = outbox.snapshot()
        print(f"Outbox: {_format_counts(snapshot)}")
        print(f"Last successful write: {_last_successful_write(outbox.path)}")
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        print(f"Outbox: unavailable ({type(exc).__name__})")
    finally:
        if outbox is not None:
            outbox.close()

    if not context.config.is_complete or not context.api_key:
        print("Remote status: unavailable; run 'hermes memory setup sibyl'")
        return

    try:
        client = _client_factory(context)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Remote status: unavailable ({type(exc).__name__})")
        return
    try:
        response = client.auth_me()
        _print_credential(response)
        latency_ms = _context_probe(client, context)
        print(f"Context read: ok ({latency_ms:.1f} ms, exposure recording disabled)")
    except (SibylClientError, OSError, RuntimeError, ValueError) as exc:
        print(f"Remote status: unavailable ({type(exc).__name__})")
    finally:
        client.close()


def _credential_checks(response: AuthMeResponse, context: _OperatorContext) -> list[_Check]:
    credential = response.credential
    checks = [
        _Check(
            "authentication",
            "pass" if credential.type == "api_key" else "fail",
            "API key accepted"
            if credential.type == "api_key"
            else "use an API key instead of session authentication",
        ),
        _Check(
            "API scope",
            "pass" if "api:write" in credential.scopes else "fail",
            "api:write is present"
            if "api:write" in credential.scopes
            else "create a provider key with api:write",
        ),
        _Check(
            "project restriction",
            "pass" if set(credential.project_ids) == {context.config.project_id} else "fail",
            "key is restricted to the configured project"
            if set(credential.project_ids) == {context.config.project_id}
            else "restrict the key to exactly the configured project",
        ),
        _Check(
            "memory-space restriction",
            (
                "pass"
                if set(credential.memory_space_ids) == {context.config.memory_space_id}
                else "fail"
            ),
            "key is restricted to the configured memory space"
            if set(credential.memory_space_ids) == {context.config.memory_space_id}
            else "restrict the key to exactly the configured memory space",
        ),
        _Check(
            "agent identity",
            "pass" if credential.agent_id == context.agent_id else "fail",
            f"bound to {context.agent_id}"
            if credential.agent_id == context.agent_id
            else f"bind the key to {context.agent_id}",
        ),
        _Check(
            "capability profile",
            "pass" if credential.capability_profile == "memory_provider" else "fail",
            "memory_provider is active"
            if credential.capability_profile == "memory_provider"
            else "set capability_profile=memory_provider",
        ),
    ]
    return checks


def _write_probe(client: _Client, context: _OperatorContext) -> None:
    probe_id = uuid.uuid4().hex
    capture = build_turn_capture(
        agent_id=context.agent_id,
        agent_workspace="hermes",
        agent_identity=_active_profile_name(),
        session_id=f"doctor-{probe_id}",
        parent_session_id="",
        local_sequence=1,
        turn_number=1,
        user_content="Sibyl provider write probe",
        assistant_content="Write access verified.",
        platform="cli",
        project_id=context.config.project_id,
        memory_space_id=context.config.memory_space_id,
        hermes_version=HERMES_VERSION,
        adapter_version=ADAPTER_VERSION,
    )
    request = RawMemoryRequest(
        title=str(capture.payload["title"]),
        raw_content=str(capture.payload["raw_content"]),
        source_id=capture.identity.source_id,
        project_id=context.config.project_id,
        metadata=dict(capture.payload["metadata"]),
        provenance=dict(capture.payload["provenance"]),
    )
    response = client.remember_raw(request, idempotency_key=capture.identity.idempotency_key)
    revision = response.revision
    if revision is None:
        raise RuntimeError("write probe response omitted its revision")
    correction_id = f"{capture.identity.operation_id}-stale"
    try:
        client.apply_correction(
            capture.identity.source_id,
            CorrectionRequest(
                action="mark_stale",
                reason="hermes_provider_doctor_write_probe",
                expected_revision=revision,
                metadata={"provider_operation_id": correction_id},
            ),
            idempotency_key=f"hermes-correction-{correction_id}",
        )
    except (SibylClientError, OSError, RuntimeError, ValueError) as exc:
        raise _WriteProbeCleanupError(
            capture.identity.source_id,
            type(exc).__name__,
        ) from exc


def _doctor(*, write_probe: bool) -> bool:
    checks: list[_Check] = []
    try:
        installed_version = _installed_hermes_version()
    except importlib.metadata.PackageNotFoundError:
        installed_version = "not installed"
    checks.append(
        _Check(
            "Hermes compatibility",
            "pass" if installed_version == HERMES_VERSION else "fail",
            f"Hermes {installed_version}; provider requires {HERMES_VERSION}",
        )
    )
    plugin_manifest = Path(__file__).with_name("plugin.yaml")
    checks.append(
        _Check(
            "plugin installation",
            "pass" if plugin_manifest.is_file() else "fail",
            f"Sibyl provider {ADAPTER_VERSION} manifest found"
            if plugin_manifest.is_file()
            else "reinstall the plugin; plugin.yaml is missing",
        )
    )

    try:
        context = _operator_context()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks.append(
            _Check(
                "configuration",
                "fail",
                f"repair {config_path()}; invalid {type(exc).__name__}",
            )
        )
        _print_checks(checks)
        return False

    parsed = urlsplit(context.config.base_url)
    tls_state = "warn" if parsed.scheme == "http" and context.config.allow_insecure_http else "pass"
    tls_detail = (
        "explicit insecure HTTP override is active"
        if tls_state == "warn"
        else "URL and TLS policy are valid"
    )
    checks.extend(
        [
            _Check(
                "configuration",
                "pass" if context.config.is_complete else "fail",
                "required project and memory space are set"
                if context.config.is_complete
                else "run 'hermes memory setup sibyl' to set project and memory space",
            ),
            _Check("URL/TLS policy", tls_state, tls_detail),
            _Check(
                "API key",
                "pass" if context.api_key else "fail",
                "secret is present"
                if context.api_key
                else "set SIBYL_API_KEY with 'hermes memory setup sibyl'",
            ),
        ]
    )

    outbox: DurableOutbox | None = None
    try:
        outbox = _outbox_factory(context.home)
        snapshot = outbox.snapshot()
        checks.append(_Check("outbox", "pass", f"writable; {_format_counts(snapshot)}"))
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        checks.append(
            _Check(
                "outbox",
                "fail",
                f"make {context.home / 'state'} writable ({type(exc).__name__})",
            )
        )
    finally:
        if outbox is not None:
            outbox.close()

    if not context.config.is_complete or not context.api_key:
        checks.append(_Check("remote probes", "skip", "complete setup first"))
        _print_checks(checks)
        return False

    try:
        client = _client_factory(context)
    except (OSError, RuntimeError, ValueError) as exc:
        checks.append(
            _Check(
                "remote probe",
                "fail",
                f"verify URL, network, TLS, and credential ({type(exc).__name__})",
            )
        )
        _print_checks(checks)
        return False
    try:
        response = client.auth_me()
        checks.extend(_credential_checks(response, context))
        if any(check.state == "fail" for check in checks):
            checks.append(_Check("context read", "skip", "fix credential restrictions first"))
        else:
            latency_ms = _context_probe(client, context)
            checks.append(
                _Check(
                    "context read",
                    "pass",
                    f"{latency_ms:.1f} ms; record_exposure=false",
                )
            )
            if write_probe:
                _write_probe(client, context)
                checks.append(
                    _Check(
                        "write probe",
                        "pass",
                        "raw write succeeded and was immediately marked stale",
                    )
                )
            else:
                checks.append(
                    _Check(
                        "write probe",
                        "skip",
                        "read-only by default; pass --write-probe to mutate",
                    )
                )
    except _WriteProbeCleanupError as exc:
        checks.append(
            _Check(
                "write probe cleanup",
                "fail",
                f"mark {exc.source_id} stale manually ({exc.error_type})",
            )
        )
    except (SibylClientError, OSError, RuntimeError, ValueError) as exc:
        checks.append(
            _Check(
                "remote probe",
                "fail",
                f"verify URL, network, TLS, and credential ({type(exc).__name__})",
            )
        )
    finally:
        client.close()

    _print_checks(checks)
    return not any(check.state == "fail" for check in checks)


def _print_checks(checks: list[_Check]) -> None:
    labels = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}
    for check in checks:
        print(f"[{labels[check.state]}] {check.name}: {check.detail}")


class _RuntimeFlushRunner:
    """Reuse the runtime's exact dispatcher without starting background lifecycle work."""

    def __init__(self, context: _OperatorContext) -> None:
        self._client = _client_factory(context)
        self._outbox = _outbox_factory(context.home)
        runtime_context = RuntimeContext(
            config=context.config,
            api_key=context.api_key,
            hermes_home=str(context.home),
            session_id="operator-cli",
            parent_session_id="",
            platform="cli",
            agent_context="primary",
            agent_identity=_active_profile_name(),
            agent_workspace="hermes",
            agent_id=context.agent_id,
            kwargs={},
        )
        self._runtime = SibylRuntime(
            runtime_context,
            client=self._client,
            outbox=self._outbox,
        )

    def run(self) -> tuple[int, FlushReport]:
        response = self._client.auth_me()
        self._runtime._validate_credential(response)
        unblocked = self._outbox.unblock_auth()
        report = self._outbox.flush(
            f"operator-{uuid.uuid4().hex}",
            self._runtime._execute_operation,
        )
        return unblocked, report

    def close(self) -> None:
        self._runtime.shutdown()


def _flush_runner_factory(context: _OperatorContext) -> _RuntimeFlushRunner:
    return _RuntimeFlushRunner(context)


def _flush() -> bool:
    try:
        context = _operator_context()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Flush failed: invalid configuration ({type(exc).__name__})")
        return False
    if not context.config.is_complete or not context.api_key:
        print("Flush failed: run 'hermes memory setup sibyl' first")
        return False

    try:
        runner = _flush_runner_factory(context)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Flush failed: {type(exc).__name__}; run 'hermes sibyl doctor'")
        return False
    try:
        unblocked, report = runner.run()
    except (SibylClientError, OSError, RuntimeError, ValueError) as exc:
        print(f"Flush failed: {type(exc).__name__}; run 'hermes sibyl doctor'")
        return False
    finally:
        runner.close()

    print(f"Unblocked authentication failures: {unblocked}")
    print(f"Claimed operations: {report.claimed}")
    print(
        f"Outcomes: {', '.join(f'{key}={value}' for key, value in sorted(report.outcomes.items())) or 'none'}"
    )
    print(f"Remaining: {_format_counts(report.remaining)}")
    return report.remaining.total == 0


def _config() -> None:
    try:
        context = _operator_context()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration is invalid ({type(exc).__name__}): {config_path()}")
        return
    payload = asdict(context.config)
    payload["api_key"] = "set" if context.api_key else "missing"
    print(f"Config path: {config_path(context.home)}")
    print(json.dumps(payload, indent=2, sort_keys=True))


def sibyl_command(args: Namespace) -> None:
    command = getattr(args, "sibyl_command", None) or "status"
    if command == "status":
        _status()
        return
    if command == "doctor":
        if not _doctor(write_probe=bool(getattr(args, "write_probe", False))):
            raise SystemExit(1)
        return
    if command == "flush":
        if not _flush():
            raise SystemExit(1)
        return
    if command == "config":
        _config()
        return
    raise SystemExit(f"unknown Sibyl command: {command}")


def register_cli(subparser: ArgumentParser) -> None:
    commands = subparser.add_subparsers(dest="sibyl_command")
    commands.add_parser("status", help="Show safe provider, scope, and outbox status")
    doctor = commands.add_parser("doctor", help="Validate the provider without remote mutation")
    doctor.add_argument(
        "--write-probe",
        action="store_true",
        help="Create and immediately stale a diagnostic memory",
    )
    commands.add_parser("flush", help="Replay durable pending mutations")
    commands.add_parser("config", help="Show non-secret provider configuration")
    subparser.set_defaults(func=sibyl_command)
