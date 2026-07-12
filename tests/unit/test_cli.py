from __future__ import annotations

import argparse
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest


class FakeClient:
    def __init__(
        self,
        schemas,
        *,
        credential=None,
        error: Exception | None = None,
        correction_error: Exception | None = None,
    ) -> None:
        self.schemas = schemas
        self.credential = credential or schemas.Credential(
            type="api_key",
            id="key-1",
            scopes=("api:write",),
            project_ids=("project_home",),
            memory_space_ids=("space_home",),
            agent_id="hermes:hermes:nova",
            delegated_authority="household-agent",
            capability_profile="memory_provider",
        )
        self.error = error
        self.correction_error = correction_error
        self.context_requests = []
        self.raw_requests = []
        self.corrections = []
        self.closed = False

    def auth_me(self):
        if self.error:
            raise self.error
        return self.schemas.AuthMeResponse(
            credential=self.credential,
            user={"id": "user-1"},
            organization={"id": "org-1"},
            org_role="member",
        )

    def context_pack(self, request, *, manual: bool = False):
        self.context_requests.append((request, manual))
        return SimpleNamespace(markdown="", rendered_item_ids=(), total_items=0)

    def remember_raw(self, request, *, idempotency_key: str):
        self.raw_requests.append((request, idempotency_key))
        return SimpleNamespace(revision=1)

    def apply_correction(self, source_id: str, request, *, idempotency_key: str):
        self.corrections.append((source_id, request, idempotency_key))
        if self.correction_error:
            raise self.correction_error
        return SimpleNamespace(revision=2)

    def close(self) -> None:
        self.closed = True


def _modules(plugin_module):
    package = plugin_module.__name__
    return (
        importlib.import_module(f"{package}.cli"),
        importlib.import_module(f"{package}.config"),
        importlib.import_module(f"{package}.outbox"),
        importlib.import_module(f"{package}.schemas"),
    )


def _configure(tmp_path, monkeypatch, config_module) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SIBYL_API_KEY", "super-secret-value")
    config_module.save_provider_config(
        config_module.ProviderConfig.from_mapping(
            {
                "base_url": "https://sibyl.example/api",
                "project_id": "project_home",
                "memory_space_id": "space_home",
            }
        ),
        tmp_path,
    )


def _prepare(plugin_module, tmp_path, monkeypatch):
    cli, config, outbox, schemas = _modules(plugin_module)
    _configure(tmp_path, monkeypatch, config)
    monkeypatch.setattr(cli, "_active_profile_name", lambda: "nova")
    monkeypatch.setattr(cli, "_installed_hermes_version", lambda: "0.18.2")
    return cli, outbox, schemas


def test_status_reports_safe_identity_scope_latency_and_outbox(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    cli.sibyl_command(argparse.Namespace(sibyl_command="status"))

    output = capsys.readouterr().out
    assert "Hermes identity: hermes:hermes:nova" in output
    assert "Authenticated identity: hermes:hermes:nova" in output
    assert "Project restrictions: project_home" in output
    assert "Memory-space restrictions: space_home" in output
    assert "Context read: ok" in output
    assert "recording disabled" in output
    assert "pending=0" in output
    assert "Last successful write: none" in output
    assert "super-secret-value" not in output
    request, manual = client.context_requests[0]
    assert request.record_exposure is False
    assert request.project == "project_home"
    assert request.agent_id == "hermes:hermes:nova"
    assert manual is True
    assert client.closed is True


def test_doctor_is_read_only_by_default(plugin_module, tmp_path, monkeypatch, capsys):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=False) is True

    output = capsys.readouterr().out
    assert "[PASS] Hermes compatibility" in output
    assert "[PASS] authentication" in output
    assert "[PASS] project restriction" in output
    assert "[PASS] memory-space restriction" in output
    assert "[PASS] agent identity" in output
    assert "[PASS] capability profile" in output
    assert "[PASS] context read" in output
    assert "[PASS] outbox" in output
    assert "[SKIP] write probe: read-only by default" in output
    assert client.context_requests[0][0].record_exposure is False
    assert client.raw_requests == []
    assert client.corrections == []
    assert "super-secret-value" not in output


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"scopes": ()}, "[FAIL] API scope"),
        ({"project_ids": ()}, "[FAIL] project restriction"),
        ({"project_ids": ("project_home", "project_extra")}, "[FAIL] project restriction"),
        ({"memory_space_ids": ()}, "[FAIL] memory-space restriction"),
        ({"agent_id": "hermes:hermes:other"}, "[FAIL] agent identity"),
        ({"capability_profile": None}, "[FAIL] capability profile"),
    ],
)
def test_doctor_rejects_each_unsafe_key_shape_before_context_read(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
    changes,
    failure,
):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    safe_client = FakeClient(schemas)
    client = FakeClient(schemas, credential=replace(safe_client.credential, **changes))
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=False) is False

    output = capsys.readouterr().out
    assert failure in output
    assert "[SKIP] context read: fix credential restrictions first" in output
    assert client.context_requests == []
    assert client.raw_requests == []


def test_doctor_write_probe_creates_then_stales_diagnostic_memory(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=True) is True

    output = capsys.readouterr().out
    assert "[PASS] write probe: raw write succeeded and was immediately marked stale" in output
    raw_request, raw_key = client.raw_requests[0]
    assert raw_request.project_id == "project_home"
    assert raw_request.metadata["memory_space_id"] == "space_home"
    assert raw_request.metadata["agent_id"] == "hermes:hermes:nova"
    assert raw_key.startswith("hermes-turn-")
    source_id, correction, correction_key = client.corrections[0]
    assert source_id == raw_request.source_id
    assert correction.action == "mark_stale"
    assert correction.expected_revision == 1
    assert correction_key.startswith("hermes-correction-")


def test_doctor_write_probe_cleanup_failure_names_safe_manual_repair(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(
        schemas,
        correction_error=RuntimeError("secret correction detail"),
    )
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=True) is False

    output = capsys.readouterr().out
    raw_request, _ = client.raw_requests[0]
    assert f"[FAIL] write probe cleanup: mark {raw_request.source_id} stale manually" in output
    assert "secret correction detail" not in output


def test_doctor_outbox_failure_is_actionable(plugin_module, tmp_path, monkeypatch, capsys):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)
    monkeypatch.setattr(
        cli, "_outbox_factory", lambda home: (_ for _ in ()).throw(PermissionError())
    )

    assert cli._doctor(write_probe=False) is False

    output = capsys.readouterr().out
    assert "[FAIL] outbox: make" in output
    assert "writable (PermissionError)" in output
    assert client.context_requests == []


def test_doctor_reports_compatibility_and_configuration_failures_without_remote_work(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, config, _outbox, schemas = _modules(plugin_module)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SIBYL_API_KEY", "super-secret-value")
    config.config_path(tmp_path).write_text('{"base_url":"not-a-url"}', encoding="utf-8")
    monkeypatch.setattr(cli, "_installed_hermes_version", lambda: "0.19.0")
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=True) is False

    output = capsys.readouterr().out
    assert "[FAIL] Hermes compatibility" in output
    assert "provider requires 0.18.2" in output
    assert "[FAIL] configuration: repair" in output
    assert client.context_requests == []
    assert client.raw_requests == []
    assert "super-secret-value" not in output


def test_doctor_missing_key_skips_all_remote_probes(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, config, _outbox, schemas = _modules(plugin_module)
    _configure(tmp_path, monkeypatch, config)
    monkeypatch.delenv("SIBYL_API_KEY")
    monkeypatch.setattr(cli, "_active_profile_name", lambda: "nova")
    monkeypatch.setattr(cli, "_installed_hermes_version", lambda: "0.18.2")
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=True) is False

    output = capsys.readouterr().out
    assert "[FAIL] API key" in output
    assert "[SKIP] remote probes: complete setup first" in output
    assert client.context_requests == []
    assert client.raw_requests == []


def test_doctor_transport_failure_is_bounded_and_actionable(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, _outbox, schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    client = FakeClient(schemas, error=RuntimeError("secret server detail"))
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=False) is False

    output = capsys.readouterr().out
    assert "[FAIL] remote probe: verify URL, network, TLS, and credential (RuntimeError)" in output
    assert "secret server detail" not in output
    assert client.context_requests == []
    assert client.raw_requests == []


def test_doctor_allows_explicit_insecure_http_with_visible_warning(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, config, _outbox, schemas = _modules(plugin_module)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SIBYL_API_KEY", "super-secret-value")
    config.save_provider_config(
        config.ProviderConfig.from_mapping(
            {
                "base_url": "http://sibyl.lan/api",
                "project_id": "project_home",
                "memory_space_id": "space_home",
                "allow_insecure_http": True,
            }
        ),
        tmp_path,
    )
    monkeypatch.setattr(cli, "_active_profile_name", lambda: "nova")
    monkeypatch.setattr(cli, "_installed_hermes_version", lambda: "0.18.2")
    client = FakeClient(schemas)
    monkeypatch.setattr(cli, "_client_factory", lambda context: client)

    assert cli._doctor(write_probe=False) is True

    output = capsys.readouterr().out
    assert "[WARN] URL/TLS policy: explicit insecure HTTP override is active" in output
    assert "[PASS] context read" in output


def test_config_prints_only_non_secret_values(plugin_module, tmp_path, monkeypatch, capsys):
    cli, _outbox, _schemas = _prepare(plugin_module, tmp_path, monkeypatch)

    cli.sibyl_command(argparse.Namespace(sibyl_command="config"))

    output = capsys.readouterr().out
    assert '"api_key": "set"' in output
    assert '"project_id": "project_home"' in output
    assert "super-secret-value" not in output


def test_flush_uses_injected_runtime_seam_and_reports_every_state(
    plugin_module,
    tmp_path,
    monkeypatch,
    capsys,
):
    cli, outbox, _schemas = _prepare(plugin_module, tmp_path, monkeypatch)
    snapshot = outbox.OutboxSnapshot(
        total=0,
        by_state={},
        oldest_age_seconds=None,
        active_claims=0,
        expired_claims=0,
        dependency_blocked=0,
    )
    report = outbox.FlushReport(
        claimed=3,
        outcomes={"succeed": 2, "obsolete": 1},
        remaining=snapshot,
        duration_seconds=0.2,
        deadline_exhausted=False,
    )

    class Runner:
        closed = False

        def run(self):
            return 1, report

        def close(self):
            self.closed = True

    runner = Runner()
    monkeypatch.setattr(cli, "_flush_runner_factory", lambda context: runner)

    assert cli._flush() is True

    output = capsys.readouterr().out
    assert "Unblocked authentication failures: 1" in output
    assert "Claimed operations: 3" in output
    assert "obsolete=1, succeed=2" in output
    assert "Remaining: pending=0" in output
    assert runner.closed is True


def test_register_cli_exposes_explicit_write_probe_flag(plugin_module):
    cli, _config, _outbox, _schemas = _modules(plugin_module)
    parser = argparse.ArgumentParser()

    cli.register_cli(parser)

    args = parser.parse_args(["doctor", "--write-probe"])
    assert args.sibyl_command == "doctor"
    assert args.write_probe is True
