from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import release_gates as gates

HERMES_VALUE = "hermes-secret"


def passing_observation(scenario: gates.ScenarioDefinition) -> gates.ScenarioObservation:
    return gates.ScenarioObservation(
        scenario=scenario.slug,
        hermes_version="0.18.2",
        claimed_pass=True,
        checks=dict.fromkeys(scenario.required_checks, True),
        metrics={"sample_count": 1},
        evidence={"receipt": scenario.slug},
    )


def full_receipt(*, mode: str = "real_stack", version: str = "0.18.2"):
    return gates.aggregate_receipt(
        [passing_observation(scenario) for scenario in gates.SCENARIOS],
        mode=mode,
        hermes_version=version,
        run_id="gate_test",
        generated_at="2026-07-12T00:00:00+00:00",
    )


def settings() -> gates.StackSettings:
    return gates.StackSettings(
        hermes_url="https://hermes.example",
        sibyl_url="https://sibyl.example/api",
        profile="e2e",
        api_key="top-secret",
        hermes_session_token=HERMES_VALUE,
    )


def test_manifest_maps_exactly_five_gates_and_fifteen_numbered_scenarios():
    manifest = gates.manifest()

    assert [gate["name"] for gate in manifest["gates"]] == [gate.value for gate in gates.GateName]
    assert all(gate["required_checks"] for gate in manifest["gates"])
    assert [scenario["number"] for scenario in manifest["scenarios"]] == list(range(1, 16))
    assert len({scenario["slug"] for scenario in manifest["scenarios"]}) == 15
    assert all(scenario["gates"] for scenario in manifest["scenarios"])
    assert all(scenario["required_checks"] for scenario in manifest["scenarios"])


def test_complete_pinned_real_stack_evidence_passes_all_gates():
    receipt = full_receipt()

    assert receipt["release_eligible"] is True
    assert receipt["passed"] is True
    assert {gate["status"] for gate in receipt["gates"].values()} == {"passed"}
    assert {scenario["status"] for scenario in receipt["scenarios"]} == {"passed"}
    for gate, gate_receipt in receipt["gates"].items():
        assert tuple(gate_receipt["checks"]) == gates.GATE_REQUIRED_CHECKS[gates.GateName(gate)]


def test_gate_receipts_keep_scenario_domains_separate():
    receipt = full_receipt()

    assert receipt["gates"][gates.GateName.CONTRACT]["scenario_ids"] == ["cross-session-recall"]
    assert receipt["gates"][gates.GateName.ISOLATION]["scenario_ids"] == [
        "scope-isolation",
        "auth-failure",
        "capability-confinement",
    ]
    assert receipt["gates"][gates.GateName.PROMPT_SAFETY]["scenario_ids"] == [
        "cross-session-recall",
        "prompt-injection",
        "tool-privacy",
        "server-log-privacy",
    ]
    assert receipt["gates"][gates.GateName.LATENCY]["scenario_ids"] == [
        "cross-session-recall",
        "offline-durability",
    ]


def test_missing_scenario_fails_closed_and_marks_it_not_run():
    observations = [passing_observation(scenario) for scenario in gates.SCENARIOS[:-1]]

    receipt = gates.aggregate_receipt(
        observations,
        mode="real_stack",
        hermes_version="0.18.2",
    )

    assert receipt["passed"] is False
    assert receipt["coverage"]["reported_scenarios"] == 14
    assert receipt["scenarios"][-1]["status"] == "not_run"
    assert receipt["gates"][gates.GateName.PROMPT_SAFETY]["status"] == "failed"


def test_missing_required_check_overrides_driver_pass_claim():
    scenario = gates.SCENARIOS[0]
    observation = gates.ScenarioObservation(
        scenario=scenario.slug,
        hermes_version="0.18.2",
        claimed_pass=True,
        checks={scenario.required_checks[0]: True},
    )

    receipt = gates.aggregate_receipt(
        [observation],
        mode="real_stack",
        hermes_version="0.18.2",
    )

    first = receipt["scenarios"][0]
    assert first["status"] == "failed"
    assert first["missing_checks"] == list(scenario.required_checks[1:])
    assert receipt["passed"] is False


@pytest.mark.parametrize(
    ("mode", "version"),
    [("hermetic", "0.18.2"), ("real_stack", "main"), ("upstream_tracker", "main")],
)
def test_nonrelease_modes_and_unpinned_hermes_never_pass(mode, version):
    receipt = full_receipt(mode=mode, version=version)

    assert receipt["release_eligible"] is False
    assert receipt["passed"] is False


def test_duplicate_and_unknown_observations_fail_closed():
    observations = [passing_observation(scenario) for scenario in gates.SCENARIOS]
    observations.append(passing_observation(gates.SCENARIOS[0]))
    observations.append(
        gates.ScenarioObservation(
            scenario="future-scenario",
            hermes_version="0.18.2",
            claimed_pass=True,
            checks={},
        )
    )

    receipt = gates.aggregate_receipt(
        observations,
        mode="real_stack",
        hermes_version="0.18.2",
    )

    assert receipt["passed"] is False
    assert receipt["coverage"]["duplicate_scenarios"] == ["cross-session-recall"]
    assert receipt["coverage"]["unknown_scenarios"] == ["future-scenario"]


def test_real_stack_settings_require_explicit_url_profile_and_key():
    with pytest.raises(ValueError, match="HERMES_SIBYL_E2E_API_KEY"):
        gates.StackSettings.from_environment(
            {
                "HERMES_SIBYL_E2E_HERMES_URL": "https://hermes.example",
                "HERMES_SIBYL_E2E_SIBYL_URL": "https://sibyl.example/api",
                "HERMES_SIBYL_E2E_PROFILE": "e2e",
                "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN": HERMES_VALUE,
            }
        )


def test_real_stack_settings_reject_credentials_in_urls():
    environment = {
        "HERMES_SIBYL_E2E_HERMES_URL": "https://user:secret@hermes.example",
        "HERMES_SIBYL_E2E_SIBYL_URL": "https://sibyl.example/api",
        "HERMES_SIBYL_E2E_PROFILE": "e2e",
        "HERMES_SIBYL_E2E_API_KEY": "secret",
        "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN": HERMES_VALUE,
    }

    with pytest.raises(ValueError, match="must not contain credentials"):
        gates.StackSettings.from_environment(environment)


def test_mutating_scenario_requires_scenario_specific_consent(monkeypatch):
    called = False

    def fake_run_driver(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(gates, "run_driver", fake_run_driver)

    with pytest.raises(ValueError, match="cross-session-recall"):
        gates.run_real_stack(
            ["cross-session-recall"],
            set(),
            settings=settings(),
            timeout_seconds=1,
        )

    assert called is False


def test_confinement_requires_consent_because_it_sends_denied_mutation_probes(monkeypatch):
    scenario = next(item for item in gates.SCENARIOS if item.slug == "capability-confinement")
    called = False

    def fake_run_driver(*_args, **_kwargs):
        nonlocal called
        called = True
        return passing_observation(scenario)

    monkeypatch.setattr(
        gates,
        "run_driver",
        fake_run_driver,
    )

    with pytest.raises(ValueError, match="capability-confinement"):
        gates.run_real_stack(
            [scenario.slug],
            set(),
            settings=settings(),
            timeout_seconds=1,
        )

    assert called is False


def test_driver_protocol_keeps_key_out_of_stdin_and_redacts_receipt(monkeypatch):
    scenario = gates.SCENARIOS[0]
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        payload = {
            "scenario": scenario.slug,
            "runtime": {"hermes_version": "0.18.2"},
            "passed": True,
            "checks": dict.fromkeys(scenario.required_checks, True),
            "metrics": {
                "api_key": "top-secret",
                "detail": "Bearer top-secret",
                "echo": HERMES_VALUE,
            },
            "evidence": {"message": "value top-secret"},
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    observation = gates.run_driver(scenario, settings(), timeout_seconds=1, environment={})

    assert captured["args"][0][0] == gates.sys.executable
    assert captured["args"][0][1] == str(gates.REAL_STACK_DRIVER)
    assert "top-secret" not in captured["input"]
    assert HERMES_VALUE not in captured["input"]
    assert captured["env"]["HERMES_SIBYL_E2E_API_KEY"] == "top-secret"
    assert captured["env"]["HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN"] == HERMES_VALUE
    assert observation.metrics["api_key"] == "[REDACTED]"
    assert observation.metrics["detail"] == "Bearer [REDACTED]"
    assert observation.metrics["echo"] == "[REDACTED]"
    assert observation.evidence["message"] == "value [REDACTED]"


def test_driver_protocol_rejects_wrong_scenario_and_missing_boolean_checks(monkeypatch):
    scenario = gates.SCENARIOS[0]

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                '{"scenario":"wrong","runtime":{"hermes_version":"0.18.2"},'
                '"passed":true,"checks":{}}'
            ),
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="wrong scenario"):
        gates.run_driver(scenario, settings(), timeout_seconds=1, environment={})

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "scenario": scenario.slug,
                    "runtime": {"hermes_version": "0.18.2"},
                    "passed": True,
                    "checks": {"check": "yes"},
                }
            ),
            stderr="",
        ),
    )
    with pytest.raises(RuntimeError, match="string-to-boolean"):
        gates.run_driver(scenario, settings(), timeout_seconds=1, environment={})


def test_real_stack_receipt_uses_observed_hermes_version(monkeypatch):
    scenario = next(item for item in gates.SCENARIOS if item.slug == "capability-confinement")
    observation = passing_observation(scenario)
    monkeypatch.setattr(
        gates,
        "run_driver",
        lambda *_args, **_kwargs: gates.ScenarioObservation(
            scenario=observation.scenario,
            hermes_version="0.19.0",
            claimed_pass=observation.claimed_pass,
            checks=observation.checks,
        ),
    )

    receipt = gates.run_real_stack(
        [scenario.slug],
        {scenario.slug},
        settings=settings(),
        timeout_seconds=1,
    )

    assert receipt["runtime"]["hermes_version"] == "0.19.0"
    assert receipt["release_eligible"] is False


def test_receipt_is_stable_json_data():
    encoded = json.dumps(full_receipt(), sort_keys=True)

    assert "top-secret" not in encoded
    assert '"passed": true' in encoded
