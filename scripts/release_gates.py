#!/usr/bin/env python3
"""Aggregate Hermes/Sibyl end-to-end evidence into release-gate receipts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA_VERSION = "hermes-sibyl-release-gates/v1"
PINNED_HERMES_VERSION = "0.18.2"
REAL_STACK_DRIVER = Path(__file__).with_name("real_stack_driver.py")


class GateName(StrEnum):
    CONTRACT = "hermes-provider-contract-gate"
    ISOLATION = "hermes-memory-isolation-gate"
    INTEGRITY = "hermes-memory-integrity-gate"
    PROMPT_SAFETY = "hermes-memory-prompt-safety-gate"
    LATENCY = "hermes-memory-latency-gate"


class ScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    number: int
    slug: str
    title: str
    gates: tuple[GateName, ...]
    required_checks: tuple[str, ...]
    mutates_stack: bool = True


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        1,
        "cross-session-recall",
        "Cross-session recall",
        (GateName.CONTRACT, GateName.PROMPT_SAFETY, GateName.LATENCY),
        (
            "fresh_session_recalls_correct_source",
            "pinned_hermes_lifecycle_passed",
            "flat_plugin_discovery_passed",
            "three_tools_dispatch_through_memory_manager",
            "released_hermes_helpers_only",
            "automatic_context_within_rendered_budget",
            "automatic_prefetch_at_most_250_ms",
            "capture_outside_visible_response_path",
            "unrelated_sessions_not_serialized",
        ),
    ),
    ScenarioDefinition(
        2,
        "scope-isolation",
        "Scope isolation",
        (GateName.ISOLATION,),
        (
            "cross_project_recall_leakage_zero",
            "cross_project_mutation_success_zero",
            "raw_key_foreign_or_disallowed_success_zero",
            "conflicting_agent_identity_rejected",
            "missing_project_or_memory_space_restrictions_rejected",
        ),
    ),
    ScenarioDefinition(
        3,
        "offline-durability",
        "Offline durability",
        (GateName.INTEGRITY, GateName.LATENCY),
        (
            "failed_writes_survive_restart",
            "offline_replay_effect_exactly_once",
            "failed_sibyl_read_does_not_fail_hermes_turn",
        ),
    ),
    ScenarioDefinition(
        4,
        "prompt-injection",
        "Prompt injection",
        (GateName.PROMPT_SAFETY,),
        ("stored_prompt_injection_fenced", "stored_prompt_injection_invoked_no_tools"),
    ),
    ScenarioDefinition(
        5,
        "tool-privacy",
        "Tool privacy",
        (GateName.PROMPT_SAFETY,),
        ("tool_output_canary_absent_from_sibyl",),
    ),
    ScenarioDefinition(
        6,
        "correction",
        "Correction",
        (GateName.INTEGRITY,),
        ("correction_changes_future_recall", "correction_receipt_revision_guarded"),
    ),
    ScenarioDefinition(
        7,
        "branch-lineage",
        "Branch lineage",
        (GateName.INTEGRITY,),
        ("branch_parent_child_provenance_correct",),
    ),
    ScenarioDefinition(
        8,
        "rewind",
        "Rewind",
        (GateName.INTEGRITY,),
        ("rewind_discarded_sources_not_current",),
    ),
    ScenarioDefinition(
        9,
        "auth-failure",
        "Auth failure",
        (GateName.ISOLATION, GateName.INTEGRITY),
        ("revoked_key_reads_fail_open", "revoked_key_writes_blocked_auth"),
    ),
    ScenarioDefinition(
        10,
        "recovery",
        "Recovery",
        (GateName.INTEGRITY,),
        ("replacement_key_drains_backlog", "recovery_creates_no_duplicate_sources"),
    ),
    ScenarioDefinition(
        11,
        "capability-confinement",
        "Capability confinement",
        (GateName.ISOLATION,),
        ("memory_provider_profile_enforced_server_side", "all_disallowed_endpoints_return_403"),
    ),
    ScenarioDefinition(
        12,
        "mutation-crash-recovery",
        "Mutation crash recovery",
        (GateName.INTEGRITY,),
        (
            "raw_effect_and_receipt_exactly_once",
            "correction_effect_and_receipt_exactly_once",
            "exposure_effect_and_receipt_exactly_once",
            "malformed_or_mismatched_receipts_preserve_outbox_rows",
        ),
    ),
    ScenarioDefinition(
        13,
        "revision-conflict",
        "Revision conflict",
        (GateName.INTEGRITY,),
        ("revision_conflict_converges_without_overwrite",),
    ),
    ScenarioDefinition(
        14,
        "pre-outbox-crash",
        "Pre-outbox crash",
        (GateName.INTEGRITY,),
        ("resume_reconciliation_recovers_missing_turn_once",),
    ),
    ScenarioDefinition(
        15,
        "server-log-privacy",
        "Server log privacy",
        (GateName.PROMPT_SAFETY,),
        (
            "household_canary_absent_from_hermes_logs",
            "household_canary_absent_from_sibyl_logs",
            "credentials_request_bodies_and_raw_goals_absent_from_logs",
        ),
    ),
)

GATE_REQUIRED_CHECKS: Mapping[GateName, tuple[str, ...]] = {
    GateName.CONTRACT: (
        "pinned_hermes_lifecycle_passed",
        "flat_plugin_discovery_passed",
        "three_tools_dispatch_through_memory_manager",
        "released_hermes_helpers_only",
    ),
    GateName.ISOLATION: (
        "cross_project_recall_leakage_zero",
        "cross_project_mutation_success_zero",
        "raw_key_foreign_or_disallowed_success_zero",
        "conflicting_agent_identity_rejected",
        "missing_project_or_memory_space_restrictions_rejected",
        "memory_provider_profile_enforced_server_side",
    ),
    GateName.INTEGRITY: (
        "failed_writes_survive_restart",
        "raw_effect_and_receipt_exactly_once",
        "correction_effect_and_receipt_exactly_once",
        "exposure_effect_and_receipt_exactly_once",
        "malformed_or_mismatched_receipts_preserve_outbox_rows",
        "revision_conflict_converges_without_overwrite",
        "resume_reconciliation_recovers_missing_turn_once",
        "rewind_discarded_sources_not_current",
    ),
    GateName.PROMPT_SAFETY: (
        "stored_prompt_injection_fenced",
        "stored_prompt_injection_invoked_no_tools",
        "tool_output_canary_absent_from_sibyl",
        "credentials_request_bodies_and_raw_goals_absent_from_logs",
        "automatic_context_within_rendered_budget",
    ),
    GateName.LATENCY: (
        "automatic_prefetch_at_most_250_ms",
        "failed_sibyl_read_does_not_fail_hermes_turn",
        "capture_outside_visible_response_path",
        "unrelated_sessions_not_serialized",
    ),
}

_SCENARIO_BY_SLUG = {scenario.slug: scenario for scenario in SCENARIOS}
_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|credential|secret|token)", re.I)
_BEARER = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")


@dataclass(frozen=True, slots=True)
class StackSettings:
    hermes_url: str
    sibyl_url: str
    profile: str
    api_key: str = field(repr=False)
    hermes_session_token: str = field(repr=False)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> StackSettings:
        required = {
            "HERMES_SIBYL_E2E_HERMES_URL": "hermes_url",
            "HERMES_SIBYL_E2E_SIBYL_URL": "sibyl_url",
            "HERMES_SIBYL_E2E_PROFILE": "profile",
            "HERMES_SIBYL_E2E_API_KEY": "api_key",
            "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN": "hermes_session_token",
        }
        missing = [name for name in required if not environment.get(name, "").strip()]
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"real-stack mode requires explicit environment values: {names}")

        hermes_url = _validated_url(environment["HERMES_SIBYL_E2E_HERMES_URL"], "Hermes")
        sibyl_url = _validated_url(environment["HERMES_SIBYL_E2E_SIBYL_URL"], "Sibyl")
        profile = environment["HERMES_SIBYL_E2E_PROFILE"].strip()
        if any(character.isspace() for character in profile):
            raise ValueError("HERMES_SIBYL_E2E_PROFILE must not contain whitespace")
        return cls(
            hermes_url=hermes_url,
            sibyl_url=sibyl_url,
            profile=profile,
            api_key=environment["HERMES_SIBYL_E2E_API_KEY"].strip(),
            hermes_session_token=environment["HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN"].strip(),
        )


@dataclass(frozen=True, slots=True)
class ScenarioObservation:
    scenario: str
    hermes_version: str
    claimed_pass: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, int | float | str | bool | None] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)


def _validated_url(value: str, label: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} URL must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{label} URL must not contain credentials")
    return normalized


def _redact(value: Any, api_key: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    if isinstance(value, str):
        redacted = value.replace(api_key, "[REDACTED]") if api_key else value
        return _BEARER.sub("Bearer [REDACTED]", redacted)
    return value


def run_driver(
    scenario: ScenarioDefinition,
    settings: StackSettings,
    *,
    timeout_seconds: float,
    mutation_authorized: bool = False,
    environment: Mapping[str, str] | None = None,
) -> ScenarioObservation:
    request = {
        "schema_version": SCHEMA_VERSION,
        "scenario": {
            "number": scenario.number,
            "slug": scenario.slug,
            "title": scenario.title,
            "gates": [gate.value for gate in scenario.gates],
            "required_checks": list(scenario.required_checks),
            "mutates_stack": scenario.mutates_stack,
        },
        "stack": {
            "hermes_url": settings.hermes_url,
            "sibyl_url": settings.sibyl_url,
            "profile": settings.profile,
            "expected_hermes_version": PINNED_HERMES_VERSION,
        },
    }
    driver_environment = dict(environment or os.environ)
    driver_environment.update(
        {
            "HERMES_SIBYL_E2E_HERMES_URL": settings.hermes_url,
            "HERMES_SIBYL_E2E_SIBYL_URL": settings.sibyl_url,
            "HERMES_SIBYL_E2E_PROFILE": settings.profile,
            "HERMES_SIBYL_E2E_API_KEY": settings.api_key,
            "HERMES_SIBYL_E2E_HERMES_SESSION_TOKEN": settings.hermes_session_token,
        }
    )
    if mutation_authorized:
        driver_environment["HERMES_SIBYL_E2E_MUTATION_SCENARIO"] = scenario.slug
    else:
        driver_environment.pop("HERMES_SIBYL_E2E_MUTATION_SCENARIO", None)
    try:
        result = subprocess.run(  # noqa: S603 - repository-owned driver and interpreter
            (sys.executable, str(REAL_STACK_DRIVER), scenario.slug),
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=driver_environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"scenario driver failed: {type(exc).__name__}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"scenario driver exited with status {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("scenario driver did not emit one JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("scenario driver receipt must be a JSON object")
    if payload.get("scenario") != scenario.slug:
        raise RuntimeError("scenario driver receipt names the wrong scenario")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("hermes_version"), str):
        raise RuntimeError("scenario driver receipt must report runtime.hermes_version")
    hermes_version = runtime["hermes_version"].strip()
    if not hermes_version:
        raise RuntimeError("scenario driver receipt reported an empty Hermes version")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or any(
        not isinstance(key, str) or not isinstance(value, bool) for key, value in checks.items()
    ):
        raise RuntimeError("scenario driver checks must be a string-to-boolean object")
    metrics = payload.get("metrics", {})
    evidence = payload.get("evidence", {})
    if not isinstance(metrics, dict) or not isinstance(evidence, dict):
        raise RuntimeError("scenario driver metrics and evidence must be objects")
    return ScenarioObservation(
        scenario=scenario.slug,
        hermes_version=hermes_version,
        claimed_pass=payload.get("passed") is True,
        checks=checks,
        metrics=_redact(_redact(metrics, settings.api_key), settings.hermes_session_token),
        evidence=_redact(_redact(evidence, settings.api_key), settings.hermes_session_token),
    )


def _scenario_receipt(
    definition: ScenarioDefinition,
    observation: ScenarioObservation | None,
    *,
    duration_ms: int = 0,
    error: str = "",
) -> dict[str, Any]:
    checks = dict(observation.checks) if observation is not None else {}
    missing_checks = [check for check in definition.required_checks if check not in checks]
    failed_checks = [check for check in definition.required_checks if checks.get(check) is False]
    passed = bool(
        observation is not None
        and observation.claimed_pass
        and not missing_checks
        and not failed_checks
        and not error
    )
    return {
        "number": definition.number,
        "scenario": definition.slug,
        "title": definition.title,
        "gates": [gate.value for gate in definition.gates],
        "mutates_stack": definition.mutates_stack,
        "status": ScenarioStatus.PASSED
        if passed
        else (
            ScenarioStatus.FAILED if observation is not None or error else ScenarioStatus.NOT_RUN
        ),
        "duration_ms": duration_ms,
        "required_checks": list(definition.required_checks),
        "checks": checks,
        "missing_checks": missing_checks,
        "failed_checks": failed_checks,
        "metrics": dict(observation.metrics) if observation is not None else {},
        "evidence": dict(observation.evidence) if observation is not None else {},
        "error": error,
    }


def aggregate_receipt(
    observations: Iterable[ScenarioObservation],
    *,
    mode: str,
    hermes_version: str,
    durations_ms: Mapping[str, int] | None = None,
    errors: Mapping[str, str] | None = None,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    indexed: dict[str, ScenarioObservation] = {}
    duplicate_scenarios: list[str] = []
    unknown_scenarios: list[str] = []
    for observation in observations:
        if observation.scenario not in _SCENARIO_BY_SLUG:
            unknown_scenarios.append(observation.scenario)
        elif observation.scenario in indexed:
            duplicate_scenarios.append(observation.scenario)
        else:
            indexed[observation.scenario] = observation

    durations = durations_ms or {}
    run_errors = errors or {}
    scenario_receipts = [
        _scenario_receipt(
            definition,
            indexed.get(definition.slug),
            duration_ms=durations.get(definition.slug, 0),
            error=run_errors.get(definition.slug, ""),
        )
        for definition in SCENARIOS
    ]

    gate_receipts: dict[str, Any] = {}
    for gate in GateName:
        members = [receipt for receipt in scenario_receipts if gate.value in receipt["gates"]]
        required_checks = GATE_REQUIRED_CHECKS[gate]
        check_values = {
            check: next(
                (
                    receipt["checks"].get(check)
                    for receipt in members
                    if check in receipt["required_checks"]
                ),
                None,
            )
            for check in required_checks
        }
        passed = all(receipt["status"] == ScenarioStatus.PASSED for receipt in members) and all(
            value is True for value in check_values.values()
        )
        gate_receipts[gate.value] = {
            "status": "passed" if passed else "failed",
            "scenario_ids": [receipt["scenario"] for receipt in members],
            "checks": check_values,
            "passed_scenarios": sum(
                receipt["status"] == ScenarioStatus.PASSED for receipt in members
            ),
            "failed_scenarios": sum(
                receipt["status"] == ScenarioStatus.FAILED for receipt in members
            ),
            "not_run_scenarios": sum(
                receipt["status"] == ScenarioStatus.NOT_RUN for receipt in members
            ),
        }

    all_scenarios_passed = all(
        receipt["status"] == ScenarioStatus.PASSED for receipt in scenario_receipts
    )
    all_gates_passed = all(gate["status"] == "passed" for gate in gate_receipts.values())
    release_eligible = mode == "real_stack" and hermes_version == PINNED_HERMES_VERSION
    passed = bool(
        release_eligible
        and all_scenarios_passed
        and all_gates_passed
        and not duplicate_scenarios
        and not unknown_scenarios
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or f"gate_{uuid.uuid4().hex}",
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "mode": mode,
        "runtime": {
            "hermes_version": hermes_version,
            "pinned_hermes_version": PINNED_HERMES_VERSION,
        },
        "release_eligible": release_eligible,
        "passed": passed,
        "coverage": {
            "expected_scenarios": len(SCENARIOS),
            "reported_scenarios": len(indexed),
            "duplicate_scenarios": sorted(set(duplicate_scenarios)),
            "unknown_scenarios": sorted(set(unknown_scenarios)),
        },
        "gates": gate_receipts,
        "scenarios": scenario_receipts,
    }


def run_real_stack(
    selected_scenarios: Sequence[str],
    allowed_mutations: set[str],
    *,
    settings: StackSettings,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    unknown = sorted(set(selected_scenarios) - _SCENARIO_BY_SLUG.keys())
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(unknown)}")
    unknown_consents = sorted(allowed_mutations - _SCENARIO_BY_SLUG.keys())
    if unknown_consents:
        raise ValueError(f"unknown mutation consents: {', '.join(unknown_consents)}")
    if not selected_scenarios:
        raise ValueError("real-stack mode requires at least one explicit --scenario")
    missing_consent = sorted(
        slug
        for slug in selected_scenarios
        if _SCENARIO_BY_SLUG[slug].mutates_stack and slug not in allowed_mutations
    )
    if missing_consent:
        raise ValueError(
            "mutating scenarios require matching --allow-mutation values: "
            + ", ".join(missing_consent)
        )

    observations: list[ScenarioObservation] = []
    durations: dict[str, int] = {}
    errors: dict[str, str] = {}
    for slug in dict.fromkeys(selected_scenarios):
        scenario = _SCENARIO_BY_SLUG[slug]
        started = time.monotonic()
        try:
            observations.append(
                run_driver(
                    scenario,
                    settings,
                    timeout_seconds=timeout_seconds,
                    mutation_authorized=scenario.mutates_stack,
                    environment=environment,
                )
            )
        except RuntimeError as exc:
            errors[slug] = str(exc)
        durations[slug] = round((time.monotonic() - started) * 1000)
    observed_versions = {observation.hermes_version for observation in observations}
    hermes_version = observed_versions.pop() if len(observed_versions) == 1 else "unknown"
    return aggregate_receipt(
        observations,
        mode="real_stack",
        hermes_version=hermes_version,
        durations_ms=durations,
        errors=errors,
    )


def manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pinned_hermes_version": PINNED_HERMES_VERSION,
        "gates": [
            {
                "name": gate.value,
                "required_checks": list(GATE_REQUIRED_CHECKS[gate]),
            }
            for gate in GateName
        ],
        "scenarios": [
            {
                **asdict(scenario),
                "gates": [gate.value for gate in scenario.gates],
            }
            for scenario in SCENARIOS
        ],
    }


def _write_json(payload: Mapping[str, Any], output: str) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(content)
    else:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser("describe", help="emit the hermetic gate manifest")
    describe.add_argument("--output", default="-", help="JSON output path, or - for stdout")

    run = subparsers.add_parser("run", help="run explicitly selected real-stack scenarios")
    run.add_argument("--real-stack", action="store_true", required=True)
    run.add_argument("--scenario", action="append", default=[], choices=tuple(_SCENARIO_BY_SLUG))
    run.add_argument(
        "--allow-mutation",
        action="append",
        default=[],
        choices=tuple(_SCENARIO_BY_SLUG),
        help="explicitly authorize one selected mutating scenario",
    )
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--output", default="-", help="JSON receipt path, or - for stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "describe":
        _write_json(manifest(), args.output)
        return 0
    try:
        settings = StackSettings.from_environment(os.environ)
        receipt = run_real_stack(
            args.scenario,
            set(args.allow_mutation),
            settings=settings,
            timeout_seconds=args.timeout,
        )
    except ValueError as exc:
        print(f"release gate configuration error: {exc}", file=sys.stderr)
        return 2
    _write_json(receipt, args.output)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
