"""Shared least-privilege credential validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .schemas import Credential
    except ImportError:
        from schemas import Credential


@dataclass(frozen=True, slots=True)
class CredentialRequirement:
    project_id: str
    memory_space_id: str
    agent_id: str | None


class UnsafeCredentialError(RuntimeError):
    def __init__(self, failures: tuple[str, ...]) -> None:
        self.failures = failures
        super().__init__("; ".join(failures))


def credential_failures(
    credential: Credential,
    requirement: CredentialRequirement,
) -> tuple[str, ...]:
    failures: list[str] = []
    if credential.type != "api_key":
        failures.append("credential must be an API key")
    if "api:write" not in credential.scopes:
        failures.append("api:write scope is required")
    if set(credential.project_ids) != {requirement.project_id}:
        failures.append("key must be restricted to exactly the configured project")
    if set(credential.memory_space_ids) != {requirement.memory_space_id}:
        failures.append("key must be restricted to exactly the configured memory space")
    if requirement.agent_id is None:
        parts = (credential.agent_id or "").split(":")
        if len(parts) != 3 or parts[0] != "hermes" or not all(parts[1:]):
            failures.append("key must bind one canonical Hermes agent identity")
    elif credential.agent_id != requirement.agent_id:
        failures.append("authenticated agent identity does not match Hermes")
    if credential.capability_profile != "memory_provider":
        failures.append("capability_profile must be memory_provider")
    return tuple(failures)


def validate_credential(
    credential: Credential,
    requirement: CredentialRequirement,
) -> None:
    failures = credential_failures(credential, requirement)
    if failures:
        raise UnsafeCredentialError(failures)


__all__ = [
    "CredentialRequirement",
    "UnsafeCredentialError",
    "credential_failures",
    "validate_credential",
]
