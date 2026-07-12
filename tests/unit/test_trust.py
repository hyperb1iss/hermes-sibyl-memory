from __future__ import annotations

from typing import Any

import pytest

from schemas import Credential
from trust import CredentialRequirement, UnsafeCredentialError, validate_credential


def credential(**overrides: Any) -> Credential:
    values: dict[str, Any] = {
        "type": "api_key",
        "id": "key-1",
        "scopes": ("api:write",),
        "project_ids": ("project_home",),
        "memory_space_ids": ("space_home",),
        "agent_id": "hermes:home:nova",
        "capability_profile": "memory_provider",
    }
    values.update(overrides)
    return Credential(**values)


def requirement() -> CredentialRequirement:
    return CredentialRequirement("project_home", "space_home", "hermes:home:nova")


def test_exact_singleton_scope_is_accepted():
    validate_credential(credential(), requirement())


def test_setup_can_validate_canonical_binding_before_first_runtime_identity():
    validate_credential(
        credential(),
        CredentialRequirement("project_home", "space_home", None),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_ids", (), "exactly the configured project"),
        ("project_ids", ("project_home", "project_other"), "exactly the configured project"),
        ("memory_space_ids", (), "exactly the configured memory space"),
        ("memory_space_ids", ("space_home", "space_other"), "exactly the configured memory space"),
        ("agent_id", "hermes:home:other", "identity"),
        ("capability_profile", None, "capability_profile"),
    ],
)
def test_unsafe_credential_shapes_are_rejected(field: str, value, message: str):
    with pytest.raises(UnsafeCredentialError, match=message):
        validate_credential(credential(**{field: value}), requirement())
