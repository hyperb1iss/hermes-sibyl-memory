from __future__ import annotations

import os

import pytest

from scripts import release_gates


@pytest.fixture(scope="session")
def real_stack_settings() -> release_gates.StackSettings:
    if os.environ.get("HERMES_SIBYL_E2E") != "1":
        pytest.skip("set HERMES_SIBYL_E2E=1 to run the externally provisioned real stack")
    return release_gates.StackSettings.from_environment(os.environ)


@pytest.fixture(scope="session")
def mutation_consents() -> set[str]:
    return {
        value.strip()
        for value in os.environ.get("HERMES_SIBYL_E2E_MUTATION_SCENARIOS", "").split(",")
        if value.strip()
    }
