from __future__ import annotations

from scripts import release_gates


def test_all_real_stack_release_gates(real_stack_settings, mutation_consents):
    scenario_ids = [scenario.slug for scenario in release_gates.SCENARIOS]

    receipt = release_gates.run_real_stack(
        scenario_ids,
        mutation_consents,
        settings=real_stack_settings,
        timeout_seconds=300,
    )

    assert receipt["coverage"]["reported_scenarios"] == 15
    assert {gate["status"] for gate in receipt["gates"].values()} == {"passed"}
    assert receipt["passed"] is True
