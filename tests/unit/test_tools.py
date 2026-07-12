from __future__ import annotations

import pytest

from tools import (
    TOOL_SCHEMAS,
    CorrectionToolArgs,
    RecallToolArgs,
    RememberToolArgs,
    ToolValidationError,
)


def test_provider_exposes_exactly_three_fixed_scope_tools():
    assert [schema["name"] for schema in TOOL_SCHEMAS] == [
        "sibyl_recall",
        "sibyl_remember",
        "sibyl_correct",
    ]
    forbidden = {"base_url", "project", "project_id", "memory_space_id", "agent_id", "path"}
    for schema in TOOL_SCHEMAS:
        properties = schema["parameters"]["properties"]
        assert forbidden.isdisjoint(properties)
        assert schema["parameters"]["additionalProperties"] is False


def test_recall_validates_query_and_enums():
    assert RecallToolArgs.parse({"query": "find it"}).layer == "recall"
    with pytest.raises(ToolValidationError, match="layer"):
        RecallToolArgs.parse({"query": "find it", "layer": "admin"})
    with pytest.raises(ToolValidationError, match="8000"):
        RecallToolArgs.parse({"query": "x" * 8001})


def test_remember_bounds_and_deduplicates_tags():
    parsed = RememberToolArgs.parse(
        {
            "title": "Decision",
            "content": "Use the governed path.",
            "kind": "decision",
            "tags": ["memory", "memory", "sibyl"],
        }
    )
    assert parsed.tags == ("memory", "sibyl")
    with pytest.raises(ToolValidationError, match="kind"):
        RememberToolArgs.parse({"title": "x", "content": "y", "kind": "delete"})


@pytest.mark.parametrize(
    ("action", "required_field"),
    [
        ("revise", "revised_content"),
        ("superseded", "replacement_source_id"),
        ("duplicate", "duplicate_of_source_id"),
    ],
)
def test_correction_enforces_conditional_fields(action: str, required_field: str):
    with pytest.raises(ToolValidationError, match=required_field):
        CorrectionToolArgs.parse({"source_id": "source-1", "action": action, "reason": "incorrect"})


def test_correction_never_accepts_delete_or_redact():
    for action in ("delete", "redact"):
        with pytest.raises(ToolValidationError, match="action"):
            CorrectionToolArgs.parse({"source_id": "source-1", "action": action, "reason": "no"})
