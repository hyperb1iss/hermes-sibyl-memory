from __future__ import annotations

from capture import (
    build_turn_capture,
    canonical_request_hash,
    canonical_source,
    committed_turn_fingerprints,
    committed_turns,
    derive_turn_identity,
)


def test_canonical_source_preserves_exact_user_and_assistant_text():
    assert canonical_source("  user\n", "assistant  ") == (
        "[User]\n  user\n\n\n[Assistant]\nassistant  "
    )


def test_turn_identity_uses_delimiters_and_percent_encoded_provenance():
    identity = derive_turn_identity(
        agent_id="hermes:home:nova:one",
        session_id="signal:thread/one",
        local_sequence=7,
        turn_number=11,
        user_content="same",
        assistant_content="answer",
    )
    ambiguous = derive_turn_identity(
        agent_id="hermes:home:nova",
        session_id="one:signal:thread/one",
        local_sequence=7,
        turn_number=11,
        user_content="same",
        assistant_content="answer",
    )

    assert identity.operation_id != ambiguous.operation_id
    assert "hermes%3Ahome%3Anova%3Aone" in identity.source_id
    assert "signal%3Athread%2Fone" in identity.source_id
    assert identity.idempotency_key == f"hermes-turn-{identity.operation_id}"


def test_identical_turns_are_distinct_by_local_sequence():
    values = {
        "agent_id": "hermes:home:nova",
        "session_id": "session-1",
        "turn_number": 2,
        "user_content": "repeat",
        "assistant_content": "repeat",
    }
    first = derive_turn_identity(local_sequence=1, **values)
    second = derive_turn_identity(local_sequence=2, **values)

    assert first.turn_hash == second.turn_hash
    assert first.operation_id != second.operation_id
    assert first.source_id != second.source_id


def test_turn_number_falls_back_to_persisted_sequence():
    identity = derive_turn_identity(
        agent_id="hermes:home:nova",
        session_id="session-1",
        local_sequence=4,
        turn_number=None,
        user_content="hello",
        assistant_content="hi",
    )

    assert identity.turn_number == 4
    assert identity.turn_number_source == "local-sequence-fallback"


def test_capture_contains_only_explicit_final_turn_content():
    capture = build_turn_capture(
        agent_id="hermes:home:nova",
        agent_workspace="home",
        agent_identity="nova",
        session_id="session-1",
        parent_session_id="parent-1",
        local_sequence=3,
        turn_number=5,
        user_content="user canary",
        assistant_content="assistant canary",
        platform="signal",
        project_id="project_home",
        memory_space_id="space_home",
        hermes_version="0.18.2",
        adapter_version="0.1.0",
        participant_ids=["bliss"],
        chat_id="chat-1",
        thread_id="thread-1",
    )

    assert capture.payload["raw_content"] == (
        "[User]\nuser canary\n\n[Assistant]\nassistant canary"
    )
    assert capture.payload["memory_scope"] == "project"
    assert capture.payload["scope_key"] == "project_home"
    assert capture.payload["metadata"]["provider_operation_id"] == capture.identity.operation_id
    stored_hash = capture.payload["metadata"].pop("provider_request_hash")
    assert stored_hash == canonical_request_hash(capture.payload)
    assert capture.payload["metadata"]["contains_tool_messages"] is False


def test_reconciliation_fingerprints_ignore_tool_content():
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "tool_calls": [{"name": "secret"}]},
        {"role": "tool", "content": "TOOL CANARY"},
        {"role": "assistant", "content": "final"},
    ]
    fingerprints = committed_turn_fingerprints(messages)

    assert len(fingerprints) == 1
    assert committed_turns(messages) == [("question", "final")]
