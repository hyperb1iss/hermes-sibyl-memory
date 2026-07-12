"""Canonical Hermes turn projection and stable provider identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

SOURCE_SCHEMA = "hermes-final-turn-v1"
CAPTURE_SURFACE = "hermes_memory_provider"
ADAPTER_NAME = "hermes-sibyl-memory"
_NUL = b"\0"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def message_hash(content: str) -> str:
    return _sha256(content.encode("utf-8"))


def turn_hash(user_content: str, assistant_content: str) -> str:
    return _sha256(user_content.encode("utf-8") + _NUL + assistant_content.encode("utf-8"))


def canonical_source(user_content: str, assistant_content: str) -> str:
    return f"[User]\n{user_content}\n\n[Assistant]\n{assistant_content}"


def canonical_request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _delimited_hash(*parts: str) -> str:
    return _sha256(_NUL.join(part.encode("utf-8") for part in parts))


@dataclass(frozen=True, slots=True)
class TurnIdentity:
    operation_id: str
    source_id: str
    idempotency_key: str
    turn_hash: str
    local_sequence: int
    turn_number: int
    turn_number_source: str


def derive_turn_identity(
    *,
    agent_id: str,
    session_id: str,
    local_sequence: int,
    turn_number: int | None,
    user_content: str,
    assistant_content: str,
) -> TurnIdentity:
    if local_sequence < 1:
        raise ValueError("local_sequence must be positive")
    effective_turn = turn_number if turn_number is not None else local_sequence
    if effective_turn < 0:
        raise ValueError("turn_number must be non-negative")

    digest = turn_hash(user_content, assistant_content)
    operation_id = _delimited_hash(
        agent_id,
        session_id,
        str(local_sequence),
        str(effective_turn),
        digest,
    )
    source_id = ":".join(
        (
            "hermes",
            "turn",
            quote(agent_id, safe=""),
            quote(session_id, safe=""),
            str(local_sequence),
            str(effective_turn),
            digest[:16],
        )
    )
    return TurnIdentity(
        operation_id=operation_id,
        source_id=source_id,
        idempotency_key=f"hermes-turn-{operation_id}",
        turn_hash=digest,
        local_sequence=local_sequence,
        turn_number=effective_turn,
        turn_number_source="hermes" if turn_number is not None else "local-sequence-fallback",
    )


@dataclass(frozen=True, slots=True)
class TurnCapture:
    identity: TurnIdentity
    payload: dict[str, Any]


def build_turn_capture(
    *,
    agent_id: str,
    agent_workspace: str,
    agent_identity: str,
    session_id: str,
    parent_session_id: str,
    local_sequence: int,
    turn_number: int | None,
    user_content: str,
    assistant_content: str,
    platform: str,
    project_id: str,
    memory_space_id: str,
    hermes_version: str,
    adapter_version: str,
    participant_ids: list[str] | None = None,
    chat_id: str = "",
    thread_id: str = "",
) -> TurnCapture:
    identity = derive_turn_identity(
        agent_id=agent_id,
        session_id=session_id,
        local_sequence=local_sequence,
        turn_number=turn_number,
        user_content=user_content,
        assistant_content=assistant_content,
    )
    metadata: dict[str, Any] = {
        "source_schema": SOURCE_SCHEMA,
        "hermes_version": hermes_version,
        "agent_id": agent_id,
        "agent_workspace": agent_workspace,
        "agent_identity": agent_identity,
        "platform": platform,
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "local_sequence": local_sequence,
        "turn_number": identity.turn_number,
        "turn_number_source": identity.turn_number_source,
        "turn_hash": identity.turn_hash,
        "provider_operation_id": identity.operation_id,
        "trust_level": "untrusted_conversation",
        "contains_tool_messages": False,
        "project_id": project_id,
        "memory_space_id": memory_space_id,
    }
    provenance: dict[str, Any] = {
        "adapter": ADAPTER_NAME,
        "adapter_version": adapter_version,
        "participant_ids": list(participant_ids or []),
        "chat_id": chat_id,
        "thread_id": thread_id,
    }
    payload: dict[str, Any] = {
        "title": f"Hermes turn {session_id}/{identity.turn_number}",
        "raw_content": canonical_source(user_content, assistant_content),
        "source_id": identity.source_id,
        "memory_scope": "project",
        "scope_key": project_id,
        "project_id": project_id,
        "tags": ["hermes", "conversation", "completed-turn"],
        "capture_surface": CAPTURE_SURFACE,
        "metadata": metadata,
        "provenance": provenance,
    }
    metadata["provider_request_hash"] = canonical_request_hash(payload)
    return TurnCapture(identity=identity, payload=payload)


def committed_turns(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user" and isinstance(content, str):
            pending_user = content
            continue
        if role != "assistant" or pending_user is None or not isinstance(content, str):
            continue
        if message.get("tool_calls"):
            continue
        turns.append((pending_user, content))
        pending_user = None
    return turns


def committed_turn_fingerprints(messages: list[dict[str, Any]]) -> list[str]:
    return [turn_hash(user, assistant) for user, assistant in committed_turns(messages)]


__all__ = [
    "ADAPTER_NAME",
    "CAPTURE_SURFACE",
    "SOURCE_SCHEMA",
    "TurnCapture",
    "TurnIdentity",
    "build_turn_capture",
    "canonical_request_hash",
    "canonical_source",
    "committed_turn_fingerprints",
    "committed_turns",
    "derive_turn_identity",
    "message_hash",
    "turn_hash",
]
