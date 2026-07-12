"""Fixed-scope model tool contracts and argument validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

RecallLayer = Literal["recall", "deep_search"]
RecallIntent = Literal["general", "build", "plan", "research", "debug", "decide", "learn"]
RememberKind = Literal[
    "episode",
    "decision",
    "plan",
    "idea",
    "claim",
    "artifact",
    "procedure",
    "error_pattern",
    "session",
]
CorrectionToolAction = Literal[
    "wrong",
    "stale",
    "duplicate",
    "superseded",
    "revise",
    "sensitive",
    "hide",
    "restore",
]

_RECALL_LAYERS = {"recall", "deep_search"}
_RECALL_INTENTS = {"general", "build", "plan", "research", "debug", "decide", "learn"}
_REMEMBER_KINDS = {
    "episode",
    "decision",
    "plan",
    "idea",
    "claim",
    "artifact",
    "procedure",
    "error_pattern",
    "session",
}
CORRECTION_ACTIONS = {
    "wrong": "mark_wrong",
    "stale": "mark_stale",
    "duplicate": "mark_duplicate",
    "superseded": "supersede",
    "revise": "revise",
    "sensitive": "mark_sensitive",
    "hide": "hide",
    "restore": "restore",
}
MAX_TAGS = 32
MAX_TAG_LENGTH = 100


class ToolValidationError(ValueError):
    pass


def _required_string(args: dict[str, Any], key: str, *, maximum: int) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{key} is required")
    if len(value) > maximum:
        raise ToolValidationError(f"{key} exceeds {maximum} characters")
    return value


def _optional_string(args: dict[str, Any], key: str, *, maximum: int) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{key} must be a non-empty string")
    if len(value) > maximum:
        raise ToolValidationError(f"{key} exceeds {maximum} characters")
    return value


@dataclass(frozen=True, slots=True)
class RecallToolArgs:
    query: str
    layer: RecallLayer = "recall"
    intent: RecallIntent = "general"

    @classmethod
    def parse(cls, args: dict[str, Any]) -> RecallToolArgs:
        query = _required_string(args, "query", maximum=8_000)
        layer = args.get("layer", "recall")
        intent = args.get("intent", "general")
        if layer not in _RECALL_LAYERS:
            raise ToolValidationError("layer must be recall or deep_search")
        if intent not in _RECALL_INTENTS:
            raise ToolValidationError("intent is unsupported")
        return cls(
            query=query, layer=cast("RecallLayer", layer), intent=cast("RecallIntent", intent)
        )


@dataclass(frozen=True, slots=True)
class RememberToolArgs:
    title: str
    content: str
    kind: RememberKind
    tags: tuple[str, ...] = ()

    @classmethod
    def parse(cls, args: dict[str, Any]) -> RememberToolArgs:
        title = _required_string(args, "title", maximum=300)
        content = _required_string(args, "content", maximum=500_000)
        kind = args.get("kind")
        if kind not in _REMEMBER_KINDS:
            raise ToolValidationError("kind is unsupported")
        raw_tags = args.get("tags", [])
        if not isinstance(raw_tags, list) or len(raw_tags) > MAX_TAGS:
            raise ToolValidationError(f"tags must be an array with at most {MAX_TAGS} items")
        if any(
            not isinstance(tag, str) or not tag.strip() or len(tag) > MAX_TAG_LENGTH
            for tag in raw_tags
        ):
            raise ToolValidationError(
                f"tags must contain non-empty strings up to {MAX_TAG_LENGTH} characters"
            )
        tags = tuple(dict.fromkeys(tag.strip() for tag in raw_tags))
        return cls(
            title=title,
            content=content,
            kind=cast("RememberKind", kind),
            tags=tags,
        )


@dataclass(frozen=True, slots=True)
class CorrectionToolArgs:
    source_id: str
    action: CorrectionToolAction
    reason: str
    revised_content: str | None = None
    replacement_source_id: str | None = None
    duplicate_of_source_id: str | None = None
    apply: bool = False

    @classmethod
    def parse(cls, args: dict[str, Any]) -> CorrectionToolArgs:
        source_id = _required_string(args, "source_id", maximum=1_000)
        reason = _required_string(args, "reason", maximum=2_000)
        action = args.get("action")
        if action not in CORRECTION_ACTIONS:
            raise ToolValidationError("action is unsupported")
        revised_content = _optional_string(args, "revised_content", maximum=500_000)
        replacement_source_id = _optional_string(
            args,
            "replacement_source_id",
            maximum=500,
        )
        duplicate_of_source_id = _optional_string(
            args,
            "duplicate_of_source_id",
            maximum=500,
        )
        apply_value = args.get("apply", False)
        if not isinstance(apply_value, bool):
            raise ToolValidationError("apply must be a boolean")
        if action == "revise" and revised_content is None:
            raise ToolValidationError("revised_content is required for revise")
        if action == "superseded" and replacement_source_id is None:
            raise ToolValidationError("replacement_source_id is required for superseded")
        if action == "duplicate" and duplicate_of_source_id is None:
            raise ToolValidationError("duplicate_of_source_id is required for duplicate")
        return cls(
            source_id=source_id,
            action=cast("CorrectionToolAction", action),
            reason=reason,
            revised_content=revised_content,
            replacement_source_id=replacement_source_id,
            duplicate_of_source_id=duplicate_of_source_id,
            apply=apply_value,
        )


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "sibyl_recall",
        "description": "Recall governed Sibyl memory as untrusted evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 8000},
                "layer": {"type": "string", "enum": ["recall", "deep_search"]},
                "intent": {
                    "type": "string",
                    "enum": ["general", "build", "plan", "research", "debug", "decide", "learn"],
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sibyl_remember",
        "description": "Queue a deliberate source-preserving memory in the configured scope.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 300},
                "content": {"type": "string", "maxLength": 500000},
                "kind": {"type": "string", "enum": sorted(_REMEMBER_KINDS)},
                "tags": {
                    "type": "array",
                    "maxItems": MAX_TAGS,
                    "items": {"type": "string", "maxLength": MAX_TAG_LENGTH},
                },
            },
            "required": ["title", "content", "kind"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sibyl_correct",
        "description": "Preview or queue a revision-aware lifecycle correction.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "maxLength": 1000},
                "action": {"type": "string", "enum": sorted(CORRECTION_ACTIONS)},
                "reason": {"type": "string", "maxLength": 2000},
                "revised_content": {"type": "string", "maxLength": 500000},
                "replacement_source_id": {"type": "string", "maxLength": 500},
                "duplicate_of_source_id": {"type": "string", "maxLength": 500},
                "apply": {"type": "boolean", "default": False},
            },
            "required": ["source_id", "action", "reason"],
            "additionalProperties": False,
        },
    },
]


__all__ = [
    "CORRECTION_ACTIONS",
    "TOOL_SCHEMAS",
    "CorrectionToolArgs",
    "RecallToolArgs",
    "RememberToolArgs",
    "ToolValidationError",
]
