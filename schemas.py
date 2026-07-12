"""Strict wire models for the Sibyl memory-provider REST contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]

ContextIntent: TypeAlias = Literal[
    "build",
    "plan",
    "ideate",
    "research",
    "review",
    "debug",
    "decide",
    "learn",
    "general",
]
ContextLayer: TypeAlias = Literal["wake", "recall", "deep_search"]
CorrectionAction: TypeAlias = Literal[
    "hide",
    "mark_duplicate",
    "mark_sensitive",
    "mark_stale",
    "mark_wrong",
    "revise",
    "restore",
    "supersede",
]


class SchemaError(ValueError):
    """A Sibyl response did not match the versioned provider contract."""


def _object(value: object, path: str) -> JSONObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SchemaError(f"{path} must be an object")
    return cast("JSONObject", value)


def _array(value: object, path: str) -> list[JSONValue]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    return cast("list[JSONValue]", value)


def _required_string(value: JSONObject, key: str, path: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise SchemaError(f"{path}.{key} must be a string")
    return item


def _optional_string(value: JSONObject, key: str, path: str) -> str | None:
    item = value.get(key)
    if item is not None and not isinstance(item, str):
        raise SchemaError(f"{path}.{key} must be a string or null")
    return item


def _required_bool(value: JSONObject, key: str, path: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise SchemaError(f"{path}.{key} must be a boolean")
    return item


def _required_int(value: JSONObject, key: str, path: str, *, minimum: int = 0) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise SchemaError(f"{path}.{key} must be an integer >= {minimum}")
    return item


def _optional_int(value: JSONObject, key: str, path: str, *, minimum: int = 0) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise SchemaError(f"{path}.{key} must be an integer >= {minimum} or null")
    return item


def _required_number(value: JSONObject, key: str, path: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int | float):
        raise SchemaError(f"{path}.{key} must be a number")
    return float(item)


def _string_list_value(value: object, path: str) -> tuple[str, ...]:
    items = _array(value, path)
    if any(not isinstance(item, str) for item in items):
        raise SchemaError(f"{path} must contain only strings")
    return tuple(cast("list[str]", items))


def _required_string_list(value: JSONObject, key: str, path: str) -> tuple[str, ...]:
    if key not in value:
        raise SchemaError(f"{path}.{key} is required")
    return _string_list_value(value[key], f"{path}.{key}")


def _optional_object(value: JSONObject, key: str, path: str) -> JSONObject:
    item = value.get(key, {})
    return _object(item, f"{path}.{key}")


@dataclass(frozen=True, slots=True)
class ContextPackRequest:
    goal: str
    project: str
    agent_id: str
    intent: ContextIntent = "general"
    layer: ContextLayer = "recall"
    limit: int = 8
    include_related: bool = True
    related_limit: int = 2
    audit: bool = False
    record_exposure: bool = False
    markdown_token_budget: int = 900

    def to_json(self) -> JSONObject:
        return {
            "goal": self.goal,
            "intent": self.intent,
            "layer": self.layer,
            "project": self.project,
            "agent_id": self.agent_id,
            "limit": self.limit,
            "include_related": self.include_related,
            "related_limit": self.related_limit,
            "audit": self.audit,
            "record_exposure": self.record_exposure,
            "markdown_token_budget": self.markdown_token_budget,
        }


@dataclass(frozen=True, slots=True)
class ContextExposureRequest:
    exposed_ids: tuple[str, ...]
    project_id: str
    session_id_hash: str | None = None
    query_hash: str | None = None
    provider_operation_id: str | None = None
    automatic: bool = True

    def to_json(self) -> JSONObject:
        metadata: JSONObject = {"automatic": self.automatic}
        if self.session_id_hash is not None:
            metadata["session_id_hash"] = self.session_id_hash
        if self.query_hash is not None:
            metadata["query_hash"] = self.query_hash
        if self.provider_operation_id is not None:
            metadata["provider_operation_id"] = self.provider_operation_id
        return {
            "exposed_ids": list(self.exposed_ids),
            "project_id": self.project_id,
            "source_surface": "hermes_memory_provider",
            "metadata": metadata,
        }


@dataclass(frozen=True, slots=True)
class RawMemoryRequest:
    title: str
    raw_content: str
    source_id: str
    project_id: str
    metadata: JSONObject
    provenance: JSONObject
    tags: tuple[str, ...] = ("hermes", "conversation", "completed-turn")
    memory_scope: Literal["project"] = "project"
    capture_surface: Literal["hermes_memory_provider"] = "hermes_memory_provider"

    def to_json(self) -> JSONObject:
        return {
            "title": self.title,
            "raw_content": self.raw_content,
            "source_id": self.source_id,
            "memory_scope": self.memory_scope,
            "scope_key": self.project_id,
            "project_id": self.project_id,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
            "capture_surface": self.capture_surface,
        }


@dataclass(frozen=True, slots=True)
class CorrectionRequest:
    action: CorrectionAction
    reason: str
    replacement_source_id: str | None = None
    duplicate_of_source_id: str | None = None
    revised_content: str | None = None
    expected_revision: int | None = None
    metadata: JSONObject = field(default_factory=dict)

    def to_json(self) -> JSONObject:
        payload: JSONObject = {
            "action": self.action,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
        for key, value in (
            ("replacement_source_id", self.replacement_source_id),
            ("duplicate_of_source_id", self.duplicate_of_source_id),
            ("revised_content", self.revised_content),
            ("expected_revision", self.expected_revision),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    operation_id: str
    applied: bool
    revision: int | None
    affected_records: tuple[str, ...]
    idempotency_key: str | None
    replayed: bool

    @classmethod
    def from_json(cls, value: object, path: str = "response.mutation_receipt") -> MutationReceipt:
        data = _object(value, path)
        return cls(
            operation_id=_required_string(data, "operation_id", path),
            applied=_required_bool(data, "applied", path),
            revision=_optional_int(data, "revision", path, minimum=1),
            affected_records=_required_string_list(data, "affected_records", path),
            idempotency_key=_optional_string(data, "idempotency_key", path),
            replayed=_required_bool(data, "replayed", path),
        )


@dataclass(frozen=True, slots=True)
class Credential:
    type: Literal["api_key", "session"]
    id: str | None = None
    scopes: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()
    memory_space_ids: tuple[str, ...] = ()
    agent_id: str | None = None
    delegated_authority: str | None = None
    capability_profile: str | None = None

    @classmethod
    def from_json(cls, value: object) -> Credential:
        path = "response.credential"
        data = _object(value, path)
        credential_type = _required_string(data, "type", path)
        if credential_type not in {"api_key", "session"}:
            raise SchemaError(f"{path}.type is incompatible")
        if credential_type == "session":
            return cls(type="session")
        return cls(
            type="api_key",
            id=_required_string(data, "id", path),
            scopes=_required_string_list(data, "scopes", path),
            project_ids=_required_string_list(data, "project_ids", path),
            memory_space_ids=_required_string_list(data, "memory_space_ids", path),
            agent_id=_optional_string(data, "agent_id", path),
            delegated_authority=_optional_string(data, "delegated_authority", path),
            capability_profile=_optional_string(data, "capability_profile", path),
        )


@dataclass(frozen=True, slots=True)
class AuthMeResponse:
    credential: Credential
    user: JSONObject
    organization: JSONObject | None
    org_role: str | None

    @classmethod
    def from_json(cls, value: object) -> AuthMeResponse:
        path = "response"
        data = _object(value, path)
        organization_value = data.get("organization")
        organization = (
            None
            if organization_value is None
            else _object(organization_value, "response.organization")
        )
        return cls(
            credential=Credential.from_json(data.get("credential")),
            user=_object(data.get("user"), "response.user"),
            organization=organization,
            org_role=_optional_string(data, "org_role", path),
        )


@dataclass(frozen=True, slots=True)
class ContextRelatedItem:
    id: str
    type: str
    name: str
    relationship: str
    direction: str
    distance: int

    @classmethod
    def from_json(cls, value: object, path: str) -> ContextRelatedItem:
        data = _object(value, path)
        return cls(
            id=_required_string(data, "id", path),
            type=_required_string(data, "type", path),
            name=_required_string(data, "name", path),
            relationship=_required_string(data, "relationship", path),
            direction=_required_string(data, "direction", path),
            distance=_required_int(data, "distance", path, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class ContextItem:
    id: str
    type: str
    name: str
    content: str
    score: float
    facet: str
    reason: str
    source: str | None
    quality: JSONObject
    metadata: JSONObject
    related: tuple[ContextRelatedItem, ...]

    @classmethod
    def from_json(cls, value: object, path: str) -> ContextItem:
        data = _object(value, path)
        related = tuple(
            ContextRelatedItem.from_json(item, f"{path}.related[{index}]")
            for index, item in enumerate(_array(data.get("related"), f"{path}.related"))
        )
        return cls(
            id=_required_string(data, "id", path),
            type=_required_string(data, "type", path),
            name=_required_string(data, "name", path),
            content=_required_string(data, "content", path),
            score=_required_number(data, "score", path),
            facet=_required_string(data, "facet", path),
            reason=_required_string(data, "reason", path),
            source=_optional_string(data, "source", path),
            quality=_optional_object(data, "quality", path),
            metadata=_optional_object(data, "metadata", path),
            related=related,
        )


@dataclass(frozen=True, slots=True)
class ContextSection:
    facet: str
    title: str
    items: tuple[ContextItem, ...]

    @classmethod
    def from_json(cls, value: object, path: str) -> ContextSection:
        data = _object(value, path)
        items = tuple(
            ContextItem.from_json(item, f"{path}.items[{index}]")
            for index, item in enumerate(_array(data.get("items"), f"{path}.items"))
        )
        return cls(
            facet=_required_string(data, "facet", path),
            title=_required_string(data, "title", path),
            items=items,
        )


@dataclass(frozen=True, slots=True)
class ContextPackResponse:
    goal: str
    intent: str
    layer: str
    query: str
    domain: str | None
    project: str | None
    sections: tuple[ContextSection, ...]
    total_items: int
    usage_metadata: JSONObject
    usage_hint: str
    markdown: str | None
    rendered_item_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> ContextPackResponse:
        path = "response"
        data = _object(value, path)
        sections = tuple(
            ContextSection.from_json(item, f"response.sections[{index}]")
            for index, item in enumerate(_array(data.get("sections"), "response.sections"))
        )
        return cls(
            goal=_required_string(data, "goal", path),
            intent=_required_string(data, "intent", path),
            layer=_required_string(data, "layer", path),
            query=_required_string(data, "query", path),
            domain=_optional_string(data, "domain", path),
            project=_optional_string(data, "project", path),
            sections=sections,
            total_items=_required_int(data, "total_items", path),
            usage_metadata=_optional_object(data, "usage_metadata", path),
            usage_hint=_required_string(data, "usage_hint", path),
            markdown=_optional_string(data, "markdown", path),
            rendered_item_ids=_required_string_list(data, "rendered_item_ids", path),
        )


@dataclass(frozen=True, slots=True)
class ContextExposureResponse:
    recorded_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    denied_ids: tuple[str, ...]
    mutation_receipt: MutationReceipt

    @classmethod
    def from_json(cls, value: object) -> ContextExposureResponse:
        data = _object(value, "response")
        return cls(
            recorded_ids=_required_string_list(data, "recorded_ids", "response"),
            excluded_ids=_required_string_list(data, "excluded_ids", "response"),
            denied_ids=_required_string_list(data, "denied_ids", "response"),
            mutation_receipt=MutationReceipt.from_json(data.get("mutation_receipt")),
        )


@dataclass(frozen=True, slots=True)
class RawMemoryResponse:
    id: str
    organization_id: str
    source_id: str
    principal_id: str
    memory_scope: str
    scope_key: str | None
    title: str
    raw_content: str
    tags: tuple[str, ...]
    metadata: JSONObject
    provenance: JSONObject
    capture_surface: str | None
    revision: int
    mutation_receipt: MutationReceipt | None

    @classmethod
    def from_json(cls, value: object) -> RawMemoryResponse:
        data = _object(value, "response")
        receipt_value = data.get("mutation_receipt")
        return cls(
            id=_required_string(data, "id", "response"),
            organization_id=_required_string(data, "organization_id", "response"),
            source_id=_required_string(data, "source_id", "response"),
            principal_id=_required_string(data, "principal_id", "response"),
            memory_scope=_required_string(data, "memory_scope", "response"),
            scope_key=_optional_string(data, "scope_key", "response"),
            title=_required_string(data, "title", "response"),
            raw_content=_required_string(data, "raw_content", "response"),
            tags=_required_string_list(data, "tags", "response"),
            metadata=_optional_object(data, "metadata", "response"),
            provenance=_optional_object(data, "provenance", "response"),
            capture_surface=_optional_string(data, "capture_surface", "response"),
            revision=_required_int(data, "revision", "response", minimum=1),
            mutation_receipt=(
                None if receipt_value is None else MutationReceipt.from_json(receipt_value)
            ),
        )


@dataclass(frozen=True, slots=True)
class CorrectionResponse:
    allowed: bool
    applied: bool
    source_id: str
    action: str
    reason: str
    target_lifecycle_state: str
    target_lifecycle_flags: tuple[str, ...]
    updated_review_state: str | None
    lifecycle: JSONObject
    reflection_finding: JSONObject | None
    affected_source_ids: tuple[str, ...]
    affected_derived_ids: tuple[str, ...]
    reversible: bool
    recall_impact: JSONObject
    synthesis_impact: JSONObject
    audit_action: str
    policy_reasons: tuple[str, ...]
    metadata: JSONObject
    current_revision: int | None
    revision: int | None
    mutation_receipt: MutationReceipt | None

    @classmethod
    def from_json(cls, value: object) -> CorrectionResponse:
        data = _object(value, "response")
        reflection_value = data.get("reflection_finding")
        receipt_value = data.get("mutation_receipt")
        return cls(
            allowed=_required_bool(data, "allowed", "response"),
            applied=_required_bool(data, "applied", "response"),
            source_id=_required_string(data, "source_id", "response"),
            action=_required_string(data, "action", "response"),
            reason=_required_string(data, "reason", "response"),
            target_lifecycle_state=_required_string(data, "target_lifecycle_state", "response"),
            target_lifecycle_flags=_required_string_list(
                data, "target_lifecycle_flags", "response"
            ),
            updated_review_state=_optional_string(data, "updated_review_state", "response"),
            lifecycle=_optional_object(data, "lifecycle", "response"),
            reflection_finding=(
                None
                if reflection_value is None
                else _object(reflection_value, "response.reflection_finding")
            ),
            affected_source_ids=_required_string_list(data, "affected_source_ids", "response"),
            affected_derived_ids=_required_string_list(data, "affected_derived_ids", "response"),
            reversible=_required_bool(data, "reversible", "response"),
            recall_impact=_optional_object(data, "recall_impact", "response"),
            synthesis_impact=_optional_object(data, "synthesis_impact", "response"),
            audit_action=_required_string(data, "audit_action", "response"),
            policy_reasons=_required_string_list(data, "policy_reasons", "response"),
            metadata=_optional_object(data, "metadata", "response"),
            current_revision=_optional_int(data, "current_revision", "response", minimum=1),
            revision=_optional_int(data, "revision", "response", minimum=1),
            mutation_receipt=(
                None if receipt_value is None else MutationReceipt.from_json(receipt_value)
            ),
        )
