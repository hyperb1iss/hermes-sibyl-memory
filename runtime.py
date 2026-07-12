"""Governed recall, capture, recovery, and session lifecycle runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import Counter
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, unquote

from .capture import build_turn_capture, committed_turn_fingerprints, committed_turns, turn_hash
from .client import (
    SibylClient,
    SibylConflictError,
    SibylHTTPError,
    SibylProtocolError,
    SibylTransportError,
)
from .outbox import (
    AttemptResult,
    ClaimedOperation,
    DurableOutbox,
    NewOperation,
    OperationState,
    ResultAction,
    classify_http_failure,
)
from .recall import RecallCoordinator, RecallObservation, RecallResult
from .schemas import (
    AuthMeResponse,
    ContextExposureRequest,
    ContextPackRequest,
    CorrectionRequest,
    JSONObject,
    RawMemoryRequest,
)
from .sessions import IndexedTurn, TurnReconciliationError, TurnStartQueue, plan_rewind_corrections

if TYPE_CHECKING:
    from .provider import RuntimeContext

log = logging.getLogger(__name__)
ADAPTER_VERSION = "0.1.0"
HERMES_VERSION = "0.18.2"


def _hash_parts(*parts: str) -> str:
    return hashlib.sha256(b"\0".join(part.encode("utf-8") for part in parts)).hexdigest()


class SibylRuntime:
    def __init__(
        self,
        context: RuntimeContext,
        *,
        client: SibylClient | None = None,
        outbox: DurableOutbox | None = None,
        executor: Executor | None = None,
    ) -> None:
        self._context = context
        self._client = client or SibylClient(
            base_url=context.config.base_url,
            api_key=context.api_key,
        )
        self._outbox = outbox or DurableOutbox(context.hermes_home)
        self._executor = executor or ThreadPoolExecutor(thread_name_prefix="sibyl-provider")
        self._owns_executor = executor is None
        self._turns = TurnStartQueue()
        self._session_id = context.session_id
        self._parent_session_id = context.parent_session_id
        self._credential_safe: bool | None = None
        self._credential_error = ""
        self._credential_lock = threading.Lock()
        self._closed = False
        self._recall = RecallCoordinator(
            self._fetch_context,
            self._record_delivery,
            executor=executor,
            observe=self._observe_recall,
        )

    def initialize(self, context: RuntimeContext) -> None:
        self._outbox.mark_session_reconcile(
            self._session_id,
            parent_session_id=self._parent_session_id,
            required=True,
        )
        self._executor.submit(self._validate_and_flush)
        log.info(
            "sibyl_provider_initialized",
            extra={
                "agent_id": context.agent_id,
                "session_hash": self._session_hash(self._session_id),
                "project_id": context.config.project_id,
            },
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        if not message:
            return
        self._turns.record(self._session_id, turn_number, message)
        self._recall.schedule(message, session_id=self._session_id)
        self._executor.submit(self._validate_and_flush)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self._recall.prefetch(query, session_id=session_id or self._session_id)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._context.agent_context != "primary" or not self._context.config.automatic_capture:
            return
        active_session = session_id or self._session_id
        if messages is not None and self._outbox.session_reconcile_required(active_session):
            self._reconcile_session(
                active_session,
                messages,
                current_turn=(user_content, assistant_content),
            )
        try:
            turn_number = self._turns.consume(active_session, user_content).turn_number
        except TurnReconciliationError as exc:
            turn_number = None
            log.warning(
                "sibyl_turn_reconciliation_failed",
                extra={
                    "session_hash": self._session_hash(active_session),
                    "error_class": type(exc).__name__,
                },
            )
        operation_id = self._enqueue_capture(
            active_session,
            user_content,
            assistant_content,
            turn_number=turn_number,
        )
        log.info(
            "sibyl_capture_queued",
            extra={
                "session_hash": self._session_hash(active_session),
                "operation_id": operation_id,
                "outbox_depth": self._outbox.snapshot().total,
            },
        )
        self._executor.submit(self._validate_and_flush)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        old_session = self._session_id
        if reset:
            self._turns.clear(old_session)
            self._recall.discard_session(old_session)
        self._session_id = new_session_id
        self._parent_session_id = parent_session_id
        self._outbox.mark_session_reconcile(
            new_session_id,
            parent_session_id=parent_session_id,
            required=True,
        )
        if rewound:
            self._recall.discard_session(new_session_id)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        return json.dumps({"error": f"Unknown Sibyl tool: {tool_name}"}, separators=(",", ":"))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._recall.shutdown()
        if self._credential_safe:
            self._outbox.shutdown(
                f"shutdown-{threading.get_ident()}",
                self._execute_operation,
                timeout_seconds=4.0,
            )
        else:
            self._outbox.close()
        self._client.close()
        if self._owns_executor and isinstance(self._executor, ThreadPoolExecutor):
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _validate_and_flush(self) -> None:
        if self._closed:
            return
        try:
            self._ensure_credential_safe()
        except Exception as exc:
            log.warning(
                "sibyl_credential_probe_failed",
                extra={"error_class": type(exc).__name__},
            )
            return
        report = self._outbox.flush(
            f"runtime-{threading.get_ident()}",
            self._execute_operation,
        )
        if report.remaining.by_state.get("blocked_auth"):
            log.warning(
                "sibyl_outbox_blocked_auth",
                extra={"outbox_depth": report.remaining.total},
            )
        if report.remaining.by_state.get("dead_letter"):
            log.error(
                "sibyl_outbox_dead_letter",
                extra={"outbox_depth": report.remaining.total},
            )

    def _ensure_credential_safe(self) -> None:
        with self._credential_lock:
            if self._credential_safe is True:
                return
            if self._credential_safe is False:
                raise RuntimeError(self._credential_error or "Sibyl credential is unsafe")
            response = self._client.auth_me()
            self._validate_credential(response)
            self._credential_safe = True

    def _validate_credential(self, response: AuthMeResponse) -> None:
        credential = response.credential
        failures: list[str] = []
        if credential.type != "api_key":
            failures.append("credential must be an API key")
        if "api:write" not in credential.scopes:
            failures.append("api:write scope is required")
        if (
            not credential.project_ids
            or self._context.config.project_id not in credential.project_ids
        ):
            failures.append("configured project is not the key's restricted project")
        if (
            not credential.memory_space_ids
            or self._context.config.memory_space_id not in credential.memory_space_ids
        ):
            failures.append("configured memory space is not restricted on the key")
        if credential.agent_id != self._context.agent_id:
            failures.append("authenticated agent identity does not match Hermes")
        if credential.capability_profile != "memory_provider":
            failures.append("capability_profile must be memory_provider")
        if failures:
            self._credential_safe = False
            self._credential_error = "; ".join(failures)
            raise RuntimeError(self._credential_error)

    def _fetch_context(self, query: str, session_id: str) -> RecallResult:
        self._ensure_credential_safe()
        response = self._client.context_pack(
            ContextPackRequest(
                goal=query,
                project=self._context.config.project_id,
                agent_id=self._context.agent_id,
                limit=self._context.config.context_limit,
                related_limit=self._context.config.context_related_limit,
                markdown_token_budget=self._context.config.context_token_budget,
            )
        )
        return RecallResult(
            markdown=response.markdown or "",
            rendered_item_ids=response.rendered_item_ids,
            total_items=response.total_items,
        )

    def _record_delivery(self, session_id: str, query_hash: str, result: RecallResult) -> None:
        if not result.rendered_item_ids:
            return
        session_hash = self._session_hash(session_id)
        operation_id = _hash_parts(
            self._context.agent_id,
            session_hash,
            query_hash,
            *result.rendered_item_ids,
        )
        request = ContextExposureRequest(
            exposed_ids=result.rendered_item_ids,
            project_id=self._context.config.project_id,
            session_id_hash=session_hash,
            query_hash=query_hash,
        )
        self._outbox.enqueue(
            NewOperation(
                operation_id=operation_id,
                kind="context_exposure",
                endpoint="/api/memory/expose",
                request=request.to_json(),
                idempotency_key=f"hermes-context-{operation_id}",
            )
        )
        self._executor.submit(self._validate_and_flush)

    def _enqueue_capture(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        turn_number: int | None,
    ) -> str:
        local_sequence = self._outbox.reserve_sequence(
            session_id,
            parent_session_id=self._parent_session_id,
        )
        capture = build_turn_capture(
            agent_id=self._context.agent_id,
            agent_workspace=self._context.agent_workspace,
            agent_identity=self._context.agent_identity,
            session_id=session_id,
            parent_session_id=self._parent_session_id,
            local_sequence=local_sequence,
            turn_number=turn_number,
            user_content=user_content,
            assistant_content=assistant_content,
            platform=self._context.platform,
            project_id=self._context.config.project_id,
            memory_space_id=self._context.config.memory_space_id,
            hermes_version=HERMES_VERSION,
            adapter_version=ADAPTER_VERSION,
            participant_ids=self._participant_ids(),
            chat_id=self._stable_context_value("chat_id", "group_id", "conversation_id"),
            thread_id=self._stable_context_value("thread_id"),
        )
        request = cast("JSONObject", capture.payload)
        self._outbox.enqueue(
            NewOperation(
                operation_id=capture.identity.operation_id,
                kind="raw_capture",
                endpoint="/api/memory/raw",
                request=request,
                idempotency_key=capture.identity.idempotency_key,
            )
        )
        self._outbox.index_turn(
            session_id=session_id,
            local_sequence=local_sequence,
            operation_id=capture.identity.operation_id,
            source_id=capture.identity.source_id,
            turn_hash=capture.identity.turn_hash,
        )
        return capture.identity.operation_id

    def _reconcile_session(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        current_turn: tuple[str, str],
    ) -> None:
        transcript_turns = committed_turns(messages)
        prior_turns = transcript_turns
        if transcript_turns and transcript_turns[-1] == current_turn:
            prior_turns = transcript_turns[:-1]
        indexed_records = self._outbox.indexed_turns(session_id)
        indexed_counts = Counter(record.turn_hash for record in indexed_records)
        seen: Counter[str] = Counter()
        for user_content, assistant_content in prior_turns:
            digest = turn_hash(user_content, assistant_content)
            seen[digest] += 1
            if seen[digest] > indexed_counts[digest]:
                self._enqueue_capture(
                    session_id,
                    user_content,
                    assistant_content,
                    turn_number=None,
                )

        indexed = [
            IndexedTurn(
                operation_id=record.operation_id,
                source_id=record.source_id,
                turn_hash=record.turn_hash,
                local_sequence=record.local_sequence,
                revision=record.revision,
                queued=record.queued,
            )
            for record in self._outbox.indexed_turns(session_id)
        ]
        for correction in plan_rewind_corrections(
            indexed,
            committed_turn_fingerprints(messages),
        ):
            provider_operation_id = _hash_parts(
                self._context.agent_id,
                correction.operation_id,
            )
            request = CorrectionRequest(
                action="mark_stale",
                reason="hermes_session_rewind",
                expected_revision=correction.expected_revision,
                metadata={"provider_operation_id": provider_operation_id},
            )
            self._outbox.enqueue(
                NewOperation(
                    operation_id=provider_operation_id,
                    kind="rewind_correction",
                    endpoint=(
                        f"/api/memory/inspect/{quote(correction.source_id, safe='')}/corrections"
                    ),
                    request=request.to_json(),
                    idempotency_key=f"hermes-correction-{provider_operation_id}",
                    depends_on_operation_id=correction.depends_on_operation_id,
                )
            )
        self._outbox.mark_session_reconcile(session_id, required=False)
        log.info(
            "sibyl_rewind_reconciled",
            extra={"session_hash": self._session_hash(session_id)},
        )

    def _execute_operation(self, operation: ClaimedOperation) -> AttemptResult:
        try:
            if operation.kind == "raw_capture":
                response = self._client.remember_raw(
                    self._raw_request(operation.request),
                    idempotency_key=operation.idempotency_key,
                )
                self._outbox.record_turn_revision(operation.operation_id, response.revision)
                receipt = response.mutation_receipt
                return AttemptResult.success(
                    replayed=bool(receipt and receipt.replayed),
                    status=200,
                )
            if operation.kind == "context_exposure":
                response = self._client.expose_context(
                    self._exposure_request(operation.request),
                    idempotency_key=operation.idempotency_key,
                )
                return AttemptResult.success(
                    replayed=response.mutation_receipt.replayed,
                    status=200,
                )
            if operation.endpoint.endswith("/corrections"):
                source_id = self._correction_source_id(operation.endpoint)
                request = self._correction_request(operation.request)
                if operation.state is OperationState.RECONCILE_REVISION:
                    return self._reconcile_correction(operation, source_id, request)
                response = self._client.apply_correction(
                    source_id,
                    request,
                    idempotency_key=operation.idempotency_key,
                )
                receipt = response.mutation_receipt
                return AttemptResult.success(
                    replayed=bool(receipt and receipt.replayed),
                    status=200,
                )
            return AttemptResult.failure(
                ResultAction.DEAD_LETTER,
                error="unsupported outbox operation",
            )
        except SibylConflictError as exc:
            return AttemptResult.failure(
                classify_http_failure(
                    status=exc.status_code,
                    error_code=exc.error,
                    is_correction=operation.endpoint.endswith("/corrections"),
                    dependency_pending=operation.depends_on_operation_id is not None,
                ),
                status=exc.status_code,
                error=exc.error,
            )
        except SibylHTTPError as exc:
            return AttemptResult.failure(
                classify_http_failure(
                    status=exc.status_code,
                    error_code=exc.error,
                    is_correction=operation.endpoint.endswith("/corrections"),
                    dependency_pending=operation.depends_on_operation_id is not None,
                ),
                status=exc.status_code,
                error=exc.error,
            )
        except SibylTransportError as exc:
            return AttemptResult.failure(ResultAction.RETRY, error=exc.error_type)
        except SibylProtocolError as exc:
            return AttemptResult.failure(
                ResultAction.DEAD_LETTER,
                status=exc.status_code,
                error="invalid Sibyl response contract",
            )
        except Exception as exc:
            return AttemptResult.failure(ResultAction.RETRY, error=type(exc).__name__)

    def _reconcile_correction(
        self,
        operation: ClaimedOperation,
        source_id: str,
        request: CorrectionRequest,
    ) -> AttemptResult:
        preview = self._client.preview_correction(source_id, request)
        if preview.target_lifecycle_state == "erased" or (
            request.action == "mark_stale" and preview.target_lifecycle_state == "stale"
        ):
            return AttemptResult.failure(ResultAction.OBSOLETE, status=200)
        current_revision = preview.current_revision or preview.revision
        if current_revision is None:
            return AttemptResult.failure(
                ResultAction.DEAD_LETTER,
                error="correction preview omitted current revision",
            )
        replacement_id = _hash_parts(
            operation.operation_id,
            str(current_revision),
            request.action,
        )
        replacement_request = CorrectionRequest(
            action=request.action,
            reason=request.reason,
            replacement_source_id=request.replacement_source_id,
            duplicate_of_source_id=request.duplicate_of_source_id,
            revised_content=request.revised_content,
            expected_revision=current_revision,
            metadata={**request.metadata, "provider_operation_id": replacement_id},
        )
        self._outbox.supersede(
            operation,
            NewOperation(
                operation_id=replacement_id,
                kind=operation.kind,
                endpoint=operation.endpoint,
                request=replacement_request.to_json(),
                idempotency_key=f"hermes-correction-{replacement_id}",
                depends_on_operation_id=operation.depends_on_operation_id,
            ),
        )
        return AttemptResult.failure(ResultAction.SUPERSEDED, status=200)

    @staticmethod
    def _raw_request(data: dict[str, Any]) -> RawMemoryRequest:
        return RawMemoryRequest(
            title=str(data["title"]),
            raw_content=str(data["raw_content"]),
            source_id=str(data["source_id"]),
            project_id=str(data["project_id"]),
            metadata=cast("JSONObject", data["metadata"]),
            provenance=cast("JSONObject", data["provenance"]),
            tags=tuple(str(value) for value in data.get("tags", [])),
        )

    @staticmethod
    def _exposure_request(data: dict[str, Any]) -> ContextExposureRequest:
        metadata = cast("dict[str, Any]", data.get("metadata", {}))
        return ContextExposureRequest(
            exposed_ids=tuple(str(value) for value in data["exposed_ids"]),
            project_id=str(data["project_id"]),
            session_id_hash=cast("str | None", metadata.get("session_id_hash")),
            query_hash=cast("str | None", metadata.get("query_hash")),
            automatic=bool(metadata.get("automatic", True)),
        )

    @staticmethod
    def _correction_request(data: dict[str, Any]) -> CorrectionRequest:
        return CorrectionRequest(
            action=data["action"],
            reason=str(data["reason"]),
            replacement_source_id=cast("str | None", data.get("replacement_source_id")),
            duplicate_of_source_id=cast("str | None", data.get("duplicate_of_source_id")),
            revised_content=cast("str | None", data.get("revised_content")),
            expected_revision=cast("int | None", data.get("expected_revision")),
            metadata=cast("JSONObject", data.get("metadata", {})),
        )

    @staticmethod
    def _correction_source_id(endpoint: str) -> str:
        marker = "/api/memory/inspect/"
        if not endpoint.startswith(marker) or not endpoint.endswith("/corrections"):
            raise ValueError("invalid correction endpoint")
        return unquote(endpoint[len(marker) : -len("/corrections")])

    def _participant_ids(self) -> list[str]:
        values = [
            self._stable_context_value("user_id"),
            self._stable_context_value("user_id_alt"),
        ]
        return list(dict.fromkeys(value for value in values if value))

    def _stable_context_value(self, *keys: str) -> str:
        for key in keys:
            value = self._context.kwargs.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _session_hash(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _observe_recall(observation: RecallObservation) -> None:
        level = (
            logging.WARNING if observation.event.endswith(("failed", "missed")) else logging.INFO
        )
        log.log(
            level,
            observation.event,
            extra={
                "session_hash": observation.session_hash,
                "query_hash": observation.query_hash,
                "duration_ms": observation.duration_ms,
                "item_count": observation.item_count,
                "request_id": observation.request_id,
                "error_class": observation.error_class,
            },
        )


def build_runtime(context: RuntimeContext) -> SibylRuntime:
    return SibylRuntime(context)


__all__ = ["SibylRuntime", "build_runtime"]
