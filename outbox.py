"""Durable, process-safe mutation outbox for the Sibyl provider."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

SCHEMA_VERSION = 2
DEFAULT_BUSY_TIMEOUT_MS = 5_000
DEFAULT_LEASE_SECONDS = 30.0
DEFAULT_PAGE_SIZE = 64
MAX_ERROR_LENGTH = 2_000

_SECRET_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "cookie",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "token",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(
        r"(?i)((?:api[-_ ]?key|access[-_ ]?token|authorization|password|secret)\s*"
        r"[:=]\s*)('[^']*'|\"[^\"]*\"|[^\s,;]+)"
    ),
)


class OperationState(StrEnum):
    PENDING = "pending"
    PENDING_RECOVERY = "pending_recovery"
    BLOCKED_AUTH = "blocked_auth"
    DEAD_LETTER = "dead_letter"
    RECONCILE_REVISION = "reconcile_revision"


class RecoveryKind(StrEnum):
    NONE = "none"
    RAW_CAPTURE = "raw_capture"
    CORRECTION = "correction"
    EXPOSURE = "exposure"


class Outcome(StrEnum):
    SUCCEEDED = "succeeded"
    REPLAYED = "replayed"
    OBSOLETE = "obsolete"
    SUPERSEDED = "superseded"


class ResultAction(StrEnum):
    SUCCEED = "succeed"
    RETRY = "retry"
    BLOCK_AUTH = "block_auth"
    DEAD_LETTER = "dead_letter"
    RECOVER = "recover"
    RECONCILE_REVISION = "reconcile_revision"
    OBSOLETE = "obsolete"


class OutboxError(RuntimeError):
    """Base error for durable outbox invariants."""


class OperationConflictError(OutboxError):
    """A stable operation ID was reused for a different mutation."""


class ClaimLostError(OutboxError):
    """A worker attempted to mutate an operation after losing its lease."""


class OutboxClosedError(OutboxError):
    """The outbox was used after shutdown."""


@dataclass(frozen=True, slots=True)
class NewOperation:
    operation_id: str
    kind: str
    endpoint: str
    request: Mapping[str, JsonValue]
    idempotency_key: str
    depends_on_operation_id: str | None = None
    http_method: str = "POST"


@dataclass(frozen=True, slots=True)
class IndexedTurnRecord:
    session_id: str
    local_sequence: int
    operation_id: str
    source_id: str
    turn_hash: str
    revision: int | None
    queued: bool


@dataclass(frozen=True, slots=True)
class ClaimedOperation:
    operation_id: str
    depends_on_operation_id: str | None
    kind: str
    endpoint: str
    request: dict[str, JsonValue]
    idempotency_key: str
    state: OperationState
    created_at: str
    updated_at: str
    attempts: int
    last_status: int | None
    last_error: str | None
    http_method: str
    claim_token: str
    claimed_by: str
    lease_expires_at: str
    recovery_kind: RecoveryKind


@dataclass(frozen=True, slots=True)
class AttemptResult:
    action: ResultAction
    status: int | None = None
    error: str | None = None
    replayed: bool = False

    @classmethod
    def success(cls, *, replayed: bool = False, status: int | None = None) -> AttemptResult:
        return cls(ResultAction.SUCCEED, status=status, replayed=replayed)

    @classmethod
    def failure(
        cls,
        action: ResultAction,
        *,
        status: int | None = None,
        error: str | None = None,
    ) -> AttemptResult:
        if action is ResultAction.SUCCEED:
            raise ValueError("failure action cannot be succeed")
        return cls(action, status=status, error=error)


@dataclass(frozen=True, slots=True)
class OutboxSnapshot:
    total: int
    by_state: dict[str, int]
    oldest_age_seconds: float | None
    active_claims: int
    expired_claims: int
    dependency_blocked: int


@dataclass(frozen=True, slots=True)
class FlushReport:
    claimed: int
    outcomes: dict[str, int]
    remaining: OutboxSnapshot
    duration_seconds: float
    deadline_exhausted: bool


@dataclass(slots=True)
class _FlushAccumulator:
    claimed: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat(timespec="microseconds")


def _canonical_request(request: Mapping[str, JsonValue]) -> str:
    _reject_secrets(request)
    try:
        return json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("outbox request must be finite JSON") from exc


def _reject_secrets(value: object, *, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in _SECRET_KEYS:
                raise ValueError(f"secret field {path}.{key} cannot be stored in the outbox")
            _reject_secrets(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secrets(nested, path=f"{path}[{index}]")


def redact_error(error: object | None) -> str | None:
    if error is None:
        return None
    sanitized = " ".join(str(error).split())
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
    return sanitized[:MAX_ERROR_LENGTH]


def recovery_kind_for_endpoint(endpoint: str) -> RecoveryKind:
    path = endpoint.split("?", 1)[0].rstrip("/")
    if path.endswith("/memory/raw"):
        return RecoveryKind.RAW_CAPTURE
    if path.endswith("/memory/expose"):
        return RecoveryKind.EXPOSURE
    if path.endswith("/corrections"):
        return RecoveryKind.CORRECTION
    return RecoveryKind.NONE


def classify_http_failure(
    *,
    status: int | None,
    error_code: str | None = None,
    is_correction: bool = False,
    dependency_pending: bool = False,
) -> ResultAction:
    if status is None or status == 429 or status >= 500:
        return ResultAction.RETRY
    if status in {401, 403}:
        return ResultAction.BLOCK_AUTH
    if status == 404 and is_correction:
        return ResultAction.RETRY if dependency_pending else ResultAction.OBSOLETE
    if status == 409:
        if error_code == "idempotency_lock_held":
            return ResultAction.RETRY
        if error_code == "idempotency_interrupted_pending":
            return ResultAction.RECOVER
        if error_code == "idempotency_payload_mismatch":
            return ResultAction.DEAD_LETTER
        if error_code == "revision_conflict":
            return ResultAction.RECONCILE_REVISION
        return ResultAction.DEAD_LETTER
    return ResultAction.DEAD_LETTER


class DurableOutbox:
    """SQLite outbox whose leases permit concurrent workers and processes."""

    def __init__(
        self,
        hermes_home: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> None:
        if busy_timeout_ms < 1:
            raise ValueError("busy_timeout_ms must be positive")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.path = Path(hermes_home).expanduser() / "state" / "sibyl-outbox.sqlite3"
        self._busy_timeout_ms = busy_timeout_ms
        self._lease_seconds = lease_seconds
        self._closed = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise OutboxClosedError("outbox is closed")
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current_version > SCHEMA_VERSION:
                raise OutboxError(
                    f"outbox schema {current_version} is newer than supported {SCHEMA_VERSION}"
                )
            if current_version == 0:
                self._create_schema(connection)
            elif current_version == 1:
                self._migrate_session_index(connection)
            elif current_version != SCHEMA_VERSION:
                raise OutboxError(f"no migration from outbox schema {current_version}")
        finally:
            connection.close()
        self._secure_files()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE operations (
                operation_id TEXT PRIMARY KEY,
                depends_on_operation_id TEXT,
                kind TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                request_json TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                http_method TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_status INTEGER,
                last_error TEXT,
                claim_token TEXT,
                claimed_by TEXT,
                lease_expires_at TEXT,
                CHECK (attempts >= 0),
                CHECK (state IN (
                    'pending', 'pending_recovery', 'blocked_auth',
                    'dead_letter', 'reconcile_revision'
                ))
            );
            CREATE INDEX operations_replay_order
                ON operations(state, created_at, operation_id);
            CREATE INDEX operations_dependency
                ON operations(depends_on_operation_id);
            CREATE INDEX operations_lease
                ON operations(lease_expires_at) WHERE claim_token IS NOT NULL;

            CREATE TABLE operation_history (
                operation_id TEXT PRIMARY KEY,
                outcome TEXT NOT NULL,
                kind TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                http_method TEXT NOT NULL,
                depends_on_operation_id TEXT,
                attempts INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                replacement_operation_id TEXT,
                last_status INTEGER,
                last_error TEXT,
                CHECK (outcome IN ('succeeded', 'replayed', 'obsolete', 'superseded'))
            );
            CREATE INDEX operation_history_outcome
                ON operation_history(outcome, completed_at);

            CREATE TABLE session_state (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL DEFAULT '',
                next_sequence INTEGER NOT NULL DEFAULT 1,
                reconcile_required INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                CHECK (next_sequence >= 1),
                CHECK (reconcile_required IN (0, 1))
            );
            CREATE TABLE session_turns (
                session_id TEXT NOT NULL,
                local_sequence INTEGER NOT NULL,
                operation_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL UNIQUE,
                turn_hash TEXT NOT NULL,
                revision INTEGER,
                PRIMARY KEY (session_id, local_sequence),
                FOREIGN KEY (session_id) REFERENCES session_state(session_id),
                CHECK (local_sequence >= 1),
                CHECK (revision IS NULL OR revision >= 1)
            );
            CREATE INDEX session_turns_hash ON session_turns(session_id, turn_hash);
            PRAGMA user_version = 2;
            COMMIT;
            """
        )

    def _migrate_session_index(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE session_state (
                session_id TEXT PRIMARY KEY,
                parent_session_id TEXT NOT NULL DEFAULT '',
                next_sequence INTEGER NOT NULL DEFAULT 1,
                reconcile_required INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                CHECK (next_sequence >= 1),
                CHECK (reconcile_required IN (0, 1))
            );
            CREATE TABLE session_turns (
                session_id TEXT NOT NULL,
                local_sequence INTEGER NOT NULL,
                operation_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL UNIQUE,
                turn_hash TEXT NOT NULL,
                revision INTEGER,
                PRIMARY KEY (session_id, local_sequence),
                FOREIGN KEY (session_id) REFERENCES session_state(session_id),
                CHECK (local_sequence >= 1),
                CHECK (revision IS NULL OR revision >= 1)
            );
            CREATE INDEX session_turns_hash ON session_turns(session_id, turn_hash);
            PRAGMA user_version = 2;
            COMMIT;
            """
        )

    def _secure_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self.path}{suffix}")
            if path.exists():
                os.chmod(path, 0o600)

    def enqueue(self, operation: NewOperation) -> bool:
        self._validate_operation(operation)
        request_json = _canonical_request(operation.request)
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        now = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT operation_id, kind, endpoint, request_hash, idempotency_key,
                       http_method, depends_on_operation_id
                FROM operations WHERE operation_id = ?
                UNION ALL
                SELECT operation_id, kind, endpoint, request_hash, idempotency_key,
                       http_method, depends_on_operation_id
                FROM operation_history WHERE operation_id = ?
                LIMIT 1
                """,
                (operation.operation_id, operation.operation_id),
            ).fetchone()
            if existing is not None:
                self._assert_same_operation(existing, operation, request_hash)
                connection.commit()
                return False
            connection.execute(
                """
                INSERT INTO operations (
                    operation_id, depends_on_operation_id, kind, endpoint,
                    request_json, request_hash, idempotency_key, http_method,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    operation.operation_id,
                    operation.depends_on_operation_id,
                    operation.kind,
                    operation.endpoint,
                    request_json,
                    request_hash,
                    operation.idempotency_key,
                    operation.http_method.upper(),
                    now,
                    now,
                ),
            )
            connection.commit()
        self._secure_files()
        return True

    def reserve_sequence(self, session_id: str, *, parent_session_id: str = "") -> int:
        if not session_id:
            raise ValueError("session_id is required")
        now = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO session_state (
                    session_id, parent_session_id, next_sequence,
                    reconcile_required, updated_at
                ) VALUES (?, ?, 1, 1, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    parent_session_id = CASE
                        WHEN excluded.parent_session_id != ''
                        THEN excluded.parent_session_id
                        ELSE session_state.parent_session_id
                    END,
                    updated_at = excluded.updated_at
                """,
                (session_id, parent_session_id, now),
            )
            sequence = int(
                connection.execute(
                    "SELECT next_sequence FROM session_state WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )
            connection.execute(
                "UPDATE session_state SET next_sequence = ?, updated_at = ? WHERE session_id = ?",
                (sequence + 1, now, session_id),
            )
            connection.commit()
        return sequence

    def index_turn(
        self,
        *,
        session_id: str,
        local_sequence: int,
        operation_id: str,
        source_id: str,
        turn_hash: str,
    ) -> bool:
        if not all((session_id, operation_id, source_id, turn_hash)) or local_sequence < 1:
            raise ValueError("indexed turn fields are required")
        now = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO session_state (
                    session_id, next_sequence, reconcile_required, updated_at
                ) VALUES (?, ?, 1, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    next_sequence = MAX(session_state.next_sequence, excluded.next_sequence),
                    updated_at = excluded.updated_at
                """,
                (session_id, local_sequence + 1, now),
            )
            existing = connection.execute(
                """
                SELECT operation_id, source_id, turn_hash
                FROM session_turns
                WHERE session_id = ? AND local_sequence = ?
                """,
                (session_id, local_sequence),
            ).fetchone()
            if existing is not None:
                actual = (existing["operation_id"], existing["source_id"], existing["turn_hash"])
                expected = (operation_id, source_id, turn_hash)
                if actual != expected:
                    connection.rollback()
                    raise OperationConflictError(
                        f"session {session_id} sequence {local_sequence} already identifies another turn"
                    )
                connection.commit()
                return False
            connection.execute(
                """
                INSERT INTO session_turns (
                    session_id, local_sequence, operation_id, source_id, turn_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, local_sequence, operation_id, source_id, turn_hash),
            )
            connection.commit()
        return True

    def mark_session_reconcile(
        self,
        session_id: str,
        *,
        parent_session_id: str = "",
        required: bool = True,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_state (
                    session_id, parent_session_id, reconcile_required, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    parent_session_id = CASE
                        WHEN excluded.parent_session_id != ''
                        THEN excluded.parent_session_id
                        ELSE session_state.parent_session_id
                    END,
                    reconcile_required = excluded.reconcile_required,
                    updated_at = excluded.updated_at
                """,
                (session_id, parent_session_id, int(required), _timestamp()),
            )

    def session_reconcile_required(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT reconcile_required FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return bool(row and row["reconcile_required"])

    def indexed_turns(self, session_id: str) -> list[IndexedTurnRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT turn.*, EXISTS (
                    SELECT 1 FROM operations AS operation
                    WHERE operation.operation_id = turn.operation_id
                ) AS queued
                FROM session_turns AS turn
                WHERE turn.session_id = ?
                ORDER BY turn.local_sequence
                """,
                (session_id,),
            ).fetchall()
        return [
            IndexedTurnRecord(
                session_id=row["session_id"],
                local_sequence=int(row["local_sequence"]),
                operation_id=row["operation_id"],
                source_id=row["source_id"],
                turn_hash=row["turn_hash"],
                revision=int(row["revision"]) if row["revision"] is not None else None,
                queued=bool(row["queued"]),
            )
            for row in rows
        ]

    def record_turn_revision(self, operation_id: str, revision: int) -> None:
        if revision < 1:
            raise ValueError("revision must be positive")
        with self._connect() as connection:
            connection.execute(
                "UPDATE session_turns SET revision = ? WHERE operation_id = ?",
                (revision, operation_id),
            )

    def _validate_operation(self, operation: NewOperation) -> None:
        if not operation.operation_id or not operation.idempotency_key:
            raise ValueError("operation_id and idempotency_key are required")
        if not operation.kind:
            raise ValueError("kind is required")
        if (
            not operation.endpoint.startswith("/")
            or "://" in operation.endpoint
            or "?" in operation.endpoint
            or "#" in operation.endpoint
        ):
            raise ValueError("endpoint must be an absolute path, not a URL")
        if operation.http_method.upper() not in {"POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("http_method must be a mutation method")
        if operation.depends_on_operation_id == operation.operation_id:
            raise ValueError("an operation cannot depend on itself")

    def _assert_same_operation(
        self,
        existing: sqlite3.Row,
        operation: NewOperation,
        request_hash: str,
    ) -> None:
        expected = (
            operation.kind,
            operation.endpoint,
            request_hash,
            operation.idempotency_key,
            operation.http_method.upper(),
            operation.depends_on_operation_id,
        )
        actual = (
            existing["kind"],
            existing["endpoint"],
            existing["request_hash"],
            existing["idempotency_key"],
            existing["http_method"],
            existing["depends_on_operation_id"],
        )
        if actual != expected:
            raise OperationConflictError(
                f"operation {operation.operation_id} already exists with a different mutation"
            )

    def claim_page(
        self,
        worker_id: str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        lease_seconds: float | None = None,
        ready_before: str | None = None,
    ) -> list[ClaimedOperation]:
        if not worker_id:
            raise ValueError("worker_id is required")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        lease_duration = self._lease_seconds if lease_seconds is None else lease_seconds
        if lease_duration <= 0:
            raise ValueError("lease_seconds must be positive")

        now_value = _utc_now()
        now = _timestamp(now_value)
        lease_expires_at = _timestamp(now_value + timedelta(seconds=lease_duration))
        claims: list[ClaimedOperation] = []
        ready_cutoff = ready_before or _timestamp(now_value + timedelta(days=365_000))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM operations AS candidate
                WHERE candidate.state IN ('pending', 'pending_recovery', 'reconcile_revision')
                  AND candidate.updated_at < ?
                  AND (candidate.claim_token IS NULL OR candidate.lease_expires_at <= ?)
                  AND (
                    candidate.depends_on_operation_id IS NULL
                    OR EXISTS (
                        SELECT 1 FROM operation_history AS dependency
                        WHERE dependency.operation_id = candidate.depends_on_operation_id
                          AND dependency.outcome IN ('succeeded', 'replayed')
                    )
                  )
                ORDER BY candidate.created_at, candidate.operation_id
                LIMIT ?
                """,
                (ready_cutoff, now, page_size),
            ).fetchall()
            for row in rows:
                claim_token = uuid.uuid4().hex
                changed = connection.execute(
                    """
                    UPDATE operations
                    SET claim_token = ?, claimed_by = ?, lease_expires_at = ?,
                        attempts = attempts + 1, updated_at = ?
                    WHERE operation_id = ?
                      AND (claim_token IS NULL OR lease_expires_at <= ?)
                    """,
                    (claim_token, worker_id, lease_expires_at, now, row["operation_id"], now),
                ).rowcount
                if changed:
                    claims.append(
                        self._claimed_from_row(
                            row,
                            claim_token=claim_token,
                            worker_id=worker_id,
                            lease_expires_at=lease_expires_at,
                        )
                    )
            connection.commit()
        self._secure_files()
        return claims

    def _claimed_from_row(
        self,
        row: sqlite3.Row,
        *,
        claim_token: str,
        worker_id: str,
        lease_expires_at: str,
    ) -> ClaimedOperation:
        request = json.loads(row["request_json"])
        if not isinstance(request, dict):
            raise OutboxError(f"operation {row['operation_id']} has a non-object request")
        return ClaimedOperation(
            operation_id=row["operation_id"],
            depends_on_operation_id=row["depends_on_operation_id"],
            kind=row["kind"],
            endpoint=row["endpoint"],
            request=request,
            idempotency_key=row["idempotency_key"],
            state=OperationState(row["state"]),
            created_at=row["created_at"],
            updated_at=_timestamp(),
            attempts=int(row["attempts"]) + 1,
            last_status=row["last_status"],
            last_error=row["last_error"],
            http_method=row["http_method"],
            claim_token=claim_token,
            claimed_by=worker_id,
            lease_expires_at=lease_expires_at,
            recovery_kind=recovery_kind_for_endpoint(row["endpoint"]),
        )

    def renew_claim(
        self, operation: ClaimedOperation, *, lease_seconds: float | None = None
    ) -> str:
        lease_duration = self._lease_seconds if lease_seconds is None else lease_seconds
        if lease_duration <= 0:
            raise ValueError("lease_seconds must be positive")
        expires_at = _timestamp(_utc_now() + timedelta(seconds=lease_duration))
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE operations SET lease_expires_at = ?, updated_at = ?
                WHERE operation_id = ? AND claim_token = ?
                """,
                (expires_at, _timestamp(), operation.operation_id, operation.claim_token),
            ).rowcount
        if not changed:
            raise ClaimLostError(f"claim lost for operation {operation.operation_id}")
        return expires_at

    def apply_result(self, operation: ClaimedOperation, result: AttemptResult) -> None:
        if result.action is ResultAction.SUCCEED:
            outcome = Outcome.REPLAYED if result.replayed else Outcome.SUCCEEDED
            self._archive_and_delete(operation, outcome=outcome, result=result)
            return
        if result.action is ResultAction.OBSOLETE:
            self._archive_and_delete(operation, outcome=Outcome.OBSOLETE, result=result)
            return

        state_by_action = {
            ResultAction.RETRY: OperationState.PENDING,
            ResultAction.BLOCK_AUTH: OperationState.BLOCKED_AUTH,
            ResultAction.DEAD_LETTER: OperationState.DEAD_LETTER,
            ResultAction.RECOVER: OperationState.PENDING_RECOVERY,
            ResultAction.RECONCILE_REVISION: OperationState.RECONCILE_REVISION,
        }
        state = state_by_action[result.action]
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE operations
                SET state = ?, updated_at = ?, last_status = ?, last_error = ?,
                    claim_token = NULL, claimed_by = NULL, lease_expires_at = NULL
                WHERE operation_id = ? AND claim_token = ?
                """,
                (
                    state.value,
                    _timestamp(),
                    result.status,
                    redact_error(result.error),
                    operation.operation_id,
                    operation.claim_token,
                ),
            ).rowcount
        if not changed:
            raise ClaimLostError(f"claim lost for operation {operation.operation_id}")

    def _archive_and_delete(
        self,
        operation: ClaimedOperation,
        *,
        outcome: Outcome,
        result: AttemptResult,
        replacement_operation_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ? AND claim_token = ?",
                (operation.operation_id, operation.claim_token),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ClaimLostError(f"claim lost for operation {operation.operation_id}")
            connection.execute(
                """
                INSERT INTO operation_history (
                    operation_id, outcome, kind, endpoint, idempotency_key,
                    request_hash, http_method, depends_on_operation_id, attempts,
                    completed_at, replacement_operation_id, last_status, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["operation_id"],
                    outcome.value,
                    row["kind"],
                    row["endpoint"],
                    row["idempotency_key"],
                    row["request_hash"],
                    row["http_method"],
                    row["depends_on_operation_id"],
                    row["attempts"],
                    _timestamp(),
                    replacement_operation_id,
                    result.status,
                    redact_error(result.error),
                ),
            )
            connection.execute(
                "DELETE FROM operations WHERE operation_id = ?", (operation.operation_id,)
            )
            connection.commit()

    def supersede(self, operation: ClaimedOperation, replacement: NewOperation) -> None:
        self._validate_operation(replacement)
        if replacement.operation_id == operation.operation_id:
            raise ValueError("replacement must have a new logical operation ID")
        request_json = _canonical_request(replacement.request)
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        now = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ? AND claim_token = ?",
                (operation.operation_id, operation.claim_token),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ClaimLostError(f"claim lost for operation {operation.operation_id}")
            collision = connection.execute(
                """
                SELECT operation_id FROM operations WHERE operation_id = ?
                UNION ALL
                SELECT operation_id FROM operation_history WHERE operation_id = ?
                LIMIT 1
                """,
                (replacement.operation_id, replacement.operation_id),
            ).fetchone()
            if collision is not None:
                connection.rollback()
                raise OperationConflictError(
                    f"replacement {replacement.operation_id} already exists"
                )
            connection.execute(
                """
                INSERT INTO operations (
                    operation_id, depends_on_operation_id, kind, endpoint,
                    request_json, request_hash, idempotency_key, http_method,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    replacement.operation_id,
                    replacement.depends_on_operation_id,
                    replacement.kind,
                    replacement.endpoint,
                    request_json,
                    request_hash,
                    replacement.idempotency_key,
                    replacement.http_method.upper(),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO operation_history (
                    operation_id, outcome, kind, endpoint, idempotency_key,
                    request_hash, http_method, depends_on_operation_id, attempts,
                    completed_at, replacement_operation_id, last_status, last_error
                ) VALUES (?, 'superseded', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["operation_id"],
                    row["kind"],
                    row["endpoint"],
                    row["idempotency_key"],
                    row["request_hash"],
                    row["http_method"],
                    row["depends_on_operation_id"],
                    row["attempts"],
                    now,
                    replacement.operation_id,
                    row["last_status"],
                    row["last_error"],
                ),
            )
            connection.execute(
                "DELETE FROM operations WHERE operation_id = ?", (operation.operation_id,)
            )
            connection.commit()

    def unblock_auth(self) -> int:
        with self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE operations
                SET state = 'pending', updated_at = ?, last_error = NULL
                WHERE state = 'blocked_auth'
                """,
                (_timestamp(),),
            ).rowcount
        return changed

    def snapshot(self) -> OutboxSnapshot:
        now = _timestamp()
        with self._connect() as connection:
            state_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM operations GROUP BY state"
            ).fetchall()
            oldest = connection.execute("SELECT MIN(created_at) FROM operations").fetchone()[0]
            active_claims = int(
                connection.execute(
                    "SELECT COUNT(*) FROM operations WHERE claim_token IS NOT NULL AND lease_expires_at > ?",
                    (now,),
                ).fetchone()[0]
            )
            expired_claims = int(
                connection.execute(
                    "SELECT COUNT(*) FROM operations WHERE claim_token IS NOT NULL AND lease_expires_at <= ?",
                    (now,),
                ).fetchone()[0]
            )
            dependency_blocked = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM operations AS candidate
                    WHERE candidate.depends_on_operation_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM operation_history AS dependency
                          WHERE dependency.operation_id = candidate.depends_on_operation_id
                            AND dependency.outcome IN ('succeeded', 'replayed')
                      )
                    """
                ).fetchone()[0]
            )
        by_state = {str(row["state"]): int(row["count"]) for row in state_rows}
        total = sum(by_state.values())
        oldest_age = None
        if oldest:
            oldest_age = max(0.0, (_utc_now() - datetime.fromisoformat(oldest)).total_seconds())
        return OutboxSnapshot(
            total=total,
            by_state=by_state,
            oldest_age_seconds=oldest_age,
            active_claims=active_claims,
            expired_claims=expired_claims,
            dependency_blocked=dependency_blocked,
        )

    def history_outcome(self, operation_id: str) -> Outcome | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT outcome FROM operation_history WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return Outcome(row["outcome"]) if row else None

    def flush(
        self,
        worker_id: str,
        executor: Callable[[ClaimedOperation], AttemptResult],
        *,
        timeout_seconds: float | None = None,
        page_size: int = 1,
    ) -> FlushReport:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        started = time.monotonic()
        deadline = None if timeout_seconds is None else started + timeout_seconds
        ready_before = _timestamp()
        accumulator = _FlushAccumulator()
        deadline_exhausted = False

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                deadline_exhausted = True
                break
            claims = self.claim_page(
                worker_id,
                page_size=page_size,
                ready_before=ready_before,
            )
            if not claims:
                break
            for index, operation in enumerate(claims):
                if deadline is not None and time.monotonic() >= deadline:
                    deadline_exhausted = True
                    self._release_unattempted(claims[index:])
                    break
                accumulator.claimed += 1
                try:
                    result = executor(operation)
                except Exception as exc:
                    result = AttemptResult.failure(ResultAction.RETRY, error=str(exc))
                self.apply_result(operation, result)
                accumulator.outcomes[result.action.value] += 1
            if deadline_exhausted:
                break

        return FlushReport(
            claimed=accumulator.claimed,
            outcomes=dict(accumulator.outcomes),
            remaining=self.snapshot(),
            duration_seconds=time.monotonic() - started,
            deadline_exhausted=deadline_exhausted,
        )

    def _release_unattempted(self, operations: list[ClaimedOperation]) -> None:
        if not operations:
            return
        now = _timestamp()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for operation in operations:
                connection.execute(
                    """
                    UPDATE operations
                    SET attempts = attempts - 1, updated_at = ?, claim_token = NULL,
                        claimed_by = NULL, lease_expires_at = NULL
                    WHERE operation_id = ? AND claim_token = ?
                    """,
                    (now, operation.operation_id, operation.claim_token),
                )
            connection.commit()

    def close(self) -> None:
        self._closed = True

    def shutdown(
        self,
        worker_id: str,
        executor: Callable[[ClaimedOperation], AttemptResult],
        *,
        timeout_seconds: float = 4.0,
        page_size: int = 1,
    ) -> FlushReport:
        report = self.flush(
            worker_id,
            executor,
            timeout_seconds=timeout_seconds,
            page_size=page_size,
        )
        self.close()
        return report


__all__ = [
    "AttemptResult",
    "ClaimLostError",
    "ClaimedOperation",
    "DurableOutbox",
    "FlushReport",
    "IndexedTurnRecord",
    "NewOperation",
    "OperationConflictError",
    "OperationState",
    "OutboxClosedError",
    "OutboxSnapshot",
    "Outcome",
    "RecoveryKind",
    "ResultAction",
    "classify_http_failure",
    "recovery_kind_for_endpoint",
    "redact_error",
]
