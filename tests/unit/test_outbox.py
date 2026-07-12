from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import pytest

from outbox import (
    SCHEMA_VERSION,
    AttemptResult,
    ClaimLostError,
    DurableOutbox,
    NewOperation,
    OperationConflictError,
    OperationState,
    Outcome,
    RecoveryKind,
    ResultAction,
    classify_http_failure,
    redact_error,
)

if TYPE_CHECKING:
    from pathlib import Path


def operation(
    operation_id: str,
    *,
    endpoint: str = "/api/memory/raw",
    depends_on: str | None = None,
    value: str = "memory",
) -> NewOperation:
    return NewOperation(
        operation_id=operation_id,
        kind="raw_capture",
        endpoint=endpoint,
        request={"raw_content": value, "metadata": {"provider_operation_id": operation_id}},
        idempotency_key=f"hermes-{operation_id}",
        depends_on_operation_id=depends_on,
    )


def test_initializes_secure_wal_database_under_hermes_home(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)

    assert outbox.path == tmp_path / "state" / "sibyl-outbox.sqlite3"
    assert outbox.path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(outbox.path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_enqueue_is_idempotent_but_rejects_operation_identity_reuse(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    original = operation("same")

    assert outbox.enqueue(original) is True
    assert outbox.enqueue(original) is False

    with pytest.raises(OperationConflictError):
        outbox.enqueue(operation("same", value="different"))


def test_outbox_refuses_secrets_in_persisted_payload(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    unsafe = NewOperation(
        operation_id="unsafe",
        kind="raw_capture",
        endpoint="/api/memory/raw",
        request={"metadata": {"api_key": "sibyl-secret"}},
        idempotency_key="hermes-unsafe",
    )

    with pytest.raises(ValueError, match="secret field"):
        outbox.enqueue(unsafe)

    assert outbox.snapshot().total == 0


def test_concurrent_instances_claim_each_operation_once(tmp_path: Path) -> None:
    producer = DurableOutbox(tmp_path)
    for index in range(24):
        producer.enqueue(operation(f"op-{index:02d}"))

    barrier = threading.Barrier(3)
    claimed: list[list[str]] = [[], []]

    def claim(worker_index: int) -> None:
        outbox = DurableOutbox(tmp_path)
        barrier.wait()
        claimed[worker_index] = [
            item.operation_id for item in outbox.claim_page(f"worker-{worker_index}", page_size=24)
        ]

    threads = [threading.Thread(target=claim, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    combined = claimed[0] + claimed[1]
    assert len(combined) == 24
    assert len(set(combined)) == 24


def test_expired_lease_can_be_reclaimed_and_old_owner_is_fenced(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path, lease_seconds=0.01)
    outbox.enqueue(operation("leased"))
    stale = outbox.claim_page("stale-worker")[0]
    time.sleep(0.02)
    replacement = outbox.claim_page("replacement-worker")[0]

    assert replacement.claim_token != stale.claim_token
    with pytest.raises(ClaimLostError):
        outbox.apply_result(stale, AttemptResult.success())

    outbox.apply_result(replacement, AttemptResult.success())
    assert outbox.history_outcome("leased") is Outcome.SUCCEEDED


def test_success_tombstone_unblocks_dependent_rewind_correction(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("source"))
    outbox.enqueue(
        operation(
            "rewind",
            endpoint="/api/memory/inspect/source/corrections",
            depends_on="source",
        )
    )

    source = outbox.claim_page("worker")[0]
    assert source.operation_id == "source"
    assert source.recovery_kind is RecoveryKind.RAW_CAPTURE
    outbox.apply_result(source, AttemptResult.success(replayed=True, status=200))

    correction = outbox.claim_page("worker")[0]
    assert correction.operation_id == "rewind"
    assert correction.recovery_kind is RecoveryKind.CORRECTION
    assert outbox.history_outcome("source") is Outcome.REPLAYED


def test_retryable_failure_is_not_hot_retried_in_same_flush(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("network"))
    attempts: list[str] = []

    report = outbox.flush(
        "worker",
        lambda claimed: (
            attempts.append(claimed.operation_id)
            or AttemptResult.failure(ResultAction.RETRY, error="network timeout")
        ),
    )

    assert attempts == ["network"]
    assert report.claimed == 1
    assert report.outcomes == {"retry": 1}
    assert report.remaining.by_state == {"pending": 1}


@pytest.mark.parametrize(
    ("status", "code", "correction", "dependency_pending", "expected"),
    [
        (None, None, False, False, ResultAction.RETRY),
        (429, None, False, False, ResultAction.RETRY),
        (503, None, False, False, ResultAction.RETRY),
        (401, None, False, False, ResultAction.BLOCK_AUTH),
        (403, None, False, False, ResultAction.BLOCK_AUTH),
        (409, "idempotency_lock_held", False, False, ResultAction.RETRY),
        (409, None, False, False, ResultAction.DEAD_LETTER),
        (409, "idempotency_interrupted_pending", False, False, ResultAction.RECOVER),
        (409, "idempotency_payload_mismatch", False, False, ResultAction.DEAD_LETTER),
        (409, "revision_conflict", True, False, ResultAction.RECONCILE_REVISION),
        (404, None, True, True, ResultAction.RETRY),
        (404, None, True, False, ResultAction.OBSOLETE),
        (422, None, False, False, ResultAction.DEAD_LETTER),
    ],
)
def test_failure_classification(
    status: int | None,
    code: str | None,
    correction: bool,
    dependency_pending: bool,
    expected: ResultAction,
) -> None:
    assert (
        classify_http_failure(
            status=status,
            error_code=code,
            is_correction=correction,
            dependency_pending=dependency_pending,
        )
        is expected
    )


def test_interrupted_pending_and_revision_conflict_retain_explicit_states(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("recovery"))
    recovery = outbox.claim_page("worker")[0]
    outbox.apply_result(
        recovery,
        AttemptResult.failure(
            ResultAction.RECOVER,
            status=409,
            error="idempotency_interrupted_pending",
        ),
    )

    recovered = outbox.claim_page("recovery-worker")[0]
    assert recovered.state is OperationState.PENDING_RECOVERY
    assert recovered.recovery_kind is RecoveryKind.RAW_CAPTURE
    outbox.apply_result(
        recovered,
        AttemptResult.failure(ResultAction.RECONCILE_REVISION, status=409),
    )
    revision = outbox.claim_page("revision-worker")[0]
    assert revision.state is OperationState.RECONCILE_REVISION


def test_auth_failure_stops_hot_retry_until_explicitly_unblocked(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("auth"))
    claimed = outbox.claim_page("worker")[0]
    outbox.apply_result(
        claimed,
        AttemptResult.failure(ResultAction.BLOCK_AUTH, status=401, error="unauthorized"),
    )

    assert outbox.claim_page("other-worker") == []
    assert outbox.snapshot().by_state == {"blocked_auth": 1}
    assert outbox.unblock_auth() == 1
    assert outbox.claim_page("other-worker")[0].operation_id == "auth"


def test_obsolete_correction_becomes_audit_tombstone(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("obsolete", endpoint="/api/memory/inspect/gone/corrections"))
    claimed = outbox.claim_page("worker")[0]

    outbox.apply_result(
        claimed,
        AttemptResult.failure(ResultAction.OBSOLETE, status=404, error="target erased"),
    )

    assert outbox.snapshot().total == 0
    assert outbox.history_outcome("obsolete") is Outcome.OBSOLETE


def test_revision_reconcile_can_atomically_supersede_with_new_intent(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("old", endpoint="/api/memory/inspect/source/corrections"))
    old = outbox.claim_page("worker")[0]

    outbox.supersede(
        old,
        operation(
            "new",
            endpoint="/api/memory/inspect/source/corrections",
            value="revision-2",
        ),
    )

    assert outbox.history_outcome("old") is Outcome.SUPERSEDED
    assert outbox.claim_page("worker")[0].operation_id == "new"


def test_snapshot_reports_backlog_age_claims_and_dependencies(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("source"))
    outbox.enqueue(operation("dependent", depends_on="source"))
    outbox.claim_page("worker", page_size=1)

    snapshot = outbox.snapshot()
    assert snapshot.total == 2
    assert snapshot.oldest_age_seconds is not None
    assert snapshot.oldest_age_seconds >= 0
    assert snapshot.active_claims == 1
    assert snapshot.dependency_blocked == 1


def test_error_redaction_covers_headers_assignments_and_storage(tmp_path: Path) -> None:
    error = "Authorization: Bearer top-secret api_key=another-secret path=/api/memory/raw"
    redacted = redact_error(error)
    assert redacted is not None
    assert "top-secret" not in redacted
    assert "another-secret" not in redacted
    assert "path=/api/memory/raw" in redacted

    outbox = DurableOutbox(tmp_path)
    outbox.enqueue(operation("redacted"))
    claimed = outbox.claim_page("worker")[0]
    outbox.apply_result(
        claimed,
        AttemptResult.failure(ResultAction.DEAD_LETTER, status=422, error=error),
    )
    with sqlite3.connect(outbox.path) as connection:
        stored_error = connection.execute(
            "SELECT last_error FROM operations WHERE operation_id = 'redacted'"
        ).fetchone()[0]
    assert "top-secret" not in stored_error
    assert "another-secret" not in stored_error


def test_shutdown_flush_is_bounded_between_attempts_and_reports_remaining(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    for index in range(3):
        outbox.enqueue(operation(f"shutdown-{index}"))

    def slow_success(_claimed) -> AttemptResult:
        time.sleep(0.015)
        return AttemptResult.success()

    report = outbox.shutdown(
        "shutdown-worker",
        slow_success,
        timeout_seconds=0.001,
        page_size=1,
    )

    assert report.deadline_exhausted is True
    assert report.duration_seconds < 4.0
    assert report.claimed in {0, 1}
    assert report.remaining.total == 3 - report.claimed
    assert os.path.exists(outbox.path)


def test_session_sequences_and_turn_index_survive_reopen(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    assert outbox.reserve_sequence("session-1", parent_session_id="parent-1") == 1
    assert outbox.reserve_sequence("session-1") == 2
    assert outbox.index_turn(
        session_id="session-1",
        local_sequence=1,
        operation_id="operation-1",
        source_id="source-1",
        turn_hash="hash-1",
    )
    outbox.close()

    reopened = DurableOutbox(tmp_path)
    indexed = reopened.indexed_turns("session-1")

    assert indexed[0].operation_id == "operation-1"
    assert indexed[0].local_sequence == 1
    assert reopened.reserve_sequence("session-1") == 3
    assert reopened.session_reconcile_required("session-1") is True


def test_turn_index_reports_queued_state_and_revision(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    queued = operation("operation-1")
    outbox.enqueue(queued)
    outbox.index_turn(
        session_id="session-1",
        local_sequence=1,
        operation_id=queued.operation_id,
        source_id="source-1",
        turn_hash="hash-1",
    )

    assert outbox.indexed_turns("session-1")[0].queued is True
    outbox.record_turn_revision("operation-1", 4)
    claimed = outbox.claim_page("worker-1")[0]
    outbox.apply_result(claimed, AttemptResult.success(status=200))

    indexed = outbox.indexed_turns("session-1")[0]
    assert indexed.queued is False
    assert indexed.revision == 4


def test_session_reconcile_flag_is_durable(tmp_path: Path) -> None:
    outbox = DurableOutbox(tmp_path)
    outbox.mark_session_reconcile("session-1", required=True)
    assert outbox.session_reconcile_required("session-1") is True

    outbox.mark_session_reconcile("session-1", required=False)
    assert outbox.session_reconcile_required("session-1") is False
