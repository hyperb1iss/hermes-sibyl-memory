from __future__ import annotations

import pytest

from capture import turn_hash
from sessions import IndexedTurn, TurnReconciliationError, TurnStartQueue, plan_rewind_corrections


def test_completed_turn_consumes_matching_fifo_entry():
    queue = TurnStartQueue()
    queue.record("session-1", 4, "first")
    queue.record("session-1", 5, "second")

    assert queue.consume("session-1", "first").turn_number == 4
    assert queue.consume("session-1", "second").turn_number == 5


def test_interrupted_older_turn_is_discarded_when_newer_completed_turn_matches():
    queue = TurnStartQueue()
    queue.record("session-1", 4, "first")
    queue.record("session-1", 5, "second")

    assert queue.consume("session-1", "second").turn_number == 5
    with pytest.raises(TurnReconciliationError, match="no turn-start"):
        queue.consume("session-1", "first")


def test_unknown_mismatch_is_explicit_and_retains_queue():
    queue = TurnStartQueue()
    queue.record("session-1", 4, "first")

    with pytest.raises(TurnReconciliationError, match="any queued"):
        queue.consume("session-1", "unknown")

    assert queue.consume("session-1", "first").turn_number == 4


def test_sessions_are_independent():
    queue = TurnStartQueue()
    queue.record("one", 1, "one")
    queue.record("two", 7, "two")

    assert queue.consume("two", "two").turn_number == 7
    assert queue.consume("one", "one").turn_number == 1


def test_rewind_marks_only_absent_turns_stale():
    first_hash = turn_hash("one", "answer one")
    second_hash = turn_hash("two", "answer two")
    indexed = [
        IndexedTurn("operation-1", "source-1", first_hash, 1, revision=1),
        IndexedTurn("operation-2", "source-2", second_hash, 2, revision=3),
    ]

    corrections = plan_rewind_corrections(indexed, [first_hash])

    assert len(corrections) == 1
    assert corrections[0].source_id == "source-2"
    assert corrections[0].expected_revision == 3
    assert corrections[0].depends_on_operation_id is None


def test_queued_source_correction_waits_for_source_write():
    indexed = [
        IndexedTurn(
            "operation-1",
            "source-1",
            turn_hash("one", "answer"),
            1,
            queued=True,
        )
    ]

    correction = plan_rewind_corrections(indexed, [])[0]

    assert correction.expected_revision is None
    assert correction.depends_on_operation_id == "operation-1"


def test_identical_turn_counts_preserve_legitimate_repeats():
    repeated = turn_hash("same", "same")
    indexed = [
        IndexedTurn("operation-1", "source-1", repeated, 1),
        IndexedTurn("operation-2", "source-2", repeated, 2),
    ]

    corrections = plan_rewind_corrections(indexed, [repeated])

    assert [item.source_id for item in corrections] == ["source-2"]
