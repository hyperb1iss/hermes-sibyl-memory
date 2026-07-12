"""Session-local turn matching and rewind reconciliation plans."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from threading import Lock

try:
    from .capture import message_hash
except ImportError:
    from capture import message_hash


class TurnReconciliationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PendingTurn:
    turn_number: int
    user_message_hash: str


class TurnStartQueue:
    def __init__(self) -> None:
        self._turns: dict[str, deque[PendingTurn]] = defaultdict(deque)
        self._lock = Lock()

    def record(self, session_id: str, turn_number: int, user_content: str) -> PendingTurn:
        pending = PendingTurn(
            turn_number=turn_number,
            user_message_hash=message_hash(user_content),
        )
        with self._lock:
            self._turns[session_id].append(pending)
        return pending

    def consume(self, session_id: str, user_content: str) -> PendingTurn:
        actual_hash = message_hash(user_content)
        with self._lock:
            queue = self._turns.get(session_id)
            if not queue:
                raise TurnReconciliationError(f"no turn-start record for session {session_id!r}")
            pending = queue[0]
            if pending.user_message_hash != actual_hash:
                raise TurnReconciliationError(
                    "completed turn does not match the oldest turn-start record: "
                    f"session={session_id!r} turn={pending.turn_number}"
                )
            queue.popleft()
            if not queue:
                self._turns.pop(session_id, None)
            return pending

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)


@dataclass(frozen=True, slots=True)
class IndexedTurn:
    operation_id: str
    source_id: str
    turn_hash: str
    local_sequence: int
    revision: int | None = None
    queued: bool = False


@dataclass(frozen=True, slots=True)
class StaleCorrectionIntent:
    operation_id: str
    source_id: str
    expected_revision: int | None
    depends_on_operation_id: str | None


def plan_rewind_corrections(
    indexed_turns: list[IndexedTurn],
    committed_fingerprints: list[str],
) -> list[StaleCorrectionIntent]:
    remaining = Counter(committed_fingerprints)
    corrections: list[StaleCorrectionIntent] = []
    for turn in sorted(indexed_turns, key=lambda item: item.local_sequence):
        if remaining[turn.turn_hash] > 0:
            remaining[turn.turn_hash] -= 1
            continue
        corrections.append(
            StaleCorrectionIntent(
                operation_id=f"rewind-stale-{turn.operation_id}",
                source_id=turn.source_id,
                expected_revision=turn.revision,
                depends_on_operation_id=turn.operation_id if turn.queued else None,
            )
        )
    return corrections


__all__ = [
    "IndexedTurn",
    "PendingTurn",
    "StaleCorrectionIntent",
    "TurnReconciliationError",
    "TurnStartQueue",
    "plan_rewind_corrections",
]
