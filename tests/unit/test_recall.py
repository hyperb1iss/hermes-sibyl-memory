from __future__ import annotations

import time
from concurrent.futures import Future

from recall import RecallCoordinator, RecallResult


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class PendingExecutor:
    def __init__(self):
        self.future = Future()

    def submit(self, function, *args):
        return self.future


def _result(markdown: str = "# Memory") -> RecallResult:
    return RecallResult(
        markdown=markdown,
        rendered_item_ids=("decision_1",),
        total_items=1,
        request_id="request-1",
    )


def test_current_query_returns_markdown_and_records_exact_delivery():
    deliveries = []
    coordinator = RecallCoordinator(
        lambda query, session: _result(f"{session}:{query}"),
        lambda session, digest, result: deliveries.append((session, digest, result)),
        executor=ImmediateExecutor(),
    )
    coordinator.schedule("current", session_id="session-1")

    markdown = coordinator.prefetch("current", session_id="session-1")

    assert markdown == "session-1:current"
    assert deliveries[0][2].rendered_item_ids == ("decision_1",)


def test_different_query_never_receives_cached_context():
    deliveries = []
    coordinator = RecallCoordinator(
        lambda query, session: _result(),
        lambda *values: deliveries.append(values),
        executor=ImmediateExecutor(),
    )
    coordinator.schedule("old", session_id="session-1")

    assert coordinator.prefetch("new", session_id="session-1") == ""
    assert deliveries == []


def test_slow_result_misses_hot_path_and_is_never_acknowledged():
    executor = PendingExecutor()
    deliveries = []
    observations = []
    coordinator = RecallCoordinator(
        lambda query, session: _result(),
        lambda *values: deliveries.append(values),
        executor=executor,
        hot_wait_seconds=0.001,
        observe=observations.append,
    )
    coordinator.schedule("slow", session_id="session-1")

    assert coordinator.prefetch("slow", session_id="session-1") == ""
    executor.future.set_result(_result())
    time.sleep(0.01)

    assert deliveries == []
    assert observations[0].event == "sibyl_context_hot_wait_missed"
    assert observations[-1].event == "sibyl_context_ready_late"


def test_fetch_failure_fails_open_without_content_or_delivery():
    deliveries = []
    observations = []

    def fail(query, session):
        raise RuntimeError("offline")

    coordinator = RecallCoordinator(
        fail,
        lambda *values: deliveries.append(values),
        executor=ImmediateExecutor(),
        observe=observations.append,
    )
    coordinator.schedule("current", session_id="session-1")

    assert coordinator.prefetch("current", session_id="session-1") == ""
    assert deliveries == []
    assert observations[-1].event == "sibyl_context_failed"
    assert observations[-1].error_class == "RuntimeError"


def test_session_switch_discards_only_that_sessions_future():
    executor = PendingExecutor()
    coordinator = RecallCoordinator(
        lambda query, session: _result(),
        lambda *values: None,
        executor=executor,
    )
    coordinator.schedule("current", session_id="session-1")
    coordinator.discard_session("session-1")

    assert coordinator.prefetch("current", session_id="session-1") == ""
