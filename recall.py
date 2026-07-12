"""Current-query automatic recall with a strict hot-path budget."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol


def query_digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RecallResult:
    markdown: str
    rendered_item_ids: tuple[str, ...]
    total_items: int
    request_id: str = ""
    rendered_token_estimate: int | None = None


@dataclass(frozen=True, slots=True)
class RecallObservation:
    event: str
    session_hash: str
    query_hash: str
    duration_ms: float
    item_count: int = 0
    request_id: str = ""
    error_class: str = ""


FetchContext = Callable[[str, str], RecallResult]
RecordDelivery = Callable[[str, str, RecallResult], None]
ObserveRecall = Callable[[RecallObservation], None]


class Submitter(Protocol):
    def submit(self, function: Callable[..., RecallResult], *args: str) -> Future[RecallResult]: ...


class RecallCoordinator:
    def __init__(
        self,
        fetch_context: FetchContext,
        record_delivery: RecordDelivery,
        *,
        hot_wait_seconds: float = 0.25,
        executor: Submitter | None = None,
        observe: ObserveRecall | None = None,
    ) -> None:
        self._fetch_context = fetch_context
        self._record_delivery = record_delivery
        self._hot_wait_seconds = hot_wait_seconds
        self._executor = executor or ThreadPoolExecutor(thread_name_prefix="sibyl-recall")
        self._owns_executor = executor is None
        self._observe = observe or (lambda observation: None)
        self._futures: dict[tuple[str, str], tuple[float, Future[RecallResult]]] = {}
        self._lock = Lock()

    def schedule(self, query: str, *, session_id: str) -> None:
        if not query:
            return
        key = (session_id, query_digest(query))
        with self._lock:
            previous = self._futures.pop(key, None)
            if previous is not None:
                previous[1].cancel()
            self._futures[key] = (
                time.monotonic(),
                self._executor.submit(self._fetch_context, query, session_id),
            )

    def prefetch(self, query: str, *, session_id: str) -> str:
        query_hash = query_digest(query)
        key = (session_id, query_hash)
        with self._lock:
            pending = self._futures.pop(key, None)
        if pending is None:
            return ""

        started_at, future = pending
        try:
            result = future.result(timeout=self._hot_wait_seconds)
        except TimeoutError:
            self._observe(
                self._observation(
                    "sibyl_context_hot_wait_missed",
                    session_id,
                    query_hash,
                    started_at,
                )
            )
            future.add_done_callback(
                lambda completed: self._observe_late_result(
                    completed,
                    session_id=session_id,
                    query_hash=query_hash,
                    started_at=started_at,
                )
            )
            return ""
        except Exception as exc:
            self._observe(
                self._observation(
                    "sibyl_context_failed",
                    session_id,
                    query_hash,
                    started_at,
                    error_class=type(exc).__name__,
                )
            )
            return ""

        self._observe(
            self._observation(
                "sibyl_context_ready",
                session_id,
                query_hash,
                started_at,
                item_count=result.total_items,
                request_id=result.request_id,
            )
        )
        if not result.markdown.strip():
            return ""
        self._record_delivery(session_id, query_hash, result)
        return result.markdown

    def discard_session(self, session_id: str) -> None:
        with self._lock:
            keys = [key for key in self._futures if key[0] == session_id]
            futures = [self._futures.pop(key)[1] for key in keys]
        for future in futures:
            future.cancel()

    def shutdown(self) -> None:
        if self._owns_executor and isinstance(self._executor, ThreadPoolExecutor):
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _observe_late_result(
        self,
        future: Future[RecallResult],
        *,
        session_id: str,
        query_hash: str,
        started_at: float,
    ) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._observe(
                self._observation(
                    "sibyl_context_failed",
                    session_id,
                    query_hash,
                    started_at,
                    error_class=type(exc).__name__,
                )
            )
            return
        self._observe(
            self._observation(
                "sibyl_context_ready_late",
                session_id,
                query_hash,
                started_at,
                item_count=result.total_items,
                request_id=result.request_id,
            )
        )

    @staticmethod
    def _observation(
        event: str,
        session_id: str,
        query_hash: str,
        started_at: float,
        **values: Any,
    ) -> RecallObservation:
        return RecallObservation(
            event=event,
            session_hash=hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16],
            query_hash=query_hash,
            duration_ms=(time.monotonic() - started_at) * 1000,
            **values,
        )


__all__ = [
    "RecallCoordinator",
    "RecallObservation",
    "RecallResult",
    "query_digest",
]
