"""Synchronous, retry-free HTTP client for Sibyl's provider contract."""

from __future__ import annotations

import json
import re
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar, cast
from urllib.parse import quote

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

from .schemas import (
    AuthMeResponse,
    ContextExposureRequest,
    ContextExposureResponse,
    ContextPackRequest,
    ContextPackResponse,
    CorrectionRequest,
    CorrectionResponse,
    JSONObject,
    MutationReceipt,
    RawMemoryRequest,
    RawMemoryResponse,
    SchemaError,
)

BACKGROUND_CONTEXT_TIMEOUT_SECONDS = 5.0
MANUAL_CONTEXT_TIMEOUT_SECONDS = 10.0
MUTATION_TIMEOUT_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 10.0
MAX_ERROR_BODY_BYTES = 16 * 1024
_MAX_ERROR_PREVIEW_CHARS = 512
_REQUEST_ID_HEADER = "X-Request-ID"
_IDEMPOTENCY_HEADER = "Idempotency-Key"
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ResponseT = TypeVar("_ResponseT")


class IdempotencyErrorKind(StrEnum):
    LOCK_HELD = "idempotency_lock_held"
    PAYLOAD_MISMATCH = "idempotency_payload_mismatch"
    INTERRUPTED_PENDING = "idempotency_interrupted_pending"
    RECEIPT_PENDING = "idempotency_receipt_pending"


class SibylClientError(RuntimeError):
    """Base class for errors safe to surface outside the HTTP boundary."""


class SibylTransportError(SibylClientError):
    def __init__(self, *, request_id: str, error_type: str) -> None:
        self.request_id = request_id
        self.error_type = error_type
        super().__init__(f"Sibyl transport failed ({error_type}, request {request_id})")


class SibylProtocolError(SibylClientError):
    def __init__(
        self,
        message: str,
        *,
        request_id: str,
        status_code: int | None = None,
        body_preview: str = "",
    ) -> None:
        self.request_id = request_id
        self.status_code = status_code
        self.body_preview = body_preview
        super().__init__(f"{message} (request {request_id})")


class SibylHTTPError(SibylClientError):
    def __init__(
        self,
        *,
        status_code: int,
        error: str,
        message: str,
        request_id: str,
        remediation: str | None,
        details: JSONObject,
    ) -> None:
        self.status_code = status_code
        self.error = error
        self.server_message = message
        self.request_id = request_id
        self.remediation = remediation
        self.details = details
        try:
            self.idempotency_kind = IdempotencyErrorKind(error)
        except ValueError:
            self.idempotency_kind = None
        self.revision_conflict = error == "revision_conflict"
        super().__init__(f"Sibyl HTTP {status_code}: {error} (request {request_id})")


class SibylConflictError(SibylHTTPError):
    """A structured HTTP 409, including idempotency and revision conflicts."""


def _request_id() -> str:
    return f"req_hermes_{uuid.uuid4().hex}"


def _timeout(seconds: float) -> httpx.Timeout:
    connection_budget = min(seconds, 5.0)
    return httpx.Timeout(
        seconds,
        connect=connection_budget,
        pool=connection_budget,
    )


def _bounded_error_body(response: httpx.Response) -> tuple[bytes, bool]:
    body = bytearray()
    for chunk in response.iter_bytes():
        remaining = MAX_ERROR_BODY_BYTES - len(body)
        if remaining <= 0:
            return bytes(body), True
        body.extend(chunk[:remaining])
        if len(chunk) > remaining:
            return bytes(body), True
    return bytes(body), False


def _safe_preview(body: bytes) -> str:
    decoded = body.decode("utf-8", errors="replace")
    cleaned = _CONTROL_CHARACTERS.sub("?", decoded).replace("\r", " ").replace("\n", " ")
    if len(cleaned) > _MAX_ERROR_PREVIEW_CHARS:
        return f"{cleaned[:_MAX_ERROR_PREVIEW_CHARS]}…"
    return cleaned


def _json_object(body: bytes) -> JSONObject:
    value: object = json.loads(body)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("JSON body must be an object")
    return cast("JSONObject", value)


def _optional_error_string(payload: JSONObject, key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"error.{key} must be a string or null")
    return value


class SibylClient:
    """Typed provider client with fixed paths, budgets, and trust boundary."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        verify_tls: bool = True,
        transport: httpx.BaseTransport | None = None,
        request_id_factory: Callable[[], str] = _request_id,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        normalized_url = base_url.rstrip("/") + "/"
        self._request_id_factory = request_id_factory
        selected_transport = transport or httpx.HTTPTransport(retries=0, verify=verify_tls)
        self._client = httpx.Client(
            base_url=normalized_url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "hermes-sibyl-memory/0.1.0",
            },
            follow_redirects=False,
            transport=selected_transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SibylClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def auth_me(self) -> AuthMeResponse:
        return self._request(
            "GET",
            "auth/me",
            parser=AuthMeResponse.from_json,
            timeout_seconds=PROBE_TIMEOUT_SECONDS,
        )

    def context_pack(
        self,
        request: ContextPackRequest,
        *,
        manual: bool = False,
    ) -> ContextPackResponse:
        return self._request(
            "POST",
            "context/pack",
            payload=request.to_json(),
            parser=ContextPackResponse.from_json,
            timeout_seconds=(
                MANUAL_CONTEXT_TIMEOUT_SECONDS if manual else BACKGROUND_CONTEXT_TIMEOUT_SECONDS
            ),
        )

    def expose_context(
        self,
        request: ContextExposureRequest,
        *,
        idempotency_key: str,
    ) -> ContextExposureResponse:
        response = self._request(
            "POST",
            "memory/expose",
            payload=request.to_json(),
            parser=ContextExposureResponse.from_json,
            timeout_seconds=MUTATION_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
            receipt=lambda value: value.mutation_receipt,
        )
        return response

    def remember_raw(
        self,
        request: RawMemoryRequest,
        *,
        idempotency_key: str,
    ) -> RawMemoryResponse:
        response = self._request(
            "POST",
            "memory/raw",
            payload=request.to_json(),
            parser=RawMemoryResponse.from_json,
            timeout_seconds=MUTATION_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
            receipt=lambda value: value.mutation_receipt,
        )
        return response

    def preview_correction(
        self,
        source_id: str,
        request: CorrectionRequest,
    ) -> CorrectionResponse:
        return self._request(
            "POST",
            self._correction_path(source_id, preview=True),
            payload=request.to_json(),
            parser=CorrectionResponse.from_json,
            timeout_seconds=MUTATION_TIMEOUT_SECONDS,
        )

    def apply_correction(
        self,
        source_id: str,
        request: CorrectionRequest,
        *,
        idempotency_key: str,
    ) -> CorrectionResponse:
        response = self._request(
            "POST",
            self._correction_path(source_id, preview=False),
            payload=request.to_json(),
            parser=CorrectionResponse.from_json,
            timeout_seconds=MUTATION_TIMEOUT_SECONDS,
            idempotency_key=idempotency_key,
            receipt=lambda value: value.mutation_receipt,
        )
        return response

    @staticmethod
    def _correction_path(source_id: str, *, preview: bool) -> str:
        if not source_id:
            raise ValueError("source_id is required")
        suffix = "/preview" if preview else ""
        return f"memory/inspect/{quote(source_id, safe='')}/corrections{suffix}"

    @staticmethod
    def _require_matching_receipt(
        receipt: MutationReceipt | None,
        idempotency_key: str,
    ) -> None:
        if receipt is None:
            raise SchemaError("response.mutation_receipt is required")
        if receipt.idempotency_key != idempotency_key:
            raise SchemaError("response.mutation_receipt.idempotency_key does not match request")

    def _request(
        self,
        method: str,
        path: str,
        *,
        parser: Callable[[object], _ResponseT],
        timeout_seconds: float,
        payload: JSONObject | None = None,
        idempotency_key: str | None = None,
        receipt: Callable[[_ResponseT], MutationReceipt | None] | None = None,
    ) -> _ResponseT:
        request_id = self._request_id_factory()
        headers = {_REQUEST_ID_HEADER: request_id}
        if idempotency_key is not None:
            normalized_key = idempotency_key.strip()
            if not normalized_key or len(normalized_key) > 255:
                raise ValueError("idempotency_key must contain 1-255 non-whitespace characters")
            headers[_IDEMPOTENCY_HEADER] = normalized_key
        try:
            with self._client.stream(
                method,
                path,
                json=payload,
                headers=headers,
                timeout=_timeout(timeout_seconds),
            ) as response:
                if response.is_error:
                    self._raise_http_error(response, request_id=request_id)
                body = response.read()
        except SibylClientError:
            raise
        except httpx.HTTPError as exc:
            raise SibylTransportError(
                request_id=request_id,
                error_type=type(exc).__name__,
            ) from exc

        try:
            response_payload = _json_object(body)
            parsed = parser(response_payload)
            if receipt is not None:
                if idempotency_key is None:
                    raise SchemaError("receipt validation requires an idempotency key")
                self._require_matching_receipt(receipt(parsed), idempotency_key)
            return parsed
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, SchemaError) as exc:
            raise SibylProtocolError(
                "Sibyl returned a malformed JSON success response",
                request_id=request_id,
                status_code=response.status_code,
            ) from exc

    @staticmethod
    def _raise_http_error(response: httpx.Response, *, request_id: str) -> None:
        body, truncated = _bounded_error_body(response)
        response_request_id = response.headers.get(_REQUEST_ID_HEADER, request_id)
        preview = _safe_preview(body)
        if truncated:
            raise SibylProtocolError(
                "Sibyl error response exceeded the safe body limit",
                request_id=response_request_id,
                status_code=response.status_code,
                body_preview=preview,
            )
        try:
            payload = _json_object(body)
            error = payload.get("error")
            message = payload.get("message")
            if not isinstance(error, str) or not isinstance(message, str):
                raise ValueError("error response requires string error and message fields")
            error_request_id = _optional_error_string(payload, "request_id")
            remediation = _optional_error_string(payload, "remediation")
            details_value = payload.get("details", {})
            if not isinstance(details_value, dict) or any(
                not isinstance(key, str) for key in details_value
            ):
                raise ValueError("error.details must be an object")
            details = cast("JSONObject", details_value)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise SibylProtocolError(
                "Sibyl returned a non-contract JSON error response",
                request_id=response_request_id,
                status_code=response.status_code,
                body_preview=preview,
            ) from exc

        error_type = SibylConflictError if response.status_code == 409 else SibylHTTPError
        raise error_type(
            status_code=response.status_code,
            error=error,
            message=message,
            request_id=error_request_id or response_request_id,
            remediation=remediation,
            details=details,
        )
