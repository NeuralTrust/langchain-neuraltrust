"""Sync and async httpx client for TrustGuard ``POST /v1/evaluate``."""

from __future__ import annotations

import asyncio
import ssl
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import httpx

from langchain_neuraltrust._types import (
    AUTH_FAILED,
    AUTH_HTTP_STATUSES,
    DEFAULT_MAX_RETRIES,
    ENTITLEMENTS,
    EVALUATE_PATH,
    KNOWN_STATUSES,
    REQUEST_FAILED,
    RETRYABLE_HTTP_STATUSES,
    UNKNOWN_VERDICT,
    UNREACHABLE,
    UNREACHABLE_HTTP_STATUSES,
    TrustGuardAuthError,
    TrustGuardEntitlementError,
    TrustGuardError,
    TrustGuardRequestError,
    TrustGuardStatus,
    TrustGuardUnknownVerdictError,
    TrustGuardUnreachableError,
    TrustGuardVerdict,
)
from langchain_neuraltrust._version import __version__

_USER_AGENT = f"langchain-neuraltrust/{__version__}"
_UNREACHABLE_EXC = (httpx.TimeoutException, httpx.ConnectError)
Send = Callable[[], httpx.Response]
ASend = Callable[[], Awaitable[httpx.Response]]


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_tls_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _map_status_error(exc: httpx.HTTPStatusError) -> TrustGuardError:
    status_code = exc.response.status_code
    if status_code in AUTH_HTTP_STATUSES:
        return TrustGuardAuthError(AUTH_FAILED)
    if status_code == 503:
        return TrustGuardEntitlementError(ENTITLEMENTS)
    if status_code in UNREACHABLE_HTTP_STATUSES or status_code == 429:
        return TrustGuardUnreachableError(UNREACHABLE)
    return TrustGuardRequestError(REQUEST_FAILED)


def _map_request_error(exc: httpx.RequestError) -> TrustGuardError:
    """Connect/timeout are unreachable; TLS and everything else fail closed."""
    if isinstance(exc, httpx.ConnectError) and _is_tls_failure(exc):
        return TrustGuardRequestError(REQUEST_FAILED)
    if isinstance(exc, _UNREACHABLE_EXC):
        return TrustGuardUnreachableError(UNREACHABLE)
    if isinstance(exc, httpx.DecodingError):
        return TrustGuardUnknownVerdictError(UNKNOWN_VERDICT)
    return TrustGuardRequestError(REQUEST_FAILED)


def parse_evaluate_response(response: httpx.Response) -> TrustGuardVerdict:
    """Parse a 2xx TrustGuard body.

    Non-JSON 200 is an unknown verdict, not unreachable.
    """
    try:
        parsed: object = response.json()
    except ValueError as exc:
        raise TrustGuardUnknownVerdictError(UNKNOWN_VERDICT) from exc
    if not isinstance(parsed, dict):
        raise TrustGuardUnknownVerdictError(UNKNOWN_VERDICT)
    status = parsed.get("status")
    if not isinstance(status, str) or status.lower() not in KNOWN_STATUSES:
        raise TrustGuardUnknownVerdictError(UNKNOWN_VERDICT)
    transformed = parsed.get("transformed_payload")
    return TrustGuardVerdict(
        status=cast(TrustGuardStatus, status.lower()),
        trace_id=_as_optional_str(parsed.get("trace_id")),
        request_id=_as_optional_str(parsed.get("request_id")),
        findings=parsed.get("findings"),
        transformed_payload=transformed if isinstance(transformed, Mapping) else None,
        raw=parsed,
    )


def _interpret_response(response: httpx.Response) -> TrustGuardVerdict:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _map_status_error(exc) from exc
    return parse_evaluate_response(response)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    if isinstance(raw, str):
        try:
            return min(float(raw), 5.0)
        except ValueError:
            pass
    backoff = 0.25 * (2**attempt)
    capped = 2.0 if backoff > 2.0 else backoff
    return float(capped)


def _close_async_client(client: httpx.AsyncClient) -> None:
    if client.is_closed:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(client.aclose())
        except RuntimeError:
            return
        return
    loop.create_task(client.aclose())


class TrustGuardClient:
    """Thin wrapper around TrustGuard evaluate.

    Inject ``client`` / ``async_client`` in tests. Owned async clients are keyed
    per event loop so concurrent loops do not thrash a single slot.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        timeout: float,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._injected_sync = client
        self._injected_async = async_client
        self._owned_sync: httpx.Client | None = None
        self._owned_async: dict[int, httpx.AsyncClient] = {}
        self._sync_lock = threading.Lock()

    @property
    def url(self) -> str:
        """Absolute evaluate URL."""
        return f"{self.api_base}{EVALUATE_PATH}"

    def headers(self) -> dict[str, str]:
        """Auth and content headers for evaluate."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }

    def _post_kwargs(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "url": self.url,
            "json": dict(body),
            "headers": self.headers(),
            "timeout": self.timeout,
        }

    def evaluate(self, body: Mapping[str, Any]) -> TrustGuardVerdict:
        """Call evaluate synchronously, retrying 429/502/504 and timeouts."""

        def send() -> httpx.Response:
            return self._sync_http().post(**self._post_kwargs(body))

        return _interpret_response(self._retry_sync(send))

    async def aevaluate(self, body: Mapping[str, Any]) -> TrustGuardVerdict:
        """Call evaluate asynchronously, retrying 429/502/504 and timeouts."""

        async def send() -> httpx.Response:
            return await self._async_http().post(**self._post_kwargs(body))

        return _interpret_response(await self._retry_async(send))

    def close(self) -> None:
        """Close owned sync and async clients when possible."""
        if self._injected_sync is None and self._owned_sync is not None:
            self._owned_sync.close()
            self._owned_sync = None
        if self._injected_async is not None:
            return
        pending = list(self._owned_async.values())
        self._owned_async = {}
        for client in pending:
            _close_async_client(client)

    async def aclose(self) -> None:
        """Close owned async clients."""
        if self._injected_async is not None:
            return
        pending = list(self._owned_async.values())
        self._owned_async = {}
        for client in pending:
            if not client.is_closed:
                await client.aclose()

    def _sync_http(self) -> httpx.Client:
        if self._injected_sync is not None:
            return self._injected_sync
        client = self._owned_sync
        if client is None or client.is_closed:
            with self._sync_lock:
                client = self._owned_sync
                if client is None or client.is_closed:
                    client = httpx.Client(timeout=self.timeout)
                    self._owned_sync = client
        return client

    def _async_http(self) -> httpx.AsyncClient:
        if self._injected_async is not None:
            return self._injected_async
        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._owned_async.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout)
            self._owned_async[key] = client
        return client

    def _retry_sync(self, send: Send) -> httpx.Response:
        last_error: TrustGuardError | None = None
        last_response: httpx.Response | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = send()
            except httpx.RequestError as exc:
                mapped = _map_request_error(exc)
                last_error = mapped
                if (
                    isinstance(mapped, TrustGuardUnreachableError)
                    and attempt < self.max_retries
                ):
                    time.sleep(_retry_delay(httpx.Response(0), attempt))
                    continue
                raise mapped from exc
            last_response = response
            if (
                response.status_code in RETRYABLE_HTTP_STATUSES
                and attempt < self.max_retries
            ):
                time.sleep(_retry_delay(response, attempt))
                continue
            return response
        if last_response is not None:
            return last_response
        raise last_error or TrustGuardUnreachableError(UNREACHABLE)

    async def _retry_async(self, send: ASend) -> httpx.Response:
        last_error: TrustGuardError | None = None
        last_response: httpx.Response | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await send()
            except httpx.RequestError as exc:
                mapped = _map_request_error(exc)
                last_error = mapped
                if (
                    isinstance(mapped, TrustGuardUnreachableError)
                    and attempt < self.max_retries
                ):
                    await asyncio.sleep(_retry_delay(httpx.Response(0), attempt))
                    continue
                raise mapped from exc
            last_response = response
            if (
                response.status_code in RETRYABLE_HTTP_STATUSES
                and attempt < self.max_retries
            ):
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            return response
        if last_response is not None:
            return last_response
        raise last_error or TrustGuardUnreachableError(UNREACHABLE)
