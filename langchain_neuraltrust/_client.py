"""Sync and async httpx client for TrustGuard ``POST /v1/evaluate``."""

from __future__ import annotations

import asyncio
import ssl
import threading
from collections.abc import Mapping
from typing import Any, cast

import httpx

from langchain_neuraltrust._types import (
    AUTH_FAILED,
    AUTH_HTTP_STATUSES,
    ENTITLEMENTS,
    EVALUATE_PATH,
    KNOWN_STATUSES,
    REQUEST_FAILED,
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


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_tls_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        name = type(current).__name__.lower()
        text = str(current).lower()
        if "ssl" in name or "ssl" in text or "certificate verify" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _map_status_error(exc: httpx.HTTPStatusError) -> TrustGuardError:
    status_code = exc.response.status_code
    if status_code in AUTH_HTTP_STATUSES:
        return TrustGuardAuthError(AUTH_FAILED)
    if status_code == 503:
        return TrustGuardEntitlementError(ENTITLEMENTS)
    if status_code in UNREACHABLE_HTTP_STATUSES:
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
    """Parse a 2xx TrustGuard body. Non-JSON 200 is an unknown verdict, not unreachable."""
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


class TrustGuardClient:
    """Thin wrapper around TrustGuard evaluate. Inject ``client`` / ``async_client`` in tests."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        timeout: float,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._owns_client = client is None
        self._owns_async_client = async_client is None
        self._client = client
        self._async_client = async_client
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._sync_lock = threading.Lock()
        self._async_lock = threading.Lock()

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

    def evaluate(self, body: Mapping[str, Any]) -> TrustGuardVerdict:
        """Call evaluate synchronously."""
        try:
            response = self._sync().post(
                self.url,
                json=dict(body),
                headers=self.headers(),
                timeout=self.timeout,
            )
        except httpx.RequestError as exc:
            raise _map_request_error(exc) from exc
        return _interpret_response(response)

    async def aevaluate(self, body: Mapping[str, Any]) -> TrustGuardVerdict:
        """Call evaluate asynchronously."""
        try:
            return await self._apost(body)
        except RuntimeError:
            if not self._owns_async_client:
                raise
            self._drop_async_client()
            try:
                return await self._apost(body)
            except RuntimeError as exc:
                raise TrustGuardRequestError(REQUEST_FAILED) from exc

    async def _apost(self, body: Mapping[str, Any]) -> TrustGuardVerdict:
        try:
            response = await self._async().post(
                self.url,
                json=dict(body),
                headers=self.headers(),
                timeout=self.timeout,
            )
        except httpx.RequestError as exc:
            raise _map_request_error(exc) from exc
        return _interpret_response(response)

    def close(self) -> None:
        """Close the owned sync client."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close the owned async client."""
        if self._owns_async_client and self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

    def _sync(self) -> httpx.Client:
        if not self._owns_client:
            assert self._client is not None
            return self._client
        client = self._client
        if client is None or client.is_closed:
            with self._sync_lock:
                client = self._client
                if client is None or client.is_closed:
                    self._client = httpx.Client(timeout=self.timeout)
        assert self._client is not None
        return self._client

    def _async(self) -> httpx.AsyncClient:
        if not self._owns_async_client:
            assert self._async_client is not None
            return self._async_client
        loop = asyncio.get_running_loop()
        client = self._async_client
        if client is None or client.is_closed or self._async_loop is not loop:
            with self._async_lock:
                client = self._async_client
                if client is None or client.is_closed or self._async_loop is not loop:
                    self._async_client = httpx.AsyncClient(timeout=self.timeout)
                    self._async_loop = loop
        assert self._async_client is not None
        return self._async_client

    def _drop_async_client(self) -> None:
        self._async_client = None
        self._async_loop = None
