from __future__ import annotations

import httpx
import pytest
import respx

from langchain_neuraltrust._client import TrustGuardClient
from langchain_neuraltrust._types import (
    AUTH_FAILED,
    ENTITLEMENTS,
    REQUEST_FAILED,
    UNKNOWN_VERDICT,
    UNREACHABLE,
    TrustGuardAuthError,
    TrustGuardEntitlementError,
    TrustGuardRequestError,
    TrustGuardUnknownVerdictError,
    TrustGuardUnreachableError,
)

URL = "https://trustguard.neuraltrust.ai/v1/evaluate"


def _client() -> TrustGuardClient:
    return TrustGuardClient(
        api_key="tgk_test",
        api_base="https://trustguard.neuraltrust.ai/",
        timeout=5.0,
    )


def _body() -> dict[str, object]:
    return {
        "payload": {"messages": [{"role": "user", "content": "hello"}]},
        "direction": "input",
        "protocol": "llm",
        "attributes": {"content_type": "application/json", "model": {"name": ""}},
    }


@respx.mock
def test_evaluate_parses_known_status() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ALLOW",
                "trace_id": "tr-1",
                "request_id": "rq-1",
                "findings": [],
            },
        )
    )
    verdict = _client().evaluate(_body())
    assert verdict.status == "allow"
    assert verdict.trace_id == "tr-1"
    assert verdict.request_id == "rq-1"
    assert verdict.findings == []


@respx.mock
def test_evaluate_sends_bearer_and_json() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"status": "allow"}))
    _client().evaluate(_body())
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tgk_test"
    assert request.headers["Content-Type"] == "application/json"
    assert request.url.path == "/v1/evaluate"


@pytest.mark.asyncio
@respx.mock
async def test_aevaluate_allow() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"status": "report"}))
    verdict = await _client().aevaluate(_body())
    assert verdict.status == "report"


@pytest.mark.parametrize("status_code", [401, 403])
@respx.mock
def test_auth_errors_fail_closed(status_code: int) -> None:
    respx.post(URL).mock(return_value=httpx.Response(status_code, text="nope"))
    with pytest.raises(TrustGuardAuthError, match=AUTH_FAILED):
        _client().evaluate(_body())


@respx.mock
def test_entitlements_503() -> None:
    respx.post(URL).mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(TrustGuardEntitlementError, match=ENTITLEMENTS):
        _client().evaluate(_body())


@pytest.mark.parametrize("status_code", [502, 504])
@respx.mock
def test_bad_gateway_is_unreachable(status_code: int) -> None:
    respx.post(URL).mock(return_value=httpx.Response(status_code, text="gw"))
    with pytest.raises(TrustGuardUnreachableError, match=UNREACHABLE):
        _client().evaluate(_body())


@respx.mock
def test_429_is_request_failed() -> None:
    respx.post(URL).mock(return_value=httpx.Response(429, text="slow"))
    with pytest.raises(TrustGuardRequestError, match=REQUEST_FAILED):
        _client().evaluate(_body())


@respx.mock
def test_timeout_is_unreachable() -> None:
    respx.post(URL).mock(side_effect=httpx.TimeoutException("timeout"))
    with pytest.raises(TrustGuardUnreachableError, match=UNREACHABLE):
        _client().evaluate(_body())


@respx.mock
def test_connect_error_is_unreachable() -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(TrustGuardUnreachableError, match=UNREACHABLE):
        _client().evaluate(_body())


@respx.mock
def test_non_json_200_is_unknown_verdict() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, text="not-json"))
    with pytest.raises(TrustGuardUnknownVerdictError, match=UNKNOWN_VERDICT):
        _client().evaluate(_body())


@respx.mock
def test_json_array_200_is_unknown_verdict() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=["allow"]))
    with pytest.raises(TrustGuardUnknownVerdictError, match=UNKNOWN_VERDICT):
        _client().evaluate(_body())


@respx.mock
def test_unknown_status_is_unknown_verdict() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"status": "maybe"}))
    with pytest.raises(TrustGuardUnknownVerdictError, match=UNKNOWN_VERDICT):
        _client().evaluate(_body())


@respx.mock
def test_missing_status_is_unknown_verdict() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    with pytest.raises(TrustGuardUnknownVerdictError, match=UNKNOWN_VERDICT):
        _client().evaluate(_body())


@respx.mock
def test_remote_protocol_error_is_request_failed() -> None:
    respx.post(URL).mock(side_effect=httpx.RemoteProtocolError("oops"))
    with pytest.raises(TrustGuardRequestError, match=REQUEST_FAILED):
        _client().evaluate(_body())


@respx.mock
def test_decoding_error_is_unknown_verdict() -> None:
    respx.post(URL).mock(side_effect=httpx.DecodingError("bad gzip"))
    with pytest.raises(TrustGuardUnknownVerdictError, match=UNKNOWN_VERDICT):
        _client().evaluate(_body())


@respx.mock
def test_user_agent_includes_package_version() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"status": "allow"}))
    _client().evaluate(_body())
    from langchain_neuraltrust._version import __version__

    assert route.calls.last.request.headers["User-Agent"] == f"langchain-neuraltrust/{__version__}"


def test_tls_error_is_request_failed_not_unreachable() -> None:
    import ssl

    from langchain_neuraltrust._client import _map_request_error

    failure = httpx.ConnectError("certificate verify failed")
    failure.__cause__ = ssl.SSLError("certificate verify failed")
    mapped = _map_request_error(failure)
    assert isinstance(mapped, TrustGuardRequestError)
    assert str(mapped) == REQUEST_FAILED


def test_async_client_is_recreated_for_a_new_event_loop() -> None:
    import asyncio

    client = _client()

    async def grab() -> httpx.AsyncClient:
        return client._async()

    first = asyncio.run(grab())
    second = asyncio.run(grab())
    assert first is not second
