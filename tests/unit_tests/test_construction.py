from __future__ import annotations

import os

import pytest

from langchain_neuraltrust import TrustGuardMiddleware
from langchain_neuraltrust._types import DEFAULT_API_BASE, MISSING_API_KEY


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValueError, match="API key is required"):
        TrustGuardMiddleware()


def test_env_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTGUARD_API_KEY", "tgk_env")
    monkeypatch.setenv("TRUSTGUARD_API_BASE", "https://example.test/")
    monkeypatch.setenv("TRUSTGUARD_COLLECTOR_KEY", "tgcol_env")
    monkeypatch.setenv("TRUSTGUARD_SESSION_ID", "sess-env")
    middleware = TrustGuardMiddleware()
    assert middleware.api_key == "tgk_env"
    assert middleware.api_base == "https://example.test"
    assert middleware.collector_key == "tgcol_env"
    assert middleware.session_id == "sess-env"


def test_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTGUARD_API_KEY", "tgk_env")
    monkeypatch.setenv("TRUSTGUARD_API_BASE", "https://env.test")
    middleware = TrustGuardMiddleware(
        api_key="tgk_arg",
        api_base="https://arg.test/",
        collector_key="tgcol_arg",
    )
    assert middleware.api_key == "tgk_arg"
    assert middleware.api_base == "https://arg.test"
    assert middleware.collector_key == "tgcol_arg"


def test_env_timeout_and_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTGUARD_API_KEY", "tgk_env")
    monkeypatch.setenv("TRUSTGUARD_TIMEOUT", "8.5")
    monkeypatch.setenv("TRUSTGUARD_MODEL_NAME", "from-env")
    middleware = TrustGuardMiddleware()
    assert middleware.timeout == 8.5
    assert middleware.model_name == "from-env"


def test_default_api_base_and_timeout() -> None:
    middleware = TrustGuardMiddleware(api_key="tgk_test")
    assert middleware.api_base == DEFAULT_API_BASE
    assert middleware.timeout == 5.0
    assert middleware.unreachable_fallback == "fail_closed"
    assert middleware.exit_behavior == "end"


def test_invalid_exit_behavior() -> None:
    with pytest.raises(ValueError, match="exit_behavior"):
        TrustGuardMiddleware(api_key="tgk_test", exit_behavior="explode")  # type: ignore[arg-type]


def test_invalid_unreachable_fallback() -> None:
    with pytest.raises(ValueError, match="unreachable_fallback"):
        TrustGuardMiddleware(api_key="tgk_test", unreachable_fallback="shrug")  # type: ignore[arg-type]


def test_missing_key_message() -> None:
    with pytest.raises(ValueError, match=MISSING_API_KEY.split(".")[0]):
        TrustGuardMiddleware()
    assert "TRUSTGUARD_API_KEY" not in os.environ


def test_http_api_base_rejected() -> None:
    with pytest.raises(ValueError, match="https"):
        TrustGuardMiddleware(api_key="tgk_test", api_base="http://trustguard.example")


def test_localhost_http_api_base_allowed() -> None:
    middleware = TrustGuardMiddleware(
        api_key="tgk_test", api_base="http://localhost:8080/"
    )
    assert middleware.api_base == "http://localhost:8080"


def test_api_base_rejects_query_string() -> None:
    with pytest.raises(ValueError, match="https"):
        TrustGuardMiddleware(api_key="tgk_test", api_base="https://x.example?a=b")


def test_api_base_rejects_missing_hostname() -> None:
    with pytest.raises(ValueError, match="https"):
        TrustGuardMiddleware(api_key="tgk_test", api_base="https:/x.example")


def test_payload_tools_convert_langchain_tool() -> None:
    from langchain.tools import tool

    @tool
    def search(q: str) -> str:
        """Search."""
        return q

    middleware = TrustGuardMiddleware(api_key="tgk_test", payload_tools=[search])
    assert middleware.payload_tools is not None
    assert middleware.payload_tools[0]["function"]["name"] == "search"


def test_payload_tools_reject_arbitrary_objects() -> None:
    with pytest.raises(ValueError, match="payload_tools"):
        TrustGuardMiddleware(api_key="tgk_test", payload_tools=[object()])
