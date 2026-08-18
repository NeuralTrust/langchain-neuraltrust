"""Creds-gated live suite against langchain-demo-* collectors.

Skipped unless ``tests/integration_tests/.creds.json`` exists. That file is gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain.messages import HumanMessage

from langchain_neuraltrust import TrustGuardMiddleware
from langchain_neuraltrust._types import BLOCKED

CREDS_PATH = Path(__file__).resolve().parent / ".creds.json"

pytestmark = pytest.mark.live


def _load_creds() -> dict[str, Any]:
    if not CREDS_PATH.is_file():
        pytest.skip("tests/integration_tests/.creds.json is not present")
    return json.loads(CREDS_PATH.read_text())


def _mw(entry: dict[str, Any], **kwargs: Any) -> TrustGuardMiddleware:
    return TrustGuardMiddleware(
        api_key=entry["api_key"],
        collector_key=entry.get("collector_key"),
        api_base=entry.get("api_base"),
        check_output=False,
        timeout=15.0,
        **kwargs,
    )


def _entry(creds: dict[str, Any], name: str) -> dict[str, Any]:
    collectors = creds.get("collectors") or creds
    if name not in collectors:
        pytest.skip(f"collector {name} missing from .creds.json")
    return collectors[name]


def test_live_allow() -> None:
    creds = _load_creds()
    result = _mw(_entry(creds, "allow")).before_model(
        {"messages": [HumanMessage(content="What is the capital of France?")]},
        runtime=None,  # type: ignore[arg-type]
    )
    assert result is None


def test_live_block() -> None:
    creds = _load_creds()
    result = _mw(_entry(creds, "block")).before_model(
        {"messages": [HumanMessage(content="this prompt is forbidden")]},
        runtime=None,  # type: ignore[arg-type]
    )
    assert result is not None
    assert result["jump_to"] == "end"
    assert BLOCKED in result["messages"][0].content


def test_live_report() -> None:
    creds = _load_creds()
    seen: list[str] = []
    human = HumanMessage(content="this prompt is forbidden")
    mw = _mw(_entry(creds, "report"), on_violation=lambda v, _s: seen.append(v.status))
    result = mw.before_model(
        {"messages": [human]},
        runtime=None,  # type: ignore[arg-type]
    )
    if result is None:
        pytest.fail("expected report metadata on the human message")
    assert "jump_to" not in result
    assert result["messages"][0].additional_kwargs["trustguard"]["status"] == "report"
    assert seen == ["report"]


def test_live_transform() -> None:
    creds = _load_creds()
    human = HumanMessage(content="My SSN is 123-45-6789", id="live-hm")
    result = _mw(_entry(creds, "transform")).before_model(
        {"messages": [human]},
        runtime=None,  # type: ignore[arg-type]
    )
    assert result is not None
    updated = result["messages"][0]
    assert updated.id == "live-hm"
    assert "123-45-6789" not in str(updated.content)
    assert isinstance(updated, HumanMessage)


def test_live_skips_without_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    missing = Path("/tmp/missing-creds.json")
    monkeypatch.setattr(sys.modules[__name__], "CREDS_PATH", missing)
    with pytest.raises(pytest.skip.Exception):
        _load_creds()
