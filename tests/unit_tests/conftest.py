from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

ENV_KEYS = (
    "TRUSTGUARD_API_KEY",
    "TRUSTGUARD_API_BASE",
    "TRUSTGUARD_COLLECTOR_KEY",
    "TRUSTGUARD_SESSION_ID",
    "TRUSTGUARD_MODEL_NAME",
    "TRUSTGUARD_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _clear_trustguard_env() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_neuraltrust._client.time.sleep", lambda _s: None)

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("langchain_neuraltrust._client.asyncio.sleep", _no_sleep)
