from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

ENV_KEYS = (
    "TRUSTGUARD_API_KEY",
    "TRUSTGUARD_API_BASE",
    "TRUSTGUARD_COLLECTOR_KEY",
    "TRUSTGUARD_SESSION_ID",
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
