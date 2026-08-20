"""Shared types, constants, and errors for the TrustGuard client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

DEFAULT_API_BASE: Final = "https://trustguard.neuraltrust.ai"
EVALUATE_PATH: Final = "/v1/evaluate"
DEFAULT_TIMEOUT: Final = 5.0
DEFAULT_MAX_RETRIES: Final = 2

STATUS_ALLOW: Final = "allow"
STATUS_BLOCK: Final = "block"
STATUS_TRANSFORM: Final = "transform"
STATUS_REPORT: Final = "report"

KNOWN_STATUSES: Final = frozenset(
    {STATUS_ALLOW, STATUS_BLOCK, STATUS_TRANSFORM, STATUS_REPORT}
)
UNREACHABLE_HTTP_STATUSES: Final = frozenset({502, 504})
RETRYABLE_HTTP_STATUSES: Final = frozenset({429, 502, 504})
AUTH_HTTP_STATUSES: Final = frozenset({401, 403})

TrustGuardStatus = Literal["allow", "block", "transform", "report"]
ExitBehavior = Literal["end", "error", "replace"]
UnreachableFallback = Literal["fail_closed", "fail_open"]
EvaluateDirection = Literal["input", "output"]
HookStage = Literal["input", "output", "tool"]
ViolationStage = Literal["input", "output", "tool", "tool_call"]
OutcomeKind = Literal[
    "allow", "report", "block", "transform", "fail_closed", "fail_open", "reraise"
]

TRANSFORM_MISSING: Final = "TrustGuard transform missing payload"
UNKNOWN_VERDICT: Final = "TrustGuard returned an unknown verdict"
UNREACHABLE: Final = "TrustGuard guardrail service unreachable"
AUTH_FAILED: Final = "TrustGuard authentication failed"
ENTITLEMENTS: Final = "TrustGuard entitlements unavailable"
REQUEST_FAILED: Final = "TrustGuard request failed"
MISSING_API_KEY: Final = (
    "TrustGuard API key is required. Set TRUSTGUARD_API_KEY or pass api_key."
)
BLOCKED: Final = "Blocked by NeuralTrust TrustGuard."


@dataclass(frozen=True, slots=True)
class TrustGuardVerdict:
    """Parsed TrustGuard evaluate response."""

    status: TrustGuardStatus
    trace_id: str | None = None
    request_id: str | None = None
    findings: object | None = None
    transformed_payload: Mapping[str, object] | None = None
    raw: Mapping[str, object] | None = None


class TrustGuardError(Exception):
    """Base error for TrustGuard failures."""


class TrustGuardBlockedError(TrustGuardError):
    """Raised when TrustGuard blocks and exit_behavior is set to error."""

    def __init__(
        self,
        message: str,
        *,
        verdict: TrustGuardVerdict | None = None,
        stage: ViolationStage = "input",
    ) -> None:
        super().__init__(message)
        self.verdict = verdict
        self.stage = stage


class TrustGuardUnreachableError(TrustGuardError):
    """Transport failure eligible for ``unreachable_fallback``.

    Covers connect errors, timeouts, HTTP 502/504, and exhausted HTTP 429 retries.
    """


class TrustGuardAuthError(TrustGuardError):
    """HTTP 401/403. Always fail closed."""


class TrustGuardEntitlementError(TrustGuardError):
    """HTTP 503 entitlements. Always fail closed."""


class TrustGuardRequestError(TrustGuardError):
    """Other HTTP or unexpected request failures. Always fail closed."""


class TrustGuardUnknownVerdictError(TrustGuardError):
    """Unknown or unparseable verdict, including non-JSON HTTP 200. Always fail closed."""


class TrustGuardTransformError(TrustGuardError):
    """Unusable ``transformed_payload``. Always fail closed."""

    def __init__(self, reason: str) -> None:
        super().__init__(TRANSFORM_MISSING)
        self.reason = reason
