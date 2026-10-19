"""Single verdict and recovery classifier for hooks and tool gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from langgraph.errors import GraphBubbleUp

from langchain_neuraltrust._types import (
    BLOCKED,
    REQUEST_FAILED,
    STATUS_ALLOW,
    STATUS_BLOCK,
    STATUS_REPORT,
    STATUS_SKIP,
    STATUS_TRANSFORM,
    UNKNOWN_VERDICT,
    UNREACHABLE,
    OutcomeKind,
    TrustGuardBlockedError,
    TrustGuardError,
    TrustGuardUnreachableError,
    TrustGuardVerdict,
    UnreachableFallback,
)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Closed set of actions the adapters may take."""

    kind: OutcomeKind
    text: str = ""
    verdict: TrustGuardVerdict | None = None
    fire: bool = False
    error: BaseException | None = None


def block_text(verdict: TrustGuardVerdict) -> str:
    """Human-readable block message, including trace id when present."""
    if verdict.trace_id:
        return f"{BLOCKED} trace_id={verdict.trace_id}"
    return BLOCKED


def _contains_report_finding(findings: object) -> bool:
    """Return whether a Skip response still carries an earlier Report finding."""
    if not isinstance(findings, list):
        return False
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        outcome = finding.get("outcome")
        if isinstance(outcome, Mapping) and outcome.get("action") == STATUS_REPORT:
            return True
    return False


def classify_verdict(verdict: TrustGuardVerdict) -> Outcome:
    """Map a parsed evaluate body onto one outcome.

    ``skip`` continues like ``allow``. When a Report finding is already in that
    response, the findings are recorded and ``on_violation`` is not fired.
    """
    status = verdict.status
    if status == STATUS_ALLOW:
        return Outcome(kind="allow", verdict=verdict)
    if status == STATUS_SKIP:
        if _contains_report_finding(verdict.findings):
            return Outcome(kind="report", verdict=verdict, fire=False)
        return Outcome(kind="allow", verdict=verdict)
    if status == STATUS_REPORT:
        return Outcome(kind="report", verdict=verdict, fire=True)
    if status == STATUS_BLOCK:
        return Outcome(kind="block", text=block_text(verdict), verdict=verdict, fire=True)
    if status == STATUS_TRANSFORM:
        return Outcome(kind="transform", verdict=verdict, fire=True)
    return Outcome(kind="fail_closed", text=UNKNOWN_VERDICT, verdict=verdict)


def classify_exception(
    exc: BaseException, *, unreachable_fallback: UnreachableFallback
) -> Outcome:
    """Map a transport or programming failure onto one outcome."""
    if isinstance(exc, GraphBubbleUp | TrustGuardBlockedError):
        return Outcome(kind="reraise", error=exc)
    if isinstance(exc, TrustGuardUnreachableError):
        text = str(exc) or UNREACHABLE
        if unreachable_fallback == "fail_open":
            return Outcome(kind="fail_open", text=text)
        return Outcome(kind="fail_closed", text=text)
    if isinstance(exc, TrustGuardError):
        return Outcome(kind="fail_closed", text=str(exc) or REQUEST_FAILED)
    return Outcome(kind="fail_closed", text=REQUEST_FAILED)
