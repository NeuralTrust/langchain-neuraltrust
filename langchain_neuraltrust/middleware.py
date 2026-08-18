"""LangChain ``AgentMiddleware`` that evaluates traffic with NeuralTrust TrustGuard."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_neuraltrust._client import TrustGuardClient
from langchain_neuraltrust._payload import (
    apply_transform_to_messages,
    apply_transform_to_tool_call,
    extract_input_payload,
    extract_output_payload,
    extract_tool_call_payload,
    extract_tool_results_payload,
    last_index_of,
    tool_result_span,
)
from langchain_neuraltrust._types import (
    BLOCKED,
    DEFAULT_API_BASE,
    DEFAULT_TIMEOUT,
    MISSING_API_KEY,
    REQUEST_FAILED,
    STATUS_ALLOW,
    STATUS_BLOCK,
    STATUS_REPORT,
    STATUS_TRANSFORM,
    TRANSFORM_MISSING,
    UNKNOWN_VERDICT,
    UNREACHABLE,
    EvaluateDirection,
    ExitBehavior,
    TrustGuardBlockedError,
    TrustGuardError,
    TrustGuardTransformError,
    TrustGuardUnreachableError,
    TrustGuardVerdict,
    UnreachableFallback,
    ViolationStage,
)

OnViolation = Callable[[TrustGuardVerdict, ViolationStage], None]
ToolCallResult = ToolMessage | Command[Any]
ToolHandler = Callable[[ToolCallRequest], ToolCallResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolCallResult]]
EvaluateFn = Callable[[dict[str, Any]], TrustGuardVerdict]
AEvaluateFn = Callable[[dict[str, Any]], Awaitable[TrustGuardVerdict]]


@dataclass(frozen=True, slots=True)
class _EvalJob:
    stage: ViolationStage
    direction: EvaluateDirection
    payload: dict[str, Any]
    span: tuple[int, int]
    replace_index: int | None


@dataclass(frozen=True, slots=True)
class _HookUpdate:
    messages: list[BaseMessage] | None = None
    jump_to: Literal["end"] | None = None

    def as_dict(self) -> dict[str, Any] | None:
        if self.jump_to is not None:
            return {"jump_to": self.jump_to, "messages": self.messages or []}
        if self.messages is not None:
            return {"messages": self.messages}
        return None


@dataclass(frozen=True, slots=True)
class _ToolDecision:
    kind: Literal["proceed", "raise", "reject", "rewrite"]
    tool_call: dict[str, Any] | None = None
    message: ToolMessage | None = None
    error: str | None = None
    verdict: TrustGuardVerdict | None = None


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


class TrustGuardMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Evaluate LangChain agent traffic with TrustGuard ``POST /v1/evaluate``.

    Hooks: ``before_model`` / ``abefore_model``, ``after_model`` / ``aafter_model``,
    and optionally ``wrap_tool_call`` / ``awrap_tool_call``.

    ``on_violation`` is invoked synchronously from both sync and async hooks.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        collector_key: str | None = None,
        session_id: str | None = None,
        model_name: str | None = None,
        check_input: bool = True,
        check_output: bool = True,
        check_tool_results: bool = False,
        check_tool_calls: bool = False,
        exit_behavior: ExitBehavior = "end",
        unreachable_fallback: UnreachableFallback = "fail_closed",
        timeout: float = DEFAULT_TIMEOUT,
        payload_tools: Sequence[object] | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        on_violation: OnViolation | None = None,
    ) -> None:
        """Create the middleware.

        Args:
            api_key: TrustGuard API key (``tgk_...``). Falls back to ``TRUSTGUARD_API_KEY``.
            api_base: TrustGuard base URL. Falls back to ``TRUSTGUARD_API_BASE``, then
                ``https://trustguard.neuraltrust.ai``.
            collector_key: Optional collector key (``tgcol_...``). Falls back to
                ``TRUSTGUARD_COLLECTOR_KEY``.
            session_id: Optional session id forwarded as ``session_id``. Falls back to
                ``TRUSTGUARD_SESSION_ID``.
            model_name: Optional model name sent in ``attributes.model.name``.
            check_input: Evaluate conversation messages in ``before_model``.
            check_output: Evaluate the last AI message in ``after_model``.
            check_tool_results: Evaluate tool outputs in ``before_model``.
            check_tool_calls: Gate tool execution in ``wrap_tool_call``.
            exit_behavior: How to handle ``block`` (and fail-closed errors):
                ``end``, ``error``, or ``replace``.
            unreachable_fallback: ``fail_closed`` or ``fail_open`` for connect errors,
                timeouts, and HTTP 502/504 only.
            timeout: HTTP timeout in seconds.
            payload_tools: Optional JSON-serializable OpenAI tool dicts, or LangChain
                tools (converted at construction). Arbitrary objects fail construction.
            client: Optional ``httpx.Client`` for connection reuse, proxies, or tests.
            async_client: Optional ``httpx.AsyncClient`` for the same. Owned clients are
                recreated if the event loop changes.
            on_violation: Optional callback for report/block/transform verdicts.
                Called synchronously from both ``invoke`` and ``ainvoke``. Exceptions
                from this callback propagate and are not mapped to a TrustGuard failure.
        """
        super().__init__()
        resolved_key = api_key or _env("TRUSTGUARD_API_KEY")
        if not resolved_key:
            raise ValueError(MISSING_API_KEY)
        if exit_behavior not in ("end", "error", "replace"):
            raise ValueError("exit_behavior must be 'end', 'error', or 'replace'")
        if unreachable_fallback not in ("fail_closed", "fail_open"):
            raise ValueError("unreachable_fallback must be 'fail_closed' or 'fail_open'")

        self.api_key = resolved_key
        self.api_base = _validated_api_base(
            api_base or _env("TRUSTGUARD_API_BASE") or DEFAULT_API_BASE
        )
        self.collector_key = collector_key or _env("TRUSTGUARD_COLLECTOR_KEY") or ""
        self.session_id = session_id or _env("TRUSTGUARD_SESSION_ID") or ""
        self.model_name = model_name or ""
        self.check_input = check_input
        self.check_output = check_output
        self.check_tool_results = check_tool_results
        self.check_tool_calls = check_tool_calls
        self.exit_behavior: ExitBehavior = exit_behavior
        self.unreachable_fallback: UnreachableFallback = unreachable_fallback
        self.timeout = float(timeout)
        self.payload_tools = _coerce_payload_tools(payload_tools)
        self.on_violation = on_violation
        self._client = TrustGuardClient(
            api_key=self.api_key,
            api_base=self.api_base,
            timeout=self.timeout,
            client=client,
            async_client=async_client,
        )

    def evaluate_body(
        self,
        payload: dict[str, Any],
        direction: EvaluateDirection,
        runtime: object | None = None,
    ) -> dict[str, Any]:
        """Build the TrustGuard evaluate request body."""
        body: dict[str, Any] = {
            "payload": payload,
            "direction": direction,
            "protocol": "llm",
            "attributes": {
                "content_type": "application/json",
                "model": {"name": self._model_name(runtime)},
            },
        }
        if self.collector_key:
            body["collector_key"] = self.collector_key
        if self.session_id:
            body["session_id"] = self.session_id
        return body

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Evaluate input and optional tool results before the model is called."""
        return self._run_stages(state, runtime, self._before_stage_names(), self._client.evaluate)

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Evaluate the last AI message after the model is called."""
        return self._run_stages(state, runtime, self._after_stage_names(), self._client.evaluate)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_model`."""
        return await self._arun_stages(
            state, runtime, self._before_stage_names(), self._client.aevaluate
        )

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_model`."""
        return await self._arun_stages(
            state, runtime, self._after_stage_names(), self._client.aevaluate
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResult:
        """Gate a tool call before it executes."""
        if not self.check_tool_calls:
            return handler(request)
        decision = self._decide_tool(request, self._client.evaluate)
        return self._apply_tool_decision(request, handler, decision)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResult:
        """Async version of :meth:`wrap_tool_call`."""
        if not self.check_tool_calls:
            return await handler(request)
        decision = await self._adecide_tool(request, self._client.aevaluate)
        return await self._aapply_tool_decision(request, handler, decision)

    def _before_stage_names(self) -> tuple[ViolationStage, ...]:
        stages: list[ViolationStage] = []
        if self.check_tool_results:
            stages.append("tool")
        if self.check_input:
            stages.append("input")
        return tuple(stages)

    def _after_stage_names(self) -> tuple[ViolationStage, ...]:
        return ("output",) if self.check_output else ()

    def _run_stages(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
        stages: Sequence[ViolationStage],
        evaluate: EvaluateFn,
    ) -> dict[str, Any] | None:
        messages: list[BaseMessage] = list(state.get("messages") or [])
        if not stages or not messages:
            return None
        working = messages
        modified = False
        for stage in stages:
            job = self._job_for(stage, working)
            if job is None:
                continue
            update = self._eval_job(job, working, runtime, evaluate)
            if update is None:
                continue
            if update.jump_to is not None:
                return update.as_dict()
            if update.messages is not None:
                working = update.messages
                modified = True
        if modified:
            return {"messages": working}
        return None

    async def _arun_stages(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
        stages: Sequence[ViolationStage],
        evaluate: AEvaluateFn,
    ) -> dict[str, Any] | None:
        messages: list[BaseMessage] = list(state.get("messages") or [])
        if not stages or not messages:
            return None
        working = messages
        modified = False
        for stage in stages:
            job = self._job_for(stage, working)
            if job is None:
                continue
            update = await self._aeval_job(job, working, runtime, evaluate)
            if update is None:
                continue
            if update.jump_to is not None:
                return update.as_dict()
            if update.messages is not None:
                working = update.messages
                modified = True
        if modified:
            return {"messages": working}
        return None

    def _job_for(self, stage: ViolationStage, messages: list[BaseMessage]) -> _EvalJob | None:
        if stage == "tool":
            payload = extract_tool_results_payload(messages)
            start, end = tool_result_span(messages)
            if payload is None or start is None or end is None:
                return None
            return _EvalJob(stage, "input", payload, (start, end), end - 1)
        if stage == "input":
            return _EvalJob(
                stage,
                "input",
                extract_input_payload(messages, tools=self.payload_tools),
                (0, len(messages)),
                last_index_of(messages, HumanMessage),
            )
        payload = extract_output_payload(messages)
        index = last_index_of(messages, AIMessage)
        if payload is None or index is None:
            return None
        return _EvalJob(stage, "output", payload, (index, index + 1), index)

    def _eval_job(
        self,
        job: _EvalJob,
        messages: list[BaseMessage],
        runtime: Runtime[Any],
        evaluate: EvaluateFn,
    ) -> _HookUpdate | None:
        try:
            verdict = evaluate(self.evaluate_body(job.payload, job.direction, runtime))
        except Exception as exc:
            return self._recover_hook(exc, job, messages)
        return self._apply_verdict(messages, verdict, job)

    async def _aeval_job(
        self,
        job: _EvalJob,
        messages: list[BaseMessage],
        runtime: Runtime[Any],
        evaluate: AEvaluateFn,
    ) -> _HookUpdate | None:
        try:
            verdict = await evaluate(self.evaluate_body(job.payload, job.direction, runtime))
        except Exception as exc:
            return self._recover_hook(exc, job, messages)
        return self._apply_verdict(messages, verdict, job)

    def _apply_verdict(
        self,
        messages: list[BaseMessage],
        verdict: TrustGuardVerdict,
        job: _EvalJob,
    ) -> _HookUpdate | None:
        if verdict.status == STATUS_ALLOW:
            return None
        if verdict.status == STATUS_REPORT:
            self._fire(verdict, job.stage)
            return self._attach_report(messages, job.replace_index, verdict)
        if verdict.status == STATUS_BLOCK:
            return self._handle_block(messages, job=job, verdict=verdict)
        if verdict.status != STATUS_TRANSFORM:
            return self._fail_closed(
                UNKNOWN_VERDICT,
                verdict=verdict,
                stage=job.stage,
                messages=messages,
                span=job.span,
            )
        self._fire(verdict, job.stage)
        try:
            rewritten = apply_transform_to_messages(
                messages, verdict.transformed_payload, span=job.span
            )
        except TrustGuardTransformError:
            return self._fail_closed(
                TRANSFORM_MISSING,
                verdict=verdict,
                stage=job.stage,
                messages=messages,
                span=job.span,
            )
        return _HookUpdate(messages=rewritten)

    def _decide_tool(self, request: ToolCallRequest, evaluate: EvaluateFn) -> _ToolDecision:
        try:
            verdict = evaluate(
                self.evaluate_body(
                    extract_tool_call_payload(request.tool_call), "input", request.runtime
                )
            )
        except Exception as exc:
            return self._recover_tool(exc, request)
        return self._tool_decision(request, verdict)

    async def _adecide_tool(self, request: ToolCallRequest, evaluate: AEvaluateFn) -> _ToolDecision:
        try:
            verdict = await evaluate(
                self.evaluate_body(
                    extract_tool_call_payload(request.tool_call), "input", request.runtime
                )
            )
        except Exception as exc:
            return self._recover_tool(exc, request)
        return self._tool_decision(request, verdict)

    def _tool_decision(self, request: ToolCallRequest, verdict: TrustGuardVerdict) -> _ToolDecision:
        if verdict.status in {STATUS_ALLOW, STATUS_REPORT}:
            if verdict.status == STATUS_REPORT:
                self._fire(verdict, "tool_call")
            return _ToolDecision(kind="proceed")
        if verdict.status == STATUS_BLOCK:
            self._fire(verdict, "tool_call")
            if self.exit_behavior == "error":
                return _ToolDecision(
                    kind="raise", error=self._block_message(verdict), verdict=verdict
                )
            return _ToolDecision(
                kind="reject", message=self._tool_error(request, self._block_message(verdict))
            )
        if verdict.status != STATUS_TRANSFORM:
            return _ToolDecision(kind="reject", message=self._tool_error(request, UNKNOWN_VERDICT))
        self._fire(verdict, "tool_call")
        try:
            return _ToolDecision(
                kind="rewrite",
                tool_call=apply_transform_to_tool_call(
                    request.tool_call, verdict.transformed_payload
                ),
            )
        except TrustGuardTransformError:
            return _ToolDecision(
                kind="reject", message=self._tool_error(request, TRANSFORM_MISSING)
            )

    def _apply_tool_decision(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
        decision: _ToolDecision,
    ) -> ToolCallResult:
        if decision.kind == "proceed":
            return handler(request)
        if decision.kind == "raise":
            raise TrustGuardBlockedError(
                decision.error or REQUEST_FAILED, verdict=decision.verdict, stage="tool_call"
            )
        if decision.kind == "rewrite" and decision.tool_call is not None:
            return handler(self._with_tool_call(request, decision.tool_call))
        return decision.message or self._tool_error(request, REQUEST_FAILED)

    async def _aapply_tool_decision(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
        decision: _ToolDecision,
    ) -> ToolCallResult:
        if decision.kind == "proceed":
            return await handler(request)
        if decision.kind == "raise":
            raise TrustGuardBlockedError(
                decision.error or REQUEST_FAILED, verdict=decision.verdict, stage="tool_call"
            )
        if decision.kind == "rewrite" and decision.tool_call is not None:
            return await handler(self._with_tool_call(request, decision.tool_call))
        return decision.message or self._tool_error(request, REQUEST_FAILED)

    def _with_tool_call(
        self, request: ToolCallRequest, tool_call: dict[str, Any]
    ) -> ToolCallRequest:
        return request.override(tool_call=cast(ToolCall, tool_call))

    def _handle_unreachable(
        self, error: TrustGuardUnreachableError, job: _EvalJob, messages: list[BaseMessage]
    ) -> _HookUpdate | None:
        if self.unreachable_fallback == "fail_open":
            return None
        return self._fail_closed(
            str(error) or UNREACHABLE, stage=job.stage, messages=messages, span=job.span
        )

    def _fail_closed(
        self,
        message: str,
        *,
        verdict: TrustGuardVerdict | None = None,
        stage: ViolationStage,
        messages: list[BaseMessage] | None = None,
        span: tuple[int, int] | None = None,
    ) -> _HookUpdate:
        if self.exit_behavior == "error":
            raise TrustGuardBlockedError(message, verdict=verdict, stage=stage)
        if messages is not None and span is not None:
            return _HookUpdate(messages=_end_messages(messages, span, message), jump_to="end")
        return _HookUpdate(messages=[AIMessage(content=message)], jump_to="end")

    def _recover_hook(
        self, exc: BaseException, job: _EvalJob, messages: list[BaseMessage]
    ) -> _HookUpdate | None:
        if isinstance(exc, TrustGuardBlockedError | GraphBubbleUp):
            raise exc
        if isinstance(exc, TrustGuardUnreachableError):
            return self._handle_unreachable(exc, job, messages)
        if isinstance(exc, TrustGuardError):
            return self._fail_closed(
                str(exc) or REQUEST_FAILED, stage=job.stage, messages=messages, span=job.span
            )
        return self._fail_closed(REQUEST_FAILED, stage=job.stage, messages=messages, span=job.span)

    def _recover_tool(self, exc: BaseException, request: ToolCallRequest) -> _ToolDecision:
        if isinstance(exc, GraphBubbleUp):
            raise exc
        if isinstance(exc, TrustGuardBlockedError):
            return _ToolDecision(kind="raise", error=str(exc), verdict=exc.verdict)
        if isinstance(exc, TrustGuardUnreachableError):
            if self.unreachable_fallback == "fail_open":
                return _ToolDecision(kind="proceed")
            return _ToolDecision(kind="reject", message=self._tool_error(request, UNREACHABLE))
        if isinstance(exc, TrustGuardError):
            return _ToolDecision(
                kind="reject", message=self._tool_error(request, str(exc) or REQUEST_FAILED)
            )
        return _ToolDecision(kind="reject", message=self._tool_error(request, REQUEST_FAILED))

    def _handle_block(
        self,
        messages: list[BaseMessage],
        *,
        job: _EvalJob,
        verdict: TrustGuardVerdict,
    ) -> _HookUpdate:
        text = self._block_message(verdict)
        self._fire(verdict, job.stage)
        if self.exit_behavior == "error":
            raise TrustGuardBlockedError(text, verdict=verdict, stage=job.stage)
        if self.exit_behavior == "end":
            return _HookUpdate(messages=_end_messages(messages, job.span, text), jump_to="end")
        working = list(messages)
        start, end = job.span
        for index in range(start, end):
            if isinstance(working[index], SystemMessage):
                continue
            working[index] = _neutralize(working[index], text)
        return _HookUpdate(messages=working)

    def _attach_report(
        self,
        messages: list[BaseMessage],
        index: int | None,
        verdict: TrustGuardVerdict,
    ) -> _HookUpdate | None:
        if index is None:
            return None
        working = list(messages)
        message = working[index]
        extra = dict(message.additional_kwargs)
        extra["trustguard"] = {
            "status": STATUS_REPORT,
            "trace_id": verdict.trace_id,
            "request_id": verdict.request_id,
            "findings": verdict.findings,
        }
        working[index] = message.model_copy(update={"additional_kwargs": extra})
        return _HookUpdate(messages=working)

    def _block_message(self, verdict: TrustGuardVerdict) -> str:
        if verdict.trace_id:
            return f"{BLOCKED} trace_id={verdict.trace_id}"
        return BLOCKED

    def _fire(self, verdict: TrustGuardVerdict, stage: ViolationStage) -> None:
        if self.on_violation is None:
            return
        self.on_violation(verdict, stage)

    def _tool_error(self, request: ToolCallRequest, content: str) -> ToolMessage:
        tool_call_id = str(request.tool_call.get("id") or "")
        return ToolMessage(content=content, tool_call_id=tool_call_id, status="error")

    def _model_name(self, runtime: object | None) -> str:
        if self.model_name:
            return self.model_name
        if runtime is None:
            return ""
        context = getattr(runtime, "context", None)
        if isinstance(context, dict):
            model = context.get("model")
            if isinstance(model, str):
                return model
        return ""

    def close(self) -> None:
        """Close owned HTTP clients."""
        self._client.close()

    async def aclose(self) -> None:
        """Close owned async HTTP clients."""
        await self._client.aclose()


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _validated_api_base(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return url.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return url.rstrip("/")
    raise ValueError("TrustGuard api_base must be an https URL")


def _coerce_payload_tools(tools: Sequence[object] | None) -> list[object] | None:
    if not tools:
        return None
    converted: list[object] = []
    for item in tools:
        if isinstance(item, Mapping):
            converted.append(dict(item))
            continue
        try:
            converted.append(convert_to_openai_tool(cast(Any, item)))
        except Exception as exc:
            raise ValueError(
                "payload_tools must be JSON-serializable tool dicts or LangChain tools"
            ) from exc
    try:
        json.dumps(converted)
    except TypeError as exc:
        raise ValueError("payload_tools must be JSON serializable") from exc
    return converted


def _neutralize(message: BaseMessage, text: str) -> BaseMessage:
    updates: dict[str, Any] = {"content": text}
    if isinstance(message, AIMessage):
        updates["tool_calls"] = []
        updates["invalid_tool_calls"] = []
    return message.model_copy(update=updates)


def _end_messages(
    messages: Sequence[BaseMessage], span: tuple[int, int], text: str
) -> list[BaseMessage]:
    start, end = span
    removals: list[BaseMessage] = []
    for message in messages[start:end]:
        if isinstance(message, SystemMessage):
            continue
        if message.id:
            removals.append(RemoveMessage(id=message.id))
    return [*removals, AIMessage(content=text)]
