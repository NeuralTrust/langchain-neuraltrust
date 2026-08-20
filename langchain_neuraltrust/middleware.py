"""LangChain ``AgentMiddleware`` that evaluates traffic with NeuralTrust TrustGuard."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, get_args
from urllib.parse import urlparse

import httpx
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_neuraltrust._client import TrustGuardClient
from langchain_neuraltrust._payload import (
    apply_transform_to_messages,
    apply_transform_to_tool_call,
    end_messages,
    extract_input_payload,
    extract_output_payload,
    extract_tool_call_payload,
    extract_tool_results,
    last_index_of,
    neutralize,
)
from langchain_neuraltrust._policy import Outcome, classify_exception, classify_verdict
from langchain_neuraltrust._types import (
    DEFAULT_API_BASE,
    DEFAULT_TIMEOUT,
    MISSING_API_KEY,
    REQUEST_FAILED,
    TRANSFORM_MISSING,
    ExitBehavior,
    HookStage,
    TrustGuardBlockedError,
    TrustGuardTransformError,
    TrustGuardVerdict,
    UnreachableFallback,
    ViolationStage,
)

OnViolation = Callable[[TrustGuardVerdict, ViolationStage], None]
ToolCallResult = ToolMessage | Command[Any]
ToolHandler = Callable[[ToolCallRequest], ToolCallResult]
AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[ToolCallResult]]
GatedTool = ToolCallRequest | ToolMessage
EvaluateBody = dict[str, Any]
StageDriver = Generator[EvaluateBody, TrustGuardVerdict, dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class _EvalJob:
    stage: HookStage
    payload: dict[str, Any]
    targets: tuple[int, ...]
    end_targets: tuple[int, ...]
    replace_targets: tuple[int, ...]
    replace_index: int | None

    @property
    def direction(self) -> str:
        return "output" if self.stage == "output" else "input"


@dataclass(frozen=True, slots=True)
class _Jump:
    messages: list[BaseMessage]

    def as_dict(self) -> dict[str, Any]:
        return {"jump_to": "end", "messages": self.messages}


@dataclass(frozen=True, slots=True)
class _Rewrite:
    messages: list[BaseMessage]


HookUpdate = _Jump | _Rewrite | None


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _context_value(runtime: object | None, name: str) -> str:
    if runtime is None:
        return ""
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping):
        value = context.get(name)
    else:
        value = getattr(context, name, None)
    return value if isinstance(value, str) else ""


class TrustGuardMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Evaluate LangChain agent traffic with NeuralTrust TrustGuard.

    Runs ``before_model`` / ``after_model`` (and their async twins) against
    ``POST {api_base}/v1/evaluate``. Set ``check_tool_calls=True`` to also wrap
    tool execution. ``on_violation`` is invoked synchronously from both sync and
    async hooks.

    Environment fallbacks: ``TRUSTGUARD_API_KEY``, ``TRUSTGUARD_API_BASE``,
    ``TRUSTGUARD_COLLECTOR_KEY``, ``TRUSTGUARD_SESSION_ID``,
    ``TRUSTGUARD_MODEL_NAME``, ``TRUSTGUARD_TIMEOUT``.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_neuraltrust import TrustGuardMiddleware

        agent = create_agent(
            model="gpt-4o-mini",
            tools=tools,
            middleware=[
                TrustGuardMiddleware(
                    check_input=True,
                    check_output=True,
                    payload_tools=tools,
                )
            ],
        )
        result = agent.invoke({"messages": [("human", "Hello")]})
        ```
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
        timeout: float | None = None,
        payload_tools: Sequence[object] | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        on_violation: OnViolation | None = None,
    ) -> None:
        """Create the middleware.

        Args:
            api_key: TrustGuard API key (``tgk_...``). Falls back to
                ``TRUSTGUARD_API_KEY``.
            api_base: TrustGuard base URL. Falls back to ``TRUSTGUARD_API_BASE``,
                then ``https://trustguard.neuraltrust.ai``.
            collector_key: Optional collector key (``tgcol_...``). Falls back to
                ``TRUSTGUARD_COLLECTOR_KEY``.
            session_id: Optional session id forwarded as ``session_id``. Falls
                back to ``TRUSTGUARD_SESSION_ID``, then
                ``runtime.execution_info.thread_id``.
            model_name: Optional model name sent in ``attributes.model.name``.
                Falls back to ``TRUSTGUARD_MODEL_NAME``, then runtime context.
            check_input: Evaluate conversation messages in ``before_model``.
            check_output: Evaluate the last AI message in ``after_model``.
            check_tool_results: Evaluate tool outputs in ``before_model``. Skipped
                when ``check_input`` is also true, because the conversation
                payload already includes those tool messages.
            check_tool_calls: Gate tool execution in ``wrap_tool_call``. When
                false, those methods are not registered on the class.
            exit_behavior: How to handle ``block`` and fail-closed errors:
                ``end``, ``error``, or ``replace``. The same value applies to
                hooks and ``wrap_tool_call``.
            unreachable_fallback: ``fail_closed`` or ``fail_open`` for connect
                errors, timeouts, HTTP 502/504, and exhausted HTTP 429 retries.
            timeout: HTTP timeout in seconds. Falls back to
                ``TRUSTGUARD_TIMEOUT``, then ``5.0``.
            payload_tools: Optional JSON-serializable OpenAI tool dicts, or
                LangChain tools (converted at construction). Do not pass this as
                ``tools``: ``AgentMiddleware.tools`` is reserved by
                ``create_agent``.
            client: Optional ``httpx.Client`` for connection reuse, proxies, or
                tests.
            async_client: Optional ``httpx.AsyncClient`` for the same. Owned
                clients are keyed per event loop.
            on_violation: Optional callback for report/block/transform verdicts.
                Called synchronously from both ``invoke`` and ``ainvoke``.
                Exceptions from this callback propagate and are not mapped to a
                TrustGuard failure.
        """
        super().__init__()
        resolved_key = api_key or _env("TRUSTGUARD_API_KEY")
        if not resolved_key:
            raise ValueError(MISSING_API_KEY)
        if exit_behavior not in get_args(ExitBehavior):
            raise ValueError("exit_behavior must be 'end', 'error', or 'replace'")
        if unreachable_fallback not in get_args(UnreachableFallback):
            raise ValueError("unreachable_fallback must be 'fail_closed' or 'fail_open'")

        env_timeout = _env("TRUSTGUARD_TIMEOUT")
        if timeout is not None:
            resolved_timeout = float(timeout)
        elif env_timeout:
            resolved_timeout = float(env_timeout)
        else:
            resolved_timeout = DEFAULT_TIMEOUT

        self.api_key = resolved_key
        self.api_base = _validated_api_base(
            api_base or _env("TRUSTGUARD_API_BASE") or DEFAULT_API_BASE
        )
        self.collector_key = collector_key or _env("TRUSTGUARD_COLLECTOR_KEY") or ""
        self.session_id = session_id or _env("TRUSTGUARD_SESSION_ID") or ""
        self.model_name = model_name or _env("TRUSTGUARD_MODEL_NAME") or ""
        self.check_input = check_input
        self.check_output = check_output
        self.check_tool_results = check_tool_results
        self.check_tool_calls = check_tool_calls
        self.exit_behavior: ExitBehavior = exit_behavior
        self.unreachable_fallback: UnreachableFallback = unreachable_fallback
        self.timeout = resolved_timeout
        self.payload_tools = _coerce_payload_tools(payload_tools)
        self.on_violation = on_violation
        self._client = TrustGuardClient(
            api_key=self.api_key,
            api_base=self.api_base,
            timeout=self.timeout,
            client=client,
            async_client=async_client,
        )
        if check_tool_calls:
            self.__class__ = _TrustGuardToolCallMiddleware  # noqa: PLC3002

    def _evaluate_body(
        self,
        payload: dict[str, Any],
        direction: str,
        runtime: object | None = None,
    ) -> dict[str, Any]:
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
        session_id = self._session_id(runtime)
        if session_id:
            body["session_id"] = session_id
        return body

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Evaluate input and optional tool results before the model is called.

        Args:
            state: Agent state containing ``messages``.
            runtime: LangGraph runtime.

        Returns:
            A state update, or ``None`` when the conversation is unchanged.
        """
        return _drive(
            self._stage_driver(
                list(state.get("messages") or []), runtime, self._before_stage_names()
            ),
            self._client.evaluate,
        )

    @hook_config(can_jump_to=["end"])
    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Evaluate the last AI message after the model is called.

        Args:
            state: Agent state containing ``messages``.
            runtime: LangGraph runtime.

        Returns:
            A state update, or ``None`` when the conversation is unchanged.
        """
        return _drive(
            self._stage_driver(
                list(state.get("messages") or []), runtime, self._after_stage_names()
            ),
            self._client.evaluate,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_model`."""
        return await _adrive(
            self._stage_driver(
                list(state.get("messages") or []), runtime, self._before_stage_names()
            ),
            self._client.aevaluate,
        )

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of :meth:`after_model`."""
        return await _adrive(
            self._stage_driver(
                list(state.get("messages") or []), runtime, self._after_stage_names()
            ),
            self._client.aevaluate,
        )

    def _before_stage_names(self) -> tuple[HookStage, ...]:
        stages: list[HookStage] = []
        if self.check_tool_results and not self.check_input:
            stages.append("tool")
        if self.check_input:
            stages.append("input")
        return tuple(stages)

    def _after_stage_names(self) -> tuple[HookStage, ...]:
        return ("output",) if self.check_output else ()

    def _stage_driver(
        self,
        messages: list[BaseMessage],
        runtime: Runtime[Any],
        stages: Sequence[HookStage],
    ) -> StageDriver:
        working = messages
        if not stages or not working:
            yield from ()
            return None
        modified = False
        for stage in stages:
            try:
                job = self._job_for(stage, working)
            except Exception as exc:
                dummy = _EvalJob(stage, {}, tuple(range(len(working))), (), (), None)
                update = self._render_hook(
                    working,
                    dummy,
                    classify_exception(
                        exc, unreachable_fallback=self.unreachable_fallback
                    ),
                )
                working, early, changed = _consume_update(working, update)
                if early is not None:
                    return early
                modified = modified or changed
                continue
            if job is None:
                continue
            try:
                verdict = yield self._evaluate_body(job.payload, job.direction, runtime)
            except Exception as exc:
                outcome = classify_exception(
                    exc, unreachable_fallback=self.unreachable_fallback
                )
            else:
                outcome = classify_verdict(verdict)
            update = self._render_hook(working, job, outcome)
            working, early, changed = _consume_update(working, update)
            if early is not None:
                return early
            modified = modified or changed
        return {"messages": working} if modified else None

    def _job_for(self, stage: HookStage, messages: list[BaseMessage]) -> _EvalJob | None:
        if stage == "tool":
            extracted = extract_tool_results(messages)
            if extracted is None:
                return None
            indices, payload = extracted
            last_ai = last_index_of(messages, AIMessage)
            block = (last_ai, *indices) if last_ai is not None else indices
            return _EvalJob(stage, payload, indices, block, block, indices[-1])
        if stage == "input":
            all_idx = tuple(range(len(messages)))
            last_human = last_index_of(messages, HumanMessage)
            end_targets = (
                tuple(range(last_human, len(messages)))
                if last_human is not None
                else all_idx
            )
            return _EvalJob(
                stage,
                extract_input_payload(messages, tools=self.payload_tools),
                all_idx,
                end_targets,
                all_idx,
                last_human,
            )
        if stage == "output":
            output_payload = extract_output_payload(messages)
            index = last_index_of(messages, AIMessage)
            if output_payload is None or index is None:
                return None
            targets = (index,)
            return _EvalJob(stage, output_payload, targets, targets, targets, index)
        raise ValueError(f"unknown hook stage: {stage}")

    def _render_hook(
        self, messages: list[BaseMessage], job: _EvalJob, outcome: Outcome
    ) -> HookUpdate:
        if outcome.kind == "reraise":
            if outcome.error is None:
                raise TrustGuardBlockedError(REQUEST_FAILED, stage=job.stage)
            raise outcome.error
        if outcome.kind in {"allow", "fail_open"}:
            return None
        if outcome.kind == "report":
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, job.stage)
            return self._attach_report(messages, job.replace_index, outcome.verdict)
        if outcome.kind == "block":
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, job.stage)
            return self._handle_block(
                messages, job=job, text=outcome.text, verdict=outcome.verdict
            )
        if outcome.kind == "transform":
            payload = outcome.verdict.transformed_payload if outcome.verdict else None
            try:
                rewritten = apply_transform_to_messages(
                    messages, payload, indices=job.targets
                )
            except TrustGuardTransformError:
                return self._fail_closed(
                    TRANSFORM_MISSING,
                    verdict=outcome.verdict,
                    stage=job.stage,
                    messages=messages,
                    targets=job.end_targets,
                )
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, job.stage)
            return _Rewrite(rewritten)
        return self._fail_closed(
            outcome.text or REQUEST_FAILED,
            verdict=outcome.verdict,
            stage=job.stage,
            messages=messages,
            targets=job.end_targets,
        )

    def _render_tool(self, request: ToolCallRequest, outcome: Outcome) -> GatedTool:
        if outcome.kind == "reraise":
            if outcome.error is None:
                raise TrustGuardBlockedError(REQUEST_FAILED, stage="tool_call")
            raise outcome.error
        if outcome.kind in {"allow", "fail_open"}:
            return request
        if outcome.kind == "report":
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, "tool_call")
            return request
        if outcome.kind == "block":
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, "tool_call")
            return self._tool_closed(request, outcome.text, outcome.verdict)
        if outcome.kind == "transform":
            payload = outcome.verdict.transformed_payload if outcome.verdict else None
            try:
                rewritten = apply_transform_to_tool_call(request.tool_call, payload)
            except TrustGuardTransformError:
                return self._tool_closed(request, TRANSFORM_MISSING, outcome.verdict)
            if outcome.fire and outcome.verdict is not None:
                self._fire(outcome.verdict, "tool_call")
            return request.override(tool_call=rewritten)
        return self._tool_closed(request, outcome.text or REQUEST_FAILED, outcome.verdict)

    def _fail_closed(
        self,
        message: str,
        *,
        verdict: TrustGuardVerdict | None = None,
        stage: ViolationStage,
        messages: list[BaseMessage] | None = None,
        targets: tuple[int, ...] | None = None,
    ) -> _Jump:
        if self.exit_behavior == "error":
            raise TrustGuardBlockedError(message, verdict=verdict, stage=stage)
        if messages is not None and targets is not None:
            return _Jump(end_messages(messages, targets, message))
        return _Jump([AIMessage(content=message)])

    def _tool_closed(
        self,
        request: ToolCallRequest,
        message: str,
        verdict: TrustGuardVerdict | None = None,
    ) -> ToolMessage:
        if self.exit_behavior == "error":
            raise TrustGuardBlockedError(message, verdict=verdict, stage="tool_call")
        tool_call_id = str(request.tool_call.get("id") or "")
        return ToolMessage(content=message, tool_call_id=tool_call_id, status="error")

    def _handle_block(
        self,
        messages: list[BaseMessage],
        *,
        job: _EvalJob,
        text: str,
        verdict: TrustGuardVerdict | None,
    ) -> HookUpdate:
        if self.exit_behavior == "error":
            raise TrustGuardBlockedError(text, verdict=verdict, stage=job.stage)
        if self.exit_behavior == "end":
            return _Jump(end_messages(messages, job.end_targets, text))
        working = list(messages)
        for index in job.replace_targets:
            if isinstance(working[index], SystemMessage):
                continue
            working[index] = neutralize(working[index], text)
        return _Rewrite(working)

    def _attach_report(
        self,
        messages: list[BaseMessage],
        index: int | None,
        verdict: TrustGuardVerdict | None,
    ) -> HookUpdate:
        if verdict is None:
            return None
        if index is None:
            index = next(
                (
                    i
                    for i in range(len(messages) - 1, -1, -1)
                    if not isinstance(messages[i], SystemMessage)
                ),
                len(messages) - 1 if messages else None,
            )
        if index is None:
            return None
        working = list(messages)
        message = working[index]
        extra = dict(message.additional_kwargs)
        extra["trustguard"] = {
            "status": verdict.status,
            "trace_id": verdict.trace_id,
            "request_id": verdict.request_id,
            "findings": verdict.findings,
        }
        working[index] = message.model_copy(update={"additional_kwargs": extra})
        return _Rewrite(working)

    def _fire(self, verdict: TrustGuardVerdict, stage: ViolationStage) -> None:
        if self.on_violation is None:
            return
        self.on_violation(verdict, stage)

    def _model_name(self, runtime: object | None) -> str:
        return self.model_name or _context_value(runtime, "model")

    def _session_id(self, runtime: object | None) -> str:
        if self.session_id:
            return self.session_id
        if runtime is None:
            return ""
        info = getattr(runtime, "execution_info", None)
        thread_id = getattr(info, "thread_id", None)
        if isinstance(thread_id, str) and thread_id:
            return thread_id
        config = getattr(runtime, "config", None)
        if isinstance(config, Mapping):
            configurable = config.get("configurable")
            if isinstance(configurable, Mapping):
                value = configurable.get("thread_id")
                if isinstance(value, str) and value:
                    return value
        return ""

    def close(self) -> None:
        """Close owned HTTP clients. Prefer :meth:`aclose` after ``ainvoke``."""
        self._client.close()

    async def aclose(self) -> None:
        """Close owned async HTTP clients."""
        await self._client.aclose()


class _TrustGuardToolCallMiddleware(TrustGuardMiddleware):
    """Same middleware with ``wrap_tool_call`` registered for ``create_agent``."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolCallResult:
        """Gate a tool call before it executes.

        Args:
            request: Pending tool call.
            handler: Next wrapper or the tool.

        Returns:
            The handler result, or a ``ToolMessage(status="error")`` when blocked.
        """
        gated = self._gate_tool(request)
        if isinstance(gated, ToolMessage):
            return gated
        return handler(gated)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolCallResult:
        """Async version of :meth:`wrap_tool_call`."""
        gated = await self._agate_tool(request)
        if isinstance(gated, ToolMessage):
            return gated
        return await handler(gated)

    def _gate_tool(self, request: ToolCallRequest) -> GatedTool:
        try:
            verdict = self._client.evaluate(
                self._evaluate_body(
                    extract_tool_call_payload(request.tool_call),
                    "input",
                    request.runtime,
                )
            )
        except Exception as exc:
            return self._render_tool(
                request,
                classify_exception(exc, unreachable_fallback=self.unreachable_fallback),
            )
        return self._render_tool(request, classify_verdict(verdict))

    async def _agate_tool(self, request: ToolCallRequest) -> GatedTool:
        try:
            verdict = await self._client.aevaluate(
                self._evaluate_body(
                    extract_tool_call_payload(request.tool_call),
                    "input",
                    request.runtime,
                )
            )
        except Exception as exc:
            return self._render_tool(
                request,
                classify_exception(exc, unreachable_fallback=self.unreachable_fallback),
            )
        return self._render_tool(request, classify_verdict(verdict))


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _validated_api_base(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query or parsed.fragment or parsed.params or not parsed.hostname:
        raise ValueError("TrustGuard api_base must be an https URL")
    if parsed.scheme == "https":
        return url.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS:
        return url.rstrip("/")
    raise ValueError("TrustGuard api_base must be an https URL")


def _coerce_payload_tools(tools: Sequence[object] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for item in tools:
        if isinstance(item, Mapping):
            converted.append(dict(item))
            continue
        try:
            converted.append(convert_to_openai_tool(cast(Any, item)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                "payload_tools must be JSON-serializable tool dicts or LangChain tools"
            ) from exc
    try:
        json.dumps(converted)
    except TypeError as exc:
        raise ValueError("payload_tools must be JSON serializable") from exc
    return converted


def _consume_update(
    working: list[BaseMessage], update: HookUpdate
) -> tuple[list[BaseMessage], dict[str, Any] | None, bool]:
    if update is None:
        return working, None, False
    if isinstance(update, _Jump):
        return working, update.as_dict(), False
    return update.messages, None, True


def _drive(
    driver: StageDriver, evaluate: Callable[[EvaluateBody], TrustGuardVerdict]
) -> dict[str, Any] | None:
    try:
        body = next(driver)
        while True:
            try:
                verdict = evaluate(body)
            except Exception as exc:
                body = driver.throw(exc)
            else:
                body = driver.send(verdict)
    except StopIteration as done:
        value = done.value
        return value if value is None or isinstance(value, dict) else None


async def _adrive(
    driver: StageDriver,
    evaluate: Callable[[EvaluateBody], Awaitable[TrustGuardVerdict]],
) -> dict[str, Any] | None:
    try:
        body = next(driver)
        while True:
            try:
                verdict = await evaluate(body)
            except Exception as exc:
                body = driver.throw(exc)
            else:
                body = driver.send(verdict)
    except StopIteration as done:
        value = done.value
        return value if value is None or isinstance(value, dict) else None
