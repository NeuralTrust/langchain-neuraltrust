"""Paired extract/apply codec between LangChain messages and TrustGuard payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_neuraltrust._types import TRANSFORM_MISSING, TrustGuardTransformError

JsonObject = dict[str, Any]
_TEXT_PART_KEYS = frozenset({"type", "text"})


def last_index_of(messages: Sequence[BaseMessage], message_type: type[BaseMessage]) -> int | None:
    """Return the last index of ``message_type``, or ``None``."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], message_type):
            return index
    return None


def role_of(message: BaseMessage) -> str:
    """Map a LangChain message to an OpenAI-style role."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    type_name = getattr(message, "type", None)
    if type_name == "human":
        return "user"
    if type_name == "ai":
        return "assistant"
    if isinstance(type_name, str) and type_name:
        return type_name
    return "user"


def _openai_function_call(call_id: str, name: str, args: object) -> JsonObject:
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _openai_tool_calls(message: AIMessage) -> list[JsonObject] | None:
    if message.tool_calls:
        return [
            _openai_function_call(
                str(tool_call.get("id") or ""),
                str(tool_call.get("name") or ""),
                tool_call.get("args"),
            )
            for tool_call in message.tool_calls
        ]
    raw = message.additional_kwargs.get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return None
    converted = [item for item in raw if isinstance(item, dict)]
    return converted or None


def message_to_payload(message: BaseMessage) -> JsonObject:
    """Serialize one LangChain message to the TrustGuard/OpenAI chat shape."""
    payload: JsonObject = {"role": role_of(message)}
    if message.content is not None:
        payload["content"] = message.content
    if isinstance(message, AIMessage):
        tool_calls = _openai_tool_calls(message)
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
    if isinstance(message, ToolMessage):
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            payload["name"] = message.name
    return payload


def extract_input_payload(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[object] | None = None,
) -> JsonObject:
    """Build the evaluate payload for ``direction=input``."""
    payload: JsonObject = {"messages": [message_to_payload(message) for message in messages]}
    if tools:
        payload["tools"] = list(tools)
    return payload


def extract_output_payload(messages: Sequence[BaseMessage]) -> JsonObject | None:
    """Build the evaluate payload for the last AI message."""
    index = last_index_of(messages, AIMessage)
    if index is None:
        return None
    return {"messages": [message_to_payload(messages[index])]}


def extract_tool_results_payload(messages: Sequence[BaseMessage]) -> JsonObject | None:
    """Build the evaluate payload for tool results after the last AI message."""
    start, end = tool_result_span(messages)
    if start is None or end is None or start >= end:
        return None
    return {"messages": [message_to_payload(message) for message in messages[start:end]]}


def extract_tool_call_payload(tool_call: Mapping[str, Any]) -> JsonObject:
    """Build the evaluate payload for a single pending tool call."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _openai_function_call(
                        str(tool_call.get("id") or ""),
                        str(tool_call.get("name") or ""),
                        tool_call.get("args"),
                    )
                ],
            }
        ]
    }


def tool_result_span(messages: Sequence[BaseMessage]) -> tuple[int | None, int | None]:
    """Return ``[start, end)`` covering ToolMessages after the last AI message."""
    last_ai = last_index_of(messages, AIMessage)
    if last_ai is None:
        return None, None
    start = last_ai + 1
    while start < len(messages) and not isinstance(messages[start], ToolMessage):
        start += 1
    end = start
    while end < len(messages) and isinstance(messages[end], ToolMessage):
        end += 1
    if start >= end:
        return None, None
    return start, end


def _single_text_coverable(content: object) -> bool:
    """True when a single ``input`` string can replace this content without dropping data."""
    if isinstance(content, str):
        return True
    if not isinstance(content, list) or len(content) != 1:
        return False
    part = content[0]
    if isinstance(part, str):
        return True
    if not isinstance(part, Mapping):
        return False
    part_type = part.get("type")
    if part_type is not None and part_type != "text":
        return False
    return set(part.keys()) <= _TEXT_PART_KEYS


def _redacted_content(content: object, redacted: str) -> object:
    if not _single_text_coverable(content):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    if isinstance(content, str):
        return redacted
    if not isinstance(content, list) or not content:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    part = content[0]
    if isinstance(part, str):
        return [redacted]
    if not isinstance(part, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    return [{**dict(part), "text": redacted}]


def _parse_tool_args(raw: object) -> JsonObject:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise TrustGuardTransformError(TRANSFORM_MISSING) from exc
    if not isinstance(raw, dict):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    return raw


def _incoming_tool_call(item: object) -> tuple[str, JsonObject, str | None]:
    """Normalize a TrustGuard tool call to ``(name, args, id or None)``."""
    if not isinstance(item, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    if "name" in item and "args" in item:
        call_id = item.get("id")
        return (
            str(item.get("name") or ""),
            _parse_tool_args(item["args"]),
            call_id if isinstance(call_id, str) and call_id else None,
        )
    function = item.get("function")
    if not isinstance(function, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    call_id = item.get("id")
    return (
        str(function.get("name") or ""),
        _parse_tool_args(function.get("arguments", "{}")),
        call_id if isinstance(call_id, str) and call_id else None,
    )


def _original_tool_identity(original: object) -> tuple[str, str]:
    if isinstance(original, Mapping):
        return str(original.get("name") or ""), str(original.get("id") or "")
    return "", ""


def _apply_tool_calls(
    original_calls: Sequence[object], incoming_calls: Sequence[object]
) -> list[JsonObject]:
    if len(incoming_calls) != len(original_calls):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    converted: list[JsonObject] = []
    for original, incoming in zip(original_calls, incoming_calls, strict=True):
        name, args, incoming_id = _incoming_tool_call(incoming)
        orig_name, orig_id = _original_tool_identity(original)
        if name and orig_name and name != orig_name:
            raise TrustGuardTransformError(TRANSFORM_MISSING)
        if incoming_id and orig_id and incoming_id != orig_id:
            raise TrustGuardTransformError(TRANSFORM_MISSING)
        converted.append(
            {
                "name": orig_name or name,
                "args": args,
                "id": orig_id or incoming_id or "",
                "type": "tool_call",
            }
        )
    return converted


def _original_tool_calls(message: BaseMessage) -> list[object] | None:
    if isinstance(message, AIMessage) and message.tool_calls:
        return list(message.tool_calls)
    raw = message.additional_kwargs.get("tool_calls")
    return raw if isinstance(raw, list) else None


def _apply_one(original: BaseMessage, incoming: object) -> BaseMessage:
    if not isinstance(incoming, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    incoming_role = incoming.get("role")
    if isinstance(incoming_role, str) and incoming_role and incoming_role != role_of(original):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    updates: JsonObject = {}
    if "content" in incoming:
        content = incoming["content"]
        if isinstance(content, str):
            updates["content"] = _redacted_content(original.content, content)
        elif isinstance(content, list):
            original_content = original.content
            if not isinstance(original_content, list) or len(original_content) != len(content):
                raise TrustGuardTransformError(TRANSFORM_MISSING)
            updates["content"] = content
        else:
            raise TrustGuardTransformError(TRANSFORM_MISSING)
    if "tool_calls" in incoming:
        incoming_calls = incoming["tool_calls"]
        original_calls = _original_tool_calls(original)
        if not isinstance(incoming_calls, list) or original_calls is None:
            raise TrustGuardTransformError(TRANSFORM_MISSING)
        updates["tool_calls"] = _apply_tool_calls(original_calls, incoming_calls)
    if not updates:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    return original.model_copy(update=updates)


def apply_transform_to_messages(
    messages: Sequence[BaseMessage],
    transformed: object,
    *,
    span: tuple[int, int] | None = None,
) -> list[BaseMessage]:
    """Rewrite ``messages[span]`` from ``transformed_payload``, preserving ids.

    ``span`` is ``[start, end)``. ``None`` means the full conversation.
    Empty, shorter, or unusable transforms raise :class:`TrustGuardTransformError`.
    """
    if not isinstance(transformed, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)

    start, end = span if span is not None else (0, len(messages))
    if start < 0 or end > len(messages) or start >= end:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    span_len = end - start

    raw_messages = transformed.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        if len(raw_messages) < span_len:
            raise TrustGuardTransformError(TRANSFORM_MISSING)
        applied = raw_messages[-span_len:]
        rewritten = list(messages)
        for offset, incoming in enumerate(applied):
            rewritten[start + offset] = _apply_one(messages[start + offset], incoming)
        return rewritten

    raw_input = transformed.get("input")
    if not isinstance(raw_input, str) or not raw_input:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    rewritten = list(messages)
    rewritten[end - 1] = rewritten[end - 1].model_copy(
        update={"content": _redacted_content(messages[end - 1].content, raw_input)}
    )
    return rewritten


def apply_transform_to_tool_call(
    tool_call: Mapping[str, Any],
    transformed: object,
) -> dict[str, Any]:
    """Rewrite tool-call arguments from a transform. Fail closed on injection or mismatch."""
    if not isinstance(transformed, Mapping):
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    raw_messages = transformed.get("messages")
    incoming_calls: object = None
    if isinstance(raw_messages, list) and raw_messages:
        last = raw_messages[-1]
        if not isinstance(last, Mapping):
            raise TrustGuardTransformError(TRANSFORM_MISSING)
        incoming_calls = last.get("tool_calls")
    elif isinstance(transformed.get("input"), str) and transformed["input"]:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    if not isinstance(incoming_calls, list) or len(incoming_calls) != 1:
        raise TrustGuardTransformError(TRANSFORM_MISSING)
    rewritten = _apply_tool_calls([tool_call], incoming_calls)[0]
    next_call = dict(tool_call)
    next_call["args"] = rewritten["args"]
    next_call["name"] = rewritten["name"]
    next_call["id"] = rewritten["id"]
    return next_call
