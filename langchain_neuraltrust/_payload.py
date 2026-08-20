"""Paired extract/apply codec between LangChain messages and TrustGuard payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.tool import ToolCall

from langchain_neuraltrust._types import TrustGuardTransformError

JsonObject = dict[str, Any]
_TEXT_PART_KEYS = frozenset({"type", "text"})


def _reject(reason: str) -> NoReturn:
    raise TrustGuardTransformError(reason)


def last_index_of(
    messages: Sequence[BaseMessage], message_type: type[BaseMessage]
) -> int | None:
    """Return the last index of ``message_type``, or ``None``."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], message_type):
            return index
    return None


def role_of(message: BaseMessage) -> str:
    """Map a LangChain message to an OpenAI-style role.

    Uses ``convert_to_openai_messages``. Unknown types fail closed.
    """
    try:
        converted = convert_to_openai_messages(message, text_format="string")
    except (TypeError, ValueError) as exc:
        raise TrustGuardTransformError("unknown_role") from exc
    if isinstance(converted, list):
        converted = converted[0] if converted else {}
    if isinstance(converted, Mapping):
        role = converted.get("role")
        if isinstance(role, str) and role:
            return role
    _reject("unknown_role")


def message_to_payload(message: BaseMessage) -> JsonObject:
    """Serialize one LangChain message to the TrustGuard/OpenAI chat shape."""
    try:
        converted = convert_to_openai_messages(message, text_format="string")
    except (TypeError, ValueError) as exc:
        raise TrustGuardTransformError("unserializable_message") from exc
    if isinstance(converted, list):
        converted = converted[0] if converted else {}
    if not isinstance(converted, dict):
        _reject("unserializable_message")
    payload = dict(converted)
    if isinstance(message.content, list):
        payload["content"] = message.content
    return payload


def extract_input_payload(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[object] | None = None,
) -> JsonObject:
    """Build the evaluate payload for ``direction=input``."""
    payload: JsonObject = {
        "messages": [message_to_payload(message) for message in messages]
    }
    if tools:
        payload["tools"] = list(tools)
    return payload


def extract_output_payload(messages: Sequence[BaseMessage]) -> JsonObject | None:
    """Build the evaluate payload for the last AI message."""
    index = last_index_of(messages, AIMessage)
    if index is None:
        return None
    return {"messages": [message_to_payload(messages[index])]}


def tool_result_indices(messages: Sequence[BaseMessage]) -> tuple[int, ...]:
    """Return indexes of every ``ToolMessage`` after the last AI message."""
    last_ai = last_index_of(messages, AIMessage)
    if last_ai is None:
        return ()
    return tuple(
        index
        for index in range(last_ai + 1, len(messages))
        if isinstance(messages[index], ToolMessage)
    )


def extract_tool_results(
    messages: Sequence[BaseMessage],
) -> tuple[tuple[int, ...], JsonObject] | None:
    """Return tool-result indexes and payload, or ``None`` when none exist."""
    indices = tool_result_indices(messages)
    if not indices:
        return None
    payload = {"messages": [message_to_payload(messages[index]) for index in indices]}
    return indices, payload


def extract_tool_results_payload(messages: Sequence[BaseMessage]) -> JsonObject | None:
    """Build the evaluate payload for tool results after the last AI message."""
    extracted = extract_tool_results(messages)
    return None if extracted is None else extracted[1]


def extract_tool_call_payload(tool_call: Mapping[str, Any]) -> JsonObject:
    """Build the evaluate payload for a single pending tool call."""
    arguments = tool_call.get("args")
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
    return {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(tool_call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str(tool_call.get("name") or ""),
                            "arguments": serialized,
                        },
                    }
                ],
            }
        ]
    }


def _is_text_part(part: object) -> bool:
    if isinstance(part, str):
        return True
    if not isinstance(part, Mapping):
        return False
    if part.get("type") not in (None, "text"):
        return False
    return set(part.keys()) <= _TEXT_PART_KEYS


def _redacted_content(content: object, redacted: str) -> object:
    if isinstance(content, str):
        return redacted
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, str):
            return [redacted]
        if _is_text_part(part) and isinstance(part, Mapping):
            return [{**dict(part), "text": redacted}]
    _reject("not_text_coverable")


def _apply_content_part(original: object, incoming: object) -> object:
    if isinstance(original, str):
        if not isinstance(incoming, str):
            _reject("content_part_type")
        return incoming
    if isinstance(original, Mapping) and isinstance(incoming, Mapping):
        if not _is_text_part(original) or not _is_text_part(incoming):
            _reject("content_part_keys")
        text = incoming.get("text")
        if not isinstance(text, str):
            _reject("content_part_text")
        return {**dict(original), "text": text}
    _reject("content_part_type")


def _apply_content_list(original_content: object, incoming: list[object]) -> list[object]:
    if not isinstance(original_content, list) or len(original_content) != len(incoming):
        _reject("content_length")
    return [
        _apply_content_part(original, incoming_part)
        for original, incoming_part in zip(original_content, incoming, strict=True)
    ]


def _parse_tool_args(raw: object) -> JsonObject:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise TrustGuardTransformError("tool_args_json") from exc
    if not isinstance(raw, dict):
        _reject("tool_args_type")
    return raw


def _incoming_tool_call(item: object) -> tuple[str, JsonObject, str | None]:
    """Normalize a TrustGuard tool call to ``(name, args, id or None)``."""
    if not isinstance(item, Mapping):
        _reject("tool_call_shape")
    if "name" in item and "args" in item:
        call_id = item.get("id")
        return (
            str(item.get("name") or ""),
            _parse_tool_args(item["args"]),
            call_id if isinstance(call_id, str) and call_id else None,
        )
    function = item.get("function")
    if not isinstance(function, Mapping):
        _reject("tool_call_shape")
    call_id = item.get("id")
    return (
        str(function.get("name") or ""),
        _parse_tool_args(function.get("arguments", "{}")),
        call_id if isinstance(call_id, str) and call_id else None,
    )


def _original_tool_identity(original: object) -> tuple[str, str]:
    if not isinstance(original, Mapping):
        _reject("tool_identity")
    name = str(original.get("name") or "")
    call_id = str(original.get("id") or "")
    function = original.get("function")
    if isinstance(function, Mapping):
        name = name or str(function.get("name") or "")
    if not name or not call_id:
        _reject("tool_identity")
    return name, call_id


def _apply_tool_calls(
    original_calls: Sequence[object], incoming_calls: Sequence[object]
) -> list[JsonObject]:
    if len(incoming_calls) != len(original_calls):
        _reject("tool_call_count")
    converted: list[JsonObject] = []
    for original, incoming in zip(original_calls, incoming_calls, strict=True):
        name, args, incoming_id = _incoming_tool_call(incoming)
        orig_name, orig_id = _original_tool_identity(original)
        if name and name != orig_name:
            _reject("tool_name_mismatch")
        if incoming_id and incoming_id != orig_id:
            _reject("tool_id_mismatch")
        converted.append(
            {
                "name": orig_name,
                "args": args,
                "id": orig_id,
                "type": "tool_call",
            }
        )
    return converted


def _original_tool_calls(message: BaseMessage) -> list[object] | None:
    if isinstance(message, AIMessage) and message.tool_calls:
        return list(message.tool_calls)
    raw = message.additional_kwargs.get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(item, Mapping) for item in raw):
        _reject("tool_identity")
    return raw


def _apply_one(original: BaseMessage, incoming: object) -> BaseMessage:
    if not isinstance(incoming, Mapping):
        _reject("message_shape")
    incoming_role = incoming.get("role")
    if not isinstance(incoming_role, str) or not incoming_role:
        _reject("role_missing")
    if incoming_role != role_of(original):
        _reject("role_mismatch")
    updates: JsonObject = {}
    if "content" in incoming:
        content = incoming["content"]
        if isinstance(content, str):
            updates["content"] = _redacted_content(original.content, content)
        elif isinstance(content, list):
            updates["content"] = _apply_content_list(original.content, content)
        else:
            _reject("content_type")
    if "tool_calls" in incoming:
        incoming_calls = incoming["tool_calls"]
        original_calls = _original_tool_calls(original)
        if not isinstance(incoming_calls, list) or original_calls is None:
            _reject("tool_calls_missing")
        updates["tool_calls"] = _apply_tool_calls(original_calls, incoming_calls)
    if not updates:
        _reject("empty_transform")
    return original.model_copy(update=updates)


def _targets(
    messages: Sequence[BaseMessage],
    span: tuple[int, int] | None,
    indices: Sequence[int] | None,
) -> list[int]:
    if indices is not None:
        targets = list(indices)
    else:
        start, end = span if span is not None else (0, len(messages))
        if start < 0 or end > len(messages) or start >= end:
            _reject("span")
        targets = list(range(start, end))
    if not targets or any(index < 0 or index >= len(messages) for index in targets):
        _reject("span")
    return targets


def apply_transform_to_messages(
    messages: Sequence[BaseMessage],
    transformed: object,
    *,
    span: tuple[int, int] | None = None,
    indices: Sequence[int] | None = None,
) -> list[BaseMessage]:
    """Rewrite targeted messages from ``transformed_payload``, preserving ids.

    ``span`` is ``[start, end)``. ``indices`` selects specific positions.
    ``None`` for both means the full conversation. Empty, shorter, longer, or
    unusable transforms raise :class:`TrustGuardTransformError`.
    """
    if not isinstance(transformed, Mapping):
        _reject("missing_payload")

    targets = _targets(messages, span, indices)
    span_len = len(targets)

    raw_messages = transformed.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        if len(raw_messages) != span_len:
            _reject("message_count")
        rewritten = list(messages)
        for index, incoming in zip(targets, raw_messages, strict=True):
            rewritten[index] = _apply_one(messages[index], incoming)
        return rewritten

    raw_input = transformed.get("input")
    if not isinstance(raw_input, str) or not raw_input:
        _reject("missing_payload")
    if span_len != 1:
        _reject("input_span")
    target = targets[0]
    rewritten = list(messages)
    rewritten[target] = rewritten[target].model_copy(
        update={"content": _redacted_content(messages[target].content, raw_input)}
    )
    return rewritten


def apply_transform_to_tool_call(
    tool_call: Mapping[str, Any],
    transformed: object,
) -> ToolCall:
    """Rewrite tool-call arguments from a transform.

    Fail closed on injection or mismatch.
    """
    if not isinstance(transformed, Mapping):
        _reject("missing_payload")
    raw_messages = transformed.get("messages")
    incoming_calls: object = None
    if isinstance(raw_messages, list) and raw_messages:
        last = raw_messages[-1]
        if not isinstance(last, Mapping):
            _reject("message_shape")
        incoming_calls = last.get("tool_calls")
    elif isinstance(transformed.get("input"), str) and transformed["input"]:
        _reject("input_on_tool_call")
    if not isinstance(incoming_calls, list) or len(incoming_calls) != 1:
        _reject("tool_call_count")
    rewritten = _apply_tool_calls([tool_call], incoming_calls)[0]
    next_call = dict(tool_call)
    next_call["args"] = rewritten["args"]
    next_call["name"] = rewritten["name"]
    next_call["id"] = rewritten["id"]
    next_call["type"] = "tool_call"
    return next_call  # type: ignore[return-value]


def neutralize(message: BaseMessage, text: str) -> BaseMessage:
    """Rewrite a blocked message so it cannot invoke or answer a tool call."""
    extra = dict(message.additional_kwargs)
    extra.pop("tool_calls", None)
    extra.pop("function_call", None)
    if isinstance(message, ToolMessage):
        return HumanMessage(content=text, id=message.id, additional_kwargs=extra)
    updates: dict[str, Any] = {"content": text, "additional_kwargs": extra}
    if isinstance(message, AIMessage):
        updates["tool_calls"] = []
        updates["invalid_tool_calls"] = []
    return message.model_copy(update=updates)


def end_messages(
    messages: Sequence[BaseMessage], targets: Sequence[int], text: str
) -> list[BaseMessage]:
    """Remove targeted messages and append a blocked ``AIMessage``.

    Messages without ids are neutralized in the returned snapshot so blocked
    content is never left unredacted in the hook update.
    """
    removals: list[BaseMessage] = []
    missing_id = False
    target_set = set(targets)
    for index in targets:
        message = messages[index]
        if isinstance(message, SystemMessage):
            continue
        if message.id:
            removals.append(RemoveMessage(id=message.id))
        else:
            missing_id = True
    if missing_id:
        working: list[BaseMessage] = []
        for index, message in enumerate(messages):
            if index in target_set and not isinstance(message, SystemMessage):
                assigned = message
                if not assigned.id:
                    assigned = assigned.model_copy(update={"id": str(uuid4())})
                working.append(neutralize(assigned, text))
            else:
                working.append(message)
        return [*working, AIMessage(content=text)]
    return [*removals, AIMessage(content=text)]
