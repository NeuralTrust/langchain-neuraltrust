# langchain-neuraltrust

LangChain 1.x middleware for [NeuralTrust TrustGuard](https://neuraltrust.ai). Evaluates agent input, model output, and optionally tool traffic with `POST /v1/evaluate`.

## Install

```bash
pip install langchain-neuraltrust
```

or:

```bash
uv add langchain-neuraltrust
```

## Configure

```python
from langchain.agents import create_agent
from langchain_neuraltrust import TrustGuardMiddleware

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    middleware=[
        TrustGuardMiddleware(
            # api_key="tgk_...",          # or TRUSTGUARD_API_KEY
            # collector_key="tgcol_...",  # or TRUSTGUARD_COLLECTOR_KEY
            check_input=True,
            check_output=True,
            payload_tools=tools,          # not `tools=` — that name is reserved
        )
    ],
)
```

Call `close()` after `invoke`, or `await aclose()` after `ainvoke`, when the middleware owns the HTTP clients.

| Setting | Env var | Default |
|---|---|---|
| `api_key` | `TRUSTGUARD_API_KEY` | required |
| `api_base` | `TRUSTGUARD_API_BASE` | `https://trustguard.neuraltrust.ai` |
| `collector_key` | `TRUSTGUARD_COLLECTOR_KEY` | omitted from the body when unset |
| `session_id` | `TRUSTGUARD_SESSION_ID` | omitted, then `runtime.execution_info.thread_id` |
| `model_name` | `TRUSTGUARD_MODEL_NAME` | omitted, then `runtime.context.model` |
| `timeout` | `TRUSTGUARD_TIMEOUT` | `5.0` seconds |

`payload_tools` is included on input-stage evaluate payloads. Pass OpenAI tool dicts or LangChain tools. Do not set `TrustGuardMiddleware.tools`; `create_agent` reserves that attribute.

## Verdicts

| TrustGuard | Middleware |
|---|---|
| `allow` | Continue |
| `report` | Continue. Fires `on_violation` and stores findings on `additional_kwargs["trustguard"]` |
| `block` | Honors `exit_behavior` |
| `transform` | Rewrites the matching messages **preserving `message.id`** so LangChain replaces instead of appending |

`exit_behavior`:

- `end` (default) — jump to the end of the agent with an `AIMessage`. On **input**, only the current turn is removed (from the last `HumanMessage` through the end), so earlier conversation is kept. On **output**, the last AI message is removed. On **tool results**, the originating `AIMessage` is removed with the `ToolMessage`s so the thread is not left with unanswered `tool_calls`. `SystemMessage`s are never removed.
- `error` — raise `TrustGuardBlockedError` (hooks and `wrap_tool_call`)
- `replace` — rewrite **every non-system message in the evaluated span** in place and continue. `SystemMessage`s are left intact. Blocked `AIMessage`s have `tool_calls` and `additional_kwargs["tool_calls"]` cleared. Blocked `ToolMessage`s are converted to `HumanMessage`s so they cannot orphan a tool response.

`wrap_tool_call` blocks return a `ToolMessage(status="error")` and do not call the tool, unless `exit_behavior="error"`, which raises `TrustGuardBlockedError`. Fail-closed errors on the tool path follow the same rule. Exceptions raised by the tool handler (including LangGraph interrupts) propagate.

`on_violation` is a synchronous callback. It runs from both `invoke` and `ainvoke`. Exceptions from the callback propagate; they are not turned into a TrustGuard failure.

## Fail-closed

`unreachable_fallback` applies only to connect errors, timeouts, HTTP 502/504, and HTTP 429 after retries are exhausted. Those statuses are retried with backoff (honoring `Retry-After` when present) before the fallback is applied.

HTTP 401/403, 503 entitlements, other 4xx/5xx, non-JSON 200, unknown verdicts, and unusable transforms always fail closed — including when `unreachable_fallback="fail_open"`.

Transforms fail closed when they:

- return only `{"input": ...}` for a multi-message span
- return a `messages` array whose length differs from the evaluated span
- omit `role` or disagree with the original message role
- swap list-content block types or inject non-text parts
- rewrite a tool name or id, or fill in a missing original identity

## Streaming

`after_model` sees the assembled `AIMessage` after the model call finishes. Token-level streaming is not evaluated mid-stream. Do not rely on this middleware to redact tokens as they leave the provider.

## Hooks

| Flag | Hook |
|---|---|
| `check_input=True` | `before_model` / `abefore_model` |
| `check_output=True` | `after_model` / `aafter_model` |
| `check_tool_results=True` | tool outputs, during `before_model` (skipped when `check_input` is also true) |
| `check_tool_calls=True` | `wrap_tool_call` / `awrap_tool_call` |

If both `check_input` and `check_tool_results` are true, tool output is evaluated once as part of the conversation payload.

Both sync (`invoke`) and async (`ainvoke`) paths are implemented.

## Develop

```bash
make install
make lint
make typing
make test-unit
```

Live tests against the prod demo tenant run only when `tests/integration_tests/.creds.json` is present (gitignored). The `langchain-demo-*` collectors reuse the demo policies: allow is a no-op, block/report trigger on the keyword `forbidden`, and transform redacts an SSN such as `123-45-6789`.
