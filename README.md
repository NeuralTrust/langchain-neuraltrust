# langchain-neuraltrust

LangChain 1.x middleware for [NeuralTrust TrustGuard](https://neuraltrust.ai). Evaluates agent input, model output, and optionally tool traffic with `POST /v1/evaluate`.

This repository is private while the package is iterated. It is not on PyPI yet.

## Install

```bash
pip install langchain-neuraltrust
```

Until the first release, install from this repo:

```bash
pip install -e .
```

## Configure

```python
from langchain.agents import create_agent
from langchain_neuraltrust import TrustGuardMiddleware

agent = create_agent(
    model="gpt-4o-mini",
    middleware=[
        TrustGuardMiddleware(
            # api_key="tgk_...",          # or TRUSTGUARD_API_KEY
            # collector_key="tgcol_...",  # or TRUSTGUARD_COLLECTOR_KEY
            check_input=True,
            check_output=True,
        )
    ],
    tools=[],
)
```

| Setting | Env var | Default |
|---|---|---|
| `api_key` | `TRUSTGUARD_API_KEY` | required |
| `api_base` | `TRUSTGUARD_API_BASE` | `https://trustguard.neuraltrust.ai` |
| `collector_key` | `TRUSTGUARD_COLLECTOR_KEY` | omitted from the body when unset |
| `session_id` | `TRUSTGUARD_SESSION_ID` | omitted from the body when unset |

## Verdicts

| TrustGuard | Middleware |
|---|---|
| `allow` | Continue |
| `report` | Continue. Fires `on_violation` and stores findings on `additional_kwargs["trustguard"]` |
| `block` | Honors `exit_behavior` |
| `transform` | Rewrites the matching messages **preserving `message.id`** so LangChain replaces instead of appending |

`exit_behavior`:

- `end` (default) — jump to the end of the agent with an `AIMessage`. Messages in the evaluated span (except `SystemMessage`) are removed via `RemoveMessage` so blocked content does not remain in the thread.
- `error` — raise `TrustGuardBlockedError`
- `replace` — rewrite **every non-system message in the evaluated span** in place and continue. `SystemMessage`s are left intact. Blocked `AIMessage`s have `tool_calls` cleared so tools cannot run.

`wrap_tool_call` blocks return a `ToolMessage(status="error")` and do not call the tool. Exceptions raised by the tool handler (including LangGraph interrupts) propagate.

`on_violation` is a synchronous callback. It runs from both `invoke` and `ainvoke`. Exceptions from the callback propagate; they are not turned into a TrustGuard failure.

## Fail-closed

`unreachable_fallback` applies only to connect errors, timeouts, and HTTP 502/504.

HTTP 401/403, 503 entitlements, other 4xx/5xx, non-JSON 200, unknown verdicts, and unusable transforms always fail closed — including when `unreachable_fallback="fail_open"`.

Transforms that only change `role`, send `content: null`, inject extra content blocks, or disagree with the original message role fail closed.

## Streaming

`after_model` sees the assembled `AIMessage` after the model call finishes. Token-level streaming is not evaluated mid-stream. Do not rely on this middleware to redact tokens as they leave the provider.

## Hooks

| Flag | Hook |
|---|---|
| `check_input=True` | `before_model` / `abefore_model` |
| `check_output=True` | `after_model` / `aafter_model` |
| `check_tool_results=True` | tool outputs, during `before_model` |
| `check_tool_calls=True` | `wrap_tool_call` / `awrap_tool_call` |

Both sync (`invoke`) and async (`ainvoke`) paths are implemented.

## Develop

```bash
make install
make lint
make typing
make test-unit
```

Live tests against the prod demo tenant run only when `tests/integration_tests/.creds.json` is present (gitignored). The `langchain-demo-*` collectors reuse the demo policies: allow is a no-op, block/report trigger on the keyword `forbidden`, and transform redacts an SSN such as `123-45-6789`.
