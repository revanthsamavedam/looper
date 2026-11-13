# Custom Looper

A minimal agent loop built on the [Hermes Agent architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) — not a fork, a re-implementation of the shape, small enough to read in one sitting.

## Hermes -> Looper map

| Hermes | Looper (this repo) | Notes |
|---|---|---|
| `AIAgent` / `run_agent.py` facade | `looper/loop.py:Looper` — `chat()` / `run_conversation()` | Same two entry points |
| `agent/conversation_loop.py` + `turn_*.py` | `loop.py` — prep, preflight, API call, dispatch, recovery | One sync loop, like Hermes |
| `agent/prompt_builder.py` | `looper/prompt.py` | SOUL + USER + MEMORY + skills + tool list, cached |
| `hermes_cli/runtime_provider.py` (18+ providers, 3 API modes) | `looper/provider.py` | **V1: `chat_completions` only** (OpenAI / OpenRouter / Nous / DeepSeek / Ollama). Anthropic native is a documented v2, not a half-implementation |
| `tools/registry.py` | `looper/registry.py` | Self-register at import, `check_fn`, dispatch wraps errors as JSON |
| `model_tools.py` / `toolsets.py` | `registry.get_definitions()` + `tools.py` | Toolsets: `file`, `terminal`, `agent` |
| `tools/approval.py` | `looper/approval.py` | Dangerous patterns, per-session + permanent allowlist |
| `agent/context_compressor.py` | `looper/compress.py` | Preflight >50%, protect last 20, never split tool pairs |
| `hermes_state.py` (SQLite + FTS5) | `looper/state.py` | Sessions, lineage on compression, FTS5 `session_search` |
| `gateway/`, `cron/`, `acp_adapter/` | **Out of scope v1** | CLI + Python library only; gateway is the next layer |

## Tools (v1)

- `file`: `read_file`, `write_file`, `list_files`
- `terminal`: `terminal` (dangerous commands ask first)
- `agent` (intercepted by the loop, Hermes-style): `todo`, `memory`, `session_search`, `delegate_task` (sub-looper, max 10 turns)

## Run it

```bash
# stdlib only — no pip install needed
export LOOPER_API_KEY=...            # or OPENAI_API_KEY
export LOOPER_MODEL=openai/gpt-4o-mini   # or openrouter/..., ollama/llama3.1, ...

python -m looper.cli "read README.md and summarise it in 3 bullets"
python -m looper.cli --chat
```

Python library:

```python
from looper import Looper
agent = Looper(model="openrouter/anthropic/claude-sonnet-4.6")
print(agent.chat("Fix the bug in main.py"))
result = agent.run_conversation("...", task_id="task_abc123")  # full dict: final_response, messages, turns, usage
```

No key? Run the mock demo + tests — they use a scripted provider, no network:

```bash
python examples/demo_mock.py
python tests/test_looper.py
```

## Config

| Env | Meaning |
|---|---|
| `LOOPER_MODEL` | e.g. `openai/gpt-4o-mini`, `openrouter/...`, `ollama/...` |
| `LOOPER_API_KEY` / `LOOPER_BASE_URL` | Override key / endpoint |
| `LOOPER_FALLBACK_MODELS` | Comma-separated fallback models |
| `LOOPER_HOME` | `~/.looper` by default — SOUL.md, USER.md, MEMORY.md, skills/, sessions.db |
| `LOOPER_AUTO_APPROVE` | `1` skips dangerous-command prompts (tests / trusted batch only) |

## What to decide next

1. **Provider** — OpenAI, OpenRouter, Ollama local, or Nous?
2. **First job** — coding, research, file chores?
3. **Next layer** — gateway (Telegram/WhatsApp), cron jobs, or MCP tools?
