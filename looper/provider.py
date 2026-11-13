"""Provider resolution — Hermes `hermes_cli/runtime_provider.py`, shrunk.

Hermes maps (provider, model) -> (api_mode, api_key, base_url) for 18+
providers and 3 API modes. V1 of the looper keeps ONE mode —
``chat_completions`` (OpenAI-compatible) — because every provider
Revanth uses (OpenAI, OpenRouter, Nous, DeepSeek, local Ollama/vLLM)
speaks it. Anthropic native mode is a documented v2, not a silent
half-implementation.

Resolution order (same as Hermes):
  1. explicit args to Looper()
  2. LOOPER_MODEL / LOOPER_BASE_URL / LOOPER_API_KEY env
  3. model-prefix defaults (openrouter/, anthropic/ is rejected with a
     clear error in v1, ollama/ -> localhost:11434)
  4. OPENAI_API_KEY / OPENAI_BASE_URL fallback

The client itself is stdlib-only (urllib) — no openai/httpx dependency,
so the looper runs on a bare Python 3.12.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ResolvedProvider:
    model: str
    api_key: str
    base_url: str
    api_mode: str = "chat_completions"


_PREFIX_BASE_URLS = {
    "openrouter/": "https://openrouter.ai/api/v1",
    "deepseek/": "https://api.deepseek.com/v1",
    "nous/": "https://api.nousresearch.com/v1",
    "ollama/": "http://localhost:11434/v1",
    "openai/": "https://api.openai.com/v1",
}


def resolve_provider(model: str | None = None, api_key: str | None = None,
                     base_url: str | None = None) -> ResolvedProvider:
    model = model or os.environ.get("LOOPER_MODEL") or "openai/gpt-4o-mini"
    base_url = base_url or os.environ.get("LOOPER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("LOOPER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not base_url:
        for prefix, url in _PREFIX_BASE_URLS.items():
            if model.startswith(prefix):
                base_url = url
                # strip the provider prefix for the wire model name, except openai/
                if prefix != "openai/":
                    model = model[len(prefix):]
                break
        else:
            base_url = "https://api.openai.com/v1"
    if model.startswith("anthropic/"):
        raise ValueError(
            "anthropic/ models need the native Anthropic Messages API — "
            "not in looper v1 (chat_completions only). Use an OpenAI-compatible "
            "endpoint, or route Anthropic through OpenRouter: openrouter/anthropic/claude-sonnet-4.6")
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise ValueError(
            "no API key: set LOOPER_API_KEY (or OPENAI_API_KEY), or pass api_key= to Looper(). "
            "Local Ollama needs no key.")
    return ResolvedProvider(model=model, api_key=api_key, base_url=base_url.rstrip("/"))


def _post_json(url: str, payload: dict, api_key: str, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"provider HTTP {exc.code}: {body}") from exc


class OpenAICompatibleClient:
    """One chat_completions call. Returns Hermes-normalised dict:
    {content: str, tool_calls: [{id, name, arguments: dict}]}"""

    def __init__(self, provider: ResolvedProvider, fallback_models: list[str] | None = None):
        self.provider = provider
        self.fallback_models = fallback_models or [
            m.strip() for m in os.environ.get("LOOPER_FALLBACK_MODELS", "").split(",") if m.strip()
        ]

    def call(self, messages: list[dict], tools: list[dict]) -> dict:
        last_err: Exception | None = None
        for model in [self.provider.model, *self.fallback_models]:
            payload = {"model": model, "messages": messages, "temperature": 0.2}
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            for attempt in range(3):  # 429/5xx retry with backoff
                try:
                    data = _post_json(f"{self.provider.base_url}/chat/completions", payload,
                                      self.provider.api_key)
                    msg = data["choices"][0]["message"]
                    calls = []
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw_arguments": fn.get("arguments") or ""}
                        calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                                      "arguments": args})
                    return {"content": msg.get("content") or "", "tool_calls": calls,
                            "model": data.get("model", model),
                            "usage": data.get("usage", {})}
                except Exception as exc:  # noqa: BLE001 — retry, then fall back
                    last_err = exc
                    if "429" in str(exc) or "500" in str(exc) or "502" in str(exc) or "503" in str(exc):
                        time.sleep(2 ** attempt)
                        continue
                    break
        raise RuntimeError(f"provider call failed (tried {self.provider.model}"
                           f"{' + fallbacks' if self.fallback_models else ''}): {last_err}")
