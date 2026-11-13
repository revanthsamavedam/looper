"""The looper — Hermes `AIAgent.run_conversation()`, shrunk.

One synchronous imperative loop (Hermes is deliberately sync too):
  prompt -> provider call -> tool dispatch -> loop
  until the model answers in text, or the iteration budget runs out.

Turn lifecycle, mirroring Hermes `agent/turn_*.py`:
  1. prep: task_id, append user msg, build/reuse cached system prompt
  2. preflight compression (>50% context)
  3. API call (interruptible in Hermes; here: timeout + KeyboardInterrupt)
  4. parse: tool_calls -> dispatch (parallel via ThreadPoolExecutor),
     text -> persist, flush memory, return
  5. on error: retry, then fall back to the next provider model

Agent-loop tools (intercepted BEFORE registry dispatch, Hermes-style):
  todo, memory, session_search, delegate_task — they need agent state.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import tools as _tools  # noqa: F401 — import registers the built-ins
from .approval import Approver
from .compress import compress, estimate_tokens, needs_compression
from .prompt import PromptBuilder, home_dir
from .provider import OpenAICompatibleClient, resolve_provider
from .registry import registry
from .state import SessionStore

AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}


class IterationBudget:
    """Hermes IterationBudget: max_turns per agent, subagents capped lower."""
    def __init__(self, max_turns: int = 30):
        self.max_turns = max_turns
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_turns - self.used)


class Looper:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_turns: int = 30,
                 session_id: str | None = None, store: SessionStore | None = None,
                 client=None, quiet_mode: bool = True, home: Path | None = None,
                 on_tool=None, on_step=None, auto_approve: bool = False):
        self.provider = resolve_provider(model, api_key, base_url) if client is None else None
        self.client = client or OpenAICompatibleClient(self.provider)
        self.budget = IterationBudget(max_turns)
        self.home = home or home_dir()
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = store or SessionStore()
        self.session_id = session_id or self.store.new_session()
        self.prompt_builder = PromptBuilder(self.home)
        self.approver = Approver(self.home, auto_approve=auto_approve)
        self.quiet_mode = quiet_mode
        self.on_tool = on_tool  # fn(name, args) / fn(name, result)
        self.on_step = on_step  # fn(turn_no, state)
        self.todos: list[dict] = []
        self._stop = False

    # -- agent-loop tools -------------------------------------------------
    def _agent_tool(self, name: str, args: dict, task_id: str) -> str | None:
        if name == "todo":
            action = args.get("action", "list")
            if action == "add":
                self.todos.append({"item": args.get("item", ""), "done": False})
            elif action == "done":
                for t in self.todos:
                    if t["item"] == args.get("item"):
                        t["done"] = True
            return json.dumps({"todos": self.todos})
        if name == "memory":
            text = args.get("text", "")
            p = self.home / "MEMORY.md"
            p.write_text((p.read_text() if p.exists() else "") + f"\n- {text}\n")
            self.prompt_builder._cached = None  # force prompt rebuild
            return json.dumps({"saved": text})
        if name == "session_search":
            return json.dumps({"results": self.store.search(args.get("query", ""), 10)})
        if name == "delegate_task":
            child = Looper(max_turns=min(10, self.budget.remaining), client=self.client,
                           store=self.store, quiet_mode=True, home=self.home,
                           auto_approve=self.approver.auto_approve)
            return json.dumps({"subagent_result": child.chat(args.get("task", ""))})
        return None

    # -- the loop ---------------------------------------------------------
    def run_conversation(self, user_message: str, system_message: str | None = None,
                         task_id: str | None = None) -> dict:
        task_id = task_id or uuid.uuid4().hex[:12]
        history: list[dict] = [{"role": "system",
                                "content": system_message or self.prompt_builder.build()},
                               {"role": "user", "content": user_message}]
        # resume: load persisted history (minus the system prompt we just built)
        persisted = [m for m in self.store.history(self.session_id) if m["role"] != "system"]
        if persisted:
            history = history[:1] + persisted + history[1:]
        self.store.append(self.session_id, "user", user_message)
        start = time.time()

        while self.budget.remaining > 0 and not self._stop:
            self.budget.used += 1
            turn = self.budget.used
            if self.on_step:
                self.on_step(turn, {"remaining": self.budget.remaining})
            if needs_compression(history):
                history, summary = compress(history)
                self.session_id = self.store.child_session(self.session_id, summary)
            try:
                tools = registry.get_definitions()
                # agent-loop tool schemas (registered as stubs for the model to see)
                tools += [{"type": "function", "function": s} for s in _AGENT_TOOL_SCHEMAS]
                resp = self.client.call(history, tools)
            except Exception as exc:  # noqa: BLE001 — Hermes: retry/fallback, then surface
                history.append({"role": "assistant",
                                "content": f"[provider error, stopping: {exc}]"})
                break
            assistant_msg: dict = {"role": "assistant", "content": resp.get("content") or ""}
            if resp.get("tool_calls"):
                # internal simplified calls -> OpenAI wire format for history/provider
                assistant_msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for tc in resp["tool_calls"]]
            history.append(assistant_msg)
            self.store.append(self.session_id, "assistant", assistant_msg["content"],
                              assistant_msg.get("tool_calls"))

            if not resp.get("tool_calls"):
                return {"final_response": resp.get("content") or "",
                        "messages": history, "task_id": task_id,
                        "turns": turn, "seconds": round(time.time() - start, 1),
                        "model": resp.get("model"), "usage": resp.get("usage", {})}

            # dispatch: parallel, results reinserted in call order (Hermes rule)
            def _run(tc):
                if self.on_tool:
                    self.on_tool(tc["name"], tc["arguments"])
                intercepted = self._agent_tool(tc["name"], tc["arguments"], task_id)
                if intercepted is not None:
                    return tc["id"], intercepted
                return tc["id"], registry.dispatch(tc["name"], tc["arguments"])

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = dict(pool.map(_run, resp["tool_calls"]))
            for tc in resp["tool_calls"]:
                result = results[tc["id"]]
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                self.store.append(self.session_id, "tool", result)
                if self.on_tool:
                    self.on_tool(tc["name"], result)

        return {"final_response": history[-1].get("content", "") if history else "",
                "messages": history, "task_id": task_id, "turns": self.budget.used,
                "seconds": round(time.time() - start, 1),
                "stopped": "budget exhausted" if self.budget.remaining == 0 else "interrupted"}

    def chat(self, message: str) -> str:
        """Thin wrapper around run_conversation(), like Hermes chat()."""
        return self.run_conversation(message).get("final_response", "")

    def stop(self):
        self._stop = True


_AGENT_TOOL_SCHEMAS = [
    {"name": "todo", "description": "Track tasks: add/list/done items.",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["add", "list", "done"]},
         "item": {"type": "string"}}, "required": ["action"]}},
    {"name": "memory", "description": "Save a durable fact to MEMORY.md.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"]}},
    {"name": "session_search", "description": "Full-text search over past sessions.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "delegate_task", "description": "Spawn a sub-looper (max 10 turns) for a subtask.",
     "parameters": {"type": "object", "properties": {"task": {"type": "string"}},
                    "required": ["task"]}},
]
