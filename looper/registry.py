"""Tool registry — Hermes `tools/registry.py`, shrunk.

Hermes rule kept verbatim: every tool self-registers at import time
with ``registry.register(...)``. No central import list to maintain —
add a file, it shows up. Handlers MUST return a string (Hermes
returns a JSON string); dispatch wraps any exception so the model
always gets a well-formed result, never a traceback.
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("looper.registry")


@dataclass
class ToolEntry:
    name: str
    toolset: str
    schema: dict  # OpenAI function schema: {name, description, parameters}
    handler: Callable[..., str]
    check_fn: Optional[Callable[[], bool]] = None
    is_async: bool = False
    description: str = ""
    emoji: str = "🔧"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, *, name, toolset, schema, handler, check_fn=None,
                 is_async=False, description="", emoji="🔧", override=False):
        if name in self._tools and self._tools[name].toolset != toolset and not override:
            log.error("register: %s already registered by toolset %s, refusing shadow",
                      name, self._tools[name].toolset)
            raise ValueError(f"tool {name!r} already registered by {self._tools[name].toolset}")
        # Schema description is authoritative for the model (Hermes rule).
        if "description" not in schema:
            schema = {**schema, "description": description}
        self._tools[name] = ToolEntry(
            name=name, toolset=toolset, schema=schema, handler=handler,
            check_fn=check_fn, is_async=is_async,
            description=description or schema.get("description", ""), emoji=emoji)

    def get(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def _available(self, entry: ToolEntry, _cache: dict) -> bool:
        if entry.check_fn is None:
            return True
        key = id(entry.check_fn)
        if key not in _cache:
            try:
                _cache[key] = bool(entry.check_fn())
            except Exception:
                _cache[key] = False  # exceptions = unavailable (fail-safe)
        return _cache[key]

    def get_definitions(self, toolsets: Optional[set[str]] = None) -> list[dict]:
        """OpenAI-format function definitions, filtered by toolset + check_fn."""
        out, cache = [], {}
        for entry in self._tools.values():
            if toolsets is not None and entry.toolset not in toolsets:
                continue
            if not self._available(entry, cache):
                continue
            out.append({"type": "function", "function": entry.schema})
        return out

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """registry.dispatch(): look up, call, wrap errors as JSON."""
        entry = self._tools.get(name)
        if entry is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = entry.handler(**args, **kwargs) if kwargs else entry.handler(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:  # noqa: BLE001 — model must get JSON, not a crash
            log.exception("tool %s failed", name)
            return json.dumps({"error": f"Tool execution failed: {type(exc).__name__}: {exc}"})


# Singleton, like Hermes.
registry = ToolRegistry()


def discover_builtin_tools(package: str = "looper.tools") -> None:
    """Import every tools module that self-registers (AST-free version:
    we import the bundled tools package; new files register on import)."""
    try:
        importlib.import_module(package)
    except ImportError:
        log.debug("no tools package %s", package)


def tool(name: str, toolset: str, description: str, parameters: dict,
         check_fn=None, emoji: str = "🔧"):
    """Decorator sugar over registry.register() — one decorator per tool."""
    def deco(fn):
        schema = {"name": name, "description": description, "parameters": parameters}
        registry.register(name=name, toolset=toolset, schema=schema,
                          handler=fn, check_fn=check_fn, description=description, emoji=emoji)
        return fn
    return deco
