"""Prompt builder — Hermes `agent/prompt_builder.py`, shrunk.

Assembles the system prompt from, in order:
  1. SOUL.md / USER.md / MEMORY.md in the looper home (~/.looper/ or LOOPER_HOME)
  2. skills/*.md — one short line per skill file, full text only when named
  3. tool guidance — the registered tools, so the model knows what exists
  4. the task instruction passed to run_conversation()

Cached: rebuilt only when the source files' mtimes change.
"""
from __future__ import annotations

import os
from pathlib import Path

from .registry import registry

DEFAULT_SOUL = "You are Looper, a careful agent. Do the task with tools, verify before claiming done, say what you could not verify."
DEFAULT_USER = "User: Revanth."


def home_dir() -> Path:
    return Path(os.environ.get("LOOPER_HOME", Path.home() / ".looper"))


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _skill_index(skills_dir: Path) -> str:
    if not skills_dir.is_dir():
        return ""
    lines = []
    for p in sorted(skills_dir.glob("*.md")):
        first = (p.read_text(errors="replace").splitlines() or [""])[0].strip()
        lines.append(f"- {p.stem}: {first}")
    return "\n".join(lines)


class PromptBuilder:
    def __init__(self, home: Path | None = None):
        self.home = home or home_dir()
        self._cache_key: tuple | None = None
        self._cached: str | None = None

    def build(self, extra_instruction: str = "") -> str:
        home = self.home
        files = [home / "SOUL.md", home / "USER.md", home / "MEMORY.md"]
        mtimes = tuple(p.stat().st_mtime if p.exists() else 0 for p in files)
        skills_dir = home / "skills"
        skill_files = tuple(sorted(skills_dir.glob("*.md"))) if skills_dir.is_dir() else ()
        key = (mtimes, tuple(p.stat().st_mtime for p in skill_files), extra_instruction,
               tuple(registry.names()))
        if key == self._cache_key and self._cached is not None:
            return self._cached

        soul = _read(files[0]) or DEFAULT_SOUL
        user = _read(files[1]) or DEFAULT_USER
        memory = _read(files[2])
        tools = "\n".join(
            f"- {e.name}: {e.description}" for e in
            (registry.get(n) for n in registry.names()) if e)
        skills = _skill_index(skills_dir)

        parts = [f"# Identity\n{soul}", f"# User\n{user}"]
        if memory:
            parts.append(f"# Memory (durable facts — do not contradict without evidence)\n{memory}")
        if skills:
            parts.append(f"# Skills (read the full file only when the task needs it)\n{skills}")
        if tools:
            parts.append(
                "# Tools\nCall tools to act; do not claim an action you did not take. "
                "When a tool returns {\"error\": ...}, say so and try a different approach.\n" + tools)
        parts.append(
            "# Rules\n- Loop until the task is done or you hit the iteration budget; then summarise what is done and what is not.\n"
            "- Never present a tool result you did not receive. Verify file writes by reading back.\n"
            "- Prefer the smallest safe change that satisfies the request.")
        if extra_instruction:
            parts.append(f"# Task instruction\n{extra_instruction}")
        self._cache_key, self._cached = key, "\n\n".join(parts)
        return self._cached
