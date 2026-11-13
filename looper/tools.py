"""Built-in tools — Hermes `tools/*.py`, shrunk to 6 tools.

Each tool is one function + one @tool decorator = one
registry.register() at import time. Adding a tool is adding a file
(or a function here) — no central list to edit.

Toolsets: file, terminal, agent (loop-intercepted tools live in
loop.py, not here — same split as Hermes: todo/memory/session_search/
delegate need agent state, so the loop handles them before dispatch).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .approval import Approver
from .registry import tool

_OBJ = {"type": "object", "properties": {}, "required": []}


def _props(**kwargs) -> dict:
    return {"type": "object", "properties": kwargs,
            "required": [k for k, v in kwargs.items() if v.get("required")]}
def _s(desc: str, required: bool = False) -> dict:
    return {"type": "string", "description": desc, "required": required}
def _i(desc: str) -> dict:
    return {"type": "integer", "description": desc}


# --- file toolset -------------------------------------------------------
@tool("list_files", "file", "List files in a directory (relative paths, max 100).",
      _props(path=_s("Directory to list"), recursive=_i("1 to recurse, 0 for top level only")), emoji="📁")
def list_files(path: str = ".", recursive: int = 0) -> str:
    root = Path(path)
    if not root.is_dir():
        return json.dumps({"error": f"not a directory: {path}"})
    files = sorted(p.relative_to(root).as_posix() for p in
                   (root.rglob("*") if recursive else root.iterdir()) if p.is_file())[:100]
    return json.dumps({"path": path, "files": files})


@tool("read_file", "file", "Read a text file. Returns content (truncated at 20k chars).",
      _props(path=_s("File to read", required=True)), emoji="📖")
def read_file(path: str) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({"path": path, "content": text[:20000],
                       "truncated": len(text) > 20000})


@tool("write_file", "file", "Write text to a file, creating parent dirs. Returns bytes written.",
      _props(path=_s("File to write", required=True), content=_s("Full file content", required=True)), emoji="✏️")
def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return json.dumps({"path": path, "bytes": len(content.encode())})


# --- terminal toolset ---------------------------------------------------
_APPROVER = Approver()

@tool("terminal", "terminal", "Run a shell command (timeout 60s). Dangerous commands ask first.",
      _props(command=_s("Shell command", required=True), cwd=_s("Working directory"),
             timeout=_i("Timeout in seconds (default 60)")), emoji="💻")
def terminal(command: str, cwd: str = ".", timeout: int = 60) -> str:
    if not _APPROVER.approve(command):
        return json.dumps({"error": "command denied by user approval", "command": command})
    try:
        r = subprocess.run(command, shell=True, cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"timed out after {timeout}s", "command": command})
    out = (r.stdout + r.stderr)[-8000:]
    return json.dumps({"command": command, "exit_code": r.returncode, "output": out})
