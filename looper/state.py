"""Session storage — Hermes `hermes_state.py`, shrunk.

SQLite, one row per message, FTS5 full-text search for the
`session_search` tool. Lineage: compression creates a child session
(parent_session_id), so a resumed session can trace back to the
original — same idea as Hermes, one table instead of five.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            from .prompt import home_dir
            path = home_dir() / "sessions.db"
        self.path = Path(path) if str(path) != ":memory:" else ":memory:"
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
          session_id TEXT PRIMARY KEY, parent_session_id TEXT,
          created_at REAL, updated_at REAL, title TEXT);
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
          role TEXT, content TEXT, tool_calls TEXT, ts REAL);
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
          USING fts5(content, content='messages', content_rowid='id');
        """)
        self.conn.commit()

    def new_session(self, parent: str | None = None, title: str = "") -> str:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self.conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)",
                          (sid, parent, now, now, title))
        self.conn.commit()
        return sid

    def append(self, session_id: str, role: str, content: str, tool_calls: list | None = None):
        cur = self.conn.execute(
            "INSERT INTO messages(session_id, role, content, tool_calls, ts) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(tool_calls or []), time.time()))
        self.conn.execute("INSERT INTO messages_fts(rowid, content) VALUES (?,?)",
                          (cur.lastrowid, content or ""))
        self.conn.execute("UPDATE sessions SET updated_at=? WHERE session_id=?",
                          (time.time(), session_id))
        self.conn.commit()

    def history(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, tool_calls FROM messages WHERE session_id=? ORDER BY id",
            (session_id,)).fetchall()
        out = []
        for role, content, tc in rows:
            m: dict = {"role": role, "content": content or ""}
            calls = json.loads(tc or "[]")
            if calls:
                m["tool_calls"] = calls
            out.append(m)
        return out

    def search(self, query: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT m.session_id, m.role, m.content FROM messages_fts f "
            "JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? LIMIT ?",
            (query, limit)).fetchall()
        return [{"session_id": s, "role": r, "content": (c or "")[:500]} for s, r, c in rows]

    def child_session(self, parent: str, summary: str) -> str:
        """Compression lineage: new session, summary as first message."""
        child = self.new_session(parent=parent, title="compressed")
        self.append(child, "system", f"[Compressed history summary]\n{summary}")
        return child
