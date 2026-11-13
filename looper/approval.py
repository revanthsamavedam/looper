"""Dangerous-command approval — Hermes `tools/approval.py`, shrunk.

Same idea, smaller list: pattern -> description. The terminal tool
checks before executing. Approval is per-session (approve once, no
re-prompt for the same pattern), plus a permanent allowlist in
~/.looper/command_allowlist. LOOPER_AUTO_APPROVE=1 skips prompts —
for tests / trusted batch runs only, never the default.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-z]*r[a-z]*|--recursive)\b", "recursive delete"),
    (r"\bmkfs\b|\bdd\s+.*of=/dev/", "filesystem format / raw disk write"),
    (r"\bDROP\s+TABLE\b|\bDELETE\s+FROM\b(?!\s+\w+\s+WHERE)", "destructive SQL"),
    (r">\s*/etc/", "overwrite system config"),
    (r"\bsystemctl\s+(stop|disable)\b", "stop/disable a service"),
    (r"curl\s+.*\|\s*(sh|bash)\b", "remote code execution (curl | sh)"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}", "fork bomb"),
    (r"\bkill\s+-9\b", "force-kill process"),
]


def detect_dangerous_command(command: str) -> str | None:
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return description
    return None


class Approver:
    def __init__(self, home: Path | None = None, auto_approve: bool | None = None):
        self.home = home or Path(os.environ.get("LOOPER_HOME", Path.home() / ".looper"))
        self.auto_approve = (os.environ.get("LOOPER_AUTO_APPROVE") == "1"
                             if auto_approve is None else auto_approve)
        self._session_approved: set[str] = set()
        self._allowlist = self._load_allowlist()

    def _load_allowlist(self) -> set[str]:
        try:
            return {l.strip() for l in (self.home / "command_allowlist").read_text().splitlines()
                    if l.strip()}
        except OSError:
            return set()

    def approve(self, command: str, ask=None) -> bool:
        """Return True if the command may run. `ask` is a callback
        (description, command) -> bool; defaults to a CLI prompt."""
        reason = detect_dangerous_command(command)
        if reason is None:
            return True
        if self.auto_approve or reason in self._session_approved or reason in self._allowlist:
            return True
        if ask is not None:
            ok = ask(reason, command)
        else:
            ans = input(f"\n⚠ Dangerous command ({reason}):\n  {command}\nApprove? [y/N/a(ll this session)] ").strip().lower()
            ok = ans in ("y", "yes")
            if ans == "a":
                self._session_approved.add(reason)
                return True
        if ok:
            self._session_approved.add(reason)
        return ok
