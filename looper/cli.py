"""CLI entry point — Hermes `cli.py`, one-shot + REPL only in v1.

Usage:
  python -m looper.cli "summarise README.md"          # one-shot
  python -m looper.cli --chat                          # REPL, /quit to exit
  python -m looper.cli "do X" --model openrouter/... --max-turns 10 --session abc
  LOOPER_AUTO_APPROVE=1 python -m looper.cli "run the tests"
"""
from __future__ import annotations

import argparse
import sys

from .loop import Looper


def main(argv=None):
    ap = argparse.ArgumentParser(prog="looper", description="Custom Hermes-style agent looper")
    ap.add_argument("task", nargs="?", help="one-shot task")
    ap.add_argument("--chat", action="store_true", help="interactive REPL")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--session", default=None)
    ap.add_argument("--auto-approve", action="store_true")
    args = ap.parse_args(argv)

    def on_tool(name, payload):
        print(f"  [tool {name}] {str(payload)[:200]}", file=sys.stderr)

    agent = Looper(model=args.model, max_turns=args.max_turns, session_id=args.session,
                   on_tool=on_tool, auto_approve=args.auto_approve)
    if args.chat or not args.task:
        print("looper — type a task, /quit to exit, /stop to stop the current run")
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line in ("/quit", "/exit"):
                break
            if not line:
                continue
            print(agent.chat(line))
        return 0
    print(agent.chat(args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
