"""Tests for the looper — mock provider, no API key, no network.

Run:  python -m pytest tests/ -q   (or: python tests/test_looper.py)
The mock is a scripted provider: a queue of responses the loop consumes
in order, exactly like a real model returning tool_calls then text.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from looper import Looper, SessionStore  # noqa: E402
from looper.compress import compress, estimate_tokens, needs_compression  # noqa: E402
from looper.registry import registry  # noqa: E402


class ScriptedClient:
    """Fake provider: pop responses in order."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def call(self, messages, tools):
        self.calls.append((messages, tools))
        if not self.script:
            return {"content": "done", "tool_calls": []}
        return self.script.pop(0)


def _looper(script, **kw):
    tmp = tempfile.mkdtemp()
    return Looper(client=ScriptedClient(script), store=SessionStore(":memory:"),
                  home=Path(tmp), auto_approve=True, **kw)


def test_tool_loop_reads_file_then_answers():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "note.txt"
        p.write_text("hello looper")
        script = [
            {"content": "", "tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": {"path": str(p)}}]},
            {"content": "The file says: hello looper", "tool_calls": []},
        ]
        agent = _looper(script)
        out = agent.chat(f"read {p} and tell me what it says")
        assert "hello looper" in out, out
        assert len(agent.client.calls) == 2
        # tool result was fed back into history
        last_call_msgs = agent.client.calls[-1][0]
        assert any(m.get("role") == "tool" and "hello looper" in m.get("content", "")
                   for m in last_call_msgs)


def test_unknown_tool_returns_error_json_not_crash():
    script = [
        {"content": "", "tool_calls": [
            {"id": "c1", "name": "nope_tool", "arguments": {}}]},
        {"content": "that tool does not exist, stopping", "tool_calls": []},
    ]
    assert "does not exist" in _looper(script).chat("use nope_tool")


def test_budget_exhausted_stops():
    script = [{"content": "", "tool_calls": [
        {"id": "c1", "name": "list_files", "arguments": {"path": "."}}]}] * 10
    r = _looper(script, max_turns=2).run_conversation("list files forever")
    assert r["turns"] == 2 and r.get("stopped") == "budget exhausted"


def test_registry_self_registration():
    assert "read_file" in registry.names()
    assert "terminal" in registry.names()
    defs = registry.get_definitions()
    assert any(d["function"]["name"] == "write_file" for d in defs)


def test_compression_protects_last_and_keeps_tool_pairs():
    msgs = [{"role": "system", "content": "s"}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 200})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "y" * 200})
    assert needs_compression(msgs, context_window=2000)
    new, _ = compress(msgs, protect_last_n=4)
    assert new[0]["role"] == "system" and len(new) <= 4 + 2
    assert estimate_tokens(new) < estimate_tokens(msgs)
    assert new[-1]["content"].startswith("answer 29")


def test_session_search_and_memory_tool():
    script = [
        {"content": "", "tool_calls": [{"id": "c1", "name": "memory", "arguments": {"text": "likes chai"}}]},
        {"content": "saved", "tool_calls": []},
    ]
    agent = _looper(script)
    agent.chat("remember I like chai")
    assert "chai" in (agent.home / "MEMORY.md").read_text()
    assert agent.store.search("chai")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
