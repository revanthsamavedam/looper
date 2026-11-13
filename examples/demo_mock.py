"""Demo with no API key: the looper drives a scripted 'model'.

Run:  python examples/demo_mock.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_looper import ScriptedClient  # noqa: E402
from looper import Looper, SessionStore  # noqa: E402

script = [
    {"content": "", "tool_calls": [
        {"id": "c1", "name": "write_file",
         "arguments": {"path": "/tmp/looper-demo.txt", "content": "Looper was here\n"}}]},
    {"content": "Wrote /tmp/looper-demo.txt — verified by the write_file result.", "tool_calls": []},
]
agent = Looper(client=ScriptedClient(script), store=SessionStore(":memory:"),
               auto_approve=True)
print(agent.chat("write a file saying looper was here"))
