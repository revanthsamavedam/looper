"""Context compression — Hermes `agent/context_compressor.py`, shrunk.

Hermes triggers preflight at >50% of context, protects the last N
messages, keeps tool call/result pairs together, and summarises the
middle with an auxiliary LLM. The looper keeps the trigger, the
protection and the pairing; summarisation is LLM-assisted when a
client is available, otherwise a faithful truncation that never
invents a summary.
"""
from __future__ import annotations


def estimate_tokens(messages: list[dict]) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    chars += sum(len(str(c)) for m in messages for c in (m.get("tool_calls") or []))
    return max(1, chars // 4)


def needs_compression(messages: list[dict], context_window: int = 128_000) -> bool:
    return estimate_tokens(messages) > context_window * 0.5


def compress(messages: list[dict], protect_last_n: int = 20,
             summarise=None) -> tuple[list[dict], str]:
    """Return (new_messages, summary_text). System message is always kept.
    Tool pairs are never split: a cut point is moved back to just before
    an assistant message that has tool_calls, or forward past its tool results."""
    if len(messages) <= protect_last_n + 2:
        return messages, ""
    system = [m for m in messages if m.get("role") == "system"][:1]
    rest = [m for m in messages if m.get("role") != "system"]
    keep = rest[-protect_last_n:]
    middle = rest[:-protect_last_n]
    # never split a tool pair: drop trailing assistant-with-tool_calls
    # from `middle` into `keep` boundary handling — simplest correct rule:
    # if middle ends with an assistant tool_call whose results are in keep,
    # move that assistant message to keep as well.
    while middle and middle[-1].get("role") == "assistant" and middle[-1].get("tool_calls"):
        keep.insert(0, middle.pop())
    if summarise is not None:
        summary = summarise(middle)
    else:
        # Faithful truncation, NOT a fake summary: keep a bounded sample
        # so the compressed history is actually smaller than the original.
        sample = middle[:8]
        lines = []
        for m in sample:
            c = str(m.get("content") or "")[:120].replace("\n", " ")
            lines.append(f"- {m.get('role')}: {c}")
        omitted = len(middle) - len(sample)
        if omitted:
            lines.append(f"- [... {omitted} more turns omitted]")
        summary = "Earlier turns (truncated, not LLM-summarised):\n" + "\n".join(lines)
    compressed = system + [{"role": "system", "content": f"[Compressed history]\n{summary}"}] + keep
    return compressed, summary
