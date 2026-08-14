"""Unit tests for ensure_user_message_present (agent/message_sanitization.py).

Guarantees a genuine user turn survives in the outgoing request so strict chat
templates (Qwen3 raises "No user query found in messages") don't fail the call.
"""

from __future__ import annotations

from agent.message_sanitization import ensure_user_message_present


def _genuine_user_count(messages):
    return sum(
        1
        for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and not m["content"].startswith("<tool_response>")
    )


# ----- SHOULD inject -----

def test_injects_when_no_user_turn():
    msgs = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
        {"role": "assistant", "content": "Working on it."},
    ]
    assert ensure_user_message_present(msgs) is True
    # inserted right after the single leading system message
    assert msgs[1]["role"] == "user"
    assert msgs[0]["role"] == "system"
    assert _genuine_user_count(msgs) == 1


def test_tool_response_wrapped_user_does_not_count():
    # A user message that is ONLY a <tool_response> wrapper is not a real query
    # (mirrors the Qwen template's own test) -> still needs injection.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "<tool_response>ok</tool_response>"},
        {"role": "assistant", "content": "done"},
    ]
    assert ensure_user_message_present(msgs) is True
    assert _genuine_user_count(msgs) == 1


def test_inserts_after_multiple_system_messages():
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "assistant", "content": "hi"},
    ]
    ensure_user_message_present(msgs)
    assert [m["role"] for m in msgs[:3]] == ["system", "system", "user"]


# ----- should NOT inject -----

def test_noop_when_genuine_user_present():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Do the thing."},
        {"role": "assistant", "content": "ok"},
    ]
    before = len(msgs)
    assert ensure_user_message_present(msgs) is False
    assert len(msgs) == before


def test_noop_on_multimodal_user_content():
    # list-of-parts content (text + image) still counts as a genuine user turn.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "look at this"}, {"type": "image_url"}]},
    ]
    assert ensure_user_message_present(msgs) is False
