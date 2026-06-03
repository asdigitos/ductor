"""Tests for SlackStreamEditor's rolling/recovery behavior and clean fallback."""

from __future__ import annotations

from typing import Any

from ductor_slack.cli.stream_events import ToolUseEvent
from ductor_slack.messenger.slack import streaming
from ductor_slack.messenger.slack.streaming import SlackStreamEditor, _ToolActivity


def _answer_phase(editor: SlackStreamEditor) -> SlackStreamEditor:
    """Put the editor in the answer phase, where markdown flows into the body."""
    editor._card_started = True
    editor._answer_phase = True
    return editor


class FakeSlackError(Exception):
    """Mimics slack_sdk.SlackApiError enough for _slack_error_code/_retry_after."""

    def __init__(self, code: str) -> None:
        super().__init__(f"slack api error: {code}")
        self.response = {"error": code}


class FakeStream:
    """Records appends/stop. `raises` is consumed one entry per append call."""

    def __init__(self, raises: list[Exception | None] | None = None) -> None:
        self.appends: list[tuple[str, Any]] = []
        self.stopped = False
        self.stop_args: tuple[Any, Any] | None = None
        self._raises = list(raises or [])

    async def append(self, *, markdown_text: str | None = None, chunks: Any = None) -> None:
        if self._raises:
            exc = self._raises.pop(0)
            if exc is not None:
                raise exc
        if markdown_text is not None:
            self.appends.append(("md", markdown_text))
        if chunks is not None:
            self.appends.append(("chunks", chunks))

    async def stop(self, *, markdown_text: str | None = None, chunks: Any = None) -> None:
        self.stopped = True
        self.stop_args = (markdown_text, chunks)

    def md_text(self) -> str:
        return "".join(t for kind, t in self.appends if kind == "md")


class FakeClient:
    def __init__(self, streams: list[FakeStream]) -> None:
        self._streams = list(streams)
        self.kwargs: list[dict[str, Any]] = []

    async def chat_stream(self, **kwargs: Any) -> FakeStream:
        self.kwargs.append(kwargs)
        return self._streams.pop(0)


def _editor(client: FakeClient) -> SlackStreamEditor:
    return SlackStreamEditor(client, "C123", thread_ts="1.1")


async def test_msg_too_long_rolls_to_new_message_with_continuation_header() -> None:
    s0 = FakeStream(raises=[FakeSlackError("msg_too_long")])
    s1 = FakeStream()
    client = FakeClient([s0, s1])
    editor = _answer_phase(_editor(client))

    await editor._append_markdown("hello world")

    # First (alive) message is stopped cleanly, text continues on a fresh stream.
    assert s0.stopped is True
    assert s1.md_text() == f"{streaming._CONTINUATION_HEADER}hello world"
    assert editor._rolled is True
    # Continuation stream must NOT carry a plan card (option A).
    assert "task_display_mode" not in client.kwargs[1]
    assert client.kwargs[0].get("task_display_mode") == "plan"


async def test_message_not_in_streaming_state_discards_without_stopping() -> None:
    s0 = FakeStream(raises=[FakeSlackError("message_not_in_streaming_state")])
    s1 = FakeStream()
    client = FakeClient([s0, s1])
    editor = _answer_phase(_editor(client))

    await editor._append_markdown("resume here")

    # Dead stream is discarded, never stopped.
    assert s0.stopped is False
    assert s1.md_text().endswith("resume here")
    assert editor._rolled is True


async def test_proactive_roll_before_exceeding_length_cap(monkeypatch: Any) -> None:
    monkeypatch.setattr(streaming, "_MAX_STREAM_CHARS", 20)
    s0 = FakeStream()
    s1 = FakeStream()
    client = FakeClient([s0, s1])
    editor = _answer_phase(_editor(client))

    await editor._append_markdown("a" * 15)  # fits on s0
    await editor._append_markdown("b" * 15)  # would exceed cap -> roll to s1

    assert s0.md_text() == "a" * 15
    assert s0.stopped is True
    assert s1.md_text() == f"{streaming._CONTINUATION_HEADER}{'b' * 15}"


async def test_rate_limit_backs_off_and_retries_same_stream(monkeypatch: Any) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)
    s0 = FakeStream(raises=[FakeSlackError("ratelimited")])
    client = FakeClient([s0])
    editor = _answer_phase(_editor(client))

    await editor._append_markdown("payload")

    assert slept, "expected a backoff sleep"
    assert s0.md_text() == "payload"  # retried on the same stream
    assert editor._rolled is False


async def test_thinking_goes_to_plan_card_not_body() -> None:
    s0 = FakeStream()
    client = FakeClient([s0])
    editor = _editor(client)

    await editor.on_thinking("reasoning about the problem")

    # Thinking must never hit the message body (option A) — only plan/task chunks.
    assert s0.md_text() == ""
    assert all(kind == "chunks" for kind, _ in s0.appends)
    flat = [c for kind, payload in s0.appends if kind == "chunks" for c in payload]
    analyze = [c for c in flat if c.get("id") == "analyze"]
    assert analyze, "expected an analyze task chunk"
    assert "reasoning about the problem" in analyze[-1].get("details", "")


async def test_answer_streams_to_body() -> None:
    s0 = FakeStream()
    client = FakeClient([s0])
    editor = _editor(client)

    await editor.on_delta("the final answer")

    assert s0.md_text() == "the final answer"


async def test_long_thinking_rolls_into_detail_tail_not_many_messages() -> None:
    s0 = FakeStream()
    client = FakeClient([s0])
    editor = _editor(client)

    # Feed far more thinking than the detail cap; it must stay one message,
    # with the card detail showing only the freshest tail window.
    for i in range(50):
        await editor.on_thinking(f"thinking step {i} " + "x" * 50)

    assert s0.md_text() == ""  # nothing in the body
    assert len(client.kwargs) == 1  # no rolling, single message
    flat = [c for kind, payload in s0.appends if kind == "chunks" for c in payload]
    analyze = [c for c in flat if c.get("id") == "analyze"]
    detail = analyze[-1].get("details", "")
    assert len(detail) <= streaming._DETAIL_LIMIT
    assert "step 49" in detail  # freshest content retained


async def test_tool_rows_paginate_to_new_cards(monkeypatch: Any) -> None:
    monkeypatch.setattr(streaming, "_MAX_TOOLS_PER_CARD", 3)
    streams = [FakeStream() for _ in range(5)]
    client = FakeClient(streams)
    editor = _editor(client)

    for i in range(7):
        await editor.on_tool(
            ToolUseEvent(type="assistant", tool_name="Shell", parameters={"command": f"cmd {i}"})
        )

    # 3 tool rows per card → card0: tools 1-3, card1: 4-6, card2: 7  → 3 plan cards.
    assert len(client.kwargs) == 3
    assert all(kw.get("task_display_mode") == "plan" for kw in client.kwargs)
    # Each card after the first reopens with its own plan header.
    plan_updates = [
        c
        for s in streams
        for kind, payload in s.appends
        if kind == "chunks"
        for c in payload
        if c["type"] == "plan_update"
    ]
    assert len(plan_updates) == 3

    # Every tool call is its own row with a stable, sequential id — full record.
    in_progress = [
        c
        for s in streams
        for kind, payload in s.appends
        if kind == "chunks"
        for c in payload
        if str(c.get("id", "")).startswith("tool-") and c["status"] == "in_progress"
    ]
    assert [c["id"] for c in in_progress] == [f"tool-{n}" for n in range(1, 8)]
    # Earlier cards were finalized as we paginated; the current one stays open.
    assert streams[0].stopped is True
    assert streams[1].stopped is True
    assert streams[2].stopped is False


async def test_non_recoverable_error_degrades_to_fallback() -> None:
    s0 = FakeStream(raises=[FakeSlackError("invalid_blocks")])
    client = FakeClient([s0])
    editor = _editor(client)

    await editor._append_markdown("oops")

    assert editor._native_failed is True


async def test_fallback_renders_clean_tool_summary_without_raw_markers(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_send_rich(_client: Any, _channel: str, text: str, _opts: Any) -> None:
        captured["text"] = text

    monkeypatch.setattr(streaming, "send_rich", fake_send_rich)

    editor = _editor(FakeClient([]))
    editor._native_failed = True
    editor._thinking_parts = ["Let me check the docs."]
    editor._used_tools = True
    editor._tool_count = 3
    editor._tool_history = [
        _ToolActivity(label="WebSearch", target="example.com"),
        _ToolActivity(label="Read", target="/tmp/x.py"),
    ]

    await editor.finalize("Here is the answer.")

    text = captured["text"]
    assert "[TOOL:" not in text  # no leaked raw markers
    assert "🔧 *Tools used* (3)" in text
    assert "WebSearch: example.com" in text
    assert "Here is the answer." in text
