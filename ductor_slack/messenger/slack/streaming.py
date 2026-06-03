"""Native chat stream support for Slack with graceful fallback.

Rendering model
---------------
Each agent turn is rendered as one or more Slack *plan cards* (one streamed
message each, ``task_display_mode="plan"``). A card holds task rows:

* ``analyze`` — folds the streamed reasoning into its ``details`` (option A:
  thinking never reaches the message body, only the freshest tail window).
* one row per tool call — the full tool-call record, each with its own ✓.
* ``respond`` — drafting the final answer; the answer text streams into the
  body of the card that is active when the answer starts.

When a card fills up (``_MAX_TOOLS_PER_CARD`` tool rows, kept safely under
Slack's per-plan task cap) the next tool call rolls onto a fresh plan card —
so the complete history stays visible across multiple cards, never one card
per tool. A very long answer body (Slack caps ``markdown_text`` at 12k) rolls
onto a plain continuation message.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ductor_slack.cli.stream_events import ToolUseEvent
from ductor_slack.messenger.slack.sender import SlackSendOpts, send_rich
from ductor_slack.text.response_format import normalize_tool_name

logger = logging.getLogger(__name__)

_STREAM_BUFFER_SIZE = 64
_MAX_TOOL_HISTORY = 8
_PLAN_TITLE = "Working on your request"
_ANALYZE_TASK_ID = "analyze"
_ANALYZE_TASK_TITLE = "Understand request"
_RESPOND_TASK_ID = "respond"
_RESPOND_TASK_TITLE = "Draft response"
_TITLE_LIMIT = 140
_DETAIL_LIMIT = 280
# Slack rejects a streamed message once its rendered markdown grows past a hard
# length cap (`msg_too_long`); the SDK documents the limit as 12,000 chars. We
# roll to a fresh continuation message before reaching it.
_MAX_STREAM_CHARS = 11000
# Slack rejects a plan whose `tasks` array exceeds 50 entries. We paginate tool
# rows well under that (leaving room for analyze/respond) and roll to a new
# plan card when a card fills up.
_MAX_TOOLS_PER_CARD = 45
_CONTINUATION_HEADER = "_(continued)_\n\n"
# A single write is retried this many times across recoverable failures
# (length roll, dead-stream roll, rate-limit backoff) before giving up.
_MAX_STREAM_WRITE_ATTEMPTS = 3
# Recoverable by rolling onto a new stream message rather than degrading.
# - msg_too_long: current message is still alive; stop it cleanly, continue in a new one.
# - message_not_in_streaming_state: current message is already dead; discard and reopen.
_ROLLABLE_ERRORS = frozenset({"msg_too_long", "message_not_in_streaming_state"})
# Recoverable by backing off and retrying the same stream.
_RATELIMIT_ERRORS = frozenset({"ratelimited", "rate_limited"})
_SYSTEM_LABELS: dict[str, str] = {
    "thinking": "Thinking",
    "compacting": "Compacting",
    "recovering": "Recovering",
    "timeout_warning": "Timeout approaching",
    "timeout_extended": "Timeout extended",
}


@dataclass(slots=True)
class _ToolActivity:
    label: str
    target: str = ""


class SlackStreamEditor:
    """Render Slack output through the native chat streaming APIs."""

    def __init__(  # noqa: PLR0913
        self,
        client: Any,
        channel_id: str,
        *,
        thread_ts: str,
        recipient_user_id: str | None = None,
        recipient_team_id: str | None = None,
        edit_interval_seconds: float = 1.0,
    ) -> None:
        del edit_interval_seconds
        self._client = client
        self._channel_id = channel_id
        self._thread_ts = thread_ts
        self._recipient_user_id = recipient_user_id
        self._recipient_team_id = recipient_team_id
        self._stream: Any | None = None
        self._native_failed = False
        self._thinking_parts: list[str] = []
        self._answer_parts: list[str] = []
        self._status: str | None = None
        self._answer_started = False
        self._answer_phase = False
        self._tool_history: list[_ToolActivity] = []
        self._tool_count = 0
        self._used_tools = False
        # --- plan-card state ---
        self._card_started = False  # card 0 (analyze) has been emitted
        self._card_is_plan = True  # current card renders a plan block (vs plain text)
        self._card_plan_emitted = False  # plan_update header sent on the current card
        self._analyze_done = False
        self._analyze_detail_sent = ""
        self._respond_started = False
        self._tool_seq = 0  # global, for stable per-tool task ids
        self._tools_on_card = 0
        self._active_tool_id: str | None = None
        self._active_tool_title = ""
        # --- stream length / rolling state ---
        self._stream_chars = 0  # markdown chars written to the current message
        self._rolled = False
        self._pending_continuation_header = False

    # ------------------------------------------------------------------ events

    async def on_thinking(self, text: str) -> None:
        """Record reasoning and surface it in the analyze task, not the message body."""
        if not text.strip():
            return
        self._thinking_parts.append(text)
        self._status = "thinking"
        await self._ensure_card0()
        if self._analyze_done:
            return
        detail = self._analysis_detail()
        if detail == self._analyze_detail_sent:
            return
        self._analyze_detail_sent = detail
        await self._emit([self._task(_ANALYZE_TASK_ID, _ANALYZE_TASK_TITLE, "in_progress", detail)])

    async def on_tool(self, tool: ToolUseEvent | str) -> None:
        """Render each tool call as its own task row (paginating across cards)."""
        activity = self._tool_activity(tool)
        self._tool_history.append(activity)
        self._tool_history = self._tool_history[-_MAX_TOOL_HISTORY:]
        self._tool_count += 1
        self._used_tools = True
        await self._ensure_card0()

        batch: list[dict[str, object]] = []
        if not self._analyze_done:
            self._analyze_done = True
            batch.append(self._task(_ANALYZE_TASK_ID, _ANALYZE_TASK_TITLE, "complete", self._analysis_detail()))
        if self._active_tool_id is not None:
            batch.append(self._task(self._active_tool_id, self._active_tool_title, "complete"))
            self._active_tool_id = None

        # Paginate: a full card flushes its pending completions, then rolls.
        if self._tools_on_card >= _MAX_TOOLS_PER_CARD:
            await self._emit(batch)
            batch = []
            await self._open_new_card()

        self._tool_seq += 1
        task_id = f"tool-{self._tool_seq}"
        title = self._tool_title(activity)
        self._active_tool_id = task_id
        self._active_tool_title = title
        self._tools_on_card += 1
        batch.append(self._task(task_id, title, "in_progress"))
        await self._emit(batch)

    async def on_delta(self, text: str) -> None:
        """Stream assistant answer text into the message body."""
        if not text:
            return
        await self._ensure_card0()
        self._answer_phase = True
        batch: list[dict[str, object]] = []
        if not self._analyze_done:
            self._analyze_done = True
            batch.append(self._task(_ANALYZE_TASK_ID, _ANALYZE_TASK_TITLE, "complete", self._analysis_detail()))
        if self._active_tool_id is not None:
            batch.append(self._task(self._active_tool_id, self._active_tool_title, "complete"))
            self._active_tool_id = None
        if not self._respond_started:
            self._respond_started = True
            batch.append(self._task(_RESPOND_TASK_ID, _RESPOND_TASK_TITLE, "in_progress"))
        await self._emit(batch)

        self._answer_parts.append(text)
        self._answer_started = True
        await self._append_markdown(text)

    async def on_system(self, status: str | None) -> None:
        """Track transient system status."""
        self._status = status
        if status is None or status == "thinking":
            return
        label = _SYSTEM_LABELS.get(status)
        if label:
            await self._append_markdown(f"\n\n_{label}_")

    async def finalize(self, final_text: str | None) -> None:
        """Finalize the stream, falling back to a single rich message if needed."""
        if final_text and not "".join(self._answer_parts).strip():
            self._answer_parts = [final_text]
        if self._native_failed:
            await self._send_fallback(final_text)
            return

        stop_text = None
        if final_text and not self._answer_started:
            stop_text = final_text
        if stop_text and self._pending_continuation_header:
            stop_text = _CONTINUATION_HEADER + stop_text
            self._pending_continuation_header = False
        stop_chunks = self._finalize_plan_chunks(bool(final_text))
        try:
            stream = await self._ensure_stream()
            await stream.stop(markdown_text=stop_text, chunks=stop_chunks)
        except Exception as exc:
            self._mark_native_failure(exc)
            await self._send_fallback(final_text)

    # ------------------------------------------------------------- plan chunks

    async def _ensure_card0(self) -> None:
        """Emit the first plan card (analyze in progress) on first activity."""
        if self._card_started:
            return
        self._card_started = True
        self._analyze_detail_sent = self._analysis_detail()
        await self._emit(
            [self._task(_ANALYZE_TASK_ID, _ANALYZE_TASK_TITLE, "in_progress", self._analyze_detail_sent)]
        )

    async def _open_new_card(self) -> None:
        """Roll onto a fresh plan card to continue the tool-call record."""
        await self._roll_stream(stop_current=True)

    async def _emit(self, chunks: list[dict[str, object]]) -> None:
        """Send plan/task chunks to the current card, prepending the plan header once."""
        if not chunks or not self._card_is_plan:
            return
        out: list[dict[str, object]] = []
        if not self._card_plan_emitted:
            out.append({"type": "plan_update", "title": _PLAN_TITLE})
            self._card_plan_emitted = True
        out.extend(chunks)
        await self._append_chunks(out)

    def _finalize_plan_chunks(self, has_final_text: bool) -> list[dict[str, object]] | None:
        """Chunks to mark the current card's tasks complete on stop()."""
        if not self._card_started or not self._card_is_plan:
            return None
        chunks: list[dict[str, object]] = []
        if not self._card_plan_emitted:
            self._card_plan_emitted = True
            chunks.append({"type": "plan_update", "title": _PLAN_TITLE})
        if not self._analyze_done:
            self._analyze_done = True
            chunks.append(self._task(_ANALYZE_TASK_ID, _ANALYZE_TASK_TITLE, "complete", self._analysis_detail()))
        if self._active_tool_id is not None:
            chunks.append(self._task(self._active_tool_id, self._active_tool_title, "complete"))
            self._active_tool_id = None
        if self._answer_started or self._respond_started or has_final_text:
            chunks.append(self._task(_RESPOND_TASK_ID, _RESPOND_TASK_TITLE, "complete"))
        return chunks or None

    def _task(
        self,
        task_id: str,
        title: str,
        status: str,
        details: str | None = None,
    ) -> dict[str, object]:
        chunk: dict[str, object] = {
            "type": "task_update",
            "id": task_id,
            "title": self._limit_title(title),
            "status": status,
        }
        if details:
            chunk["details"] = self._limit_detail(details)
        return chunk

    def _tool_title(self, activity: _ToolActivity) -> str:
        if activity.target:
            return f"{activity.label}: {activity.target}"
        return activity.label

    # --------------------------------------------------------------- transport

    async def _append_markdown(self, text: str) -> None:
        if not text or self._native_failed:
            return
        await self._write(markdown_text=text, account_len=len(text))

    async def _append_chunks(self, chunks: list[dict[str, object]]) -> None:
        if not chunks or self._native_failed:
            return
        await self._write(chunks=chunks)

    async def _write(
        self,
        *,
        markdown_text: str | None = None,
        chunks: list[dict[str, object]] | None = None,
        account_len: int = 0,
    ) -> None:
        """Append to the native stream, rolling or backing off on recoverable errors."""
        for attempt in range(_MAX_STREAM_WRITE_ATTEMPTS):
            # Proactively roll before we would exceed Slack's per-message cap.
            if (
                account_len
                and self._stream is not None
                and self._stream_chars + account_len > _MAX_STREAM_CHARS
            ):
                await self._roll_stream(stop_current=True)
            try:
                stream = await self._ensure_stream()
                if markdown_text is not None:
                    payload = markdown_text
                    if self._pending_continuation_header:
                        payload = _CONTINUATION_HEADER + markdown_text
                    await stream.append(markdown_text=payload)
                    self._pending_continuation_header = False
                if chunks is not None:
                    await stream.append(chunks=chunks)
            except Exception as exc:
                if not await self._handle_write_error(exc, attempt):
                    return
            else:
                self._stream_chars += account_len
                return
        self._mark_native_failure(RuntimeError("Slack stream write retries exhausted"))

    async def _handle_write_error(self, exc: Exception, attempt: int) -> bool:
        """Return True if the caller should retry the write, False if handled terminally."""
        code = self._slack_error_code(exc)
        is_last = attempt >= _MAX_STREAM_WRITE_ATTEMPTS - 1
        if code in _ROLLABLE_ERRORS and not is_last:
            # msg_too_long → message is still alive, stop it cleanly before reopening.
            # message_not_in_streaming_state → message is dead, discard without stopping.
            await self._roll_stream(stop_current=(code == "msg_too_long"))
            logger.info("Slack stream rolled after %s (attempt %d)", code, attempt + 1)
            return True
        if code in _RATELIMIT_ERRORS and not is_last:
            await asyncio.sleep(self._retry_after(exc))
            logger.info("Slack stream rate-limited; backed off (attempt %d)", attempt + 1)
            return True
        self._mark_native_failure(exc)
        return False

    async def _roll_stream(self, *, stop_current: bool) -> None:
        """Finalize/discard the current stream and arm a fresh continuation card.

        During the answer phase the continuation is a plain text message; during
        the analyze/tool phase it is a fresh plan card so the tool record keeps
        rendering across cards.
        """
        old = self._stream
        self._stream = None
        self._stream_chars = 0
        self._rolled = True
        self._tools_on_card = 0
        self._card_plan_emitted = False
        self._card_is_plan = not self._answer_phase
        self._pending_continuation_header = self._answer_phase
        if old is not None and stop_current:
            try:
                await old.stop()
            except Exception as exc:
                logger.debug("Ignored error stopping rolled Slack stream: %r", exc)

    @staticmethod
    def _slack_error_code(exc: Exception) -> str:
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                err = resp.get("error")
            except Exception:
                err = None
            if err:
                return str(err)
        text = str(exc)
        for code in (*_ROLLABLE_ERRORS, *_RATELIMIT_ERRORS):
            if code in text:
                return code
        return ""

    @staticmethod
    def _retry_after(exc: Exception) -> float:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        if isinstance(headers, dict):
            value = headers.get("Retry-After") or headers.get("retry-after")
            if value:
                try:
                    return min(float(value), 30.0)
                except (TypeError, ValueError):
                    pass
        return 1.0

    async def _ensure_stream(self) -> Any:
        if self._stream is not None:
            return self._stream
        kwargs: dict[str, object] = {
            "channel": self._channel_id,
            "thread_ts": self._thread_ts,
            "buffer_size": _STREAM_BUFFER_SIZE,
        }
        # Use plan mode only for cards that actually carry a plan block; plain
        # continuation messages (long-answer overflow) and no-activity turns omit it.
        if self._card_is_plan and self._card_started:
            kwargs["task_display_mode"] = "plan"
        if self._recipient_team_id is not None:
            kwargs["recipient_team_id"] = self._recipient_team_id
        if self._recipient_user_id is not None:
            kwargs["recipient_user_id"] = self._recipient_user_id
        self._stream = await self._client.chat_stream(**kwargs)
        return self._stream

    def _mark_native_failure(self, exc: Exception) -> None:
        if self._native_failed:
            return
        self._native_failed = True
        logger.warning("Slack native stream failed; falling back to plain reply: %r", exc)

    # ---------------------------------------------------------------- detail/text

    def _analysis_detail(self) -> str:
        joined = " ".join("".join(self._thinking_parts).split())
        if not joined:
            return "Understanding the request"
        return self._limit_detail_tail(joined)

    def _tool_activity(self, tool: ToolUseEvent | str) -> _ToolActivity:
        label = normalize_tool_name(str(getattr(tool, "tool_name", tool)))
        parameters = getattr(tool, "parameters", None)
        return _ToolActivity(label=label, target=self._tool_target(parameters))

    def _tool_target(self, parameters: dict[str, Any] | None) -> str:
        if not parameters:
            return ""
        return self._limit_detail(
            self._string_param(parameters, ("url", "uri"), transform=self._compact_url)
            or self._string_param(parameters, ("query", "q", "search", "pattern"))
            or self._string_param(parameters, ("path", "file_path", "file", "filepath", "directory", "dir"))
            or self._string_param(parameters, ("cmd", "command"))
            or self._list_param(parameters, ("urls", "paths"))
        )

    def _string_param(
        self,
        parameters: dict[str, Any],
        keys: tuple[str, ...],
        *,
        transform: Any = None,
    ) -> str:
        for key in keys:
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = " ".join(value.split())
                return transform(cleaned) if callable(transform) else cleaned
        return ""

    def _list_param(self, parameters: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = parameters.get(key)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    suffix = f" (+{len(value) - 1} more)" if len(value) > 1 else ""
                    return f"{first.strip()}{suffix}"
        return ""

    def _compact_url(self, url: str) -> str:
        compact = url.replace("https://", "").replace("http://", "")
        return compact.rstrip("/")

    def _tool_details(self) -> str:
        collapsed: list[tuple[_ToolActivity, int]] = []
        for activity in self._tool_history:
            if collapsed and collapsed[-1][0] == activity:
                previous, count = collapsed[-1]
                collapsed[-1] = (previous, count + 1)
            else:
                collapsed.append((activity, 1))
        lines = [
            self._tool_detail_line(activity, count)
            for activity, count in collapsed[-_MAX_TOOL_HISTORY:]
        ]
        details = "\n".join(lines)
        return self._limit_detail(details)

    def _tool_detail_line(self, activity: _ToolActivity, count: int) -> str:
        label = f"{activity.label} x{count}" if count > 1 else activity.label
        if activity.target:
            return f"- {label}: {activity.target}"
        return f"- {label}"

    def _limit_title(self, text: str) -> str:
        one_line = " ".join(text.split())
        if len(one_line) <= _TITLE_LIMIT:
            return one_line
        return one_line[: _TITLE_LIMIT - 1].rstrip() + "…"

    def _limit_detail(self, text: str) -> str:
        if len(text) <= _DETAIL_LIMIT:
            return text
        return text[: _DETAIL_LIMIT - 1].rstrip() + "…"

    def _limit_detail_tail(self, text: str) -> str:
        """Keep the freshest tail of `text` within the detail cap (rolling window)."""
        if len(text) <= _DETAIL_LIMIT:
            return text
        return "…" + text[-(_DETAIL_LIMIT - 1) :].lstrip()

    # ------------------------------------------------------------------ fallback

    async def _send_fallback(self, final_text: str | None) -> None:
        rendered = self._render_fallback(final_text)
        if rendered:
            await send_rich(
                self._client,
                self._channel_id,
                rendered,
                SlackSendOpts(thread_ts=self._thread_ts),
            )

    def _render_fallback(self, final_text: str | None) -> str:
        sections: list[str] = []
        thinking = "".join(self._thinking_parts).strip()
        answer = "".join(self._answer_parts).strip() or (final_text or "").strip()

        if thinking:
            sections.append(f"💭 *Thinking*\n{thinking}")
        elif self._status is not None:
            label = _SYSTEM_LABELS.get(self._status or "")
            if label:
                sections.append(f"💭 *{label}*")

        if self._used_tools:
            summary = self._tool_summary()
            if summary:
                sections.append(summary)

        if answer:
            sections.append(answer)

        if not sections:
            sections.append("…")
        return "\n\n".join(sections).strip()

    def _tool_summary(self) -> str:
        details = self._tool_details()
        if not details:
            return ""
        header = f"🔧 *Tools used* ({self._tool_count})" if self._tool_count else "🔧 *Tools used*"
        return f"{header}\n{details}"
