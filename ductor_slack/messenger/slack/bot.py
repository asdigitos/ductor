"""Slack transport bot using Socket Mode."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ductor_slack.bus.bus import MessageBus
from ductor_slack.bus.lock_pool import LockPool
from ductor_slack.commands import BOT_COMMANDS, MULTIAGENT_SUB_COMMANDS
from ductor_slack.config import AgentConfig
from ductor_slack.files.allowed_roots import resolve_allowed_roots
from ductor_slack.i18n import t
from ductor_slack.infra.version import get_current_version
from ductor_slack.messenger.commands import classify_command
from ductor_slack.messenger.notifications import NotificationService
from ductor_slack.messenger.slack.id_map import SlackIdMap
from ductor_slack.messenger.slack.sender import (
    SlackSendOpts,
    send_rich,
)
from ductor_slack.messenger.slack.sender import (
    add_reaction as slack_add_reaction,
)
from ductor_slack.messenger.slack.sender import (
    remove_reaction as slack_remove_reaction,
)
from ductor_slack.session.key import SessionKey
from ductor_slack.text.response_format import SEP, fmt

if TYPE_CHECKING:
    from ductor_slack.infra.updater import UpdateObserver
    from ductor_slack.multiagent.bus import AsyncInterAgentResult
    from ductor_slack.orchestrator.core import Orchestrator
    from ductor_slack.tasks.models import TaskResult
    from ductor_slack.workspace.paths import DuctorPaths

_SlackSocketModeHandler: Any
_SlackAsyncApp: Any

try:
    from slack_bolt.adapter.socket_mode.async_handler import (
        AsyncSocketModeHandler as _SlackSocketModeHandler,
    )
    from slack_bolt.async_app import AsyncApp as _SlackAsyncApp
    from slack_sdk.errors import SlackApiError as _SlackApiError

    _SLACK_AVAILABLE = True
except ImportError:  # pragma: no cover - import fallback
    _SLACK_AVAILABLE = False
    _SlackSocketModeHandler = object
    _SlackAsyncApp = object
    _SlackApiError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_DEFAULT_MENTIONED_THREAD_TTL_SECONDS = 3600.0
_DEFAULT_MENTIONED_THREAD_MAX_SIZE = 200
_DEFAULT_THREAD_CONTEXT_CACHE_MAX_SIZE = 200
_DEFAULT_RECENT_EVENT_TTL_SECONDS = 120.0
_DEFAULT_RECENT_EVENT_MAX_SIZE = 500
_MENTION_ORDER_DELAY_SECONDS = 0.2
_PEER_EDIT_DEBOUNCE_SECONDS = 1.0
_DEFAULT_PEER_TURN_BUDGET = 2
_MAX_PEER_TURN_BUDGET = 50
_PEER_TURN_BUDGET_CACHE_MAX = 200
_PEER_TURN_BUDGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:max[\s_-]*turns?|at\s*most|no\s*more\s*than|up\s*to|limit(?:\s*of)?)"
        r"\D{0,8}(\d+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d+)\s*(?:turns?|rounds?|exchanges?|replies?)\b", re.IGNORECASE),
    re.compile(r"(?:最多|不超过|不多于|不得超过|限定?|上限)\D{0,4}(\d+)"),
    re.compile(r"(\d+)\s*(?:轮|次|回合)"),
)
_MESSAGE_COMMANDS_WITH_ARGS = frozenset(
    {"agent_restart", "agent_start", "agent_stop", "model", "session", "showfiles"}
)
_SLACK_MENTION_RE = re.compile(r"<@([A-Z0-9]+)>")


@dataclass(slots=True)
class _ThreadContextCache:
    """Cache entry for fetched Slack thread context."""

    content: str
    fetched_at: float = field(default_factory=time.monotonic)
    message_count: int = 0


def _restart_marker_path(ductor_home: str) -> Path:
    """Return the restart marker path."""
    return Path(ductor_home).expanduser() / "restart-requested"


def _slack_response_data(response: object) -> dict[str, Any]:
    """Return the dict body of a Slack SDK response.

    `slack_sdk.web.async_slack_response.AsyncSlackResponse` is dict-like but
    not a `dict` instance, so plain `isinstance(response, dict)` is False and
    silently drops the payload. Fall back to its `.data` attribute.
    """
    if isinstance(response, dict):
        return response
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return data
    return {}


def _slack_ts_is_at_or_after(candidate_ts: str, current_ts: str) -> bool:
    """Return whether *candidate_ts* is the current Slack message or later."""
    try:
        return Decimal(candidate_ts) >= Decimal(current_ts)
    except (InvalidOperation, ValueError):
        return candidate_ts >= current_ts


class SlackNotificationService:
    """Notification service implementation for Slack."""

    def __init__(self, bot: SlackBot) -> None:
        self._bot = bot

    async def notify(self, chat_id: int, text: str) -> None:
        channel_id = self._bot.id_map.int_to_channel(chat_id)
        if channel_id:
            await self._bot._send_rich(channel_id, text)
            return
        logger.warning("notify: cannot resolve chat_id=%d to Slack channel, falling back", chat_id)
        await self.notify_all(text)

    async def notify_all(self, text: str) -> None:
        await self._bot.broadcast(text)


class SlackBot:
    """Slack bot implementing ``BotProtocol``."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        agent_name: str = "main",
        bus: MessageBus | None = None,
        lock_pool: LockPool | None = None,
    ) -> None:
        if not _SLACK_AVAILABLE:
            raise ImportError(
                "slack-bolt is required for Slack transport. "
                "Install with: pip install 'ductor-slack[slack]'"
            ) from None

        self._config = config
        self._agent_name = agent_name
        self._store_path = Path(config.ductor_home).expanduser() / "slack_store"
        self._store_path.mkdir(parents=True, exist_ok=True)

        self._app: Any = _SlackAsyncApp(token=config.slack.bot_token)
        self._socket_handler: Any | None = None
        self._socket_task: asyncio.Task[None] | None = None
        self._lock_pool = lock_pool or LockPool()
        self._bus = bus or MessageBus(lock_pool=self._lock_pool)
        self._id_map = SlackIdMap(self._store_path)

        from ductor_slack.messenger.slack.transport import SlackTransport

        self._bus.register_transport(SlackTransport(self))
        self._orchestrator: Orchestrator | None = None
        self._startup_hooks: list[Callable[[], Awaitable[None]]] = []
        self._notification_service: NotificationService = SlackNotificationService(self)
        self._abort_all_callback: Callable[[], Awaitable[int]] | None = None
        self._exit_code = 0
        self._update_observer: UpdateObserver | None = None
        self._restart_watcher: asyncio.Task[None] | None = None
        self._bot_user_id = ""
        self._bot_id = ""
        self._bot_name = "slack-bot"
        self._team_id = ""
        self._last_active_channel: str | None = None
        self._mentioned_threads: dict[tuple[str, str], float] = {}
        self._recent_events: dict[tuple[str, str], float] = {}
        self._pending_peer_edit_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._processed_peer_edit_signatures: dict[tuple[str, str], str] = {}
        self._peer_turn_budget_cache: dict[tuple[str, str, str], int] = {}
        self._user_name_cache: dict[str, str] = {}
        self._thread_context_cache: dict[str, _ThreadContextCache] = {}
        self._MENTIONED_THREAD_TTL = _DEFAULT_MENTIONED_THREAD_TTL_SECONDS
        self._MENTIONED_THREAD_MAX_SIZE = _DEFAULT_MENTIONED_THREAD_MAX_SIZE
        self._RECENT_EVENT_TTL = _DEFAULT_RECENT_EVENT_TTL_SECONDS
        self._RECENT_EVENT_MAX_SIZE = _DEFAULT_RECENT_EVENT_MAX_SIZE
        self._THREAD_CACHE_TTL = 60.0
        self._THREAD_CONTEXT_CACHE_MAX_SIZE = _DEFAULT_THREAD_CONTEXT_CACHE_MAX_SIZE

        self._register_handlers()

    @property
    def orchestrator(self) -> Orchestrator | None:
        return self._orchestrator

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def notification_service(self) -> NotificationService:
        return self._notification_service

    @property
    def id_map(self) -> SlackIdMap:
        return self._id_map

    @property
    def client(self) -> Any:
        return self._app.client

    @property
    def bot_name(self) -> str:
        return self._bot_name

    def register_startup_hook(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._startup_hooks.append(hook)

    def set_abort_all_callback(self, callback: Callable[[], Awaitable[int]]) -> None:
        self._abort_all_callback = callback

    def file_roots(self, paths: DuctorPaths) -> list[Path] | None:
        return resolve_allowed_roots(self._config.file_access, paths.workspace)

    async def run(self) -> int:
        auth = await self.client.auth_test()
        self._bot_user_id = str(auth.get("user_id", ""))
        self._bot_id = str(auth.get("bot_id", ""))
        self._bot_name = str(auth.get("user", "slack-bot"))
        self._team_id = str(auth.get("team_id", ""))

        from ductor_slack.messenger.slack.startup import run_slack_startup

        await run_slack_startup(self)

        self._restart_watcher = asyncio.create_task(self._watch_restart_marker())
        self._socket_handler = _SlackSocketModeHandler(self._app, self._config.slack.app_token)
        handler: Any = self._socket_handler
        self._socket_task = asyncio.create_task(handler.start_async())
        try:
            await self._socket_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Slack Socket Mode exited with error, requesting restart")
            from ductor_slack.infra.restart import EXIT_RESTART

            self._exit_code = EXIT_RESTART
        return self._exit_code

    async def shutdown(self) -> None:
        pending_peer_edit_tasks = list(self._pending_peer_edit_tasks.values())
        for task in pending_peer_edit_tasks:
            task.cancel()
        for task in pending_peer_edit_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_peer_edit_tasks.clear()

        if self._restart_watcher:
            self._restart_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._restart_watcher

        if self._socket_handler:
            handler: Any = self._socket_handler
            with contextlib.suppress(Exception):
                await handler.close_async()
        if self._socket_task and not self._socket_task.done():
            self._socket_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._socket_task

        if self._update_observer:
            await self._update_observer.stop()
        if self._orchestrator:
            await self._orchestrator.shutdown()

        logger.info("SlackBot shut down")

    def _register_handlers(self) -> None:
        self._app.event("message")(self._handle_message_event)
        self._app.event("app_mention")(self._handle_mention_event)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        await self._on_message(event)

    async def _handle_mention_event(self, event: dict[str, Any]) -> None:
        await self._on_message(event)

    async def _on_message(self, event: dict[str, Any]) -> None:
        subtype = str(event.get("subtype", "") or "")
        if subtype == "message_deleted":
            return
        if subtype == "message_changed":
            await self._schedule_peer_edit_processing(event)
            return
        if self._is_peer_bot_inbound(event):
            # Streaming bots post a placeholder first then edit it to the final
            # text. Route the initial post through the same debounce path so we
            # only act on the stable text and dispatch once per peer message.
            await self._schedule_peer_edit_processing(event)
            return
        await self._process_message_event(event)

    def _is_peer_bot_inbound(self, event: dict[str, Any]) -> bool:
        bot_id = str(event.get("bot_id", "") or "")
        app_id = self._extract_app_id(event)
        subtype = str(event.get("subtype", "") or "")
        if not (bot_id or app_id or subtype == "bot_message"):
            return False
        user_id = str(event.get("user", "") or "")
        return not self._is_self_sender(user_id, bot_id)

    async def _process_message_event(
        self,
        event: dict[str, Any],
        *,
        skip_recent_event_check: bool = False,
    ) -> None:
        subtype = str(event.get("subtype", "") or "")
        user_id = str(event.get("user", "") or "")
        bot_id = str(event.get("bot_id", "") or "")
        app_id = self._extract_app_id(event)
        channel_id = str(event.get("channel", "") or "")
        text = str(event.get("text", "") or "").strip()
        if not channel_id or not text:
            return

        is_dm = str(event.get("channel_type", "") or "") == "im"
        if not self._is_authorized(
            channel_id,
            user_id,
            is_dm=is_dm,
            bot_id=bot_id,
            app_id=app_id,
        ):
            return

        thread_ts = str(event.get("thread_ts", "") or "")
        ts = str(event.get("ts", "") or "")
        if not skip_recent_event_check and self._should_skip_recent_event(channel_id, ts):
            logger.debug("Skipping duplicate Slack event for %s/%s", channel_id, ts)
            return
        reply_thread_ts = thread_ts or (ts if not is_dm else "")
        is_thread_reply = bool(reply_thread_ts and reply_thread_ts != ts)
        has_thread_session = bool(
            reply_thread_ts
            and await self._has_active_session_for_thread(channel_id, reply_thread_ts)
        )
        prepared_text = await self._prepare_inbound_text(
            event,
            channel_id=channel_id,
            text=text,
            is_dm=is_dm,
            is_thread_reply=is_thread_reply,
            reply_thread_ts=reply_thread_ts,
            has_thread_session=has_thread_session,
            bot_id=bot_id,
            app_id=app_id,
            subtype=subtype,
        )
        if prepared_text is None:
            return
        text = prepared_text
        if not await self._should_process_peer_turn(
            event,
            channel_id=channel_id,
            thread_ts=reply_thread_ts,
            current_ts=ts,
            is_dm=is_dm,
            bot_id=bot_id,
            app_id=app_id,
            subtype=subtype,
        ):
            return
        await self._delay_for_mention_order(text=str(event.get("text", "") or ""), is_dm=is_dm)

        chat_id = self._id_map.channel_to_int(channel_id)
        topic_id = (
            self._id_map.thread_to_int(channel_id, reply_thread_ts) if reply_thread_ts else None
        )
        key = SessionKey.for_transport("sl", chat_id, topic_id)
        self._last_active_channel = channel_id
        force_fresh_session = not is_dm

        command_text = self._normalize_command_text(text)
        async with self._processing_reaction(channel_id, ts):
            if command_text is not None:
                await self._handle_command(
                    command_text,
                    channel_id,
                    key,
                    reply_thread_ts or None,
                    stream_thread_ts=reply_thread_ts or ts,
                    recipient_user_id=user_id,
                )
                return

            if is_thread_reply and reply_thread_ts and not is_dm:
                thread_context = await self._fetch_thread_context(
                    channel_id=channel_id,
                    thread_ts=reply_thread_ts,
                    current_ts=ts,
                )
                if thread_context:
                    text = thread_context + text

            await self._dispatch_with_lock(
                key,
                text,
                channel_id,
                reply_thread_ts or None,
                stream_thread_ts=reply_thread_ts or ts,
                recipient_user_id=user_id,
                force_fresh_session=force_fresh_session,
            )

    async def _schedule_peer_edit_processing(self, event: dict[str, Any]) -> None:
        message = self._normalize_message_changed_event(event)
        if message is None:
            return
        channel_id = str(message.get("channel", "") or "")
        message_ts = str(message.get("ts", "") or "")
        if not channel_id or not message_ts:
            return
        user_id = str(message.get("user", "") or "")
        bot_id = str(message.get("bot_id", "") or "")
        app_id = self._extract_app_id(message)
        is_dm = str(message.get("channel_type", "") or "") == "im"
        if not self._is_authorized(
            channel_id,
            user_id,
            is_dm=is_dm,
            bot_id=bot_id,
            app_id=app_id,
        ):
            return
        if not (bot_id or app_id or str(message.get("subtype", "") or "") == "bot_message"):
            return

        key = (channel_id, message_ts)
        signature = self._peer_edit_signature(message)
        if self._processed_peer_edit_signatures.get(key) == signature:
            return

        existing = self._pending_peer_edit_tasks.pop(key, None)
        if existing is not None:
            existing.cancel()

        task = asyncio.create_task(self._debounce_peer_edit(key, message, signature))
        self._pending_peer_edit_tasks[key] = task

    def _normalize_message_changed_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if str(event.get("subtype", "") or "") == "message_changed":
            message = event.get("message")
            if not isinstance(message, dict):
                return None
            normalized = dict(message)
        else:
            normalized = dict(event)
        normalized.setdefault("channel", str(event.get("channel", "") or ""))
        channel_type = str(event.get("channel_type", "") or "")
        if channel_type:
            normalized.setdefault("channel_type", channel_type)
        subtype = str(normalized.get("subtype", "") or "")
        if not subtype and (normalized.get("bot_id") or self._extract_app_id(normalized)):
            normalized["subtype"] = "bot_message"
        return normalized

    def _peer_edit_signature(self, event: dict[str, Any]) -> str:
        return "|".join(
            [
                str(event.get("ts", "") or ""),
                str(event.get("thread_ts", "") or ""),
                str(event.get("bot_id", "") or ""),
                self._extract_app_id(event),
                str(event.get("text", "") or ""),
            ]
        )

    async def _debounce_peer_edit(
        self,
        key: tuple[str, str],
        event: dict[str, Any],
        signature: str,
    ) -> None:
        try:
            await asyncio.sleep(_PEER_EDIT_DEBOUNCE_SECONDS)
            if self._processed_peer_edit_signatures.get(key) == signature:
                return
            await self._process_message_event(event, skip_recent_event_check=True)
            self._processed_peer_edit_signatures[key] = signature
        finally:
            current = self._pending_peer_edit_tasks.get(key)
            if current is asyncio.current_task():
                self._pending_peer_edit_tasks.pop(key, None)

    async def _should_process_peer_turn(  # noqa: PLR0913
        self,
        event: dict[str, Any],
        *,
        channel_id: str,
        thread_ts: str,
        current_ts: str,
        is_dm: bool,
        bot_id: str,
        app_id: str,
        subtype: str,
    ) -> bool:
        if is_dm or not thread_ts:
            return True
        if not (bot_id or app_id or subtype == "bot_message"):
            return True
        messages = await self._fetch_thread_messages(channel_id=channel_id, thread_ts=thread_ts)
        if not messages:
            logger.info(
                "peer_turn_check ts=%s: empty thread snapshot -> ALLOW",
                current_ts,
            )
            return True
        anchor = self._latest_human_anchor(messages, current_ts=current_ts)
        if anchor is None:
            logger.info(
                "peer_turn_check ts=%s: no human anchor in %d msgs -> ALLOW",
                current_ts,
                len(messages),
            )
            return True
        anchor_ts, anchor_text = anchor
        peer_turns = self._count_peer_turns_since_anchor(
            messages,
            anchor_ts=anchor_ts,
            current_ts=current_ts,
        )
        if peer_turns <= 0:
            logger.info(
                "peer_turn_check ts=%s anchor=%s msgs=%d peer_turns=0 -> ALLOW",
                current_ts,
                anchor_ts,
                len(messages),
            )
            return True
        budget = await self._resolve_peer_turn_budget(
            channel_id=channel_id,
            thread_ts=thread_ts,
            anchor_ts=anchor_ts,
            anchor_text=anchor_text,
        )
        decision = peer_turns < budget
        logger.info(
            "peer_turn_check ts=%s anchor=%s msgs=%d peer_turns=%d budget=%d -> %s",
            current_ts,
            anchor_ts,
            len(messages),
            peer_turns,
            budget,
            "ALLOW" if decision else "SUPPRESS",
        )
        return decision

    async def _fetch_thread_messages(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        limit: int = 100,
    ) -> list[object]:
        try:
            response = await self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=limit,
                inclusive=True,
            )
        except Exception:
            logger.warning(
                "Failed to fetch Slack thread messages for %s/%s",
                channel_id,
                thread_ts,
                exc_info=True,
            )
            return []
        data = _slack_response_data(response)
        messages = data.get("messages", [])
        return messages if isinstance(messages, list) else []

    def _latest_human_anchor(
        self,
        messages: list[object],
        *,
        current_ts: str,
    ) -> tuple[str, str] | None:
        latest: tuple[str, str] | None = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_ts = str(msg.get("ts", "") or "")
            if not msg_ts or _slack_ts_is_at_or_after(msg_ts, current_ts):
                continue
            if self._looks_like_bot_message(msg):
                continue
            msg_text = str(msg.get("text", "") or "").strip()
            if not msg_text:
                continue
            latest = (msg_ts, msg_text)
        return latest

    def _looks_like_bot_message(self, msg: dict[str, Any]) -> bool:
        msg_user = str(msg.get("user", "") or "")
        msg_bot_id = str(msg.get("bot_id", "") or "")
        if self._is_self_sender(msg_user, msg_bot_id):
            return True
        if msg_bot_id:
            return True
        if self._extract_app_id(msg):
            return True
        return str(msg.get("subtype", "") or "") == "bot_message"

    def _count_peer_turns_since_anchor(
        self,
        messages: list[object],
        *,
        anchor_ts: str,
        current_ts: str,
    ) -> int:
        count = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_ts = str(msg.get("ts", "") or "")
            if not msg_ts or _slack_ts_is_at_or_after(anchor_ts, msg_ts):
                continue
            if _slack_ts_is_at_or_after(msg_ts, current_ts) and msg_ts != current_ts:
                continue
            if self._looks_like_bot_message(msg):
                count += 1
        return count

    async def _resolve_peer_turn_budget(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        anchor_ts: str,
        anchor_text: str,
    ) -> int:
        cache_key = (channel_id, thread_ts, anchor_ts)
        cached = self._peer_turn_budget_cache.get(cache_key)
        if cached is not None:
            return cached
        budget = await self._extract_peer_turn_budget(anchor_text)
        if len(self._peer_turn_budget_cache) >= _PEER_TURN_BUDGET_CACHE_MAX:
            oldest = next(iter(self._peer_turn_budget_cache))
            self._peer_turn_budget_cache.pop(oldest, None)
        self._peer_turn_budget_cache[cache_key] = budget
        return budget

    async def _extract_peer_turn_budget(self, anchor_text: str) -> int:
        for pattern in _PEER_TURN_BUDGET_PATTERNS:
            match = pattern.search(anchor_text)
            if match is None:
                continue
            try:
                turns = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if turns <= 0:
                continue
            return max(1, min(_MAX_PEER_TURN_BUDGET, turns))
        return _DEFAULT_PEER_TURN_BUDGET

    async def _prepare_inbound_text(  # noqa: PLR0913
        self,
        event: dict[str, Any],
        *,
        channel_id: str,
        text: str,
        is_dm: bool,
        is_thread_reply: bool,
        reply_thread_ts: str,
        has_thread_session: bool,
        bot_id: str,
        app_id: str,
        subtype: str,
    ) -> str | None:
        if not is_dm and self._config.group_mention_only:
            is_mentioned = bool(self._bot_user_id and f"<@{self._bot_user_id}>" in text)
            in_mentioned_thread = bool(
                is_thread_reply
                and reply_thread_ts
                and self._has_recent_mentioned_thread(channel_id, reply_thread_ts)
            )
            if not is_mentioned and not in_mentioned_thread and not has_thread_session:
                return None
            text = self._strip_mention(text)
            if is_mentioned and reply_thread_ts:
                self._mark_mentioned_thread(channel_id, reply_thread_ts)
        if bot_id or app_id or subtype == "bot_message":
            peer_name = await self._resolve_peer_name(event, channel_id=channel_id)
            return self._wrap_peer_message(text, peer_name)
        return text

    async def _delay_for_mention_order(self, *, text: str, is_dm: bool) -> None:
        delay_seconds = self._mention_order_delay_seconds(text=text, is_dm=is_dm)
        if delay_seconds <= 0:
            return
        await asyncio.sleep(delay_seconds)

    def _mention_order_delay_seconds(self, *, text: str, is_dm: bool) -> float:
        if is_dm or not self._bot_user_id:
            return 0.0
        mentioned_ids: list[str] = []
        seen: set[str] = set()
        for match in _SLACK_MENTION_RE.finditer(text):
            mentioned_id = match.group(1)
            if mentioned_id in seen:
                continue
            seen.add(mentioned_id)
            mentioned_ids.append(mentioned_id)
        if self._bot_user_id not in seen:
            return 0.0
        return mentioned_ids.index(self._bot_user_id) * _MENTION_ORDER_DELAY_SECONDS

    async def _handle_command(  # noqa: PLR0913
        self,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
        *,
        stream_thread_ts: str | None = None,
        recipient_user_id: str | None = None,
    ) -> None:
        cmd = text.split(maxsplit=1)[0].lower().lstrip("/")
        handler = self._COMMAND_DISPATCH.get(cmd)
        if handler is not None:
            if cmd in self._IMMEDIATE_COMMANDS:
                await handler(self, text=text, channel_id=channel_id, key=key, thread_ts=thread_ts)
            else:
                await self._run_handler_with_lock(
                    handler,
                    text=text,
                    channel_id=channel_id,
                    key=key,
                    thread_ts=thread_ts,
                )
        elif classify_command(cmd) in ("orchestrator", "multiagent"):
            await self._cmd_orchestrator(
                text=text, channel_id=channel_id, key=key, thread_ts=thread_ts
            )
        else:
            await self._dispatch_with_lock(
                key,
                text,
                channel_id,
                thread_ts,
                stream_thread_ts=stream_thread_ts,
                recipient_user_id=recipient_user_id,
            )

    async def _dispatch_with_lock(  # noqa: PLR0913
        self,
        key: SessionKey,
        text: str,
        channel_id: str,
        thread_ts: str | None,
        *,
        stream_thread_ts: str | None = None,
        recipient_user_id: str | None = None,
        force_fresh_session: bool = False,
    ) -> None:
        lock = self._lock_pool.get(key.lock_key)
        async with lock:
            await self._dispatch_message(
                key,
                text,
                channel_id,
                thread_ts,
                stream_thread_ts=stream_thread_ts,
                recipient_user_id=recipient_user_id,
                force_fresh_session=force_fresh_session,
            )

    async def _run_handler_with_lock(
        self,
        handler: Callable[..., Awaitable[None]],
        **kwargs: object,
    ) -> None:
        key = kwargs["key"]
        assert isinstance(key, SessionKey)
        lock = self._lock_pool.get(key.lock_key)
        async with lock:
            await handler(self, **kwargs)

    async def _dispatch_message(  # noqa: PLR0913
        self,
        key: SessionKey,
        text: str,
        channel_id: str,
        thread_ts: str | None,
        *,
        stream_thread_ts: str | None = None,
        recipient_user_id: str | None = None,
        force_fresh_session: bool = False,
    ) -> None:
        if self._config.streaming.enabled:
            await self._run_streaming(
                key,
                text,
                channel_id,
                stream_thread_ts or thread_ts,
                recipient_user_id=recipient_user_id,
                force_fresh_session=force_fresh_session,
            )
            return
        await self._run_non_streaming(
            key,
            text,
            channel_id,
            thread_ts,
            force_fresh_session=force_fresh_session,
        )

    async def _run_streaming(  # noqa: PLR0913
        self,
        key: SessionKey,
        text: str,
        channel_id: str,
        thread_ts: str | None,
        *,
        recipient_user_id: str | None = None,
        force_fresh_session: bool = False,
    ) -> None:
        orch = self._orchestrator
        if orch is None or thread_ts is None:
            return

        from ductor_slack.messenger.slack.streaming import SlackStreamEditor

        editor = SlackStreamEditor(
            self.client,
            channel_id,
            thread_ts=thread_ts,
            recipient_user_id=recipient_user_id if not channel_id.startswith("D") else None,
            recipient_team_id=self._team_id or None if not channel_id.startswith("D") else None,
            edit_interval_seconds=self._config.streaming.edit_interval_seconds,
        )
        result = await orch.handle_message_streaming(
            key,
            text,
            force_fresh_session=force_fresh_session,
            on_text_delta=editor.on_delta,
            on_thinking_delta=editor.on_thinking,
            on_tool_activity=editor.on_tool,
            on_system_status=editor.on_system,
        )
        self._maybe_append_footer(result)
        await editor.finalize(result.text)

    async def _run_non_streaming(
        self,
        key: SessionKey,
        text: str,
        channel_id: str,
        thread_ts: str | None,
        *,
        force_fresh_session: bool = False,
    ) -> None:
        orch = self._orchestrator
        if orch is None:
            return
        result = await orch.handle_message(key, text, force_fresh_session=force_fresh_session)
        self._maybe_append_footer(result)
        if result.text:
            await self._send_rich(channel_id, result.text, thread_ts=thread_ts)

    async def _cmd_stop(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text
        orch = self._orchestrator
        if orch:
            killed = await orch.abort(key.chat_id)
            msg = t("abort_all.done", count=killed) if killed else t("abort_all.nothing")
        else:
            msg = t("abort_all.nothing")
        await self._send_rich(channel_id, msg, thread_ts=thread_ts)

    async def _cmd_interrupt(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text
        orch = self._orchestrator
        if orch:
            interrupted = orch.interrupt(key.chat_id)
            msg = t("interrupt.done", count=interrupted) if interrupted else t("interrupt.nothing")
            await self._send_rich(channel_id, msg, thread_ts=thread_ts)

    async def _cmd_stop_all(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text, key
        orch = self._orchestrator
        killed = await orch.abort_all() if orch else 0
        if self._abort_all_callback:
            killed += await self._abort_all_callback()
        msg = t("abort_all.done", count=killed) if killed else t("abort_all.nothing")
        await self._send_rich(channel_id, msg, thread_ts=thread_ts)

    async def _cmd_restart(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text, key
        from ductor_slack.infra.restart import EXIT_RESTART, write_restart_marker

        marker = _restart_marker_path(self._config.ductor_home)
        write_restart_marker(marker_path=marker)
        await self._send_rich(
            channel_id,
            fmt(t("startup.restart_header"), SEP, t("startup.restart_body")),
            thread_ts=thread_ts,
        )
        self._exit_code = EXIT_RESTART
        if self._socket_task and not self._socket_task.done():
            self._socket_task.cancel()

    async def _cmd_new(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text
        orch = self._orchestrator
        if orch:
            result = await orch.handle_message(key, "/new")
            if result and result.text:
                await self._send_rich(channel_id, result.text, thread_ts=thread_ts)

    async def _cmd_help(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text, key
        await self._send_rich(channel_id, self._build_help_text(), thread_ts=thread_ts)

    async def _cmd_info(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text, key
        text_out = fmt(
            t("info.header"),
            t("info.version", version=get_current_version()),
            SEP,
            t("info.slack_description"),
        )
        await self._send_rich(channel_id, text_out, thread_ts=thread_ts)

    async def _cmd_agent_commands(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del text, key
        lines = [
            "Slack sub-agents use the same multi-agent runtime.",
            "",
            "`/agents` — list all agents and their status",
            "`/agent_start <name>` — start a sub-agent",
            "`/agent_stop <name>` — stop a sub-agent",
            "`/agent_restart <name>` — restart a sub-agent",
        ]
        await self._send_rich(
            channel_id,
            fmt(t("agents.system_header"), SEP, "\n".join(lines)),
            thread_ts=thread_ts,
        )

    async def _cmd_showfiles(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        del key
        orch = self._orchestrator
        if not orch:
            return
        from ductor_slack.messenger.matrix.file_browser import format_file_listing

        parts = text.split(None, 1)
        subdir = parts[1].strip() if len(parts) > 1 else ""
        listing = await asyncio.to_thread(format_file_listing, orch.paths, subdir)
        await self._send_rich(channel_id, listing, thread_ts=thread_ts)

    async def _cmd_session(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        parts = text.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            await self._send_rich(
                channel_id,
                fmt(
                    t("session_help.header"),
                    SEP,
                    "`/session <prompt>` — start a background session\n"
                    "`/sessions` — list running sessions\n"
                    "`/stop` — stop the active run",
                ),
                thread_ts=thread_ts,
            )
            return
        await self._dispatch_message(key, text, channel_id, thread_ts)

    async def _cmd_orchestrator(
        self,
        *,
        text: str,
        channel_id: str,
        key: SessionKey,
        thread_ts: str | None,
    ) -> None:
        orch = self._orchestrator
        if not orch:
            return
        result = await orch.handle_message(key, text)
        if result and result.text:
            await self._send_rich(channel_id, result.text, thread_ts=thread_ts)

    _COMMAND_DISPATCH: dict[str, Callable[..., Awaitable[None]]] = {
        "stop": _cmd_stop,
        "stop_all": _cmd_stop_all,
        "interrupt": _cmd_interrupt,
        "restart": _cmd_restart,
        "new": _cmd_new,
        "help": _cmd_help,
        "start": _cmd_help,
        "info": _cmd_info,
        "agent_commands": _cmd_agent_commands,
        "showfiles": _cmd_showfiles,
        "session": _cmd_session,
    }

    _IMMEDIATE_COMMANDS: frozenset[str] = frozenset(
        {
            "stop",
            "stop_all",
            "interrupt",
            "restart",
            "help",
            "start",
            "info",
            "agent_commands",
            "showfiles",
        }
    )

    def _build_help_text(self) -> str:
        cmd_desc = {**dict(BOT_COMMANDS), **dict(MULTIAGENT_SUB_COMMANDS)}

        def _line(command: str) -> str:
            description = cmd_desc.get(command, "")
            return f"`/{command}` — {description}" if description else f"`/{command}`"

        return fmt(
            t("help.header"),
            SEP,
            f"**{t('help.cat_daily')}**\n{_line('new')}\n{_line('stop')}\n{_line('stop_all')}\n"
            f"{_line('model')}\n{_line('status')}\n{_line('memory')}",
            f"**{t('help.cat_automation')}**\n{_line('session')}\n{_line('tasks')}\n{_line('cron')}",
            f"**{t('help.cat_multiagent')}**\n{_line('agent_commands')}\n{_line('agents')}\n"
            f"{_line('agent_start')}\n{_line('agent_stop')}\n{_line('agent_restart')}",
            f"**{t('help.cat_browse')}**\n{_line('showfiles')}\n{_line('info')}\n{_line('help')}",
            f"**{t('help.cat_maintenance')}**\n{_line('diagnose')}\n{_line('upgrade')}\n{_line('restart')}",
            SEP,
            t("help.slack_footer"),
        )

    @staticmethod
    def _extract_app_id(event: dict[str, Any]) -> str:
        app_id = str(event.get("app_id", "") or "")
        if app_id:
            return app_id
        bot_profile = event.get("bot_profile")
        if isinstance(bot_profile, dict):
            return str(bot_profile.get("app_id", "") or "")
        return ""

    def _is_self_sender(self, user_id: str, bot_id: str) -> bool:
        return (bool(user_id) and user_id == self._bot_user_id) or (
            bool(bot_id) and bool(self._bot_id) and bot_id == self._bot_id
        )

    def _is_allowed_bot_sender(self, bot_id: str, app_id: str) -> bool:
        slack = self._config.slack
        bot_ok = bool(bot_id) and bot_id in slack.allowed_bot_ids
        app_ok = bool(app_id) and app_id in slack.allowed_app_ids
        return bot_ok or app_ok

    def _is_authorized(
        self,
        channel_id: str,
        user_id: str,
        *,
        is_dm: bool,
        bot_id: str = "",
        app_id: str = "",
    ) -> bool:
        slack = self._config.slack
        channel_ok = is_dm or not slack.allowed_channels or channel_id in slack.allowed_channels
        if not channel_ok:
            return False
        if self._is_self_sender(user_id, bot_id):
            return False
        if bot_id or app_id or not user_id:
            return self._is_allowed_bot_sender(bot_id, app_id)
        if self._config.group_mention_only and not is_dm:
            return channel_ok
        return not slack.allowed_users or user_id in slack.allowed_users

    def _is_message_addressed(self, channel_id: str, thread_ts: str, text: str) -> bool:
        if self._bot_user_id and f"<@{self._bot_user_id}>" in text:
            return True
        return self._has_recent_mentioned_thread(channel_id, thread_ts)

    def _normalize_command_text(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        parts = stripped.split(None, 1)
        raw_cmd = parts[0]
        cmd = raw_cmd.lower().lstrip("/")
        if classify_command(cmd) == "unknown":
            return None
        has_args = len(parts) > 1 and bool(parts[1].strip())
        if not raw_cmd.startswith("/") and has_args and cmd not in _MESSAGE_COMMANDS_WITH_ARGS:
            return None
        suffix = f" {parts[1].strip()}" if has_args else ""
        return f"/{cmd}{suffix}"

    def _prune_mentioned_threads(self, now: float) -> None:
        if self._MENTIONED_THREAD_TTL > 0:
            cutoff = now - self._MENTIONED_THREAD_TTL
            expired = [key for key, seen_at in self._mentioned_threads.items() if seen_at < cutoff]
            for key in expired:
                del self._mentioned_threads[key]
        max_size = max(1, self._MENTIONED_THREAD_MAX_SIZE)
        while len(self._mentioned_threads) > max_size:
            oldest = next(iter(self._mentioned_threads))
            del self._mentioned_threads[oldest]

    def _prune_recent_events(self, now: float) -> None:
        if self._RECENT_EVENT_TTL > 0:
            cutoff = now - self._RECENT_EVENT_TTL
            expired = [key for key, seen_at in self._recent_events.items() if seen_at < cutoff]
            for key in expired:
                del self._recent_events[key]
        max_size = max(1, self._RECENT_EVENT_MAX_SIZE)
        while len(self._recent_events) > max_size:
            oldest = next(iter(self._recent_events))
            del self._recent_events[oldest]

    def _should_skip_recent_event(self, channel_id: str, message_ts: str) -> bool:
        if not channel_id or not message_ts:
            return False
        now = time.monotonic()
        self._prune_recent_events(now)
        key = (channel_id, message_ts)
        seen_at = self._recent_events.get(key)
        if seen_at is not None:
            if self._RECENT_EVENT_TTL <= 0 or now - seen_at < self._RECENT_EVENT_TTL:
                return True
            del self._recent_events[key]
        self._recent_events[key] = now
        self._prune_recent_events(now)
        return False

    def _mark_mentioned_thread(self, channel_id: str, thread_ts: str) -> None:
        now = time.monotonic()
        key = (channel_id, thread_ts)
        self._mentioned_threads.pop(key, None)
        self._mentioned_threads[key] = now
        self._prune_mentioned_threads(now)

    def _has_recent_mentioned_thread(self, channel_id: str, thread_ts: str) -> bool:
        if not thread_ts:
            return False
        now = time.monotonic()
        self._prune_mentioned_threads(now)
        key = (channel_id, thread_ts)
        seen_at = self._mentioned_threads.get(key)
        if seen_at is None:
            return False
        if self._MENTIONED_THREAD_TTL > 0 and now - seen_at >= self._MENTIONED_THREAD_TTL:
            del self._mentioned_threads[key]
            return False
        return True

    async def _has_active_session_for_thread(self, channel_id: str, thread_ts: str) -> bool:
        """Return whether this Slack thread already has a fresh persisted session."""
        orch = self._orchestrator
        if orch is None:
            return False
        chat_id = self._id_map.channel_to_int(channel_id)
        topic_id = self._id_map.thread_to_int(channel_id, thread_ts)
        sessions = await orch._sessions.list_active_for_chat(chat_id)
        for session in sessions:
            if session.topic_id == topic_id and bool(session.session_id):
                return True
        return False

    async def _resolve_user_name(self, user_id: str, *, channel_id: str) -> str:
        """Resolve a Slack user ID to a display name with a small in-memory cache."""
        if not user_id:
            return "unknown"
        cached = self._user_name_cache.get(user_id)
        if cached:
            return cached
        try:
            response = await self.client.users_info(user=user_id)
            data = _slack_response_data(response)
            user = data.get("user", {}) if isinstance(data, dict) else {}
            profile = user.get("profile", {}) if isinstance(user, dict) else {}
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("real_name")
                or user.get("name")
                or user_id
            )
        except Exception:
            logger.debug(
                "Failed to resolve Slack user name in channel %s", channel_id, exc_info=True
            )
            name = user_id
        resolved = str(name).strip() or user_id
        self._user_name_cache[user_id] = resolved
        return resolved

    async def _resolve_peer_name(self, event: dict[str, Any], *, channel_id: str) -> str:
        bot_profile = event.get("bot_profile")
        if isinstance(bot_profile, dict):
            profile_name = str(bot_profile.get("name", "") or "").strip()
            if profile_name:
                return profile_name
        username = str(event.get("username", "") or "").strip()
        if username:
            return username
        user_id = str(event.get("user", "") or "")
        if user_id:
            user_name = await self._resolve_user_name(user_id, channel_id=channel_id)
            if user_name and user_name != "unknown":
                return user_name
        app_id = self._extract_app_id(event)
        if app_id:
            return app_id
        bot_id = str(event.get("bot_id", "") or "")
        if bot_id:
            return bot_id
        return "unknown"

    def _wrap_peer_message(self, text: str, peer_name: str) -> str:
        speaker = peer_name or "unknown"
        return (
            f'[Message from peer agent "{speaker}". This is NOT a user request and NOT your '
            f'sub-agent. They are an independent agent in this thread. Respond with your own '
            f'perspective as "{self._bot_name}"; do not repeat their points, do not mirror '
            "their structure or wording, and do not speak on their behalf. If you agree, say "
            "so briefly and move forward; if you disagree, push back.]\n\n"
            f"{text}"
        )

    async def _fetch_thread_context(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        current_ts: str,
        limit: int = 30,
    ) -> str:
        """Fetch earlier Slack thread messages for the current non-DM turn snapshot."""
        cache_key = f"{channel_id}:{thread_ts}:{current_ts}"
        now = time.monotonic()
        self._prune_thread_context_cache(now)
        cached = self._thread_context_cache.get(cache_key)
        if cached:
            return cached.content

        try:
            response = await self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=limit + 1,
                inclusive=True,
            )
        except Exception:
            logger.warning(
                "Failed to fetch Slack thread context for %s/%s",
                channel_id,
                thread_ts,
                exc_info=True,
            )
            return ""

        data = _slack_response_data(response)
        messages = data.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return ""

        context_parts, has_peer_agent = await self._build_thread_context_parts(
            messages=messages,
            channel_id=channel_id,
            thread_ts=thread_ts,
            current_ts=current_ts,
        )
        return self._cache_thread_context(
            cache_key=cache_key,
            content_parts=context_parts,
            fetched_at=now,
            has_peer_agent=has_peer_agent,
        )

    async def _build_thread_context_parts(
        self,
        *,
        messages: list[object],
        channel_id: str,
        thread_ts: str,
        current_ts: str,
    ) -> tuple[list[str], bool]:
        """Build normalized thread-history lines from Slack reply payloads."""
        context_parts: list[str] = []
        has_peer_agent = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_ts = str(msg.get("ts", "") or "")
            if not msg_ts or _slack_ts_is_at_or_after(msg_ts, current_ts):
                continue
            msg_user = str(msg.get("user", "") or "")
            msg_bot_id = str(msg.get("bot_id", "") or "")
            msg_app_id = self._extract_app_id(msg)
            if self._is_self_sender(msg_user, msg_bot_id):
                continue
            if (msg_bot_id or msg.get("subtype") == "bot_message" or msg_app_id) and not (
                self._is_allowed_bot_sender(msg_bot_id, msg_app_id)
            ):
                continue
            msg_text = str(msg.get("text", "") or "").strip()
            if not msg_text:
                continue
            if self._bot_user_id:
                msg_text = msg_text.replace(f"<@{self._bot_user_id}>", "").strip()
            if not msg_text:
                continue
            if msg_bot_id or msg_app_id or msg.get("subtype") == "bot_message":
                speaker = f"peer agent {await self._resolve_peer_name(msg, channel_id=channel_id)}"
                has_peer_agent = True
            else:
                speaker = await self._resolve_user_name(msg_user, channel_id=channel_id)
            prefix = "[thread parent] " if msg_ts == thread_ts else ""
            context_parts.append(f"{prefix}{speaker}: {msg_text}")
        return context_parts, has_peer_agent

    def _cache_thread_context(
        self,
        *,
        cache_key: str,
        content_parts: list[str],
        fetched_at: float,
        has_peer_agent: bool = False,
    ) -> str:
        """Persist thread-context cache entry and return the formatted content."""
        self._prune_thread_context_cache(fetched_at)
        if not content_parts:
            self._thread_context_cache.pop(cache_key, None)
            self._thread_context_cache[cache_key] = _ThreadContextCache(
                content="",
                fetched_at=fetched_at,
            )
            self._prune_thread_context_cache(fetched_at)
            return ""

        header = "[Thread context — prior messages in this thread (not yet in conversation history)."
        if has_peer_agent:
            header += (
                f'\nYou are "{self._bot_name}". Lines tagged "peer agent X" are from other '
                "independent agents — do not mirror or speak for them."
            )
        header += "]\n"
        content = header + "\n".join(content_parts) + "\n[End of thread context]\n\n"
        self._thread_context_cache.pop(cache_key, None)
        self._thread_context_cache[cache_key] = _ThreadContextCache(
            content=content,
            fetched_at=fetched_at,
            message_count=len(content_parts),
        )
        self._prune_thread_context_cache(fetched_at)
        return content

    def _prune_thread_context_cache(self, now: float) -> None:
        if self._THREAD_CACHE_TTL > 0:
            cutoff = now - self._THREAD_CACHE_TTL
            expired = [
                key
                for key, entry in self._thread_context_cache.items()
                if entry.fetched_at < cutoff
            ]
            for key in expired:
                del self._thread_context_cache[key]
        max_size = max(1, self._THREAD_CONTEXT_CACHE_MAX_SIZE)
        while len(self._thread_context_cache) > max_size:
            oldest = next(iter(self._thread_context_cache))
            del self._thread_context_cache[oldest]

    def _strip_mention(self, text: str) -> str:
        if not self._bot_user_id:
            return text
        return text.replace(f"<@{self._bot_user_id}>", "").strip()

    def _maybe_append_footer(self, result: object) -> None:
        from ductor_slack.orchestrator.registry import OrchestratorResult

        if not isinstance(result, OrchestratorResult):
            return
        if not self._config.scene.technical_footer or not result.model_name:
            return
        from ductor_slack.text.response_format import format_technical_footer

        footer = format_technical_footer(
            result.model_name,
            result.total_tokens,
            result.input_tokens,
            result.cost_usd,
            result.duration_ms,
        )
        result.text += footer

    async def _send_rich(
        self,
        channel_id: str,
        text: str,
        *,
        thread_ts: str | None = None,
    ) -> str | None:
        return await send_rich(self.client, channel_id, text, SlackSendOpts(thread_ts=thread_ts))

    @contextlib.asynccontextmanager
    async def _processing_reaction(self, channel_id: str, message_ts: str) -> Any:
        added = await self._add_processing_reaction(channel_id, message_ts)
        try:
            yield
        finally:
            if added:
                await self._remove_processing_reaction(channel_id, message_ts)

    async def _add_processing_reaction(self, channel_id: str, message_ts: str) -> bool:
        try:
            await slack_add_reaction(self.client, channel_id, message_ts, "eyes")
        except Exception as exc:
            self._log_reaction_error("add", exc)
            return False
        return True

    async def _remove_processing_reaction(self, channel_id: str, message_ts: str) -> None:
        try:
            await slack_remove_reaction(self.client, channel_id, message_ts, "eyes")
        except Exception as exc:
            self._log_reaction_error("remove", exc)

    def _log_reaction_error(self, action: str, exc: Exception) -> None:
        response = getattr(exc, "response", None)
        error_code = None
        if isinstance(response, dict):
            error_code = response.get("error")
        elif response is not None:
            data = getattr(response, "data", None)
            if isinstance(data, dict):
                error_code = data.get("error")
        if error_code == "missing_scope":
            logger.warning(
                "Slack reactions.%s failed: missing reactions:write scope",
                action,
            )
            return
        logger.debug("Failed to %s processing reaction: %r", action, exc)

    def _broadcast_channels(self) -> list[str]:
        channels = list(self._config.slack.allowed_channels)
        if not channels and self._last_active_channel:
            channels = [self._last_active_channel]
        return channels

    async def broadcast(self, text: str) -> None:
        channels = self._broadcast_channels()
        if not channels:
            logger.warning("Slack broadcast: no channels available, message lost: %s", text[:80])
            return
        failures: list[tuple[str, str]] = []
        for channel_id in channels:
            try:
                await self._send_rich(channel_id, text)
            except _SlackApiError as exc:
                err = (exc.response or {}).get("error", "unknown") if hasattr(exc, "response") else "unknown"
                failures.append((channel_id, err))
                logger.warning(
                    "Slack broadcast: skipping channel=%s error=%s (continuing with remaining channels)",
                    channel_id,
                    err,
                )
        if failures:
            logger.warning(
                "Slack broadcast: %d/%d channel(s) failed: %s",
                len(failures),
                len(channels),
                ", ".join(f"{ch}={err}" for ch, err in failures),
            )

    async def notify_startup(self, text: str) -> None:
        await self._notification_service.notify_all(text)

    async def notify_upgrade(self, text: str) -> None:
        await self._notification_service.notify_all(text)

    async def on_async_interagent_result(self, result: AsyncInterAgentResult) -> None:
        from ductor_slack.bus.adapters import from_interagent_result

        chat_id = self._default_chat_id()
        if not chat_id:
            logger.warning(
                "No chat_id for async interagent result (task=%s) — delivering to all channels",
                result.task_id,
            )
            text = result.result_text or f"Inter-agent result from {result.recipient}"
            await self._notification_service.notify_all(text)
            return
        env = from_interagent_result(result, chat_id)
        env.transport = "sl"
        await self._bus.submit(env)

    async def on_task_result(self, result: TaskResult) -> None:
        from ductor_slack.bus.adapters import from_task_result

        env = from_task_result(result)
        env.transport = "sl"
        await self._bus.submit(env)

    async def on_task_question(
        self,
        task_id: str,
        question: str,
        prompt_preview: str,
        chat_id: int,
        thread_id: int | None = None,
    ) -> None:
        from ductor_slack.bus.adapters import from_task_question

        if not chat_id:
            chat_id = self._default_chat_id()
        env = from_task_question(task_id, question, prompt_preview, chat_id, topic_id=thread_id)
        env.transport = "sl"
        await self._bus.submit(env)

    def _default_chat_id(self) -> int:
        if self._config.slack.allowed_channels:
            return self._id_map.channel_to_int(self._config.slack.allowed_channels[0])
        if self._last_active_channel:
            return self._id_map.channel_to_int(self._last_active_channel)
        logger.warning("No default chat_id: no allowed_channels and no active channel yet")
        return 0

    async def _watch_restart_marker(self) -> None:
        from ductor_slack.infra.restart import EXIT_RESTART

        marker = _restart_marker_path(self._config.ductor_home)
        while True:
            await asyncio.sleep(2)
            if marker.exists():
                logger.info("Restart marker detected")
                self._exit_code = EXIT_RESTART
                if self._socket_task and not self._socket_task.done():
                    self._socket_task.cancel()
                break
