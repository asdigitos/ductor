from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ductor_slack.cli.stream_events import ToolUseEvent
from ductor_slack.config import AgentConfig
from ductor_slack.messenger.slack.bot import SlackBot, _ThreadContextCache
from ductor_slack.orchestrator.registry import OrchestratorResult
from ductor_slack.session.manager import SessionData


@contextlib.asynccontextmanager
async def _instant_peer_debounce(bot: SlackBot):
    """Run the body with peer-edit debounce sleeps no-op'd, then flush pending tasks.

    Peer bot inbound messages now go through the same debounced path as
    `message_changed`, so callers that need to assert post-dispatch state must
    wait for the scheduled task to finish.
    """
    with patch("ductor_slack.messenger.slack.bot.asyncio.sleep", new=AsyncMock()):
        yield
        pending = list(bot._pending_peer_edit_tasks.values())
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _make_bot() -> SlackBot:
    bot = object.__new__(SlackBot)
    bot._config = AgentConfig(
        transport="slack",
        slack={
            "bot_token": "xoxb-test",
            "app_token": "xapp-test",
            "allowed_channels": ["C123"],
            "allowed_users": ["U123"],
        },
        group_mention_only=True,
    )
    bot._agent_name = "main"
    bot._app = SimpleNamespace(client=AsyncMock())
    bot._lock_pool = MagicMock()
    bot._bus = MagicMock()
    bot._id_map = MagicMock()
    bot._id_map.channel_to_int.return_value = 11
    bot._id_map.thread_to_int.return_value = 22
    bot._orchestrator = MagicMock()
    bot._orchestrator._sessions.list_active_for_chat = AsyncMock(return_value=[])
    bot._startup_hooks = []
    bot._bot_user_id = "B123"
    bot._bot_id = "BOTSELF"
    bot._bot_name = "ductor"
    bot._team_id = "T123"
    bot._last_active_channel = None
    bot._mentioned_threads = {}
    bot._recent_events = {}
    bot._pending_peer_edit_tasks = {}
    bot._processed_peer_edit_signatures = {}
    bot._peer_turn_budget_cache = {}
    bot._user_name_cache = {}
    bot._thread_context_cache = {}
    bot._MENTIONED_THREAD_TTL = 3600.0
    bot._MENTIONED_THREAD_MAX_SIZE = 200
    bot._RECENT_EVENT_TTL = 120.0
    bot._RECENT_EVENT_MAX_SIZE = 500
    bot._THREAD_CACHE_TTL = 60.0
    bot._THREAD_CONTEXT_CACHE_MAX_SIZE = 200
    bot._dispatch_with_lock = AsyncMock()
    bot._handle_command = AsyncMock()
    bot._send_rich = AsyncMock()
    return bot


class TestThreadSessionLookup:
    async def test_detects_active_thread_session(self) -> None:
        bot = _make_bot()
        active = SessionData(chat_id=11, transport="sl", topic_id=22, provider_sessions={})
        active.session_id = "sid-1"
        bot._orchestrator._sessions.list_active_for_chat.return_value = [active]

        result = await bot._has_active_session_for_thread("C123", "1710000000.123")

        assert result is True


class TestMentionedThreadCache:
    def test_prunes_expired_entries(self) -> None:
        bot = _make_bot()
        now = time.monotonic()
        bot._MENTIONED_THREAD_TTL = 10.0
        bot._mentioned_threads = {
            ("C123", "old"): now - 20.0,
            ("C123", "fresh"): now - 1.0,
        }

        with patch("ductor_slack.messenger.slack.bot.time") as mock_time:
            mock_time.monotonic.return_value = now
            bot._mark_mentioned_thread("C123", "new")

        assert ("C123", "old") not in bot._mentioned_threads
        assert ("C123", "fresh") in bot._mentioned_threads
        assert ("C123", "new") in bot._mentioned_threads

    def test_enforces_max_size(self) -> None:
        bot = _make_bot()
        bot._MENTIONED_THREAD_MAX_SIZE = 2
        now = time.monotonic()

        with patch("ductor_slack.messenger.slack.bot.time") as mock_time:
            mock_time.monotonic.return_value = now
            bot._mark_mentioned_thread("C123", "one")
            mock_time.monotonic.return_value = now + 1.0
            bot._mark_mentioned_thread("C123", "two")
            mock_time.monotonic.return_value = now + 2.0
            bot._mark_mentioned_thread("C123", "three")

        assert list(bot._mentioned_threads) == [("C123", "two"), ("C123", "three")]


class TestRecentEventCache:
    def test_prunes_expired_entries(self) -> None:
        bot = _make_bot()
        now = time.monotonic()
        bot._RECENT_EVENT_TTL = 10.0
        bot._recent_events = {
            ("C123", "old"): now - 20.0,
            ("C123", "fresh"): now - 1.0,
        }

        with patch("ductor_slack.messenger.slack.bot.time") as mock_time:
            mock_time.monotonic.return_value = now
            skipped = bot._should_skip_recent_event("C123", "new")

        assert skipped is False
        assert ("C123", "old") not in bot._recent_events
        assert ("C123", "fresh") in bot._recent_events
        assert ("C123", "new") in bot._recent_events

    def test_marks_duplicate_within_ttl(self) -> None:
        bot = _make_bot()
        now = time.monotonic()

        with patch("ductor_slack.messenger.slack.bot.time") as mock_time:
            mock_time.monotonic.return_value = now
            assert bot._should_skip_recent_event("C123", "1710000000.456") is False
            mock_time.monotonic.return_value = now + 1.0
            assert bot._should_skip_recent_event("C123", "1710000000.456") is True


class TestThreadContextFetching:
    async def test_fetches_prior_thread_messages_once(self) -> None:
        bot = _make_bot()
        bot._app.client.conversations_replies.return_value = {
            "messages": [
                {"ts": "1710000000.100", "user": "U111", "text": "First context"},
                {"ts": "1710000000.123", "user": "U222", "text": "<@B123> Parent message"},
                {
                    "ts": "1710000000.200",
                    "bot_id": "BOT",
                    "bot_profile": {"name": "reviewer"},
                    "text": "Bot output",
                },
                {"ts": "1710000000.300", "user": "U333", "text": "Current message"},
                {"ts": "1710000000.301", "user": "U444", "text": "Future message"},
            ]
        }
        bot._resolve_user_name = AsyncMock(side_effect=["Alice", "Bob"])
        bot._config.slack.allowed_bot_ids = ["BOT"]

        content = await bot._fetch_thread_context(
            channel_id="C123",
            thread_ts="1710000000.123",
            current_ts="1710000000.300",
        )

        assert "Alice: First context" in content
        assert "[thread parent] Bob: Parent message" in content
        assert "peer agent reviewer: Bot output" in content
        assert 'You are "ductor". Lines tagged "peer agent X"' in content
        assert "Current message" not in content
        assert "Future message" not in content

        again = await bot._fetch_thread_context(
            channel_id="C123",
            thread_ts="1710000000.123",
            current_ts="1710000000.300",
        )
        assert again == content
        bot._app.client.conversations_replies.assert_awaited_once()

    async def test_thread_context_cache_is_scoped_to_current_ts(self) -> None:
        bot = _make_bot()
        bot._app.client.conversations_replies.return_value = {
            "messages": [
                {"ts": "1710000000.100", "user": "U111", "text": "Parent"},
                {"ts": "1710000000.200", "user": "U222", "text": "First reply"},
                {"ts": "1710000000.300", "user": "U333", "text": "Second reply"},
            ]
        }
        bot._resolve_user_name = AsyncMock(side_effect=["Alice", "Bob", "Cara"])

        first = await bot._fetch_thread_context(
            channel_id="C123",
            thread_ts="1710000000.100",
            current_ts="1710000000.200",
        )
        second = await bot._fetch_thread_context(
            channel_id="C123",
            thread_ts="1710000000.100",
            current_ts="1710000000.300",
        )

        assert "First reply" not in first
        assert "First reply" in second
        assert bot._app.client.conversations_replies.await_count == 2

    def test_prunes_expired_cached_context_entries(self) -> None:
        bot = _make_bot()
        now = time.monotonic()
        bot._THREAD_CACHE_TTL = 10.0
        bot._thread_context_cache = {
            "expired": _ThreadContextCache(content="old", fetched_at=now - 20.0),
            "fresh": _ThreadContextCache(content="keep", fetched_at=now - 1.0),
        }

        bot._cache_thread_context(cache_key="new", content_parts=["Alice: hi"], fetched_at=now)

        assert "expired" not in bot._thread_context_cache
        assert "fresh" in bot._thread_context_cache
        assert "new" in bot._thread_context_cache

    def test_enforces_thread_context_cache_max_size(self) -> None:
        bot = _make_bot()
        bot._THREAD_CONTEXT_CACHE_MAX_SIZE = 2
        now = time.monotonic()

        bot._cache_thread_context(cache_key="one", content_parts=["a"], fetched_at=now)
        bot._cache_thread_context(cache_key="two", content_parts=["b"], fetched_at=now + 1.0)
        bot._cache_thread_context(cache_key="three", content_parts=["c"], fetched_at=now + 2.0)

        assert list(bot._thread_context_cache) == ["two", "three"]


class TestCommandPresentation:
    async def test_info_uses_slack_i18n_description(self) -> None:
        bot = _make_bot()

        await bot._cmd_info(text="/info", channel_id="C123", key=MagicMock(), thread_ts=None)

        sent_text = bot._send_rich.await_args.args[1]
        assert "Slack" in sent_text
        assert "Matrix" not in sent_text

    def test_help_uses_slack_footer(self) -> None:
        bot = _make_bot()

        help_text = bot._build_help_text()

        assert "`help` or `/help`" in help_text
        assert "`!`" not in help_text


class TestMessageRouting:
    async def test_on_message_wraps_processing_reaction(self) -> None:
        bot = _make_bot()
        bot._add_processing_reaction = AsyncMock(return_value=True)
        bot._remove_processing_reaction = AsyncMock()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "im",
                "ts": "1710000000.456",
                "text": "hello",
            }
        )

        bot._add_processing_reaction.assert_awaited_once_with("C123", "1710000000.456")
        bot._remove_processing_reaction.assert_awaited_once_with("C123", "1710000000.456")

    async def test_app_mention_event_routes_like_message(self) -> None:
        bot = _make_bot()
        bot._on_message = AsyncMock()

        event = {
            "user": "U123",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1710000000.456",
            "text": "<@B123> status",
        }

        await bot._handle_mention_event(event)

        bot._on_message.assert_awaited_once_with(event)

    async def test_dedupes_message_and_app_mention_for_same_slack_post(self) -> None:
        bot = _make_bot()

        event = {
            "user": "U123",
            "channel": "C123",
            "channel_type": "channel",
            "ts": "1710000000.456",
            "text": "<@B123> how big is your memory file",
        }

        await bot._handle_message_event(event)
        await bot._handle_mention_event(event)

    def test_mention_order_delay_is_based_on_unique_mention_position(self) -> None:
        bot = _make_bot()

        delay = bot._mention_order_delay_seconds(
            text="<@B999> first <@B123> second <@B123> repeat",
            is_dm=False,
        )

        assert delay == 0.2

    async def test_delays_processing_for_later_mentions(self) -> None:
        bot = _make_bot()

        with patch("ductor_slack.messenger.slack.bot.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await bot._on_message(
                {
                    "user": "U123",
                    "channel": "C123",
                    "channel_type": "channel",
                    "ts": "1710000000.456",
                    "text": "<@B999> check Neon, <@B123> check Supabase",
                }
            )

        mock_sleep.assert_awaited_once_with(0.2)
        bot._dispatch_with_lock.assert_awaited_once()

    async def test_allows_explicitly_allowlisted_bot_id(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["B456"]

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B456",
                    "channel": "C123",
                    "channel_type": "im",
                    "subtype": "bot_message",
                    "ts": "1710000000.456",
                    "text": "hello from allowed bot",
                }
            )

        bot._dispatch_with_lock.assert_awaited_once()

    async def test_allows_explicitly_allowlisted_app_id(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_app_ids = ["A456"]

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B999",
                    "app_id": "A456",
                    "channel": "C123",
                    "channel_type": "im",
                    "subtype": "bot_message",
                    "ts": "1710000000.456",
                    "text": "hello from allowed app",
                }
            )

        bot._dispatch_with_lock.assert_awaited_once()

    async def test_rejects_unallowlisted_bot_message(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = []
        bot._config.slack.allowed_app_ids = []

        await bot._on_message(
            {
                "bot_id": "B456",
                "channel": "C123",
                "channel_type": "im",
                "subtype": "bot_message",
                "ts": "1710000000.456",
                "text": "hello from blocked bot",
            }
        )

        bot._dispatch_with_lock.assert_not_awaited()

    async def test_rejects_own_bot_message_even_if_allowlisted(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["BOTSELF"]

        await bot._on_message(
            {
                "bot_id": "BOTSELF",
                "channel": "C123",
                "channel_type": "im",
                "subtype": "bot_message",
                "ts": "1710000000.456",
                "text": "loop me maybe",
            }
        )

        bot._dispatch_with_lock.assert_not_awaited()

    async def test_thread_context_keeps_allowlisted_bot_messages(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_bot_ids = ["B456"]

        content, has_peer_agent = await bot._build_thread_context_parts(
            messages=[
                {
                    "ts": "1710000000.100",
                    "bot_id": "B456",
                    "bot_profile": {"name": "Allowed Bot"},
                    "text": "Bot context",
                },
                {"ts": "1710000000.200", "bot_id": "B789", "text": "Blocked context"},
            ],
            channel_id="C123",
            thread_ts="1710000000.100",
            current_ts="1710000000.300",
        )

        assert content == ["[thread parent] peer agent Allowed Bot: Bot context"]
        assert has_peer_agent is True

    async def test_wraps_allowlisted_peer_bot_before_dispatch(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["B456"]

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B456",
                    "bot_profile": {"name": "reviewer"},
                    "channel": "C123",
                    "channel_type": "im",
                    "subtype": "bot_message",
                    "ts": "1710000000.456",
                    "text": "hello from allowed bot",
                }
            )

        bot._dispatch_with_lock.assert_awaited_once()
        wrapped = bot._dispatch_with_lock.await_args.args[1]
        assert 'Message from peer agent "reviewer"' in wrapped
        assert wrapped.endswith("hello from allowed bot")

    async def test_suppresses_peer_reply_after_budget_is_exhausted(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["B456"]
        bot._app.client.conversations_replies.return_value = {
            "messages": [
                {"ts": "1710000000.100", "user": "U123", "text": "不要超过 4 轮"},
                {"ts": "1710000000.200", "bot_id": "BOTSELF", "text": "一"},
                {"ts": "1710000000.300", "bot_id": "B456", "text": "二"},
                {"ts": "1710000000.400", "bot_id": "BOTSELF", "text": "三"},
                {"ts": "1710000000.500", "bot_id": "B456", "text": "四"},
            ]
        }
        bot._extract_peer_turn_budget = AsyncMock(return_value=4)

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B456",
                    "channel": "C123",
                    "channel_type": "channel",
                    "thread_ts": "1710000000.100",
                    "subtype": "bot_message",
                    "ts": "1710000000.500",
                    "text": "四",
                }
            )

        bot._dispatch_with_lock.assert_not_awaited()

    async def test_peer_turn_budget_falls_back_when_no_number_present(self) -> None:
        bot = _make_bot()

        budget = await bot._extract_peer_turn_budget("开始对话")

        assert budget == 2

    async def test_peer_turn_budget_extracts_chinese_round_count(self) -> None:
        bot = _make_bot()

        assert await bot._extract_peer_turn_budget("不要超过 4 轮") == 4
        assert await bot._extract_peer_turn_budget("最多 6 次回复") == 6

    async def test_peer_turn_budget_extracts_english_phrasing(self) -> None:
        bot = _make_bot()

        assert await bot._extract_peer_turn_budget("max_turns: 7") == 7
        assert await bot._extract_peer_turn_budget("no more than 3 turns please") == 3
        assert await bot._extract_peer_turn_budget("limit 5 rounds") == 5

    async def test_peer_turn_budget_caps_extracted_value(self) -> None:
        bot = _make_bot()

        assert await bot._extract_peer_turn_budget("max_turns 9999") == 50

    async def test_counts_peer_turns_when_thread_snapshot_lacks_app_id(self) -> None:
        # Slack's conversations.replies sometimes omits bot_profile/app_id even
        # though the live message event carried it. The peer-turn counter must
        # still recognise those messages as bot activity so the budget kicks in.
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = []
        bot._config.slack.allowed_app_ids = ["A_PEER"]
        bot._app.client.conversations_replies.return_value = {
            "messages": [
                {"ts": "1710000000.100", "user": "U123", "text": "聊 4 轮"},
                {"ts": "1710000000.200", "bot_id": "BOTSELF", "text": "一"},
                {"ts": "1710000000.300", "bot_id": "B_PEER", "text": "二"},
                {"ts": "1710000000.400", "bot_id": "BOTSELF", "text": "三"},
                {"ts": "1710000000.500", "bot_id": "B_PEER", "text": "四"},
            ]
        }

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B_PEER",
                    "app_id": "A_PEER",
                    "channel": "C123",
                    "channel_type": "channel",
                    "thread_ts": "1710000000.100",
                    "subtype": "bot_message",
                    "ts": "1710000000.500",
                    "text": "四",
                }
            )

        bot._dispatch_with_lock.assert_not_awaited()

    async def test_unallowlisted_peer_bot_does_not_reset_anchor(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = ["U123"]
        bot._config.slack.allowed_bot_ids = ["B456"]
        bot._app.client.conversations_replies.return_value = {
            "messages": [
                {"ts": "1710000000.100", "user": "U123", "text": "聊 2 轮"},
                {"ts": "1710000000.200", "bot_id": "BOTSELF", "text": "一"},
                {"ts": "1710000000.300", "bot_id": "B999", "text": "stranger bot chimes in"},
                {"ts": "1710000000.400", "bot_id": "B456", "text": "二"},
            ]
        }

        async with _instant_peer_debounce(bot):
            await bot._on_message(
                {
                    "bot_id": "B456",
                    "channel": "C123",
                    "channel_type": "channel",
                    "thread_ts": "1710000000.100",
                    "subtype": "bot_message",
                    "ts": "1710000000.400",
                    "text": "二",
                }
            )

        bot._dispatch_with_lock.assert_not_awaited()

    async def test_debounces_peer_bot_message_changed_to_final_text(self) -> None:
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["B456"]
        gate = asyncio.Event()

        async def _sleep(_delay: float) -> None:
            await gate.wait()

        first_event = {
            "subtype": "message_changed",
            "channel": "C123",
            "message": {
                "bot_id": "B456",
                "bot_profile": {"name": "reviewer"},
                "channel": "C123",
                "channel_type": "im",
                "subtype": "bot_message",
                "ts": "1710000000.456",
                "text": "draft",
            },
        }
        second_event = {
            "subtype": "message_changed",
            "channel": "C123",
            "message": {
                "bot_id": "B456",
                "bot_profile": {"name": "reviewer"},
                "channel": "C123",
                "channel_type": "im",
                "subtype": "bot_message",
                "ts": "1710000000.456",
                "text": "final answer",
            },
        }

        with patch("ductor_slack.messenger.slack.bot.asyncio.sleep", side_effect=_sleep):
            await bot._on_message(first_event)
            first_task = bot._pending_peer_edit_tasks[("C123", "1710000000.456")]
            await bot._on_message(second_event)
            second_task = bot._pending_peer_edit_tasks[("C123", "1710000000.456")]
            gate.set()
            await second_task
            with contextlib.suppress(asyncio.CancelledError):
                await first_task

        bot._dispatch_with_lock.assert_awaited_once()
        wrapped = bot._dispatch_with_lock.await_args.args[1]
        assert wrapped.endswith("final answer")
        assert "draft" not in wrapped

    async def test_peer_bot_initial_post_then_edit_dispatches_once(self) -> None:
        # Streaming bots first POST a placeholder then EDIT it. The receiver
        # used to dispatch twice (once for the initial event, once for the
        # debounced final edit). With the unified debounce, the initial post
        # is treated like an edit and only the final stable text dispatches.
        bot = _make_bot()
        bot._config.slack.allowed_users = []
        bot._config.slack.allowed_bot_ids = ["B456"]
        gate = asyncio.Event()

        async def _sleep(_delay: float) -> None:
            await gate.wait()

        initial_event = {
            "bot_id": "B456",
            "bot_profile": {"name": "reviewer"},
            "channel": "C123",
            "channel_type": "im",
            "subtype": "bot_message",
            "ts": "1710000000.789",
            "text": "Working on your request",
        }
        edit_event = {
            "subtype": "message_changed",
            "channel": "C123",
            "message": {
                "bot_id": "B456",
                "bot_profile": {"name": "reviewer"},
                "channel": "C123",
                "channel_type": "im",
                "subtype": "bot_message",
                "ts": "1710000000.789",
                "text": "actual reply",
            },
        }

        with patch("ductor_slack.messenger.slack.bot.asyncio.sleep", side_effect=_sleep):
            await bot._on_message(initial_event)
            first_task = bot._pending_peer_edit_tasks[("C123", "1710000000.789")]
            await bot._on_message(edit_event)
            second_task = bot._pending_peer_edit_tasks[("C123", "1710000000.789")]
            gate.set()
            await second_task
            with contextlib.suppress(asyncio.CancelledError):
                await first_task

        bot._dispatch_with_lock.assert_awaited_once()
        wrapped = bot._dispatch_with_lock.await_args.args[1]
        assert wrapped.endswith("actual reply")
        assert "Working on your request" not in wrapped

    async def test_backfills_first_thread_reply_after_mention(self) -> None:
        bot = _make_bot()
        bot._fetch_thread_context = AsyncMock(return_value="[ctx]\n")

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "channel",
                "thread_ts": "1710000000.123",
                "ts": "1710000000.456",
                "text": "<@B123> help here",
            }
        )

        bot._dispatch_with_lock.assert_awaited_once()
        assert bot._dispatch_with_lock.await_args.args[1] == "[ctx]\nhelp here"
        assert ("C123", "1710000000.123") in bot._mentioned_threads

    async def test_existing_thread_session_routes_without_mention(self) -> None:
        bot = _make_bot()
        active = SessionData(chat_id=11, transport="sl", topic_id=22, provider_sessions={})
        active.session_id = "sid-1"
        bot._orchestrator._sessions.list_active_for_chat.return_value = [active]
        bot._fetch_thread_context = AsyncMock(return_value="[ctx]\n")

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "channel",
                "thread_ts": "1710000000.123",
                "ts": "1710000000.456",
                "text": "follow-up without mention",
            }
        )

        bot._dispatch_with_lock.assert_awaited_once()
        bot._fetch_thread_context.assert_awaited_once()
        assert bot._dispatch_with_lock.await_args.args[1] == "[ctx]\nfollow-up without mention"

    async def test_non_dm_messages_force_fresh_session(self) -> None:
        bot = _make_bot()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "channel",
                "ts": "1710000000.456",
                "text": "<@B123> help here",
            }
        )

        assert bot._dispatch_with_lock.await_args.kwargs["force_fresh_session"] is True

    async def test_dm_messages_keep_session_resume(self) -> None:
        bot = _make_bot()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "D123",
                "channel_type": "im",
                "ts": "1710000000.456",
                "text": "help here",
            }
        )

        assert bot._dispatch_with_lock.await_args.kwargs["force_fresh_session"] is False

    async def test_routes_bare_message_command_without_leading_slash(self) -> None:
        bot = _make_bot()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "im",
                "ts": "1710000000.456",
                "text": "status",
            }
        )

        bot._handle_command.assert_awaited_once()
        assert bot._handle_command.await_args.args[0] == "/status"
        bot._dispatch_with_lock.assert_not_awaited()

    async def test_routes_bare_message_command_with_arguments_when_supported(self) -> None:
        bot = _make_bot()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "im",
                "ts": "1710000000.456",
                "text": "model gpt-5.4",
            }
        )

        bot._handle_command.assert_awaited_once()
        assert bot._handle_command.await_args.args[0] == "/model gpt-5.4"

    async def test_non_command_text_with_extra_words_stays_a_message(self) -> None:
        bot = _make_bot()

        await bot._on_message(
            {
                "user": "U123",
                "channel": "C123",
                "channel_type": "im",
                "ts": "1710000000.456",
                "text": "help me debug this",
            }
        )

        bot._handle_command.assert_not_awaited()
        bot._dispatch_with_lock.assert_awaited_once()

    async def test_run_streaming_updates_single_slack_message(self) -> None:
        bot = _make_bot()
        bot._config.streaming.edit_interval_seconds = 0.0
        streamer = MagicMock()
        streamer.append = AsyncMock()
        streamer.stop = AsyncMock()
        bot._app.client.chat_stream = AsyncMock(return_value=streamer)

        async def _fake_stream(
            key: object,
            text: str,
            *,
            force_fresh_session: bool = False,
            on_text_delta: object = None,
            on_thinking_delta: object = None,
            on_tool_activity: object = None,
            on_system_status: object = None,
        ) -> OrchestratorResult:
            assert key is not None
            assert text == "hello"
            assert force_fresh_session is False
            assert on_thinking_delta is not None
            assert on_text_delta is not None
            await on_thinking_delta("step 1")
            await on_tool_activity(
                ToolUseEvent(
                    type="assistant",
                    tool_name="ToolSearch",
                    parameters={"query": "slack thinking steps ai agents"},
                )
            )
            await on_tool_activity(
                ToolUseEvent(
                    type="assistant",
                    tool_name="WebFetch",
                    parameters={"url": "https://slack.dev/slack-thinking-steps-ai-agents/"},
                )
            )
            await on_text_delta("final")
            await on_system_status(None)
            return OrchestratorResult(text="final")

        bot._orchestrator.handle_message_streaming = _fake_stream

        await bot._run_streaming(
            MagicMock(),
            "hello",
            "C123",
            "1710000000.123",
            recipient_user_id="U123",
        )

        bot._app.client.chat_stream.assert_awaited_once_with(
            channel="C123",
            thread_ts="1710000000.123",
            recipient_team_id="T123",
            recipient_user_id="U123",
            task_display_mode="plan",
            buffer_size=64,
        )
        # Option A: reasoning never reaches the message body — only the answer does.
        markdown_writes = [
            call.kwargs.get("markdown_text", "")
            for call in streamer.append.await_args_list
            if call.kwargs.get("markdown_text")
        ]
        assert markdown_writes == ["final"]
        assert all("💭 *Thinking*" not in md for md in markdown_writes)
        # Gather every plan/task chunk across the append stream and the final stop.
        all_chunks = [
            chunk
            for call in streamer.append.await_args_list
            if call.kwargs.get("chunks")
            for chunk in call.kwargs["chunks"]
        ]
        all_chunks += list(streamer.stop.await_args.kwargs.get("chunks") or [])

        plan_updates = [c for c in all_chunks if c["type"] == "plan_update"]
        assert len(plan_updates) == 1
        assert plan_updates[0]["title"] == "Working on your request"

        def statuses(task_id: str) -> list[str]:
            return [c["status"] for c in all_chunks if c.get("id") == task_id]

        assert statuses("analyze")[0] == "in_progress"
        assert statuses("analyze")[-1] == "complete"
        assert "step 1" in [c.get("details") for c in all_chunks if c.get("id") == "analyze"]

        # Each tool call is its own row (full record), not folded into one "tools" task.
        assert all(c.get("id") != "tools" for c in all_chunks)
        titles = {c["id"]: c["title"] for c in all_chunks if str(c.get("id", "")).startswith("tool-")}
        assert titles["tool-1"] == "Search: slack thinking steps ai agents"
        assert titles["tool-2"] == "Web fetch: slack.dev/slack-thinking-steps-ai-agents"
        assert statuses("tool-1")[-1] == "complete"
        assert statuses("tool-2")[-1] == "complete"

        assert statuses("respond")[-1] == "complete"
        streamer.stop.assert_awaited_once()

    async def test_run_streaming_in_dm_omits_recipient_context(self) -> None:
        bot = _make_bot()
        streamer = MagicMock()
        streamer.append = AsyncMock()
        streamer.stop = AsyncMock()
        bot._app.client.chat_stream = AsyncMock(return_value=streamer)
        bot._orchestrator.handle_message_streaming = AsyncMock(
            return_value=OrchestratorResult(text="ok")
        )

        await bot._run_streaming(
            MagicMock(),
            "hello",
            "D123",
            "1710000000.123",
            recipient_user_id="U123",
        )

        # No callbacks fired → no plan activity, so the bare turn streams a plain
        # message (no task_display_mode). In a DM, recipient context is omitted.
        bot._app.client.chat_stream.assert_awaited_once_with(
            channel="D123",
            thread_ts="1710000000.123",
            buffer_size=64,
        )

    async def test_run_streaming_can_force_fresh_session(self) -> None:
        bot = _make_bot()
        streamer = MagicMock()
        streamer.append = AsyncMock()
        streamer.stop = AsyncMock()
        bot._app.client.chat_stream = AsyncMock(return_value=streamer)
        bot._orchestrator.handle_message_streaming = AsyncMock(
            return_value=OrchestratorResult(text="ok")
        )

        await bot._run_streaming(
            MagicMock(),
            "hello",
            "C123",
            "1710000000.123",
            recipient_user_id="U123",
            force_fresh_session=True,
        )

        assert (
            bot._orchestrator.handle_message_streaming.await_args.kwargs["force_fresh_session"]
            is True
        )

    async def test_run_streaming_falls_back_when_native_streaming_fails(self) -> None:
        bot = _make_bot()
        streamer = MagicMock()
        streamer.append = AsyncMock(side_effect=RuntimeError("boom"))
        streamer.stop = AsyncMock()
        bot._app.client.chat_stream = AsyncMock(return_value=streamer)

        async def _fake_stream(
            key: object,
            text: str,
            *,
            force_fresh_session: bool = False,
            on_text_delta: object = None,
            on_thinking_delta: object = None,
            on_tool_activity: object = None,
            on_system_status: object = None,
        ) -> OrchestratorResult:
            assert key is not None
            assert text == "hello"
            assert force_fresh_session is False
            assert on_thinking_delta is not None
            assert on_text_delta is not None
            await on_thinking_delta("step 1")
            await on_tool_activity(
                ToolUseEvent(type="assistant", tool_name="bash", parameters={"cmd": "pwd"})
            )
            await on_text_delta("final")
            await on_system_status(None)
            return OrchestratorResult(text="final")

        bot._orchestrator.handle_message_streaming = _fake_stream

        with patch(
            "ductor_slack.messenger.slack.streaming.send_rich",
            new_callable=AsyncMock,
        ) as mock_send_rich:
            await bot._run_streaming(
                MagicMock(),
                "hello",
                "C123",
                "1710000000.123",
                recipient_user_id="U123",
            )

        mock_send_rich.assert_awaited_once()
        sent_text = mock_send_rich.await_args.args[2]
        assert "💭 *Thinking*" in sent_text
        assert "final" in sent_text
        streamer.stop.assert_not_awaited()

    async def test_run_non_streaming_can_force_fresh_session(self) -> None:
        bot = _make_bot()
        bot._orchestrator.handle_message = AsyncMock(return_value=OrchestratorResult(text="ok"))

        await bot._run_non_streaming(
            MagicMock(),
            "hello",
            "C123",
            "1710000000.123",
            force_fresh_session=True,
        )

        assert bot._orchestrator.handle_message.await_args.kwargs["force_fresh_session"] is True
