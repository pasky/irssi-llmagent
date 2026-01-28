"""Tests for Discord room monitor behavior."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from muaddib.agentic_actor.actor import AgentResult
from muaddib.main import MuaddibAgent
from muaddib.rooms.discord.monitor import DiscordRoomMonitor


@pytest.mark.asyncio
async def test_discord_reply_mentions_author_and_strips_prefix(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.get_context.return_value = [{"role": "user", "content": "hi"}]
    agent.history.add_message = AsyncMock(return_value=1)
    agent.history.log_llm_call = AsyncMock(return_value=1)
    agent.history.update_llm_call_response = AsyncMock()
    agent.irc_monitor = SimpleNamespace()
    agent.irc_monitor._run_actor = AsyncMock(
        return_value=AgentResult(
            text="Pasky: hello there",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )
    )

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))
    monitor.rate_limiter = MagicMock()
    monitor.rate_limiter.check_limit.return_value = True

    message = MagicMock()
    message.reply = AsyncMock()

    await monitor.handle_highlight(
        message,
        server_tag="discord:_DM",
        channel_name="pasky_1",
        arc="discord:_DM#pasky_1",
        nick="pasky",
        mynick="Muaddib",
        trigger_message_id=1,
    )

    message.reply.assert_awaited_once()
    reply_args, reply_kwargs = message.reply.call_args
    assert reply_args[0] == "hello there"
    assert reply_kwargs["mention_author"] is True

    history_args, history_kwargs = agent.history.add_message.call_args
    assert history_args[0] == "discord:_DM"
    assert history_args[1] == "pasky_1"
    assert history_args[2] == "hello there"
    assert history_kwargs["mode"] == "EASY_SERIOUS"
