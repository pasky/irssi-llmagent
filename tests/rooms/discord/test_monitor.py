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
    agent.history.get_arc_cost_today = AsyncMock(return_value=0)
    agent.chronicle = AsyncMock()
    agent.model_router = AsyncMock()

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))
    monitor.command_handler.rate_limiter = MagicMock()
    monitor.command_handler.rate_limiter.check_limit.return_value = True
    monitor.command_handler.autochronicler.check_and_chronicle = AsyncMock(return_value=False)
    monitor.command_handler._run_actor = AsyncMock(
        return_value=AgentResult(
            text="Pasky: hello there",
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            tool_calls_count=0,
            primary_model=None,
        )
    )

    message = MagicMock()
    message.reply = AsyncMock()
    message.author.bot = False
    message.author.display_name = "pasky"
    message.author.id = 1
    message.guild = None
    message.clean_content = "!s hi"
    message.content = "!s hi"
    message.id = 123
    message.channel = MagicMock()

    bot_user = MagicMock()
    bot_user.display_name = "Muaddib"
    bot_user.id = 999
    monitor.client._connection.user = bot_user

    await monitor.process_message_event(message)

    message.reply.assert_awaited_once()
    reply_args, reply_kwargs = message.reply.call_args
    assert reply_args[0] == "hello there"
    assert reply_kwargs["mention_author"] is True

    history_args, history_kwargs = agent.history.add_message.call_args
    assert history_args[0] == "discord:_DM"
    assert history_args[1] == "pasky_1"
    assert history_args[2] == "hello there"
    assert history_kwargs["mode"] == "EASY_SERIOUS"


@pytest.mark.asyncio
async def test_discord_sender_chains_replies(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.add_message = AsyncMock(return_value=1)

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))

    original_message = MagicMock()
    first_reply = MagicMock()
    second_reply = MagicMock()
    original_message.reply = AsyncMock(return_value=first_reply)
    first_reply.reply = AsyncMock(return_value=second_reply)

    async def handle_command(**kwargs):
        reply_sender = kwargs["reply_sender"]
        await reply_sender("first")
        await reply_sender("second")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    bot_user = MagicMock()
    bot_user.display_name = "Muaddib"
    bot_user.id = 999
    monitor.client._connection.user = bot_user

    original_message.author.display_name = "pasky"
    original_message.author.id = 1
    original_message.author.bot = False
    original_message.guild = None
    original_message.clean_content = "hello"
    original_message.content = "hello"
    original_message.id = 321

    await monitor.process_message_event(original_message)

    original_message.reply.assert_awaited_once_with("first", mention_author=True)
    first_reply.reply.assert_awaited_once_with("second", mention_author=False)


@pytest.mark.asyncio
async def test_discord_attachments_are_appended_to_message_text(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.add_message = AsyncMock(return_value=1)

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))
    monitor.command_handler.handle_passive_message = AsyncMock()

    bot_user = MagicMock()
    bot_user.display_name = "Muaddib"
    bot_user.id = 999
    monitor.client._connection.user = bot_user

    attachment = MagicMock()
    attachment.content_type = "image/png"
    attachment.filename = "cat.png"
    attachment.size = 1234
    attachment.url = "https://cdn.discordapp.com/example/cat.png"

    message = MagicMock()
    message.author.display_name = "pasky"
    message.author.id = 1
    message.author.bot = False
    message.guild = MagicMock()
    message.clean_content = "hello"
    message.content = "hello"
    message.id = 456
    message.attachments = [attachment]
    message.channel = MagicMock()
    message.mentions = []

    await monitor.process_message_event(message)

    monitor.command_handler.handle_passive_message.assert_awaited_once()
    kwargs = monitor.command_handler.handle_passive_message.call_args.kwargs

    assert kwargs["message"] == (
        "hello\n\n"
        "[Attachments]\n"
        "1. image/png (filename: cat.png) (size: 1234): https://cdn.discordapp.com/example/cat.png\n"
        "[/Attachments]"
    )


@pytest.mark.asyncio
async def test_discord_custom_emoji_are_normalized(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.add_message = AsyncMock(return_value=1)

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))
    monitor.command_handler.handle_passive_message = AsyncMock()

    bot_user = MagicMock()
    bot_user.display_name = "Muaddib"
    bot_user.id = 999
    monitor.client._connection.user = bot_user

    message = MagicMock()
    message.author.display_name = "pasky"
    message.author.id = 1
    message.author.bot = False
    message.guild = MagicMock()
    message.clean_content = "<a:blobDance:590880199063896084> wow <:blobWave:1234>"
    message.content = "<a:blobDance:590880199063896084> wow <:blobWave:1234>"
    message.id = 789
    message.attachments = []
    message.channel = MagicMock()
    message.mentions = []

    await monitor.process_message_event(message)

    monitor.command_handler.handle_passive_message.assert_awaited_once()
    kwargs = monitor.command_handler.handle_passive_message.call_args.kwargs
    assert kwargs["message"] == ":blobDance: wow :blobWave:"


async def test_discord_ignores_own_messages(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.add_message = AsyncMock(return_value=1)

    monitor = DiscordRoomMonitor(cast(MuaddibAgent, agent))
    monitor.command_handler.handle_command = AsyncMock()
    monitor.command_handler.handle_passive_message = AsyncMock()

    bot_user = MagicMock()
    bot_user.display_name = "Muaddib"
    bot_user.id = 999
    monitor.client._connection.user = bot_user

    message = MagicMock()
    message.author.display_name = "Muaddib"
    message.author.id = 999
    message.author.bot = True
    message.guild = None
    message.clean_content = "hello"
    message.content = "hello"
    message.id = 900
    message.channel = MagicMock()

    await monitor.process_message_event(message)

    agent.history.add_message.assert_not_awaited()
    monitor.command_handler.handle_command.assert_not_awaited()
    monitor.command_handler.handle_passive_message.assert_not_awaited()
