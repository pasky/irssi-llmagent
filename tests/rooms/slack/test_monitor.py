"""Tests for Slack room monitor behavior."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from muaddib.main import MuaddibAgent
from muaddib.rooms.slack.monitor import SlackRoomMonitor


def build_monitor(test_config):
    agent = SimpleNamespace()
    agent.config = test_config
    agent.history = AsyncMock()
    agent.history.add_message = AsyncMock(return_value=1)
    agent.history.get_message_id_by_platform_id = AsyncMock(return_value=None)

    monitor = SlackRoomMonitor(cast(MuaddibAgent, agent))
    monitor.command_handler.handle_command = AsyncMock()
    monitor.command_handler.handle_passive_message = AsyncMock()

    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_update = AsyncMock()

    monitor._get_client = AsyncMock(return_value=client)
    monitor._get_bot_user_id = AsyncMock(return_value="B1")
    monitor._get_channel_name = AsyncMock(return_value="general")

    async def display_name(team_id: str, client_ref, user_id: str) -> str:
        return "Muaddib" if user_id == "B1" else "pasky"

    monitor._get_user_display_name = AsyncMock(side_effect=display_name)
    return monitor, agent, client


@pytest.mark.asyncio
async def test_slack_mention_triggers_command_and_threads_reply(test_config):
    monitor, agent, client = build_monitor(test_config)

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("hello")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "app_mention",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> hi there",
        "ts": "1700000000.1234",
    }

    await monitor.process_message_event(body, event, is_direct=True)

    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_command.assert_awaited_once()
    kwargs = handle_command.call_args.kwargs
    assert kwargs["server_tag"] == "slack:Rossum"
    assert kwargs["channel_name"] == "#general"
    assert kwargs["message"] == "hi there"
    assert kwargs["response_thread_id"] == "1700000000.1234"

    client.chat_postMessage.assert_awaited_once()
    _, send_kwargs = client.chat_postMessage.call_args
    assert send_kwargs["channel"] == "C123"
    assert send_kwargs["text"] == "hello"
    assert send_kwargs["thread_ts"] == "1700000000.1234"


@pytest.mark.asyncio
async def test_slack_dm_does_not_start_thread_by_default(test_config):
    monitor, agent, client = build_monitor(test_config)

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("hello")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "D123",
        "channel_type": "im",
        "user": "U1",
        "text": "hello",
        "ts": "1700000000.2222",
    }

    await monitor.process_message_event(body, event, is_direct=True)

    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_command.assert_awaited_once()
    kwargs = handle_command.call_args.kwargs
    assert kwargs["channel_name"] == "pasky_U1"
    assert kwargs["response_thread_id"] is None

    client.chat_postMessage.assert_awaited_once()
    _, send_kwargs = client.chat_postMessage.call_args
    assert send_kwargs["channel"] == "D123"
    assert send_kwargs["text"] == "hello"
    assert "thread_ts" not in send_kwargs


@pytest.mark.asyncio
async def test_slack_passive_message_routes_to_proactive(test_config):
    monitor, agent, client = build_monitor(test_config)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "hello",
        "ts": "1700000000.3333",
    }

    await monitor.process_message_event(body, event, is_direct=False)

    handle_passive = cast(AsyncMock, monitor.command_handler.handle_passive_message)
    handle_passive.assert_awaited_once()
    kwargs = handle_passive.call_args.kwargs
    assert kwargs["message"] == "hello"
    assert kwargs["channel_name"] == "#general"


@pytest.mark.asyncio
async def test_slack_attachments_include_secrets_and_block(test_config):
    monitor, agent, client = build_monitor(test_config)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "hello",
        "ts": "1700000000.4444",
        "files": [
            {
                "mimetype": "image/png",
                "name": "cat.png",
                "size": 1234,
                "url_private": "https://files.slack.com/files-pri/T123/cat.png",
            }
        ],
    }

    await monitor.process_message_event(body, event, is_direct=False)

    handle_passive = cast(AsyncMock, monitor.command_handler.handle_passive_message)
    handle_passive.assert_awaited_once()
    kwargs = handle_passive.call_args.kwargs

    assert kwargs["message"] == (
        "hello\n\n"
        "[Attachments]\n"
        "1. image/png (filename: cat.png) (size: 1234): "
        "https://files.slack.com/files-pri/T123/cat.png\n"
        "[/Attachments]"
    )
    secrets = kwargs["secrets"]
    assert secrets["http_header_prefixes"]["https://files.slack.com/"]["Authorization"] == (
        "Bearer xoxb-mock-token"
    )


@pytest.mark.asyncio
async def test_slack_reply_edit_debounce_combines_messages(test_config):
    monitor, agent, client = build_monitor(test_config)
    monitor._now = MagicMock(side_effect=[0.0, 1.0])

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("first")
        await kwargs["reply_sender"]("second")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "app_mention",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> hello",
        "ts": "1700000000.5555",
    }

    await monitor.process_message_event(body, event, is_direct=True)

    client.chat_postMessage.assert_awaited_once()
    client.chat_update.assert_awaited_once()
    _, update_kwargs = client.chat_update.call_args
    assert update_kwargs["text"] == "first\nsecond"


@pytest.mark.asyncio
async def test_slack_ignores_own_messages(test_config):
    monitor, agent, client = build_monitor(test_config)
    monitor._get_bot_user_id = AsyncMock(return_value="U1")

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "hello",
        "ts": "1700000000.6666",
    }

    await monitor.process_message_event(body, event, is_direct=False)

    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_passive = cast(AsyncMock, monitor.command_handler.handle_passive_message)
    handle_command.assert_not_awaited()
    handle_passive.assert_not_awaited()
    agent.history.add_message.assert_not_awaited()
