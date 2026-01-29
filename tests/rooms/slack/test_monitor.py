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


@pytest.mark.asyncio
async def test_slack_mention_with_files_triggers_command(test_config):
    """Test that mentions with file attachments are processed correctly.

    This ensures files are included when bot is mentioned (previously app_mention
    events could miss files, but now we detect mentions in message events).
    """
    monitor, agent, client = build_monitor(test_config)

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("I see your image!")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> what's in this image?",
        "ts": "1700000000.7777",
        "files": [
            {
                "mimetype": "image/png",
                "name": "screenshot.png",
                "size": 5678,
                "url_private": "https://files.slack.com/files-pri/T123/screenshot.png",
            }
        ],
    }

    await monitor.process_message_event(body, event, is_direct=True)

    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_command.assert_awaited_once()
    kwargs = handle_command.call_args.kwargs

    # Verify the message includes the attachment block
    assert "[Attachments]" in kwargs["message"]
    assert "screenshot.png" in kwargs["message"]
    assert "https://files.slack.com/files-pri/T123/screenshot.png" in kwargs["message"]

    # Verify secrets are passed for file access
    assert kwargs["secrets"] is not None
    assert "http_header_prefixes" in kwargs["secrets"]


@pytest.mark.asyncio
async def test_slack_handle_message_detects_mention(test_config):
    """Test that _handle_message correctly detects bot mentions and routes to command handler."""
    monitor, agent, client = build_monitor(test_config)
    monitor.bot_user_ids["T123"] = "B1"

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("hello")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> hi there",
        "ts": "1700000000.8888",
    }

    # Call _handle_message directly (simulating Slack event)
    await monitor._handle_message(body, event, AsyncMock())

    # Should route to handle_command (is_direct=True) because of mention
    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_slack_handle_message_no_mention_routes_passive(test_config):
    """Test that _handle_message routes non-mentions to passive handler."""
    monitor, agent, client = build_monitor(test_config)
    monitor.bot_user_ids["T123"] = "B1"

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "just a regular message",
        "ts": "1700000000.9999",
    }

    # Call _handle_message directly
    await monitor._handle_message(body, event, AsyncMock())

    # Should route to handle_passive_message (is_direct=False)
    handle_passive = cast(AsyncMock, monitor.command_handler.handle_passive_message)
    handle_passive.assert_awaited_once()


@pytest.mark.asyncio
async def test_slack_reply_formats_mentions(test_config):
    """Test that @DisplayName in replies gets converted to <@USER_ID> for Slack."""
    monitor, agent, client = build_monitor(test_config)
    # Populate the user_id_cache (reverse lookup)
    monitor.user_id_cache["T123"] = {"petr baudis": "U1", "muaddib": "B1"}

    async def handle_command(**kwargs):
        # Response includes @mention that should be converted
        await kwargs["reply_sender"]("@Petr Baudis, here's your answer!")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> hello",
        "ts": "1700000000.1111",
    }

    await monitor.process_message_event(body, event, is_direct=True)

    client.chat_postMessage.assert_awaited_once()
    _, send_kwargs = client.chat_postMessage.call_args
    # The @Petr Baudis should be converted to <@U1>
    assert send_kwargs["text"] == "<@U1>, here's your answer!"


@pytest.mark.asyncio
async def test_slack_file_share_subtype_is_processed(test_config):
    """Test that file_share subtype messages are processed (not filtered out)."""
    monitor, agent, client = build_monitor(test_config)
    monitor.bot_user_ids["T123"] = "B1"

    async def handle_command(**kwargs):
        await kwargs["reply_sender"]("I see your file!")

    monitor.command_handler.handle_command = AsyncMock(side_effect=handle_command)

    body = {"team_id": "T123"}
    event = {
        "type": "message",
        "subtype": "file_share",
        "channel": "C123",
        "channel_type": "channel",
        "user": "U1",
        "text": "<@B1> check this out",
        "ts": "1700000000.2222",
        "files": [
            {
                "mimetype": "image/png",
                "name": "screenshot.png",
                "url_private": "https://files.slack.com/files-pri/T123/screenshot.png",
            }
        ],
    }

    # Call _handle_message directly (simulates Slack event)
    await monitor._handle_message(body, event, AsyncMock())

    # Should be processed as a command (is_direct=True due to mention)
    handle_command = cast(AsyncMock, monitor.command_handler.handle_command)
    handle_command.assert_awaited_once()
    kwargs = handle_command.call_args.kwargs
    assert "[Attachments]" in kwargs["message"]
    assert "screenshot.png" in kwargs["message"]
