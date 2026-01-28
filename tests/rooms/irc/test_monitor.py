"""Tests for IRC monitor functionality."""

from unittest.mock import AsyncMock

import pytest

from muaddib.main import MuaddibAgent


class TestIRCMonitor:
    """Test IRC monitor IRC-specific functionality."""

    @pytest.mark.asyncio
    async def test_get_mynick_caching(self, temp_config_file):
        agent = MuaddibAgent(temp_config_file)

        mock_sender = AsyncMock()
        mock_sender.get_server_nick.return_value = "testbot"
        agent.irc_monitor.varlink_sender = mock_sender

        nick1 = await agent.irc_monitor.get_mynick("irc.libera.chat")
        assert nick1 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1

        nick2 = await agent.irc_monitor.get_mynick("irc.libera.chat")
        assert nick2 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1

    @pytest.mark.asyncio
    async def test_message_addressing_detection(self, temp_config_file):
        agent = MuaddibAgent(temp_config_file)
        agent.irc_monitor.server_nicks["test"] = "mybot"

        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.irc_monitor.command_handler.handle_command = AsyncMock()

        event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "testuser",
            "message": "mybot: hello there",
        }

        await agent.irc_monitor.process_message_event(event)

        agent.irc_monitor.command_handler.handle_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_privmsg_commands_without_nick_prefix(self, temp_config_file):
        agent = MuaddibAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.command_handler.handle_command = AsyncMock()
        agent.irc_monitor.server_nicks["test"] = "mybot"

        event = {
            "type": "message",
            "subtype": "private",
            "server": "test",
            "target": "mybot",
            "nick": "testuser",
            "message": "hello there",
        }

        await agent.irc_monitor.process_message_event(event)

        agent.irc_monitor.command_handler.handle_command.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_channel_messages_without_nick_prefix_ignored(self, temp_config_file):
        agent = MuaddibAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.server_nicks["test"] = "mybot"
        agent.irc_monitor.command_handler.handle_command = AsyncMock()
        agent.irc_monitor.command_handler.handle_passive_message = AsyncMock()

        event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "testuser",
            "message": "just a regular message",
        }

        await agent.irc_monitor.process_message_event(event)

        agent.irc_monitor.command_handler.handle_command.assert_not_called()
        agent.irc_monitor.command_handler.handle_passive_message.assert_awaited_once()
