"""Tests for main application functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent


class TestIRSSILLMAgent:
    """Test main agent functionality."""

    def test_load_config(self, temp_config_file):
        """Test configuration loading."""
        agent = IRSSILLMAgent(temp_config_file)
        assert agent.config is not None
        assert "anthropic" in agent.config
        assert "varlink" in agent.config

    def test_should_ignore_user(self, temp_config_file):
        """Test user ignoring functionality."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.config["behavior"]["ignore_users"] = ["spammer", "BadBot"]

        assert agent.should_ignore_user("spammer") is True
        assert agent.should_ignore_user("SPAMMER") is True  # Case insensitive
        assert agent.should_ignore_user("gooduser") is False

    @pytest.mark.asyncio
    async def test_get_mynick_caching(self, temp_config_file):
        """Test that bot nick is cached per server."""
        agent = IRSSILLMAgent(temp_config_file)

        # Mock the varlink sender
        mock_sender = AsyncMock()
        mock_sender.get_server_nick.return_value = "testbot"
        agent.varlink_sender = mock_sender

        # First call should query the server
        nick1 = await agent.get_mynick("irc.libera.chat")
        assert nick1 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1

        # Second call should use cache
        nick2 = await agent.get_mynick("irc.libera.chat")
        assert nick2 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_message_addressing_detection(self, temp_config_file):
        """Test that messages addressing the bot are detected correctly."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.server_nicks["test"] = "mybot"

        # Mock dependencies
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(return_value=[])

        # Mock API clients
        with patch("irssi_llmagent.main.AnthropicClient") as mock_claude:
            mock_claude.return_value.__aenter__.return_value.call_claude = AsyncMock(
                return_value="Test response"
            )

            # Test message addressing the bot
            event = {
                "type": "message",
                "subtype": "public",
                "server": "test",
                "target": "#test",
                "nick": "testuser",
                "message": "mybot: hello there",
            }

            await agent.process_message_event(event)

            # Should call handle_command
            agent.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_help_command(self, temp_config_file):
        """Test help command functionality."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()

        await agent.handle_command("test", "#test", "#test", "user", "!h", "mybot")

        # Should send help message
        agent.varlink_sender.send_message.assert_called_once()
        call_args = agent.varlink_sender.send_message.call_args[0]
        assert "Claude" in call_args[1]  # Help text should mention Claude

    @pytest.mark.asyncio
    async def test_rate_limiting_triggers(self, temp_config_file):
        """Test that rate limiting prevents excessive requests."""
        agent = IRSSILLMAgent(temp_config_file)
        # Mock rate limiter to simulate limit exceeded
        agent.rate_limiter.check_limit = lambda: False
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()

        event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "testuser",
            "message": "mybot: hello",
        }

        agent.server_nicks["test"] = "mybot"
        await agent.process_message_event(event)

        # Should send rate limiting message
        agent.varlink_sender.send_message.assert_called_once()
        call_args = agent.varlink_sender.send_message.call_args[0]
        assert "rate limiting" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_serious_vs_sarcastic_mode(self, temp_config_file):
        """Test serious vs sarcastic Claude mode selection."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(return_value=[])

        with patch("irssi_llmagent.main.AnthropicClient") as mock_claude_class:
            mock_claude = AsyncMock()
            mock_claude.call_claude = AsyncMock(return_value="Response")
            mock_claude_class.return_value.__aenter__.return_value = mock_claude

            # Test serious mode
            await agent.handle_command(
                "test", "#test", "#test", "user", "!s tell me facts", "mybot"
            )

            # Should use serious model
            call_args = mock_claude.call_claude.call_args
            system_prompt = call_args[0][1]
            model = call_args[0][2]

            assert "friendly, straight" in system_prompt.lower()
            assert model == agent.config["anthropic"]["serious_model"]

            # Test sarcastic mode (default)
            mock_claude.call_claude.reset_mock()
            await agent.handle_command("test", "#test", "#test", "user", "tell me jokes", "mybot")

            call_args = mock_claude.call_claude.call_args
            system_prompt = call_args[0][1]
            model = call_args[0][2]

            assert "sarcasm" in system_prompt.lower()
            assert model == agent.config["anthropic"]["model"]
