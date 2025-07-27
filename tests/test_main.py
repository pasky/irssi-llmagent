"""Tests for main application functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent, cli_mode


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
    async def test_serious_agent_mode(self, temp_config_file):
        """Test serious mode now uses agent."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(return_value=[{"role": "user", "content": "user search for Python news"}])

        with patch("irssi_llmagent.agent.ClaudeAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Agent response")
            mock_agent_class.return_value.__aenter__.return_value = mock_agent

            # Test serious mode (should use agent)
            await agent.handle_command(
                "test", "#test", "#test", "user", "!s search for Python news", "mybot"
            )

            # Should create and use agent
            mock_agent_class.assert_called_once_with(agent.config, "mybot")
            # Should call run_agent with context only
            mock_agent.run_agent.assert_called_once()
            call_args = mock_agent.run_agent.call_args
            assert len(call_args[0]) == 1  # Only context parameter
            context = call_args[0][0]
            assert isinstance(context, list)  # Should be context list
            agent.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_sarcastic_mode_unchanged(self, temp_config_file):
        """Test sarcastic mode still uses regular Claude."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(return_value=[])

        with patch("irssi_llmagent.main.AnthropicClient") as mock_claude_class:
            mock_claude = AsyncMock()
            mock_claude.call_claude = AsyncMock(return_value="Sarcastic response")
            mock_claude_class.return_value.__aenter__.return_value = mock_claude

            # Test sarcastic mode (default - should use regular Claude)
            await agent.handle_command("test", "#test", "#test", "user", "tell me jokes", "mybot")

            # Should use regular Claude API
            mock_claude.call_claude.assert_called_once()
            call_args = mock_claude.call_claude.call_args
            system_prompt = call_args[0][1]
            model = call_args[0][2]

            assert "sarcasm" in system_prompt.lower()
            assert model == agent.config["anthropic"]["model"]


class TestCLIMode:
    """Test CLI mode functionality."""

    @pytest.mark.asyncio
    async def test_cli_mode_sarcastic_message(self, temp_config_file):
        """Test CLI mode with sarcastic message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.main.AnthropicClient") as mock_claude_class:
                mock_claude = AsyncMock()
                mock_claude.call_claude = AsyncMock(return_value="Sarcastic response")
                mock_claude_class.return_value.__aenter__.return_value = mock_claude

                await cli_mode("tell me a joke", temp_config_file)

                # Verify Claude was called with the actual message
                mock_claude.call_claude.assert_called_once()
                call_args = mock_claude.call_claude.call_args
                context = call_args[0][0]  # First argument is context

                # Verify the user message is in the context
                assert len(context) == 1
                assert context[0]["role"] == "user"
                assert context[0]["content"] == "tell me a joke"

                # Verify output
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Simulating IRC message: tell me a joke" in call for call in print_calls)
                assert any("Sarcastic response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_perplexity_message(self, temp_config_file):
        """Test CLI mode with Perplexity message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.main.PerplexityClient") as mock_perplexity_class:
                mock_perplexity = AsyncMock()
                mock_perplexity.call_perplexity = AsyncMock(return_value="Weather is sunny")
                mock_perplexity_class.return_value.__aenter__.return_value = mock_perplexity

                await cli_mode("!p what is the weather?", temp_config_file)

                # Verify Perplexity was called with the actual message in context
                mock_perplexity.call_perplexity.assert_called_once()
                call_args = mock_perplexity.call_perplexity.call_args
                context = call_args[0][0]  # First argument is context

                # Verify the user message is in the context
                assert len(context) == 1
                assert context[0]["role"] == "user"
                assert context[0]["content"] == "!p what is the weather?"

                # Verify output
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Simulating IRC message: !p what is the weather?" in call for call in print_calls)
                assert any("Weather is sunny" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_agent_message(self, temp_config_file):
        """Test CLI mode with agent message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.agent.ClaudeAgent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run_agent = AsyncMock(return_value="Agent response")
                mock_agent_class.return_value.__aenter__.return_value = mock_agent

                await cli_mode("!s search for Python news", temp_config_file)

                # Verify agent was called with context only
                mock_agent.run_agent.assert_called_once()
                call_args = mock_agent.run_agent.call_args
                assert len(call_args[0]) == 1  # Only context parameter
                context = call_args[0][0]
                assert isinstance(context, list)  # Should be context list

                # Verify output
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Simulating IRC message: !s search for Python news" in call for call in print_calls)
                assert any("Agent response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_message_content_validation(self, temp_config_file):
        """Test that CLI mode passes actual message content, not placeholder text."""
        with patch("builtins.print"):
            with patch("irssi_llmagent.main.AnthropicClient") as mock_claude_class:
                mock_claude = AsyncMock()
                mock_claude.call_claude = AsyncMock(return_value="Response")
                mock_claude_class.return_value.__aenter__.return_value = mock_claude

                await cli_mode("specific test message", temp_config_file)

                # Verify Claude received the EXACT message content
                mock_claude.call_claude.assert_called_once()
                call_args = mock_claude.call_claude.call_args
                context = call_args[0][0]

                # This test would catch the bug where empty context resulted in "..." placeholder
                assert context[0]["content"] == "specific test message"
                assert context[0]["content"] != "..."  # Explicitly check it's not placeholder

    @pytest.mark.asyncio
    async def test_cli_mode_config_not_found(self):
        """Test CLI mode handles missing config file."""
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                await cli_mode("test query", "/nonexistent/config.json")

                mock_exit.assert_called_with(1)  # Just check it was called with 1, not once
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Config file not found" in call for call in print_calls)
