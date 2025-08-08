"""Tests for main application functionality."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent, cli_mode
from irssi_llmagent.proactive_debouncer import ProactiveDebouncer


class TestIRSSILLMAgent:
    """Test main agent functionality."""

    def test_load_config(self, temp_config_file, api_type):
        """Test configuration loading."""
        agent = IRSSILLMAgent(temp_config_file)
        assert agent.config is not None
        assert agent.config["api_type"] == api_type
        assert api_type in agent.config  # Check API-specific section exists
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

        # Mock the API client class
        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value="Test response")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

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
        assert "automatic mode" in call_args[1]  # Help text should mention automatic mode

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
    async def test_command_cancels_proactive_interjection(self, temp_config_file):
        """Test that command processing cancels pending proactive interjection for the same channel."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.proactive_debouncer = AsyncMock(spec=ProactiveDebouncer)
        agent.rate_limiter.check_limit = lambda: True
        agent.server_nicks["test"] = "mybot"

        # Configure for proactive interjecting
        agent.config["behavior"]["proactive_interjecting"] = ["#test"]

        # First, send a non-command message to trigger proactive interjection scheduling
        non_command_event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "alice",
            "message": "some random message",
        }
        await agent.process_message_event(non_command_event)

        # Verify proactive interjection was scheduled
        agent.proactive_debouncer.schedule_check.assert_called_once()

        # Now send a command message to the same channel
        command_event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "bob",
            "message": "mybot: !h",
        }

        # Mock handle_command to prevent actual command processing
        with patch.object(agent, "handle_command", new_callable=AsyncMock):
            await agent.process_message_event(command_event)

        # Verify that proactive interjection was cancelled for the channel
        agent.proactive_debouncer.cancel_channel.assert_called_once_with("#test")

    @pytest.mark.asyncio
    async def test_serious_agent_mode(self, temp_config_file):
        """Test serious mode now uses agent."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(
            return_value=[{"role": "user", "content": "user search for Python news"}]
        )

        with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Agent response")
            mock_agent_class.return_value.__aenter__.return_value = mock_agent

            # Test serious mode (should use agent)
            await agent.handle_command(
                "test", "#test", "#test", "user", "!s search for Python news", "mybot"
            )

            # Should create and use agent
            mock_agent_class.assert_called_once_with(agent.config, "mybot", "", model_override=None)
            # Should call run_agent with context only
            mock_agent.run_agent.assert_called_once()
            call_args = mock_agent.run_agent.call_args
            assert len(call_args[0]) == 1  # Only context parameter
            context = call_args[0][0]
            assert isinstance(context, list)  # Should be context list
            agent.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_sarcastic_mode_unchanged(self, temp_config_file):
        """Test sarcastic mode still uses regular API client."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(return_value=[])

        # Mock the API client class that would be returned by _get_api_client_class
        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value="Sarcastic response")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Test sarcastic mode (default - should use regular API client)
            await agent.handle_command("test", "#test", "#test", "user", "tell me jokes", "mybot")

            # Should use regular API client
            mock_client.call.assert_called_once()
            call_args = mock_client.call.call_args
            system_prompt = call_args[0][1]
            model = call_args[0][2]

            assert "sarcasm" in system_prompt.lower()
            # Get the expected model from the appropriate config section
            api_config = agent._get_api_config_section()
            assert model == api_config["model"]

    @pytest.mark.asyncio
    async def test_mode_classification(self, temp_config_file):
        """Test that mode classification works in automatic mode."""
        agent = IRSSILLMAgent(temp_config_file)

        # Mock dependencies
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(
            return_value=[{"role": "user", "content": "how do I install Python?"}]
        )

        # Mock the API client class for classification
        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            # First call for classification returns SERIOUS
            mock_client.call = AsyncMock(return_value="SERIOUS")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run_agent = AsyncMock(return_value="Agent response")
                mock_agent_class.return_value.__aenter__.return_value = mock_agent

                # Test automatic mode message that should be classified as serious
                await agent.handle_command(
                    "test", "#test", "#test", "user", "how do I install Python?", "mybot"
                )

                # Should call classify_mode first, then use serious mode (agent)
                mock_client.call.assert_called_once()
                mock_agent.run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_proactive_interjection_detection(self, temp_config_file):
        """Test proactive interjection detection in whitelisted channels."""
        agent = IRSSILLMAgent(temp_config_file)
        # Configure proactive interjection test channel with short debounce
        agent.config["behavior"]["proactive_interjecting_test"] = ["#testchannel"]
        agent.config["behavior"]["proactive_debounce_seconds"] = 0.1
        # Recreate debouncer with updated config
        agent.proactive_debouncer = ProactiveDebouncer(0.1)

        # Mock dependencies
        agent.varlink_sender = AsyncMock()
        agent.history = AsyncMock()
        agent.history.add_message = AsyncMock()
        agent.history.get_context = AsyncMock(
            return_value=[{"role": "user", "content": "I need help with Python imports"}]
        )
        agent.server_nicks["test"] = "mybot"

        # Mock the API client class for proactive decisions and classification
        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            # Mock proactive decision (score 9), mode classification (SERIOUS), and test response
            mock_client.call = AsyncMock(
                side_effect=["Should help with this: 9/10", "SERIOUS", "Test response"]
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run_agent = AsyncMock(return_value="Test response")
                mock_agent_class.return_value.__aenter__.return_value = mock_agent

                # Test message NOT addressing bot in whitelisted test channel
                event = {
                    "type": "message",
                    "subtype": "public",
                    "server": "test",
                    "target": "#testchannel",
                    "nick": "user",
                    "message": "I need help with Python imports",
                }

                await agent.process_message_event(event)

                # Wait for debounce to complete
                await asyncio.sleep(0.15)

                # Should call proactive decision and classification (agent call is separate)
                assert mock_client.call.call_count == 2
                # Should call agent in test mode (consistent with live mode)
                mock_agent.run_agent.assert_called_once()
                # Should pass the extra proactive prompt to the agent
                api_config = agent._get_api_config_section()
                expected_model = api_config["model"]
                mock_agent_class.assert_called_once_with(
                    agent.config,
                    "mybot",
                    " NOTE: This is a proactive interjection. If upon reflection you decide your contribution wouldn't add significant factual value (e.g. just an encouragement or general statement), respond with exactly 'NULL' instead of a message.",
                    model_override=expected_model,
                )

    @pytest.mark.asyncio
    async def test_proactive_interjection_configurable_threshold(self, temp_config_file):
        """Test proactive interjection with configurable threshold."""
        agent = IRSSILLMAgent(temp_config_file)

        # Test with threshold 8 - score 8 should trigger
        agent.config["behavior"]["proactive_interject_threshold"] = 8

        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value="Testing threshold: 8/10")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            context = [{"role": "user", "content": "Test message"}]
            should_interject, reason, test_mode = await agent.should_interject_proactively(context)

            assert should_interject is True
            assert "Score: 8" in reason
            assert test_mode is False

        # Test with threshold 9 - score 8 should NOT trigger
        agent.config["behavior"]["proactive_interject_threshold"] = 9

        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value="Testing threshold: 8/10")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            context = [{"role": "user", "content": "Test message"}]
            should_interject, reason, test_mode = await agent.should_interject_proactively(context)

            assert should_interject is True  # Now should trigger test mode
            assert "Score: 8" in reason
            assert test_mode is True  # Should be in test mode due to barely threshold

        # Test with threshold 9 - score 7 should NOT trigger at all
        with patch.object(agent, "api_client_class") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.call = AsyncMock(return_value="Testing threshold: 7/10")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            context = [{"role": "user", "content": "Test message"}]
            should_interject, reason, test_mode = await agent.should_interject_proactively(context)

            assert should_interject is False  # Should not trigger at all
            assert "Score: 7" in reason
            assert test_mode is False


class TestCLIMode:
    """Test CLI mode functionality."""

    @pytest.mark.asyncio
    async def test_cli_mode_sarcastic_message(self, temp_config_file):
        """Test CLI mode with sarcastic message."""
        with patch("builtins.print") as mock_print:
            # Mock the ChatHistory import in cli_mode
            with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                mock_history = AsyncMock()
                mock_history.add_message = AsyncMock()
                mock_history.get_context.return_value = [
                    {"role": "user", "content": "!S tell me a joke"}
                ]
                mock_history_class.return_value = mock_history

                # Create a real agent but mock its API client
                from irssi_llmagent.main import IRSSILLMAgent

                agent = IRSSILLMAgent(temp_config_file)

                # Mock the API client
                with patch.object(agent, "api_client_class") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.call = AsyncMock(return_value="Sarcastic response")
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    # Patch the agent creation in cli_mode
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_mode("!S tell me a joke", temp_config_file)

                        # Verify API client was called with the actual message
                        mock_client.call.assert_called_once()
                        call_args = mock_client.call.call_args
                        context = call_args[0][0]  # First argument is context

                        # Verify the user message is in the context - should be the last message
                        assert len(context) >= 1
                        assert context[-1]["role"] == "user"
                        assert "!S tell me a joke" in context[-1]["content"]

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating IRC message: !S tell me a joke" in call
                            for call in print_calls
                        )
                        assert any("Sarcastic response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_perplexity_message(self, temp_config_file):
        """Test CLI mode with Perplexity message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.main.PerplexityClient") as mock_perplexity_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!p what is the weather?"}
                    ]
                    mock_history_class.return_value = mock_history

                    mock_perplexity = AsyncMock()
                    mock_perplexity.call_perplexity = AsyncMock(return_value="Weather is sunny")
                    mock_perplexity_class.return_value.__aenter__.return_value = mock_perplexity

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_mode
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_mode("!p what is the weather?", temp_config_file)

                        # Verify Perplexity was called with the actual message in context
                        mock_perplexity.call_perplexity.assert_called_once()
                        call_args = mock_perplexity.call_perplexity.call_args
                        context = call_args[0][0]  # First argument is context

                        # Verify the user message is in the context - should be the last message
                        assert len(context) >= 1
                        assert context[-1]["role"] == "user"
                        assert "!p what is the weather?" in context[-1]["content"]

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating IRC message: !p what is the weather?" in call
                            for call in print_calls
                        )
                        assert any("Weather is sunny" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_agent_message(self, temp_config_file):
        """Test CLI mode with agent message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s search for Python news"}
                    ]
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value.__aenter__.return_value = mock_agent

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_mode
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_mode("!s search for Python news", temp_config_file)

                        # Verify agent was called with context only
                        mock_agent.run_agent.assert_called_once()
                        call_args = mock_agent.run_agent.call_args
                        assert len(call_args[0]) == 1  # Only context parameter
                        context = call_args[0][0]
                        assert isinstance(context, list)  # Should be context list
                        # Verify the user message is the last in context
                        assert "!s search for Python news" in context[-1]["content"]

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating IRC message: !s search for Python news" in call
                            for call in print_calls
                        )
                        assert any("Agent response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_mode_message_content_validation(self, temp_config_file):
        """Test that CLI mode passes actual message content, not placeholder text."""
        with patch("builtins.print"):
            with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s specific test message"}
                    ]
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value.__aenter__.return_value = mock_agent

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_mode
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_mode("!s specific test message", temp_config_file)

                        # Verify agent was called once for serious mode
                        mock_agent.run_agent.assert_called_once()
                        call_args = mock_agent.run_agent.call_args
                        context = call_args[0][0]

                        # This test would catch the bug where empty context resulted in "..." placeholder
                        # The user message should be the last message in context
                        assert "!s specific test message" in context[-1]["content"]
                        assert (
                            context[-1]["content"] != "..."
                        )  # Explicitly check it's not placeholder

    @pytest.mark.asyncio
    async def test_cli_mode_config_not_found(self):
        """Test CLI mode handles missing config file."""
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                await cli_mode("test query", "/nonexistent/config.json")

                mock_exit.assert_called_with(1)  # Just check it was called with 1, not once
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Config file not found" in call for call in print_calls)
