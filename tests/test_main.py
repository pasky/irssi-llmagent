"""Tests for main application functionality."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent, cli_mode
from irssi_llmagent.proactive_debouncer import ProactiveDebouncer


class MockAPIClient:
    """Mock API client with all required methods."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text

    def extract_text_from_response(self, r):
        return self.response_text

    def has_tool_calls(self, response):
        return False

    def extract_tool_calls(self, response):
        return None

    def format_assistant_message(self, response):
        return {"role": "assistant", "content": self.response_text}

    def format_tool_results(self, tool_results):
        return {"role": "user", "content": "Tool results"}


class TestIRSSILLMAgent:
    """Test main agent functionality."""

    def test_load_config(self, temp_config_file, api_type):
        """Test configuration loading."""
        agent = IRSSILLMAgent(temp_config_file)
        assert agent.config is not None
        assert "providers" in agent.config  # Provider sections exist
        assert "varlink" in agent.config

    def test_should_ignore_user(self, temp_config_file):
        """Test user ignoring functionality."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.config["command"]["ignore_users"] = ["spammer", "BadBot"]

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

        # Mock the model router call
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return simple text response via a fake client
            resp = {"output_text": "Test response"}

            return resp, MockAPIClient("Test response"), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ):
            # proceed

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

    def test_get_channel_mode(self, temp_config_file):
        """Test channel mode configuration retrieval."""
        agent = IRSSILLMAgent(temp_config_file)

        # Test default behavior
        agent.config["command"]["default_mode"] = "classifier"
        agent.config["command"]["channel_modes"] = {
            "#serious-work": "serious",
            "#sarcasm-corner": "sarcastic",
        }

        # Test channel-specific modes
        assert agent.get_channel_mode("#serious-work") == "serious"
        assert agent.get_channel_mode("#sarcasm-corner") == "sarcastic"

        # Test fallback to default
        assert agent.get_channel_mode("#random-channel") == "classifier"

        # Test when no config exists
        agent.config["command"] = {}
        assert agent.get_channel_mode("#any-channel") == "classifier"  # Default fallback

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
    async def test_help_command_with_channel_modes(self, temp_config_file):
        """Test help command shows channel-specific mode info."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.varlink_sender = AsyncMock()

        # Set up channel modes
        agent.config["command"]["channel_modes"] = {
            "#serious-work": "serious",
            "#sarcasm-corner": "sarcastic",
        }

        # Test help in serious channel
        await agent.handle_command("test", "#serious-work", "#serious-work", "user", "!h", "mybot")
        call_args = agent.varlink_sender.send_message.call_args[0]
        assert "default is serious agentic mode" in call_args[1]

        # Test help in sarcastic channel
        agent.varlink_sender.reset_mock()
        await agent.handle_command(
            "test", "#sarcasm-corner", "#sarcasm-corner", "user", "!h", "mybot"
        )
        call_args = agent.varlink_sender.send_message.call_args[0]
        assert "default is sarcastic mode" in call_args[1]

    @pytest.mark.asyncio
    async def test_command_debouncing_end_to_end(self, temp_config_file, temp_db_path):
        """Test end-to-end command debouncing with message consolidation."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.config["command"]["debounce"] = 1.0
        agent.varlink_sender = AsyncMock()

        # Use isolated database for this test
        from irssi_llmagent.history import ChatHistory

        agent.history = ChatHistory(temp_db_path)
        await agent.history.initialize()

        # Capture the final consolidated message
        captured_message = None

        async def capture_message(
            server, chan_name, target, nick, message, mynick, reasoning_effort="minimal"
        ):
            nonlocal captured_message
            captured_message = message

        agent._handle_sarcastic_mode = capture_message
        agent.classify_mode = AsyncMock(return_value="SARCASTIC")

        # Control timing with precise timestamp mocking
        base_time = 1000000000.0
        time_calls = 0

        def mock_time():
            nonlocal time_calls
            time_calls += 1
            # First call is when handle_command records original timestamp
            return base_time

        # Add followup messages that should be found after debounce sleep
        async def add_followup_messages():
            # Add all messages quickly at the start of debounce period
            await asyncio.sleep(0.05)  # Small delay to ensure they're after the timestamp
            await agent.history.add_message("test", "#test", "oops typo fix", "user", "mybot")
            await agent.history.add_message(
                "test", "#test", "blah", "user2", "mybot"
            )  # Different user - should be ignored
            await agent.history.add_message(
                "test", "#test2", "blahblah", "user", "mybot"
            )  # Different channel - should be ignored
            await agent.history.add_message("test", "#test", "and one more", "user", "mybot")

        with patch("time.time", side_effect=mock_time):
            # Run both tasks concurrently
            await asyncio.gather(
                agent.handle_command("test", "#test", "#test", "user", "original command", "mybot"),
                add_followup_messages(),
            )

        # Verify the consolidated message contains all parts
        assert captured_message is not None
        print(f"Captured message: '{captured_message}'")
        assert captured_message == "original command\noops typo fix\nand one more"

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
        agent.config["proactive"]["interjecting"] = ["#test"]

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
            mock_agent_class.assert_called_once_with(
                agent.config, "mybot", mode="serious", extra_prompt="", model_override=None
            )
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
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return simple text response via a fake client
            resp = {"output_text": "Sarcastic response"}

            return resp, MockAPIClient("Sarcastic response"), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ) as mock_call:
            # Test sarcastic mode (default - should use router)
            await agent.handle_command("test", "#test", "#test", "user", "tell me jokes", "mybot")

            # Should have been called
            assert mock_call.called

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

        # Mock the model router for classification
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return classification result
            resp = {"output_text": "SERIOUS"}

            return resp, MockAPIClient("SERIOUS"), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ) as mock_call:
            with patch("irssi_llmagent.main.AIAgent") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run_agent = AsyncMock(return_value="Agent response")
                mock_agent_class.return_value.__aenter__.return_value = mock_agent

                # Test automatic mode message that should be classified as serious
                await agent.handle_command(
                    "test", "#test", "#test", "user", "how do I install Python?", "mybot"
                )

                # Should call classify_mode first, then use serious mode (agent)
                assert mock_call.called
                mock_agent.run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_proactive_interjection_detection(self, temp_config_file):
        """Test proactive interjection detection in whitelisted channels."""
        agent = IRSSILLMAgent(temp_config_file)
        # Configure proactive interjection test channel with short debounce
        agent.config["proactive"]["interjecting_test"] = ["#testchannel"]
        agent.config["proactive"]["debounce_seconds"] = 0.1
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

        # Mock the router for proactive decisions and classification
        seq = ["Should help with this: 9/10", "SERIOUS"]

        async def fake_call_raw_with_model(*args, **kwargs):
            text = seq.pop(0)
            resp = {"output_text": text}

            return resp, MockAPIClient(text), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ):
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

                # Should call agent in test mode (consistent with live mode)
                mock_agent.run_agent.assert_called_once()
                # Should pass the extra proactive prompt to the agent
                mock_agent_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_proactive_interjection_configurable_threshold(self, temp_config_file):
        """Test proactive interjection with configurable threshold."""
        agent = IRSSILLMAgent(temp_config_file)

        # Test with threshold 8 - score 8 should trigger
        agent.config["proactive"]["interject_threshold"] = 8

        async def fake_call_raw_with_model(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 8/10"}
            return resp, MockAPIClient("Testing threshold: 8/10"), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ):
            context = [{"role": "user", "content": "Test message"}]
            should_interject, reason, test_mode = await agent.should_interject_proactively(context)

            assert should_interject is True
            assert "Score: 8" in reason
            assert test_mode is False

        # Test with threshold 9 - score 8 should NOT trigger
        agent.config["proactive"]["interject_threshold"] = 9

        async def fake_call_raw_with_model_8(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 8/10"}

            class C:
                def extract_text_from_response(self, r):
                    return "Testing threshold: 8/10"

            return resp, C(), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model_8),
        ):
            context = [{"role": "user", "content": "Test message"}]
            should_interject, reason, test_mode = await agent.should_interject_proactively(context)

            assert should_interject is True  # Now should trigger test mode
            assert "Score: 8" in reason
            assert test_mode is True  # Should be in test mode due to barely threshold

        # Test with threshold 9 - score 7 should NOT trigger at all
        async def fake_call_raw_with_model_7(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 7/10"}
            return resp, MockAPIClient("Testing threshold: 7/10"), None

        with patch(
            "irssi_llmagent.main.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model_7),
        ):
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

                # Create a real agent
                from irssi_llmagent.main import IRSSILLMAgent

                agent = IRSSILLMAgent(temp_config_file)

                async def fake_call_raw_with_model(*args, **kwargs):
                    resp = {"output_text": "Sarcastic response"}

                    return resp, MockAPIClient("Sarcastic response"), None

                # Patch the agent creation in cli_mode and model router
                with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                    with patch(
                        "irssi_llmagent.main.ModelRouter.call_raw_with_model",
                        new=AsyncMock(side_effect=fake_call_raw_with_model),
                    ):
                        await cli_mode("!S tell me a joke", temp_config_file)

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
