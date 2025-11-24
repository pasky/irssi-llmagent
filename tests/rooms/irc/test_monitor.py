"""Tests for IRC monitor functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent
from irssi_llmagent.rooms import ProactiveDebouncer


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


class TestIRCMonitor:
    """Test IRC monitor functionality."""

    def test_should_ignore_user(self, temp_config_file):
        """Test user ignoring functionality."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.config["rooms"]["irc"]["command"]["ignore_users"] = ["spammer", "BadBot"]

        assert agent.irc_monitor.should_ignore_user("spammer") is True
        assert agent.irc_monitor.should_ignore_user("SPAMMER") is True  # Case insensitive
        assert agent.irc_monitor.should_ignore_user("gooduser") is False

    @pytest.mark.asyncio
    async def test_get_mynick_caching(self, temp_config_file):
        """Test that bot nick is cached per server."""
        agent = IRSSILLMAgent(temp_config_file)

        # Mock the varlink sender
        mock_sender = AsyncMock()
        mock_sender.get_server_nick.return_value = "testbot"
        agent.irc_monitor.varlink_sender = mock_sender

        # First call should query the server
        nick1 = await agent.irc_monitor.get_mynick("irc.libera.chat")
        assert nick1 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1

        # Second call should use cache
        nick2 = await agent.irc_monitor.get_mynick("irc.libera.chat")
        assert nick2 == "testbot"
        assert mock_sender.get_server_nick.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_message_addressing_detection(self, temp_config_file):
        """Test that messages addressing the bot are detected correctly."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.server_nicks["test"] = "mybot"

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Mock dependencies
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Mock the model router call
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return simple text response via a fake client
            resp = {"output_text": "Test response"}

            return resp, MockAPIClient("Test response"), None

        with patch(
            "irssi_llmagent.agentic_actor.actor.ModelRouter.call_raw_with_model",
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

            await agent.irc_monitor.process_message_event(event)

            # Should call handle_command
            agent.irc_monitor.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_privmsg_commands_without_nick_prefix(self, temp_config_file):
        """Test that private messages are treated as commands without requiring nick prefix."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.rate_limiter.check_limit = lambda: True
        agent.irc_monitor.server_nicks["test"] = "mybot"

        # Mock the model router call
        async def fake_call_raw_with_model(*args, **kwargs):
            resp = {"output_text": "Test response"}
            return resp, MockAPIClient("Test response"), None

        with patch(
            "irssi_llmagent.agentic_actor.actor.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=fake_call_raw_with_model),
        ):
            # Test private message without nick prefix (should be treated as command)
            event = {
                "type": "message",
                "subtype": "private",
                "server": "test",
                "target": "mybot",
                "nick": "testuser",
                "message": "hello there",
            }

            await agent.irc_monitor.process_message_event(event)

            # Should call handle_command even without nick prefix
            agent.irc_monitor.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_channel_messages_without_nick_prefix_ignored(self, temp_config_file):
        """Test that channel messages without nick prefix are ignored (no response)."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.server_nicks["test"] = "mybot"

        # Ensure no proactive interjecting channels configured
        agent.config["rooms"]["irc"]["proactive"]["interjecting"] = []
        agent.config["rooms"]["irc"]["proactive"]["interjecting_test"] = []

        # Test channel message without nick prefix (should be ignored)
        event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "testuser",
            "message": "just a regular message",
        }

        await agent.irc_monitor.process_message_event(event)

        # Should NOT call send_message (bot should not respond)
        agent.irc_monitor.varlink_sender.send_message.assert_not_called()

    def test_get_channel_mode(self, temp_config_file):
        """Test channel mode configuration retrieval."""
        agent = IRSSILLMAgent(temp_config_file)

        # Test default behavior
        agent.config["rooms"]["irc"]["command"]["default_mode"] = "classifier"
        agent.config["rooms"]["irc"]["command"]["channel_modes"] = {
            "#serious-work": "serious",
            "#sarcasm-corner": "sarcastic",
        }

        # Test channel-specific modes
        assert agent.irc_monitor.get_channel_mode("#serious-work") == "serious"
        assert agent.irc_monitor.get_channel_mode("#sarcasm-corner") == "sarcastic"

        # Test fallback to default
        assert agent.irc_monitor.get_channel_mode("#random-channel") == "classifier"

        # Test when no config exists
        agent.config["rooms"]["irc"]["command"] = {}
        assert (
            agent.irc_monitor.get_channel_mode("#any-channel") == "classifier"
        )  # Default fallback

    @pytest.mark.asyncio
    async def test_help_command_with_channel_modes(self, shared_agent):
        """Test help command shows channel-specific mode info."""
        agent = shared_agent

        # Set up channel modes
        agent.config["rooms"]["irc"]["command"]["channel_modes"] = {
            "#serious-work": "serious",
            "#sarcasm-corner": "sarcastic",
        }

        # Test help in serious channel
        await agent.irc_monitor.handle_command(
            "test", "#serious-work", "#serious-work", "user", "!h", "mybot"
        )
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert "default is serious agentic mode" in call_args[1]

        # Test help in sarcastic channel
        agent.irc_monitor.varlink_sender.reset_mock()
        await agent.irc_monitor.handle_command(
            "test", "#sarcasm-corner", "#sarcasm-corner", "user", "!h", "mybot"
        )
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert "default is sarcastic mode" in call_args[1]

        # Test help in automatic mode
        await agent.irc_monitor.handle_command("test", "#test", "#test", "user", "!h", "mybot")
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert "default is automatic mode" in call_args[1]

    @pytest.mark.asyncio
    async def test_command_debouncing_end_to_end(self, shared_agent_with_db):
        """Test end-to-end command debouncing with message consolidation and context isolation."""
        agent = shared_agent_with_db
        agent.config["rooms"]["irc"]["command"]["debounce"] = 0.1

        # Set up pre-existing conversation context
        await agent.history.add_message("test", "#test", "earlier message", "alice", "mybot")
        await agent.history.add_message("test", "#test", "some context", "bob", "mybot")

        # Capture the final consolidated message and context
        captured_context = []

        async def capture_message_and_context(
            context,
            mynick,
            *,
            mode,
            progress_callback=None,
            **kwargs,
        ):
            nonlocal captured_context
            captured_context = context

        agent.irc_monitor._run_actor = capture_message_and_context
        agent.irc_monitor.classify_mode = AsyncMock(return_value="SARCASTIC")

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
            # Add interfering messages from other users
            await agent.history.add_message(
                "test", "#test", "interfering message", "charlie", "mybot"
            )
            await agent.history.add_message("test", "#test", "final interfering", "dave", "mybot")

        with (
            patch("irssi_llmagent.rooms.irc.monitor.time", spec=True) as mock_time_module,
            patch("irssi_llmagent.providers.ModelRouter.call_raw_with_model") as mock_router,
        ):
            # Mock time and API calls to prevent delays
            mock_time_module.time = mock_time
            mock_client = MagicMock()
            mock_client.extract_text_from_response.return_value = "Mock response"
            mock_router.return_value = ({"output_text": "Mock response"}, mock_client, None)

            # Run both tasks concurrently
            await asyncio.gather(
                agent.irc_monitor.handle_command(
                    "test", "#test", "#test", "user", "original command", "mybot"
                ),
                add_followup_messages(),
            )

        # Verify message consolidation worked
        expected_content = "original command\noops typo fix\nand one more"
        assert captured_context[-1]["content"].endswith(expected_content)

    @pytest.mark.asyncio
    async def test_explicit_command_prevents_race_condition(self, shared_agent_with_db):
        """Test that explicit commands use early context snapshot to prevent race conditions."""
        agent = shared_agent_with_db
        agent.config["rooms"]["irc"]["command"]["debounce"] = 0.05

        # Add initial messages to create context
        await agent.history.add_message("test", "#test", "initial message", "alice", "mybot")
        await agent.history.add_message("test", "#test", "second message", "charlie", "mybot")
        # The user's command message will be added by handle_command() now

        captured_context = [{"content": "ERROR: capture_context() not called"}]

        async def capture_context(
            context,
            mynick,
            *,
            mode,
            progress_callback=None,
            **kwargs,
        ):
            nonlocal captured_context
            captured_context = context

        agent.irc_monitor._run_actor = capture_context

        # Hook into get_context to add interfering message after context is retrieved
        original_get_context = agent.history.get_context

        async def hooked_get_context(server_tag: str, channel_name: str, limit: int | None = None):
            context = await original_get_context(server_tag, channel_name, limit)
            # Add interfering message after context is captured but before it's returned
            await agent.history.add_message("test", "#test", "interfering message", "bob", "mybot")
            return context

        agent.history.get_context = hooked_get_context

        # Test explicit command with interfering message added during context retrieval
        await agent.irc_monitor.handle_command(
            "test", "#test", "#test", "user", "!d be sarcastic", "mybot"
        )

        # Should contain initial messages + user's command, but NOT interfering message
        assert len(captured_context) == 3
        assert captured_context[0]["content"].endswith("initial message")
        assert captured_context[1]["content"].endswith("second message")
        assert captured_context[2]["content"].endswith("!d be sarcastic")

    @pytest.mark.asyncio
    async def test_rate_limiting_triggers(self, temp_config_file):
        """Test that rate limiting prevents excessive requests."""
        agent = IRSSILLMAgent(temp_config_file)
        # Mock rate limiter to simulate limit exceeded
        agent.irc_monitor.rate_limiter.check_limit = lambda: False
        agent.irc_monitor.varlink_sender = AsyncMock()
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

        agent.irc_monitor.server_nicks["test"] = "mybot"
        await agent.irc_monitor.process_message_event(event)

        # Should send rate limiting message
        agent.irc_monitor.varlink_sender.send_message.assert_called_once()
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert "rate limiting" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_command_cancels_proactive_interjection(self, temp_config_file):
        """Test that command processing cancels pending proactive interjection for the same channel."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.proactive_debouncer = AsyncMock(spec=ProactiveDebouncer)
        agent.irc_monitor.rate_limiter.check_limit = lambda: True
        agent.irc_monitor.server_nicks["test"] = "mybot"

        # Configure for proactive interjecting
        agent.config["rooms"]["irc"]["proactive"]["interjecting"] = ["#test"]

        # First, send a non-command message to trigger proactive interjection scheduling
        non_command_event = {
            "type": "message",
            "subtype": "public",
            "server": "test",
            "target": "#test",
            "nick": "alice",
            "message": "some random message",
        }
        await agent.irc_monitor.process_message_event(non_command_event)

        # Verify proactive interjection was scheduled
        agent.irc_monitor.proactive_debouncer.schedule_check.assert_called_once()

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
        with patch.object(agent.irc_monitor, "handle_command", new_callable=AsyncMock):
            await agent.irc_monitor.process_message_event(command_event)

        # Verify that proactive interjection was cancelled for the channel
        agent.irc_monitor.proactive_debouncer.cancel_channel.assert_called_once_with("#test")

    @pytest.mark.asyncio
    async def test_progress_callback_handles_tool_persistence(self, temp_config_file):
        """Test that IRC monitor progress callback handles tool_persistence type correctly."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()
        agent.history = AsyncMock()

        # Create a progress callback function like the one in handle_command
        server, chan_name, target, mynick = "test", "#test", "#test", "mybot"

        async def progress_cb(text: str, type: str = "progress") -> None:
            if type == "tool_persistence":
                # Store tool persistence summary as assistant_silent role
                await agent.history.add_message(
                    server, chan_name, text, mynick, mynick, False, content_template="{message}"
                )
            else:
                # Regular progress message - send to channel
                await agent.irc_monitor.varlink_sender.send_message(target, text, server)
                await agent.history.add_message(server, chan_name, text, mynick, mynick, True)

        # Test regular progress callback
        await progress_cb("Working on your request...", "progress")

        # Verify regular progress was sent via IRC
        agent.irc_monitor.varlink_sender.send_message.assert_called_with(
            target, "Working on your request...", server
        )

        # Reset mock to test tool persistence
        agent.irc_monitor.varlink_sender.reset_mock()

        # Test tool persistence callback (should store in history, not send to IRC)
        await progress_cb(
            "Tool summary: Performed web search and found 5 results about Python",
            "tool_persistence",
        )

        # Verify no message was sent to IRC for tool persistence
        agent.irc_monitor.varlink_sender.send_message.assert_not_called()

        # Verify message was stored in history with plain content template
        agent.history.add_message.assert_called_with(
            server,
            chan_name,
            "Tool summary: Performed web search and found 5 results about Python",
            mynick,
            mynick,
            False,
            content_template="{message}",
        )

    @pytest.mark.asyncio
    async def test_serious_agent_mode(self, temp_config_file):
        """Test serious mode now uses agent with chapter context."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Add test message and chapter context
        await agent.history.add_message(
            "test", "#test", "user search for Python news", "user", "mybot"
        )
        arc = "test#test"
        await agent.chronicle.append_paragraph(arc, "Previous discussion about Python")
        await agent.chronicle.append_paragraph(arc, "User asked about imports")

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Agent response")
            mock_agent_class.return_value = mock_agent

            # Test serious mode (should use agent)
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!s search for Python news", "mybot"
            )

            # Should create and use agent - verify key parameters
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert call_args[1]["config"] == agent.config
            assert call_args[1]["reasoning_effort"] == "low"
            # Model should be the serious model from config (list in this case)
            expected_model = agent.config["rooms"]["irc"]["command"]["modes"]["serious"]["model"]
            assert call_args[1]["model"] == expected_model

            # Should have chapter context prepended (includes meta message for new arc + our 2 paragraphs)
            assert "prepended_context" in call_args[1]
            prepended = call_args[1]["prepended_context"]
            assert len(prepended) >= 1  # At least 1 context message (may be just meta for new arc)

            # Verify we get some form of chapter context (may be meta message for new arc)
            assert len(prepended) >= 1
            assert prepended[0]["role"] == "user"
            assert "<context_summary>" in prepended[0]["content"]

            # Should call run_agent with context only
            mock_agent.run_agent.assert_called_once()
            call_args = mock_agent.run_agent.call_args
            assert len(call_args[0]) == 1  # Only context parameter
            context = call_args[0][0]
            assert isinstance(context, list)  # Should be context list
            agent.irc_monitor.varlink_sender.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_thinking_serious_mode_override(self, temp_config_file):
        """Test that THINKING_SERIOUS mode uses thinking_model if configured, and falls back if not."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize databases
        await agent.history.initialize()
        await agent.chronicle.initialize()

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Agent response")
            mock_agent_class.return_value = mock_agent

            # 1. Test fallback (thinking_model NOT configured)
            # Ensure thinking_model is not set
            if "thinking_model" in agent.config["rooms"]["irc"]["command"]["modes"]["serious"]:
                del agent.config["rooms"]["irc"]["command"]["modes"]["serious"]["thinking_model"]

            default_serious_model = agent.config["rooms"]["irc"]["command"]["modes"]["serious"][
                "model"
            ]

            # Trigger THINKING_SERIOUS mode with !a
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!a solve default", "mybot"
            )

            # Verify fallback to default serious model
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert call_args[1]["model"] == default_serious_model
            assert call_args[1]["reasoning_effort"] == "medium"

            # Reset mock for next test
            mock_agent_class.reset_mock()

            # 2. Test override (thinking_model IS configured)
            thinking_model = "provider:thinking-model"
            agent.config["rooms"]["irc"]["command"]["modes"]["serious"]["thinking_model"] = (
                thinking_model
            )

            # Trigger THINKING_SERIOUS mode with !a
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!a solve override", "mybot"
            )

            # Verify override with thinking_model
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert call_args[1]["model"] == thinking_model
            assert call_args[1]["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_help_command_with_thinking_model(self, shared_agent):
        """Test help command shows thinking model info when configured."""
        agent = shared_agent

        # Configure thinking_model
        thinking_model = "provider:thinking-model"
        agent.config["rooms"]["irc"]["command"]["modes"]["serious"]["thinking_model"] = (
            thinking_model
        )

        # Set up channel modes to test serious channel output
        agent.config["rooms"]["irc"]["command"]["channel_modes"] = {
            "#serious-work": "serious",
        }

        # Test help in serious channel
        await agent.irc_monitor.handle_command(
            "test", "#serious-work", "#serious-work", "user", "!h", "mybot"
        )
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert f"!a forces thinking ({thinking_model})" in call_args[1]

        # Test help in automatic mode (default)
        await agent.irc_monitor.handle_command("test", "#test", "#test", "user", "!h", "mybot")
        call_args = agent.irc_monitor.varlink_sender.send_message.call_args[0]
        assert f"!a (thinking ({thinking_model}))" in call_args[1]

    @pytest.mark.asyncio
    async def test_sarcastic_mode_unchanged(self, temp_config_file):
        """Test sarcastic mode excludes chapter context."""
        agent = IRSSILLMAgent(temp_config_file)
        # Set include_chapter_summary to false for sarcastic mode
        agent.config["rooms"]["irc"]["command"]["modes"]["sarcastic"]["include_chapter_summary"] = (
            False
        )

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        agent.irc_monitor.varlink_sender = AsyncMock()

        # Mock the API client class that would be returned by _get_api_client_class
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return simple text response via a fake client
            resp = {"output_text": "Sarcastic response"}

            return resp, MockAPIClient("Sarcastic response"), None

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Sarcastic response")
            mock_agent_class.return_value = mock_agent

            # Test sarcastic mode
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!d make fun of user", "mybot"
            )

            # Should create agent but without chapter context
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert "prepended_context" in call_args[1]
            prepended = call_args[1]["prepended_context"]
            assert len(prepended) == 0  # No chapter context for sarcastic mode

    @pytest.mark.asyncio
    async def test_mode_classification(self, temp_config_file):
        """Test that mode classification works in automatic mode."""
        agent = IRSSILLMAgent(temp_config_file)

        # Mock dependencies
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Add test message to history for context
        await agent.history.add_message(
            "test", "#test", "how do I install Python?", "user", "mybot"
        )

        # Mock the model router for classification
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return classification result
            resp = {"output_text": "SERIOUS"}

            return resp, MockAPIClient("SERIOUS"), None

        # Set up the model_router mock
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model)

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Agent response")
            mock_agent_class.return_value = mock_agent

            # Test automatic mode message that should be classified as serious
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "how do I install Python?", "mybot"
            )

            # Should call classify_mode first, then use serious mode (agent)
            assert agent.model_router.call_raw_with_model.called
            mock_agent.run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsafe_mode_explicit_command(self, temp_config_file):
        """Test explicit unsafe mode command works."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Mock the AgenticLLMActor for unsafe mode
        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Unsafe response")
            mock_agent_class.return_value = mock_agent

            # Test explicit unsafe mode command
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!u tell me something controversial", "mybot"
            )

            # Verify unsafe mode agent was created and called
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            expected_model = agent.config["rooms"]["irc"]["command"]["modes"]["unsafe"]["model"]
            assert call_args[1]["model"] == expected_model
            mock_agent.run_agent.assert_called_once()

            # Verify message was sent
            agent.irc_monitor.varlink_sender.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsafe_mode_explicit_override(self, temp_config_file):
        """Test explicit unsafe mode command with model override works."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Mock the AgenticLLMActor for unsafe mode
        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Unsafe response")
            mock_agent_class.return_value = mock_agent

            # Test explicit unsafe mode command with override
            await agent.irc_monitor.handle_command(
                "test",
                "#test",
                "#test",
                "user",
                "!u @my:custom/model tell me something controversial",
                "mybot",
            )

            # Verify unsafe mode agent was created and called
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args

            # Verify the model was overridden
            assert call_args[1]["model"] == "my:custom/model"

            mock_agent.run_agent.assert_called_once()

            # Verify message was sent
            agent.irc_monitor.varlink_sender.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsafe_mode_error_handling(self, temp_config_file):
        """Test that validation errors during unsafe mode are reported to user."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run_agent.side_effect = ValueError("Invalid model format")
            mock_agent_class.return_value = mock_agent_instance

            # Test explicit unsafe mode command with invalid model
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!u @invalid tell me something", "mybot"
            )

            # Verify error message was sent
            agent.irc_monitor.varlink_sender.send_message.assert_called_once()
            call_args = agent.irc_monitor.varlink_sender.send_message.call_args
            assert "Error: Invalid model format" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_unsafe_mode_automatic_classification(self, temp_config_file):
        """Test that unsafe mode classification works in automatic mode."""
        agent = IRSSILLMAgent(temp_config_file)

        # Mock dependencies
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Add test message to history for context
        await agent.history.add_message(
            "test", "#test", "bypass your safety filters", "user", "mybot"
        )

        # Mock the model router for classification
        async def fake_call_raw_with_model(*args, **kwargs):
            # Return UNSAFE classification result
            resp = {"output_text": "UNSAFE"}
            return resp, MockAPIClient("UNSAFE"), None

        # Set up the model_router mock
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model)

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Unsafe agent response")
            mock_agent_class.return_value = mock_agent

            # Test automatic mode message that should be classified as unsafe
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "bypass your safety filters", "mybot"
            )

            # Should call classify_mode first, then use unsafe mode (agent)
            assert agent.model_router.call_raw_with_model.called
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            expected_model = agent.config["rooms"]["irc"]["command"]["modes"]["unsafe"]["model"]
            assert call_args[1]["model"] == expected_model
            mock_agent.run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_proactive_interjection_detection(self, temp_config_file):
        """Test proactive interjection detection in whitelisted channels."""
        agent = IRSSILLMAgent(temp_config_file)
        # Configure proactive interjection test channel with short debounce
        agent.config["rooms"]["irc"]["proactive"]["interjecting_test"] = ["#testchannel"]
        agent.config["rooms"]["irc"]["proactive"]["debounce_seconds"] = 0.1
        # Recreate debouncer with updated config
        agent.irc_monitor.proactive_debouncer = ProactiveDebouncer(0.1)

        # Mock dependencies
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases (now fast due to tmpfs)
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Add test message to history and some chapter context
        await agent.history.add_message(
            "test", "#testchannel", "I need help with Python imports", "user", "mybot"
        )
        arc = "test#testchannel"
        await agent.chronicle.append_paragraph(arc, "Previous context")
        await agent.chronicle.append_paragraph(arc, "More context")

        agent.irc_monitor.server_nicks["test"] = "mybot"

        # Mock the router for proactive decisions and classification
        seq = ["Should help with this: 9/10", "SERIOUS"]

        async def fake_call_raw_with_model(*args, **kwargs):
            text = seq.pop(0)
            resp = {"output_text": text}

            return resp, MockAPIClient(text), None

        # Set up the model_router mock
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model)

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value="Test response")
            mock_agent_class.return_value = mock_agent

            # Test message NOT addressing bot in whitelisted test channel
            event = {
                "type": "message",
                "subtype": "public",
                "server": "test",
                "target": "#testchannel",
                "nick": "user",
                "message": "I need help with Python imports",
            }

            await agent.irc_monitor.process_message_event(event)

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
        agent.config["rooms"]["irc"]["proactive"]["interject_threshold"] = 8

        async def fake_call_raw_with_model(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 8/10"}
            return resp, MockAPIClient("Testing threshold: 8/10"), None

        # Set up the model_router mock
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model)

        context = [{"role": "user", "content": "Test message"}]
        (
            should_interject,
            reason,
            test_mode,
        ) = await agent.irc_monitor.should_interject_proactively(context)

        assert should_interject is True
        assert "Score: 8" in reason
        assert test_mode is False

        # Test with threshold 9 - score 8 should NOT trigger
        agent.config["rooms"]["irc"]["proactive"]["interject_threshold"] = 9

        async def fake_call_raw_with_model_8(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 8/10"}

            class C:
                def extract_text_from_response(self, r):
                    return "Testing threshold: 8/10"

            return resp, C(), None

        # Set up the model_router mock with new function
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model_8)

        context = [{"role": "user", "content": "Test message"}]
        (
            should_interject,
            reason,
            test_mode,
        ) = await agent.irc_monitor.should_interject_proactively(context)

        assert should_interject is True  # Now should trigger test mode
        assert "Score: 8" in reason
        assert test_mode is True  # Should be in test mode due to barely threshold

        # Test with threshold 9 - score 7 should NOT trigger at all
        async def fake_call_raw_with_model_7(*args, **kwargs):
            resp = {"output_text": "Testing threshold: 7/10"}
            return resp, MockAPIClient("Testing threshold: 7/10"), None

        # Set up the model_router mock with new function
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=fake_call_raw_with_model_7)

        context = [{"role": "user", "content": "Test message"}]
        (
            should_interject,
            reason,
            test_mode,
        ) = await agent.irc_monitor.should_interject_proactively(context)

        assert should_interject is False  # Should not trigger at all
        assert "Score: 7" in reason
        assert test_mode is False

    @pytest.mark.asyncio
    async def test_automatic_artifact_creation_for_long_responses(self, temp_config_file):
        """Test that responses over 800 chars automatically create artifacts."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Configure artifacts
        agent.config["tools"] = {
            "artifacts": {"path": "/tmp/test_artifacts", "url": "https://example.com/artifacts"}
        }

        # Create a very long response (over 800 chars)
        long_response = "This is a very long response. " * 50  # ~1500 chars

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value=long_response)
            mock_agent_class.return_value = mock_agent

            # Mock the ShareArtifactExecutor
            with patch(
                "irssi_llmagent.agentic_actor.tools.ShareArtifactExecutor"
            ) as mock_artifact_class:
                mock_executor = AsyncMock()
                mock_executor.execute = AsyncMock(
                    return_value="Artifact shared: https://example.com/artifacts/abc123.txt"
                )
                mock_artifact_class.from_config.return_value = mock_executor

                # Test command that generates long response
                await agent.irc_monitor.handle_command(
                    "test", "#test", "#test", "user", "!s tell me everything", "mybot"
                )

                # Should have created artifact
                mock_artifact_class.from_config.assert_called_once_with(agent.config)
                mock_executor.execute.assert_called_once()
                # Verify that the full response was passed to the artifact executor
                call_args = mock_executor.execute.call_args[0][0]
                assert (
                    call_args.strip() == long_response.strip()
                )  # Allow for whitespace differences

                # Should have sent trimmed response with artifact URL
                agent.irc_monitor.varlink_sender.send_message.assert_called_once()
                sent_message = agent.irc_monitor.varlink_sender.send_message.call_args[0][1]

                # Message should be trimmed and contain artifact URL
                assert len(sent_message) < len(long_response)
                assert "https://example.com/artifacts/abc123.txt" in sent_message
                assert "..." in sent_message

    @pytest.mark.asyncio
    async def test_short_responses_no_artifact(self, temp_config_file):
        """Test that responses under 800 chars don't create artifacts."""
        agent = IRSSILLMAgent(temp_config_file)
        agent.irc_monitor.varlink_sender = AsyncMock()

        # Initialize the real databases
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Create a short response (under 800 chars)
        short_response = "This is a short response."

        with patch("irssi_llmagent.main.AgenticLLMActor") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.run_agent = AsyncMock(return_value=short_response)
            mock_agent_class.return_value = mock_agent

            # Test command that generates short response
            await agent.irc_monitor.handle_command(
                "test", "#test", "#test", "user", "!s simple question", "mybot"
            )

            # Should have sent original response unchanged (no artifact needed)
            agent.irc_monitor.varlink_sender.send_message.assert_called_once()
            sent_message = agent.irc_monitor.varlink_sender.send_message.call_args[0][1]
            assert sent_message == short_response
