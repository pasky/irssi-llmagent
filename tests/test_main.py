"""Tests for main application functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.main import IRSSILLMAgent, cli_message


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
        assert "rooms" in agent.config
        assert "irc" in agent.config["rooms"]
        assert "varlink" in agent.config["rooms"]["irc"]


class TestCLIMode:
    """Test CLI mode functionality."""

    @pytest.mark.asyncio
    async def test_cli_message_sarcastic_message(self, temp_config_file):
        """Test CLI mode with sarcastic message."""
        with patch("builtins.print") as mock_print:
            # Mock the ChatHistory import in cli_message
            with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                mock_history = AsyncMock()
                mock_history.add_message = AsyncMock()
                mock_history.get_context.return_value = [
                    {"role": "user", "content": "!S tell me a joke"}
                ]
                # Add new chronicling methods
                mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                mock_history.mark_chronicled = AsyncMock()
                mock_history_class.return_value = mock_history

                # Create a real agent
                from irssi_llmagent.main import IRSSILLMAgent

                agent = IRSSILLMAgent(temp_config_file)

                async def fake_call_raw_with_model(*args, **kwargs):
                    resp = {"output_text": "Sarcastic response"}

                    return resp, MockAPIClient("Sarcastic response"), None

                # Patch the agent creation in cli_message and model router
                with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                    with patch(
                        "irssi_llmagent.agentic_actor.actor.ModelRouter.call_raw_with_model",
                        new=AsyncMock(side_effect=fake_call_raw_with_model),
                    ):
                        await cli_message("!S tell me a joke", temp_config_file)

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating IRC message: !S tell me a joke" in call
                            for call in print_calls
                        )
                        assert any("Sarcastic response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_perplexity_message(self, temp_config_file):
        """Test CLI mode with Perplexity message."""
        with patch("builtins.print") as mock_print:
            with patch(
                "irssi_llmagent.rooms.irc.monitor.PerplexityClient"
            ) as mock_perplexity_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!p what is the weather?"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_perplexity = AsyncMock()
                    mock_perplexity.call_perplexity = AsyncMock(return_value="Weather is sunny")
                    mock_perplexity_class.return_value.__aenter__.return_value = mock_perplexity

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_message("!p what is the weather?", temp_config_file)

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
    async def test_cli_message_agent_message(self, temp_config_file):
        """Test CLI mode with agent message."""
        with patch("builtins.print") as mock_print:
            with patch("irssi_llmagent.rooms.irc.monitor.AgenticLLMActor") as mock_agent_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s search for Python news"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value.__aenter__.return_value = mock_agent

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_message("!s search for Python news", temp_config_file)

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
    async def test_cli_message_message_content_validation(self, temp_config_file):
        """Test that CLI mode passes actual message content, not placeholder text."""
        with patch("builtins.print"):
            with patch("irssi_llmagent.rooms.irc.monitor.AgenticLLMActor") as mock_agent_class:
                with patch("irssi_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s specific test message"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value.__aenter__.return_value = mock_agent

                    # Create a real agent
                    from irssi_llmagent.main import IRSSILLMAgent

                    agent = IRSSILLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("irssi_llmagent.main.IRSSILLMAgent", return_value=agent):
                        await cli_message("!s specific test message", temp_config_file)

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
    async def test_cli_message_config_not_found(self):
        """Test CLI mode handles missing config file."""
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                await cli_message("test query", "/nonexistent/config.json")

                mock_exit.assert_called_with(1)  # Just check it was called with 1, not once
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Config file not found" in call for call in print_calls)
