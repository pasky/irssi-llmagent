"""Tests for main application functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from muaddib.agentic_actor.actor import AgentResult
from muaddib.main import MuaddibAgent, cli_message
from muaddib.providers import ModelSpec, UsageInfo


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


class TestMuaddibAgent:
    """Test main agent functionality."""

    def test_load_config(self, temp_config_file, api_type):
        """Test configuration loading."""
        agent = MuaddibAgent(temp_config_file)
        assert agent.config is not None
        assert "providers" in agent.config  # Provider sections exist
        assert "rooms" in agent.config
        assert "irc" in agent.config["rooms"]
        assert "discord" in agent.config["rooms"]
        assert "varlink" in agent.config["rooms"]["irc"]


class TestCLIMode:
    """Test CLI mode functionality."""

    @pytest.mark.asyncio
    async def test_cli_message_sarcastic_message(self, temp_config_file):
        """Test CLI mode with sarcastic message."""
        with patch("builtins.print") as mock_print:
            # Mock the ChatHistory import in cli_message
            with patch("muaddib.main.ChatHistory") as mock_history_class:
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
                from muaddib.main import MuaddibAgent

                agent = MuaddibAgent(temp_config_file)

                async def fake_call_raw_with_model(*args, **kwargs):
                    resp = {"output_text": "Sarcastic response"}

                    return (
                        resp,
                        MockAPIClient("Sarcastic response"),
                        ModelSpec("test", "model"),
                        UsageInfo(None, None, None),
                    )

                # Patch the agent creation in cli_message and model router
                with patch("muaddib.main.MuaddibAgent", return_value=agent):
                    with patch(
                        "muaddib.agentic_actor.actor.ModelRouter.call_raw_with_model",
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
    async def test_cli_message_agent_message(self, temp_config_file):
        """Test CLI mode with agent message."""
        with patch("builtins.print") as mock_print:
            # Mock AgenticLLMActor to return a fake response
            with patch("muaddib.main.AgenticLLMActor") as mock_agent_class:
                mock_agent = AsyncMock()
                mock_agent.run_agent = AsyncMock(
                    return_value=AgentResult(
                        text="Agent response",
                        total_input_tokens=100,
                        total_output_tokens=50,
                        total_cost=0.01,
                        tool_calls_count=3,
                    )
                )
                mock_agent_class.return_value = mock_agent

                await cli_message("!s search for Python news", temp_config_file)

                # Verify agent was called (means serious mode was triggered)
                assert mock_agent.run_agent.called

                # Verify output shows simulation and response via reply_sender
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any(
                    "Simulating IRC message: !s search for Python news" in call
                    for call in print_calls
                )
                assert any("📤 Bot response: Agent response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_message_content_validation(self, temp_config_file):
        """Test that CLI mode processes message content correctly."""
        with patch("builtins.print") as mock_print:
            # Mock AgenticLLMActor to capture the context it receives
            with patch("muaddib.main.AgenticLLMActor") as mock_agent_class:
                mock_agent = AsyncMock()
                captured_context: list[dict[str, str]] | None = None

                async def capture_run_agent(context, **kwargs):
                    nonlocal captured_context
                    captured_context = context
                    return AgentResult(
                        text="Agent response",
                        total_input_tokens=100,
                        total_output_tokens=50,
                        total_cost=0.01,
                        tool_calls_count=3,
                    )

                mock_agent.run_agent = AsyncMock(side_effect=capture_run_agent)
                mock_agent_class.return_value = mock_agent

                await cli_message("!s specific test message", temp_config_file)

                # Verify agent was called and context contains the message
                assert mock_agent.run_agent.called
                assert captured_context is not None

                # The last message must be the actual user message content (not a placeholder)
                assert isinstance(captured_context, list)
                assert captured_context, "Context must not be empty"
                assert "specific test message" in captured_context[-1]["content"]
                assert captured_context[-1]["content"] != "..."

                # Verify output shows the response via reply_sender
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("📤 Bot response: Agent response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_config_not_found(self):
        """Test CLI mode handles missing config file."""
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                await cli_message("test query", "/nonexistent/config.json")

                mock_exit.assert_called_with(1)  # Just check it was called with 1, not once
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Config file not found" in call for call in print_calls)
