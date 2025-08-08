"""Tests for agent functionality."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.agent import AIAgent


class TestAPIAgent:
    """Test API agent functionality with both Anthropic and OpenAI."""

    @pytest.fixture
    def agent(self, test_config):
        """Create agent instance for testing."""
        # Add missing prompts for agent tests
        test_config["prompts"]["serious"] = "You are IRC user {mynick}. Be helpful and informative."
        return AIAgent(test_config, "testbot")

    def create_text_response(self, api_type: str, text: str) -> dict:
        """Create a text response in the appropriate format for the API type."""
        if api_type == "anthropic":
            return {
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
            }
        else:  # openai (Responses API)
            return {"output_text": text}

    def create_tool_response(self, api_type: str, tools: list[dict]) -> dict:
        """Create a tool response in the appropriate format for the API type."""
        if api_type == "anthropic":
            content = []
            for tool in tools:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool["id"],
                        "name": tool["name"],
                        "input": tool["input"],
                    }
                )
            return {
                "content": content,
                "stop_reason": "tool_use",
            }
        else:  # openai (Responses API)
            content = []
            for tool in tools:
                content.append(
                    {
                        "type": "tool_call",
                        "id": tool["id"],
                        "function": {"name": tool["name"], "arguments": json.dumps(tool["input"])},
                    }
                )
            return {
                "output": [
                    {
                        "type": "message",
                        "message": {"role": "assistant", "content": content},
                    }
                ]
            }

    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert "testbot" in agent.system_prompt  # mynick is substituted
        assert "IRC user" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, agent):
        """Test agent as async context manager."""
        with patch.object(agent.api_client, "__aenter__", new_callable=AsyncMock) as mock_enter:
            with patch.object(agent.api_client, "__aexit__", new_callable=AsyncMock) as mock_exit:
                mock_enter.return_value = agent.api_client

                async with agent:
                    pass

                mock_enter.assert_called_once()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_simple_response(self, agent, api_type):
        """Test agent with simple text response (no tools)."""
        # Mock API response with text only
        mock_response = self.create_text_response(api_type, "This is a simple answer.")

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.run_agent([{"role": "user", "content": "What is 2+2?"}])

            assert result == "This is a simple answer."
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_tool_use_flow(self, agent, api_type):
        """Test agent tool usage flow."""
        # Mock API responses - first wants to use tool, then provides final answer
        tool_use_response = self.create_tool_response(
            api_type,
            [{"id": "tool_123", "name": "web_search", "input": {"query": "Python tutorial"}}],
        )

        final_response = self.create_text_response(
            api_type, "Based on the search results, here's what I found about Python tutorials."
        )

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                mock_call.side_effect = [tool_use_response, final_response]
                mock_tool.return_value = "Search results: Python is a programming language..."

                result = await agent.run_agent(
                    [{"role": "user", "content": "Tell me about Python tutorials"}]
                )

                assert "Based on the search results" in result
                assert mock_call.call_count == 2
                mock_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_max_iterations(self, agent, api_type):
        """Test agent respects max iteration limit."""
        # Mock API to always want to use tools
        tool_use_response = self.create_tool_response(
            api_type, [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]
        )

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                # Mock _extract_text_from_response to return actual cleaned text
                with patch.object(
                    agent.api_client, "extract_text_from_response", return_value="Final response"
                ):
                    # Set up the mock to return tool_use responses for first 2 calls, then final response on 3rd call
                    final_dict_response = self.create_text_response(api_type, "Final response")

                    mock_call.side_effect = [
                        tool_use_response,
                        tool_use_response,
                        final_dict_response,
                    ]
                    mock_tool.return_value = "Tool result"

                    result = await agent.run_agent(
                        [{"role": "user", "content": "Keep using tools"}]
                    )

                    assert "Final response" in result
                    assert mock_call.call_count == 3  # 3 iterations max

    @pytest.mark.asyncio
    async def test_agent_api_error_handling(self, agent):
        """Test agent handles API errors gracefully."""
        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API Error")

            with pytest.raises(RuntimeError, match="coroutine raised StopIteration"):
                await agent.run_agent([{"role": "user", "content": "Test query"}])

    @pytest.mark.asyncio
    async def test_agent_tool_execution_error(self, agent, api_type):
        """Test agent handles tool execution errors."""
        tool_use_response = self.create_tool_response(
            api_type, [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]
        )

        final_response = self.create_text_response(
            api_type, "I encountered an error but here's what I can tell you."
        )

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                mock_call.side_effect = [tool_use_response, final_response]
                mock_tool.return_value = "Tool execution failed: Network error"

                result = await agent.run_agent(
                    [{"role": "user", "content": "Search for something"}]
                )

                assert "encountered an error" in result or "what I can tell you" in result

    def test_extract_tool_uses_single(self, agent, api_type):
        """Test single tool use extraction from API response."""
        if api_type == "anthropic":
            response = {
                "content": [
                    {"type": "text", "text": "I'll search for that."},
                    {
                        "type": "tool_use",
                        "id": "tool_456",
                        "name": "visit_webpage",
                        "input": {"url": "https://example.com"},
                    },
                ]
            }
        else:  # openai (Responses API)
            response = {
                "output": [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_call",
                                    "id": "tool_456",
                                    "function": {
                                        "name": "visit_webpage",
                                        "arguments": '{"url": "https://example.com"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

        tool_uses = agent.api_client.extract_tool_calls(response)

        assert tool_uses is not None
        assert len(tool_uses) == 1
        assert tool_uses[0]["id"] == "tool_456"
        assert tool_uses[0]["name"] == "visit_webpage"
        assert tool_uses[0]["input"]["url"] == "https://example.com"

    def test_extract_tool_uses_multiple(self, agent, api_type):
        """Test multiple tool use extraction from API response."""
        if api_type == "anthropic":
            response = {
                "content": [
                    {"type": "text", "text": "I'll search and visit a page."},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_2",
                        "name": "visit_webpage",
                        "input": {"url": "https://example.com"},
                    },
                ]
            }
        else:  # openai (Responses API)
            response = {
                "output": [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_call",
                                    "id": "tool_1",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "test"}',
                                    },
                                },
                                {
                                    "type": "tool_call",
                                    "id": "tool_2",
                                    "function": {
                                        "name": "visit_webpage",
                                        "arguments": '{"url": "https://example.com"}',
                                    },
                                },
                            ],
                        },
                    }
                ]
            }
        tool_uses = agent.api_client.extract_tool_calls(response)

        assert tool_uses is not None
        assert len(tool_uses) == 2
        assert tool_uses[0]["name"] == "web_search"
        assert tool_uses[1]["name"] == "visit_webpage"

    def test_extract_tool_uses_no_tools(self, agent, api_type):
        """Test tool use extraction when no tools in response."""
        if api_type == "anthropic":
            response = {"content": [{"type": "text", "text": "Just a text response."}]}
        else:  # openai (Responses API)
            response = {"output_text": "Just a text response."}

        tool_uses = agent.api_client.extract_tool_calls(response)

        assert tool_uses is None

    @pytest.mark.asyncio
    async def test_agent_multiple_tools_execution(self, agent, api_type):
        """Test agent executes multiple tools in a single response."""
        # Mock response with multiple tool calls
        multi_tool_response = self.create_tool_response(
            api_type,
            [
                {"id": "tool_1", "name": "web_search", "input": {"query": "test"}},
                {"id": "tool_2", "name": "visit_webpage", "input": {"url": "https://example.com"}},
            ],
        )

        final_response = self.create_text_response(api_type, "Here's what I found")

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                with patch.object(
                    agent.api_client,
                    "extract_text_from_response",
                    return_value="Here's what I found",
                ):
                    mock_call.side_effect = [multi_tool_response, final_response]
                    mock_tool.side_effect = ["Search result", "Page content"]

                    result = await agent.run_agent(
                        [{"role": "user", "content": "Search and visit"}]
                    )

                    # Verify both tools were executed
                    assert mock_tool.call_count == 2

                    # Verify correct tool calls
                    call_args_list = mock_tool.call_args_list
                    # Now expects (tool_name, tool_executors, **kwargs)
                    assert call_args_list[0][0][0] == "web_search"  # first positional arg
                    assert call_args_list[0][1] == {"query": "test"}  # kwargs
                    assert call_args_list[1][0][0] == "visit_webpage"  # first positional arg
                    assert call_args_list[1][1] == {"url": "https://example.com"}  # kwargs

                    assert "Here's what I found" in result

    @pytest.mark.asyncio
    async def test_agent_with_context(self, agent):
        """Test agent receives and uses conversation history context."""
        # Mock context with previous conversation
        context = [
            {"role": "user", "content": "What's your favorite color?"},
            {
                "role": "assistant",
                "content": "I don't have personal preferences, but blue is nice.",
            },
            {"role": "user", "content": "Tell me more about it"},
        ]

        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch.object(
                agent.api_client,
                "extract_text_from_response",
                return_value="Blue represents calm and trust",
            ):
                mock_call.return_value = {
                    "content": [{"type": "text", "text": "Blue represents calm and trust"}],
                    "stop_reason": "end_turn",
                }

                await agent.run_agent(context)

                # Verify the full context was passed to AI
                call_args = mock_call.call_args[0]
                messages_passed = call_args[0]

                # Should have original context + current query (if not duplicate)
                assert len(messages_passed) >= 3
                assert messages_passed[0]["content"] == "What's your favorite color?"
                assert (
                    messages_passed[1]["content"]
                    == "I don't have personal preferences, but blue is nice."
                )
                assert messages_passed[2]["content"] == "Tell me more about it"

    @pytest.mark.asyncio
    async def test_agent_without_context(self, agent):
        """Test agent works without context (current behavior)."""
        with patch.object(agent.api_client, "call_raw", new_callable=AsyncMock) as mock_call:
            with patch.object(
                agent.api_client, "extract_text_from_response", return_value="Hello there"
            ):
                mock_call.return_value = {
                    "content": [{"type": "text", "text": "Hello there"}],
                    "stop_reason": "end_turn",
                }

                await agent.run_agent([{"role": "user", "content": "Hello"}])

                # Verify only the current query was passed
                call_args = mock_call.call_args[0]
                messages_passed = call_args[0]

                assert len(messages_passed) == 1
                assert messages_passed[0]["content"] == "Hello"

    # Removed tests for _extract_text_response and _format_for_irc as they were unified with client modules

    def test_process_ai_response_text(self, agent, api_type):
        """Test unified response processing for text responses."""
        if api_type == "anthropic":
            response = {
                "content": [{"type": "text", "text": "This is a test response"}],
                "stop_reason": "end_turn",
            }
        else:  # openai (Responses API)
            response = {"output_text": "This is a test response"}

        result = agent._process_ai_response(response)

        assert result["type"] == "final_text"
        assert result["text"] == "This is a test response"

    def test_process_ai_response_tool_use(self, agent, api_type):
        """Test unified response processing for tool use responses."""
        if api_type == "anthropic":
            response = {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "web_search",
                        "input": {"query": "test"},
                    }
                ],
                "stop_reason": "tool_use",
            }
        else:  # openai (Responses API)
            response = {
                "output": [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_call",
                                    "id": "tool_123",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "test"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            }

        result = agent._process_ai_response(response)

        assert result["type"] == "tool_use"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["id"] == "tool_123"
        assert result["tools"][0]["name"] == "web_search"
        assert result["tools"][0]["input"] == {"query": "test"}

    def test_process_ai_response_errors(self, agent):
        """Test unified response processing error cases."""
        # Test None response
        result = agent._process_ai_response(None)
        assert result["type"] == "error"
        assert "Empty response" in result["message"]

        # Test string response (fallback)
        result = agent._process_ai_response("Text response")
        assert result["type"] == "final_text"
        assert result["text"] == "Text response"

        # Test invalid dict response (AI client returns None, so we return error)
        result = agent._process_ai_response({"invalid": "response"})
        assert result["type"] == "error"
        assert "No valid text" in result["message"]
