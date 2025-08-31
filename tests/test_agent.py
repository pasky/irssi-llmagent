"""Tests for agent functionality."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.agent import AIAgent
from irssi_llmagent.model_router import ModelRouter, ModelSpec


class TestAPIAgent:
    """Test API agent functionality with both Anthropic and OpenAI."""

    @pytest.fixture
    def agent(self, test_config):
        """Create agent instance for testing."""
        # Add missing prompts for agent tests
        test_config["command"]["prompts"][
            "serious"
        ] = "You are IRC user {mynick}. Be helpful and informative. Available models: serious={serious_model}, sarcastic={sarcastic_model}."
        return AIAgent(test_config, "testbot", mode="serious")

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
        system_prompt = agent._get_system_prompt("test:model")
        assert "testbot" in system_prompt  # mynick is substituted
        assert "IRC user" in system_prompt
        assert (
            "serious=" in system_prompt and "dummy-serious" in system_prompt
        )  # serious model is substituted
        assert (
            "sarcastic=" in system_prompt and "dummy-sarcastic" in system_prompt
        )  # sarcastic model is substituted

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, agent):
        """Test agent as async context manager."""
        with patch.object(ModelRouter, "__aenter__", new_callable=AsyncMock) as mock_enter:
            with patch.object(ModelRouter, "__aexit__", new_callable=AsyncMock) as mock_exit:
                mock_enter.return_value = ModelRouter(agent.config)

                async with agent:
                    pass

                mock_enter.assert_called_once()
                mock_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_simple_response(self, agent, api_type):
        """Test agent with simple text response (no tools)."""
        # Mock API response with text only
        mock_response = self.create_text_response(api_type, "This is a simple answer.")

        class FakeClient:
            def extract_text_from_response(self, r):
                return "This is a simple answer."

            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": "ok"}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        async def fake_call_raw_with_model(*args, **kwargs):
            return mock_response, FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
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

        class FakeClient:
            def extract_text_from_response(self, r):
                if r is final_response:
                    return (
                        "Based on the search results, here's what I found about Python tutorials."
                    )
                return ""

            def has_tool_calls(self, r):
                return r is tool_use_response

            def extract_tool_calls(self, r):
                if r is tool_use_response:
                    return [
                        {
                            "id": "tool_123",
                            "name": "web_search",
                            "input": {"query": "Python tutorial"},
                        }
                    ]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        seq = [tool_use_response, final_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
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

        class FakeClient:
            def extract_text_from_response(self, r):
                if isinstance(r, dict) and (
                    r.get("output_text") == "Final response"
                    or any(
                        (c.get("type") == "text" and c.get("text") == "Final response")
                        for c in r.get("content", [])
                    )
                ):
                    return "Final response"
                return ""

            def has_tool_calls(self, r):
                return r == tool_use_response

            def extract_tool_calls(self, r):
                if r == tool_use_response:
                    return [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        final_dict_response = self.create_text_response(api_type, "Final response")
        seq = [tool_use_response, tool_use_response, final_dict_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                mock_tool.return_value = "Tool result"

                result = await agent.run_agent([{"role": "user", "content": "Keep using tools"}])

                assert "Final response" in result
                assert mock_call.call_count == 3  # 3 iterations max

    @pytest.mark.asyncio
    async def test_agent_api_error_handling(self, agent):
        """Test agent handles API errors gracefully."""

        async def fake_call_raw_with_model(*args, **kwargs):
            raise Exception("API Error")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
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

        class FakeClient:
            def extract_text_from_response(self, r):
                if r is final_response:
                    return "I encountered an error but here's what I can tell you."
                return ""

            def has_tool_calls(self, r):
                return r is tool_use_response

            def extract_tool_calls(self, r):
                if r is tool_use_response:
                    return [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        seq = [tool_use_response, final_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
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

        class FakeClient:
            def extract_tool_calls(self, r):
                # Reuse provider shapes
                if "content" in r:
                    return [
                        {
                            "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input", {}),
                        }
                        for b in r.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ] or None
                outputs = r.get("output") or []
                calls = []
                for item in outputs:
                    if item.get("type") == "message":
                        msg = item.get("message") if isinstance(item.get("message"), dict) else item
                        for c in msg.get("content", []):
                            if c.get("type") == "tool_call":
                                fn = c.get("function", {})
                                import json as _json

                                args = fn.get("arguments")
                                if isinstance(args, str):
                                    try:
                                        args = _json.loads(args)
                                    except Exception:
                                        args = {}
                                calls.append(
                                    {"id": c.get("id"), "name": fn.get("name"), "input": args}
                                )
                return calls or None

        tool_uses = FakeClient().extract_tool_calls(response)

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

        class FakeClient:
            def extract_tool_calls(self, r):
                # Reuse provider shapes
                if "content" in r:
                    return [
                        {
                            "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input", {}),
                        }
                        for b in r.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ] or None
                outputs = r.get("output") or []
                calls = []
                for item in outputs:
                    if item.get("type") == "message":
                        msg = item.get("message") if isinstance(item.get("message"), dict) else item
                        for c in msg.get("content", []):
                            if c.get("type") == "tool_call":
                                fn = c.get("function", {})
                                import json as _json

                                args = fn.get("arguments")
                                if isinstance(args, str):
                                    try:
                                        args = _json.loads(args)
                                    except Exception:
                                        args = {}
                                calls.append(
                                    {"id": c.get("id"), "name": fn.get("name"), "input": args}
                                )
                return calls or None

        tool_uses = FakeClient().extract_tool_calls(response)

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

        class FakeClient:
            def extract_tool_calls(self, r):
                # Reuse provider shapes
                if "content" in r:
                    return [
                        {
                            "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input", {}),
                        }
                        for b in r.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ] or None
                outputs = r.get("output") or []
                calls = []
                for item in outputs:
                    if item.get("type") == "message":
                        msg = item.get("message") if isinstance(item.get("message"), dict) else item
                        for c in msg.get("content", []):
                            if c.get("type") == "tool_call":
                                fn = c.get("function", {})
                                import json as _json

                                args = fn.get("arguments")
                                if isinstance(args, str):
                                    try:
                                        args = _json.loads(args)
                                    except Exception:
                                        args = {}
                                calls.append(
                                    {"id": c.get("id"), "name": fn.get("name"), "input": args}
                                )
                return calls or None

        tool_uses = FakeClient().extract_tool_calls(response)

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

        class FakeClient:
            def extract_text_from_response(self, r):
                if r is final_response:
                    return "Here's what I found"
                return ""

            def has_tool_calls(self, r):
                return r is multi_tool_response

            def extract_tool_calls(self, r):
                if r is multi_tool_response:
                    return [
                        {"id": "tool_1", "name": "web_search", "input": {"query": "test"}},
                        {
                            "id": "tool_2",
                            "name": "visit_webpage",
                            "input": {"url": "https://example.com"},
                        },
                    ]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        seq = [multi_tool_response, final_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch("irssi_llmagent.agent.execute_tool", new_callable=AsyncMock) as mock_tool:
                mock_tool.side_effect = ["Search result", "Page content"]

                result = await agent.run_agent([{"role": "user", "content": "Search and visit"}])

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

        class FakeClient:
            def extract_text_from_response(self, r):
                return "Blue represents calm and trust"

            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        async def fake_call_raw_with_model(model, messages, system_prompt, **kwargs):
            return (
                {
                    "content": [{"type": "text", "text": "Blue represents calm and trust"}],
                    "stop_reason": "end_turn",
                },
                FakeClient(),
                ModelSpec("anthropic", "dummy"),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            await agent.run_agent(context)

            # Verify the full context was passed to AI
            call_args = mock_call.call_args[0]
            # args were (model, messages, system_prompt)
            messages_passed = call_args[1]

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

        class FakeClient:
            def extract_text_from_response(self, r):
                return "Hello there"

            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        async def fake_call_raw_with_model(model, messages, system_prompt, **kwargs):
            return (
                {
                    "content": [{"type": "text", "text": "Hello there"}],
                    "stop_reason": "end_turn",
                },
                FakeClient(),
                ModelSpec("anthropic", "dummy"),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            await agent.run_agent([{"role": "user", "content": "Hello"}])

            # Verify only the current query was passed
            call_args = mock_call.call_args[0]
            messages_passed = call_args[1]

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

        class FakeClient:
            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def extract_text_from_response(self, r):
                return "This is a test response"

        result = agent._process_ai_response_provider(response, FakeClient())

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

        class FakeClient:
            def has_tool_calls(self, r):
                return True

            def extract_tool_calls(self, r):
                return [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]

            def extract_text_from_response(self, r):
                return ""

        result = agent._process_ai_response_provider(response, FakeClient())

        assert result["type"] == "tool_use"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["id"] == "tool_123"
        assert result["tools"][0]["name"] == "web_search"
        assert result["tools"][0]["input"] == {"query": "test"}

    def test_process_ai_response_errors(self, agent):
        """Test unified response processing error cases."""

        # Test None response
        # Using provider-aware path now
        class FakeClient:
            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def extract_text_from_response(self, r):
                return ""

        result = agent._process_ai_response_provider(None, FakeClient())
        assert result["type"] == "error"
        assert "Empty response" in result["message"]

        # Test string response (fallback)
        result = agent._process_ai_response_provider("Text response", FakeClient())
        assert result["type"] == "final_text"
        assert result["text"] == "Text response"

        # Test invalid dict response (AI client returns None, so we return error)
        result = agent._process_ai_response_provider({"invalid": "response"}, FakeClient())
        assert result["type"] == "error"
        assert "No valid text" in result["message"]

    @pytest.mark.asyncio
    async def test_claude_thinking_budget_with_non_auto_tool_choice(self):
        """Test Claude's special handling when thinking budget is set with non-auto tool_choice."""

        # Create a Claude client instance for direct testing
        from irssi_llmagent.anthropic import AnthropicClient

        test_config = {
            "providers": {
                "anthropic": {"key": "test-key", "url": "https://api.anthropic.com/v1/messages"}
            }
        }

        claude_client = AnthropicClient(test_config)

        # Mock the HTTP session to capture the payload
        captured_payload = {}

        class MockResponse:
            def __init__(self):
                self.ok = True
                self.status = 200

            async def json(self):
                return {
                    "content": [{"type": "text", "text": "Final answer"}],
                    "stop_reason": "end_turn",
                }

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            def post(self, url, json=None):
                captured_payload.update(json or {})
                return MockResponse()

            async def close(self):
                pass

        # Test the scenario: thinking budget + non-auto tool_choice
        messages = [{"role": "user", "content": "Test query"}]

        with patch("aiohttp.ClientSession", MockSession):
            async with claude_client:
                await claude_client.call_raw(
                    messages,
                    "Test system prompt",
                    "claude-3-5-sonnet-20241022",
                    tools=[{"name": "final_answer", "description": "test tool"}],
                    tool_choice=["final_answer"],  # Non-auto tool choice
                    reasoning_effort="medium",  # Sets thinking budget
                )

        # Verify the special case was handled correctly
        assert "thinking" in captured_payload  # Thinking budget was set
        assert captured_payload["thinking"]["budget_tokens"] == 4096  # Medium effort

        # Assert exact literal value of messages[]
        expected_messages = [
            {"role": "user", "content": "Test query"},
            {
                "role": "user",
                "content": "<meta>only tool ['final_answer'] may be called now</meta>",
            },
        ]
        assert captured_payload["messages"] == expected_messages

        # Verify tool_choice is not set in payload due to thinking budget incompatibility
        assert "tool_choice" not in captured_payload

    @pytest.mark.asyncio
    async def test_openai_thinking_budget_with_non_auto_tool_choice(self):
        """Test OpenAI's handling when reasoning effort is set with non-auto tool_choice."""

        # Create an OpenAI client instance for direct testing
        from irssi_llmagent.openai import OpenAIClient

        test_config = {"providers": {"openai": {"key": "test-key"}}}

        openai_client = OpenAIClient(test_config)

        # Mock the OpenAI SDK client to capture the payload
        captured_kwargs = {}

        class MockResponse:
            def model_dump(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Final answer",
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {"name": "final_answer", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                }

        class MockAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                self.chat = self.MockChat()

            class MockChat:
                def __init__(self):
                    self.completions = self.MockCompletions()

                class MockCompletions:
                    async def create(self, **kwargs):
                        captured_kwargs.update(kwargs)
                        return MockResponse()

        # Test the scenario: reasoning effort + non-auto tool_choice
        messages = [{"role": "user", "content": "Test query"}]

        with patch("irssi_llmagent.openai._AsyncOpenAI", MockAsyncOpenAI):
            async with openai_client:
                await openai_client.call_raw(
                    messages,
                    "Test system prompt",
                    "gpt-5",
                    tools=[
                        {"name": "final_answer", "description": "test tool", "input_schema": {}}
                    ],
                    tool_choice=["final_answer"],  # Non-auto tool choice
                    reasoning_effort="medium",  # Sets reasoning effort
                )

        # Assert exact literal value of messages (reasoning models use developer role)
        expected_messages = [
            {"role": "developer", "content": "Test system prompt"},
            {"role": "user", "content": "Test query"},
        ]
        assert captured_kwargs["messages"] == expected_messages

        # Chat Completions API doesn't use reasoning parameter for non-o1 models

        # Verify tool_choice was set correctly for Chat Completions API (with allowed_tools)
        expected_tool_choice = captured_kwargs["tool_choice"]
        assert expected_tool_choice == {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "required",
                "tools": [{"type": "function", "function": {"name": "final_answer"}}],
            },
        }

        # Verify tools were converted properly for Chat Completions format
        assert len(captured_kwargs["tools"]) == 1
        assert captured_kwargs["tools"][0]["type"] == "function"
        assert captured_kwargs["tools"][0]["function"]["name"] == "final_answer"

    @pytest.mark.asyncio
    async def test_openai_api_reasoning_vs_legacy_model_handling(self):
        """Test OpenAI API handles reasoning models vs legacy models differently."""
        from irssi_llmagent.openai import OpenAIClient

        config = {"providers": {"openai": {"key": "test-key"}}}

        # Mock to capture API calls
        captured_kwargs = {}

        class MockResponse:
            def model_dump(self):
                return {"choices": [{"message": {"content": "test response"}}]}

        class MockAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                self.chat = self.MockChat()

            class MockChat:
                def __init__(self):
                    self.completions = self.MockCompletions()

                class MockCompletions:
                    async def create(self, **kwargs):
                        captured_kwargs.update(kwargs)
                        return MockResponse()

        # Test legacy model (gpt-4o)
        captured_kwargs.clear()
        client = OpenAIClient(config)

        with patch("irssi_llmagent.openai._AsyncOpenAI", MockAsyncOpenAI):
            async with client:
                await client.call_raw(
                    [{"role": "user", "content": "Test message"}],
                    "Test system prompt",
                    "gpt-4o",  # Legacy model
                    tools=[
                        {
                            "name": "tool_a",
                            "description": "Tool A",
                            "input_schema": {"type": "object"},
                        },
                        {
                            "name": "tool_b",
                            "description": "Tool B",
                            "input_schema": {"type": "object"},
                        },
                    ],
                    tool_choice=["tool_a", "tool_b"],
                    reasoning_effort="high",
                )

        # Legacy model should use system role and max_tokens
        assert captured_kwargs["messages"][0]["role"] == "system"
        assert "max_tokens" in captured_kwargs
        assert "max_completion_tokens" not in captured_kwargs
        # Legacy should use meta messages for tool choice and reasoning
        assert "tool_choice" not in captured_kwargs
        assert "reasoning_effort" not in captured_kwargs
        meta_messages = [
            msg for msg in captured_kwargs["messages"] if "meta>" in str(msg.get("content", ""))
        ]
        assert len(meta_messages) >= 2  # reasoning + tool choice meta messages

        # Test reasoning model (gpt-5)
        captured_kwargs.clear()

        with patch("irssi_llmagent.openai._AsyncOpenAI", MockAsyncOpenAI):
            async with client:
                await client.call_raw(
                    [{"role": "user", "content": "Test message"}],
                    "Test system prompt",
                    "gpt-5",  # Reasoning model
                    tools=[
                        {
                            "name": "tool_c",
                            "description": "Tool C",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    tool_choice=["tool_c"],
                    reasoning_effort="medium",
                )

        # Reasoning model should use developer role and max_completion_tokens
        assert captured_kwargs["messages"][0]["role"] == "developer"
        assert "max_completion_tokens" in captured_kwargs
        assert "max_tokens" not in captured_kwargs
        # Reasoning should use direct API parameters
        assert "reasoning_effort" in captured_kwargs
        assert "tool_choice" in captured_kwargs

    @pytest.mark.asyncio
    async def test_agent_max_tokens_tool_retry(self, agent):
        """Test agent handles max_tokens truncated tool calls by retrying."""
        # Create truncated Claude response first, then successful response
        truncated_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "web_search",
                    "input": {},  # Empty input due to truncation
                }
            ],
            "stop_reason": "max_tokens",
        }
        final_response = {
            "content": [{"type": "text", "text": "Here's the final answer."}],
            "stop_reason": "end_turn",
        }

        class FakeClient:
            def extract_text_from_response(self, r):
                if r is final_response:
                    return "Here's the final answer."
                return ""

            def has_tool_calls(self, r):
                return r.get("stop_reason") == "tool_use"

            def extract_tool_calls(self, r):
                # Only return tool calls for non-truncated responses
                if r.get("stop_reason") == "tool_use":
                    return [{"id": "tool_123", "name": "web_search", "input": {"query": "test"}}]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": r.get("content", [])}

            def format_tool_results(self, results):
                return {"role": "user", "content": results}

        seq = [truncated_response, final_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("anthropic", "dummy")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            result = await agent.run_agent([{"role": "user", "content": "Search for something"}])

            # Should have made 2 calls - first truncated, then retry
            assert mock_call.call_count == 2
            assert "final answer" in result
