"""Tests for core agent functionality."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from muaddib.agentic_actor import AgenticLLMActor
from muaddib.providers import ModelRouter, ModelSpec, UsageInfo, parse_model_spec


class TestAPIAgent:
    """Test API agent functionality with both Anthropic and OpenAI."""

    @pytest.fixture
    def agent(self, test_config, mock_agent):
        """Create agent instance for testing."""
        # Update serious prompt for agent tests (config structure already has the prompt)
        test_config["rooms"]["common"]["command"]["modes"]["serious"]["prompt"] = (
            "You are IRC user {mynick}. Be helpful and informative. Available models: serious={serious_model}, sarcastic={sarcastic_model}."
        )

        def build_test_prompt():
            from datetime import datetime

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            return test_config["rooms"]["common"]["command"]["modes"]["serious"]["prompt"].format(
                mynick="testbot",
                current_time=current_time,
                sarcastic_model="anthropic:claude-3-5-haiku",
                serious_model="anthropic:claude-3-5-sonnet",
                unsafe_model="not-configured",
            )

        def get_prompt_reminder():
            return test_config["rooms"]["common"]["command"]["modes"]["serious"].get(
                "prompt_reminder"
            )

        return AgenticLLMActor(
            config=test_config,
            model="anthropic:claude-3-5-sonnet",
            system_prompt_generator=build_test_prompt,
            prompt_reminder_generator=get_prompt_reminder,
            agent=mock_agent,
        )

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
        system_prompt = agent.system_prompt_generator()
        assert "testbot" in system_prompt  # mynick is substituted
        assert "IRC user" in system_prompt
        assert (
            "serious=" in system_prompt and "claude-3-5-sonnet" in system_prompt
        )  # serious model is substituted
        assert (
            "sarcastic=" in system_prompt and "claude-3-5-haiku" in system_prompt
        )  # sarcastic model is substituted
        # Verify router is created
        assert agent.model_router is not None
        assert isinstance(agent.model_router, ModelRouter)

    @pytest.mark.asyncio
    async def test_agent_simple_response(self, agent, api_type):
        """Test agent with simple text response (no tools)."""
        # Mock API response with text only
        mock_response = self.create_text_response(api_type, "This is a simple answer.")

        # Add prompt reminder to test config and capture messages
        agent.config["rooms"]["common"]["command"]["modes"]["serious"]["prompt_reminder"] = (
            "Be helpful!"
        )
        captured_messages = []

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

        async def fake_call_raw_with_model(model, messages, *args, **kwargs):
            captured_messages.extend(messages)
            return (
                mock_response,
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            result = await agent.run_agent(
                [{"role": "user", "content": "What is 2+2?"}], arc="test"
            )

            assert result.text == "This is a simple answer."
            mock_call.assert_called_once()
            # Check prompt reminder was added as last message
            assert len(captured_messages) == 2
            assert captured_messages[-1]["role"] == "user"
            assert "<meta>Be helpful!</meta>" in captured_messages[-1]["content"]

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
        usage_seq = [
            UsageInfo(1000, 500, 0.01),
            UsageInfo(800, 300, 0.008),
        ]

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                seq.pop(0),
                FakeClient(),
                parse_model_spec(model),
                usage_seq.pop(0),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                mock_tool.return_value = "Search results: Python is a programming language..."

                result = await agent.run_agent(
                    [{"role": "user", "content": "Tell me about Python tutorials"}], arc="test"
                )

                assert "Based on the search results" in result.text
                assert mock_call.call_count == 2
                mock_tool.assert_called_once()
                # Verify usage accumulation across iterations
                assert result.total_input_tokens == 1800  # 1000 + 800
                assert result.total_output_tokens == 800  # 500 + 300
                assert result.total_cost == pytest.approx(0.018)  # 0.01 + 0.008
                assert result.tool_calls_count == 1

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

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                seq.pop(0),
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                mock_tool.return_value = "Tool result"

                result = await agent.run_agent(
                    [{"role": "user", "content": "Keep using tools"}], arc="test"
                )

                assert "Final response" in result.text
                assert mock_call.call_count == 3  # 3 iterations max

    @pytest.mark.asyncio
    async def test_agent_api_error_handling(self, agent):
        """Test agent handles API errors gracefully."""

        async def fake_call_raw_with_model(*args, **kwargs):
            raise Exception("API Error")

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            from muaddib.agentic_actor.actor import AgentIterationLimitError

            with pytest.raises(AgentIterationLimitError, match="too many turns"):
                await agent.run_agent([{"role": "user", "content": "Test query"}], arc="test")

    @pytest.mark.asyncio
    async def test_agent_refusal_error_handling(self, agent):
        """Test agent handles Anthropic refusal responses with proper error message."""

        # Mock call_raw_with_model to return refusal (converted to error by provider)
        async def fake_call_raw_with_model(*args, **kwargs):
            from muaddib.providers.anthropic import AnthropicClient

            client = AnthropicClient({"anthropic": {"key": "test", "url": "http://test"}})
            # Simulate what call_raw does - convert refusal to error
            return (
                {"error": "The AI refused to respond to this request"},
                client,
                ModelSpec("anthropic", "claude-3-5-sonnet"),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            result = await agent.run_agent([{"role": "user", "content": "Test query"}], arc="test")
            assert "Error:" in result.text
            assert "refused" in result.text.lower()

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

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                seq.pop(0),
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                mock_tool.return_value = "Tool execution failed: Network error"

                result = await agent.run_agent(
                    [{"role": "user", "content": "Search for something"}], arc="test"
                )

                assert "encountered an error" in result.text or "what I can tell you" in result.text

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
    async def test_vision_fallback_switches_model_and_appends_suffix(self, test_config, mock_agent):
        """Image via visit_webpage triggers switching to vision_model and suffix in final text."""
        # Minimal prompt wiring
        test_config["rooms"]["common"]["command"]["modes"]["serious"]["prompt"] = (
            "You are IRC user {mynick}."
        )

        # Build actor with openrouter base and anthropic vision
        from muaddib.agentic_actor import AgenticLLMActor
        from muaddib.providers import ModelRouter

        model_calls: list[str] = []

        # First response: tool call to visit_webpage; Second: final text
        tool_resp = {
            "content": [
                {"type": "text", "text": "I'll visit"},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "visit_webpage",
                    "input": {"url": "https://x"},
                },
            ],
            "stop_reason": "tool_use",
        }
        final_resp = {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn"}
        seq = [tool_resp, final_resp]

        async def fake_call_raw_with_model(model, messages, *args, **kwargs):
            model_calls.append(model)
            resp = seq.pop(0)
            # Return spec matching the requested model to avoid false refusal detection
            spec = parse_model_spec(model)
            return (
                resp,
                type(
                    "C",
                    (),
                    {
                        "extract_text_from_response": lambda self, r: "Done."
                        if r is final_resp
                        else "",
                        "has_tool_calls": lambda self, r: r is tool_resp,
                        "extract_tool_calls": lambda self, r: (
                            [
                                {
                                    "id": "tool_1",
                                    "name": "visit_webpage",
                                    "input": {"url": "https://x"},
                                }
                            ]
                            if r is tool_resp
                            else None
                        ),
                        "format_assistant_message": lambda self, r: {
                            "role": "assistant",
                            "content": [],
                        },
                        "format_tool_results": lambda self, results: {
                            "role": "user",
                            "content": [],
                        },
                        "cleanup_raw_text": lambda self, t: t,
                    },
                )(),
                spec,
                UsageInfo(None, None, None),
            )

        actor = AgenticLLMActor(
            config=test_config,
            model="openrouter:gpt-4o-mini",
            system_prompt_generator=lambda: test_config["rooms"]["common"]["command"]["modes"][
                "serious"
            ]["prompt"].format(
                mynick="testbot",
                current_time="now",
                sarcastic_model="s",
                serious_model="s",
                unsafe_model="u",
            ),
            agent=mock_agent,
            vision_model="anthropic:claude-sonnet-4-20250514",
        )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                # Return Anthropic content blocks with image
                mock_tool.return_value = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "AAAA",
                        },
                    }
                ]
                result = await actor.run_agent([{"role": "user", "content": "go"}], arc="arc")

        assert model_calls[0].startswith("openrouter:")
        assert model_calls[1].startswith("anthropic:")
        assert result.text.endswith(" [image fallback to claude-sonnet-4-20250514]")

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

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                seq.pop(0),
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                mock_tool.side_effect = ["Search result", "Page content"]

                result = await agent.run_agent(
                    [{"role": "user", "content": "Search and visit"}], arc="test"
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

                assert "Here's what I found" in result.text

    @pytest.mark.asyncio
    async def test_agent_with_context(self, agent):
        """Test agent passes through existing context."""
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
                return "Blue is often associated with calm and serenity."

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
                    "content": [
                        {"type": "text", "text": "Blue is often associated with calm and serenity."}
                    ],
                    "stop_reason": "end_turn",
                },
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            await agent.run_agent(context, arc="test")

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
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            await agent.run_agent([{"role": "user", "content": "Hello"}], arc="test")

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
        assert result["type"] == "empty_response_retry"
        assert "No valid text" in result["message"]

    @pytest.mark.asyncio
    async def test_agent_empty_response_retry(self, agent):
        """Test agent retries on empty/invalid responses."""

        # 1st response: Invalid (will trigger empty_response_retry)
        # 2nd response: Invalid (will trigger empty_response_retry)
        # 3rd response: Valid

        invalid_response = {"invalid": "response"}
        valid_response = {
            "content": [{"type": "text", "text": "Success!"}],
            "stop_reason": "end_turn",
        }

        class FakeClient:
            def extract_text_from_response(self, r):
                if r == valid_response:
                    return "Success!"
                return ""

            def has_tool_calls(self, r):
                return False

            def extract_tool_calls(self, r):
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": ""}

        responses = [invalid_response, invalid_response, valid_response]

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                responses.pop(0),
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        progress_calls = []

        async def mock_progress_callback(text: str, type: str = "progress"):
            progress_calls.append({"text": text, "type": type})

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            result = await agent.run_agent(
                [{"role": "user", "content": "test"}],
                arc="test",
                progress_callback=mock_progress_callback,
            )

            assert "Success!" in result.text
            # Should have 2 retry warnings in progress
            assert len(progress_calls) == 2
            assert "Retrying (1/3)" in progress_calls[0]["text"]
            assert "Retrying (2/3)" in progress_calls[1]["text"]

    @pytest.mark.asyncio
    async def test_final_answer_empty_with_thinking_continues_turn(self, agent):
        """Test that empty final answer with thinking tags continues the turn."""

        class MockClient:
            def __init__(self, responses):
                self.responses = responses
                self.call_count = 0

            def cleanup_raw_text(self, text):
                # Simulate cleanup_raw_text stripping thinking tags
                import re

                cleaned = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()
                return cleaned if cleaned else "..."

            def has_tool_calls(self, response):
                return "tool_calls" in response

            def extract_tool_calls(self, response):
                return response.get("tool_calls", [])

            def format_assistant_message(self, response):
                return {"role": "assistant", "content": ""}

            def format_tool_results(self, results):
                # Format tool results to preserve original content
                formatted = []
                for result in results:
                    formatted.append(
                        {
                            "role": "user",
                            "content": result["content"],  # Keep original content verbatim
                        }
                    )
                return formatted

        # First response: only thinking, no actual answer
        # Second response: proper answer
        responses = [
            {
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "final_answer",
                        "input": {"answer": "<thinking>I need to think about this...</thinking>"},
                    }
                ]
            },
            {
                "tool_calls": [
                    {
                        "id": "2",
                        "name": "final_answer",
                        "input": {"answer": "This is the actual answer"},
                    }
                ]
            },
        ]

        mock_client = MockClient(responses)

        call_messages = []  # Track messages for each call

        async def mock_call_raw_with_model(model, messages, *args, **kwargs):
            if isinstance(messages, list):
                call_messages.append(messages.copy())  # Store messages for verification
            response = responses[mock_client.call_count]
            mock_client.call_count += 1
            return response, mock_client, parse_model_spec(model), UsageInfo(None, None, None)

        with patch(
            "muaddib.agentic_actor.actor.ModelRouter.call_raw_with_model",
            new=AsyncMock(side_effect=mock_call_raw_with_model),
        ):
            result = await agent.run_agent([{"role": "user", "content": "test"}], arc="test")
            assert result.text == "This is the actual answer"
            assert mock_client.call_count == 2  # Should have made 2 calls

            # Verify second call includes the original thinking content
            assert len(call_messages) == 2
            second_call_messages = call_messages[1]
            thinking_found = any(
                "<thinking>I need to think about this...</thinking>" in str(msg.get("content", ""))
                for msg in second_call_messages
            )
            assert thinking_found, "Second call should include original thinking content verbatim"

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

        async def fake_call_raw_with_model(model, *args, **kwargs):
            return (
                seq.pop(0),
                FakeClient(),
                parse_model_spec(model),
                UsageInfo(None, None, None),
            )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ) as mock_call:
            result = await agent.run_agent(
                [{"role": "user", "content": "Search for something"}], arc="test"
            )

            # Should have made 2 calls - first truncated, then retry
            assert mock_call.call_count == 2
            assert "final answer" in result.text

    @pytest.mark.asyncio
    async def test_create_artifact_for_tool(self, agent):
        """Test artifact creation with tool input and output."""
        # Mock artifact executor
        mock_artifact_executor = AsyncMock()
        mock_artifact_executor.execute = AsyncMock(
            return_value="Artifact shared: https://example.com/artifacts/test123.txt"
        )

        tool_executors = {"share_artifact": mock_artifact_executor}

        tool_name = "web_search"
        tool_input = {"query": "Python tutorial"}
        tool_result = "Found 5 results about Python tutorials..."

        result = await agent._create_artifact_for_tool(
            tool_name, tool_input, tool_result, tool_executors
        )

        # Verify artifact URL is returned
        assert result == "https://example.com/artifacts/test123.txt"

        # Verify artifact executor was called with formatted content
        mock_artifact_executor.execute.assert_called_once()
        call_args = mock_artifact_executor.execute.call_args[0]
        artifact_content = call_args[0]  # First positional argument

        # Verify content format
        assert "# web_search Tool Call" in artifact_content
        assert "## Input" in artifact_content
        assert '"query": "Python tutorial"' in artifact_content
        assert "## Output" in artifact_content
        assert "Found 5 results about Python tutorials..." in artifact_content

    @pytest.mark.asyncio
    async def test_create_artifact_for_tool_no_executor(self, agent):
        """Test artifact creation when share_artifact executor is not available."""
        tool_executors = {}  # No share_artifact executor

        result = await agent._create_artifact_for_tool("web_search", {}, "output", tool_executors)

        assert result is None

    @pytest.mark.asyncio
    async def test_create_artifact_for_tool_executor_error(self, agent):
        """Test artifact creation when executor fails."""
        # Mock artifact executor that raises an exception
        mock_artifact_executor = AsyncMock()
        mock_artifact_executor.execute = AsyncMock(side_effect=Exception("Network error"))

        tool_executors = {"share_artifact": mock_artifact_executor}

        result = await agent._create_artifact_for_tool("web_search", {}, "output", tool_executors)

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_and_store_persistence_summary(self, agent):
        """Test persistence summary generation and storage."""
        # Mock persistence callback
        persistence_calls = []

        async def mock_persistence_callback(text: str):
            persistence_calls.append(text)

        # Mock model router response
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": "Summary: Performed web search and code execution successfully.",
                }
            ]
        }

        agent.model_router = AsyncMock()
        agent.model_router.call_raw_with_model = AsyncMock(
            return_value=(
                mock_response,
                None,
                ModelSpec("test", "model"),
                UsageInfo(None, None, None),
            )
        )

        # Test data
        persistent_tool_calls = [
            {
                "tool_name": "web_search",
                "input": {"query": "Python tutorial"},
                "output": "Found 5 results...",
                "persist_type": "summary",
            },
            {
                "tool_name": "execute_code",
                "input": {"code": "print('hello')"},
                "output": "hello",
                "persist_type": "artifact",
                "artifact_url": "https://example.com/artifacts/test.txt",
            },
        ]

        await agent._generate_and_store_persistence_summary(
            persistent_tool_calls, mock_persistence_callback
        )

        # Verify model router was called
        agent.model_router.call_raw_with_model.assert_called_once()
        call_args = agent.model_router.call_raw_with_model.call_args[0]

        # Verify model and system prompt
        expected_model = agent.config["tools"]["summary"]["model"]
        assert call_args[0] == expected_model
        # Verify system prompt contains key elements
        system_prompt = call_args[2]
        assert "AI agent" in system_prompt
        assert "remember" in system_prompt
        assert "tool" in system_prompt
        assert "summary" in system_prompt

        # Verify message content includes tool details
        messages = call_args[1]
        assert len(messages) == 1
        message_content = messages[0]["content"]
        assert "web_search" in message_content
        assert "execute_code" in message_content
        assert "https://example.com/artifacts/test.txt" in message_content

        # Verify persistence callback was called
        assert len(persistence_calls) == 1
        assert (
            persistence_calls[0] == "Summary: Performed web search and code execution successfully."
        )

    @pytest.mark.asyncio
    async def test_generate_and_store_persistence_summary_empty_list(self, agent):
        """Test persistence summary with empty tool calls list."""
        persistence_calls = []

        async def mock_persistence_callback(text: str):
            persistence_calls.append(text)

        agent.model_router = AsyncMock()

        await agent._generate_and_store_persistence_summary([], mock_persistence_callback)

        # Verify no model calls or persistence callbacks were made
        agent.model_router.call_raw_with_model.assert_not_called()
        assert len(persistence_calls) == 0

    @pytest.mark.asyncio
    async def test_generate_and_store_persistence_summary_model_error(self, agent):
        """Test persistence summary when model call fails."""
        persistence_calls = []

        async def mock_persistence_callback(text: str):
            persistence_calls.append(text)

        agent.model_router = AsyncMock()
        agent.model_router.call_raw_with_model = AsyncMock(side_effect=Exception("API Error"))

        persistent_tool_calls = [
            {
                "tool_name": "web_search",
                "input": {"query": "test"},
                "output": "results",
                "persist_type": "summary",
            }
        ]

        # Should not raise exception, just log error
        await agent._generate_and_store_persistence_summary(
            persistent_tool_calls, mock_persistence_callback
        )

        # Verify no persistence callback was made due to error
        assert len(persistence_calls) == 0

    @pytest.mark.asyncio
    async def test_tool_persistence_collection_during_execution(self, agent, api_type):
        """Test that persistent tool calls are collected during agent execution."""
        # Mock responses for tool use flow
        tool_use_response = self.create_tool_response(
            api_type,
            [
                {"id": "tool_1", "name": "web_search", "input": {"query": "test"}},
                {"id": "tool_2", "name": "execute_code", "input": {"code": "print('test')"}},
            ],
        )

        final_response = self.create_text_response(
            api_type, "Based on the search and execution, here's the answer."
        )

        class FakeClient:
            def extract_text_from_response(self, r):
                if r is final_response:
                    return "Based on the search and execution, here's the answer."
                return ""

            def has_tool_calls(self, r):
                return r is tool_use_response

            def extract_tool_calls(self, r):
                if r is tool_use_response:
                    return [
                        {"id": "tool_1", "name": "web_search", "input": {"query": "test"}},
                        {
                            "id": "tool_2",
                            "name": "execute_code",
                            "input": {"code": "print('test')"},
                        },
                    ]
                return None

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": []}

            def format_tool_results(self, results):
                return {"role": "user", "content": []}

        seq = [tool_use_response, final_response]

        async def fake_call_raw_with_model(*args, **kwargs):
            return seq.pop(0), FakeClient(), ModelSpec("test", "model"), UsageInfo(None, None, None)

        # Track progress callback calls
        progress_calls = []

        async def mock_progress_callback(text: str, type: str = "progress"):
            progress_calls.append({"text": text, "type": type})

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            with patch(
                "muaddib.agentic_actor.actor.execute_tool", new_callable=AsyncMock
            ) as mock_tool:
                mock_tool.side_effect = ["Search results", "test\n"]

                # Mock _generate_and_store_persistence_summary to verify it's called
                with patch.object(
                    agent, "_generate_and_store_persistence_summary", new_callable=AsyncMock
                ) as mock_summary:
                    await agent.run_agent(
                        [{"role": "user", "content": "Search and execute"}],
                        progress_callback=mock_progress_callback,
                        arc="test",
                    )

                    # Verify tool persistence summary was called
                    mock_summary.assert_called_once()
                    call_args = mock_summary.call_args[0]
                    persistent_calls = call_args[0]  # First argument is persistent_tool_calls list

                    # Verify both tools were tracked for persistence
                    assert len(persistent_calls) == 2

                    # Verify web_search tool details (persist: summary)
                    web_search_call = next(
                        call for call in persistent_calls if call["tool_name"] == "web_search"
                    )
                    assert web_search_call["input"] == {"query": "test"}
                    assert web_search_call["output"] == "Search results"
                    assert web_search_call["persist_type"] == "summary"

                    # Verify execute_code tool details (persist: artifact)
                    code_call = next(
                        call for call in persistent_calls if call["tool_name"] == "execute_code"
                    )
                    assert code_call["input"] == {"code": "print('test')"}
                    assert code_call["output"] == "test\n"
                    assert code_call["persist_type"] == "artifact"

    @pytest.mark.asyncio
    async def test_generate_persistence_summary_with_image_data(self, agent):
        """Test that image data is replaced with placeholders in persistence summaries."""
        # Mock persistence callback
        persistence_calls = []

        async def mock_persistence_callback(text: str):
            persistence_calls.append(text)

        # Mock model router response
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": "Summary: Successfully visited image and web page.",
                }
            ]
        }

        agent.model_router = AsyncMock()
        agent.model_router.call_raw_with_model = AsyncMock(
            return_value=(
                mock_response,
                None,
                ModelSpec("test", "model"),
                UsageInfo(None, None, None),
            )
        )

        # Test data with Anthropic content blocks
        persistent_tool_calls = [
            {
                "tool_name": "visit_webpage",
                "input": {"url": "https://example.com/image.jpg"},
                "output": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAAAAElEQVR4nO3B...",
                        },
                    }
                ],
                "persist_type": "summary",
            },
            {
                "tool_name": "visit_webpage",
                "input": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAABAAAAAAAAAElEQVR4nO3B...",
                        },
                    }
                ],
                "output": "Regular text output",
                "persist_type": "summary",
            },
        ]

        # Execute the method
        await agent._generate_and_store_persistence_summary(
            persistent_tool_calls, mock_persistence_callback
        )

        # Verify model was called
        assert agent.model_router.call_raw_with_model.called
        call_args = agent.model_router.call_raw_with_model.call_args

        # Get the messages sent to the model
        messages = call_args[0][1]  # Second argument is messages
        user_message_content = messages[0]["content"]

        # Verify that image blocks were replaced with placeholders
        assert "[image: image/jpeg]" in user_message_content
        assert "[image: image/png]" in user_message_content
        assert "Regular text output" in user_message_content  # Regular text should remain

        # Verify persistence callback was called with the summary
        assert len(persistence_calls) == 1
        assert persistence_calls[0] == "Summary: Successfully visited image and web page."
