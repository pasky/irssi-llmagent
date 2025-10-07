"""Tests for OpenAI-specific functionality."""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.providers.openai import OpenAIClient


class TestOpenAISpecificBehavior:
    """Test OpenAI-specific behaviors like reasoning effort handling."""

    @pytest.mark.asyncio
    async def test_openai_thinking_budget_with_non_auto_tool_choice(self):
        """Test OpenAI's handling when reasoning effort is set with non-auto tool_choice."""

        # Create an OpenAI client instance for direct testing
        from irssi_llmagent.providers.openai import OpenAIClient

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

        with patch("irssi_llmagent.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
            openai_client = OpenAIClient(test_config)
            await openai_client.call_raw(
                messages,
                "Test system prompt",
                "gpt-5",
                tools=[{"name": "final_answer", "description": "test tool", "input_schema": {}}],
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
        from irssi_llmagent.providers.openai import OpenAIClient

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

        with patch("irssi_llmagent.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
            client = OpenAIClient(config)
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

        with patch("irssi_llmagent.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
            client = OpenAIClient(config)
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


class TestOpenRouterClient:
    """Test OpenRouter client functionality."""

    @pytest.mark.asyncio
    async def test_openrouter_provider_routing_parsing(self):
        """Test OpenRouter provider routing syntax parsing."""
        from irssi_llmagent.providers.openai import OpenRouterClient

        test_config = {"providers": {"openrouter": {"key": "test-key"}}}
        client = OpenRouterClient(test_config)

        # Test without provider routing
        extra_body, model_name = client._get_extra_body("gpt-4o")
        assert extra_body is None
        assert model_name is None

        # Test with provider routing
        extra_body, model_name = client._get_extra_body("moonshot/kimi-k2#groq,moonshotai")
        assert extra_body == {"provider": {"only": ["groq", "moonshotai"]}}
        assert model_name == "moonshot/kimi-k2"

        # Test with single provider
        extra_body, model_name = client._get_extra_body("gpt-4o#anthropic")
        assert extra_body == {"provider": {"only": ["anthropic"]}}
        assert model_name == "gpt-4o"

        # Test with empty provider list
        extra_body, model_name = client._get_extra_body("gpt-4o#")
        assert extra_body is None
        assert model_name is None

    @pytest.mark.asyncio
    async def test_openrouter_call_raw_with_provider_routing(self):
        """Test OpenRouter call_raw method with provider routing."""
        from irssi_llmagent.providers.openai import OpenRouterClient

        test_config = {"providers": {"openrouter": {"key": "test-key"}}}
        client = OpenRouterClient(test_config)

        # Mock the OpenAI SDK client to capture the payload
        captured_kwargs = {}

        class MockResponse:
            def model_dump(self):
                return {"choices": [{"message": {"role": "assistant", "content": "Test response"}}]}

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MockResponse()

        async def capture_kwargs(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return MockResponse()

        mock_client.chat.completions.create.side_effect = capture_kwargs

        # Test with provider routing
        client._client = mock_client
        await client.call_raw(
            context=[],
            system_prompt="Test prompt",
            model="moonshot/kimi-k2#groq,moonshotai",
        )

        # Should have called with extra_body containing provider config
        assert "extra_body" in captured_kwargs
        assert captured_kwargs["extra_body"]["provider"]["only"] == ["groq", "moonshotai"]
        assert captured_kwargs["model"] == "moonshot/kimi-k2"


class TestOpenAIImageHandling:
    """Test OpenAI-specific image handling in tool results."""

    def test_openai_image_formatting(self):
        """Test OpenAI client formats image tool results correctly."""
        client = OpenAIClient({"openai": {"key": "test-key", "model": "gpt-4-vision-preview"}})

        # Mock image data (small PNG)
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        image_b64 = base64.b64encode(png_data).decode()

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "test-123",
                "content": f"IMAGE_DATA:image/png:{len(png_data)}:{image_b64}",
            }
        ]

        result = client.format_tool_results(tool_results)

        assert isinstance(result, list)
        assert len(result) == 2  # Tool result + image message

        # Check tool result
        tool_result = result[0]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "test-123"
        assert "Downloaded image (image/png" in tool_result["content"]

        # Check image message
        image_msg = result[1]
        assert image_msg["role"] == "user"
        assert len(image_msg["content"]) == 2  # Text + image
        assert image_msg["content"][0]["type"] == "text"
        assert image_msg["content"][1]["type"] == "image_url"
        assert f"data:image/png;base64,{image_b64}" in image_msg["content"][1]["image_url"]["url"]
