"""Tests for OpenAI-specific functionality."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from muaddib.providers.openai import OpenAIClient


class TestOpenAISpecificBehavior:
    """Test OpenAI-specific behaviors like reasoning effort handling."""

    @pytest.mark.asyncio
    async def test_openai_thinking_budget_with_non_auto_tool_choice(self):
        """Test OpenAI's handling when reasoning effort is set with non-auto tool_choice."""

        # Create an OpenAI client instance for direct testing
        from muaddib.providers.openai import OpenAIClient

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

        with patch("muaddib.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
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
        from muaddib.providers.openai import OpenAIClient

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

        with patch("muaddib.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
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

        with patch("muaddib.providers.openai._AsyncOpenAI", MockAsyncOpenAI):
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
    async def test_openrouter_reasoning_details_preservation(self):
        """Test that reasoning_details are preserved in assistant messages."""
        from muaddib.providers.openai import OpenRouterClient

        test_config = {"providers": {"openrouter": {"key": "test-key"}}}
        client = OpenRouterClient(test_config)

        # Mock response with reasoning_details
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me solve this problem",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search_web",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ],
                        "reasoning_details": [
                            {
                                "id": "rd_1",
                                "format": "anthropic-claude-v1",
                                "type": "reasoning.text",
                                "text": "I need to search for information...",
                                "index": 0,
                            }
                        ],
                    }
                }
            ]
        }

        # Test that reasoning_details are extracted and preserved
        formatted_msg = client.format_assistant_message(mock_response)

        assert formatted_msg["role"] == "assistant"
        assert formatted_msg["content"] == "Let me solve this problem"
        assert "reasoning_details" in formatted_msg
        assert len(formatted_msg["reasoning_details"]) == 1
        assert formatted_msg["reasoning_details"][0]["type"] == "reasoning.text"

        # Test that reasoning_details are passed back in context
        captured_kwargs = {}

        class MockResponse:
            def model_dump(self):
                return {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = MockResponse()

        async def capture_kwargs(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return MockResponse()

        mock_client.chat.completions.create.side_effect = capture_kwargs

        # Send context with reasoning_details followed by tool result
        # This simulates the continuation of a reasoning flow after tool execution
        client._client = mock_client
        await client.call_raw(
            context=[
                {"role": "user", "content": "Search for test"},
                formatted_msg,
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Search results: Python is great",
                },
            ],
            system_prompt="Test",
            model="anthropic/claude-sonnet-4",
        )

        # Verify reasoning_details were included in the messages
        messages = captured_kwargs.get("messages", [])
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_messages) > 0, f"Expected assistant messages in: {messages}"
        assert "reasoning_details" in assistant_messages[0]
        assert assistant_messages[0]["reasoning_details"] == formatted_msg["reasoning_details"]

    @pytest.mark.asyncio
    async def test_openrouter_provider_routing_parsing(self):
        """Test OpenRouter provider routing syntax parsing."""
        from muaddib.providers.openai import OpenRouterClient

        test_config = {"providers": {"openrouter": {"key": "test-key"}}}
        client = OpenRouterClient(test_config)

        # Test without provider routing (still includes usage tracking)
        extra_body, model_name = client._get_extra_body("gpt-4o")
        assert extra_body == {"usage": {"include": True}}
        assert model_name is None

        # Test with provider routing
        extra_body, model_name = client._get_extra_body("moonshot/kimi-k2#groq,moonshotai")
        assert extra_body == {
            "provider": {"only": ["groq", "moonshotai"]},
            "usage": {"include": True},
        }
        assert model_name == "moonshot/kimi-k2"

        # Test with single provider
        extra_body, model_name = client._get_extra_body("gpt-4o#anthropic")
        assert extra_body == {"provider": {"only": ["anthropic"]}, "usage": {"include": True}}
        assert model_name == "gpt-4o"

        # Test with empty provider list (still includes usage tracking)
        extra_body, model_name = client._get_extra_body("gpt-4o#")
        assert extra_body == {"usage": {"include": True}}
        assert model_name is None

    @pytest.mark.asyncio
    async def test_openrouter_call_raw_with_provider_routing(self):
        """Test OpenRouter call_raw method with provider routing."""
        from muaddib.providers.openai import OpenRouterClient

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

        # Use Anthropic content blocks format
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "test-123",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    }
                ],
            }
        ]

        result = client.format_tool_results(tool_results)

        assert isinstance(result, list)
        assert len(result) == 2  # Tool result + image message

        # Check tool result
        tool_result = result[0]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "test-123"
        assert "Images returned by tool" in tool_result["content"]

        # Check image message
        image_msg = result[1]
        assert image_msg["role"] == "user"
        assert len(image_msg["content"]) == 2  # Text + image
        assert image_msg["content"][0]["type"] == "text"
        assert image_msg["content"][1]["type"] == "image_url"
        assert f"data:image/png;base64,{image_b64}" in image_msg["content"][1]["image_url"]["url"]

    def test_openai_plain_text_formatting(self):
        """Test OpenAI client formats plain text tool results correctly."""
        client = OpenAIClient({"openai": {"key": "test-key", "model": "gpt-4"}})

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "test-123",
                "content": "This is plain text result from a tool",
            }
        ]

        result = client.format_tool_results(tool_results)

        assert isinstance(result, list)
        assert len(result) == 1  # Only tool message, no image message

        # Check tool result
        tool_result = result[0]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "test-123"
        assert tool_result["content"] == "This is plain text result from a tool"


class TestOpenAIContentSafetyRefusal:
    """Test OpenAI content safety error handling."""

    @pytest.mark.asyncio
    async def test_openai_content_safety_refusal(self):
        """Test that OpenAI content safety errors are converted to refusal format."""
        from openai import BadRequestError

        from muaddib.providers.openai import OpenAIClient

        test_config = {"providers": {"openai": {"key": "test-key"}}}
        client = OpenAIClient(test_config)

        # Create a mock BadRequestError with the actual safety error structure
        mock_response = httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Invalid prompt: we've limited access to this content for safety reasons.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_prompt",
                }
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        # Note: OpenAI SDK does body.get("error", body) so the exception body is the inner dict
        error = BadRequestError(
            "Error code: 400",
            response=mock_response,
            body={
                "message": "Invalid prompt: we've limited access to this content for safety reasons.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_prompt",
            },
        )

        # Mock the OpenAI client to raise the error
        with patch.object(client._client.chat.completions, "create", side_effect=error):
            response = await client.call_raw(
                context=[{"role": "user", "content": "test message"}],
                system_prompt="Test",
                model="gpt-4",
            )

        # Should return refusal error with the actual message from OpenAI
        assert "error" in response
        assert (
            "Invalid prompt: we've limited access to this content for safety reasons."
            in response["error"]
        )
        assert "(consider !u)" in response["error"]

    @pytest.mark.asyncio
    async def test_openai_other_errors_not_converted(self):
        """Test that non-safety OpenAI errors are not converted to refusal format."""
        from openai import BadRequestError

        from muaddib.providers.openai import OpenAIClient

        test_config = {"providers": {"openai": {"key": "test-key"}}}
        client = OpenAIClient(test_config)

        # Create a mock BadRequestError with a different error code
        mock_response = httpx.Response(
            status_code=400,
            json={
                "error": {
                    "message": "Invalid request: missing required parameter",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "missing_parameter",
                }
            },
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        # Note: OpenAI SDK does body.get("error", body) so the exception body is the inner dict
        error = BadRequestError(
            "Error code: 400",
            response=mock_response,
            body={
                "message": "Invalid request: missing required parameter",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "missing_parameter",
            },
        )

        # Mock the OpenAI client to raise the error
        with patch.object(client._client.chat.completions, "create", side_effect=error):
            response = await client.call_raw(
                context=[{"role": "user", "content": "test message"}],
                system_prompt="Test",
                model="gpt-4",
            )

        # Should return generic API error, not refusal
        assert "error" in response
        assert "API error:" in response["error"]
        assert "The AI refused to respond" not in response["error"]


class TestOpenAIRetry:
    """Test OpenAI client retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_500_error_in_response(self):
        """Test retry when 200 OK response contains 500 error code (OpenRouter style)."""
        from muaddib.providers.openai import OpenAIClient

        test_config = {"providers": {"openai": {"key": "test-key"}}}
        client = OpenAIClient(test_config)

        # Mock response with error
        error_response = {
            "id": None,
            "choices": None,
            "error": {"message": "Internal Server Error", "code": 500},
        }

        # Mock response with success
        success_response = {
            "id": "123",
            "choices": [{"message": {"role": "assistant", "content": "Success"}}],
        }

        class MockResponse:
            def __init__(self, data):
                self.data = data

            def model_dump(self):
                return self.data

        # Mock the OpenAI client
        mock_completions = AsyncMock()
        # Side effect: return error twice, then success
        mock_completions.create.side_effect = [
            MockResponse(error_response),
            MockResponse(error_response),
            MockResponse(success_response),
        ]

        # Replace _client with MagicMock to avoid property setter issues
        client._client = MagicMock()
        client._client.chat.completions = mock_completions

        # We need to mock asyncio.sleep to avoid waiting
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            response = await client.call_raw(
                context=[{"role": "user", "content": "test"}], system_prompt="test", model="gpt-4"
            )

            # Check that it retried
            assert mock_completions.create.call_count == 3
            assert mock_sleep.call_count == 2

            # Check result
            assert response["choices"][0]["message"]["content"] == "Success"

    @pytest.mark.asyncio
    async def test_retry_on_500_exception(self):
        """Test retry when API raises 500 status exception."""
        from muaddib.providers.openai import OpenAIClient

        test_config = {"providers": {"openai": {"key": "test-key"}}}
        client = OpenAIClient(test_config)

        # Mocking a generic exception with status_code attribute for simplicity/robustness
        class Mock500Error(Exception):
            status_code = 500

        success_response = {
            "id": "123",
            "choices": [{"message": {"role": "assistant", "content": "Success"}}],
        }

        class MockResponse:
            def __init__(self, data):
                self.data = data

            def model_dump(self):
                return self.data

        mock_completions = AsyncMock()
        mock_completions.create.side_effect = [
            Mock500Error("Server Error"),
            MockResponse(success_response),
        ]

        # Replace _client with MagicMock to avoid property setter issues
        client._client = MagicMock()
        client._client.chat.completions = mock_completions

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            response = await client.call_raw(
                context=[{"role": "user", "content": "test"}], system_prompt="test", model="gpt-4"
            )

            assert mock_completions.create.call_count == 2
            assert mock_sleep.call_count == 1
            assert response["choices"][0]["message"]["content"] == "Success"
