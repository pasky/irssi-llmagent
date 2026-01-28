"""Tests for Anthropic-specific functionality."""

import base64
import json
from unittest.mock import patch

import pytest

from muaddib.providers.anthropic import AnthropicClient


class TestAnthropicSpecificBehavior:
    """Test Anthropic-specific behaviors like thinking budget handling."""

    @pytest.mark.asyncio
    async def test_claude_thinking_budget_with_non_auto_tool_choice(self):
        """Test Claude's special handling when thinking budget is set with non-auto tool_choice."""

        # Create a Claude client instance for direct testing
        from muaddib.providers.anthropic import AnthropicClient

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

            def raise_for_status(self):
                if not self.ok:
                    import aiohttp

                    raise aiohttp.ClientError(f"HTTP status {self.status}")

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

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                await self.close()

        # Test the scenario: thinking budget + non-auto tool_choice
        messages = [{"role": "user", "content": "Test query"}]

        with patch("aiohttp.ClientSession", MockSession):
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

        # Assert exact literal value of messages[] (with cache_control on last original message)
        expected_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Test query",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": "<meta>only tool ['final_answer'] may be called now</meta>",
            },
        ]
        assert captured_payload["messages"] == expected_messages

        # Verify tool_choice is not set in payload due to thinking budget incompatibility
        assert "tool_choice" not in captured_payload


class TestAnthropicRetryLogic:
    """Test Anthropic API retry logic for 529 errors."""

    @pytest.mark.asyncio
    async def test_529_retry_with_eventual_success(self):
        """Test that 529 errors are retried with exponential backoff until success."""
        config = {
            "providers": {
                "anthropic": {"key": "test-key", "url": "https://api.anthropic.com/v1/messages"}
            }
        }

        call_count = 0

        class MockResponse:
            def __init__(self, status, data):
                self.status = status
                self.ok = status == 200
                self.reason = "Overloaded" if status == 529 else "OK"
                self._data = data
                self.request_info = None
                self.history = ()

            async def text(self):
                return json.dumps(self._data)

            def raise_for_status(self):
                if not self.ok:
                    raise Exception(f"{self.status} {self.reason}")

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockContextManager:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *args):
                pass

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Fail first 2 attempts with 529, succeed on 3rd
            if call_count <= 2:
                response = MockResponse(
                    529,
                    {
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "Overloaded"},
                    },
                )
            else:
                response = MockResponse(200, {"content": [{"type": "text", "text": "Success!"}]})

            return MockContextManager(response)

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            def post(self, *args, **kwargs):
                return mock_post(*args, **kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        client = AnthropicClient(config)
        with patch("aiohttp.ClientSession", MockSession):
            with patch("asyncio.sleep") as mock_sleep:
                result = await client.call_raw(
                    context=[{"role": "user", "content": "test"}],
                    system_prompt="test",
                    model="claude-3-sonnet-20240229",
                )

                # Should have made 3 attempts total
                assert call_count == 3
                # Should have called sleep twice (for retries)
                assert mock_sleep.call_count == 2
                # Should have succeeded eventually
                assert "content" in result
                assert result["content"][0]["text"] == "Success!"

    @pytest.mark.asyncio
    async def test_529_retry_exhausted(self):
        """Test that after exhausting retries, 529 error is returned."""
        config = {
            "providers": {
                "anthropic": {"key": "test-key", "url": "https://api.anthropic.com/v1/messages"}
            }
        }

        call_count = 0

        class MockResponse:
            def __init__(self, status, data):
                self.status = status
                self.ok = status == 200
                self.reason = "Overloaded"
                self._data = data
                self.request_info = None
                self.history = ()

            async def text(self):
                return json.dumps(self._data)

            def raise_for_status(self):
                if not self.ok:
                    raise Exception(f"{self.status} {self.reason}")

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockContextManager:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *args):
                pass

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Always fail with 529
            response = MockResponse(
                529,
                {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
            )
            return MockContextManager(response)

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            def post(self, *args, **kwargs):
                return mock_post(*args, **kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        client = AnthropicClient(config)
        with patch("aiohttp.ClientSession", MockSession):
            with patch("asyncio.sleep") as mock_sleep:
                result = await client.call_raw(
                    context=[{"role": "user", "content": "test"}],
                    system_prompt="test",
                    model="claude-3-sonnet-20240229",
                )

                # Should have made 5 attempts total (all backoff_delays)
                assert call_count == 5
                # Should have called sleep 4 times (for retries)
                assert mock_sleep.call_count == 4
                # Should have failed with error
                assert "error" in result
                assert "529" in result["error"]


class TestAnthropicImageHandling:
    """Test Anthropic-specific image handling in tool results."""

    def test_claude_image_formatting(self):
        """Test Claude client formats image tool results correctly."""
        client = AnthropicClient(
            {"anthropic": {"key": "test-key", "model": "claude-3-sonnet-20240229"}}
        )

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

        assert result["role"] == "user"
        content = result["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "test-123"
        assert content["content"][0]["type"] == "image"
        assert content["content"][0]["source"]["type"] == "base64"
        assert content["content"][0]["source"]["media_type"] == "image/png"
        assert content["content"][0]["source"]["data"] == image_b64
