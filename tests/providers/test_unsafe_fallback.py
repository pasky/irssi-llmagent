"""Tests for refusal fallback model."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from muaddib.agentic_actor import AgenticLLMActor
from muaddib.providers import ModelRouter, ModelSpec, UsageInfo


class TestRefusalFallback:
    """Test automatic fallback on content safety refusals."""

    @pytest.mark.asyncio
    async def test_anthropic_refusal_with_fallback(self):
        """Test Anthropic refusal triggers fallback model."""
        config = {
            "router": {"refusal_fallback_model": "anthropic:claude-sonnet-4-unsafe"},
            "providers": {"anthropic": {"key": "test-key", "url": "https://api.anthropic.com"}},
        }

        router = ModelRouter(config)

        # Track call count to return different responses
        call_count = [0]

        async def mock_call_raw(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (safe model) refuses
                return {"error": "The AI refused to respond to this request (consider !u)"}
            else:
                # Second call (unsafe model) succeeds
                return {"content": [{"type": "text", "text": "Unsafe response"}]}

        # Use Mock for client, AsyncMock only for call_raw
        mock_client = Mock()
        mock_client.call_raw = AsyncMock(side_effect=mock_call_raw)
        mock_client.extract_usage.return_value = UsageInfo(10, 20, 0.001)

        with patch.object(router, "_ensure_client", return_value=mock_client):
            response, client, spec, _ = await router.call_raw_with_model(
                "anthropic:claude-sonnet-4",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return fallback response with model prefix
        assert response == {
            "content": [
                {
                    "type": "text",
                    "text": "Unsafe response [refusal fallback to claude-sonnet-4-unsafe]",
                }
            ]
        }
        assert spec.provider == "anthropic"
        assert spec.name == "claude-sonnet-4-unsafe"
        assert call_count[0] == 2  # Both calls should have been made

    @pytest.mark.asyncio
    async def test_openai_refusal_with_fallback(self):
        """Test OpenAI refusal triggers fallback model."""
        config = {
            "router": {"refusal_fallback_model": "openai:gpt-4o-unsafe"},
            "providers": {"openai": {"key": "test-key"}},
        }

        router = ModelRouter(config)

        # Use Mock for clients, AsyncMock only for call_raw
        safe_client = Mock()
        safe_client.call_raw = AsyncMock(
            return_value={
                "error": "Invalid prompt: we've limited access to this content for safety reasons. (consider !u)"
            }
        )

        unsafe_client = Mock()
        unsafe_client.call_raw = AsyncMock(
            return_value={"choices": [{"message": {"content": "Unsafe response"}}]}
        )
        unsafe_client.extract_usage.return_value = UsageInfo(10, 20, 0.001)

        # Mock client creation
        call_count = [0]

        def get_client(provider):
            if provider == "openai":
                call_count[0] += 1
                if call_count[0] == 1:
                    return safe_client
                return unsafe_client
            return safe_client

        with patch.object(router, "_ensure_client", side_effect=get_client):
            response, client, spec, _ = await router.call_raw_with_model(
                "openai:gpt-4o",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return fallback response with model prefix
        assert response == {
            "choices": [
                {"message": {"content": "Unsafe response [refusal fallback to gpt-4o-unsafe]"}}
            ]
        }
        assert spec.provider == "openai"
        assert spec.name == "gpt-4o-unsafe"

    @pytest.mark.asyncio
    async def test_cross_provider_fallback(self):
        """Test fallback to a different provider."""
        config = {
            "router": {"refusal_fallback_model": "openrouter:some-unsafe-model"},
            "providers": {
                "anthropic": {"key": "test-key", "url": "https://api.anthropic.com"},
                "openrouter": {"key": "test-key"},
            },
        }

        router = ModelRouter(config)

        # Use Mock for clients, AsyncMock only for call_raw
        anthropic_client = Mock()
        anthropic_client.call_raw = AsyncMock(
            return_value={"error": "The AI refused to respond to this request (consider !u)"}
        )

        openrouter_client = Mock()
        openrouter_client.call_raw = AsyncMock(
            return_value={"choices": [{"message": {"content": "Cross-provider unsafe response"}}]}
        )
        openrouter_client.extract_usage.return_value = UsageInfo(10, 20, 0.001)

        # Mock client creation
        def get_client(provider):
            if provider == "anthropic":
                return anthropic_client
            return openrouter_client

        with patch.object(router, "_ensure_client", side_effect=get_client):
            response, client, spec, _ = await router.call_raw_with_model(
                "anthropic:claude-sonnet-4",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return response from OpenRouter with model prefix
        assert response == {
            "choices": [
                {
                    "message": {
                        "content": "Cross-provider unsafe response [refusal fallback to some-unsafe-model]"
                    }
                }
            ]
        }
        assert spec.provider == "openrouter"
        assert spec.name == "some-unsafe-model"

    @pytest.mark.asyncio
    async def test_non_refusal_errors_not_retried(self):
        """Test that non-refusal errors don't trigger fallback."""
        config = {
            "router": {"refusal_fallback_model": "anthropic:claude-sonnet-4-unsafe"},
            "providers": {"anthropic": {"key": "test-key", "url": "https://api.anthropic.com"}},
        }

        router = ModelRouter(config)

        # Use Mock for client, AsyncMock only for call_raw
        client = Mock()
        client.call_raw = AsyncMock(return_value={"error": "API error: connection timeout"})
        client.extract_usage.return_value = UsageInfo(10, 20, 0.001)

        with patch.object(router, "_ensure_client", return_value=client):
            response, returned_client, spec, _ = await router.call_raw_with_model(
                "anthropic:claude-sonnet-4",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return original error without retrying (no "consider !u" marker)
        assert response == {"error": "API error: connection timeout"}
        assert client.call_raw.call_count == 1  # Only called once

    @pytest.mark.asyncio
    async def test_agentic_loop_sticks_to_fallback_model(self, test_config, mock_agent):
        """Test that after refusal fallback, all subsequent iterations use fallback model."""
        test_config["router"] = {"refusal_fallback_model": "deepseek:deepseek-reasoner"}

        models_called = []

        def create_tool_response():
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "final_answer",
                        "input": {"answer": "Done"},
                    }
                ],
                "stop_reason": "tool_use",
            }

        async def fake_call_raw_with_model(model, messages, *args, **kwargs):
            models_called.append(model)

            # First call: refusal (returns different spec to simulate fallback)
            if len(models_called) == 1:
                return (
                    create_tool_response(),
                    FakeClient(),
                    ModelSpec("deepseek", "deepseek-reasoner"),  # Fallback was used
                    UsageInfo(10, 20, 0.001),
                )
            # Second call: should now be using fallback model
            return (
                {"content": [{"type": "text", "text": "Final answer"}], "stop_reason": "end_turn"},
                FakeClient(),
                ModelSpec("deepseek", "deepseek-reasoner"),
                UsageInfo(10, 20, 0.001),
            )

        class FakeClient:
            def extract_text_from_response(self, r):
                for block in r.get("content", []):
                    if block.get("type") == "text":
                        return block.get("text", "")
                return ""

            def has_tool_calls(self, r):
                return any(b.get("type") == "tool_use" for b in r.get("content", []))

            def extract_tool_calls(self, r):
                return [b for b in r.get("content", []) if b.get("type") == "tool_use"]

            def format_assistant_message(self, r):
                return {"role": "assistant", "content": r.get("content", [])}

            def format_tool_results(self, results):
                return {"role": "user", "content": results}

        agent = AgenticLLMActor(
            config=test_config,
            model="anthropic:claude-sonnet-4",
            system_prompt_generator=lambda: "Test prompt",
            prompt_reminder_generator=lambda: None,
            agent=mock_agent,
        )

        with patch.object(
            ModelRouter, "call_raw_with_model", new=AsyncMock(side_effect=fake_call_raw_with_model)
        ):
            result = await agent.run_agent([{"role": "user", "content": "test"}], arc="test")

        # First call was with original model, second should be with fallback
        assert len(models_called) == 2
        assert models_called[0] == "anthropic:claude-sonnet-4"
        assert models_called[1] == "deepseek:deepseek-reasoner"
        # Final result should have the fallback suffix
        assert result.text.endswith("[refusal fallback to deepseek-reasoner]")
