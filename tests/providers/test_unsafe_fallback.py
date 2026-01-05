"""Tests for refusal fallback model."""

from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.providers import ModelRouter


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

        # Mock client that returns refusal on first call, success on second
        mock_client = AsyncMock()

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

        mock_client.call_raw.side_effect = mock_call_raw

        with patch.object(router, "_ensure_client", return_value=mock_client):
            response, client, spec, _ = await router.call_raw_with_model(
                "anthropic:claude-sonnet-4",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return fallback response with model prefix
        assert response == {
            "content": [{"type": "text", "text": "[claude-sonnet-4-unsafe] Unsafe response"}]
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

        # Mock both clients
        safe_client = AsyncMock()
        unsafe_client = AsyncMock()

        # Safe client refuses
        safe_client.call_raw.return_value = {
            "error": "Invalid prompt: we've limited access to this content for safety reasons. (consider !u)"
        }

        # Unsafe client succeeds
        unsafe_client.call_raw.return_value = {
            "choices": [{"message": {"content": "Unsafe response"}}]
        }

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
            "choices": [{"message": {"content": "[gpt-4o-unsafe] Unsafe response"}}]
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

        # Mock both clients
        anthropic_client = AsyncMock()
        openrouter_client = AsyncMock()

        # Anthropic refuses
        anthropic_client.call_raw.return_value = {
            "error": "The AI refused to respond to this request (consider !u)"
        }

        # OpenRouter succeeds
        openrouter_client.call_raw.return_value = {
            "choices": [{"message": {"content": "Cross-provider unsafe response"}}]
        }

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
                {"message": {"content": "[some-unsafe-model] Cross-provider unsafe response"}}
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

        # Mock client
        client = AsyncMock()
        client.call_raw.return_value = {"error": "API error: connection timeout"}

        with patch.object(router, "_ensure_client", return_value=client):
            response, returned_client, spec, _ = await router.call_raw_with_model(
                "anthropic:claude-sonnet-4",
                [{"role": "user", "content": "test"}],
                "system prompt",
            )

        # Should return original error without retrying (no "consider !u" marker)
        assert response == {"error": "API error: connection timeout"}
        assert client.call_raw.call_count == 1  # Only called once
