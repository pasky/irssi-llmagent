"""Tests for ContextReducer functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muaddib.context_reducer import ContextReducer


class TestContextReducer:
    """Test the ContextReducer class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with model router."""
        agent = MagicMock()
        agent.model_router.call_raw_with_model = AsyncMock()
        return agent

    @pytest.fixture
    def reducer_config(self):
        """Config with context_reducer enabled at root level."""
        return {
            "model": "anthropic:claude-3-5-haiku-20241022",
            "prompt": "Condense this conversation. Output [USER]: and [ASSISTANT]: messages.",
        }

    @pytest.mark.asyncio
    async def test_no_config_returns_context_minus_triggering(self, mock_agent):
        """When context_reducer not configured, return context without triggering."""
        mock_agent.config = {}
        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "question"},
        ]

        result = await reducer.reduce(context, "system prompt")
        assert len(result) == 2
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_is_configured_property(self, mock_agent, reducer_config):
        """is_configured returns True only when both model and prompt are set."""
        mock_agent.config = {}
        reducer = ContextReducer(mock_agent)
        assert not reducer.is_configured

        mock_agent.config = {"context_reducer": {"model": "test:model"}}
        reducer = ContextReducer(mock_agent)
        assert not reducer.is_configured

        mock_agent.config = {"context_reducer": {"prompt": "test prompt"}}
        reducer = ContextReducer(mock_agent)
        assert not reducer.is_configured

        mock_agent.config = {"context_reducer": reducer_config}
        reducer = ContextReducer(mock_agent)
        assert reducer.is_configured

    @pytest.mark.asyncio
    async def test_short_context_returns_empty(self, mock_agent, reducer_config):
        """Context with only triggering message returns empty (caller appends triggering)."""
        mock_agent.config = {"context_reducer": reducer_config}
        reducer = ContextReducer(mock_agent)

        context = [{"role": "user", "content": "single message"}]
        result = await reducer.reduce(context, "system prompt")
        assert result == []

    @pytest.mark.asyncio
    async def test_reduces_context_preserving_structure(
        self, mock_agent, reducer_config, mock_model_call
    ):
        """Reducer condenses context while preserving user/assistant structure."""
        mock_agent.config = {"context_reducer": reducer_config}

        mock_agent.model_router.call_raw_with_model = AsyncMock(
            side_effect=mock_model_call(
                "[USER]: Summary of earlier discussion\n[ASSISTANT]: Key response"
            )
        )

        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "long discussion part 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "long discussion part 2"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "triggering message"},
        ]

        result = await reducer.reduce(context, "You are a helpful assistant")

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "Summary" in result[0]["content"]
        assert result[1]["role"] == "assistant"
        assert "Key response" in result[1]["content"]

    @pytest.mark.asyncio
    async def test_formats_context_with_system_prompt_and_triggering(
        self, mock_agent, reducer_config, mock_model_call
    ):
        """Reducer includes system prompt, history, and triggering message for relevance."""
        mock_agent.config = {"context_reducer": reducer_config}

        call_mock = AsyncMock(side_effect=mock_model_call("[USER]: condensed"))
        mock_agent.model_router.call_raw_with_model = call_mock

        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "triggering question"},
        ]

        await reducer.reduce(context, "You are TestBot on IRC")

        call_args = call_mock.call_args
        user_content = call_args[0][1][0]["content"]
        assert "AGENT SYSTEM PROMPT" in user_content
        assert "TestBot" in user_content
        assert "msg1" in user_content
        assert "triggering question" in user_content
        assert "TRIGGERING INPUT" in user_content

    @pytest.mark.asyncio
    async def test_fallback_on_parse_failure(self, mock_agent, reducer_config, mock_model_call):
        """When reducer output can't be parsed, treat as single user message."""
        mock_agent.config = {"context_reducer": reducer_config}

        mock_agent.model_router.call_raw_with_model = AsyncMock(
            side_effect=mock_model_call("Just some plain text summary without markers")
        )

        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "triggering"},
        ]

        result = await reducer.reduce(context, "system prompt")

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "plain text summary" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_error_returns_original_context(self, mock_agent, reducer_config):
        """On API error, return original context (minus triggering message)."""
        mock_agent.config = {"context_reducer": reducer_config}

        mock_agent.model_router.call_raw_with_model = AsyncMock(side_effect=Exception("API error"))

        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "triggering"},
        ]

        result = await reducer.reduce(context, "system prompt")

        assert len(result) == 2
        assert result[0]["content"] == "msg1"
        assert result[1]["content"] == "resp1"

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_original(self, mock_agent):
        """If prompt not configured, return original context."""
        mock_agent.config = {"context_reducer": {"model": "anthropic:claude-3-5-haiku-20241022"}}

        reducer = ContextReducer(mock_agent)

        context = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "triggering"},
        ]

        result = await reducer.reduce(context, "system prompt")

        assert len(result) == 1
        assert result[0]["content"] == "msg1"

    @pytest.mark.asyncio
    async def test_chapter_in_full_context_is_reduced(
        self, mock_agent, reducer_config, mock_model_call
    ):
        """Chapter context passed as part of full context gets reduced."""
        mock_agent.config = {"context_reducer": reducer_config}

        call_mock = AsyncMock(side_effect=mock_model_call("[USER]: condensed with chapter"))
        mock_agent.model_router.call_raw_with_model = call_mock

        reducer = ContextReducer(mock_agent)

        chapter_msg = {
            "role": "user",
            "content": "<context_summary>Earlier summary</context_summary>",
        }
        context = [
            chapter_msg,
            {"role": "user", "content": "recent msg"},
            {"role": "user", "content": "triggering"},
        ]

        result = await reducer.reduce(context, "system prompt")

        call_args = call_mock.call_args
        user_content = call_args[0][1][0]["content"]
        assert "Earlier summary" in user_content

        assert len(result) == 1
        assert "condensed with chapter" in result[0]["content"]
