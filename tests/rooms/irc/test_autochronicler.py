"""Tests for AutoChronicler functionality."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from muaddib.history import ChatHistory
from muaddib.rooms.autochronicler import AutoChronicler


class TestAutoChronicler:
    """Test the AutoChronicler class."""

    @pytest.fixture
    def mock_history(self):
        """Create a proper mock ChatHistory with all required methods."""
        # Create a mock that inherits from ChatHistory to ensure all methods exist
        history = AsyncMock(spec=ChatHistory)
        history.count_recent_unchronicled = AsyncMock(return_value=0)
        history.mark_chronicled = AsyncMock()
        return history

    @pytest.fixture
    def mock_monitor(self, mock_history):
        """Create a mock IRC monitor."""
        monitor = MagicMock()
        monitor.agent.model_router.call_raw_with_model = AsyncMock()
        monitor.agent.model_router.extract_text_from_response = Mock(return_value="Test summary")
        monitor.agent.chronicle.append_paragraph = AsyncMock()
        monitor.agent.chronicle.get_or_open_current_chapter = AsyncMock(return_value={"id": 123})
        monitor.agent.chronicle.read_chapter = AsyncMock(
            return_value=["Previous context", "More context"]
        )
        monitor.agent.chronicle.db_path = ":memory:"  # For chapter functions
        monitor.agent.config = {"chronicler": {"model": "test:model", "paragraphs_per_chapter": 10}}
        # Mock the history reference that autochronicler now uses
        monitor.agent.history = mock_history
        # Mock the chronicle's get_chapter_context_messages method
        monitor.agent.chronicle.get_chapter_context_messages = AsyncMock(
            return_value=[
                {"role": "user", "content": "<context_summary>Previous context</context_summary>"},
                {"role": "user", "content": "<context_summary>More context</context_summary>"},
            ]
        )
        return monitor

    @pytest.mark.asyncio
    async def test_no_chronicling_needed(self, mock_history, mock_monitor):
        """Test that no chronicling occurs when below threshold."""
        mock_history.count_recent_unchronicled.return_value = 5
        autochronicler = AutoChronicler(mock_history, mock_monitor)

        result = await autochronicler.check_and_chronicle(
            "user1", "testserver", "#testchannel", max_size=10
        )

        assert result is False
        mock_history.count_recent_unchronicled.assert_called_once_with(
            "testserver", "#testchannel", days=7
        )

    @pytest.mark.asyncio
    @patch("muaddib.rooms.autochronicler.chapter_append_paragraph")
    async def test_chronicling_triggered(
        self, mock_chapter_append, mock_history, mock_monitor, mock_model_call
    ):
        """Test that chronicling is triggered when above threshold."""
        mock_history.count_recent_unchronicled.return_value = 15
        mock_history.get_full_history.return_value = [
            {
                "id": 1,
                "nick": "user1",
                "message": "<user1> test message",
                "role": "user",
                "timestamp": "2025-01-01 10:00:00",
            },
            {
                "id": 2,
                "nick": "user2",
                "message": "<user2> another message",
                "role": "user",
                "timestamp": "2025-01-01 10:01:00",
            },
        ]

        # Set up the model mock for the chronicler
        mock_monitor.agent.model_router.call_raw_with_model = AsyncMock(
            side_effect=mock_model_call("Chronicled test messages")
        )

        # Mock chapter_append_paragraph to avoid database operations
        mock_chapter_append.return_value = AsyncMock()

        autochronicler = AutoChronicler(mock_history, mock_monitor)

        result = await autochronicler.check_and_chronicle(
            "user1", "testserver", "#testchannel", max_size=10
        )

        assert result is True
        mock_history.count_recent_unchronicled.assert_called_once_with(
            "testserver", "#testchannel", days=7
        )
        mock_history.get_full_history.assert_called_once_with(
            "testserver",
            "#testchannel",
            limit=20,  # 15 + MESSAGE_OVERLAP(5)
        )
        mock_history.mark_chronicled.assert_called_once_with([1, 2], 123)

    @pytest.mark.asyncio
    @patch("muaddib.rooms.autochronicler.chapter_append_paragraph")
    async def test_run_chronicler_creates_proper_context(
        self, mock_chapter_append, mock_history, mock_monitor, mock_model_call
    ):
        """Test that _run_chronicler creates proper context from messages."""
        messages = [
            {
                "id": 1,
                "nick": "user1",
                "message": "<user1> hello world",
                "role": "user",
                "timestamp": "2025-01-01 10:00:00",
            },
            {
                "id": 2,
                "nick": "user2",
                "message": "<user2> how are you?",
                "role": "user",
                "timestamp": "2025-01-01 10:01:00",
            },
        ]

        # Use the generalized mock pattern
        mock_call = AsyncMock(side_effect=mock_model_call("Test chronicle entry"))
        mock_monitor.agent.model_router.call_raw_with_model = mock_call

        # Mock chapter_append_paragraph to avoid database operations
        mock_chapter_append.return_value = AsyncMock()

        autochronicler = AutoChronicler(mock_history, mock_monitor)
        result = await autochronicler._run_chronicler("user1", "test#arc", messages)

        assert result == 123  # Expected chapter ID

        # Verify the model router was called with correct parameters
        mock_call.assert_called_once()
        call_args = mock_call.call_args

        # Check that the system prompt and context were passed correctly
        assert call_args.kwargs["system_prompt"] is not None
        assert "chronicle" in call_args.kwargs["system_prompt"].lower()
        assert call_args.kwargs["max_tokens"] == 1024

        # Check that there are three context messages: 2 chapter contexts + message content
        context_messages = call_args.kwargs["context"]
        assert len(context_messages) == 3

        # First two messages should be chapter context
        assert context_messages[0]["role"] == "user"
        assert (
            "<context_summary>Previous context</context_summary>" in context_messages[0]["content"]
        )
        assert context_messages[1]["role"] == "user"
        assert "<context_summary>More context</context_summary>" in context_messages[1]["content"]

        # Third message should be the actual messages to chronicle
        user_prompt = context_messages[2]["content"]
        assert "[2025-01-01 10:00" in user_prompt
        assert "<user1> hello world" in user_prompt
        assert "<user2> how are you?" in user_prompt
