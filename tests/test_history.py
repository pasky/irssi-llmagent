"""Tests for chat history functionality."""

import pytest

from irssi_llmagent.history import ChatHistory


class TestChatHistory:
    """Test chat history persistence and retrieval."""

    @pytest.mark.asyncio
    async def test_initialize_database(self, temp_db_path):
        """Test database initialization."""
        history = ChatHistory(temp_db_path, inference_limit=3)
        await history.initialize()
        # Should not raise any exceptions

    @pytest.mark.asyncio
    async def test_add_and_retrieve_messages(self, temp_db_path):
        """Test adding messages and retrieving context."""
        history = ChatHistory(temp_db_path, inference_limit=3)
        await history.initialize()

        server = "irc.libera.chat"
        channel = "#test"
        mynick = "testbot"

        # Add some messages
        await history.add_message(server, channel, "Hello world", "user1", mynick)
        await history.add_message(server, channel, "Hi there!", "testbot", mynick)
        await history.add_message(server, channel, "How are you?", "user2", mynick)

        # Retrieve context
        context = await history.get_context(server, channel)

        assert len(context) == 3
        assert context[0]["content"].endswith("<user1> Hello world")
        assert context[0]["role"] == "user"
        assert context[1]["content"].endswith("<testbot> Hi there!")
        assert context[1]["role"] == "assistant"
        assert context[2]["content"].endswith("<user2> How are you?")
        assert context[2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_inference_limit(self, temp_db_path):
        """Test that context respects inference limit."""
        history = ChatHistory(temp_db_path, inference_limit=2)
        await history.initialize()

        server = "irc.libera.chat"
        channel = "#test"
        mynick = "testbot"

        # Add more messages than the limit
        for i in range(5):
            await history.add_message(server, channel, f"Message {i}", f"user{i}", mynick)

        context = await history.get_context(server, channel)

        # Should only return the last 2 messages
        assert len(context) == 2
        assert "Message 3" in context[0]["content"]
        assert "Message 4" in context[1]["content"]

    @pytest.mark.asyncio
    async def test_separate_channels(self, temp_db_path):
        """Test that different channels maintain separate histories."""
        history = ChatHistory(temp_db_path, inference_limit=5)
        await history.initialize()

        server = "irc.libera.chat"
        mynick = "testbot"

        # Add messages to different channels
        await history.add_message(server, "#channel1", "Message in channel 1", "user1", mynick)
        await history.add_message(server, "#channel2", "Message in channel 2", "user2", mynick)

        context1 = await history.get_context(server, "#channel1")
        context2 = await history.get_context(server, "#channel2")

        assert len(context1) == 1
        assert len(context2) == 1
        assert "channel 1" in context1[0]["content"]
        assert "channel 2" in context2[0]["content"]

    @pytest.mark.asyncio
    async def test_full_history_retrieval(self, temp_db_path):
        """Test retrieving full history without limits."""
        history = ChatHistory(temp_db_path, inference_limit=2)
        await history.initialize()

        server = "irc.libera.chat"
        channel = "#test"
        mynick = "testbot"

        # Add several messages
        for i in range(5):
            await history.add_message(server, channel, f"Message {i}", f"user{i}", mynick)

        # Get limited context vs full history
        context = await history.get_context(server, channel)
        full_history = await history.get_full_history(server, channel)

        assert len(context) == 2  # Limited by inference_limit
        assert len(full_history) == 5  # All messages
