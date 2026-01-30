"""Tests for chat history functionality."""

import asyncio
import time

import pytest

from muaddib.history import ChatHistory
from muaddib.rooms.message import RoomMessage


def make_msg(
    server: str, channel: str, content: str, nick: str, mynick: str, **kwargs
) -> RoomMessage:
    """Helper to build RoomMessage for tests."""
    return RoomMessage(
        server_tag=server, channel_name=channel, content=content, nick=nick, mynick=mynick, **kwargs
    )


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
        await history.add_message(make_msg(server, channel, "Hello world", "user1", mynick))
        await history.add_message(make_msg(server, channel, "Hi there!", "testbot", mynick))
        await history.add_message(make_msg(server, channel, "How are you?", "user2", mynick))

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
            await history.add_message(make_msg(server, channel, f"Message {i}", f"user{i}", mynick))

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
        await history.add_message(
            make_msg(server, "#channel1", "Message in channel 1", "user1", mynick)
        )
        await history.add_message(
            make_msg(server, "#channel2", "Message in channel 2", "user2", mynick)
        )

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
            await history.add_message(make_msg(server, channel, f"Message {i}", f"user{i}", mynick))

        # Get limited context vs full history
        context = await history.get_context(server, channel)
        full_history = await history.get_full_history(server, channel)

        assert len(context) == 2  # Limited by inference_limit
        assert len(full_history) == 5  # All messages

    @pytest.mark.asyncio
    async def test_get_recent_messages_since(self, temp_db_path):
        """Test retrieving messages from a user since a timestamp for debouncing."""
        history = ChatHistory(temp_db_path)
        await history.initialize()

        server = "testserver"
        channel = "#testchan"
        mynick = "bot"

        # Add initial message and record timestamp
        await history.add_message(make_msg(server, channel, "initial command", "user1", mynick))
        original_time = time.time()

        # Small delay to ensure timestamp difference (needs to be >1s due to integer conversion)
        await asyncio.sleep(1.01)
        await history.add_message(make_msg(server, channel, "oops typo", "user1", mynick))
        await history.add_message(make_msg(server, channel, "one more thing", "user1", mynick))

        # Add message from different user (should not be included)
        await history.add_message(make_msg(server, channel, "unrelated", "user2", mynick))

        # Test the query
        followups = await history.get_recent_messages_since(server, channel, "user1", original_time)

        assert len(followups) == 2
        assert followups[0]["message"] == "oops typo"
        assert followups[1]["message"] == "one more thing"

        # Test with different user returns empty
        other_followups = await history.get_recent_messages_since(
            server, channel, "user2", original_time
        )
        assert len(other_followups) == 1
        assert other_followups[0]["message"] == "unrelated"

    @pytest.mark.asyncio
    async def test_thread_context_filters_channel_messages(self, temp_db_path):
        """Test that thread context includes channel history up to starter only."""
        history = ChatHistory(temp_db_path, inference_limit=10)
        await history.initialize()

        server = "discord:test"
        channel = "#general"
        mynick = "bot"

        await history.add_message(
            make_msg(server, channel, "before 1", "user1", mynick, platform_id="1")
        )
        await history.add_message(
            make_msg(server, channel, "before 2", "user2", mynick, platform_id="2")
        )
        await history.add_message(
            make_msg(server, channel, "starter", "user1", mynick, platform_id="100")
        )
        starter_id = await history.get_message_id_by_platform_id(server, channel, "100")
        assert starter_id is not None

        await history.add_message(
            make_msg(server, channel, "after", "user3", mynick, platform_id="3")
        )
        await history.add_message(
            make_msg(
                server, channel, "thread 1", "user1", mynick, thread_id="100", platform_id="101"
            )
        )
        await history.add_message(
            make_msg(
                server, channel, "thread 2", "user2", mynick, thread_id="100", platform_id="102"
            )
        )

        thread_context = await history.get_context(
            server,
            channel,
            limit=10,
            thread_id="100",
            thread_starter_id=starter_id,
        )
        thread_contents = [entry["content"] for entry in thread_context]
        assert any("before 1" in content for content in thread_contents)
        assert any("before 2" in content for content in thread_contents)
        assert any("starter" in content for content in thread_contents)
        assert any("thread 1" in content for content in thread_contents)
        assert any("thread 2" in content for content in thread_contents)
        assert not any("after" in content for content in thread_contents)

        channel_context = await history.get_context(server, channel, limit=10)
        channel_contents = [entry["content"] for entry in channel_context]
        assert any("after" in content for content in channel_contents)
        assert any("starter" in content for content in channel_contents)
        assert not any("thread 1" in content for content in channel_contents)

    @pytest.mark.asyncio
    async def test_add_message_with_custom_template(self, temp_db_path):
        """Test adding messages with custom role (for assistant_silent)."""
        history = ChatHistory(temp_db_path, inference_limit=5)
        await history.initialize()

        server = "irc.libera.chat"
        channel = "#test"
        mynick = "testbot"

        # Add regular messages
        await history.add_message(make_msg(server, channel, "Hello", "user1", mynick))
        await history.add_message(make_msg(server, channel, "Hi there!", mynick, mynick))

        # Add message with custom role (tool persistence)
        await history.add_message(
            make_msg(
                server,
                channel,
                "Tool summary: Performed web search and code execution",
                mynick,
                mynick,
            ),
            content_template="[internal monologue] {message}",
        )

        # Retrieve context
        context = await history.get_context(server, channel)
        assert len(context) == 3

        # Verify the custom role message content (should not be wrapped with <nick>)
        custom_role_msg = context[-1]
        assert custom_role_msg["role"] == "assistant"
        assert custom_role_msg["content"].endswith(
            "] [internal monologue] Tool summary: Performed web search and code execution"
        )

        # Verify regular messages still have proper formatting
        user_msg = context[0]
        assert "<user1>" in user_msg["content"]
        assistant_msg = context[1]
        assert "<testbot>" in assistant_msg["content"]

    @pytest.mark.asyncio
    async def test_mode_tracking_in_context(self, temp_db_path):
        """Test that mode is stored and prefixed in context for assistant messages."""
        history = ChatHistory(temp_db_path, inference_limit=10)
        await history.initialize()

        server = "irc.libera.chat"
        channel = "#test"
        mynick = "testbot"

        # Add user message (no mode)
        await history.add_message(make_msg(server, channel, "Tell me a joke", "user1", mynick))

        # Add assistant responses with different modes
        await history.add_message(
            make_msg(server, channel, "Why did the chicken...", mynick, mynick), mode="SARCASTIC"
        )
        await history.add_message(
            make_msg(server, channel, "Here's the answer", mynick, mynick), mode="EASY_SERIOUS"
        )
        await history.add_message(
            make_msg(server, channel, "Deep analysis", mynick, mynick), mode="THINKING_SERIOUS"
        )
        await history.add_message(
            make_msg(server, channel, "Unsafe response", mynick, mynick), mode="UNSAFE"
        )
        # Assistant message without mode (legacy)
        await history.add_message(make_msg(server, channel, "No mode set", mynick, mynick))

        context = await history.get_context(server, channel)
        assert len(context) == 6

        # User message should have no mode prefix
        assert context[0]["content"].startswith("[")
        assert not context[0]["content"].startswith("!")

        # SARCASTIC -> !d prefix
        assert context[1]["content"].startswith("!d [")
        assert "Why did the chicken" in context[1]["content"]

        # EASY_SERIOUS -> !s prefix
        assert context[2]["content"].startswith("!s [")

        # THINKING_SERIOUS -> !a prefix
        assert context[3]["content"].startswith("!a [")

        # UNSAFE -> !u prefix
        assert context[4]["content"].startswith("!u [")

        # No mode -> no prefix
        assert context[5]["content"].startswith("[")
        assert not context[5]["content"].startswith("!")

    @pytest.mark.asyncio
    async def test_mode_to_prefix_helper(self):
        """Test the _mode_to_prefix static method."""
        assert ChatHistory._mode_to_prefix(None) == ""
        assert ChatHistory._mode_to_prefix("SARCASTIC") == "!d "
        assert ChatHistory._mode_to_prefix("EASY_SERIOUS") == "!s "
        assert ChatHistory._mode_to_prefix("THINKING_SERIOUS") == "!a "
        assert ChatHistory._mode_to_prefix("UNSAFE") == "!u "
        assert ChatHistory._mode_to_prefix("UNKNOWN") == ""

    @pytest.mark.asyncio
    async def test_update_message_by_platform_id(self, temp_db_path):
        """Test updating message content by platform ID."""
        history = ChatHistory(temp_db_path, inference_limit=5)
        await history.initialize()

        server = "discord:test"
        channel = "#general"
        mynick = "bot"

        # Add message with platform_id
        await history.add_message(
            make_msg(server, channel, "original content", "user1", mynick, platform_id="123456")
        )

        # Update the message
        updated = await history.update_message_by_platform_id(
            server, channel, "123456", "edited content", "user1"
        )
        assert updated is True

        # Verify the update
        context = await history.get_context(server, channel)
        assert len(context) == 1
        assert "edited content" in context[0]["content"]
        assert "original content" not in context[0]["content"]

    @pytest.mark.asyncio
    async def test_update_message_nonexistent_platform_id(self, temp_db_path):
        """Test updating message with non-existent platform ID returns False."""
        history = ChatHistory(temp_db_path, inference_limit=5)
        await history.initialize()

        server = "discord:test"
        channel = "#general"

        # Try to update non-existent message
        updated = await history.update_message_by_platform_id(
            server, channel, "nonexistent", "new content", "user1"
        )
        assert updated is False
