"""Tests for proactive debouncer functionality."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from muaddib.rooms import ProactiveDebouncer
from muaddib.rooms.message import RoomMessage


async def noop_reply_sender(_: str) -> None:
    return None


noop_sender = noop_reply_sender


def channel_key(server: str, chan_name: str) -> str:
    return f"{server}#{chan_name}"


def make_msg(
    server: str, chan_name: str, nick: str, message: str, mynick: str = "bot"
) -> RoomMessage:
    return RoomMessage(
        server_tag=server,
        channel_name=chan_name,
        nick=nick,
        mynick=mynick,
        content=message,
    )


class TestProactiveDebouncer:
    """Test proactive debouncing behavior."""

    @pytest.fixture
    def debouncer(self):
        """Create a debouncer with short timeout for testing."""
        return ProactiveDebouncer(debounce_seconds=0.1)

    @pytest.fixture
    def callback_tracker(self):
        """Track callback invocations."""
        calls: list[dict] = []

        async def track_callback(
            msg: RoomMessage,
            reply_sender: Callable[[str], Awaitable[None]],
        ):
            calls.append(
                {
                    "server": msg.server_tag,
                    "chan_name": msg.channel_name,
                    "nick": msg.nick,
                    "message": msg.content,
                    "mynick": msg.mynick,
                    "reply_sender": reply_sender,
                    "thread_id": msg.thread_id,
                    "thread_starter_id": msg.thread_starter_id,
                    "secrets": msg.secrets,
                }
            )

        track_callback.calls = calls  # type: ignore[attr-defined]
        return track_callback

    @pytest.mark.asyncio
    async def test_single_message_processed(self, debouncer, callback_tracker):
        """Test that a single message gets processed after debounce."""
        msg = make_msg("freenode", "#test", "alice", "hello world")
        await debouncer.schedule_check(
            msg,
            channel_key("freenode", "#test"),
            noop_sender,
            callback_tracker,
        )

        # Should be pending
        assert debouncer.is_pending(channel_key("freenode", "#test"))
        assert channel_key("freenode", "#test") in debouncer.get_pending_channels()

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Should have been processed
        assert not debouncer.is_pending(channel_key("freenode", "#test"))
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "hello world"
        assert callback_tracker.calls[0]["chan_name"] == "#test"

    @pytest.mark.asyncio
    async def test_multiple_messages_only_last_processed(self, debouncer, callback_tracker):
        """Test that only the last message in a burst gets processed."""
        key = channel_key("freenode", "#test")
        # Send three messages quickly
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "alice", "first message"),
            key,
            noop_sender,
            callback_tracker,
        )
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "bob", "second message"),
            key,
            noop_sender,
            callback_tracker,
        )
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "charlie", "third message"),
            key,
            noop_sender,
            callback_tracker,
        )

        # Should still be pending
        assert debouncer.is_pending(key)

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Only the last message should have been processed
        assert not debouncer.is_pending(key)
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "third message"
        assert callback_tracker.calls[0]["nick"] == "charlie"

    @pytest.mark.asyncio
    async def test_different_channels_independent(self, debouncer, callback_tracker):
        """Test that different channels are processed independently."""
        key1 = channel_key("freenode", "#test1")
        key2 = channel_key("freenode", "#test2")
        # Send messages to different channels
        await debouncer.schedule_check(
            make_msg("freenode", "#test1", "alice", "message1"),
            key1,
            noop_sender,
            callback_tracker,
        )
        await debouncer.schedule_check(
            make_msg("freenode", "#test2", "bob", "message2"),
            key2,
            noop_sender,
            callback_tracker,
        )

        # Both should be pending
        assert debouncer.is_pending(key1)
        assert debouncer.is_pending(key2)
        assert len(debouncer.get_pending_channels()) == 2

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Both should have been processed
        assert not debouncer.is_pending(key1)
        assert not debouncer.is_pending(key2)
        assert len(callback_tracker.calls) == 2

        # Check both messages were processed
        messages = [call["message"] for call in callback_tracker.calls]
        assert "message1" in messages
        assert "message2" in messages

    @pytest.mark.asyncio
    async def test_message_during_debounce_resets_timer(self, debouncer, callback_tracker):
        """Test that a new message during debounce resets the timer."""
        key = channel_key("freenode", "#test")
        # Send first message
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "alice", "first"),
            key,
            noop_sender,
            callback_tracker,
        )

        # Wait halfway through debounce
        await asyncio.sleep(0.05)

        # Send second message (should reset timer)
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "bob", "second"),
            key,
            noop_sender,
            callback_tracker,
        )

        # Wait for original debounce time (should not have triggered yet)
        await asyncio.sleep(0.07)  # Total 0.12s from first message, 0.07s from second
        assert len(callback_tracker.calls) == 0

        # Wait for second debounce to complete
        await asyncio.sleep(0.05)  # Total 0.12s from second message

        # Only second message should be processed
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "second"

    @pytest.mark.asyncio
    async def test_cancel_all(self, debouncer, callback_tracker):
        """Test cancelling all pending debounced checks."""
        key1 = channel_key("freenode", "#test1")
        key2 = channel_key("freenode", "#test2")
        # Schedule multiple checks
        await debouncer.schedule_check(
            make_msg("freenode", "#test1", "alice", "message1"),
            key1,
            noop_sender,
            callback_tracker,
        )
        await debouncer.schedule_check(
            make_msg("freenode", "#test2", "bob", "message2"),
            key2,
            noop_sender,
            callback_tracker,
        )

        # Should be pending
        assert len(debouncer.get_pending_channels()) == 2

        # Cancel all
        await debouncer.cancel_all()

        # Should be cleared
        assert len(debouncer.get_pending_channels()) == 0
        assert not debouncer.is_pending(key1)
        assert not debouncer.is_pending(key2)

        # Wait to ensure no callbacks fire
        await asyncio.sleep(0.15)
        assert len(callback_tracker.calls) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self, debouncer):
        """Test that callback exceptions are handled gracefully."""

        async def failing_callback(
            msg: RoomMessage,
            reply_sender: Callable[[str], Awaitable[None]],
        ):
            raise ValueError("Test error")

        key = channel_key("freenode", "#test")
        # Should not raise exception
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "alice", "hello"),
            key,
            noop_sender,
            failing_callback,
        )

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Debouncer should still work after exception
        assert not debouncer.is_pending(key)

    @pytest.mark.asyncio
    async def test_zero_debounce_time(self, callback_tracker):
        """Test debouncer with zero debounce time."""
        debouncer = ProactiveDebouncer(debounce_seconds=0.0)

        await debouncer.schedule_check(
            make_msg("freenode", "#test", "alice", "instant"),
            channel_key("freenode", "#test"),
            noop_sender,
            callback_tracker,
        )

        # Should process immediately
        await asyncio.sleep(0.01)
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "instant"

    @pytest.mark.asyncio
    async def test_concurrent_same_channel(self, debouncer, callback_tracker):
        """Test concurrent scheduling for the same channel."""
        key = channel_key("freenode", "#test")
        # Schedule multiple messages concurrently
        tasks = []
        for i in range(5):
            task = debouncer.schedule_check(
                make_msg("freenode", "#test", f"user{i}", f"message{i}"),
                key,
                noop_sender,
                callback_tracker,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Only one message should be processed (the last one)
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "message4"
        assert callback_tracker.calls[0]["nick"] == "user4"

    @pytest.mark.asyncio
    async def test_cancel_channel_specific(self, debouncer, callback_tracker):
        """Test cancelling a specific channel's debounced check."""
        key1 = channel_key("freenode", "#test1")
        key2 = channel_key("freenode", "#test2")
        # Schedule checks for multiple channels
        await debouncer.schedule_check(
            make_msg("freenode", "#test1", "alice", "message1"),
            key1,
            noop_sender,
            callback_tracker,
        )
        await debouncer.schedule_check(
            make_msg("freenode", "#test2", "bob", "message2"),
            key2,
            noop_sender,
            callback_tracker,
        )

        # Both should be pending
        assert debouncer.is_pending(key1)
        assert debouncer.is_pending(key2)
        assert len(debouncer.get_pending_channels()) == 2

        # Cancel only one channel
        await debouncer.cancel_channel(key1)

        # Only test1 should be cancelled
        assert not debouncer.is_pending(key1)
        assert debouncer.is_pending(key2)
        assert len(debouncer.get_pending_channels()) == 1

        # Wait for remaining debounce
        await asyncio.sleep(0.15)

        # Only message2 should have been processed
        assert len(callback_tracker.calls) == 1
        assert callback_tracker.calls[0]["message"] == "message2"
        assert callback_tracker.calls[0]["chan_name"] == "#test2"

    @pytest.mark.asyncio
    async def test_cancel_channel_no_pending(self, debouncer):
        """Test cancelling a channel with no pending check."""
        # Should not raise exception
        await debouncer.cancel_channel(channel_key("freenode", "#nonexistent"))
        assert not debouncer.is_pending(channel_key("freenode", "#nonexistent"))

    @pytest.mark.asyncio
    async def test_cancel_channel_during_debounce(self, debouncer, callback_tracker):
        """Test cancelling a channel while its debounce is in progress."""
        key = channel_key("freenode", "#test")
        # Schedule a check
        await debouncer.schedule_check(
            make_msg("freenode", "#test", "alice", "message"),
            key,
            noop_sender,
            callback_tracker,
        )

        assert debouncer.is_pending(key)

        # Wait partway through debounce
        await asyncio.sleep(0.05)

        # Cancel the channel
        await debouncer.cancel_channel(key)

        # Should be cancelled immediately
        assert not debouncer.is_pending(key)

        # Wait for original debounce time to complete
        await asyncio.sleep(0.1)

        # No callback should have been invoked
        assert len(callback_tracker.calls) == 0
