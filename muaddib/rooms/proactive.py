"""Debounce proactive interjections per channel."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..message_logging import MessageLoggingContext
from ..paths import get_logs_dir

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """Represents a message pending debounced processing."""

    server: str
    chan_name: str
    channel_key: str
    nick: str
    message: str
    mynick: str
    reply_sender: Callable[[str], Awaitable[None]]
    timestamp: float
    thread_id: str | None = None
    thread_starter_id: int | None = None
    secrets: dict[str, Any] | None = None


class ProactiveDebouncer:
    """Debounces proactive interjections per channel.

    When multiple messages arrive in quick succession on the same channel,
    only the last message will be checked for proactive interjection after
    the debounce period expires.
    """

    def __init__(self, debounce_seconds: float = 15.0):
        """Initialize the debouncer.

        Args:
            debounce_seconds: Time to wait before processing the latest message
        """
        self.debounce_seconds = debounce_seconds
        self._pending_timers: dict[str, asyncio.Task] = {}
        self._pending_messages: dict[str, PendingMessage] = {}
        self._channel_locks: dict[str, asyncio.Lock] = {}

    def _get_channel_lock(self, channel_key: str) -> asyncio.Lock:
        """Get or create a lock for the specific channel."""
        if channel_key not in self._channel_locks:
            self._channel_locks[channel_key] = asyncio.Lock()
        return self._channel_locks[channel_key]

    async def schedule_check(
        self,
        server: str,
        chan_name: str,
        channel_key: str,
        nick: str,
        message: str,
        mynick: str,
        reply_sender: Callable[[str], Awaitable[None]],
        check_callback: Callable[
            [
                str,
                str,
                str,
                str,
                str,
                Callable[[str], Awaitable[None]],
                str | None,
                int | None,
                dict[str, Any] | None,
            ],
            Awaitable[None],
        ],
        thread_id: str | None = None,
        thread_starter_id: int | None = None,
        secrets: dict[str, Any] | None = None,
    ) -> None:
        """Schedule a debounced proactive check for this channel.

        Args:
            server: IRC server tag
            chan_name: Channel name
            channel_key: Server-qualified channel key
            nick: Nick who sent the message
            message: Message content
            mynick: Bot's nickname
            reply_sender: Send function for replies
            check_callback: Async function to call with message data after debounce
            thread_id: Optional platform thread identifier
            thread_starter_id: Internal ID of the thread starter message
        """
        channel_lock = self._get_channel_lock(channel_key)

        async with channel_lock:
            # Cancel existing timer for this channel
            if channel_key in self._pending_timers:
                self._pending_timers[channel_key].cancel()
                logger.debug(f"Cancelled previous debounce timer for {channel_key}")

            # Store latest message
            self._pending_messages[channel_key] = PendingMessage(
                server,
                chan_name,
                channel_key,
                nick,
                message,
                mynick,
                reply_sender,
                time.time(),
                thread_id=thread_id,
                thread_starter_id=thread_starter_id,
                secrets=secrets,
            )
            logger.debug(f"Scheduled debounced check for {channel_key}: {message[:100]}...")

            # Schedule new debounced check
            self._pending_timers[channel_key] = asyncio.create_task(
                self._debounced_check(channel_key, check_callback)
            )

    async def _debounced_check(
        self,
        channel_key: str,
        check_callback: Callable[
            [
                str,
                str,
                str,
                str,
                str,
                Callable[[str], Awaitable[None]],
                str | None,
                int | None,
                dict[str, Any] | None,
            ],
            Awaitable[None],
        ],
    ) -> None:
        """Execute the debounced check after delay."""
        try:
            await asyncio.sleep(self.debounce_seconds)

            channel_lock = self._get_channel_lock(channel_key)
            async with channel_lock:
                if channel_key in self._pending_messages:
                    msg = self._pending_messages[channel_key]
                    logger.debug(
                        "Executing debounced proactive check for %s: %s...",
                        channel_key,
                        msg.message[:100],
                    )

                    # Execute with fresh logging context for this proactive check
                    arc = f"{msg.server}#{msg.chan_name}"
                    with MessageLoggingContext(
                        arc, f"proactive-{msg.nick}", msg.message, get_logs_dir()
                    ):
                        await check_callback(
                            msg.server,
                            msg.chan_name,
                            msg.nick,
                            msg.message,
                            msg.mynick,
                            msg.reply_sender,
                            msg.thread_id,
                            msg.thread_starter_id,
                            msg.secrets,
                        )

                    # Cleanup
                    del self._pending_messages[channel_key]
                    if channel_key in self._pending_timers:
                        del self._pending_timers[channel_key]

        except asyncio.CancelledError:
            logger.debug(f"Debounced check cancelled for {channel_key}")
            raise
        except Exception as e:
            logger.error(f"Error in debounced check for {channel_key}: {e}")
            # Cleanup even on exception
            channel_lock = self._get_channel_lock(channel_key)
            async with channel_lock:
                if channel_key in self._pending_messages:
                    del self._pending_messages[channel_key]
                if channel_key in self._pending_timers:
                    del self._pending_timers[channel_key]

    async def cancel_all(self) -> None:
        """Cancel all pending debounced checks."""
        for timer in self._pending_timers.values():
            timer.cancel()

        # Wait for all tasks to complete cancellation
        if self._pending_timers:
            await asyncio.gather(*self._pending_timers.values(), return_exceptions=True)

        self._pending_timers.clear()
        self._pending_messages.clear()
        logger.debug("Cancelled all pending debounced checks")

    def get_pending_channels(self) -> list[str]:
        """Get list of channels with pending debounced checks."""
        return list(self._pending_messages.keys())

    def is_pending(self, channel_key: str) -> bool:
        """Check if a channel has a pending debounced check."""
        return channel_key in self._pending_messages

    async def cancel_channel(self, channel_key: str) -> None:
        """Cancel pending debounced check for a specific channel.

        Args:
            channel_key: Server-qualified channel key to cancel check for
        """
        channel_lock = self._get_channel_lock(channel_key)

        async with channel_lock:
            if channel_key in self._pending_timers:
                self._pending_timers[channel_key].cancel()
                logger.debug(
                    "Cancelled debounced check for %s due to command processing",
                    channel_key,
                )

                # Wait for the task to complete cancellation
                with contextlib.suppress(asyncio.CancelledError):
                    await self._pending_timers[channel_key]

                # Cleanup
                del self._pending_timers[channel_key]
                if channel_key in self._pending_messages:
                    del self._pending_messages[channel_key]
