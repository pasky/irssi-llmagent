"""Debounce proactive interjections per channel."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..message_logging import MessageLoggingContext
from .message import RoomMessage

logger = logging.getLogger(__name__)


@dataclass
class PendingCheck:
    """A pending proactive check with message and reply mechanism."""

    message: RoomMessage
    channel_key: str
    reply_sender: Callable[[str], Awaitable[None]]
    timestamp: float


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
        self._pending_messages: dict[str, PendingCheck] = {}
        self._channel_locks: dict[str, asyncio.Lock] = {}

    def _get_channel_lock(self, channel_key: str) -> asyncio.Lock:
        """Get or create a lock for the specific channel."""
        if channel_key not in self._channel_locks:
            self._channel_locks[channel_key] = asyncio.Lock()
        return self._channel_locks[channel_key]

    async def schedule_check(
        self,
        msg: RoomMessage,
        channel_key: str,
        reply_sender: Callable[[str], Awaitable[None]],
        check_callback: Callable[
            [RoomMessage, Callable[[str], Awaitable[None]]],
            Awaitable[None],
        ],
    ) -> None:
        """Schedule a debounced proactive check for this channel.

        Args:
            msg: The room message to check
            channel_key: Server-qualified channel key
            reply_sender: Send function for replies
            check_callback: Async function to call with message after debounce
        """
        channel_lock = self._get_channel_lock(channel_key)

        async with channel_lock:
            # Cancel existing timer for this channel
            if channel_key in self._pending_timers:
                self._pending_timers[channel_key].cancel()
                logger.debug(f"Cancelled previous debounce timer for {channel_key}")

            # Store latest message
            self._pending_messages[channel_key] = PendingCheck(
                message=msg,
                channel_key=channel_key,
                reply_sender=reply_sender,
                timestamp=time.time(),
            )
            logger.debug(f"Scheduled debounced check for {channel_key}: {msg.content[:100]}...")

            # Schedule new debounced check
            self._pending_timers[channel_key] = asyncio.create_task(
                self._debounced_check(channel_key, check_callback)
            )

    async def _debounced_check(
        self,
        channel_key: str,
        check_callback: Callable[
            [RoomMessage, Callable[[str], Awaitable[None]]],
            Awaitable[None],
        ],
    ) -> None:
        """Execute the debounced check after delay."""
        try:
            await asyncio.sleep(self.debounce_seconds)

            channel_lock = self._get_channel_lock(channel_key)
            async with channel_lock:
                if channel_key in self._pending_messages:
                    pending = self._pending_messages[channel_key]
                    logger.debug(
                        "Executing debounced proactive check for %s: %s...",
                        channel_key,
                        pending.message.content[:100],
                    )

                    # Execute with fresh logging context for this proactive check
                    msg = pending.message
                    with MessageLoggingContext(msg.arc, f"proactive-{msg.nick}", msg.content):
                        await check_callback(msg, pending.reply_sender)

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
