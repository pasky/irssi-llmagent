"""Debounce proactive interjections per channel."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """Represents a message pending debounced processing."""

    server: str
    chan_name: str
    nick: str
    message: str
    mynick: str
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
        self._pending_messages: dict[str, PendingMessage] = {}
        self._channel_locks: dict[str, asyncio.Lock] = {}

    def _get_channel_lock(self, chan_name: str) -> asyncio.Lock:
        """Get or create a lock for the specific channel."""
        if chan_name not in self._channel_locks:
            self._channel_locks[chan_name] = asyncio.Lock()
        return self._channel_locks[chan_name]

    async def schedule_check(
        self,
        server: str,
        chan_name: str,
        nick: str,
        message: str,
        mynick: str,
        check_callback: Callable[[str, str, str, str, str], Awaitable[None]],
    ) -> None:
        """Schedule a debounced proactive check for this channel.

        Args:
            server: IRC server tag
            chan_name: Channel name
            nick: Nick who sent the message
            message: Message content
            mynick: Bot's nickname
            check_callback: Async function to call with message data after debounce
        """
        channel_lock = self._get_channel_lock(chan_name)

        async with channel_lock:
            # Cancel existing timer for this channel
            if chan_name in self._pending_timers:
                self._pending_timers[chan_name].cancel()
                logger.debug(f"Cancelled previous debounce timer for {chan_name}")

            # Store latest message
            self._pending_messages[chan_name] = PendingMessage(
                server, chan_name, nick, message, mynick, time.time()
            )
            logger.debug(f"Scheduled debounced check for {chan_name}: {message[:100]}...")

            # Schedule new debounced check
            self._pending_timers[chan_name] = asyncio.create_task(
                self._debounced_check(chan_name, check_callback)
            )

    async def _debounced_check(
        self,
        chan_name: str,
        check_callback: Callable[[str, str, str, str, str], Awaitable[None]],
    ) -> None:
        """Execute the debounced check after delay."""
        try:
            await asyncio.sleep(self.debounce_seconds)

            channel_lock = self._get_channel_lock(chan_name)
            async with channel_lock:
                if chan_name in self._pending_messages:
                    msg = self._pending_messages[chan_name]
                    logger.debug(
                        f"Executing debounced proactive check for {chan_name}: {msg.message[:100]}..."
                    )

                    # Execute with fresh context at check time
                    await check_callback(
                        msg.server, msg.chan_name, msg.nick, msg.message, msg.mynick
                    )

                    # Cleanup
                    del self._pending_messages[chan_name]
                    if chan_name in self._pending_timers:
                        del self._pending_timers[chan_name]

        except asyncio.CancelledError:
            logger.debug(f"Debounced check cancelled for {chan_name}")
            raise
        except Exception as e:
            logger.error(f"Error in debounced check for {chan_name}: {e}")
            # Cleanup even on exception
            channel_lock = self._get_channel_lock(chan_name)
            async with channel_lock:
                if chan_name in self._pending_messages:
                    del self._pending_messages[chan_name]
                if chan_name in self._pending_timers:
                    del self._pending_timers[chan_name]

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

    def is_pending(self, chan_name: str) -> bool:
        """Check if a channel has a pending debounced check."""
        return chan_name in self._pending_messages

    async def cancel_channel(self, chan_name: str) -> None:
        """Cancel pending debounced check for a specific channel.

        Args:
            chan_name: Channel name to cancel check for
        """
        channel_lock = self._get_channel_lock(chan_name)

        async with channel_lock:
            if chan_name in self._pending_timers:
                self._pending_timers[chan_name].cancel()
                logger.debug(f"Cancelled debounced check for {chan_name} due to command processing")

                # Wait for the task to complete cancellation
                with contextlib.suppress(asyncio.CancelledError):
                    await self._pending_timers[chan_name]

                # Cleanup
                del self._pending_timers[chan_name]
                if chan_name in self._pending_messages:
                    del self._pending_messages[chan_name]
