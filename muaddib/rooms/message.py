"""Shared message abstraction for room monitors."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoomMessage:
    """A message in a room conversation."""

    # Location
    server_tag: str
    channel_name: str

    # Participants
    nick: str  # sender
    mynick: str  # bot

    # Content
    content: str

    # Platform threading
    platform_id: str | None = None
    thread_id: str | None = None
    thread_starter_id: int | None = None  # looked up from platform thread_id

    # Response thread override (Slack may start new threads for replies)
    # If None, responses go to thread_id
    _response_thread_id: str | None = field(default=None, repr=False)

    # Platform-specific auth for attachments etc.
    secrets: dict[str, Any] | None = None

    @property
    def arc(self) -> str:
        """Return arc identifier for logging/history."""
        return f"{self.server_tag}#{self.channel_name}"

    @property
    def response_thread_id(self) -> str | None:
        """Thread ID for responses (defaults to thread_id)."""
        return self._response_thread_id if self._response_thread_id is not None else self.thread_id
