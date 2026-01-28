"""Muaddib - AI chatbot for IRC via irssi-varlink."""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

__version__ = "1.0.0"

_background_tasks: set[asyncio.Task[Any]] = set()
_logger = logging.getLogger(__name__)


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
    """Create a background task that won't be garbage collected."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)

    def _on_done(t: asyncio.Task[Any]) -> None:
        _background_tasks.discard(t)
        if not t.cancelled() and (exc := t.exception()):
            _logger.error(f"Background task {t.get_name()!r} failed: {exc}")

    task.add_done_callback(_on_done)
    return task
