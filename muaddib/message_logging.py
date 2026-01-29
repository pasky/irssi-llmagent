"""Per-message logging infrastructure using contextvars.

Each incoming message gets its own log file with all processing traces,
organized by arc (server#channel or server#nick) and timestamp.

Directory structure:
  logs/
    YYYY-MM-DD/
      server#channel/
        HH-MM-SS-nick-message_prefix.log
      server#nick/
        HH-MM-SS-nick-message_prefix.log
      system.log  # startup, global events
"""

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Context variable holding current message context
_message_context: ContextVar["MessageContext | None"] = ContextVar("message_context", default=None)


@dataclass
class MessageContext:
    """Context for the current message being processed."""

    arc: str  # e.g., "libera#python" or "libera#someuser"
    nick: str  # sender nick
    message_preview: str  # first ~20 chars of message
    timestamp: datetime
    log_path: Path

    @classmethod
    def create(cls, arc: str, nick: str, message: str, logs_dir: Path) -> "MessageContext":
        """Create a new message context and set up the log file path."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        # Sanitize message preview for filename (first 50 chars, alphanumeric + underscore)
        preview = re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]", "_", message[:50])).strip("_").lower()
        if not preview:
            preview = "msg"

        # Sanitize arc for filesystem (replace problematic chars)
        arc_safe = arc.replace("/", "_").replace("\\", "_")

        # Build path: logs/YYYY-MM-DD/arc/HH-MM-SS-nick-preview.log
        log_dir = logs_dir / date_str / arc_safe
        log_filename = f"{time_str}-{nick}-{preview}.log"
        log_path = log_dir / log_filename

        return cls(
            arc=arc,
            nick=nick,
            message_preview=preview,
            timestamp=now,
            log_path=log_path,
        )


def set_message_context(ctx: MessageContext | None) -> None:
    """Set the current message context."""
    _message_context.set(ctx)


def get_message_context() -> MessageContext | None:
    """Get the current message context."""
    return _message_context.get()


class MessageContextHandler(logging.Handler):
    """Logging handler that routes logs to per-message files based on context.

    When a message context is active, logs go to that message's log file.
    When no context is active, logs go to a per-day system.log file.

    Uses LRU eviction to limit the number of open file handles.
    """

    MAX_OPEN_HANDLES = 100

    def __init__(self, logs_dir: str | Path, level: int = logging.DEBUG):
        super().__init__(level)
        self.logs_dir = Path(logs_dir)

        # LRU cache of open file handles (path -> handler), limited size
        self._file_handles: dict[Path, logging.FileHandler] = {}

        # Formatter for log messages
        self._formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def _get_system_log_path(self, record: logging.LogRecord) -> Path:
        date_str = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d")
        return self.logs_dir / date_str / "system.log"

    def _get_handler_for_path(self, path: Path) -> logging.FileHandler:
        """Get or create a file handler for the given path, with LRU eviction."""
        if path in self._file_handles:
            # Move to end (most recently used)
            handler = self._file_handles.pop(path)
            self._file_handles[path] = handler
            return handler

        # Evict oldest handles if at capacity
        while len(self._file_handles) >= self.MAX_OPEN_HANDLES:
            oldest_path, oldest_handler = next(iter(self._file_handles.items()))
            oldest_handler.close()
            del self._file_handles[oldest_path]

        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(self._formatter)
        self._file_handles[path] = handler
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the appropriate file."""
        try:
            ctx = get_message_context()
            path = ctx.log_path if ctx is not None else self._get_system_log_path(record)
            handler = self._get_handler_for_path(path)
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close all open file handles."""
        for handler in self._file_handles.values():
            handler.close()
        self._file_handles.clear()
        super().close()


class MessageLoggingContext:
    """Context manager for message-scoped logging.

    Usage:
        with MessageLoggingContext(arc, nick, message):
            # All logging in this block goes to the message's log file
            logger.info("Processing message...")
    """

    def __init__(self, arc: str, nick: str, message: str):
        self.arc = arc
        self.nick = nick
        self.message = message
        self._token = None
        self._ctx: MessageContext | None = None

    def __enter__(self) -> MessageContext:
        from .paths import get_logs_dir

        self._ctx = MessageContext.create(self.arc, self.nick, self.message, get_logs_dir())
        logging.getLogger(__name__).info(f"Starting message log: {self._ctx.log_path}")
        self._token = _message_context.set(self._ctx)
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None and self._ctx is not None:
            ctx = self._ctx
            _message_context.reset(self._token)
            logging.getLogger(__name__).info(f"Finished message log: {ctx.log_path}")
        return None
