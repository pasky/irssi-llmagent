"""Chat history management with SQLite persistence."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class ChatHistory:
    """Persistent chat history using SQLite with configurable limits for inference."""

    def __init__(self, db_path: str = "chat_history.db", inference_limit: int = 5):
        # Handle in-memory database path specially
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path).expanduser()
        self.inference_limit = inference_limit
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_tag TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    nick TEXT NOT NULL,
                    message TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    chapter_id INTEGER NULL
                )
            """
            ) as _:
                pass
            async with db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_server_channel
                ON chat_messages (server_tag, channel_name, timestamp)
            """
            ) as _:
                pass
            async with db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chapter_id
                ON chat_messages (chapter_id)
            """
            ) as _:
                pass
            await db.commit()
            logger.debug(f"Initialized chat history database: {self.db_path}")

    async def add_message(
        self,
        server_tag: str,
        channel_name: str,
        message: str,
        nick: str,
        mynick: str,
        is_response: bool = False,
        content_template: str = "<{nick}> {message}",
    ) -> None:
        """Add a message to the chat history."""
        role = "assistant" if nick.lower() == mynick.lower() else "user"
        content = content_template.format(nick=nick, message=message)

        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                    INSERT INTO chat_messages
                    (server_tag, channel_name, nick, message, role)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                (server_tag, channel_name, nick, content, role),
            ) as _,
        ):
            await db.commit()

        logger.debug(f"Added message to history: {server_tag}/{channel_name} - {nick}: {message}")

    async def get_context(
        self, server_tag: str, channel_name: str, limit: int | None = None
    ) -> list[dict[str, str]]:
        """Get recent chat context for inference (limited by inference_limit or provided limit)."""
        inference_limit = limit if limit is not None else self.inference_limit
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                    SELECT message, role, strftime('%H:%M', timestamp) as time_only FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                (server_tag, channel_name, inference_limit),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        # Reverse to get chronological order
        rows_list = list(rows)
        rows_list.reverse()
        context = [{"role": str(row[1]), "content": f"[{row[2]}] {row[0]}"} for row in rows_list]
        logger.debug(f"Retrieved {len(context)} messages for context: {server_tag}/{channel_name}")
        return context

    async def get_full_history(
        self, server_tag: str, channel_name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get full chat history for analysis (not limited by inference_limit)."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            query = """
                    SELECT id, nick, message, role, timestamp FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    ORDER BY timestamp DESC
                """
            params = [server_tag, channel_name]

            if limit:
                query += " LIMIT ?"
                params.append(str(limit))

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        rows_list = list(rows)
        rows_list.reverse()
        history = [
            {
                "id": int(row[0]),
                "nick": str(row[1]),
                "message": str(row[2]),
                "role": str(row[3]),
                "timestamp": str(row[4]),
            }
            for row in rows_list
        ]
        return history

    async def cleanup_old_messages(self, days: int = 30) -> int:
        """Remove messages older than specified days."""
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "DELETE FROM chat_messages WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (days,),
            ) as cursor,
        ):
            await db.commit()
            return cursor.rowcount

    async def get_recent_messages_since(
        self, server_tag: str, channel_name: str, nick: str, timestamp: float
    ) -> list[dict[str, str]]:
        """Get messages from a specific user since a given timestamp."""
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                SELECT message, timestamp FROM chat_messages
                WHERE server_tag = ? AND channel_name = ? AND nick = ?
                AND strftime('%s', timestamp) > ?
                ORDER BY timestamp ASC
                """,
                # strftime('%s', timestamp) converts SQLite datetime to Unix timestamp for comparison
                (server_tag, channel_name, nick, str(int(timestamp))),
            ) as cursor,
        ):
            rows = await cursor.fetchall()

        # Extract message text (message field stores "<nick> text", strip the prefix)
        messages = []
        for row in rows:
            content = str(row[0])
            # Find first "> " and take everything after it
            if "> " in content:
                message_text = content.split("> ", 1)[1]
                messages.append({"message": message_text, "timestamp": str(row[1])})

        logger.debug(f"Found {len(messages)} followup messages from {nick} since {timestamp}")
        return messages

    async def count_recent_unchronicled(
        self, server_tag: str, channel_name: str, days: int = 7
    ) -> int:
        """Count unchronicled messages from the last N days."""
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                SELECT COUNT(*) FROM chat_messages
                WHERE server_tag = ? AND channel_name = ?
                AND chapter_id IS NULL
                AND timestamp >= datetime('now', '-' || ? || ' days')
                """,
                (server_tag, channel_name, days),
            ) as cursor,
        ):
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def mark_chronicled(self, message_ids: list[int], chapter_id: int) -> None:
        """Mark messages as chronicled by setting their chapter_id."""
        if not message_ids:
            return

        async with self._lock, aiosqlite.connect(self.db_path) as db:
            placeholders = ",".join("?" * len(message_ids))
            async with db.execute(
                f"""
                UPDATE chat_messages
                SET chapter_id = ?
                WHERE id IN ({placeholders})
                """,
                [chapter_id] + message_ids,
            ) as _:
                await db.commit()
            logger.debug(
                f"Marked {len(message_ids)} messages as chronicled in chapter {chapter_id}"
            )

    async def close(self) -> None:
        """Close database connections."""
        # aiosqlite handles connection cleanup automatically
        pass
