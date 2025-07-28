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
        self.db_path = Path(db_path).expanduser()
        self.inference_limit = inference_limit
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    server_tag TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    nick TEXT NOT NULL,
                    message TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_server_channel
                ON chat_messages (server_tag, channel_name, timestamp)
            """
            )
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
    ) -> None:
        """Add a message to the chat history."""
        role = "assistant" if nick.lower() == mynick.lower() else "user"
        content = f"<{nick}> {message}"

        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                    INSERT INTO chat_messages
                    (server_tag, channel_name, nick, message, role)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                (server_tag, channel_name, nick, content, role),
            )
            await db.commit()

        logger.debug(f"Added message to history: {server_tag}/{channel_name} - {nick}: {message}")

    async def get_context(self, server_tag: str, channel_name: str) -> list[dict[str, str]]:
        """Get recent chat context for inference (limited by inference_limit)."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                    SELECT message, role FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                (server_tag, channel_name, self.inference_limit),
            )
            rows = await cursor.fetchall()

        # Reverse to get chronological order
        rows_list = list(rows)
        rows_list.reverse()
        context = [{"role": str(row[1]), "content": str(row[0])} for row in rows_list]
        logger.debug(f"Retrieved {len(context)} messages for context: {server_tag}/{channel_name}")
        return context

    async def get_full_history(
        self, server_tag: str, channel_name: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get full chat history for analysis (not limited by inference_limit)."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            query = """
                    SELECT nick, message, role, timestamp FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    ORDER BY timestamp DESC
                """
            params = [server_tag, channel_name]

            if limit:
                query += " LIMIT ?"
                params.append(str(limit))

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        rows_list = list(rows)
        rows_list.reverse()
        history = [
            {
                "nick": str(row[0]),
                "message": str(row[1]),
                "role": str(row[2]),
                "timestamp": str(row[3]),
            }
            for row in rows_list
        ]
        return history

    async def cleanup_old_messages(self, days: int = 30) -> int:
        """Remove messages older than specified days."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM chat_messages WHERE timestamp < datetime('now', '-' || ? || ' days')",
                (days,),
            )
            await db.commit()
            return cursor.rowcount

    async def close(self) -> None:
        """Close database connections."""
        # aiosqlite handles connection cleanup automatically
        pass
