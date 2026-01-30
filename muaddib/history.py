"""Chat history management with SQLite persistence."""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from .rooms.message import RoomMessage

logger = logging.getLogger(__name__)


class ChatHistory:
    """Persistent chat history using SQLite with configurable limits for inference."""

    def __init__(self, db_path: str, inference_limit: int = 5):
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
                    chapter_id INTEGER NULL,
                    mode TEXT NULL,
                    llm_call_id INTEGER NULL,
                    platform_id TEXT NULL,
                    thread_id TEXT NULL
                )
            """
            ) as _:
                pass
            async with db.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL,
                    call_type TEXT,
                    arc_name TEXT
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
            async with db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_llm_calls_arc
                ON llm_calls (arc_name, timestamp)
            """
            ) as _:
                pass
            # Migrate existing tables: add mode column if missing
            async with db.execute("PRAGMA table_info(chat_messages)") as cursor:
                columns = [row[1] for row in await cursor.fetchall()]
            if "mode" not in columns:
                await db.execute("ALTER TABLE chat_messages ADD COLUMN mode TEXT NULL")
            if "llm_call_id" not in columns:
                await db.execute("ALTER TABLE chat_messages ADD COLUMN llm_call_id INTEGER NULL")
            if "platform_id" not in columns:
                await db.execute("ALTER TABLE chat_messages ADD COLUMN platform_id TEXT NULL")
            if "thread_id" not in columns:
                await db.execute("ALTER TABLE chat_messages ADD COLUMN thread_id TEXT NULL")
            async with db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_platform_id
                ON chat_messages (server_tag, channel_name, platform_id)
            """
            ) as _:
                pass
            # Migrate llm_calls: add trigger/response message links
            async with db.execute("PRAGMA table_info(llm_calls)") as cursor:
                llm_columns = [row[1] for row in await cursor.fetchall()]
            if "trigger_message_id" not in llm_columns:
                await db.execute("ALTER TABLE llm_calls ADD COLUMN trigger_message_id INTEGER NULL")
            if "response_message_id" not in llm_columns:
                await db.execute(
                    "ALTER TABLE llm_calls ADD COLUMN response_message_id INTEGER NULL"
                )
                # Backfill response_message_id from existing chat_messages.llm_call_id
                await db.execute(
                    """
                    UPDATE llm_calls SET response_message_id = (
                        SELECT id FROM chat_messages WHERE llm_call_id = llm_calls.id LIMIT 1
                    ) WHERE response_message_id IS NULL
                    """
                )
            await db.commit()
            logger.debug(f"Initialized chat history database: {self.db_path}")

    async def log_llm_call(
        self,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cost: float | None,
        call_type: str | None = None,
        arc_name: str | None = None,
        trigger_message_id: int | None = None,
    ) -> int:
        """Log an LLM API call and return its ID."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                INSERT INTO llm_calls
                (provider, model, input_tokens, output_tokens, cost, call_type, arc_name,
                 trigger_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    cost,
                    call_type,
                    arc_name,
                    trigger_message_id,
                ),
            ) as cursor:
                call_id = cursor.lastrowid
            await db.commit()

        cost_str = f"${cost:.6f}" if cost is not None else "N/A"
        logger.debug(
            f"Logged LLM call: {provider}:{model} in={input_tokens} out={output_tokens} "
            f"cost={cost_str} type={call_type} arc={arc_name} trigger={trigger_message_id}"
        )
        return call_id or 0

    async def update_llm_call_response(self, call_id: int, response_message_id: int) -> None:
        """Update an LLM call with the response message ID."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE llm_calls SET response_message_id = ? WHERE id = ?",
                (response_message_id, call_id),
            )
            await db.commit()

    async def add_message(
        self,
        msg: "RoomMessage",
        *,
        mode: str | None = None,
        llm_call_id: int | None = None,
        content_template: str = "<{nick}> {message}",
        role: str | None = None,
    ) -> int:
        """Add a RoomMessage to history. Returns the message ID.

        The message's response_thread_id is used for the thread_id (defaults to thread_id for incoming).
        """
        if role is None:
            role = "assistant" if msg.nick.lower() == msg.mynick.lower() else "user"
        content = content_template.format(nick=msg.nick, message=msg.content)

        async with self._lock, aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                INSERT INTO chat_messages
                (server_tag, channel_name, nick, message, role, mode, llm_call_id, platform_id, thread_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.server_tag,
                    msg.channel_name,
                    msg.nick,
                    content,
                    role,
                    mode,
                    llm_call_id,
                    msg.platform_id,
                    msg.response_thread_id,
                ),
            ) as cursor:
                message_id = cursor.lastrowid
            await db.commit()

        logger.debug(
            f"Added message to history: {msg.server_tag}/{msg.channel_name} - {msg.nick}: {msg.content}"
        )
        return message_id or 0

    async def get_context_for_message(
        self,
        msg: "RoomMessage",
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Get conversation context for a RoomMessage."""
        return await self.get_context(
            msg.server_tag,
            msg.channel_name,
            limit,
            thread_id=msg.thread_id,
            thread_starter_id=msg.thread_starter_id,
        )

    async def get_context(
        self,
        server_tag: str,
        channel_name: str,
        limit: int | None = None,
        thread_id: str | None = None,
        thread_starter_id: int | None = None,
    ) -> list[dict[str, str]]:
        """Get recent chat context for inference (limited by inference_limit or provided limit).

        For assistant messages with auto-routed mode, inserts a mode command prefix
        (e.g. !s for sarcastic) before the timestamp so models can learn/reproduce routing.
        """
        inference_limit = limit if limit is not None else self.inference_limit

        assert thread_id or thread_starter_id is None

        if thread_id:
            if thread_starter_id is not None:
                query = """
                    SELECT message, role, strftime('%H:%M', timestamp) as time_only, mode
                    FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    AND ((thread_id IS NULL AND id <= ?) OR thread_id = ?)
                    ORDER BY id DESC
                    LIMIT ?
                    """
                params = (server_tag, channel_name, thread_starter_id, thread_id, inference_limit)
            else:
                query = """
                    SELECT message, role, strftime('%H:%M', timestamp) as time_only, mode
                    FROM chat_messages
                    WHERE server_tag = ? AND channel_name = ?
                    AND (thread_id IS NULL OR thread_id = ?)
                    ORDER BY id DESC
                    LIMIT ?
                    """
                params = (server_tag, channel_name, thread_id, inference_limit)
        else:
            query = """
                SELECT message, role, strftime('%H:%M', timestamp) as time_only, mode
                FROM chat_messages
                WHERE server_tag = ? AND channel_name = ? AND thread_id IS NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """
            params = (server_tag, channel_name, inference_limit)

        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(query, params) as cursor,
        ):
            rows = await cursor.fetchall()

        # Reverse to get chronological order
        rows_list = list(rows)
        rows_list.reverse()
        context = []
        for row in rows_list:
            message, role, time_only, mode = row[0], str(row[1]), row[2], row[3]
            mode_prefix = self._mode_to_prefix(mode) if role == "assistant" and mode else ""
            context.append({"role": role, "content": f"{mode_prefix}[{time_only}] {message}"})
        logger.debug(f"Retrieved {len(context)} messages for context: {server_tag}/{channel_name}")
        return context

    @staticmethod
    def _mode_to_prefix(mode: str | None) -> str:
        """Convert internal mode name to IRC command prefix for context."""
        if not mode:
            return ""
        mode_prefixes = {
            "SARCASTIC": "!d ",
            "EASY_SERIOUS": "!s ",
            "THINKING_SERIOUS": "!a ",
            "UNSAFE": "!u ",
        }
        return mode_prefixes.get(mode, "")

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
        self,
        server_tag: str,
        channel_name: str,
        nick: str,
        timestamp: float,
        thread_id: str | None = None,
    ) -> list[dict[str, str]]:
        """Get messages from a specific user since a given timestamp."""
        if thread_id:
            query = """
                SELECT message, timestamp FROM chat_messages
                WHERE server_tag = ? AND channel_name = ? AND nick = ?
                AND strftime('%s', timestamp) > ? AND thread_id = ?
                ORDER BY timestamp ASC
                """
            params = (server_tag, channel_name, nick, str(int(timestamp)), thread_id)
        else:
            query = """
                SELECT message, timestamp FROM chat_messages
                WHERE server_tag = ? AND channel_name = ? AND nick = ?
                AND strftime('%s', timestamp) > ? AND thread_id IS NULL
                ORDER BY timestamp ASC
                """
            params = (server_tag, channel_name, nick, str(int(timestamp)))

        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(query, params) as cursor,
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

    async def get_message_id_by_platform_id(
        self, server_tag: str, channel_name: str, platform_id: str
    ) -> int | None:
        """Look up internal message ID by platform message ID."""
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                SELECT id FROM chat_messages
                WHERE server_tag = ? AND channel_name = ? AND platform_id = ?
                LIMIT 1
                """,
                (server_tag, channel_name, platform_id),
            ) as cursor,
        ):
            row = await cursor.fetchone()

        return int(row[0]) if row else None

    async def update_message_by_platform_id(
        self,
        server_tag: str,
        channel_name: str,
        platform_id: str,
        new_content: str,
        nick: str,
        content_template: str = "<{nick}> {message}",
    ) -> bool:
        """Update message content by platform ID. Returns True if updated."""
        formatted_content = content_template.format(nick=nick, message=new_content)
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                UPDATE chat_messages
                SET message = ?
                WHERE server_tag = ? AND channel_name = ? AND platform_id = ?
                """,
                (formatted_content, server_tag, channel_name, platform_id),
            ) as cursor:
                updated = cursor.rowcount > 0
            await db.commit()

        if updated:
            logger.debug(
                f"Updated message {platform_id} in {server_tag}/{channel_name}: {new_content}"
            )
        return updated

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

    async def get_arc_cost_today(self, arc_name: str) -> float:
        """Get total LLM cost for an arc since midnight today."""
        async with (
            self._lock,
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                SELECT COALESCE(SUM(cost), 0) FROM llm_calls
                WHERE arc_name = ?
                AND timestamp >= date('now', 'localtime')
                """,
                (arc_name,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
            return float(row[0]) if row else 0.0

    async def close(self) -> None:
        """Close database connections."""
        # aiosqlite handles connection cleanup automatically
        pass
