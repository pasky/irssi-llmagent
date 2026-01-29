"""Chronicle storage (arcs → chapters → paragraphs) using SQLite (async)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from . import QuestStatus

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    id: int
    arc_id: int
    opened_at: str
    closed_at: str | None
    meta_json: str | None


class Chronicle:
    """Persistent chronicle separate from IRC chat history.

    Arcs are required for all operations. Each arc has at most one open chapter.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS arcs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL UNIQUE,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chapters (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  arc_id INTEGER NOT NULL,
                  opened_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  closed_at DATETIME,
                  meta_json TEXT,
                  FOREIGN KEY (arc_id) REFERENCES arcs(id) ON DELETE CASCADE
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_chapters_arc_open
                ON chapters(arc_id)
                WHERE closed_at IS NULL;

                CREATE TABLE IF NOT EXISTS paragraphs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chapter_id INTEGER NOT NULL,
                  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                  content TEXT NOT NULL,
                  FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_paragraphs_chapter_ts
                ON paragraphs(chapter_id, ts);

                CREATE INDEX IF NOT EXISTS idx_chapters_arc_opened
                ON chapters(arc_id, opened_at);

                CREATE TABLE IF NOT EXISTS quests (
                  id TEXT PRIMARY KEY,
                  arc_id INTEGER NOT NULL,
                  parent_id TEXT,
                  status TEXT NOT NULL CHECK(status IN ('ongoing', 'in_step', 'finished')),
                  last_state TEXT,
                  plan TEXT,
                  resume_at TEXT,
                  created_by_paragraph_id INTEGER,
                  last_updated_by_paragraph_id INTEGER,
                  FOREIGN KEY (arc_id) REFERENCES arcs(id),
                  FOREIGN KEY (parent_id) REFERENCES quests(id),
                  FOREIGN KEY (created_by_paragraph_id) REFERENCES paragraphs(id),
                  FOREIGN KEY (last_updated_by_paragraph_id) REFERENCES paragraphs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_quests_arc_status
                ON quests(arc_id, status);

                CREATE INDEX IF NOT EXISTS idx_quests_parent
                ON quests(parent_id);
                """
            )
            await db.commit()

            # Migration: add resume_at column if missing (for existing DBs)
            async with db.execute("PRAGMA table_info(quests)") as cur:
                columns = {row[1] for row in await cur.fetchall()}
            if "resume_at" not in columns:
                await db.execute("ALTER TABLE quests ADD COLUMN resume_at TEXT")
                await db.commit()

    async def _get_or_create_arc(self, arc: str) -> tuple[int, bool]:
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT id FROM arcs WHERE name = ?", (arc,)) as cur:
                row = await cur.fetchone()
                if row:
                    return int(row[0]), False
            async with db.execute("INSERT INTO arcs(name) VALUES (?)", (arc,)) as cur:
                await db.commit()
                last_id = int(cur.lastrowid or 0)
            return last_id, True

    async def _get_open_chapter_row(self, arc_id: int) -> Chapter | None:
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT id, arc_id, opened_at, closed_at, meta_json FROM chapters"
                " WHERE arc_id = ? AND closed_at IS NULL",
                (arc_id,),
            ) as cur,
        ):
            row = await cur.fetchone()
            if not row:
                return None
            return Chapter(int(row[0]), int(row[1]), str(row[2]), row[3], row[4])

    async def _open_new_chapter(self, arc_id: int) -> Chapter:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "INSERT INTO chapters(arc_id) VALUES (?)",
                (arc_id,),
            ) as cur:
                await db.commit()
                chapter_id = int(cur.lastrowid or 0)
            async with db.execute(
                "SELECT id, arc_id, opened_at, closed_at, meta_json FROM chapters WHERE id = ?",
                (chapter_id,),
            ) as cur:
                row = await cur.fetchone()
                assert row is not None
                return Chapter(int(row[0]), int(row[1]), str(row[2]), row[3], row[4])

    async def get_or_open_current_chapter(self, arc: str) -> dict[str, Any]:
        arc_id, new_arc = await self._get_or_create_arc(arc)
        async with self._lock:
            chapter = await self._get_open_chapter_row(arc_id)
            if not chapter:
                chapter = await self._open_new_chapter(arc_id)
        if new_arc:
            await self.append_paragraph(
                arc, "<meta>This is a beginning of an entirely new story arc!</meta>"
            )
        return {
            "id": chapter.id,
            "arc_id": chapter.arc_id,
            "opened_at": chapter.opened_at,
            "closed_at": chapter.closed_at,
            "meta_json": chapter.meta_json,
        }

    async def append_paragraph(self, arc: str, content: str) -> dict[str, Any]:
        if not content or not content.strip():
            raise ValueError("content must be non-empty")
        chapter = await self.get_or_open_current_chapter(arc)
        chapter_id = int(chapter["id"])
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "INSERT INTO paragraphs(chapter_id, content) VALUES (?, ?)",
                (chapter_id, content),
            ) as cur:
                await db.commit()
                para_id = int(cur.lastrowid or 0)
            async with db.execute(
                "SELECT id, chapter_id, ts, content FROM paragraphs WHERE id = ?",
                (para_id,),
            ) as cur:
                row = await cur.fetchone()
                assert row is not None
        return {
            "id": int(row[0]),
            "chapter_id": int(row[1]),
            "ts": str(row[2]),
            "content": str(row[3]),
        }

    async def _resolve_chapter_id(
        self, arc: str, chapter_id: int | None
    ) -> tuple[int | None, Chapter | None]:
        arc_id, _ = await self._get_or_create_arc(arc)
        if chapter_id is not None:
            async with (
                aiosqlite.connect(self.db_path) as db,
                db.execute(
                    "SELECT id, arc_id, opened_at, closed_at, meta_json FROM chapters WHERE id = ? AND arc_id = ?",
                    (chapter_id, arc_id),
                ) as cur,
            ):
                row = await cur.fetchone()
                if not row:
                    return None, None
                return int(row[0]), Chapter(int(row[0]), int(row[1]), str(row[2]), row[3], row[4])
        # try open chapter first
        open_ch = await self._get_open_chapter_row(arc_id)
        if open_ch:
            return open_ch.id, open_ch
        # fallback to latest closed
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT id, arc_id, opened_at, closed_at, meta_json FROM chapters"
                " WHERE arc_id = ? ORDER BY opened_at DESC LIMIT 1",
                (arc_id,),
            ) as cur,
        ):
            row = await cur.fetchone()
            if not row:
                return None, None
            return int(row[0]), Chapter(int(row[0]), int(row[1]), str(row[2]), row[3], row[4])

    async def _resolve_relative_chapter_id(
        self, arc: str, relative_chapter_id: int
    ) -> tuple[int | None, Chapter | None]:
        """Resolve a relative chapter ID (0=current, -1=previous, -2=two chapters back, etc.)"""
        arc_id, _ = await self._get_or_create_arc(arc)

        # Get all chapters for this arc, ordered by opened_at (oldest first)
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT id, arc_id, opened_at, closed_at, meta_json FROM chapters"
                " WHERE arc_id = ? ORDER BY opened_at ASC",
                (arc_id,),
            ) as cur,
        ):
            rows = await cur.fetchall()

        if not rows:
            return None, None

        # Find current chapter (open chapter or latest chapter)
        open_ch = await self._get_open_chapter_row(arc_id)
        current_chapter_id = open_ch.id if open_ch else max(int(row[0]) for row in rows)

        # Build ordered list of chapter IDs (oldest to newest)
        chapter_ids = [int(row[0]) for row in rows]

        # Find index of current chapter
        try:
            current_index = chapter_ids.index(current_chapter_id)
        except ValueError:
            return None, None

        # Calculate target index based on relative offset
        target_index = current_index + relative_chapter_id

        if target_index < 0 or target_index >= len(chapter_ids):
            return None, None

        target_chapter_id = chapter_ids[target_index]

        # Find the row data for the target chapter
        for row in rows:
            if int(row[0]) == target_chapter_id:
                return int(row[0]), Chapter(int(row[0]), int(row[1]), str(row[2]), row[3], row[4])

        return None, None

    async def read_chapter(self, chapter_id: int) -> list[str]:
        """Read all paragraphs from a chapter."""
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT content FROM paragraphs WHERE chapter_id = ? ORDER BY ts ASC", (chapter_id,)
            ) as cursor,
        ):
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_chapter_context_messages(self, arc: str) -> list[dict[str, str]]:
        """Get chapter context as user messages for prepending to conversation.

        Args:
            arc: Arc name in format "server#channel"

        Returns:
            List of context messages to prepend, each wrapped in <context_summary> tags
        """
        current_chapter = await self.get_or_open_current_chapter(arc)
        chapter_id = current_chapter["id"]
        paragraphs = await self.read_chapter(int(chapter_id))
        context_messages = []
        for paragraph in paragraphs:
            context_messages.append(
                {"role": "user", "content": f"<context_summary>{paragraph}</context_summary>"}
            )

        return context_messages

    async def render_chapter(
        self, arc: str, chapter_id: int | None = None, last_n: int | None = None
    ) -> str:
        chap_id, chap = await self._resolve_chapter_id(arc, chapter_id)
        if chap_id is None or chap is None:
            return f"# Arc: {arc} — No chapters yet\n\n(Empty)"

        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT ts, content FROM paragraphs WHERE chapter_id = ? ORDER BY ts ASC",
                (chap_id,),
            ) as cur,
        ):
            rows = await cur.fetchall()

        rows_list = list(rows)
        if last_n is not None and last_n > 0:
            rows_list = rows_list[-last_n:]

        # Format markdown
        title = f"# Arc: {arc} — Chapter {chap.id} (opened {chap.opened_at.split('.')[0]})"
        if chap.closed_at:
            title += f", closed {str(chap.closed_at).split('.')[0]}"
        lines = [title, "", "Paragraphs:"]
        for ts, content in rows_list:
            hhmm = str(ts)[11:16] if len(str(ts)) >= 16 else str(ts)
            lines.append(f"[{hhmm}] {content}")
        if len(rows_list) == 0:
            lines.append("(No paragraphs)")
        return "\n".join(lines)

    async def render_chapter_relative(
        self, arc: str, relative_chapter_id: int, last_n: int | None = None
    ) -> str:
        """Render a chapter using relative chapter ID (0=current, -1=previous, etc.)"""
        chap_id, chap = await self._resolve_relative_chapter_id(arc, relative_chapter_id)
        if chap_id is None or chap is None:
            return f"# Arc: {arc} — No chapters at relative offset {relative_chapter_id}\n\n(Empty)"

        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT ts, content FROM paragraphs WHERE chapter_id = ? ORDER BY ts ASC",
                (chap_id,),
            ) as cur,
        ):
            rows = await cur.fetchall()

        rows_list = list(rows)
        if last_n is not None and last_n > 0:
            rows_list = rows_list[-last_n:]

        # Format markdown with relative offset info
        relative_desc = (
            "current"
            if relative_chapter_id == 0
            else f"{abs(relative_chapter_id)} chapter{'s' if abs(relative_chapter_id) > 1 else ''} {'back' if relative_chapter_id < 0 else 'forward'}"
        )
        title = f"# Arc: {arc} — Chapter {chap.id} ({relative_desc}, opened {chap.opened_at.split('.')[0]})"
        if chap.closed_at:
            title += f", closed {str(chap.closed_at).split('.')[0]}"
        lines = [title, "", "Paragraphs:"]
        for ts, content in rows_list:
            hhmm = str(ts)[11:16] if len(str(ts)) >= 16 else str(ts)
            lines.append(f"[{hhmm}] {content}")
        if len(rows_list) == 0:
            lines.append("(No paragraphs)")
        return "\n".join(lines)

    # --- Quest management methods ---

    async def quest_start(
        self,
        quest_id: str,
        arc: str,
        paragraph_id: int,
        state_text: str,
        parent_id: str | None = None,
    ) -> None:
        """Create a new quest with status=ONGOING."""
        arc_id, _ = await self._get_or_create_arc(arc)
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO quests
                   (id, arc_id, parent_id, status, last_state,
                    created_by_paragraph_id, last_updated_by_paragraph_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    quest_id,
                    arc_id,
                    parent_id,
                    QuestStatus.ONGOING.value,
                    state_text,
                    paragraph_id,
                    paragraph_id,
                ),
            )
            await db.commit()
        logger.debug(f"Quest started: {quest_id} (parent={parent_id})")

    async def quest_update(self, quest_id: str, state_text: str, paragraph_id: int) -> None:
        """Update last_state and last_updated_by_paragraph_id for a quest."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE quests
                   SET last_state = ?, last_updated_by_paragraph_id = ?
                   WHERE id = ?""",
                (state_text, paragraph_id, quest_id),
            )
            await db.commit()
        logger.debug(f"Quest updated: {quest_id}")

    async def quest_finish(self, quest_id: str, paragraph_id: int) -> None:
        """Set quest status to FINISHED."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE quests
                   SET status = ?, last_updated_by_paragraph_id = ?
                   WHERE id = ?""",
                (QuestStatus.FINISHED.value, paragraph_id, quest_id),
            )
            await db.commit()
        logger.debug(f"Quest finished: {quest_id}")

    async def quest_set_status(self, quest_id: str, status: QuestStatus) -> None:
        """Set quest status directly (for ONGOING <-> IN_STEP transitions)."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE quests SET status = ? WHERE id = ?",
                (status.value, quest_id),
            )
            await db.commit()
        logger.debug(f"Quest {quest_id} status set to {status.value}")

    async def quest_try_transition(
        self, quest_id: str, from_status: QuestStatus, to_status: QuestStatus
    ) -> bool:
        """Atomically transition quest status if current status matches.

        Returns True if successful, False if quest was not in expected status.
        """
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE quests SET status = ? WHERE id = ? AND status = ?",
                (to_status.value, quest_id, from_status.value),
            )
            await db.commit()
            if cursor.rowcount > 0:
                logger.debug(
                    f"Quest {quest_id} transitioned {from_status.value} -> {to_status.value}"
                )
                return True
            logger.debug(f"Quest {quest_id} not transitioned (not {from_status.value})")
            return False

    async def quest_get(self, quest_id: str) -> dict[str, Any] | None:
        """Return quest row as dict, or None if not found."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT id, arc_id, parent_id, status, last_state, plan, resume_at,
                          created_by_paragraph_id, last_updated_by_paragraph_id
                   FROM quests WHERE id = ?""",
                (quest_id,),
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return dict(row)

    async def quest_set_plan(self, quest_id: str, plan: str) -> bool:
        """Set the plan for a quest. Returns True if quest existed."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE quests SET plan = ? WHERE id = ?",
                (plan, quest_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def quest_set_resume_at(self, quest_id: str, resume_at: str | None) -> bool:
        """Set the resume_at timestamp for a quest. Pass None to clear. Returns True if quest existed."""
        async with self._lock, aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE quests SET resume_at = ? WHERE id = ?",
                (resume_at, quest_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def quests_ready_for_heartbeat(
        self, arc: str, cooldown_seconds: float
    ) -> list[dict[str, Any]]:
        """Return ONGOING quests ready for heartbeat trigger.

        A quest is ready if:
        - status is ONGOING
        - last_updated paragraph.ts + cooldown < now
        - no children with status in (ONGOING, IN_STEP)
        - resume_at is NULL or now >= resume_at

        Also logs warnings for any orphaned quests (ONGOING with no active children
        but parent has active children that don't include this quest).
        """
        arc_id, _ = await self._get_or_create_arc(arc)
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT q.id, q.arc_id, q.parent_id, q.status, q.last_state,
                          q.created_by_paragraph_id, q.last_updated_by_paragraph_id
                   FROM quests q
                   JOIN paragraphs p ON q.last_updated_by_paragraph_id = p.id
                   WHERE q.arc_id = ?
                     AND q.status = ?
                     AND datetime(p.ts, '+' || ? || ' seconds') <= datetime('now')
                     AND (q.resume_at IS NULL OR datetime('now') >= datetime(q.resume_at))
                     AND NOT EXISTS (
                       SELECT 1 FROM quests c
                       WHERE c.parent_id = q.id AND c.status IN (?, ?)
                     )""",
                (
                    arc_id,
                    QuestStatus.ONGOING.value,
                    int(cooldown_seconds),
                    QuestStatus.ONGOING.value,
                    QuestStatus.IN_STEP.value,
                ),
            ) as cur:
                rows = await cur.fetchall()
                return [dict(row) for row in rows]

    async def quests_count_unfinished(self, arc: str) -> int:
        """Count quests with status != finished for an arc."""
        arc_id, _ = await self._get_or_create_arc(arc)
        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                "SELECT COUNT(*) FROM quests WHERE arc_id = ? AND status != ?",
                (arc_id, QuestStatus.FINISHED.value),
            ) as cur,
        ):
            row = await cur.fetchone()
            return int(row[0]) if row else 0
