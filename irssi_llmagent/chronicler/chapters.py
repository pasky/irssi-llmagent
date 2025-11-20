"""Chronicle chapter management - handling chapter closing/opening and automatic summaries."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Per-arc locks to prevent concurrent access to the same arc
_arc_locks: dict[str, asyncio.Lock] = {}


def _get_arc_lock(arc: str) -> asyncio.Lock:
    """Get or create a lock for the specified arc."""
    # Clean up locks from different event loops
    try:
        if arc in _arc_locks:
            # Test if the lock is still valid for current event loop
            _arc_locks[arc]._get_loop()  # type: ignore[attr-defined]
    except RuntimeError:
        # Lock is from a different event loop, remove it
        if arc in _arc_locks:
            del _arc_locks[arc]

    if arc not in _arc_locks:
        _arc_locks[arc] = asyncio.Lock()
    return _arc_locks[arc]


async def _generate_chapter_summary(agent: Any, arc: str, chapter_paragraphs: list[str]) -> str:
    """Generate a single-paragraph summary of chapter paragraphs."""
    summary_prompt = """As an AI agent, you maintain a Chronicle (arcs → chapters → paragraphs) of your experiences, plans, thoughts and observations, forming the backbone of your consciousness.

Summarize the following chronicle chapter in a single paragraph.
Focus on the key events, decisions, and developments that happened.
Keep it concise but informative, as this will serve as a recap for the next chapter.

Respond only with the summary, no preamble."""

    user_content = "\n\n".join(chapter_paragraphs)

    chronicler_config = agent.config["chronicler"]
    model = chronicler_config.get("arc_models", {}).get(arc, chronicler_config["model"])

    resp, client, _ = await agent.model_router.call_raw_with_model(
        model_str=model,
        context=[{"role": "user", "content": user_content}],
        system_prompt=summary_prompt,
    )
    return client.extract_text_from_response(resp)


async def _close_chapter(chronicle: Any, chapter_id: int, summary: str) -> None:
    """Close a chapter by setting closed_at timestamp and storing summary in meta_json."""
    import json

    meta_json = json.dumps({"summary": summary})
    async with (
        aiosqlite.connect(chronicle.db_path) as db,
        db.execute(
            "UPDATE chapters SET closed_at = CURRENT_TIMESTAMP, meta_json = ? WHERE id = ?",
            (meta_json, chapter_id),
        ) as _,
    ):
        await db.commit()
    logger.info(f"Closed chapter {chapter_id} with summary")


async def _count_paragraphs_in_chapter(chronicle: Any, chapter_id: int) -> int:
    """Count number of paragraphs in a chapter."""
    async with (
        aiosqlite.connect(chronicle.db_path) as db,
        db.execute("SELECT COUNT(*) FROM paragraphs WHERE chapter_id = ?", (chapter_id,)) as cursor,
    ):
        row = await cursor.fetchone()
        return int(row[0]) if row else 0


async def chapter_append_paragraph(arc: str, paragraph_text: str, agent: Any) -> dict[str, Any]:
    """
    Append paragraph to chronicle, handling chapter management.

    Checks if current chapter needs to be closed (based on paragraph count),
    generates summary if so, creates new chapter with recap, then appends paragraph.

    Uses per-arc locking to prevent race conditions when multiple async tasks try
    to append to the same arc simultaneously.
    """
    if not paragraph_text or not paragraph_text.strip():
        raise ValueError("paragraph_text must be non-empty")

    # Acquire lock for this arc to prevent concurrent modifications
    arc_lock = _get_arc_lock(arc)
    async with arc_lock:
        # Get current chapter for this arc
        current_chapter = await agent.chronicle.get_or_open_current_chapter(arc)
        chapter_id = current_chapter["id"]

        # Count paragraphs in current chapter
        paragraph_count = await _count_paragraphs_in_chapter(agent.chronicle, chapter_id)
        max_paragraphs = agent.config.get("chronicler", {}).get("paragraphs_per_chapter", 3)

        logger.debug(f"Chapter {chapter_id} limit check ({paragraph_count}/{max_paragraphs})")

        if paragraph_count >= max_paragraphs:
            # Get all paragraphs from current chapter
            chapter_paragraphs = await agent.chronicle.read_chapter(chapter_id)
            assert chapter_paragraphs, f"Chapter {chapter_id} should have paragraphs but found none"

            # Generate chapter summary
            summary = await _generate_chapter_summary(agent, arc, chapter_paragraphs)
            logger.info(f"Generated summary for chapter {chapter_id}: {summary[:500]}...")

            # Close current chapter with summary
            await _close_chapter(agent.chronicle, chapter_id, summary)

            # Create new chapter (this will be the new "current" chapter)
            new_chapter = await agent.chronicle.get_or_open_current_chapter(arc)
            new_chapter_id = new_chapter["id"]

            # Add recap paragraph to new chapter (use direct chronicle method to avoid recursion)
            recap_paragraph = f"Previous chapter recap: {summary}"
            await agent.chronicle.append_paragraph(arc, recap_paragraph)

            logger.debug(f"Created new chapter {new_chapter_id} with recap for arc '{arc}'")

            # Copy unresolved <quest> paragraphs from previous chapter into new chapter without triggering
            latest_by_id: dict[str, tuple[str, bool]] = {}
            import re

            quest_re = re.compile(r"<\s*quest\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)
            finished_re = re.compile(r"<\s*quest_finished\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)

            for p in chapter_paragraphs:
                mfin = finished_re.search(p)
                if mfin:
                    latest_by_id[mfin.group(1)] = (p, True)
                    continue
                m = quest_re.search(p)
                if m:
                    qid = m.group(1)
                    latest_by_id[qid] = (p, False)

            for qid, (p, is_finished) in latest_by_id.items():
                if not is_finished:
                    await agent.chronicle.append_paragraph(arc, p)  # no operator trigger
                    logger.debug(f"Copied unresolved quest {qid} into new chapter {new_chapter_id}")

        # Append the original paragraph (either to current chapter or new chapter)
        result = await agent.chronicle.append_paragraph(arc, paragraph_text)

        # Trigger quests operator if available
        await agent.quests.on_chronicle_append(arc, paragraph_text)

        return result
