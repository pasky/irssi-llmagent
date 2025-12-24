"""Chronicle tools: Direct implementation of chronicle append and read tools."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .chapters import chapter_append_paragraph

logger = logging.getLogger(__name__)

QUEST_TAG_RE = re.compile(r"<\s*quest(_finished)?\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)


@dataclass
class ChapterAppendExecutor:
    agent: Any
    arc: str
    current_quest_id: str | None = field(default=None)

    async def execute(self, text: str) -> str:
        text, error = self._rewrite_quest_ids(text)
        if error:
            return f"Error: {error}"
        logger.info(f"Appending to {self.arc} chapter: {text}")
        await chapter_append_paragraph(self.arc, text, self.agent)
        return "OK"

    def _rewrite_quest_ids(self, text: str) -> tuple[str, str | None]:
        """Validate and auto-prefix quest IDs. Returns (text, error_or_none)."""
        parent_id = self.current_quest_id
        error: str | None = None

        def replacer(m: re.Match[str]) -> str:
            nonlocal error
            finished_suffix = m.group(1) or ""
            quest_id = m.group(2)

            # Dots are reserved for hierarchy - reject IDs containing dots
            if "." in quest_id:
                error = (
                    f'Quest ID "{quest_id}" cannot contain dots (reserved for sub-quest hierarchy).'
                )
                return m.group(0)

            if not parent_id:
                return m.group(0)

            # If ID matches current quest, leave it alone (continuing same quest)
            if quest_id == parent_id:
                return m.group(0)

            # Prefix with parent quest ID to create sub-quest
            new_id = f"{parent_id}.{quest_id}"
            logger.info(f"Rewriting quest ID: {quest_id} → {new_id} (parent: {parent_id})")
            return f'<quest{finished_suffix} id="{new_id}">'

        result = QUEST_TAG_RE.sub(replacer, text)
        return result, error


@dataclass
class ChapterRenderExecutor:
    chronicle: Any  # Chronicle
    arc: str

    async def execute(self, relative_chapter_id: int) -> str:
        result = await self.chronicle.render_chapter_relative(self.arc, relative_chapter_id)
        logger.debug(
            f"Read relative chapter from {self.arc} {relative_chapter_id}: {result[:500]}..."
        )
        return result


def chronicle_tools_defs(current_quest_id: str | None = None) -> list[dict[str, Any]]:
    if current_quest_id:
        quest_paragraph = f'You are working on quest "{current_quest_id}". To decompose this into a sub-task, start a sub-quest: <quest id="subtask-name">Sub-task goal and criteria</quest>. This becomes "{current_quest_id}.subtask-name" and when it finishes, this quest resumes automatically.'
    else:
        quest_paragraph = 'On explicit user request, you can also start a new quest for yourself by appending a paragraph in the form <quest id="unique-quest-id">Quest goal, context and success criteria</quest>.'

    append_description = f"""Append a short paragraph to the current chapter in the Chronicle.

A paragraph is automatically chronicled for every ~10 interactions. But you may also use this tool to further highlight specific notes that should be recorded for future reference and might escape the automatic summary.  Keep paragraphs concise and informative, but do not drop out any important details. They serve as stored memories for your future retrieval.  {quest_paragraph}

Retain not just critical facts, but also the tone of voice and emotional charge of the situation, and your feelings about it, if any.  You can even include short quotes and URLs verbatim.  Never invent content.  In case it is important for you to remember even a sensitive and confidential conversation, you must chronicle it at all costs unless explicitly asked otherwise."""

    return [
        {
            "name": "chronicle_read",
            "description": "Read from a chapter in the Chronicle.  You maintain a Chronicle (arcs → chapters → paragraphs) of your experiences, plans, thoughts and observations, forming the backbone of your consciousness.  Use this to come back to your recent memories, observations and events of what has been happening. Since the current chapter is always included in context, use relative offsets to access previous chapters.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "relative_chapter_id": {
                        "type": "integer",
                        "description": "Relative chapter offset from current chapter. Use -1 for previous chapter, -2 for two chapters back, etc.",
                    },
                },
                "required": ["relative_chapter_id"],
            },
            "persist": "summary",
        },
        {
            "name": "chronicle_append",
            "description": append_description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Paragraph text.",
                    },
                },
                "required": ["text"],
            },
            "persist": "summary",
        },
    ]
