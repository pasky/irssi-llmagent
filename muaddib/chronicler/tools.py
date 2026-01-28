"""Chronicle tools: Direct implementation of chronicle append and read tools."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .chapters import chapter_append_paragraph

logger = logging.getLogger(__name__)

QUEST_TAG_RE = re.compile(r"<\s*quest(_finished)?\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)
VALID_QUEST_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class ChapterAppendExecutor:
    agent: Any
    arc: str

    async def execute(self, text: str) -> str:
        logger.info(f"Appending to {self.arc} chapter: {text}")
        await chapter_append_paragraph(self.arc, text, self.agent)
        return "OK"


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


def _validate_quest_id(quest_id: str) -> str | None:
    """Validate quest ID format. Returns error message or None if valid."""
    if not quest_id:
        return "Quest ID cannot be empty"
    if len(quest_id) > 64:
        return "Quest ID too long (max 64 characters)"
    if "." in quest_id:
        return "Quest ID cannot contain dots (reserved for hierarchy)"
    if not VALID_QUEST_ID_RE.match(quest_id):
        return "Quest ID can only contain letters, numbers, hyphens, and underscores"
    return None


@dataclass
class QuestStartExecutor:
    """Start a new top-level quest."""

    agent: Any
    arc: str

    async def execute(self, id: str, goal: str, success_criteria: str) -> str:
        if err := _validate_quest_id(id):
            return f"Error: {err}"

        existing = await self.agent.chronicle.quest_get(id)
        if existing:
            return f"Error: Quest '{id}' already exists"

        paragraph = f'<quest id="{id}">Goal: {goal} | Success criteria: {success_criteria}</quest>'
        logger.info(paragraph)
        await chapter_append_paragraph(self.arc, paragraph, self.agent)
        return f"Quest started: {id}"


@dataclass
class SubquestStartExecutor:
    """Start a subquest under the current quest."""

    agent: Any
    arc: str
    parent_quest_id: str

    async def execute(self, id: str, goal: str, success_criteria: str) -> str:
        if err := _validate_quest_id(id):
            return f"Error: {err}"

        full_id = f"{self.parent_quest_id}.{id}"
        existing = await self.agent.chronicle.quest_get(full_id)
        if existing:
            return f"Error: Subquest '{full_id}' already exists"

        paragraph = (
            f'<quest id="{full_id}">Goal: {goal} | Success criteria: {success_criteria}</quest>'
        )
        logger.info(paragraph)
        await chapter_append_paragraph(self.arc, paragraph, self.agent)
        return f"Subquest started: {full_id}"


@dataclass
class QuestSnoozeExecutor:
    """Snooze the current quest until a specified time."""

    agent: Any
    quest_id: str

    async def execute(self, until: str) -> str:
        time_match = re.match(r"^(\d{1,2}):(\d{2})$", until.strip())
        if not time_match:
            return "Error: Invalid time format. Use HH:MM (e.g., 14:30)"

        hour, minute = int(time_match.group(1)), int(time_match.group(2))
        if hour > 23 or minute > 59:
            return "Error: Invalid time. Hours must be 0-23, minutes 0-59"

        now = datetime.now()
        resume_local = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if resume_local <= now:
            resume_local += timedelta(days=1)

        resume_at = resume_local.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        success = await self.agent.chronicle.quest_set_resume_at(self.quest_id, resume_at)
        if not success:
            return f"Error: Quest '{self.quest_id}' not found"

        return f"Quest snoozed until {resume_at}"


def chronicle_tools_defs() -> list[dict[str, Any]]:
    append_description = """Append a short paragraph to the current chapter in the Chronicle.

A paragraph is automatically chronicled for every ~10 interactions. But you may also use this tool to further highlight specific notes that should be recorded for future reference and might escape the automatic summary. Keep paragraphs concise and informative, but do not drop out any important details. They serve as stored memories for your future retrieval.

Retain not just critical facts, but also the tone of voice and emotional charge of the situation, and your feelings about it, if any. You can even include short quotes and URLs verbatim. Never invent content. In case it is important for you to remember even a sensitive and confidential conversation, you must chronicle it at all costs unless explicitly asked otherwise."""

    return [
        {
            "name": "chronicle_read",
            "description": "Read from a chapter in the Chronicle. You maintain a Chronicle (arcs → chapters → paragraphs) of your experiences, plans, thoughts and observations, forming the backbone of your consciousness. Use this to come back to your recent memories, observations and events of what has been happening. Since the current chapter is always included in context, use relative offsets to access previous chapters.",
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


def quest_tools_defs(current_quest_id: str | None = None) -> list[dict[str, Any]]:
    """Return quest tools based on current context.

    - quest_start: only when no active quest (starting a top-level quest)
    - subquest_start: only when inside a top-level quest (no dots in ID)
    - quest_snooze: only when inside a quest

    Quest finish is handled via "CONFIRMED ACHIEVED" phrase detection at final_answer time.
    """
    tools: list[dict[str, Any]] = []

    if current_quest_id is None:
        tools.append(
            {
                "name": "quest_start",
                "description": "Start a new quest for yourself. Only use on explicit user request for a multi-step autonomous task. The quest system will periodically advance the quest until success criteria are met. MUST be called alongside final_answer in the same turn.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique quest identifier (letters, numbers, hyphens, underscores only). Example: 'check-xmas25-news-tuesday'",
                        },
                        "goal": {
                            "type": "string",
                            "description": "Clear description of what the quest should accomplish.",
                        },
                        "success_criteria": {
                            "type": "string",
                            "description": "Specific, measurable criteria for when the quest is complete.",
                        },
                    },
                    "required": ["id", "goal", "success_criteria"],
                },
                "persist": "summary",
            }
        )
    elif "." not in current_quest_id:
        # Only allow subquest_start for top-level quests (no dots in ID)
        tools.append(
            {
                "name": "subquest_start",
                "description": f'Start a subquest to fully focus on a particular task of the current quest "{current_quest_id}". When the subquest finishes, the parent quest resumes. BEFORE starting subquests, call make_plan to outline your approach - the plan will be included in context for all future quest steps and can be updated via subsequent make_plan calls. If starting multiple subquests, do not call this tool in parallel for subquests that depend on each other. MUST be called alongside final_answer in the same turn.',
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Subquest identifier (letters, numbers, hyphens, underscores only). Will be prefixed with parent quest ID.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "Clear description of what this subquest should accomplish.",
                        },
                        "success_criteria": {
                            "type": "string",
                            "description": "Specific criteria for when this subquest is complete.",
                        },
                    },
                    "required": ["id", "goal", "success_criteria"],
                },
                "persist": "summary",
            }
        )

    if current_quest_id is not None:
        tools.append(
            {
                "name": "quest_snooze",
                "description": f'Snooze the current quest "{current_quest_id}" until a specified time. MUST be called alongside final_answer in the same turn - you will be pinged to resume the quest at the specified time.',
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "until": {
                            "type": "string",
                            "description": "Time to resume the quest in HH:MM format (24-hour). If the time is in the past today, it will be interpreted as tomorrow.",
                        },
                    },
                    "required": ["until"],
                },
                "persist": "summary",
            }
        )

    return tools
