"""Chronicle tools: Direct implementation of chronicle append and read tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .chapters import chapter_append_paragraph

logger = logging.getLogger(__name__)


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


def chronicle_tools_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "chronicle_append",
            "description": """Append a short paragraph to the current chapter in the Chronicle.

You maintain a Chronicle (arcs → chapters → paragraphs) of your experiences, plans, thoughts and observations, forming the backbone of your consciousness.  Use this tool when important events happen that should be recorded for future reference.  Keep paragraphs concise and informative, but do not drop out any important details. They serve as stored memories for your future retrieval.  On explicit user request, you can also start a new quest for yourself by appending a paragraph in the form <quest id="unique-quest-id">Quest goal, context and success criteria</quest>.

Retain not just critical facts, but also the tone of voice and emotional charge of the situation, and your feelings about it, if any.  You can even include short quotes and URLs verbatim.  Never invent content.  In case it is important for you to remember even a sensitive and confidential conversation, you must chronicle it at all costs unless explicitly asked otherwise.""",
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
        {
            "name": "chronicle_read",
            "description": "Read from a chapter in the Chronicle. Use this to come back to your recent memories, observations and events of what has been happening. Since the current chapter is always included in context, use relative offsets to access previous chapters.",
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
    ]
