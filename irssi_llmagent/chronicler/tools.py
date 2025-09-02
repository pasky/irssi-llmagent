"""Chronicle tools: Direct implementation of chronicle append and read tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChapterAppendExecutor:
    chronicle: Any  # Chronicle
    arc: str

    async def execute(self, text: str) -> str:
        logger.info(f"Appending to {self.arc} chapter: text")
        await self.chronicle.append_paragraph(self.arc, text)
        return "OK"


@dataclass
class ChapterRenderExecutor:
    chronicle: Any  # Chronicle
    arc: str

    async def execute(self, chapter_id: int | None = None) -> str:
        result = await self.chronicle.render_chapter(self.arc, chapter_id=chapter_id)
        logger.debug(f"Read chapter from {self.arc} {chapter_id}: {result[:500]}...")
        return result


def chronicle_tools_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "chronicle_append",
            "description": """Append a short paragraph to the current chapter in the Chronicle.

You maintain a Chronicle (arcs → chapters → paragraphs) of your experiences, plans, thoughts and observations, forming the backbone of your consciousness.

Use when important events happen that should be recorded for future reference.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Paragraph text.  Keep paragraphs concise and informative, but do not drop out any important details. They serve as stored memories for your future retrieval.  Never invent content.  Retain not just critical facts, but also the tone of voice and emotional charge of the situation, and your feelings about it, if any.",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "chronicle_read",
            "description": "Read from a chapter in the Chronicle.  Use this to come back to your recent memories, observations and events of what has been happenning.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "chapter_id": {
                        "type": "integer",
                        "description": "By default, read from the currently open chapter.",
                    },
                },
            },
        },
    ]
