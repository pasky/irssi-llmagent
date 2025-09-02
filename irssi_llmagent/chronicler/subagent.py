"""Chronicler subagent: NLI interface backed by Chronicle storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..agentic_actor import AgenticLLMActor

if TYPE_CHECKING:
    from ..main import IRSSILLMAgent


@dataclass
class ChapterAppendExecutor:
    chronicle: Any  # Chronicle
    arc: str

    async def execute(self, arc: str, text: str) -> str:
        if arc != self.arc:
            raise ValueError(f"arc mismatch: expected '{self.arc}', got '{arc}'")
        await self.chronicle.append_paragraph(self.arc, text)
        return "OK"


@dataclass
class ChapterRenderExecutor:
    chronicle: Any  # Chronicle
    arc: str

    async def execute(
        self, arc: str, chapter_id: int | None = None, last_n: int | None = None
    ) -> str:
        if arc != self.arc:
            raise ValueError(f"arc mismatch: expected '{self.arc}', got '{arc}'")
        return await self.chronicle.render_chapter(self.arc, chapter_id=chapter_id, last_n=last_n)


def _chronicler_tools_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "chapter_append",
            "description": "Append a short paragraph to the current chapter in the given arc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arc": {"type": "string", "description": "Arc name (must match)."},
                    "text": {"type": "string", "description": "Paragraph text."},
                },
                "required": ["arc", "text"],
            },
        },
        {
            "name": "chapter_render",
            "description": "Render the specified or current chapter in the given arc as markdown.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arc": {"type": "string", "description": "Arc name (must match)."},
                    "chapter_id": {"type": "integer"},
                    "last_n": {"type": "integer"},
                },
                "required": ["arc"],
            },
        },
    ]


def _system_prompt_for_arc(arc: str) -> str:
    return (
        "You are the Chronicler. You maintain a Chronicle (arcs → chapters → paragraphs) of an AI agent experiences, plans, thoughts and observations, forming the backbone of its consciousness.\n\n"
        "Rules:\n"
        f"- You MUST operate only within the given arc: <arc>{arc}</arc>.\n"
        "- A chapter is a sequence of short paragraphs. If the arc has no open chapter, open one implicitly.\n"
        "- Keep paragraphs concise and informative, but do not drop out any important details. They serve as stored memories for your future retrieval.\n"
        "- When asked to ‘show’/‘render’, use chapter_render and then read back its output, by default summarize it, but allow explicit requests for more literal access.\n"
        "- Never invent content; if unclear, request a brief clarification.\n"
        "- Output should be terse and operational. You are an AI subagent - your output will be read by another AI, not a human.\n"
        f'- IMPORTANT: When calling tools, always use arc="{arc}" (as a plain string, not JSON).\n'
        "\nAvailable tools:\n"
        "- chapter_append(arc, text): Append a paragraph to the current chapter.\n"
        "- chapter_render(arc, chapter_id?, last_n?): Render the specified or current chapter as markdown.\n"
    )


async def run_chronicler(agent: IRSSILLMAgent, *, arc: str, instructions: str) -> str:
    # Prepare tool defs and executors
    tool_defs = _chronicler_tools_defs()
    tool_executors = {
        "chapter_append": ChapterAppendExecutor(chronicle=agent.chronicle, arc=arc),
        "chapter_render": ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc),
    }

    # Get chronicler model from config
    chronicler_model = agent.config["chronicler"]["model"]

    actor = AgenticLLMActor(
        config=agent.config,
        model=chronicler_model,
        system_prompt_generator=lambda: _system_prompt_for_arc(arc),
        reasoning_effort="minimal",
        allowed_tools=["chapter_append", "chapter_render", "final_answer"],
        additional_tools=tool_defs,
        additional_tool_executors=tool_executors,
        agent=agent,
    )

    async with actor:
        result = await actor.run_agent([{"role": "user", "content": instructions}], arc=arc)
        return result
