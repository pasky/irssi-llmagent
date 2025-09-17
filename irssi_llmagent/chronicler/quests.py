"""Quests operator: reacts to <quest> paragraphs and advances them via AgenticLLMActor.

Minimal MVP that:
- Triggers on new <quest> paragraphs appended to the chronicle
- Runs a background step using IRCRoomMonitor._run_actor with serious mode
- Appends the next <quest> or <quest_finished> paragraph
- Mirrors the paragraph to IRC and ChatHistory
- On startup can scan current chapters for unresolved quests and trigger them

Configuration (config.json):
- chronicler.quests.arcs: list of arc names (e.g. ["server#channel"]) to operate in
- chronicler.quests.instructions: instruction string appended to system prompt
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


QUEST_OPEN_RE = re.compile(r"<\s*quest\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)
QUEST_FINISHED_RE = re.compile(r"<\s*quest_finished\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)


class QuestOperator:
    """Operator that advances quests for whitelisted arcs.

    Integration points:
    - chapters.chapter_append_paragraph should call on_chronicle_append()
    - On agent start, call scan_and_trigger_open_quests()
    """

    def __init__(self, agent: Any):
        self.agent = agent

        # Use dynamic config lookups so test-time config changes are picked up
        # (no local caches of arcs/instructions).

    def _parse_quest(self, text: str) -> tuple[str | None, bool]:
        """Return (quest_id, is_finished_tag) if paragraph contains a quest tag, else (None, False)."""
        m_finished = QUEST_FINISHED_RE.search(text)
        if m_finished:
            return m_finished.group(1), True
        m_open = QUEST_OPEN_RE.search(text)
        if m_open:
            return m_open.group(1), False
        return None, False

    async def on_chronicle_append(self, arc: str, paragraph_text: str) -> None:
        """Hook to be called after a paragraph is appended to the chronicle."""
        cfg = self.agent.config["chronicler"]["quests"]
        quest_id, is_finished = self._parse_quest(paragraph_text)
        if not quest_id or is_finished:
            return
        allowed_arcs = set(cfg["arcs"])
        if arc not in allowed_arcs:
            logger.debug(f"Quest {quest_id} not in allowed: {allowed_arcs}")
            return
        logger.debug(f"Quest {quest_id} triggered: {paragraph_text}")

        asyncio.create_task(self._run_step(arc, quest_id, paragraph_text))

    async def scan_and_trigger_open_quests(self) -> None:
        """Scan current chapters of allowed arcs for unresolved quests and trigger them."""
        cfg = self.agent.config["chronicler"]["quests"]
        for arc in set(cfg["arcs"]):
            current_chapter = await self.agent.chronicle.get_or_open_current_chapter(arc)
            chapter_id = int(current_chapter["id"])
            paragraphs = await self.agent.chronicle.read_chapter(chapter_id)
            latest_by_id: dict[str, tuple[str, bool]] = {}
            for p in paragraphs:
                qid, is_finished = self._parse_quest(p)
                if qid:
                    latest_by_id[qid] = (p, is_finished)
            for qid, (para, is_finished) in latest_by_id.items():
                if not is_finished:
                    logger.debug(f"Quest {qid} triggered: {para}")
                    asyncio.create_task(self._run_step(arc, qid, para))

    async def _run_step(self, arc: str, quest_id: str, paragraph_text: str) -> None:
        """Run one quest step via Agent.run_actor and handle results."""
        logger.debug(f"Quest step run_actor for {arc} {quest_id}: {paragraph_text}")

        cfg = self.agent.config["chronicler"]["quests"]
        await asyncio.sleep(float(cfg["cooldown"]))

        server, channel = arc.split("#", 1)

        mynick = await self.agent.irc_monitor.get_mynick(server)
        if not mynick:
            logger.warning(f"QuestsOperator: could not get mynick for server {server}")
            return

        # Build context: IRC chat history (same sizing as IRCMonitor serious mode), then the quest paragraph last
        irc_cfg = self.agent.irc_monitor.irc_config
        default_size = irc_cfg["command"]["history_size"]
        serious_size = (
            irc_cfg["command"]["modes"].get("serious", {}).get("history_size", default_size)
        )
        context = await self.agent.history.get_context(server, channel, serious_size)
        context = context + [{"role": "user", "content": paragraph_text}]

        mode_cfg = dict(irc_cfg["command"]["modes"]["serious"])
        mode_cfg["prompt_reminder"] = cfg["prompt_reminder"]

        system_prompt = self.agent.irc_monitor.build_system_prompt("serious", mynick)

        # Create progress callback that stores only tool_persistence updates
        async def progress_cb(text: str, type: str = "progress") -> None:
            if type == "tool_persistence":
                await self.agent.history.add_message(
                    server,
                    channel,
                    text,
                    mynick,
                    mynick,
                    False,
                    role="assistant_silent",
                )

        try:
            response = await self.agent.run_actor(
                context,
                mode_cfg=mode_cfg,
                system_prompt=system_prompt,
                arc=arc,
                progress_callback=progress_cb,
            )
        except Exception as e:
            logger.error(f"Quest step run_actor failed for {arc} {quest_id}: {e}")
            return

        if not response or response.startswith("Error - "):
            response = f"{paragraph_text}. Previous quest call failed ({response})."

        # Infer finish from content and normalize tags minimally
        is_finished = bool(re.search(r"\bCONFIRMED\s+ACHIEVED\b", response, re.IGNORECASE))
        if is_finished and "<quest_finished" not in response:
            if re.search(r"<\s*quest\b", response, re.IGNORECASE):
                # Upgrade quest → quest_finished
                response = re.sub(
                    r"<\s*quest\b", "<quest_finished", response, count=1, flags=re.IGNORECASE
                )
                response = re.sub(
                    r"</\s*quest\s*>", "</quest_finished>", response, count=1, flags=re.IGNORECASE
                )
            else:
                response = f"<quest_finished>{response}</quest_finished>"
        # Ensure quest tags have the correct id; wrap only if there is no <quest> tag at all
        if "<quest" not in response:
            response = f"<quest>{response}</quest>"

        def _ensure_id(m: re.Match[str]) -> str:
            suffix = m.group(1) or ""
            return f'<quest{suffix} id="{quest_id}">'

        response = re.sub(r'<quest(_finished)?(\s*id=".*?")?\s*>', _ensure_id, response)
        response = response.replace("\n", "; ").strip()

        # Mirror to IRC and ChatHistory
        logger.debug(f"Quest step run_actor for {arc} {quest_id} output: {response}")
        await self.agent.irc_monitor.varlink_sender.send_message(channel, response, server)
        await self.agent.history.add_message(server, channel, response, mynick, mynick, True)

        # Appending via chapter management triggers next quest step implicitly
        from .chapters import chapter_append_paragraph

        await chapter_append_paragraph(arc, response, self.agent)
