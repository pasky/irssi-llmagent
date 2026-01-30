"""Quests operator: reacts to <quest> paragraphs and advances them via AgenticLLMActor.

Minimal MVP that:
- Triggers on new <quest> paragraphs appended to the chronicle
- Runs a background step using IRCRoomMonitor._run_actor with serious mode
- Appends the next <quest> or <quest_finished> paragraph
- Mirrors the paragraph to IRC and ChatHistory
- Heartbeat mechanism to prod ongoing quests periodically

Configuration (config.json):
- chronicler.quests.arcs: list of arc names (e.g. ["server#channel"]) to operate in
- chronicler.quests.instructions: instruction string appended to system prompt
- chronicler.quests.cooldown: seconds between quest steps (also used for heartbeat)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

import muaddib

from ..message_logging import MessageLoggingContext
from . import QuestStatus

logger = logging.getLogger(__name__)


QUEST_OPEN_RE = re.compile(r"<\s*quest\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)
QUEST_FINISHED_RE = re.compile(r"<\s*quest_finished\s+id=\"([^\"]+)\"\s*>", re.IGNORECASE)


class QuestOperator:
    """Operator that advances quests for whitelisted arcs.

    Integration points:
    - chapters.chapter_append_paragraph should call on_chronicle_append()
    - On agent start, call start_heartbeat() to begin periodic quest prodding
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self._heartbeat_task: asyncio.Task[None] | None = None

    def _parse_quest(self, text: str) -> tuple[str | None, bool]:
        """Return (quest_id, is_finished_tag) if paragraph contains a quest tag, else (None, False)."""
        m_finished = QUEST_FINISHED_RE.search(text)
        if m_finished:
            return m_finished.group(1), True
        m_open = QUEST_OPEN_RE.search(text)
        if m_open:
            return m_open.group(1), False
        return None, False

    def _extract_parent_id(self, quest_id: str) -> str | None:
        """Extract parent ID from dot-separated quest ID."""
        if "." not in quest_id:
            return None
        return quest_id.rsplit(".", 1)[0]

    async def on_chronicle_append(self, arc: str, paragraph_text: str, paragraph_id: int) -> None:
        """Hook to be called after a paragraph is appended to the chronicle.

        Updates the quests table and triggers quest steps as needed.
        """
        cfg = self.agent.config["chronicler"]["quests"]
        quest_id, is_finished = self._parse_quest(paragraph_text)
        if not quest_id:
            return
        allowed_arcs = set(cfg["arcs"])
        if arc not in allowed_arcs:
            logger.debug(f"Quest {quest_id} not in allowed: {allowed_arcs}")
            return

        # Check if quest exists in DB
        existing = await self.agent.chronicle.quest_get(quest_id)

        if existing:
            if is_finished:
                await self.agent.chronicle.quest_finish(quest_id, paragraph_id)
                logger.debug(f"Quest {quest_id} finished")
            else:
                await self.agent.chronicle.quest_update(quest_id, paragraph_text, paragraph_id)
                logger.debug(f"Quest {quest_id} updated, will continue on next heartbeat")
        else:
            assert not is_finished
            parent_id = self._extract_parent_id(quest_id)
            await self.agent.chronicle.quest_start(
                quest_id, arc, paragraph_id, paragraph_text, parent_id
            )
            logger.debug(f"Quest {quest_id} created, will start on next heartbeat")

    async def start_heartbeat(self) -> None:
        """Start the heartbeat task that periodically prods ongoing quests."""
        if self._heartbeat_task is not None:
            return
        self._heartbeat_task = muaddib.spawn(self._heartbeat_loop())
        logger.info("Quest heartbeat started")

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat task."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
            logger.info("Quest heartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        """Periodically check for quests that need prodding."""
        cfg = self.agent.config["chronicler"]["quests"]
        cooldown = float(cfg["cooldown"])

        while True:
            await asyncio.sleep(cooldown)
            try:
                await self._heartbeat_tick()
            except Exception as e:
                logger.error(f"Heartbeat tick failed: {e}")

    async def _heartbeat_tick(self) -> None:
        """Single heartbeat tick: find and prod ready quests."""
        cfg = self.agent.config["chronicler"]["quests"]
        cooldown = float(cfg["cooldown"])

        for arc in set(cfg["arcs"]):
            ready_quests = await self.agent.chronicle.quests_ready_for_heartbeat(arc, cooldown)
            for quest in ready_quests:
                quest_id = quest["id"]
                last_state = quest["last_state"]
                logger.info(f"Heartbeat prodding quest {quest_id}")
                muaddib.spawn(self._run_step(arc, quest_id, last_state))

    async def _run_step(self, arc: str, quest_id: str, paragraph_text: str) -> None:
        """Run one quest step via Agent.run_actor and handle results."""
        # Atomically claim the quest (ONGOING -> IN_STEP)
        if not await self.agent.chronicle.quest_try_transition(
            quest_id, QuestStatus.ONGOING, QuestStatus.IN_STEP
        ):
            return  # Already claimed by another runner

        try:
            with MessageLoggingContext(arc, f"quest-{quest_id}", paragraph_text):
                await self._run_step_inner(arc, quest_id, paragraph_text)
        finally:
            # Atomically release (IN_STEP -> ONGOING) if not finished during step
            await self.agent.chronicle.quest_try_transition(
                quest_id, QuestStatus.IN_STEP, QuestStatus.ONGOING
            )

    async def _run_step_inner(self, arc: str, quest_id: str, paragraph_text: str) -> None:
        """Inner quest step logic, called within logging context."""
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

        quest = await self.agent.chronicle.quest_get(quest_id)
        if quest and quest.get("plan"):
            context += [{"role": "user", "content": f"<meta><plan>{quest['plan']}</plan></meta>"}]

        context += [{"role": "user", "content": paragraph_text}]

        mode_cfg = dict(irc_cfg["command"]["modes"]["serious"])
        mode_cfg["prompt_reminder"] = cfg["prompt_reminder"].replace(
            "<quest>", f"<quest> {quest_id}"
        )

        system_prompt = self.agent.irc_monitor.build_system_prompt("serious", mynick)

        # Create persistence callback (no progress callback - quests don't send IRC updates)
        async def persistence_cb(text: str) -> None:
            from ..rooms.message import RoomMessage

            msg = RoomMessage(
                server_tag=server,
                channel_name=channel,
                nick=mynick,
                mynick=mynick,
                content=text,
            )
            await self.agent.history.add_message(msg, role="assistant_silent")

        try:
            agent_result = await self.agent.run_actor(
                context,
                mode_cfg=mode_cfg,
                system_prompt=system_prompt,
                arc=arc,
                persistence_callback=persistence_cb,
                current_quest_id=quest_id,
            )
        except Exception as e:
            logger.error(f"Quest step run_actor failed for {arc} {quest_id}: {e}")
            return

        response = agent_result.text if agent_result else None
        if not response or response.startswith("Error: "):
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

        # Mirror full response to IRC and ChatHistory
        logger.debug(f"Quest step run_actor for {arc} {quest_id} output: {response}")
        await self.agent.irc_monitor.varlink_sender.send_message(channel, response, server)

        from ..rooms.message import RoomMessage

        msg = RoomMessage(
            server_tag=server,
            channel_name=channel,
            nick=mynick,
            mynick=mynick,
            content=response,
        )
        await self.agent.history.add_message(msg, mode="THINKING_SERIOUS")

        # Append only quest XML to chronicle (triggers next quest step implicitly)
        from .chapters import chapter_append_paragraph

        quest_match = re.search(
            r"<quest(?:_finished)?[^>]*>.*?</quest(?:_finished)?>", response, re.DOTALL
        )
        quest_text = quest_match.group(0) if quest_match else response
        await chapter_append_paragraph(arc, quest_text, self.agent)
