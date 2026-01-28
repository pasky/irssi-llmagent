"""Auto-chronicling functionality for rooms."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .command import RoomCommandHandler

from ..chronicler.chapters import chapter_append_paragraph
from ..chronicler.tools import chronicle_tools_defs
from ..history import ChatHistory

logger = logging.getLogger(__name__)


class AutoChronicler:
    """Manages automatic chronicling of room messages when threshold is exceeded."""

    MAX_CHRONICLE_BATCH = 100
    MAX_LOOKBACK_DAYS = 7
    MESSAGE_OVERLAP = 5

    def __init__(
        self,
        history: ChatHistory,
        monitor: "RoomCommandHandler",
    ):
        """Initialize AutoChronicler.

        Args:
            history: ChatHistory instance for tracking messages
            monitor: RoomCommandHandler instance for running chronicling agent
        """
        self.monitor = monitor
        self._chronicling_locks = {}

    async def check_and_chronicle(
        self, mynick: str, server: str, channel: str, max_size: int
    ) -> bool:
        """Check if chronicling is needed and trigger if so."""
        arc = f"{server}#{channel}"

        if arc not in self._chronicling_locks:
            self._chronicling_locks[arc] = asyncio.Lock()

        async with self._chronicling_locks[arc]:
            unchronicled_count = await self.monitor.agent.history.count_recent_unchronicled(
                server, channel, days=self.MAX_LOOKBACK_DAYS
            )

            logger.debug("Unchronicled messages in %s: %s/%s", arc, unchronicled_count, max_size)

            if unchronicled_count < max_size:
                return False

            logger.debug(
                "Auto-chronicling triggered for %s: %s unchronicled messages",
                arc,
                unchronicled_count,
            )
            try:
                await self._auto_chronicle(
                    mynick, server, channel, arc, unchronicled_count + self.MESSAGE_OVERLAP
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error("Error during auto-chronicling for %s: %s", arc, str(e))
            return True

    async def _auto_chronicle(
        self, mynick: str, server: str, channel: str, arc: str, n_messages: int
    ) -> None:
        """Execute auto-chronicling for the given channel."""
        messages = await self.monitor.agent.history.get_full_history(
            server, channel, limit=n_messages
        )

        if not messages:
            logger.error("No unchronicled messages found for %s ??", arc)
            return

        message_ids = [msg["id"] for msg in messages]
        chapter_id = await self._run_chronicler(mynick, arc, messages)

        if chapter_id:
            await self.monitor.agent.history.mark_chronicled(message_ids, chapter_id)
            logger.debug(
                "Successfully chronicled %s messages to chapter %s", len(messages), chapter_id
            )
        else:
            logger.error("Chronicling failed for %s - no chapter_id returned", arc)

    async def _run_chronicler(
        self, mynick: str, arc: str, messages: list[dict[str, Any]]
    ) -> int | None:
        """Run chronicler to summarize and record messages to chronicle."""
        message_lines = [f"[{msg['timestamp'][:16]}] {msg['message']}" for msg in messages]
        messages_text = "\n".join(message_lines)

        chronicle_tools = chronicle_tools_defs()
        append_tool = next(t for t in chronicle_tools if t["name"] == "chronicle_append")
        system_prompt = append_tool["description"]

        context_messages = await self.monitor.agent.chronicle.get_chapter_context_messages(arc)
        user_prompt = (
            f"Review the following {len(messages)} recent IRC messages (your nick is {mynick}) "
            "and create a brief paragraph (extremely concise, 2-3 SHORT sentences max) with "
            "chronicle entry that captures key points you should remember in the future:\n\n"
            f"{messages_text}\n\nRespond only with the paragraph, no preamble."
        )
        context_messages.append({"role": "user", "content": user_prompt})

        chronicler_config = self.monitor.agent.config["chronicler"]
        chronicler_model = chronicler_config.get("arc_models", {}).get(
            arc, chronicler_config["model"]
        )
        resp, client, _, _ = await self.monitor.agent.model_router.call_raw_with_model(
            model_str=chronicler_model,
            context=context_messages,
            system_prompt=system_prompt,
            max_tokens=1024,
        )
        response = client.extract_text_from_response(resp)

        if response and response.strip():
            await chapter_append_paragraph(arc, response.strip(), self.monitor.agent)

            current_chapter = await self.monitor.agent.chronicle.get_or_open_current_chapter(arc)
            chapter_id = current_chapter["id"]

            logger.debug(
                "Chronicled %s messages for arc %s to chapter %s: %s",
                len(messages),
                arc,
                chapter_id,
                response,
            )
            return chapter_id

        logger.warning("Model %s for arc %s returned no response", chronicler_model, arc)
        return None
