"""Auto-chronicling functionality for IRC rooms."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .monitor import IRCRoomMonitor

from ...chronicler.chapters import chapter_append_paragraph
from ...chronicler.tools import chronicle_tools_defs
from ...history import ChatHistory

logger = logging.getLogger(__name__)


class AutoChronicler:
    """Manages automatic chronicling of IRC messages when threshold is exceeded."""

    # Safety limits to prevent chronicling massive message backlogs
    MAX_CHRONICLE_BATCH = 100  # Never chronicle more than 100 messages at once
    MAX_LOOKBACK_DAYS = 7  # Only look at last week of messages

    # Number of additional previously chronicled messages to smoothen the memory window
    MESSAGE_OVERLAP = 5

    def __init__(
        self,
        history: ChatHistory,
        monitor: "IRCRoomMonitor",
    ):
        """Initialize AutoChronicler.

        Args:
            history: ChatHistory instance for tracking messages
            monitor: IRCRoomMonitor instance for running chronicling agent
        """
        # Don't store history reference, use monitor.agent.history to allow mocking
        self.monitor = monitor
        self._chronicling_locks = {}  # Per-channel locks to prevent concurrent chronicling

    async def check_and_chronicle(
        self, mynick: str, server: str, channel: str, max_size: int
    ) -> bool:
        """Check if chronicling is needed and trigger if so.

        Args:
            server: IRC server name
            channel: IRC channel name
            max_size: Maximum number of unchronicled messages before triggering

        Returns:
            True if chronicling was triggered, False otherwise
        """
        arc = f"{server}#{channel}"

        # Use per-channel lock to prevent concurrent chronicling
        if arc not in self._chronicling_locks:
            self._chronicling_locks[arc] = asyncio.Lock()

        async with self._chronicling_locks[arc]:
            unchronicled_count = await self.monitor.agent.history.count_recent_unchronicled(
                server, channel, days=self.MAX_LOOKBACK_DAYS
            )

            logger.debug(f"Unchronicled messages in {arc}: {unchronicled_count}/{max_size}")

            if unchronicled_count < max_size:
                return False

            logger.debug(
                f"Auto-chronicling triggered for {arc}: {unchronicled_count} unchronicled messages"
            )
            try:
                await self._auto_chronicle(
                    mynick, server, channel, arc, unchronicled_count + self.MESSAGE_OVERLAP
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Error during auto-chronicling for {arc}: {str(e)}")
            return True

    async def _auto_chronicle(
        self, mynick: str, server: str, channel: str, arc: str, n_messages: int
    ) -> None:
        """Execute auto-chronicling for the given channel."""
        # Get unchronicled messages (limited for safety)
        messages = await self.monitor.agent.history.get_full_history(
            server, channel, limit=n_messages
        )

        if not messages:
            logger.error(f"No unchronicled messages found for {arc} ??")
            return

        # Run chronicler with message summary
        message_ids = [msg["id"] for msg in messages]
        chapter_id = await self._run_chronicler(mynick, arc, messages)

        if chapter_id:
            # Mark messages as chronicled
            await self.monitor.agent.history.mark_chronicled(message_ids, chapter_id)
            logger.debug(
                f"Successfully chronicled {len(messages)} messages to chapter {chapter_id}"
            )
        else:
            logger.error(f"Chronicling failed for {arc} - no chapter_id returned")

    async def _run_chronicler(
        self, mynick: str, arc: str, messages: list[dict[str, Any]]
    ) -> int | None:
        """Run chronicler to summarize and record messages to chronicle.

        Args:
            arc: Arc name for the chronicle
            messages: List of message dicts with id, nick, message, role, timestamp

        Returns:
            Chapter ID if successful, None if failed
        """
        # Create message summary for the model
        message_lines = []
        for msg in messages:
            message_lines.append(f"[{msg['timestamp'][:16]}] {msg['message']}")

        messages_text = "\n".join(message_lines)

        # Use the chronicle_append tool description directly to avoid duplication
        chronicle_tools = chronicle_tools_defs()
        assert chronicle_tools[0]["name"] == "chronicle_append"
        system_prompt = chronicle_tools[0]["description"]  # chronicle_append is first tool

        context_messages = await self.monitor.agent.chronicle.get_chapter_context_messages(arc)
        user_prompt = f"Review the following {len(messages)} recent IRC messages (your nick is {mynick}) and create a single paragraph with chronicle entry that captures what you should remember about it in the future:\n\n{messages_text}\n\nRespond only with the paragraph, no preamble."
        context_messages.append({"role": "user", "content": user_prompt})

        chronicler_config = self.monitor.agent.config["chronicler"]
        chronicler_model = chronicler_config.get("arc_models", {}).get(
            arc, chronicler_config["model"]
        )
        resp, client, _ = await self.monitor.agent.model_router.call_raw_with_model(
            model_str=chronicler_model,
            context=context_messages,
            system_prompt=system_prompt,
            max_tokens=1024,
        )
        response = client.extract_text_from_response(resp)

        if response and response.strip():
            # Append to chronicle with chapter management
            await chapter_append_paragraph(arc, response.strip(), self.monitor.agent)

            # Get the current chapter ID
            current_chapter = await self.monitor.agent.chronicle.get_or_open_current_chapter(arc)
            chapter_id = current_chapter["id"]

            logger.debug(
                f"Chronicled {len(messages)} messages for arc {arc} to chapter {chapter_id}: {response}"
            )
            return chapter_id
        else:
            logger.warning(f"Model {chronicler_model} for arc {arc} returned no response")

        return None
