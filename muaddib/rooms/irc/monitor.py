"""IRC room monitor for handling IRC-specific message processing."""

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

import muaddib

from ...message_logging import MessageLoggingContext
from ..command import RoomCommandHandler, get_room_config
from ..message import RoomMessage
from .varlink import VarlinkClient, VarlinkSender

if TYPE_CHECKING:
    from ...main import MuaddibAgent

logger = logging.getLogger(__name__)


class IRCRoomMonitor:
    """IRC-specific room monitor that handles varlink connections and message processing."""

    def __init__(self, agent: "MuaddibAgent") -> None:
        self.agent = agent
        self.room_config = get_room_config(self.agent.config, "irc")

        self.varlink_events = VarlinkClient(self.room_config["varlink"]["socket_path"])
        self.varlink_sender = VarlinkSender(self.room_config["varlink"]["socket_path"])
        self.command_handler = RoomCommandHandler(
            self.agent,
            "irc",
            self.room_config,
            response_cleaner=self._normalize_reply,
        )
        self.server_nicks: dict[str, str] = {}

    @staticmethod
    def _normalize_reply(text: str, nick: str) -> str:
        return text.replace("\n", "; ").strip() or text

    @property
    def irc_config(self) -> dict[str, Any]:
        """Expose merged IRC config for legacy callers."""
        return self.room_config

    async def get_mynick(self, server: str) -> str | None:
        """Get bot's nick for a server."""
        if server not in self.server_nicks:
            try:
                nick = await self.varlink_sender.get_server_nick(server)
                if nick:
                    self.server_nicks[server] = nick
                    logger.debug("Got nick for %s: %s", server, nick)
                return nick
            except Exception as e:
                logger.error("Failed to get nick for server %s: %s", server, e)
                return None
        return self.server_nicks[server]

    def build_system_prompt(self, mode: str, mynick: str, model_override: str | None = None) -> str:
        """Compatibility wrapper for shared system prompt builder."""
        return self.command_handler.build_system_prompt(mode, mynick, model_override)

    def _input_match(self, mynick: str, message: str) -> re.Match | None:
        pattern = rf"^\s*(<?.*?>\s*)?{re.escape(mynick)}[,:]\s*(.*?)$"
        return re.match(pattern, message, re.IGNORECASE)

    async def process_message_event(self, event: dict[str, Any]) -> None:
        """Process incoming IRC message events."""
        msg_type = event.get("type")
        subtype = event.get("subtype")
        server = event.get("server")
        target = event.get("target")
        nick = event.get("nick")
        message = event.get("message", "")

        logger.debug("Processing message event: %s", event)

        if msg_type != "message" or not all([server, target, nick, message]):
            logger.debug("Skipping invalid message event")
            return

        assert isinstance(server, str)
        assert isinstance(target, str)
        assert isinstance(nick, str)
        assert isinstance(message, str)

        chan_name = target if subtype == "public" else nick

        mynick = await self.get_mynick(server)
        if not mynick:
            return

        if self.command_handler.should_ignore_user(nick):
            logger.debug("Ignoring user: %s", nick)
            return

        match = self._input_match(mynick, message)
        is_private = subtype != "public"
        is_direct = bool(match) or is_private

        if match and match.group(1):
            nick = match.group(1).strip("<> ")

        cleaned_msg = match.group(2) if match else message

        async def reply_sender(text: str) -> None:
            await self.varlink_sender.send_message(chan_name, text, server)

        msg = RoomMessage(
            server_tag=server,
            channel_name=chan_name,
            nick=nick,
            mynick=mynick,
            content=cleaned_msg if is_direct else message,
        )

        trigger_message_id = await self.agent.history.add_message(msg)

        if is_direct:
            with MessageLoggingContext(msg.arc, nick, message):
                await self.command_handler.handle_command(msg, trigger_message_id, reply_sender)
            return

        await self.command_handler.handle_passive_message(msg, reply_sender)

    async def _connect_with_retry(self, max_retries: int = 5) -> bool:
        """Connect to varlink sockets with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                await self.varlink_events.connect()
                await self.varlink_sender.connect()
                await self.varlink_events.wait_for_events()
                logger.info("Successfully connected to varlink sockets")
                return True
            except Exception as e:
                wait_time = 2**attempt
                logger.warning("Connection attempt %s failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    logger.info("Retrying in %s seconds...", wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Failed to connect after %s attempts", max_retries)
                    return False
        return False

    async def run(self) -> None:
        """Run the main IRC monitor loop."""
        try:
            if not await self._connect_with_retry():
                logger.error("Could not establish connection, exiting")
                return

            logger.info("Muaddib started, waiting for IRC events...")

            while True:
                try:
                    response = await self.varlink_events.receive_response()
                    if response is None:
                        logger.warning("Connection lost, attempting to reconnect...")
                        await self.varlink_events.disconnect()
                        await self.varlink_sender.disconnect()

                        if await self._connect_with_retry():
                            logger.info("Reconnected successfully")
                            continue
                        logger.error("Failed to reconnect, exiting...")
                        break

                    if "parameters" in response and "event" in response["parameters"]:
                        event = response["parameters"]["event"]
                        muaddib.spawn(self.process_message_event(event))
                    elif "error" in response:
                        logger.error("Varlink error: %s", response["error"])
                        break
                except Exception as e:
                    logger.error("Error in main loop: %s", e)
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            logger.info("Disconnecting from varlink sockets...")
            await self.varlink_events.disconnect()
            await self.varlink_sender.disconnect()
            await self.command_handler.proactive_debouncer.cancel_all()
            logger.info("IRC monitor stopped")
