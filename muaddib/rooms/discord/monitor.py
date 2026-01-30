"""Discord room monitor for handling Discord-specific message processing."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

import discord

from ...message_logging import MessageLoggingContext
from ..command import RoomCommandHandler, get_room_config
from ..message import RoomMessage

if TYPE_CHECKING:
    from ...main import MuaddibAgent

logger = logging.getLogger(__name__)


class DiscordClient(discord.Client):
    """Discord client that forwards events to the room monitor."""

    def __init__(self, monitor: DiscordRoomMonitor, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.monitor = monitor

    async def on_ready(self) -> None:
        logger.info("Discord client connected as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        await self.monitor.process_message_event(message)

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        await self.monitor.process_message_edit(before, after)


class DiscordRoomMonitor:
    """Discord-specific room monitor that handles Discord events and message processing."""

    def __init__(self, agent: MuaddibAgent) -> None:
        self.agent = agent
        self.room_config = get_room_config(self.agent.config, "discord")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        self.client = DiscordClient(self, intents=intents)

        self.reply_edit_debounce_seconds = float(
            self.room_config.get("reply_edit_debounce_seconds", 15.0)
        )

        self.command_handler = RoomCommandHandler(
            self.agent,
            "discord",
            self.room_config,
            response_cleaner=self._strip_nick_prefix,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "_".join(name.strip().split())

    @staticmethod
    def _strip_nick_prefix(text: str, nick: str) -> str:
        cleaned_text = text.lstrip()
        prefix = f"{nick}:"
        if cleaned_text.lower().startswith(prefix.lower()):
            cleaned_text = cleaned_text[len(prefix) :].lstrip()
        return cleaned_text or text

    @staticmethod
    def _normalize_content(content: str) -> str:
        if not content:
            return content
        return re.sub(r"<a?:([0-9A-Za-z_]+):\d+>", r":\1:", content)

    def _get_channel_name(self, channel: discord.abc.Messageable) -> str:
        if isinstance(channel, discord.Thread) and channel.parent:
            return self._normalize_name(channel.parent.name)
        if isinstance(channel, discord.abc.GuildChannel):
            return self._normalize_name(channel.name)
        if isinstance(channel, discord.abc.PrivateChannel):
            return "dm"
        return "dm"

    def _get_server_tag(self, message: discord.Message) -> str:
        if message.guild:
            return f"discord:{message.guild.name}"
        return "discord:_DM"

    def _is_highlight(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True
        if self.client.user is None:
            return False
        return self.client.user in message.mentions

    def _strip_leading_mention(self, message: discord.Message, mynick: str) -> str:
        content = message.clean_content or message.content or ""
        if not content:
            return content

        if self.client.user is None:
            return self._normalize_content(content)

        mention_pattern = rf"^\s*(?:<@!?{self.client.user.id}>|{re.escape(mynick)})[:,]?\s*(.*)$"
        match = re.match(mention_pattern, message.content)
        if match:
            return self._normalize_content(match.group(1).strip())

        clean_pattern = rf"^\s*{re.escape(mynick)}[:,]?\s*(.*)$"
        match = re.match(clean_pattern, content)
        if match:
            return self._normalize_content(match.group(1).strip())

        return self._normalize_content(content)

    def _now(self) -> float:
        return time.monotonic()

    async def process_message_event(self, message: discord.Message) -> None:
        """Process incoming Discord message events."""

        logger.debug("Processing message event: %s", message)

        content = message.clean_content or message.content or ""
        content = self._normalize_content(content)

        if message.attachments:
            attachment_lines: list[str] = []
            for i, att in enumerate(message.attachments, start=1):
                meta = att.content_type or "attachment"
                if att.filename:
                    meta += f" (filename: {att.filename})"
                if getattr(att, "size", None):
                    meta += f" (size: {att.size})"
                attachment_lines.append(f"{i}. {meta}: {att.url}")

            attachments_block = "\n".join(["[Attachments]", *attachment_lines, "[/Attachments]"])
            content = f"{content}\n\n{attachments_block}" if content else attachments_block

        if not content:
            logger.debug(f"No content in message {message}")
            return

        if self.client.user is None:
            logger.debug(f"No user set in client {self.client} for message {message}")
            return

        if message.author.id == self.client.user.id:
            logger.debug("Ignoring message from self: %s", message)
            return

        server_tag = self._get_server_tag(message)
        if message.guild is None:
            normalized_name = self._normalize_name(message.author.display_name)
            channel_name = f"{normalized_name}_{message.author.id}"
        else:
            channel_name = self._get_channel_name(message.channel)
        arc = f"{server_tag}#{channel_name}"
        nick = message.author.display_name
        mynick = self.client.user.display_name

        if self.command_handler.should_ignore_user(nick):
            logger.debug("Ignoring user: %s", nick)
            return

        platform_id = str(message.id) if getattr(message, "id", None) is not None else None
        thread_id: str | None = None
        thread_starter_id: int | None = None
        if isinstance(message.channel, discord.Thread):
            thread_id = str(message.channel.id)
            thread_starter_id = await self.agent.history.get_message_id_by_platform_id(
                server_tag, channel_name, thread_id
            )

        last_reply: discord.Message | None = None
        last_reply_time: float | None = None

        async def reply_sender(text: str) -> None:
            nonlocal last_reply, last_reply_time
            now = self._now()
            if (
                last_reply is not None
                and last_reply_time is not None
                and now - last_reply_time < self.reply_edit_debounce_seconds
            ):
                existing_content = getattr(last_reply, "content", "") or ""
                combined = f"{existing_content}\n{text}" if existing_content else text
                sent_message = await last_reply.edit(content=combined)
                if sent_message is not None:
                    last_reply = sent_message
                    last_reply_time = now
                return

            reply_target = last_reply or message
            if reply_target is not None and hasattr(reply_target, "reply"):
                mention_author = last_reply is None
                sent_message = await reply_target.reply(text, mention_author=mention_author)
                if sent_message is not None:
                    last_reply = sent_message
                    last_reply_time = now
                return
            logger.warning("Missing reply context for Discord send to %s", channel_name)

        if self._is_highlight(message):
            cleaned_content = self._strip_leading_mention(message, mynick)
            msg = RoomMessage(
                server_tag=server_tag,
                channel_name=channel_name,
                nick=nick,
                mynick=mynick,
                content=cleaned_content,
                platform_id=platform_id,
                thread_id=thread_id,
                thread_starter_id=thread_starter_id,
            )
            trigger_message_id = await self.agent.history.add_message(msg)
            with MessageLoggingContext(arc, nick, content):
                async with message.channel.typing():
                    await self.command_handler.handle_command(msg, trigger_message_id, reply_sender)
            return

        msg = RoomMessage(
            server_tag=server_tag,
            channel_name=channel_name,
            nick=nick,
            mynick=mynick,
            content=content,
            platform_id=platform_id,
            thread_id=thread_id,
            thread_starter_id=thread_starter_id,
        )
        trigger_message_id = await self.agent.history.add_message(msg)
        await self.command_handler.handle_passive_message(msg, reply_sender)

    async def process_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        """Process incoming Discord message edit events."""
        if self.client.user is None:
            return

        if after.author.id == self.client.user.id:
            logger.debug("Ignoring edit from self: %s", after.id)
            return

        platform_id = str(after.id) if getattr(after, "id", None) is not None else None
        if not platform_id:
            return

        new_content = after.clean_content or after.content or ""
        new_content = self._normalize_content(new_content)
        if not new_content:
            return

        server_tag = self._get_server_tag(after)
        if after.guild is None:
            normalized_name = self._normalize_name(after.author.display_name)
            channel_name = f"{normalized_name}_{after.author.id}"
        else:
            channel_name = self._get_channel_name(after.channel)

        nick = after.author.display_name

        updated = await self.agent.history.update_message_by_platform_id(
            server_tag, channel_name, platform_id, new_content, nick
        )
        if updated:
            logger.info("Updated edited message %s in %s#%s", platform_id, server_tag, channel_name)

    async def run(self) -> None:
        """Run the main Discord monitor loop."""
        token = self.room_config.get("token")
        if not token:
            logger.error("Discord token missing in config; skipping Discord monitor")
            return

        try:
            await self.client.start(token)
        finally:
            await self.command_handler.proactive_debouncer.cancel_all()
            await self.client.close()
