"""Discord room monitor for handling Discord-specific message processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import discord

from ...message_logging import MessageLoggingContext
from ...providers import parse_model_spec
from ...rate_limiter import RateLimiter

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


class DiscordRoomMonitor:
    """Discord-specific room monitor that handles Discord events and message processing."""

    def __init__(self, agent: MuaddibAgent):
        self.agent = agent
        self.discord_config = self.agent.config["rooms"]["discord"]

        command_config = self.discord_config.get("command", {})
        irc_command_config = self.agent.config["rooms"]["irc"]["command"]
        rate_limit = command_config.get("rate_limit", irc_command_config["rate_limit"])
        rate_period = command_config.get("rate_period", irc_command_config["rate_period"])
        self.history_size = command_config.get("history_size", irc_command_config["history_size"])

        self.rate_limiter = RateLimiter(rate_limit, rate_period)

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        self.client = DiscordClient(self, intents=intents)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "_".join(name.strip().split())

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

    async def process_message_event(self, message: discord.Message) -> None:
        """Process incoming Discord message events."""
        if message.author.bot:
            return

        content = message.clean_content or message.content or ""
        if not content:
            return

        if self.client.user is None:
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

        trigger_message_id = await self.agent.history.add_message(
            server_tag, channel_name, content, nick, mynick
        )

        if not self._is_highlight(message):
            return

        with MessageLoggingContext(arc, nick, content, Path("logs")):
            await self.handle_highlight(
                message,
                server_tag,
                channel_name,
                arc,
                nick,
                mynick,
                trigger_message_id,
            )

    async def handle_highlight(
        self,
        message: discord.Message,
        server_tag: str,
        channel_name: str,
        arc: str,
        nick: str,
        mynick: str,
        trigger_message_id: int,
    ) -> None:
        """Handle Discord highlights and generate responses."""
        if not self.rate_limiter.check_limit():
            logger.warning("Rate limiting triggered for %s", nick)
            await message.reply(
                f"{nick}: Slow down a little, will you? (rate limiting)",
                mention_author=True,
            )
            return

        context = await self.agent.history.get_context(server_tag, channel_name, self.history_size)

        agent_result = await self.agent.irc_monitor._run_actor(
            context[-self.history_size :],
            mynick,
            mode="serious",
            reasoning_effort="low",
            arc=arc,
        )

        if agent_result and agent_result.text:
            response_text = agent_result.text
            cleaned_text = response_text.lstrip()
            prefix = f"{nick}:"
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix) :].lstrip()
            response_text = cleaned_text or response_text
            cost_str = f"${agent_result.total_cost:.4f}" if agent_result.total_cost else "?"
            logger.info("Sending serious response (%s) to %s: %s", cost_str, arc, response_text)

            llm_call_id = None
            if agent_result.primary_model:
                try:
                    spec = parse_model_spec(agent_result.primary_model)
                    llm_call_id = await self.agent.history.log_llm_call(
                        provider=spec.provider,
                        model=spec.name,
                        input_tokens=agent_result.total_input_tokens,
                        output_tokens=agent_result.total_output_tokens,
                        cost=agent_result.total_cost,
                        call_type="agent_run",
                        arc_name=arc,
                        trigger_message_id=trigger_message_id,
                    )
                except ValueError:
                    logger.warning("Could not parse model spec: %s", agent_result.primary_model)

            await message.reply(response_text, mention_author=True)
            response_message_id = await self.agent.history.add_message(
                server_tag,
                channel_name,
                response_text,
                mynick,
                mynick,
                True,
                mode="EASY_SERIOUS",
                llm_call_id=llm_call_id,
            )
            if llm_call_id:
                await self.agent.history.update_llm_call_response(llm_call_id, response_message_id)
        else:
            logger.info("Agent chose not to answer for %s", arc)

    async def run(self) -> None:
        """Run the main Discord monitor loop."""
        token = self.discord_config.get("token")
        if not token:
            logger.error("Discord token missing in config; skipping Discord monitor")
            return

        try:
            await self.client.start(token)
        finally:
            await self.client.close()
