"""Slack room monitor for handling Slack-specific message processing."""

from __future__ import annotations

import html
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from ...message_logging import MessageLoggingContext
from ..command import RoomCommandHandler, get_room_config
from ..message import RoomMessage

if TYPE_CHECKING:
    from ...main import MuaddibAgent

logger = logging.getLogger(__name__)

# Loading messages for typing indicator
TYPING_LOADING_MESSAGES = [
    "Thinking...",
    "Consulting the spice...",
    "The worm is turning...",
    "Prescience loading...",
]


class SlackRoomMonitor:
    """Slack-specific room monitor that handles Slack events and message processing."""

    def __init__(self, agent: MuaddibAgent) -> None:
        self.agent = agent
        self.room_config = get_room_config(self.agent.config, "slack")

        reply_start_thread = self.room_config.get("reply_start_thread", {})
        self.reply_start_thread = {
            "channel": bool(reply_start_thread.get("channel", True)),
            "dm": bool(reply_start_thread.get("dm", False)),
        }
        self.reply_edit_debounce_seconds = float(
            self.room_config.get("reply_edit_debounce_seconds", 15.0)
        )

        self.workspaces: dict[str, dict[str, str]] = self.room_config.get("workspaces", {})
        self.app_token = self.room_config.get("app_token")

        self.clients: dict[str, AsyncWebClient] = {}
        self.bot_user_ids: dict[str, str] = {}
        self.bot_display_names: dict[str, str] = {}
        # user_cache: team_id -> user_id -> display_name
        self.user_cache: dict[str, dict[str, str]] = {}
        # user_id_cache: team_id -> display_name (lowercase) -> user_id (reverse lookup)
        self.user_id_cache: dict[str, dict[str, str]] = {}
        self.channel_cache: dict[str, dict[str, str]] = {}

        default_token = None
        for workspace in self.workspaces.values():
            bot_token = workspace.get("bot_token") if isinstance(workspace, dict) else None
            if bot_token:
                default_token = bot_token
                break

        self.app = AsyncApp(token=default_token)
        # Only use message event - we detect mentions ourselves to avoid
        # duplicate events (app_mention + message) and ensure files are included
        self.app.event("message")(self._handle_message)
        # Register no-op handler for app_mention to suppress warnings
        # (we handle mentions via the message event to get file attachments)
        self.app.event("app_mention")(self._handle_app_mention_noop)

        self.command_handler = RoomCommandHandler(
            self.agent,
            "slack",
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

    def _now(self) -> float:
        return time.monotonic()

    async def _handle_app_mention_noop(self, ack) -> None:
        """No-op handler for app_mention events.

        We handle mentions via message events to ensure file attachments are included.
        This handler just acknowledges the event to suppress warnings.
        """
        await ack()

    async def _handle_message(self, body: dict[str, Any], event: dict[str, Any], ack) -> None:
        await ack()

        # Skip bot messages (we check bot_id rather than subtype to catch all bot messages)
        if event.get("bot_id"):
            return

        # Handle message edits
        subtype = event.get("subtype")
        if subtype == "message_changed":
            await self._handle_message_edit(body, event)
            return

        channel_type = event.get("channel_type")
        if channel_type == "im":
            # DMs are always direct
            await self.process_message_event(body, event, is_direct=True)
        else:
            # Ensure client is initialized to get bot_user_id for mention detection
            team_id = self._get_team_id(body)
            if team_id:
                await self._get_client(team_id)

            # Check if this message mentions the bot (like Discord's _is_highlight)
            bot_user_id = self.bot_user_ids.get(team_id) if team_id else None
            text = event.get("text") or ""
            is_mention = bool(bot_user_id and f"<@{bot_user_id}>" in text)
            await self.process_message_event(body, event, is_direct=is_mention)

    async def _handle_message_edit(self, body: dict[str, Any], event: dict[str, Any]) -> None:
        """Handle message_changed events to update history."""
        team_id = self._get_team_id(body)
        if not team_id:
            return

        client = await self._get_client(team_id)
        if not client:
            return

        bot_user_id = await self._get_bot_user_id(team_id, client)

        # The edited message is in event['message']
        message = event.get("message", {})
        user_id = message.get("user")
        if not user_id:
            return

        # Skip edits from our own bot
        if user_id == bot_user_id:
            logger.debug("Ignoring Slack edit from self")
            return

        # Skip bot edits
        if message.get("bot_id"):
            return

        channel_id = event.get("channel")
        if not channel_id:
            return

        # The original message ts is the platform_id
        platform_id = message.get("ts")
        if not platform_id:
            return

        new_text = message.get("text") or ""
        new_content = await self._normalize_content(new_text, team_id, client)
        if not new_content:
            return

        channel_type = event.get("channel_type")
        workspace_name = self._get_workspace_name(team_id)
        server_tag = f"slack:{workspace_name}"

        if channel_type == "im":
            display_name = await self._get_user_display_name(team_id, client, user_id)
            channel_name = f"{self._normalize_name(display_name)}_{user_id}"
        else:
            channel_name = await self._get_channel_name(team_id, client, channel_id)
            channel_name = f"#{channel_name}"

        nick = await self._get_user_display_name(team_id, client, user_id)

        updated = await self.agent.history.update_message_by_platform_id(
            server_tag, channel_name, platform_id, new_content, nick
        )
        if updated:
            logger.info("Updated edited message %s in %s#%s", platform_id, server_tag, channel_name)

    def _get_team_id(self, body: dict[str, Any]) -> str | None:
        return body.get("team_id") or body.get("team", {}).get("id")

    def _get_workspace_name(self, team_id: str) -> str:
        workspace = self.workspaces.get(team_id, {})
        return workspace.get("name", team_id)

    async def _get_client(self, team_id: str) -> AsyncWebClient | None:
        client = self.clients.get(team_id)
        if client is not None:
            return client

        workspace = self.workspaces.get(team_id)
        if not workspace:
            logger.error("Slack workspace %s missing from config", team_id)
            return None

        bot_token = workspace.get("bot_token")
        if not bot_token:
            logger.error("Slack workspace %s missing bot_token", team_id)
            return None

        client = AsyncWebClient(token=bot_token)
        self.clients[team_id] = client
        self.user_cache.setdefault(team_id, {})
        self.channel_cache.setdefault(team_id, {})

        try:
            auth = await client.auth_test()
            bot_user_id = auth.get("user_id")
            if bot_user_id:
                self.bot_user_ids[team_id] = bot_user_id
        except Exception as e:
            logger.error("Slack auth_test failed for %s: %s", team_id, e)

        return client

    async def _get_bot_user_id(self, team_id: str, client: AsyncWebClient) -> str | None:
        if team_id in self.bot_user_ids:
            return self.bot_user_ids[team_id]

        try:
            auth = await client.auth_test()
            bot_user_id = auth.get("user_id")
            if bot_user_id:
                self.bot_user_ids[team_id] = bot_user_id
            return bot_user_id
        except Exception as e:
            logger.error("Slack auth_test failed for %s: %s", team_id, e)
            return None

    async def _set_typing_indicator(
        self, client: AsyncWebClient, channel_id: str, thread_ts: str | None
    ) -> bool:
        """Show typing indicator in channel/thread using assistant.threads.setStatus.

        The indicator is automatically cleared when a message is sent to the thread.
        For non-threaded replies, call _clear_typing_indicator after sending.

        This requires the `assistant:write` scope. If the scope is missing or the API
        call fails, the error is logged but does not interrupt message processing.

        Returns True if indicator was set successfully, False otherwise.
        """
        # The API requires a thread_ts - if we don't have one, we can't show the indicator
        if not thread_ts:
            logger.debug("Skipping typing indicator: no thread_ts available")
            return False

        try:
            await client.assistant_threads_setStatus(
                channel_id=channel_id,
                thread_ts=thread_ts,
                status="is thinking...",
                loading_messages=TYPING_LOADING_MESSAGES,
            )
            logger.debug("Set typing indicator in channel %s thread %s", channel_id, thread_ts)
            return True
        except SlackApiError as e:
            error_code = e.response.get("error", "unknown")
            if error_code == "missing_scope":
                logger.warning(
                    "Cannot show typing indicator: missing 'assistant:write' scope. "
                    "Add this scope in your Slack app settings to enable typing indicators."
                )
            elif error_code == "thread_not_found":
                # Thread doesn't exist yet (we're about to create it), this is expected
                logger.debug("Typing indicator skipped: thread not yet created")
            else:
                logger.warning("Failed to set typing indicator: %s", e)
        except Exception as e:
            logger.warning("Unexpected error setting typing indicator: %s", e)
        return False

    async def _clear_typing_indicator(
        self, client: AsyncWebClient, channel_id: str, thread_ts: str
    ) -> None:
        """Clear typing indicator by setting empty status."""
        try:
            await client.assistant_threads_setStatus(
                channel_id=channel_id,
                thread_ts=thread_ts,
                status="",
            )
            logger.debug("Cleared typing indicator in channel %s thread %s", channel_id, thread_ts)
        except Exception as e:
            # Don't warn, this is best-effort cleanup
            logger.debug("Failed to clear typing indicator: %s", e)

    async def _get_user_display_name(
        self, team_id: str, client: AsyncWebClient, user_id: str
    ) -> str:
        cache = self.user_cache.setdefault(team_id, {})
        if user_id in cache:
            return cache[user_id]

        try:
            response = await client.users_info(user=user_id)
            user = response.get("user", {})
            profile = user.get("profile", {})
            display_name = (
                profile.get("display_name")
                or profile.get("real_name")
                or user.get("name")
                or user_id
            )
        except Exception as e:
            logger.error("Slack users_info failed for %s: %s", user_id, e)
            display_name = user_id

        cache[user_id] = display_name
        # Also populate reverse lookup for mention formatting in replies
        id_cache = self.user_id_cache.setdefault(team_id, {})
        id_cache[display_name.lower()] = user_id
        return display_name

    async def _get_channel_name(self, team_id: str, client: AsyncWebClient, channel_id: str) -> str:
        cache = self.channel_cache.setdefault(team_id, {})
        if channel_id in cache:
            return cache[channel_id]

        try:
            response = await client.conversations_info(channel=channel_id)
            channel = response.get("channel", {})
            name = channel.get("name") or channel_id
        except Exception as e:
            logger.error("Slack conversations_info failed for %s: %s", channel_id, e)
            name = channel_id

        cache[channel_id] = name
        return name

    async def _normalize_content(self, text: str, team_id: str, client: AsyncWebClient) -> str:
        content = html.unescape(text or "")

        # Replace links like <https://example.com|label> -> https://example.com
        content = re.sub(r"<(https?://[^>|]+)\|[^>]+>", r"\1", content)
        content = re.sub(r"<(https?://[^>]+)>", r"\1", content)

        # Replace channel mentions like <#C123|general> -> #general
        content = re.sub(r"<#([A-Z0-9]+)\|([^>]+)>", r"#\2", content)

        # Replace user mentions like <@U123>
        user_matches = set(re.findall(r"<@([A-Z0-9]+)>", content))
        for user_id in user_matches:
            display_name = await self._get_user_display_name(team_id, client, user_id)
            content = content.replace(f"<@{user_id}>", f"@{display_name}")

        return content

    def _format_mentions_for_slack(self, text: str, team_id: str) -> str:
        """Convert @DisplayName to <@USER_ID> for proper Slack mentions."""
        if not text:
            return text

        id_cache = self.user_id_cache.get(team_id, {})
        if not id_cache:
            logger.debug("No user_id_cache for team %s, skipping mention formatting", team_id)
            return text

        # Sort by length descending to match longer names first (e.g., "John Smith" before "John")
        sorted_names = sorted(id_cache.keys(), key=len, reverse=True)

        result = text
        for name in sorted_names:
            user_id = id_cache[name]
            # Case-insensitive match for @Name pattern
            pattern = re.compile(r"@" + re.escape(name), re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(f"<@{user_id}>", result)
                logger.debug("Mention formatted: @%s -> <@%s>", name, user_id)

        return result

    def _strip_leading_mention(self, text: str, bot_user_id: str, bot_name: str) -> str:
        if not text:
            return text

        cleaned_text = text.lstrip()
        mention_pattern = rf"^\s*(?:<@{re.escape(bot_user_id)}>\s*)+[:,]?\s*(.*)$"
        match = re.match(mention_pattern, cleaned_text)
        if match:
            return match.group(1).strip()

        name_pattern = rf"^\s*@?{re.escape(bot_name)}[:,]?\s*(.*)$"
        match = re.match(name_pattern, cleaned_text)
        if match:
            return match.group(1).strip()

        return cleaned_text

    def _build_attachment_block(self, files: list[dict[str, Any]]) -> str:
        attachment_lines: list[str] = []
        for i, file in enumerate(files, start=1):
            meta = file.get("mimetype") or file.get("filetype") or "attachment"
            filename = file.get("name") or file.get("title")
            if filename:
                meta += f" (filename: {filename})"
            size = file.get("size")
            if size:
                meta += f" (size: {size})"
            url = file.get("url_private") or file.get("url_private_download") or ""
            attachment_lines.append(f"{i}. {meta}: {url}")

        if not attachment_lines:
            return ""

        return "\n".join(["[Attachments]", *attachment_lines, "[/Attachments]"])

    def _get_reply_thread_ts(
        self, channel_type: str | None, platform_id: str | None, existing_thread_id: str | None
    ) -> str | None:
        """Determine the thread_ts to use for replies based on configuration.

        Returns the thread_ts that should be used when sending replies.
        """
        if existing_thread_id:
            return existing_thread_id

        if not platform_id:
            return None

        if channel_type == "im":
            return platform_id if self.reply_start_thread.get("dm") else None
        return platform_id if self.reply_start_thread.get("channel") else None

    async def process_message_event(
        self, body: dict[str, Any], event: dict[str, Any], *, is_direct: bool
    ) -> None:
        team_id = self._get_team_id(body)
        if not team_id:
            logger.error("Slack event missing team_id: %s", body)
            return

        client = await self._get_client(team_id)
        if not client:
            return

        bot_user_id = await self._get_bot_user_id(team_id, client)
        if not bot_user_id:
            return

        user_id = event.get("user")
        if not user_id:
            return

        if user_id == bot_user_id:
            logger.debug("Ignoring Slack message from self")
            return

        channel_id = event.get("channel")
        if not channel_id:
            return

        channel_type = event.get("channel_type")
        workspace_name = self._get_workspace_name(team_id)
        server_tag = f"slack:{workspace_name}"

        if channel_type == "im":
            display_name = await self._get_user_display_name(team_id, client, user_id)
            channel_name = f"{self._normalize_name(display_name)}_{user_id}"
        else:
            channel_name = await self._get_channel_name(team_id, client, channel_id)
            channel_name = f"#{channel_name}"

        bot_name = self.bot_display_names.get(team_id)
        if not bot_name:
            bot_name = await self._get_user_display_name(team_id, client, bot_user_id)
            self.bot_display_names[team_id] = bot_name

        text = event.get("text") or ""
        if is_direct:
            text = self._strip_leading_mention(text, bot_user_id, bot_name)

        content = await self._normalize_content(text, team_id, client)

        files = event.get("files") or []
        attachment_block = self._build_attachment_block(files)
        if attachment_block:
            content = f"{content}\n\n{attachment_block}" if content else attachment_block

        if not content:
            logger.debug("No content in Slack message: %s", event)
            return

        nick = await self._get_user_display_name(team_id, client, user_id)
        mynick = bot_name

        if self.command_handler.should_ignore_user(nick):
            logger.debug("Ignoring user: %s", nick)
            return

        platform_id = event.get("ts")
        thread_id = event.get("thread_ts")
        thread_starter_id: int | None = None
        if thread_id:
            thread_starter_id = await self.agent.history.get_message_id_by_platform_id(
                server_tag, channel_name, thread_id
            )

        reply_thread_ts = self._get_reply_thread_ts(channel_type, platform_id, thread_id)

        # Fix thread_id bug: when starting a new thread, align thread_id for context
        # so that context lookup and storage both use the same thread_id
        if reply_thread_ts and not thread_id:
            thread_id = reply_thread_ts

        secrets: dict[str, Any] | None = None
        if files:
            workspace = self.workspaces.get(team_id, {})
            bot_token = workspace.get("bot_token")
            if bot_token:
                secrets = {
                    "http_header_prefixes": {
                        "https://files.slack.com/": {"Authorization": f"Bearer {bot_token}"}
                    }
                }

        msg = RoomMessage(
            server_tag=server_tag,
            channel_name=channel_name,
            nick=nick,
            mynick=mynick,
            content=content,
            platform_id=platform_id,
            thread_id=thread_id,
            thread_starter_id=thread_starter_id,
            _response_thread_id=reply_thread_ts,
            secrets=secrets,
        )

        last_reply_ts: str | None = None
        last_reply_text: str | None = None
        last_reply_time: float | None = None
        typing_indicator_thread_ts: str | None = None

        async def reply_sender(text: str) -> None:
            nonlocal last_reply_ts, last_reply_time, last_reply_text
            now = self._now()

            formatted_text = self._format_mentions_for_slack(text, team_id)

            if (
                last_reply_ts is not None
                and last_reply_time is not None
                and now - last_reply_time < self.reply_edit_debounce_seconds
            ):
                combined = (
                    f"{last_reply_text}\n{formatted_text}" if last_reply_text else formatted_text
                )
                await client.chat_update(channel=channel_id, ts=last_reply_ts, text=combined)
                last_reply_text = combined
                last_reply_time = now
            else:
                send_kwargs: dict[str, Any] = {"channel": channel_id, "text": formatted_text}
                if msg.response_thread_id:
                    send_kwargs["thread_ts"] = msg.response_thread_id
                response = await client.chat_postMessage(**send_kwargs)
                last_reply_ts = response.get("ts")
                last_reply_text = formatted_text
                last_reply_time = now

            if typing_indicator_thread_ts:
                await self._set_typing_indicator(client, channel_id, typing_indicator_thread_ts)

        trigger_message_id = await self.agent.history.add_message(msg)

        if is_direct:
            typing_thread_ts = msg.thread_id or platform_id
            typing_indicator_set = await self._set_typing_indicator(
                client, channel_id, typing_thread_ts
            )
            if typing_indicator_set:
                typing_indicator_thread_ts = typing_thread_ts

            with MessageLoggingContext(msg.arc, msg.nick, msg.content):
                await self.command_handler.handle_command(msg, trigger_message_id, reply_sender)

            if typing_indicator_set and typing_thread_ts:
                await self._clear_typing_indicator(client, channel_id, typing_thread_ts)
            return

        await self.command_handler.handle_passive_message(msg, reply_sender)

    async def run(self) -> None:
        """Run the main Slack monitor loop."""
        if not self.app_token:
            logger.error("Slack app_token missing in config; skipping Slack monitor")
            return

        if not self.workspaces:
            logger.error("Slack workspaces missing in config; skipping Slack monitor")
            return

        handler = AsyncSocketModeHandler(self.app, self.app_token)
        try:
            await handler.start_async()
        finally:
            await self.command_handler.proactive_debouncer.cancel_all()
