"""Varlink protocol client implementations for irssi communication."""

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class BaseVarlinkClient:
    """Base class for varlink protocol clients."""

    def __init__(self, socket_path: str):
        self.socket_path = os.path.expanduser(socket_path)
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connect to varlink socket."""
        self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)
        logger.debug(f"Connected to varlink socket: {self.socket_path}")

    async def disconnect(self) -> None:
        """Disconnect from varlink socket."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            self.writer = None
            self.reader = None

    async def send_call(
        self, method: str, parameters: dict[str, Any] | None = None, more: bool = False
    ) -> dict[str, Any] | None:
        """Send a varlink method call."""
        if not self.writer:
            raise ConnectionError("Not connected to varlink socket")

        call = {"method": method, "parameters": parameters or {}}
        if more:
            call["more"] = True

        message = json.dumps(call) + "\0"
        logger.debug(f"Sending varlink call: {call}")
        self.writer.write(message.encode("utf-8"))
        await self.writer.drain()

        if not more:  # Only wait for response if not streaming
            return await self.receive_response()
        return None

    async def receive_response(self) -> dict[str, Any] | None:
        """Receive a varlink response."""
        if not self.reader:
            return None

        try:
            data = await self.reader.readuntil(b"\0")
            message = data[:-1]  # Remove null terminator
            if message:
                return json.loads(message.decode("utf-8"))
        except (asyncio.IncompleteReadError, json.JSONDecodeError) as e:
            logger.error(f"Error receiving varlink response: {e}")
            return None
        return None

    async def get_server_nick(self, server: str) -> str | None:
        """Get bot's nick for a server."""
        logger.debug(f"Getting nick for server: {server}")
        response = await self.send_call("org.irssi.varlink.GetServerNick", {"server": server})

        if response and "parameters" in response:
            nick = response["parameters"].get("nick")
            logger.debug(f"Got nick for server {server}: {nick}")
            return nick
        logger.warning(f"Failed to get nick for server {server}")
        return None


class VarlinkClient(BaseVarlinkClient):
    """Async varlink protocol client for receiving events from irssi."""

    async def wait_for_events(self) -> None:
        """Start waiting for IRC events."""
        await self.send_call("org.irssi.varlink.WaitForEvent", more=True)


class VarlinkSender(BaseVarlinkClient):
    """Async varlink client for sending messages to IRC."""

    def __init__(self, socket_path: str):
        super().__init__(socket_path)
        self._send_lock = asyncio.Lock()

    async def send_call(
        self, method: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Send a varlink method call and wait for response."""
        async with self._send_lock:
            return await super().send_call(method, parameters, more=False)

    async def send_message(self, target: str, message: str, server: str) -> bool:
        """Send a message to IRC, splitting into at most two PRIVMSGs if needed.

        IRC messages are limited to 512 bytes including command and CRLF. The
        effective payload limit for a client-sent PRIVMSG is roughly:
            512 - len("PRIVMSG ") - len(target) - len(" :") - len(CRLF)
        which simplifies to: 512 - 12 - len(target)
        Note: servers prepend a source prefix and may include tags, so we keep a
        conservative safety margin to avoid server-side truncation.
        We count bytes strictly in UTF-8 and never split inside a code point.
        Prefer splitting on whitespace when possible.
        """
        # Calculate maximum payload bytes for the message text
        try:
            target_len = len(target.encode("utf-8"))
        except Exception:
            target_len = len(target)
        SAFETY_MARGIN = 60  # bytes, to account for prefix/tags on the wire
        max_payload = max(1, 512 - 12 - target_len - SAFETY_MARGIN)

        def split_once(text: str) -> tuple[str, str | None]:
            # Fast path if it fits
            b = text.encode("utf-8")
            if len(b) <= max_payload:
                return text, None

            # Walk characters while counting bytes to avoid partial code points
            byte_count = 0
            split_idx = 0  # character index to split at
            last_space_idx: int | None = None
            for i, ch in enumerate(text):
                ch_bytes = len(ch.encode("utf-8"))
                if byte_count + ch_bytes > max_payload:
                    break
                byte_count += ch_bytes
                split_idx = i + 1
                if ch.isspace():
                    last_space_idx = split_idx  # position AFTER the space

            # Prefer last whitespace if it doesn't make the first part too short
            if last_space_idx is not None and last_space_idx >= split_idx // 2:
                split_idx = last_space_idx

            head = text[:split_idx]
            tail = text[split_idx:]
            return head, tail

        first, rest = split_once(message)
        if rest is None:
            response = await self.send_call(
                "org.irssi.varlink.SendMessage",
                {"target": target, "message": first, "server": server},
            )
            if response and "parameters" in response:
                return response["parameters"].get("success", False)
            return False

        # Need a second part; ensure it also fits within one payload (truncate if not)
        # Aesthetics: if the split occurred after whitespace, avoid sending a
        # second message that starts with a space (trim exactly one if present)
        if rest.startswith(" "):
            rest = rest[1:]
        rest_bytes = rest.encode("utf-8")
        if len(rest_bytes) > max_payload:
            # Keep as many complete characters as fit
            byte_count = 0
            end_idx = 0
            for i, ch in enumerate(rest):
                ch_bytes = len(ch.encode("utf-8"))
                if byte_count + ch_bytes > max_payload:
                    break
                byte_count += ch_bytes
                end_idx = i + 1
            rest = rest[:end_idx]

        ok = True
        for part in (first, rest):
            response = await self.send_call(
                "org.irssi.varlink.SendMessage",
                {"target": target, "message": part, "server": server},
            )
            ok = ok and bool(
                response
                and "parameters" in response
                and response["parameters"].get("success", False)
            )
        return ok
