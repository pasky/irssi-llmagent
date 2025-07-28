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
        """Send a message to IRC."""
        response = await self.send_call(
            "org.irssi.varlink.SendMessage",
            {"target": target, "message": message, "server": server},
        )

        if response and "parameters" in response:
            return response["parameters"].get("success", False)
        return False
