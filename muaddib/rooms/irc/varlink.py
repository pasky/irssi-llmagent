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

            # Build a mapping from character index to cumulative byte count
            # This allows us to find split points efficiently
            char_to_bytes: list[int] = []  # char_to_bytes[i] = bytes up to and including char i
            byte_count = 0
            for ch in text:
                byte_count += len(ch.encode("utf-8"))
                char_to_bytes.append(byte_count)
            total_bytes = char_to_bytes[-1] if char_to_bytes else 0

            # Find the maximum character index that fits within max_payload
            max_char_idx = 0
            for i, cb in enumerate(char_to_bytes):
                if cb <= max_payload:
                    max_char_idx = i + 1  # split after this character
                else:
                    break

            # If nothing fits, we must split at the first char boundary that fits
            if max_char_idx == 0:
                return "", text

            # Target split point: we want both parts to fit in max_payload.
            # If total_bytes <= 2 * max_payload, we can split near the middle.
            # If total_bytes > 2 * max_payload, we MUST split so remaining fits,
            # i.e., first part must be at least (total_bytes - max_payload) bytes.
            min_first_bytes = max(0, total_bytes - max_payload)
            # Ideal target: as close to middle as possible, but at least min_first_bytes
            target_bytes = max(min_first_bytes, max_payload // 2)

            # Find all valid split points by delimiter type, ordered by preference
            # Each delimiter type: (pattern_check_func, split_after_offset)
            # split_after_offset: how many chars after the delimiter to split
            # (e.g., for ". " we split AFTER the space, so offset is 2)

            # Collect split candidates: (char_idx_after_split, delimiter_priority)
            # Lower priority number = better delimiter
            # Priority: 0=sentence, 1=semicolon, 2=comma, 3=hyphen, 4=space

            candidates: list[tuple[int, int, int]] = []  # (char_idx, priority, bytes_at_idx)

            for i in range(max_char_idx):
                if i >= len(text):
                    break
                ch = text[i]

                # Check for sentence boundary: ". " or "! " or "? "
                if ch in ".!?" and i + 1 < len(text) and text[i + 1] == " ":
                    split_at = i + 2  # after the space
                    if split_at <= max_char_idx:
                        candidates.append((split_at, 0, char_to_bytes[split_at - 1]))

                # Check for semicolon: "; "
                if ch == ";" and i + 1 < len(text) and text[i + 1] == " ":
                    split_at = i + 2
                    if split_at <= max_char_idx:
                        candidates.append((split_at, 1, char_to_bytes[split_at - 1]))

                # Check for comma: ", "
                if ch == "," and i + 1 < len(text) and text[i + 1] == " ":
                    split_at = i + 2
                    if split_at <= max_char_idx:
                        candidates.append((split_at, 2, char_to_bytes[split_at - 1]))

                # Check for hyphen: " - " (surrounded by spaces)
                if (
                    ch == "-"
                    and i > 0
                    and text[i - 1] == " "
                    and i + 1 < len(text)
                    and text[i + 1] == " "
                ):
                    split_at = i + 2  # after the trailing space
                    if split_at <= max_char_idx:
                        candidates.append((split_at, 3, char_to_bytes[split_at - 1]))

                # Check for word boundary: any whitespace
                if ch.isspace():
                    split_at = i + 1  # after the space
                    if split_at <= max_char_idx:
                        candidates.append((split_at, 4, char_to_bytes[split_at - 1]))

            # If no candidates found, fall back to hard split at max_char_idx
            if not candidates:
                return text[:max_char_idx], text[max_char_idx:]

            # Filter candidates: must leave second part small enough to fit in max_payload
            # A candidate at bytes_at leaves (total_bytes - bytes_at) for second part
            valid_candidates = [c for c in candidates if total_bytes - c[2] <= max_payload]

            # If no valid candidates, find the minimum char index that satisfies the constraint
            # (hard split without nice delimiter)
            if not valid_candidates:
                # Find first char index where remaining bytes <= max_payload
                for i, cb in enumerate(char_to_bytes):
                    if total_bytes - cb <= max_payload:
                        return text[: i + 1], text[i + 1 :]
                # Fallback (shouldn't happen if total_bytes <= 2 * max_payload)
                return text[:max_char_idx], text[max_char_idx:]

            # Select the best candidate: closest to target_bytes, preferring higher-priority delimiters
            # Strategy: for each priority level, find the candidate closest to middle
            # Use the highest-priority level that has a candidate reasonably close to middle

            def score_candidate(cand: tuple[int, int, int]) -> tuple[int, int]:
                """Return (priority, distance_from_target) for sorting."""
                char_idx, priority, bytes_at = cand
                distance = abs(bytes_at - target_bytes)
                return (priority, distance)

            # Sort by priority first, then by distance from target
            valid_candidates.sort(key=score_candidate)

            # Take the best candidate (lowest priority number, then closest to middle)
            best_char_idx = valid_candidates[0][0]

            head = text[:best_char_idx]
            tail = text[best_char_idx:]
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
            # Keep as many complete characters as fit, but leave room for "..."
            ellipsis = "..."
            ellipsis_bytes = len(ellipsis.encode("utf-8"))
            effective_payload = max_payload - ellipsis_bytes
            byte_count = 0
            end_idx = 0
            for i, ch in enumerate(rest):
                ch_bytes = len(ch.encode("utf-8"))
                if byte_count + ch_bytes > effective_payload:
                    break
                byte_count += ch_bytes
                end_idx = i + 1
            rest = rest[:end_idx] + ellipsis
            logger.warning(
                "Message exceeded 2x max IRC payload, truncated with ellipsis. "
                "Original: %d bytes, max per message: %d bytes",
                len(rest_bytes) + len(first.encode("utf-8")),
                max_payload,
            )

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
