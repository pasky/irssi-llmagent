"""Anthropic Claude API client implementation."""

import json
import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Claude API client with async support."""

    def __init__(self, config: dict[str, Any]):
        self.config = config["anthropic"]
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.config["key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "irssi-llmagent/1.0",
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call_claude(
        self, context: list[dict[str, str]], system_prompt: str, model: str
    ) -> str | None:
        """Call Claude API with context and system prompt."""
        if not self.session:
            raise RuntimeError("AnthropicClient not initialized as async context manager")

        # Coalesce consecutive user messages
        messages = []
        for msg in context:
            if msg["role"] == "user":
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += "\n" + msg["content"]
                else:
                    messages.append(msg)
            else:
                messages.append(msg)

        # Ensure first message is from user
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "..."})

        payload = {
            "model": model,
            "max_tokens": 256,
            "messages": messages,
            "system": system_prompt,
        }

        logger.info(f"Calling Anthropic API with model: {model}")
        logger.info(f"Anthropic request payload: {json.dumps(payload, indent=2)}")

        try:
            async with self.session.post(self.config["url"], json=payload) as response:
                response.raise_for_status()
                data = await response.json()

            logger.info(f"Anthropic response: {json.dumps(data, indent=2)}")

            # Check for refusal
            if data.get("stop_reason") == "refusal":
                logger.warning("Claude refusal detected")
                return "refusal, rip"

            if "content" in data and data["content"]:
                text = data["content"][0]["text"]
                logger.info(f"Claude response text: {text}")
                # Clean up response - single line only
                text = text.strip()
                text = re.sub(r"\n.*", "", text)
                text = re.sub(r"^<[^>]+>\s*", "", text)  # Remove IRC nick prefix
                logger.info(f"Cleaned Claude response: {text}")
                return text

        except aiohttp.ClientError as e:
            logger.error(f"Anthropic API error: {e}")
            return f"API error: {e}"

        return None
