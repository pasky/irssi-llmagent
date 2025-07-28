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
        self, context: list[dict], system_prompt: str, model: str
    ) -> str | None:
        """Call Claude API with context and system prompt, returning cleaned text response."""
        raw_response = await self.call_claude_raw(context, system_prompt, model)
        return self.extract_text_from_response(raw_response)

    async def call_claude_raw(
        self, context: list[dict], system_prompt: str, model: str, tools: list | None = None
    ) -> dict | None:
        """Call Claude API with context and system prompt."""
        if not self.session:
            raise RuntimeError("AnthropicClient not initialized as async context manager")

        # Coalesce consecutive user messages (only for simple string content)
        messages = []
        for msg in context:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += "\n" + msg["content"]
                else:
                    messages.append(msg)
            else:
                messages.append(msg)

        # Ensure first message is from user (only for simple text messages)
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "..."})

        # Ensure we have at least one message
        if not messages:
            messages.append({"role": "user", "content": "..."})

        payload = {
            "model": model,
            "max_tokens": 1024 if tools else 256,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = tools

        logger.info(f"Calling Anthropic API with model: {model}")
        logger.info(f"Anthropic request payload: {json.dumps(payload, indent=2)}")

        try:
            async with self.session.post(self.config["url"], json=payload) as response:
                response.raise_for_status()
                data = await response.json()

            logger.info(f"Anthropic response: {json.dumps(data, indent=2)}")
            return data

        except aiohttp.ClientError as e:
            logger.error(f"Anthropic API error: {e}")
            return {"error": f"API error: {e}"}

        return None

    def extract_text_from_response(self, response: dict | None) -> str | None:
        """Extract cleaned text from raw Claude response."""
        if not response:
            return None

        if "error" in response:
            return response["error"]

        # Check for refusal
        if response.get("stop_reason") == "refusal":
            logger.warning("Claude refusal detected")
            return "refusal, rip"

        if "content" in response and response["content"]:
            # Find text content
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text = content_block["text"]
                    logger.info(f"Claude response text: {text}")
                    # Clean up response - single line only
                    text = text.strip()
                    text = re.sub(r"\n.*", "", text)
                    text = re.sub(r"^<[^>]+>\s*", "", text)  # Remove IRC nick prefix
                    logger.info(f"Cleaned Claude response: {text}")
                    return text

        return None
