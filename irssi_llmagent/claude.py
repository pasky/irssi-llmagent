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

    async def call_claude(self, context: list[dict], system_prompt: str, model: str) -> str:
        """Call Claude API with context and system prompt, returning cleaned text response."""
        raw_response = await self.call_claude_raw(context, system_prompt, model)
        return self.extract_text_from_response(raw_response)

    async def call_claude_raw(
        self, context: list[dict], system_prompt: str, model: str, tools: list | None = None
    ) -> dict:
        """Call Claude API with context and system prompt."""
        if not self.session:
            raise RuntimeError("AnthropicClient not initialized as async context manager")

        # Keep separate user turns - Claude API handles consecutive user messages fine
        messages = context.copy()

        # Ensure first message is from user (only for simple text messages)
        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "..."})

        # Ensure we have at least one message
        if not messages:
            messages.append({"role": "user", "content": "..."})

        if messages[-1]["role"] == "assistant":
            # may happen in some race conditions with proactive checks or
            # multiple commands
            return {"error": "(wait, I just replied)"}

        payload = {
            "model": model,
            "max_tokens": 1024 if tools else 256,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = tools

        logger.debug(f"Calling Anthropic API with model: {model}")
        logger.debug(f"Anthropic request payload: {json.dumps(payload, indent=2)}")

        try:
            async with self.session.post(self.config["url"], json=payload) as response:
                response.raise_for_status()
                data = await response.json()

            logger.debug(f"Anthropic response: {json.dumps(data, indent=2)}")
            return data

        except aiohttp.ClientError as e:
            logger.error(f"Anthropic API error: {e}")
            return {"error": f"API error: {e}"}

    def extract_text_from_response(self, response: dict) -> str:
        """Extract cleaned text from raw Claude response."""

        if "error" in response:
            return ""  # response["error"]

        # Check for refusal
        if response.get("stop_reason") == "refusal":
            logger.warning("Claude refusal detected")
            return ""  # "refusal, rip"

        if "content" in response and response["content"]:
            # Find text content
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text = content_block["text"]
                    logger.debug(f"Claude response text: {text}")

                    # Clean up response
                    text = text.strip()

                    # Remove thinking tags and content if present
                    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)

                    # For IRC: single line only, take first line of remaining content
                    text = text.replace("\n", "; ").strip()

                    # Remove IRC nick prefix
                    text = re.sub(r"^(\[..:..\]\s*)?<[^>]+>\s*", "", text)

                    logger.debug(f"Cleaned Claude response: {text}")
                    return text

        return "..."
