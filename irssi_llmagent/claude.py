"""Anthropic Claude API client implementation."""

import json
import logging
from typing import Any

import aiohttp

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseAPIClient):
    """Anthropic Claude API client with async support."""

    def __init__(self, config: dict[str, Any]):
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        cfg = providers.get("anthropic", {})
        super().__init__(cfg)
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
        raw_response = await self.call_raw(context, system_prompt, model)
        return self.extract_text_from_response(raw_response)

    async def call_claude_raw(
        self, context: list[dict], system_prompt: str, model: str, tools: list | None = None
    ) -> dict:
        """Call Claude API with context and system prompt - deprecated, use call_raw."""
        return await self.call_raw(context, system_prompt, model, tools)

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: str | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call Claude API with context and system prompt."""
        if not self.session:
            raise RuntimeError("AnthropicClient not initialized as async context manager")

        # Build Claude-friendly messages (skip function_call and function_call_output artifacts)
        messages = []
        for m in context:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                messages.append({"role": m.get("role"), "content": m.get("content") or "..."})
        # Ensure first message is from user
        if messages and messages[0].get("role") != "user":
            messages.insert(0, {"role": "user", "content": "..."})
        # Ensure we have at least one message
        if not messages:
            messages.append({"role": "user", "content": "..."})

        if messages[-1]["role"] == "assistant":
            # may happen in some race conditions with proactive checks or
            # multiple commands
            return {"cancel": "(wait, I just replied)"}

        payload = {
            "model": model,
            "max_tokens": 1024 if tools else 256,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = tools
            # TODO tool_choice

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

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from Claude response format."""
        # Check for refusal
        if response.get("stop_reason") == "refusal":
            logger.warning("Claude refusal detected")
            return ""

        if "content" in response and response["content"]:
            # Find text content
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text = content_block["text"]
                    logger.debug(f"Claude response text: {text}")
                    return text or ""

        return ""

    def has_tool_calls(self, response: dict) -> bool:
        """Check if Claude response contains tool calls."""
        return response.get("stop_reason") == "tool_use"

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from Claude response format."""
        content = response.get("content", [])
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {"id": block["id"], "name": block["name"], "input": block["input"]}
                )
        return tool_uses if tool_uses else None

    def format_assistant_message(self, response: dict) -> dict:
        """Format Claude's response for conversation history."""
        content = response.get("content", [])
        return {"role": "assistant", "content": content}

    def format_tool_results(self, tool_results: list[dict]) -> dict:
        """Format tool results for Claude API."""
        return {"role": "user", "content": tool_results}
