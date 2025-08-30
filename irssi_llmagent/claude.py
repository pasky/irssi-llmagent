"""Anthropic Claude API client implementation."""

import asyncio
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
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call Claude API with context and system prompt."""
        if not self.session:
            raise RuntimeError("AnthropicClient not initialized as async context manager")

        messages = []
        for m in context:
            if m.get("tool_calls"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": "<tools>"
                        + json.dumps(m.get("tool_calls"))
                        + "</tools> <meta>! do not write like this again, use a proper tool call template</meta>",
                    }
                )
            elif m.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "user",
                        "content": "<tool_results>"
                        + json.dumps(m.get("output"))
                        + "</tool_results>",
                    }
                )
            elif m.get("role") in ("user", "assistant"):
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
            logger.debug(context)
            logger.debug(messages)
            return {"cancel": "(wait, I just replied)"}

        thinking_budget = 0
        if reasoning_effort == "low":
            thinking_budget = 1024
        elif reasoning_effort == "medium":
            thinking_budget = 4096
        elif reasoning_effort == "high":
            thinking_budget = 16000

        payload = {
            "model": model,
            "max_tokens": (1024 if tools else 256) + thinking_budget,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = tools
        if thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if tool_choice:
            # As per this documentation: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use
            # > Tool choice limitation: Tool use with thinking only supports tool_choice: {"type": "auto"} (the default) or tool_choice: {"type": "none"}. Using tool_choice: {"type": "any"} or tool_choice: {"type": "tool", "name": "..."} will result in an error because these options force tool use, which is incompatible with extended thinking.
            messages.append(
                {
                    "role": "user",
                    "content": f"<meta>only tool {tool_choice} may be called now</meta>",
                }
            )

        logger.debug(f"Calling Anthropic API with model: {model}")
        logger.debug(f"Anthropic request payload: {json.dumps(payload, indent=2)}")

        # Exponential backoff retry for 529 (overloaded) errors
        backoff_delays = [0, 2, 5, 10, 20]  # No delay, then 2s, 5s, 10s, 20s

        for attempt, delay in enumerate(backoff_delays):
            if delay > 0:
                logger.info(
                    f"Waiting {delay}s before retry {attempt + 1}/{len(backoff_delays)} for Anthropic API"
                )
                await asyncio.sleep(delay)

            try:
                async with self.session.post(self.config["url"], json=payload) as response:
                    data = await response.json()

                    if not response.ok:
                        # Check for 529 overloaded error and retry if not last attempt
                        if response.status == 529 and attempt < len(backoff_delays) - 1:
                            error_body = json.dumps(data) if data else f"HTTP {response.status}"
                            logger.warning(
                                f"Anthropic overloaded (HTTP 529), retrying in {backoff_delays[attempt + 1]}s..."
                            )
                            continue

                        error_body = json.dumps(data) if data else f"HTTP {response.status}"
                        raise aiohttp.ClientError(
                            f"Anthropic HTTP status {response.status}: {error_body}"
                        )

                logger.debug(f"Anthropic response: {json.dumps(data, indent=2)}")
                return data

            except aiohttp.ClientError as e:
                # Only retry 529 errors, fail fast on other errors
                if "HTTP status 529" in str(e) and attempt < len(backoff_delays) - 1:
                    continue
                logger.error(f"Anthropic API error: {e}")
                return {"error": f"API error: {e}"}

        # This should never be reached, but added for type safety
        return {"error": "All retry attempts exhausted"}

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
        processed_results = []
        for result in tool_results:
            content = result["content"]
            # Check if content is image data
            if isinstance(content, str) and content.startswith("IMAGE_DATA:"):
                try:
                    _, content_type, size, base64_data = content.split(":", 3)
                    # Extract the base format (e.g., "jpeg" from "image/jpeg")
                    media_type = content_type.split("/")[-1]
                    if media_type in ["jpeg", "png", "gif", "webp"]:
                        processed_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_use_id"],
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": content_type,
                                            "data": base64_data,
                                        },
                                    }
                                ],
                            }
                        )
                    else:
                        # Fallback for unsupported image types
                        processed_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_use_id"],
                                "content": f"Downloaded image ({content_type}, {size} bytes)",
                            }
                        )
                except ValueError:
                    # Malformed image data, treat as text
                    processed_results.append(result)
            else:
                # Regular text content
                processed_results.append(result)
        return {"role": "user", "content": processed_results}
