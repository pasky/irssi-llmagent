"""DeepSeek API client implementation."""

import asyncio
import json
import logging
from typing import Any

import aiohttp

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class DeepSeekClient(BaseAPIClient):
    """DeepSeek API client with Anthropic API compatibility."""

    def __init__(self, config: dict[str, Any]):
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        cfg = providers.get("deepseek", {})
        super().__init__(cfg)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        # Ensure URL ends with /v1/messages for Anthropic API compatibility
        base_url = self.config["url"].rstrip("/")
        if not base_url.endswith("/v1/messages"):
            if base_url.endswith("/anthropic"):
                base_url += "/v1/messages"
            else:
                base_url += "/v1/messages"

        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.config["key"],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "irssi-llmagent/1.0",
            }
        )
        # Store the corrected URL
        self.config = dict(self.config)
        self.config["url"] = base_url
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call DeepSeek API with context and system prompt."""
        if not self.session:
            raise RuntimeError("DeepSeekClient not initialized as async context manager")

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
            logger.debug(context)
            logger.debug(messages)
            return {"cancel": "(wait, I just replied)"}

        payload = {
            "model": model,
            "max_tokens": 4096 if tools else 256,
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = tools
        if tool_choice:
            messages.append(
                {
                    "role": "user",
                    "content": f"<meta>only tool {tool_choice} may be called now</meta>",
                }
            )

        logger.debug(f"Calling DeepSeek API with model: {model}")
        logger.debug(f"DeepSeek request payload: {json.dumps(payload, indent=2)}")

        # Exponential backoff retry for 529 (overloaded) errors
        backoff_delays = [0, 2, 5, 10, 20]  # No delay, then 2s, 5s, 10s, 20s

        for attempt, delay in enumerate(backoff_delays):
            if delay > 0:
                logger.info(
                    f"Waiting {delay}s before retry {attempt + 1}/{len(backoff_delays)} for DeepSeek API"
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
                                f"DeepSeek overloaded (HTTP 529), retrying in {backoff_delays[attempt + 1]}s..."
                            )
                            continue

                        error_body = json.dumps(data) if data else f"HTTP {response.status}"
                        raise aiohttp.ClientError(
                            f"DeepSeek HTTP status {response.status}: {error_body}"
                        )

                logger.debug(f"DeepSeek response: {json.dumps(data, indent=2)}")
                return data

            except aiohttp.ClientError as e:
                # Only retry 529 errors, fail fast on other errors
                if "HTTP status 529" in str(e) and attempt < len(backoff_delays) - 1:
                    continue
                logger.error(f"DeepSeek API error: {e}")
                return {"error": f"API error: {e}"}

        # This should never be reached, but added for type safety
        return {"error": "All retry attempts exhausted"}

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from DeepSeek response format."""
        # Check for refusal
        if response.get("stop_reason") == "refusal":
            logger.warning("DeepSeek refusal detected")
            return ""

        if "content" in response and response["content"]:
            # Find text content
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text = content_block["text"]
                    logger.debug(f"DeepSeek response text: {text}")
                    return text or ""

        return ""

    def has_tool_calls(self, response: dict) -> bool:
        """Check if DeepSeek response contains tool calls."""
        return response.get("stop_reason") == "tool_use"

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from DeepSeek response format."""
        content = response.get("content", [])
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {"id": block["id"], "name": block["name"], "input": block["input"]}
                )
        return tool_uses if tool_uses else None

    def format_assistant_message(self, response: dict) -> dict:
        """Format DeepSeek's response for conversation history."""
        content = response.get("content", [])
        return {"role": "assistant", "content": content}

    def format_tool_results(self, tool_results: list[dict]) -> dict:
        """Format tool results for DeepSeek API."""
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
