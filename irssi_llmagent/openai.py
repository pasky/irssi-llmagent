"""OpenAI API client implementation."""

import asyncio
import json
import logging
from typing import Any

import aiohttp

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseAPIClient):
    """OpenAI API client with async support."""

    def __init__(self, config: dict[str, Any]):
        # Handle both full config and openai subsection
        if "openai" in config:
            super().__init__(config["openai"])
        else:
            super().__init__(config)

        # Validate required keys
        required_keys = ["key", "model"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"OpenAI config missing required key: {key}")

        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60.0)  # 60 second timeout
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.config['key']}",
                "User-Agent": "irssi-llmagent/1.0",
            },
            timeout=timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tools to OpenAI function format."""
        converted = []
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )
        return converted

    async def call_raw(
        self, context: list[dict], system_prompt: str, model: str, tools: list | None = None
    ) -> dict:
        """Call OpenAI API with context and system prompt."""
        if not self.session:
            raise RuntimeError("OpenAIClient not initialized as async context manager")

        # Convert context to OpenAI format
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(context)

        # Ensure we have at least one user message
        if not messages or all(msg["role"] != "user" for msg in messages):
            messages.append({"role": "user", "content": "..."})

        # Check if last message is from assistant (but allow tool_calls)
        if messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
            return {"error": "(wait, I just replied)"}

        payload = {
            "model": model,
            "max_completion_tokens": self.config.get("max_tokens", 1024 if tools else 256),
            "messages": messages,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = self.config.get("tool_choice", "auto")

        logger.debug(f"Calling OpenAI API with model: {model}")
        logger.debug(f"OpenAI request payload: {json.dumps(payload, indent=2)}")

        try:
            url = self.config.get("url", "https://api.openai.com/v1/chat/completions")
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

            logger.debug(f"OpenAI response: {json.dumps(data, indent=2)}")
            return data

        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                logger.warning(f"OpenAI rate limit hit: {e}")
                return {"error": f"Rate limit: {e}"}
            elif e.status == 404:
                logger.error(
                    f"OpenAI API 404 error - check API key and model name. URL: {url}, Model: {model}"
                )
                return {"error": f"Not found: {e}"}
            elif e.status == 401:
                logger.error("OpenAI API authentication failed - check your API key")
                return {"error": f"Authentication failed: {e}"}
            logger.error(f"OpenAI API HTTP error: {e}")
            return {"error": f"HTTP error: {e}"}
        except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError) as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": f"API error: {e}"}

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from OpenAI response format."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                text = choice["message"]["content"]
                logger.debug(f"OpenAI response text: {text}")
                return text or ""
        return ""

    def has_tool_calls(self, response: dict) -> bool:
        """Check if OpenAI response contains tool calls."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            return "tool_calls" in choice.get("message", {})
        return False

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from OpenAI response format."""
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "tool_calls" in choice["message"]:
                tool_calls = choice["message"]["tool_calls"]
                tool_uses = []
                for call in tool_calls:
                    if call.get("type") == "function":
                        function = call.get("function", {})
                        # Parse arguments JSON string
                        import json

                        try:
                            arguments = json.loads(function.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse tool arguments for call {call.get('id')}: {function.get('arguments')}"
                            )
                            arguments = {}

                        tool_uses.append(
                            {
                                "id": call.get("id", ""),
                                "name": function.get("name", ""),
                                "input": arguments,
                            }
                        )
                return tool_uses if tool_uses else None
        return None

    def format_assistant_message(self, response: dict) -> dict:
        """Format OpenAI's response for conversation history."""
        if "choices" in response and response["choices"]:
            message = response["choices"][0].get("message", {})
            return {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls"),
            }
        return {"role": "assistant", "content": ""}

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Format tool results for OpenAI API."""
        # OpenAI expects tool results as separate messages with role="tool"
        return [
            {
                "role": "tool",
                "tool_call_id": result["tool_use_id"],
                "content": result["content"],
            }
            for result in tool_results
        ]
