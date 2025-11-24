"""Anthropic API client implementations for Claude and DeepSeek."""

import asyncio
import json
import logging
from typing import Any

import aiohttp

from . import BaseAPIClient

logger = logging.getLogger(__name__)


class BaseAnthropicAPIClient(BaseAPIClient):
    """Base client for Anthropic API-compatible services."""

    def __init__(self, config: dict[str, Any], provider_name: str):
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        cfg = providers.get(provider_name, {})
        super().__init__(cfg)
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
        modalities: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Call Anthropic API with context and system prompt."""

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
            elif m.get("role") == "tool":
                # Handle Chat Completions tool results from OpenAI
                messages.append(
                    {
                        "role": "user",
                        "content": "<tool_results>"
                        + json.dumps({"result": m.get("content", "")})
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
            self.logger.debug(context)
            self.logger.debug(messages)
            return {"cancel": "(wait, I just replied)"}

        thinking_budget = self._get_thinking_budget(reasoning_effort)

        payload = {
            "model": model,
            "max_tokens": max_tokens or ((4096 if tools else 256) + thinking_budget),
            "messages": messages,
            "system": system_prompt,
        }

        if tools:
            payload["tools"] = self._filter_tools(tools)

        self._handle_thinking_budget(payload, thinking_budget, messages)

        if tool_choice:
            self._handle_tool_choice(tool_choice, messages, thinking_budget > 0)

        self.logger.debug(f"Calling {self.provider_name} API with model: {model}")
        self.logger.debug(f"{self.provider_name} request payload: {json.dumps(payload, indent=2)}")

        # Simple retry policy: retry everything with exponential backoff
        backoff_delays = [0, 2, 5, 10, 20]

        for attempt, delay in enumerate(backoff_delays):
            if delay > 0:
                self.logger.info(
                    f"Waiting {delay}s before retry {attempt + 1}/{len(backoff_delays)} for {self.provider_name} API"
                )
                await asyncio.sleep(delay)

            try:
                async with (
                    aiohttp.ClientSession(
                        headers={
                            "x-api-key": self.config["key"],
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json",
                            "User-Agent": "irssi-llmagent/1.0",
                        }
                    ) as session,
                    session.post(self.config["url"], json=payload) as response,
                ):
                    response.raise_for_status()
                    data = await response.json()

                self.logger.debug(f"{self.provider_name} response: {json.dumps(data, indent=2)}")

                # Check for refusal and convert to error
                if data.get("stop_reason") == "refusal":
                    self.logger.warning(f"{self.provider_name} refusal detected")
                    return {"error": "The AI refused to respond to this request (consider !u)"}

                return data

            except Exception as e:
                if attempt < len(backoff_delays) - 1:
                    self.logger.warning(
                        f"{self.provider_name} error: {e}. Retrying in {backoff_delays[attempt + 1]}s..."
                    )
                    continue
                self.logger.error(f"{self.provider_name} error after all retries: {e}")
                return {"error": f"API error: {e}"}

        return {"error": "All retry attempts exhausted"}

    def _get_thinking_budget(self, reasoning_effort: str) -> int:
        """Get thinking budget based on reasoning effort."""
        budget_map = {
            "low": 1024,
            "medium": 4096,
            "high": 16000,
        }
        return budget_map.get(reasoning_effort, 0)

    def _filter_tools(self, tools: list[dict]) -> list[dict]:
        """Filter tool definitions to only include fields supported by Anthropic API."""
        if not tools:
            return []

        filtered_tools = []
        for tool in tools:
            # Only include fields that Anthropic API accepts
            filtered_tool = {
                "name": tool["name"],
                "description": tool["description"],
            }
            if "input_schema" in tool:
                filtered_tool["input_schema"] = tool["input_schema"]
            filtered_tools.append(filtered_tool)

        return filtered_tools

    def _handle_thinking_budget(
        self, payload: dict, thinking_budget: int, messages: list[dict]
    ) -> None:
        """Handle thinking budget - override in subclasses if needed."""
        if thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

    def _handle_tool_choice(
        self, tool_choice: list, messages: list[dict], has_thinking: bool
    ) -> None:
        """Handle tool choice constraints."""
        if has_thinking:
            # As per this documentation: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use
            # > Tool choice limitation: Tool use with thinking only supports tool_choice: {"type": "auto"} (the default) or tool_choice: {"type": "none"}. Using tool_choice: {"type": "any"} or tool_choice: {"type": "tool", "name": "..."} will result in an error because these options force tool use, which is incompatible with extended thinking.
            messages.append(
                {
                    "role": "user",
                    "content": f"<meta>only tool {tool_choice} may be called now</meta>",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"<meta>only tool {tool_choice} may be called now</meta>",
                }
            )

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from Anthropic API response format."""
        if "content" in response and response["content"]:
            # Find text content
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text = content_block["text"]
                    self.logger.debug(f"{self.provider_name} response text: {text}")
                    return text or ""

        return ""

    def has_tool_calls(self, response: dict) -> bool:
        """Check if response contains tool calls."""
        return response.get("stop_reason") == "tool_use"

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from Anthropic API response format."""
        content = response.get("content", [])
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {"id": block["id"], "name": block["name"], "input": block["input"]}
                )
        return tool_uses if tool_uses else None

    def format_assistant_message(self, response: dict) -> dict:
        """Format response for conversation history."""
        content = response.get("content", [])
        return {"role": "assistant", "content": content}

    def format_tool_results(self, tool_results: list[dict]) -> dict:
        """Format tool results for Anthropic API - just wrap in user message."""
        return {"role": "user", "content": tool_results}


class AnthropicClient(BaseAnthropicAPIClient):
    """Anthropic Claude API client with async support."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "anthropic")


class DeepSeekClient(BaseAnthropicAPIClient):
    """DeepSeek API client with Anthropic API compatibility."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "deepseek")

    def _handle_thinking_budget(
        self, payload: dict, thinking_budget: int, messages: list[dict]
    ) -> None:
        """DeepSeek doesn't support thinking budget, use meta message instead."""
        if thinking_budget:
            messages.append(
                {
                    "role": "user",
                    "content": f"<meta>Think step by step in <thinking>...</thinking> (reasoning effort: {thinking_budget} tokens)</meta>",
                }
            )
