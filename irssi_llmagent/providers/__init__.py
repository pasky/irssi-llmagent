"""AI providers module containing base classes and router."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


class BaseAPIClient(ABC):
    """Abstract base class for AI API clients."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.session = None

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    @abstractmethod
    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call API with context and system prompt, returning raw response."""
        pass

    async def call(self, context: list[dict], system_prompt: str, model: str) -> str:
        """Call API with context and system prompt, returning cleaned text response."""
        raw_response = await self.call_raw(context, system_prompt, model)
        return self.extract_text_from_response(raw_response)

    def extract_text_from_response(self, response: dict) -> str:
        """Extract cleaned text from raw API response."""
        if "cancel" in response:
            return ""

        if "error" in response:
            return f"Error - {response['error']}"

        text = self._extract_raw_text(response)
        return self.cleanup_raw_text(text)

    def cleanup_raw_text(self, text: str) -> str:
        if not text:
            return "..."

        text = text.strip()
        text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)

        return text

    @abstractmethod
    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from API-specific response format."""
        pass

    @abstractmethod
    def has_tool_calls(self, response: dict) -> bool:
        """Check if response contains tool calls."""
        pass

    @abstractmethod
    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from API-specific response format.

        Returns list of dicts with 'id', 'name', 'input' keys or None if no tools.
        """
        pass

    @abstractmethod
    def format_assistant_message(self, response: dict) -> dict:
        """Format the assistant's response for conversation history."""
        pass

    @abstractmethod
    def format_tool_results(self, tool_results: list[dict]) -> dict | list[dict]:
        """Format tool results for the next API call."""
        pass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str


def parse_model_spec(model_str: str) -> ModelSpec:
    s = (model_str or "").strip()
    if ":" in s:
        p, m = s.split(":", 1)
        return ModelSpec(p.strip(), m.strip())
    raise ValueError(
        f"Model '{s}' must be fully-qualified as provider:model (e.g., anthropic:claude-4)"
    )


class ModelRouter:
    """Hardcoded provider router for existing providers (anthropic, openai).

    Creates and holds an async client per provider for reuse during a scope.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._clients: dict[str, Any] = {}
        # No default provider; models must be fully-qualified

    async def __aenter__(self):
        # Lazy init of clients
        return self

    async def __aexit__(self, exc_type, exc, tb):
        for client in self._clients.values():
            with suppress(Exception):
                await client.__aexit__(exc_type, exc, tb)
        self._clients.clear()

    def _ensure_client(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]
        if provider == "anthropic":
            from .anthropic import AnthropicClient

            client = AnthropicClient(self.config)
        elif provider == "deepseek":
            from .anthropic import DeepSeekClient

            client = DeepSeekClient(self.config)
        elif provider == "openai":
            from .openai import OpenAIClient

            client = OpenAIClient(self.config)
        elif provider == "openrouter":
            from .openai import OpenRouterClient

            client = OpenRouterClient(self.config)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self._clients[provider] = client
        return client

    async def client_for(self, provider: str):
        client = self._ensure_client(provider)
        # Enter async context once per client
        # Use a marker attribute to avoid re-entering
        if not getattr(client, "_entered", False):
            await client.__aenter__()
            client._entered = True
        return client

    async def call_raw_with_model(
        self,
        model_str: str,
        context: list[dict],
        system_prompt: str,
        *,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
    ) -> tuple[dict, Any, ModelSpec]:
        spec = parse_model_spec(model_str)
        client = await self.client_for(spec.provider)
        resp = await client.call_raw(
            context,
            system_prompt,
            spec.name,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
        )
        return resp, client, spec
