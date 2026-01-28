"""AI providers module containing base classes and router."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Pricing in USD per 1M tokens (prompt, completion)
# Updated as of 2025-11
ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"prompt": 3.0, "completion": 15.0},
    "claude-3-5-sonnet-20241022": {"prompt": 3.0, "completion": 15.0},
    "claude-3-5-haiku-20241022": {"prompt": 0.80, "completion": 4.0},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
    "claude-opus-4-5": {"prompt": 5.0, "completion": 25.0},
    "claude-haiku-4-5-20251001": {"prompt": 1.0, "completion": 5.0},
}

OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-2024-11-20": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "o1": {"prompt": 15.0, "completion": 60.0},
    "o1-mini": {"prompt": 1.10, "completion": 4.40},
    "o3-mini": {"prompt": 1.10, "completion": 4.40},
    "gpt-4.1": {"prompt": 2.0, "completion": 8.0},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "gpt-4.1-nano": {"prompt": 0.10, "completion": 0.40},
    "gpt-5-mini": {"prompt": 0.25, "completion": 2.0},
}

DEEPSEEK_PRICING: dict[str, dict[str, float]] = {
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
}


def compute_cost(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Compute cost in USD from token counts. Returns None if model not in pricelist."""
    pricing_map = {
        "anthropic": ANTHROPIC_PRICING,
        "openai": OPENAI_PRICING,
        "deepseek": DEEPSEEK_PRICING,
    }
    pricing = pricing_map.get(provider, {})
    if model not in pricing:
        logger.warning(f"No pricing data for {provider}:{model}, cost will be None")
        return None
    rates = pricing[model]
    prompt_cost = (input_tokens or 0) * rates["prompt"] / 1_000_000
    completion_cost = (output_tokens or 0) * rates["completion"] / 1_000_000
    return prompt_cost + completion_cost


@dataclass
class UsageInfo:
    """Token usage and cost information from an LLM call."""

    input_tokens: int | None
    output_tokens: int | None
    cost: float | None  # USD, None if unknown


class BaseAPIClient(ABC):
    """Abstract base class for AI API clients."""

    provider_name: str = "unknown"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        reasoning_effort: str = "minimal",
        max_tokens: int | None = None,
    ) -> dict:
        """Call API with context and system prompt, returning raw response."""
        pass

    def extract_usage(self, response: dict, model: str) -> UsageInfo:
        """Extract usage info from response. Override in subclasses for provider-specific formats."""
        return UsageInfo(None, None, None)

    async def call(self, context: list[dict], system_prompt: str, model: str) -> str:
        """Call API with context and system prompt, returning cleaned text response."""
        raw_response = await self.call_raw(context, system_prompt, model)
        return self.extract_text_from_response(raw_response)

    def extract_text_from_response(self, response: dict) -> str:
        """Extract cleaned text from raw API response."""
        if "cancel" in response:
            return ""

        if "error" in response:
            return f"Error: {response['error']}"

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

        # Refusal fallback model
        self.refusal_fallback_model = config["router"]["refusal_fallback_model"]

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

    def client_for(self, provider: str):
        return self._ensure_client(provider)

    def _is_refusal(self, response: dict) -> bool:
        """Check if response is a content safety refusal."""
        return (
            isinstance(response, dict)
            and "error" in response
            and "(consider !u)" in response["error"]
        )

    async def call_raw_with_model(
        self,
        model_str: str,
        context: list[dict],
        system_prompt: str,
        *,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
        modalities: list[str] | None = None,
        max_tokens: int | None = None,
        add_fallback_suffix: bool = True,
    ) -> tuple[dict, Any, ModelSpec, UsageInfo]:
        spec = parse_model_spec(model_str)
        client = self.client_for(spec.provider)
        resp = await client.call_raw(
            context,
            system_prompt,
            spec.name,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            modalities=modalities,
            max_tokens=max_tokens,
        )

        # Check for content safety refusal and retry with fallback model
        if self._is_refusal(resp) and self.refusal_fallback_model:
            logger.warning(
                f"Safety refusal on {model_str}; retrying with {self.refusal_fallback_model}"
            )

            fallback_spec = parse_model_spec(self.refusal_fallback_model)
            fallback_client = self.client_for(fallback_spec.provider)
            fallback_resp = await fallback_client.call_raw(
                context,
                system_prompt,
                fallback_spec.name,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                modalities=modalities,
                max_tokens=max_tokens,
            )

            # Append fallback suffix to text responses
            if add_fallback_suffix:
                if isinstance(fallback_resp, dict) and "content" in fallback_resp:
                    # Anthropic format
                    for block in fallback_resp.get("content", []):
                        if block.get("type") == "text":
                            block["text"] += f" [refusal fallback to {fallback_spec.name}]"
                            break
                elif isinstance(fallback_resp, dict) and "choices" in fallback_resp:
                    # OpenAI format
                    choices = fallback_resp.get("choices", [])
                    if choices and "message" in choices[0]:
                        msg = choices[0]["message"]
                        if "content" in msg and isinstance(msg["content"], str):
                            msg["content"] += f" [refusal fallback to {fallback_spec.name}]"

            usage = fallback_client.extract_usage(fallback_resp, fallback_spec.name)
            return fallback_resp, fallback_client, fallback_spec, usage

        usage = client.extract_usage(resp, spec.name)
        return resp, client, spec, usage
