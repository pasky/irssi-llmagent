"""Base class for AI API clients."""

import re
from abc import ABC, abstractmethod
from datetime import datetime
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
        if not text:
            return "..."

        # Clean up response for IRC
        text = text.strip()

        # Remove thinking tags and content if present
        text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL)

        # For IRC: single line only, take first line of remaining content
        text = text.replace("\n", "; ").strip()

        # Remove IRC nick prefix
        text = re.sub(r"^(\[..:..\]\s*)?<[^>]+>\s*", "", text)

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


def build_system_prompt(config: dict[str, Any], prompt_key: str, mynick: str) -> str:
    """Build a command system prompt with standard substitutions.

    Args:
        config: Configuration dictionary
        prompt_key: Key in config["command"]["prompts"] (e.g., "serious", "sarcastic")
        mynick: IRC nickname for substitution

    Returns:
        Formatted system prompt with all substitutions applied
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Get model configurations for context
    sarcastic_model = config["command"]["models"]["sarcastic"]
    serious_cfg = config["command"]["models"]["serious"]
    serious_model = serious_cfg[0] if isinstance(serious_cfg, list) and serious_cfg else serious_cfg

    # Get the prompt template from command section
    try:
        prompt_template = config["command"]["prompts"][prompt_key]
    except KeyError:
        raise ValueError(f"Command prompt key '{prompt_key}' not found in config") from None

    return prompt_template.format(
        mynick=mynick,
        current_time=current_time,
        sarcastic_model=sarcastic_model,
        serious_model=serious_model,
    )
