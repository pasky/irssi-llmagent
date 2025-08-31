"""OpenAI client using Chat Completions API."""

from __future__ import annotations

import json
import logging
from typing import Any

from . import BaseAPIClient

logger = logging.getLogger(__name__)

try:
    # Lazy import so the package is optional until installed
    from openai import AsyncOpenAI as _AsyncOpenAI
except Exception:  # pragma: no cover - handled at runtime
    _AsyncOpenAI = None  # type: ignore


class BaseOpenAIClient(BaseAPIClient):
    """Base OpenAI API client using Chat Completions API."""

    def __init__(self, config: dict[str, Any], provider_name: str):
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        cfg = providers.get(provider_name, {})
        super().__init__(cfg)
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")

        # Validate required keys
        required_keys = ["key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"{provider_name} config missing required key: {key}")

        self._client: Any | None = None

    def get_base_url(self) -> str | None:
        return self.config.get("base_url")

    async def __aenter__(self):
        if _AsyncOpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run 'uv sync' to install dependencies."
            )
        # Allow custom base_url when provided for proxies/compat
        base_url = self.get_base_url()
        if base_url and base_url.rstrip("/").endswith("/v1"):
            self._client = _AsyncOpenAI(api_key=self.config["key"], base_url=base_url)
        else:
            # Use default API base
            self._client = _AsyncOpenAI(api_key=self.config["key"])
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # AsyncOpenAI does not require explicit close
        self._client = None

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool schema to OpenAI Chat Completion function tools."""
        converted = []
        if not tools:
            return converted
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

    def _is_reasoning_model(self, model):
        return (
            model.startswith("o1")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("gpt-5")
        )

    def _get_extra_body(self, model: str):
        return None, None

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call the OpenAI Chat Completion API and return native response dict."""
        if not self._client:
            raise RuntimeError(
                f"{self.provider_name}Client not initialized as async context manager"
            )

        # O1 and GPT-5 models use max_completion_tokens instead of max_tokens
        is_reasoning_model = self._is_reasoning_model(model)

        # Build standard chat completion messages
        messages = []

        if system_prompt:
            messages.append(
                {"role": "developer" if is_reasoning_model else "system", "content": system_prompt}
            )

        for m in context:
            if isinstance(m, dict):
                if m.get("role") in ("user", "assistant", "tool"):
                    messages.append(m)
                elif m.get("type") == "function_call_output":
                    # Convert to tool message format
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": m.get("call_id", ""),
                            "content": m.get("output", ""),
                        }
                    )

        # Check for duplicate assistant responses
        if len(messages) >= 2 and messages[-1].get("role") == "assistant":
            return {"cancel": "(wait, I just replied)"}

        max_tokens = int(self.config.get("max_tokens", 1024 if tools else 256))

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if is_reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            if reasoning_effort and reasoning_effort != "minimal":
                messages.append(
                    {
                        "role": "user",
                        "content": f"<meta>Think step by step in <thinking>...</thinking> (reasoning effort: {reasoning_effort} - more than minimal)</meta>",
                    }
                )

        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if is_reasoning_model:
                kwargs["tool_choice"] = (
                    {
                        "type": "allowed_tools",
                        "allowed_tools": {
                            "mode": "required",
                            "tools": [
                                {"type": "function", "function": {"name": tool}}
                                for tool in tool_choice
                            ],
                        },
                    }
                    if tool_choice
                    else "auto"
                )
            elif tool_choice:
                # tool_choice with multiple tools is not supported
                messages.append(
                    {
                        "role": "user",
                        "content": f"<meta>only tool {tool_choice} may be called now</meta>",
                    }
                )

        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": "..."})

        self.logger.debug(f"Calling {self.provider_name} Chat Completion API with model: {model}")
        self.logger.debug(
            f"{self.provider_name} Chat Completion request: {json.dumps(kwargs, indent=2)}"
        )

        # Add extra_body if available
        extra_body, model_override = self._get_extra_body(model)
        if extra_body:
            kwargs["extra_body"] = extra_body
            kwargs["model"] = model_override
            self.logger.debug(f"Using extra_body: {extra_body}, model override: {model_override}")

        try:
            resp = await self._client.chat.completions.create(**kwargs)
            data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
            self.logger.debug(
                f"{self.provider_name} Chat Completion response: {json.dumps(data, indent=2)}"
            )
            return data
        except Exception as e:
            msg = repr(e)
            self.logger.error(f"{self.provider_name} Chat Completion API error: {msg}")
            return {"error": f"API error: {msg}"}

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from Chat Completion response."""
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                self.logger.debug(f"{self.provider_name} Chat Completion response text: {content}")
                return content
        return ""

    def has_tool_calls(self, response: dict) -> bool:
        """Check if Chat Completion response contains tool calls."""
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return bool(message.get("tool_calls"))
        return False

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from Chat Completion response."""
        choices = response.get("choices", [])
        if not choices:
            return None

        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return None

        tool_uses = []
        for tc in tool_calls:
            if tc.get("type") == "function":
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except Exception:
                        args_obj = {}
                else:
                    args_obj = args or {}

                tool_uses.append(
                    {
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args_obj,
                    }
                )

        return tool_uses if tool_uses else None

    def format_assistant_message(self, response: dict) -> dict:
        """Format Chat Completion assistant message for conversation history."""
        choices = response.get("choices", [])
        if not choices:
            return {"role": "assistant", "content": ""}

        message = choices[0].get("message", {})
        return {
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
        }

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Format tool results for Chat Completion API as tool messages."""
        processed_results = []
        image_contents = []

        for result in tool_results:
            content = result["content"]
            # Check if content is image data
            if isinstance(content, str) and content.startswith("IMAGE_DATA:"):
                try:
                    _, content_type, size, base64_data = content.split(":", 3)
                    media_type = content_type.split("/")[-1]
                    if media_type in ["jpeg", "png", "gif", "webp"]:
                        # Store image for separate message
                        image_contents.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{content_type};base64,{base64_data}"},
                            }
                        )
                        content = f"Downloaded image ({content_type}, {size} bytes) - Image provided separately"
                    else:
                        content = (
                            f"Downloaded image ({content_type}, {size} bytes) - Unsupported format"
                        )
                except ValueError:
                    # Malformed image data, use as-is
                    pass

            processed_results.append(
                {"role": "tool", "tool_call_id": result["tool_use_id"], "content": str(content)}
            )

        # Add image message if there are images
        if image_contents:
            processed_results.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Here are the images from the website:"}]
                    + image_contents,
                }
            )

        return processed_results


class OpenAIClient(BaseOpenAIClient):
    """OpenAI API client using Chat Completions API."""

    def __init__(self, config: dict[str, Any]):
        # Support new providers.* layout (preferred) and legacy top-level openai
        if "openai" in config:
            providers = {"openai": config["openai"]}
            super().__init__({"providers": providers}, "openai")
        else:
            super().__init__(config, "openai")


class OpenRouterClient(BaseOpenAIClient):
    """OpenRouter API client using OpenAI Chat Completions API compatibility."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "openrouter")

    def _is_reasoning_model(self, model):
        return False

    def _get_extra_body(self, model: str):
        if "#" not in model:
            return None, None

        model_name, provider_list = model.split("#", 1)
        providers = [p.strip() for p in provider_list.split(",") if p.strip()]

        if providers:
            return {"provider": {"only": providers}}, model_name
        return None, None
