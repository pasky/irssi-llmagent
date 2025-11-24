"""OpenAI client using Chat Completions API."""

from __future__ import annotations

import asyncio
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


class SoftAPIError(Exception):
    """Exception for API errors returned in 200 OK responses."""

    def __init__(self, message: str, code: int | str | None):
        self.message = message
        self.status_code = code
        super().__init__(f"API Error {code}: {message}")


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

    def get_base_url(self) -> str | None:
        return self.config.get("base_url")

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
        modalities: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Call the OpenAI Chat Completion API and return native response dict."""
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
                    # Preserve reasoning_details when passing assistant messages
                    msg = {k: v for k, v in m.items() if v is not None}
                    messages.append(msg)
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

        if max_tokens is None:
            max_tokens = int(self.config.get("max_tokens", 1024 if tools else 256))

        kwargs = {
            "model": model,
            "messages": messages,
        }

        if modalities:
            kwargs["modalities"] = modalities

        if is_reasoning_model:
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            if reasoning_effort and reasoning_effort != "minimal" and "gemini" not in model:
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

        if not messages or messages[-1].get("role") not in ("user", "tool"):
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

        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
                self.logger.debug(
                    f"{self.provider_name} Chat Completion response: {json.dumps(data, indent=2)}"
                )

                # Check for API errors returned in 200 OK response (common with proxies/OpenRouter)
                if "error" in data and isinstance(data["error"], dict):
                    error = data["error"]
                    raise SoftAPIError(error.get("message", "Unknown error"), error.get("code"))

                return data
            except Exception as e:
                # Check if we should retry based on exception type/status
                is_retryable = False
                status_code = getattr(e, "status_code", None)

                if status_code:
                    try:
                        code_int = int(status_code)
                        if code_int >= 500:
                            is_retryable = True
                    except (ValueError, TypeError):
                        pass

                if is_retryable and attempt < max_retries - 1:
                    delay = 2**attempt
                    self.logger.warning(
                        f"{self.provider_name} request failed (status {status_code}), retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Check for OpenAI content safety errors and convert to refusal format
                # Note: OpenAI SDK extracts body.get("error", body) so .body is the error dict directly
                error_body = getattr(e, "body", None)
                if isinstance(error_body, dict):
                    error_code = error_body.get("code")
                    error_message = error_body.get("message", "")
                    if error_code == "invalid_prompt" and "safety reasons" in error_message:
                        self.logger.warning(
                            f"{self.provider_name} content safety refusal: {error_message}"
                        )
                        return {"error": f"{error_message} (consider !u)"}

                msg = repr(e)
                self.logger.error(f"{self.provider_name} Chat Completion API error: {msg}")
                return {"error": f"API error: {msg}"}

        return {"error": "Max retries exceeded"}

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
        formatted = {
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
        }

        # Preserve reasoning_details for OpenRouter reasoning models
        reasoning_details = message.get("reasoning_details")
        if reasoning_details:
            formatted["reasoning_details"] = reasoning_details
            self.logger.debug(f"Preserving reasoning_details with {len(reasoning_details)} blocks")

        return formatted

    def _split_blocks_to_openai(self, blocks: list[dict]) -> tuple[str, list[dict]]:
        """Convert Anthropic content blocks to OpenAI format.

        Returns:
            (text_content, image_parts) where:
            - text_content: concatenated text from all text blocks
            - image_parts: list of image_url dicts for OpenAI format
        """
        texts = []
        image_parts = []

        for block in blocks:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64":
                    mime = source.get("media_type") or "image/png"
                    data = source.get("data") or ""
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        }
                    )

        text_content = "\n\n".join(t for t in texts if t)
        return text_content, image_parts

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Format tool results for Chat Completion API as tool messages."""
        processed_results = []
        accumulated_images = []

        for result in tool_results:
            content = result["content"]

            # Handle plain strings vs Anthropic content blocks
            if isinstance(content, str):
                text = content
                images = []
            else:
                # Content is Anthropic blocks, convert to OpenAI format
                text, images = self._split_blocks_to_openai(content)

            # Tool message must have text content
            tool_content = text or ("Images returned by tool." if images else "")
            processed_results.append(
                {"role": "tool", "tool_call_id": result["tool_use_id"], "content": tool_content}
            )

            accumulated_images.extend(images)

        # Add image message if there are images
        if accumulated_images:
            processed_results.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are the images from the tool results:"}
                    ]
                    + accumulated_images,
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
