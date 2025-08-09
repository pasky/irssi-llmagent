"""OpenAI client using the official Python SDK (Responses API)."""

from __future__ import annotations

import json
import logging
from typing import Any

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)

try:
    # Lazy import so the package is optional until installed
    from openai import AsyncOpenAI as _AsyncOpenAI
except Exception:  # pragma: no cover - handled at runtime
    _AsyncOpenAI = None  # type: ignore


class OpenAIClient(BaseAPIClient):
    """OpenAI API client backed by the Responses API via the Python SDK."""

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

        self._client: Any | None = None

    async def __aenter__(self):
        if _AsyncOpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run 'uv sync' to install dependencies."
            )
        # Allow custom base_url when provided for proxies/compat
        base_url = self.config.get("base_url")
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
        """Convert internal tool schema to OpenAI function tools."""
        converted = []
        if not tools:
            return converted
        for tool in tools:
            # Responses API expects name at top-level for function tools
            converted.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
            )
        return converted

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: str | None = None,
        reasoning_effort: str = "minimal",
    ) -> dict:
        """Call the OpenAI Responses API and return native response dict."""
        if not self._client:
            raise RuntimeError("OpenAIClient not initialized as async context manager")

        # Build messages array from context (user/assistant/tool). System prompt via instructions.
        messages = list(context)
        if not messages or all(m.get("role") != "user" for m in messages):
            messages.append({"role": "user", "content": "..."})

        # If the last role is assistant without tool_calls, avoid re-answering
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and "tool_calls" not in last:
            return {"cancel": "(wait, I just replied)"}

        # Build Responses API input as a list of chat messages
        inputs: list[dict[str, Any]] = []
        if system_prompt:
            inputs.append({"role": "system", "content": system_prompt})
        for m in messages:
            # Allow passing native function_call_output items straight through
            if m.get("type") == "function_call_output":
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.get("call_id"),
                        "output": m.get("output", "{}"),
                    }
                )
                continue
            # Convert prior assistant tool_calls into Responses function_call items
            if isinstance(m.get("tool_calls"), list):
                for tc in m["tool_calls"]:
                    if tc.get("type") == "function":
                        name = tc.get("function", {}).get("name", "")
                        args = tc.get("function", {}).get("arguments", "{}")
                        call_id = tc.get("id") or ""
                        inputs.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": args,
                            }
                        )
                # Also include the assistant message text if any
            role = m.get("role", "user")
            assert role != "tool"
            content_val = m.get("content") or ""
            inputs.append({"role": role, "content": content_val})

        max_tokens = int(self.config.get("max_tokens", 1024 if tools else 256))

        sdk_kwargs: dict[str, Any] = {
            "model": model,
            "input": inputs,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": reasoning_effort, "summary": "auto"},
        }
        if tools:
            sdk_kwargs["tools"] = self._convert_tools(tools)
            sdk_kwargs["tool_choice"] = self.config.get("tool_choice", "auto")

        logger.debug(f"Calling OpenAI Responses API with model: {model}")
        try:
            resp = await self._client.responses.create(**sdk_kwargs)
            # Convert SDK obj to plain dict for robust handling
            data = resp.model_dump() if hasattr(resp, "model_dump") else json.loads(resp.json())
            logger.debug(f"OpenAI raw Responses payload: {json.dumps(data, indent=2)}")
        except Exception as e:  # Broad catch to mirror previous network error handling
            msg = repr(e)
            logger.error(f"OpenAI API error: {msg}")
            return {"error": f"API error: {msg}"}

        # Clean up potential null error fields that confuse our unified handler
        if isinstance(data, dict) and data.get("error", None) is None:
            data.pop("error", None)

        # Return Responses-native dict
        return data

    def _extract_raw_text(self, response: dict) -> str:
        """Extract raw text content from Responses-native dict."""
        # Prefer convenience field
        text = response.get("output_text") or response.get("text")
        if isinstance(text, str) and text:
            logger.debug(f"OpenAI response text: {text}")
            return text
        # Fallback: scan outputs
        outputs = response.get("output") or response.get("outputs") or []
        acc = []
        if isinstance(outputs, list):
            for item in outputs:
                if item.get("type") == "message":
                    # Responses may nest message under 'message' or inline at top level
                    msg = item.get("message") if isinstance(item.get("message"), dict) else item
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                        for c in msg["content"]:
                            if c.get("type") in ("text", "output_text"):
                                piece = c.get("text") or c.get("value") or ""
                                if isinstance(piece, str):
                                    acc.append(piece)
        return "".join(acc)

    def has_tool_calls(self, response: dict) -> bool:
        """Check if Responses-native output contains tool calls."""
        outputs = response.get("output") or response.get("outputs") or []
        if isinstance(outputs, list):
            for item in outputs:
                if item.get("type") == "message":
                    msg = item.get("message") if isinstance(item.get("message"), dict) else item
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                        for c in msg["content"]:
                            if c.get("type") in ("tool_call", "function_call"):
                                return True
                if item.get("type") in ("tool_calls", "tool_call", "function_call"):
                    return True
        return False

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from Responses-native output."""
        tool_uses: list[dict] = []
        outputs = response.get("output") or response.get("outputs") or []
        if isinstance(outputs, list):
            for item in outputs:
                if item.get("type") == "message":
                    msg = item.get("message") if isinstance(item.get("message"), dict) else item
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                        for c in msg["content"]:
                            if c.get("type") in ("tool_call", "function_call"):
                                fc = c.get("function", {})
                                args = fc.get("arguments")
                                if isinstance(args, str):
                                    try:
                                        args_obj = json.loads(args)
                                    except Exception:
                                        args_obj = {}
                                else:
                                    args_obj = args or {}
                                tool_uses.append(
                                    {
                                        "id": c.get("id", ""),
                                        "name": fc.get("name", ""),
                                        "input": args_obj,
                                    }
                                )
                if item.get("type") in ("tool_calls", "tool_call"):
                    for call in item.get("tool_calls", [item]):
                        if not isinstance(call, dict):
                            continue
                        fc = call.get("function", {})
                        args = fc.get("arguments")
                        if isinstance(args, str):
                            try:
                                args_obj = json.loads(args)
                            except Exception:
                                args_obj = {}
                        else:
                            args_obj = args or {}
                        tool_uses.append(
                            {
                                "id": call.get("id", ""),
                                "name": fc.get("name", ""),
                                "input": args_obj,
                            }
                        )
                if item.get("type") == "function_call":
                    args = item.get("arguments")
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except Exception:
                            args_obj = {}
                    else:
                        args_obj = args or {}
                    tool_uses.append(
                        {
                            "id": item.get("call_id") or item.get("id", ""),
                            "name": item.get("name", ""),
                            "input": args_obj,
                        }
                    )
        return tool_uses if tool_uses else None

    def format_assistant_message(self, response: dict) -> dict:
        """Format assistant message for conversation history from Responses-native output."""
        content_text = self._extract_raw_text(response) or None
        tool_calls_out: list[dict] = []
        outputs = response.get("output") or response.get("outputs") or []
        if isinstance(outputs, list):
            for item in outputs:
                if item.get("type") == "message":
                    msg = item.get("message") if isinstance(item.get("message"), dict) else item
                    if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                        for c in msg["content"]:
                            if c.get("type") in ("tool_call", "function_call"):
                                fc = c.get("function", {})
                                args = fc.get("arguments")
                                args_json = (
                                    args if isinstance(args, str) else json.dumps(args or {})
                                )
                                tool_calls_out.append(
                                    {
                                        "id": c.get("id") or "tool_call",
                                        "type": "function",
                                        "function": {
                                            "name": fc.get("name", ""),
                                            "arguments": args_json,
                                        },
                                    }
                                )
                if item.get("type") == "function_call":
                    args = item.get("arguments")
                    args_json = args if isinstance(args, str) else json.dumps(args or {})
                    tool_calls_out.append(
                        {
                            "id": item.get("call_id") or item.get("id") or "tool_call",
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": args_json,
                            },
                        }
                    )
        return {"role": "assistant", "content": content_text, "tool_calls": tool_calls_out or None}

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Format tool results for OpenAI Responses API as function_call_output items."""
        return [
            {
                "type": "function_call_output",
                "call_id": result["tool_use_id"],
                "output": json.dumps({"result": result["content"]}),
            }
            for result in tool_results
        ]
