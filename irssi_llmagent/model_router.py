from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from .claude import AnthropicClient
from .openai import OpenAIClient


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
            client = AnthropicClient(self.config)
        elif provider == "openai":
            client = OpenAIClient(self.config)
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
