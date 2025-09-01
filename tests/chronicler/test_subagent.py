import json
from unittest.mock import AsyncMock, patch

import pytest

from irssi_llmagent.chronicler.subagent import run_chronicler
from irssi_llmagent.providers import ModelSpec


@pytest.mark.asyncio
async def test_chronicler_subagent_appends_then_finishes(test_config, temp_db_path, api_type):
    # Add chronicler config
    test_config["chronicler"] = {
        "model": f"{api_type}:test-model",
        "database": {"path": temp_db_path},
    }
    arc = "proj-x"

    # Construct provider-shaped responses
    if api_type == "anthropic":
        tool_use_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_abc",
                    "name": "chapter_append",
                    "input": {"arc": arc, "text": "Initial setup complete."},
                }
            ],
            "stop_reason": "tool_use",
        }
        final_response = {
            "content": [{"type": "text", "text": "OK"}],
            "stop_reason": "end_turn",
        }
    else:  # openai (Responses API)
        tool_use_response = {
            "output": [
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "id": "tool_abc",
                                "function": {
                                    "name": "chapter_append",
                                    "arguments": json.dumps(
                                        {"arc": arc, "text": "Initial setup complete."}
                                    ),
                                },
                            }
                        ],
                    },
                }
            ]
        }
        final_response = {"output_text": "OK"}

    class FakeClient:
        def extract_text_from_response(self, r):
            if r is final_response:
                return "OK"
            return ""

        def has_tool_calls(self, r):
            if api_type == "anthropic":
                return r.get("stop_reason") == "tool_use"
            else:
                return any(item.get("type") == "message" for item in r.get("output", []))

        def extract_tool_calls(self, r):
            # Match shapes similar to tests for the main actor
            if api_type == "anthropic":
                return [
                    {
                        "id": b.get("id"),
                        "name": b.get("name"),
                        "input": b.get("input", {}),
                    }
                    for b in r.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ] or None
            outputs = r.get("output") or []
            calls = []
            for item in outputs:
                if item.get("type") == "message":
                    msg = item.get("message") if isinstance(item.get("message"), dict) else item
                    for c in msg.get("content", []):
                        if c.get("type") == "tool_call":
                            fn = c.get("function", {})
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    args = {}
                            calls.append({"id": c.get("id"), "name": fn.get("name"), "input": args})
            return calls or None

        def format_assistant_message(self, r):
            if api_type == "anthropic":
                return {"role": "assistant", "content": r.get("content", [])}
            else:
                return {"role": "assistant", "content": []}

        def format_tool_results(self, results):
            return {"role": "user", "content": results}

    seq = [tool_use_response, final_response]

    async def fake_call_raw_with_model(*args, **kwargs):
        return seq.pop(0), FakeClient(), ModelSpec(api_type, "dummy")

    with patch(
        "irssi_llmagent.providers.ModelRouter.call_raw_with_model",
        new=AsyncMock(side_effect=fake_call_raw_with_model),
    ):
        # Create agent instance
        from unittest.mock import MagicMock

        from irssi_llmagent.chronicler.chronicle import Chronicle

        agent = MagicMock()
        agent.config = test_config

        # Create actual chronicle instance
        agent.chronicle = Chronicle(temp_db_path)
        await agent.chronicle.initialize()

        # Run subagent
        out = await run_chronicler(agent, arc=arc, instructions="Record: Initial setup complete.")
        assert out.strip() == "OK"

        # Verify content is stored and can be rendered
        rendered = await agent.chronicle.render_chapter(arc)
        assert "Initial setup complete." in rendered
