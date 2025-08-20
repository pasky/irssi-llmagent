import pytest

from irssi_llmagent.agent import AIAgent


class FakeAPIClient:
    def __init__(self):
        self.calls = []
        self._progress_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    # Mimic API surface used by AIAgent
    async def call_raw(self, messages, system_prompt, model, tools=None, **kwargs):
        # Record tools exposed to model
        self.calls.append(
            {"system_prompt": system_prompt, "tools": [t["name"] for t in (tools or [])]}
        )
        # First turn: request calling progress_report tool if available and not already called
        if (
            (not self._progress_called)
            and tools
            and any(t.get("name") == "progress_report" for t in tools)
        ):
            self._progress_called = True
            # Simulate Claude-style tool_use
            return {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "progress_report",
                        "input": {"text": "Searching docs for foobar"},
                    }
                ],
            }
        # Second turn (after tool results) -> final text
        return {"content": [{"type": "text", "text": "Final answer"}]}

    def has_tool_calls(self, response):
        return response.get("stop_reason") == "tool_use"

    def extract_tool_calls(self, response):
        tool_uses = []
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {"id": block["id"], "name": block["name"], "input": block["input"]}
                )
        return tool_uses

    def extract_text_from_response(self, response):
        # Return text for final_text path
        content = response.get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        return "..."

    def format_assistant_message(self, response):
        return {"role": "assistant", "content": response.get("content", [])}

    def format_tool_results(self, tool_results):
        return {"role": "user", "content": tool_results}


@pytest.mark.asyncio
async def test_progress_report_tool_emits_callback(monkeypatch):
    # Progress callback tracker
    sent = []

    async def progress_cb(text: str):
        sent.append(text)

    # Config with progress thresholds very small to trigger immediately
    config = {
        "default_provider": "anthropic",
        "providers": {"anthropic": {"url": "http://example.com", "key": "dummy"}},
        "command": {
            "models": {
                "serious": "anthropic:dummy-serious",
                "classifier": "anthropic:dummy-classifier",
            },
            "prompts": {
                "serious": "{mynick} at {current_time}",
            },
        },
        "agent": {
            "progress": {
                "threshold_seconds": 0,
                "min_interval_seconds": 0,
            }
        },
    }

    agent = AIAgent(config, mynick="bot", progress_enabled=True, progress_callback=progress_cb)

    # Patch router to use FakeAPIClient
    fake_client = FakeAPIClient()
    call_count = {"n": 0}

    async def fake_call_raw_with_model(*args, **kwargs):
        # args: (model, messages, system_prompt, ...)
        model = args[0]
        messages = args[1]
        system_prompt = args[2]
        if call_count["n"] == 0:
            call_count["n"] += 1
            resp = await fake_client.call_raw(
                messages, system_prompt, model, tools=kwargs.get("tools")
            )
            return resp, fake_client, None
        else:
            # Second call: return final text
            return {"content": [{"type": "text", "text": "Final answer"}]}, fake_client, None

    from unittest.mock import AsyncMock as _AsyncMock
    from unittest.mock import patch as _patch

    with _patch(
        "irssi_llmagent.model_router.ModelRouter.call_raw_with_model",
        new=_AsyncMock(side_effect=fake_call_raw_with_model),
    ):
        # Context can be emptyish; agent ensures a user msg
        context = [{"role": "user", "content": "Hello"}]

        # Run agent without using context manager (fake client doesn't need it here)
        result = await agent.run_agent(context)

    assert result == "Final answer"
    assert sent, "Expected progress callback to be called at least once"
    assert sent[0].startswith("Searching docs")
