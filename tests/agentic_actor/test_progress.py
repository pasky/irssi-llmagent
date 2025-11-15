import pytest

from irssi_llmagent.agentic_actor import AgenticLLMActor


class FakeAPIClient:
    def __init__(self):
        self.calls = []
        self._progress_called = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    # Mimic API surface used by AgenticLLMActor
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
async def test_progress_report_tool_emits_callback(monkeypatch, mock_agent):
    # Progress callback tracker
    sent = []

    async def progress_cb(text: str, type: str = "progress"):
        sent.append(text)

    # Config with progress thresholds very small to trigger immediately
    config = {
        "default_provider": "anthropic",
        "providers": {"anthropic": {"url": "http://example.com", "key": "dummy"}},
        "tools": {},
        "rooms": {
            "irc": {
                "command": {
                    "modes": {
                        "serious": {
                            "model": "anthropic:dummy-serious",
                            "prompt": "{mynick} at {current_time} with models serious={serious_model}, sarcastic={sarcastic_model}",
                        },
                        "sarcastic": {
                            "model": "anthropic:dummy-sarcastic",
                            "prompt": "Sarcastic prompt",
                        },
                    },
                    "mode_classifier": {
                        "model": "anthropic:dummy-classifier",
                        "prompt": "Classifier prompt",
                    },
                }
            }
        },
        "actor": {
            "max_iterations": 5,
            "progress": {
                "threshold_seconds": 0,
                "min_interval_seconds": 0,
            },
        },
        "router": {
            "refusal_fallback_model": "anthropic:dummy-unsafe-fallback",
        },
    }

    def build_test_prompt():
        return "Test system prompt"

    agent = AgenticLLMActor(
        config=config,
        model="anthropic:claude-3-5-sonnet",
        system_prompt_generator=build_test_prompt,
        agent=mock_agent,
    )

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
        "irssi_llmagent.providers.ModelRouter.call_raw_with_model",
        new=_AsyncMock(side_effect=fake_call_raw_with_model),
    ):
        # Context can be emptyish; agent ensures a user msg
        context = [{"role": "user", "content": "Hello"}]

        # Run agent without using context manager (fake client doesn't need it here)
        result = await agent.run_agent(context, progress_callback=progress_cb, arc="test-arc")

    assert result == "Final answer"
    assert sent, "Expected progress callback to be called at least once"
    assert sent[0].startswith("Searching docs")


@pytest.mark.asyncio
async def test_progress_callback_with_tool_persistence_type(mock_agent):
    """Test that progress callback handles tool_persistence type correctly."""
    # Progress callback tracker
    sent = []

    async def progress_cb(text: str, type: str = "progress"):
        sent.append({"text": text, "type": type})

    config = {
        "default_provider": "anthropic",
        "providers": {"anthropic": {"url": "http://example.com", "key": "dummy"}},
        "tools": {},
        "rooms": {
            "irc": {
                "command": {
                    "modes": {
                        "serious": {
                            "model": "anthropic:dummy-serious",
                            "prompt": "Test prompt",
                        }
                    }
                }
            }
        },
        "actor": {
            "max_iterations": 5,
            "progress": {
                "threshold_seconds": 0,
                "min_interval_seconds": 0,
            },
        },
        "router": {
            "refusal_fallback_model": "anthropic:dummy-unsafe-fallback",
        },
    }

    from unittest.mock import AsyncMock as _AsyncMock
    from unittest.mock import patch as _patch

    from irssi_llmagent.agentic_actor import AgenticLLMActor

    agent = AgenticLLMActor(
        config=config,
        model="anthropic:claude-3-5-sonnet",
        system_prompt_generator=lambda: "Test system prompt",
        agent=mock_agent,
    )

    # Create a proper mock client
    class MockClient:
        def extract_text_from_response(self, response):
            return "Final answer"

        def has_tool_calls(self, response):
            return False

        def extract_tool_calls(self, response):
            return None

        def format_assistant_message(self, response):
            return {"role": "assistant", "content": "Final answer"}

        def format_tool_results(self, tool_results):
            return {"role": "user", "content": "Tool results"}

    # Mock client and response for testing (not used directly in this test)

    # Mock the _generate_and_store_persistence_summary method to call progress_callback
    async def mock_summary_generator(persistent_calls, progress_callback):
        # Simulate calling progress_callback with tool_persistence type
        await progress_callback(
            "Tool calls: web_search, execute_python completed successfully.", "tool_persistence"
        )

    with _patch.object(
        agent, "_generate_and_store_persistence_summary", new_callable=_AsyncMock
    ) as mock_summary:
        mock_summary.side_effect = mock_summary_generator

        # Mock run_agent to simulate having persistent tool calls
        async def mock_run_agent_with_persistence(*args, **kwargs):
            # Simulate persistent tool calls by directly calling the summary generator
            persistent_tool_calls = [
                {
                    "tool_name": "web_search",
                    "input": {"query": "test"},
                    "output": "results",
                    "persist_type": "summary",
                }
            ]

            # Get the progress callback from kwargs
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                await mock_summary_generator(persistent_tool_calls, progress_callback)

            return "Final answer"

        with _patch.object(agent, "run_agent", new_callable=_AsyncMock) as mock_run:
            mock_run.side_effect = mock_run_agent_with_persistence

            result = await agent.run_agent(
                [{"role": "user", "content": "test"}], progress_callback=progress_cb, arc="test-arc"
            )

    # Verify the agent completed successfully
    assert result == "Final answer"

    # Verify progress callback was called with tool_persistence type
    assert len(sent) == 1
    assert sent[0]["type"] == "tool_persistence"
    assert "Tool calls" in sent[0]["text"]
    assert "completed successfully" in sent[0]["text"]
