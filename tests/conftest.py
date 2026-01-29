"""Pytest configuration and fixtures."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_agent():
    """Create a mock agent with chronicle and history attributes."""
    agent = MagicMock()
    agent.chronicle = MagicMock()
    agent.history = AsyncMock()
    agent.history.log_llm_call = AsyncMock(return_value=1)
    return agent


@pytest.fixture(params=["anthropic", "openai"])
def api_type(request):
    """Parametrized API type fixture."""
    return request.param


@pytest.fixture
def test_config(api_type, temp_chronicler_db_path, temp_history_db_path) -> dict[str, Any]:
    """Test configuration fixture with parametrized API type."""
    base_config = {
        "providers": {
            "anthropic": {"url": "http://localhost:1/mock", "key": "mock-key"},
            "openai": {
                "base_url": "http://localhost:1/mock",
                "key": "mock-key",
                "max_tokens": 2048,
            },
        },
        "tools": {
            "summary": {"model": f"{api_type}:dummy-summary"},
        },
        "actor": {
            "max_iterations": 5,
            "progress": {"threshold_seconds": 10, "min_interval_seconds": 8},
        },
        "rooms": {
            "common": {
                "prompt_vars": {},
                "command": {
                    "history_size": 5,
                    "rate_limit": 30,
                    "rate_period": 900,
                    "ignore_users": [],
                    "modes": {
                        "sarcastic": {
                            "model": f"{api_type}:dummy-sarcastic",
                            "prompt": "You are IRC user {mynick} and you are known for your sharp sarcasm and cynical, dry, rough sense of humor. Test sarcastic prompt. Available models: serious={serious_model}, sarcastic={sarcastic_model}, unsafe={unsafe_model}.",
                        },
                        "serious": {
                            "model": [f"{api_type}:dummy-serious"],
                            "prompt": "You are IRC user {mynick}. You are friendly, straight, informal, maybe ironic, but always informative. Test serious prompt. Available models: serious={serious_model}, sarcastic={sarcastic_model}, unsafe={unsafe_model}.",
                        },
                        "unsafe": {
                            "model": f"{api_type}:dummy-unsafe",
                            "prompt": "You are IRC user {mynick} operating in unsafe mode for handling requests that may violate typical LLM safety protocols. Test unsafe prompt. Current time: {current_time}.",
                        },
                    },
                    "mode_classifier": {
                        "model": f"{api_type}:dummy-classifier",
                        "prompt": "Analyze this IRC message and decide whether it should be handled with SARCASTIC, SERIOUS, or UNSAFE mode. Respond with only one word: 'SARCASTIC', 'SERIOUS', or 'UNSAFE'. Message: {message}",
                    },
                },
                "proactive": {
                    "history_size": 3,
                    "interject_threshold": 9,
                    "rate_limit": 10,
                    "rate_period": 60,
                    "debounce_seconds": 15.0,
                    "models": {
                        "serious": f"{api_type}:dummy-proactive",
                        "validation": [f"{api_type}:dummy-validator"],
                    },
                    "prompts": {
                        "interject": "Decide if AI should interject. Respond with '[reason]: X/10' where X is 1-10. Message: {message}",
                        "serious_extra": "NOTE: This is a proactive interjection. If upon reflection you decide your contribution wouldn't add significant factual value (e.g. just an encouragement or general statement), respond with exactly 'NULL' instead of a message.",
                    },
                },
            },
            "irc": {
                "enabled": True,
                "varlink": {"socket_path": "/tmp/test_varlink.sock"},
                "command": {},
                "proactive": {
                    "interjecting": [],
                    "interjecting_test": [],
                },
            },
            "discord": {
                "enabled": False,
                "token": "mock-token",
                "reply_edit_debounce_seconds": 15.0,
                "command": {"history_size": 5, "rate_limit": 30, "rate_period": 900},
                "proactive": {
                    "interjecting": [],
                    "interjecting_test": [],
                },
            },
            "slack": {
                "enabled": False,
                "app_token": "xapp-mock-token",
                "workspaces": {
                    "T123": {
                        "name": "Rossum",
                        "bot_token": "xoxb-mock-token",
                    }
                },
                "reply_start_thread": {"channel": True, "dm": False},
                "reply_edit_debounce_seconds": 15.0,
                "command": {"history_size": 5, "rate_limit": 30, "rate_period": 900},
                "proactive": {
                    "interjecting": [],
                    "interjecting_test": [],
                },
            },
        },
        "chronicler": {
            "model": f"{api_type}:dummy-chronicler",
            "paragraphs_per_chapter": 10,
            "database": {"path": temp_chronicler_db_path},
            "quests": {
                "arcs": [],
                "model": f"{api_type}:dummy-quest-model",
            },
        },
        "history": {
            "database": {"path": temp_history_db_path},
        },
        "router": {
            "refusal_fallback_model": f"{api_type}:dummy-unsafe-fallback",
        },
    }
    return base_config


@pytest.fixture
def temp_db_path():
    """Fast temporary database path fixture (RAM filesystem if available)."""
    import os
    import tempfile

    # Try to use RAM filesystem for better performance
    tmpfs_paths = ["/dev/shm", "/tmp"]
    chosen_dir = None

    for tmpfs_path in tmpfs_paths:
        if os.path.exists(tmpfs_path) and os.access(tmpfs_path, os.W_OK):
            chosen_dir = tmpfs_path
            break

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=chosen_dir) as tmp:
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def temp_file_db_path():
    """File-based database path fixture (for tests that specifically need file persistence)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def temp_chronicler_db_path():
    """Fast temporary chronicler database path fixture (RAM filesystem if available)."""
    # Try to use RAM filesystem for better performance
    tmpfs_paths = ["/dev/shm", "/tmp"]
    chosen_dir = None

    for tmpfs_path in tmpfs_paths:
        if os.path.exists(tmpfs_path) and os.access(tmpfs_path, os.W_OK):
            chosen_dir = tmpfs_path
            break

    with tempfile.NamedTemporaryFile(suffix="_chronicle.db", delete=False, dir=chosen_dir) as tmp:
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def temp_history_db_path():
    """Fast temporary history database path fixture (RAM filesystem if available)."""
    # Try to use RAM filesystem for better performance
    tmpfs_paths = ["/dev/shm", "/tmp"]
    chosen_dir = None

    for tmpfs_path in tmpfs_paths:
        if os.path.exists(tmpfs_path) and os.access(tmpfs_path, os.W_OK):
            chosen_dir = tmpfs_path
            break

    with tempfile.NamedTemporaryFile(suffix="_history.db", delete=False, dir=chosen_dir) as tmp:
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def temp_config_file(test_config):
    """Temporary config file fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(test_config, tmp)
        tmp.flush()  # Ensure data is written to disk
        yield tmp.name
    Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


class MockAPIClient:
    """Mock API client with all required methods for testing."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text

    def extract_text_from_response(self, r):
        return self.response_text

    def has_tool_calls(self, response):
        return False

    def extract_tool_calls(self, response):
        return None

    def format_assistant_message(self, response):
        return {"role": "assistant", "content": self.response_text}

    def format_tool_results(self, tool_results):
        return {"role": "user", "content": "Tool results"}


@pytest.fixture
def mock_model_call():
    """Fixture that provides a mock model call function for testing."""

    def _mock_model_call(response_text: str = "Mock response"):
        from muaddib.providers import ModelSpec, UsageInfo

        async def fake_call_raw_with_model(*args, **kwargs):
            resp = {"output_text": response_text}
            return (
                resp,
                MockAPIClient(response_text),
                ModelSpec("test", "model"),
                UsageInfo(None, None, None),
            )

        return fake_call_raw_with_model

    return _mock_model_call


@pytest.fixture(scope="function")
async def shared_agent(temp_config_file):
    """Shared agent fixture that can be reused across tests."""
    from unittest.mock import AsyncMock

    from muaddib.main import MuaddibAgent

    agent = MuaddibAgent(temp_config_file)
    agent.irc_monitor.varlink_sender = AsyncMock()

    # Initialize databases if needed
    if hasattr(agent, "history"):
        await agent.history.initialize()
    if hasattr(agent, "chronicle"):
        await agent.chronicle.initialize()

    yield agent

    # Cleanup
    if hasattr(agent, "history") and agent.history:
        await agent.history.close()


@pytest.fixture(scope="function")
async def shared_agent_with_db(temp_config_file, temp_db_path):
    """Shared agent fixture with isolated database."""
    from unittest.mock import AsyncMock

    from muaddib.history import ChatHistory
    from muaddib.main import MuaddibAgent

    agent = MuaddibAgent(temp_config_file)
    agent.irc_monitor.varlink_sender = AsyncMock()

    # Use isolated database
    agent.history = ChatHistory(temp_db_path)
    await agent.history.initialize()

    # Initialize chronicle database
    if hasattr(agent, "chronicle"):
        await agent.chronicle.initialize()

    yield agent

    # Cleanup
    await agent.history.close()
