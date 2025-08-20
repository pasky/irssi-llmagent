"""Pytest configuration and fixtures."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(params=["anthropic", "openai"])
def api_type(request):
    """Parametrized API type fixture."""
    return request.param


@pytest.fixture
def test_config(api_type) -> dict[str, Any]:
    """Test configuration fixture with parametrized API type."""
    base_config = {
        "providers": {
            "anthropic": {"url": "https://api.anthropic.com/v1/messages", "key": "test-key"},
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "key": "test-key",
                "max_tokens": 2048,
            },
            "perplexity": {
                "url": "https://api.perplexity.ai/chat/completions",
                "key": "test-key",
                "model": "sonar-pro",
            },
        },
        "varlink": {"socket_path": "/tmp/test_varlink.sock"},
        "command": {
            "history_size": 5,
            "rate_limit": 30,
            "rate_period": 900,
            "ignore_users": [],
            "models": {
                "sarcastic": f"{api_type}:dummy-sarcastic",
                "serious": [f"{api_type}:dummy-serious"],
                "classifier": f"{api_type}:dummy-classifier",
            },
            "prompts": {
                "serious": "You are IRC user {mynick}. You are friendly, straight, informal, maybe ironic, but always informative. Test serious prompt.",
                "sarcastic": "You are IRC user {mynick} and you are known for your sharp sarcasm and cynical, dry, rough sense of humor. Test sarcastic prompt.",
                "mode_classifier": "Analyze this IRC message and decide whether it should be handled with SARCASTIC or SERIOUS mode. Respond with only one word: 'SARCASTIC' or 'SERIOUS'. Message: {message}",
            },
        },
        "proactive": {
            "history_size": 3,
            "interjecting": [],
            "interjecting_test": [],
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
        "agent": {"progress": {"threshold_seconds": 10, "min_interval_seconds": 8}},
    }
    return base_config


@pytest.fixture
def temp_db_path():
    """Temporary database path fixture."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
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
