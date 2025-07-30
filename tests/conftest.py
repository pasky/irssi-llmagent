"""Pytest configuration and fixtures."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration fixture."""
    return {
        "anthropic": {
            "url": "https://api.anthropic.com/v1/messages",
            "key": "test-key",
            "model": "claude-3-haiku-20240307",
            "serious_model": "claude-3-sonnet-20240229",
            "classifier_model": "claude-3-5-haiku-20241022",
            "proactive_validation_models": ["claude-3-haiku-20240307"],
        },
        "perplexity": {
            "url": "https://api.perplexity.ai/chat/completions",
            "key": "test-key",
            "model": "sonar-pro",
        },
        "varlink": {"socket_path": "/tmp/test_varlink.sock"},
        "behavior": {
            "history_size": 5,
            "rate_limit": 30,
            "rate_period": 900,
            "ignore_users": [],
            "proactive_interject_threshold": 9,
        },
        "prompts": {
            "serious": "You are IRC user {mynick}. You are friendly, straight, informal, maybe ironic, but always informative. Test serious prompt.",
            "sarcastic": "You are IRC user {mynick} and you are known for your sharp sarcasm and cynical, dry, rough sense of humor. Test sarcastic prompt.",
            "mode_classifier": "Analyze this IRC message and decide whether it should be handled with SARCASTIC or SERIOUS mode. Respond with only one word: 'SARCASTIC' or 'SERIOUS'. Message: {message}",
            "proactive_interject": "Decide if AI should interject. Respond with '[reason]: X/10' where X is 1-10. Message: {message}",
            "proactive_serious_extra": "NOTE: This is a proactive interjection. If upon reflection you decide your contribution wouldn't add significant factual value (e.g. just an encouragement or general statement), respond with exactly 'NULL' instead of a message.",
        },
    }


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
