"""Shared fixtures for agentic actor tests."""

from unittest.mock import AsyncMock

import pytest

from muaddib.agentic_actor.tools import ArtifactStore, EditArtifactExecutor


@pytest.fixture
def artifacts_url():
    """Artifact base URL for tests."""
    return "https://example.com/artifacts"


@pytest.fixture
def artifact_store(tmp_path, artifacts_url):
    """Configured artifact store with temp directory."""
    return ArtifactStore(artifacts_path=str(tmp_path / "artifacts"), artifacts_url=artifacts_url)


@pytest.fixture
def webpage_visitor(artifact_store):
    """WebpageVisitorExecutor with artifact store configured."""
    from muaddib.agentic_actor.tools import WebpageVisitorExecutor

    return WebpageVisitorExecutor(artifact_store=artifact_store)


@pytest.fixture
def make_edit_executor(artifact_store):
    """Factory for creating EditArtifactExecutor with mocked visitor."""

    def _make(*, visitor_result=None, visitor_exc=None):
        visitor = AsyncMock()
        if visitor_exc:
            visitor.execute.side_effect = visitor_exc
        else:
            visitor.execute.return_value = visitor_result
        return EditArtifactExecutor(store=artifact_store, webpage_visitor=visitor), visitor

    return _make


def image_content_block(media_type="image/png"):
    """Helper to create Anthropic image content block."""
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": "..."},
        }
    ]
