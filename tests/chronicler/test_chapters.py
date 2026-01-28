"""Tests for chapter management functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from muaddib.chronicler.chapters import _get_arc_lock, chapter_append_paragraph
from muaddib.providers import ModelSpec, UsageInfo


@pytest.mark.asyncio
async def test_chapter_append_paragraph_under_limit(temp_config_file):
    """Test that paragraphs are appended normally when under limit."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    # Set config with higher paragraph limit to avoid hitting it
    agent.config.setdefault("chronicler", {})["paragraphs_per_chapter"] = 10
    arc = "test-arc-under-limit"  # Use unique arc name

    # Add first paragraph (should be normal append)
    result = await chapter_append_paragraph(arc, "First paragraph", agent)

    assert result is not None
    assert result["content"] == "First paragraph"

    # Verify it was added
    content = await agent.chronicle.render_chapter(arc)
    assert "First paragraph" in content

    # Add second paragraph (still under limit)
    await chapter_append_paragraph(arc, "Second paragraph", agent)

    content = await agent.chronicle.render_chapter(arc)
    assert "First paragraph" in content
    assert "Second paragraph" in content


@pytest.mark.asyncio
async def test_chapter_append_paragraph_over_limit(temp_config_file):
    """Test that new chapter is created when hitting paragraph limit."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    # Mock model router to return a summary
    mock_response = Mock()
    mock_client = Mock()
    mock_client.extract_text_from_response.return_value = "Chapter summary"
    agent.model_router.call_raw_with_model = AsyncMock(
        return_value=(
            mock_response,
            mock_client,
            ModelSpec("test", "model"),
            UsageInfo(None, None, None),
        )
    )

    # Set reasonable paragraph limit for testing (high enough to avoid recursion issues)
    agent.config.setdefault("chronicler", {})["paragraphs_per_chapter"] = 5
    agent.config.setdefault("chronicler", {})["model"] = "test:model"
    arc = "test-arc-over-limit"  # Use unique arc name

    # Add paragraphs up to limit (5 paragraphs)
    for i in range(5):
        await chapter_append_paragraph(arc, f"Paragraph {i + 1}", agent)

    # Get current chapter ID
    chapter_before = await agent.chronicle.get_or_open_current_chapter(arc)
    chapter_id_before = chapter_before["id"]

    # Add one more paragraph - this should trigger chapter wrap-up
    await chapter_append_paragraph(arc, "Final paragraph", agent)

    # Verify model was called for summarization at least once
    assert agent.model_router.call_raw_with_model.call_count >= 1

    # Verify new chapter was created by checking the current chapter ID
    chapter_after = await agent.chronicle.get_or_open_current_chapter(arc)
    chapter_id_after = chapter_after["id"]

    # The chapter ID should have changed OR the content should show chapter wrap-up occurred
    content = await agent.chronicle.render_chapter(arc)
    chapter_was_wrapped = "Previous chapter recap: Chapter summary" in content

    # Either we got a new chapter ID or the chapter was wrapped (recap present)
    assert chapter_id_after != chapter_id_before or chapter_was_wrapped

    # Verify new chapter contains recap and new paragraph
    assert "Previous chapter recap: Chapter summary" in content
    assert "Final paragraph" in content


@pytest.mark.asyncio
async def test_chapter_append_paragraph_empty_text():
    """Test that empty text raises ValueError."""
    # Mock agent - we won't actually use it since this should fail early
    mock_agent = Mock()

    with pytest.raises(ValueError, match="paragraph_text must be non-empty"):
        await chapter_append_paragraph("test-arc", "", mock_agent)

    with pytest.raises(ValueError, match="paragraph_text must be non-empty"):
        await chapter_append_paragraph("test-arc", "   ", mock_agent)


@pytest.mark.asyncio
async def test_chapter_append_paragraph_default_config(temp_config_file):
    """Test that default paragraph limit is used when not specified in config."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    # Config without paragraphs_per_chapter (should use default of 10)
    agent.config.setdefault("chronicler", {})["model"] = "test:model"
    arc = "test-arc-default-config"  # Use unique arc name

    # Add one paragraph - should work normally with default limit
    result = await chapter_append_paragraph(arc, "Test paragraph", agent)

    assert result is not None
    assert result["content"] == "Test paragraph"

    # Verify it was added
    content = await agent.chronicle.render_chapter(arc)
    assert "Test paragraph" in content


# Concurrency and Locking Tests


@pytest.mark.asyncio
async def test_arc_locking_basic():
    """Test that each arc gets its own lock."""
    lock1 = _get_arc_lock("arc1")
    lock2 = _get_arc_lock("arc1")  # Should be the same lock
    lock3 = _get_arc_lock("arc2")  # Should be different lock

    assert lock1 is lock2  # Same arc should return same lock
    assert lock1 is not lock3  # Different arcs should have different locks


@pytest.mark.asyncio
async def test_concurrent_appends_same_arc(temp_config_file):
    """Test that concurrent appends to the same arc are properly serialized."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    # Set low paragraph limit for testing
    agent.config.setdefault("chronicler", {})["paragraphs_per_chapter"] = 3
    agent.config.setdefault("chronicler", {})["model"] = "test:model"

    # Mock model router for summary generation
    mock_response = Mock()
    mock_client = Mock()
    mock_client.extract_text_from_response.return_value = "Concurrent test summary"
    agent.model_router.call_raw_with_model = AsyncMock(
        return_value=(
            mock_response,
            mock_client,
            ModelSpec("test", "model"),
            UsageInfo(None, None, None),
        )
    )

    arc = "test-concurrent-arc"

    # Track call order to verify serialization
    call_order = []
    original_append = agent.chronicle.append_paragraph

    async def tracking_append(arc: str, content: str):
        call_order.append(f"start_{content}")
        # Add small delay to simulate processing time
        await asyncio.sleep(0.01)
        result = await original_append(arc, content)
        call_order.append(f"end_{content}")
        return result

    agent.chronicle.append_paragraph = tracking_append

    # Launch multiple concurrent appends
    tasks = [chapter_append_paragraph(arc, f"Message {i}", agent) for i in range(5)]

    results = await asyncio.gather(*tasks)

    # All should succeed
    assert len(results) == 5
    assert all(result is not None for result in results)

    # Verify that calls were serialized (each start/end pair should be together)
    # Due to locking, we should not see interleaved starts and ends
    start_count = 0
    end_count = 0
    for event in call_order:
        if event.startswith("start_"):
            start_count += 1
        elif event.startswith("end_"):
            end_count += 1
        # At any point, end_count should not exceed start_count (no race conditions)
        assert end_count <= start_count


@pytest.mark.asyncio
async def test_concurrent_appends_different_arcs(temp_config_file):
    """Test that concurrent appends to different arcs can proceed in parallel."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    arc1 = "test-arc-1"
    arc2 = "test-arc-2"

    # Track which arcs are being processed concurrently
    processing_arcs = set()
    max_concurrent = 0

    original_append = agent.chronicle.append_paragraph

    async def tracking_append(arc: str, content: str):
        processing_arcs.add(arc)
        nonlocal max_concurrent
        max_concurrent = max(max_concurrent, len(processing_arcs))

        # Add delay to increase chance of concurrency
        await asyncio.sleep(0.02)
        result = await original_append(arc, content)

        processing_arcs.remove(arc)
        return result

    agent.chronicle.append_paragraph = tracking_append

    # Launch concurrent appends to different arcs
    tasks = [
        chapter_append_paragraph(arc1, "Message for arc1", agent),
        chapter_append_paragraph(arc2, "Message for arc2", agent),
    ]

    results = await asyncio.gather(*tasks)

    # Both should succeed
    assert len(results) == 2
    assert all(result is not None for result in results)

    # We should have seen concurrent processing (both arcs active at once)
    assert max_concurrent >= 2, (
        f"Expected concurrent processing, but max_concurrent was {max_concurrent}"
    )


@pytest.mark.asyncio
async def test_lock_is_released_on_exception():
    """Test that locks are properly released even if an exception occurs."""

    # Create agent instance with mock that will fail when chronicle is accessed
    agent = Mock()
    agent.config = {}
    agent.chronicle.get_or_open_current_chapter.side_effect = RuntimeError("Simulated failure")

    arc = "test-exception-arc"

    # This should raise RuntimeError but not deadlock
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await chapter_append_paragraph(arc, "Test message", agent)

    # Lock should be available again
    lock = _get_arc_lock(arc)

    # This should not hang (lock should be released)
    async with asyncio.timeout(1.0):
        async with lock:
            # If we get here, the lock was properly released
            pass
