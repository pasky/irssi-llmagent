import pytest

from muaddib.chronicler.tools import ChapterAppendExecutor, ChapterRenderExecutor


@pytest.mark.asyncio
async def test_chapter_append_executor(temp_config_file):
    """Test ChapterAppendExecutor directly."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    arc = "test-arc"
    executor = ChapterAppendExecutor(agent=agent, arc=arc)

    # Test successful append
    result = await executor.execute(text="Test paragraph")
    assert result == "OK"

    # Verify content was appended
    content = await agent.chronicle.render_chapter(arc)
    assert "Test paragraph" in content


@pytest.mark.asyncio
async def test_chapter_render_executor(temp_config_file):
    """Test ChapterRenderExecutor with relative chapter IDs."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    arc = "test-arc"

    # Add some content first
    await agent.chronicle.append_paragraph(arc, "First paragraph")
    await agent.chronicle.append_paragraph(arc, "Second paragraph")

    executor = ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc)

    # Test reading current chapter (relative_chapter_id = 0)
    result = await executor.execute(relative_chapter_id=0)
    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "current" in result

    # Test reading non-existent previous chapter (relative_chapter_id = -1)
    result_prev = await executor.execute(relative_chapter_id=-1)
    assert "No chapters at relative offset -1" in result_prev


@pytest.mark.asyncio
async def test_chapter_render_executor_multiple_chapters(temp_config_file):
    """Test ChapterRenderExecutor with multiple chapters for relative navigation."""
    from muaddib.main import MuaddibAgent

    # Create agent instance
    agent = MuaddibAgent(temp_config_file)
    await agent.chronicle.initialize()

    arc = "test-arc"

    # Add content to first chapter
    await agent.chronicle.append_paragraph(arc, "Chapter 1 paragraph 1")
    await agent.chronicle.append_paragraph(arc, "Chapter 1 paragraph 2")

    # Close current chapter and start a new one
    from muaddib.chronicler.chapters import _close_chapter

    current_ch = await agent.chronicle.get_or_open_current_chapter(arc)
    await _close_chapter(agent.chronicle, current_ch["id"], "First chapter summary")

    # Add content to second chapter
    await agent.chronicle.append_paragraph(arc, "Chapter 2 paragraph 1")
    await agent.chronicle.append_paragraph(arc, "Chapter 2 paragraph 2")

    executor = ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc)

    # Test reading current chapter (should be chapter 2)
    result_current = await executor.execute(relative_chapter_id=0)
    assert "Chapter 2 paragraph 1" in result_current
    assert "Chapter 2 paragraph 2" in result_current
    assert "current" in result_current

    # Test reading previous chapter (should be chapter 1)
    result_prev = await executor.execute(relative_chapter_id=-1)
    assert "Chapter 1 paragraph 1" in result_prev
    assert "Chapter 1 paragraph 2" in result_prev
    assert "1 chapter back" in result_prev

    # Test reading non-existent chapter (2 chapters back)
    result_nonexistent = await executor.execute(relative_chapter_id=-2)
    assert "No chapters at relative offset -2" in result_nonexistent
