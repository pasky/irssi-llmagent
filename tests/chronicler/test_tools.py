import pytest

from irssi_llmagent.chronicler.tools import ChapterAppendExecutor, ChapterRenderExecutor


@pytest.mark.asyncio
async def test_chapter_append_executor(temp_config_file):
    """Test ChapterAppendExecutor directly."""
    from irssi_llmagent.main import IRSSILLMAgent

    # Create agent instance
    agent = IRSSILLMAgent(temp_config_file)
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
    """Test ChapterRenderExecutor directly."""
    from irssi_llmagent.main import IRSSILLMAgent

    # Create agent instance
    agent = IRSSILLMAgent(temp_config_file)
    await agent.chronicle.initialize()

    arc = "test-arc"

    # Add some content first
    await agent.chronicle.append_paragraph(arc, "First paragraph")
    await agent.chronicle.append_paragraph(arc, "Second paragraph")

    executor = ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc)

    # Test successful render
    result = await executor.execute()
    assert "First paragraph" in result
    assert "Second paragraph" in result
