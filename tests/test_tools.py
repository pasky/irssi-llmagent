"""Tests for tool functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from irssi_llmagent.tools import TOOLS, WebpageVisitorExecutor, WebSearchExecutor, execute_tool


class TestToolExecutors:
    """Test tool executor functionality."""

    @pytest.mark.asyncio
    async def test_web_search_executor(self):
        """Test web search executor."""
        executor = WebSearchExecutor(max_results=3)

        mock_results = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Description 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Description 2"},
        ]

        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
            mock_ddgs.text.return_value = mock_results

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock()
                mock_executor.return_value = mock_results
                mock_loop.return_value.run_in_executor = mock_executor

                result = await executor.execute("test query")

                assert "## Search Results" in result
                assert "Result 1" in result
                assert "https://example.com/1" in result
                assert "Description 1" in result

    @pytest.mark.asyncio
    async def test_web_search_no_results(self):
        """Test web search with no results."""
        executor = WebSearchExecutor()

        with patch("ddgs.DDGS") as mock_ddgs_class:
            mock_ddgs = MagicMock()
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs
            mock_ddgs.text.return_value = []

            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock()
                mock_executor.return_value = []
                mock_loop.return_value.run_in_executor = mock_executor

                result = await executor.execute("no results query")

                assert "No search results found" in result

    @pytest.mark.asyncio
    async def test_webpage_visitor_invalid_url(self):
        """Test webpage visitor with invalid URL."""
        executor = WebpageVisitorExecutor()

        with pytest.raises(ValueError, match="Invalid URL"):
            await executor.execute("not-a-url")


class TestToolRegistry:
    """Test tool registry and execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_web_search(self):
        """Test executing web search tool."""
        with patch.object(WebSearchExecutor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Search results"

            result = await execute_tool("web_search", query="test")

            assert result == "Search results"
            mock_execute.assert_called_once_with(query="test")

    @pytest.mark.asyncio
    async def test_execute_tool_visit_webpage(self):
        """Test executing webpage visit tool."""
        with patch.object(
            WebpageVisitorExecutor, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = "Webpage content"

            result = await execute_tool("visit_webpage", url="https://example.com")

            assert result == "Webpage content"
            mock_execute.assert_called_once_with(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool("unknown_tool", param="value")


class TestToolDefinitions:
    """Test tool definitions."""

    def test_tools_structure(self):
        """Test that tools have correct structure."""
        assert isinstance(TOOLS, list)
        assert len(TOOLS) == 2

        # Check web_search tool
        web_search = next(tool for tool in TOOLS if tool["name"] == "web_search")
        assert "description" in web_search
        assert "input_schema" in web_search
        assert web_search["input_schema"]["required"] == ["query"]

        # Check visit_webpage tool
        visit_webpage = next(tool for tool in TOOLS if tool["name"] == "visit_webpage")
        assert "description" in visit_webpage
        assert "input_schema" in visit_webpage
        assert visit_webpage["input_schema"]["required"] == ["url"]
