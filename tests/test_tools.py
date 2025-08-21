"""Tests for tool functionality."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from irssi_llmagent.claude import AnthropicClient
from irssi_llmagent.openai import OpenAIClient
from irssi_llmagent.tools import (
    PythonExecutorE2B,
    WebpageVisitorExecutor,
    WebSearchExecutor,
    create_tool_executors,
    execute_tool,
)


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

    @pytest.mark.asyncio
    async def test_webpage_visitor_image_content(self):
        """Test webpage visitor with image URL."""
        executor = WebpageVisitorExecutor()

        # Create proper async context manager mock
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "image/png"}
        mock_response.read = AsyncMock(
            return_value=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        )  # PNG header
        mock_response.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Create async context manager for session.get
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_context)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await executor.execute("https://example.com/image.png")

            assert result.startswith("IMAGE_DATA:image/png:")
            # Check for base64-encoded PNG header
            import base64

            expected_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
            assert expected_b64 in result

    @pytest.mark.asyncio
    async def test_python_executor_e2b_success(self):
        """Test Python executor with successful execution."""
        executor = PythonExecutorE2B()

        mock_execution = MagicMock()
        mock_execution.text = None
        mock_logs = MagicMock()
        mock_logs.stdout = ["Hello, World!\n"]
        mock_logs.stderr = []
        mock_execution.logs = mock_logs
        mock_execution.results = None

        mock_sandbox = MagicMock()
        mock_sandbox.run_code.return_value = mock_execution
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)

        with patch("e2b_code_interpreter.Sandbox", return_value=mock_sandbox):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = AsyncMock()
                mock_executor.return_value = mock_execution
                mock_loop.return_value.run_in_executor = mock_executor

                result = await executor.execute("print('Hello, World!')")

                assert "**Output:**" in result
                assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_python_executor_e2b_import_error(self):
        """Test Python executor when e2b package is not available."""
        executor = PythonExecutorE2B()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'e2b_code_interpreter'")
        ):
            result = await executor.execute("print('test')")

            assert "e2b-code-interpreter package not installed" in result


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
    async def test_execute_tool_python_executor(self):
        """Test executing Python code tool."""
        with patch.object(PythonExecutorE2B, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Code output"

            result = await execute_tool("execute_python", code="print('test')")

            assert result == "Code output"
            mock_execute.assert_called_once_with(code="print('test')")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool("unknown_tool", param="value")


class TestToolDefinitions:
    """Test tool definitions."""

    def test_create_tool_executors_with_config(self):
        """Test that tool executors are created with configuration."""
        config = {"providers": {"e2b": {"api_key": "test-key-123"}}}

        executors = create_tool_executors(config)

        assert "execute_python" in executors
        python_executor = executors["execute_python"]
        assert isinstance(python_executor, PythonExecutorE2B)
        assert python_executor.api_key == "test-key-123"

    def test_create_tool_executors_without_config(self):
        """Test that tool executors are created without configuration."""
        executors = create_tool_executors()

        assert "execute_python" in executors
        python_executor = executors["execute_python"]
        assert isinstance(python_executor, PythonExecutorE2B)
        assert python_executor.api_key is None


class TestImageHandling:
    """Test image handling in tool results for different providers."""

    def test_claude_image_formatting(self):
        """Test Claude client formats image tool results correctly."""
        client = AnthropicClient(
            {"anthropic": {"key": "test-key", "model": "claude-3-sonnet-20240229"}}
        )

        # Mock image data (small PNG)
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        image_b64 = base64.b64encode(png_data).decode()

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "test-123",
                "content": f"IMAGE_DATA:image/png:{len(png_data)}:{image_b64}",
            }
        ]

        result = client.format_tool_results(tool_results)

        assert result["role"] == "user"
        content = result["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "test-123"
        assert content["content"][0]["type"] == "image"
        assert content["content"][0]["source"]["type"] == "base64"
        assert content["content"][0]["source"]["media_type"] == "image/png"
        assert content["content"][0]["source"]["data"] == image_b64

    def test_openai_image_formatting(self):
        """Test OpenAI client formats image tool results correctly."""
        client = OpenAIClient({"openai": {"key": "test-key", "model": "gpt-4-vision-preview"}})

        # Mock image data (small PNG)
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        image_b64 = base64.b64encode(png_data).decode()

        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": "test-123",
                "content": f"IMAGE_DATA:image/png:{len(png_data)}:{image_b64}",
            }
        ]

        result = client.format_tool_results(tool_results)

        assert isinstance(result, list)
        assert len(result) == 2  # Tool result + image message

        # Check tool result
        tool_result = result[0]
        assert tool_result["type"] == "function_call_output"
        assert tool_result["call_id"] == "test-123"
        import json

        output = json.loads(tool_result["output"])
        assert "Downloaded image (image/png" in output["result"]

        # Check image message
        image_msg = result[1]
        assert image_msg["role"] == "user"
        assert len(image_msg["content"]) == 2  # Text + image
        assert image_msg["content"][0]["type"] == "input_text"
        assert image_msg["content"][1]["type"] == "input_image"
        assert f"data:image/png;base64,{image_b64}" in image_msg["content"][1]["image_url"]
