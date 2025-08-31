"""Tests for tool functionality."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from irssi_llmagent.tools import (
    PythonExecutorE2B,
    ShareArtifactExecutor,
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

        # Create mock for HEAD response
        mock_head_response = AsyncMock()
        mock_head_response.headers = {"content-type": "image/png"}
        mock_head_response.raise_for_status = MagicMock()

        # Create mock for GET response
        mock_get_response = AsyncMock()
        mock_get_response.headers = {"content-type": "image/png"}
        mock_get_response.read = AsyncMock(
            return_value=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        )  # PNG header
        mock_get_response.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Create async context managers for session.head and session.get
        mock_head_context = AsyncMock()
        mock_head_context.__aenter__ = AsyncMock(return_value=mock_head_response)
        mock_head_context.__aexit__ = AsyncMock(return_value=None)

        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.head = MagicMock(return_value=mock_head_context)
        mock_session.get = MagicMock(return_value=mock_get_context)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await executor.execute("https://example.com/image.png")

            assert result.startswith("IMAGE_DATA:image/png:")
            # Check for base64-encoded PNG header
            import base64

            expected_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
            assert expected_b64 in result

    @pytest.mark.asyncio
    async def test_webpage_visitor_http_451_backoff(self):
        """Test webpage visitor HTTP 451 backoff retry logic."""
        progress_calls = []

        async def mock_progress_callback(text: str):
            progress_calls.append(text)

        executor = WebpageVisitorExecutor(progress_callback=mock_progress_callback)

        # Mock head response (non-image)
        mock_head_response = AsyncMock()
        mock_head_response.headers = {"content-type": "text/html"}
        mock_head_response.raise_for_status = MagicMock()

        # Mock GET responses: first 451, then 451, then success
        mock_get_response_451_1 = AsyncMock()
        mock_get_response_451_1.status = 451
        mock_get_response_451_1.request_info = MagicMock()
        mock_get_response_451_1.history = []
        mock_get_response_451_1.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=mock_get_response_451_1.request_info,
                history=mock_get_response_451_1.history,
                status=451,
                message="Unavailable For Legal Reasons",
            )
        )

        mock_get_response_451_2 = AsyncMock()
        mock_get_response_451_2.status = 451
        mock_get_response_451_2.request_info = MagicMock()
        mock_get_response_451_2.history = []
        mock_get_response_451_2.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=mock_get_response_451_2.request_info,
                history=mock_get_response_451_2.history,
                status=451,
                message="Unavailable For Legal Reasons",
            )
        )

        mock_get_response_success = AsyncMock()
        mock_get_response_success.status = 200
        mock_get_response_success.text = AsyncMock(
            return_value="# Success\n\nContent loaded successfully"
        )
        mock_get_response_success.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Setup context managers
        mock_head_context = AsyncMock()
        mock_head_context.__aenter__ = AsyncMock(return_value=mock_head_response)
        mock_head_context.__aexit__ = AsyncMock(return_value=None)

        mock_get_context_451_1 = AsyncMock()
        mock_get_context_451_1.__aenter__ = AsyncMock(return_value=mock_get_response_451_1)
        mock_get_context_451_1.__aexit__ = AsyncMock(return_value=None)

        mock_get_context_451_2 = AsyncMock()
        mock_get_context_451_2.__aenter__ = AsyncMock(return_value=mock_get_response_451_2)
        mock_get_context_451_2.__aexit__ = AsyncMock(return_value=None)

        mock_get_context_success = AsyncMock()
        mock_get_context_success.__aenter__ = AsyncMock(return_value=mock_get_response_success)
        mock_get_context_success.__aexit__ = AsyncMock(return_value=None)

        mock_session.head = MagicMock(return_value=mock_head_context)
        mock_session.get = MagicMock(
            side_effect=[mock_get_context_451_1, mock_get_context_451_2, mock_get_context_success]
        )

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep") as mock_sleep:
                result = await executor.execute("https://example.com/test")

                # Verify result contains expected content
                assert "Content loaded successfully" in result

                # Verify progress callbacks were made
                assert len(progress_calls) == 2
                assert "r.jina.ai HTTP 451" in progress_calls[0]
                assert "r.jina.ai HTTP 451" in progress_calls[1]

                # Verify asyncio.sleep was called with correct delays
                mock_sleep.assert_has_calls([call(30), call(90)])

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

    @pytest.mark.asyncio
    async def test_share_artifact_executor_success(self):
        """Test artifact sharing with valid configuration."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"

            executor = ShareArtifactExecutor(
                artifacts_path=artifacts_path, artifacts_url=artifacts_url
            )

            content = "#!/bin/bash\necho 'Hello, World!'"

            result = await executor.execute(content)

            # Verify return format
            assert result.startswith("Artifact shared: https://example.com/artifacts/")
            assert result.endswith(".txt")

            # Extract filename from result
            url = result.split(": ")[1]
            filename = url.split("/")[-1]

            # Verify file was created
            artifacts_dir = Path(artifacts_path)
            artifact_file = artifacts_dir / filename
            assert artifact_file.exists()

            # Verify content
            file_content = artifact_file.read_text()
            assert file_content == content

            # Verify UUID format (32 hex chars)
            uuid_part = filename.replace(".txt", "")
            assert len(uuid_part) == 32
            assert all(c in "0123456789abcdef" for c in uuid_part)

    @pytest.mark.asyncio
    async def test_share_artifact_executor_missing_config(self):
        """Test artifact sharing with missing configuration."""
        executor = ShareArtifactExecutor()

        result = await executor.execute("test content")

        assert (
            result
            == "Error: artifacts.path and artifacts.url must be configured to share artifacts"
        )

    @pytest.mark.asyncio
    async def test_share_artifact_executor_write_error(self):
        """Test artifact sharing with write error."""
        executor = ShareArtifactExecutor(
            artifacts_path="/nonexistent/readonly/path",
            artifacts_url="https://example.com/artifacts",
        )

        result = await executor.execute("test content")

        assert result.startswith("Error: Failed to create artifacts directory:")


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
    async def test_execute_make_plan_tool(self):
        """Test executing make_plan tool."""
        result = await execute_tool("make_plan", plan="Test plan for searching news")
        assert result.startswith("OK")

    @pytest.mark.asyncio
    async def test_execute_share_artifact_tool(self):
        """Test executing share_artifact tool."""
        with patch.object(ShareArtifactExecutor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Artifact shared: https://example.com/artifacts/123.txt"

            result = await execute_tool("share_artifact", content="test content")

            assert result == "Artifact shared: https://example.com/artifacts/123.txt"
            mock_execute.assert_called_once_with(content="test content")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool("unknown_tool", param="value")


class TestToolDefinitions:
    """Test tool definitions."""

    def test_create_tool_executors_with_config(self):
        """Test that tool executors are created with configuration."""
        config = {"tools": {"e2b": {"api_key": "test-key-123"}}}

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

    def test_make_plan_tool_in_tools_list(self):
        """Test that make_plan tool is included in TOOLS list."""
        from irssi_llmagent.tools import TOOLS

        tool_names = [tool["name"] for tool in TOOLS]
        assert "make_plan" in tool_names

        # Find the make_plan tool and verify its structure
        make_plan_tool = next(tool for tool in TOOLS if tool["name"] == "make_plan")
        assert make_plan_tool["description"]
        assert "input_schema" in make_plan_tool
        assert "properties" in make_plan_tool["input_schema"]
        assert "plan" in make_plan_tool["input_schema"]["properties"]
        assert "required" in make_plan_tool["input_schema"]
        assert "plan" in make_plan_tool["input_schema"]["required"]

    def test_share_artifact_tool_in_tools_list(self):
        """Test that share_artifact tool is included in TOOLS list."""
        from irssi_llmagent.tools import TOOLS

        tool_names = [tool["name"] for tool in TOOLS]
        assert "share_artifact" in tool_names

        # Find the share_artifact tool and verify its structure
        share_artifact_tool = next(tool for tool in TOOLS if tool["name"] == "share_artifact")
        assert share_artifact_tool["description"]
        assert "input_schema" in share_artifact_tool
        assert "properties" in share_artifact_tool["input_schema"]
        assert "content" in share_artifact_tool["input_schema"]["properties"]
        assert "required" in share_artifact_tool["input_schema"]
        assert "content" in share_artifact_tool["input_schema"]["required"]
        # Verify no description or filename parameters
        assert "description" not in share_artifact_tool["input_schema"]["properties"]
        assert "filename" not in share_artifact_tool["input_schema"]["properties"]
