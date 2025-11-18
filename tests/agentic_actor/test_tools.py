"""Tests for tool functionality."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from irssi_llmagent.agentic_actor.tools import (
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

            # Result should be Anthropic content blocks with image
            assert isinstance(result, list)
            assert len(result) == 1
            first_block = result[0]
            assert isinstance(first_block, dict)
            assert first_block["type"] == "image"
            source = first_block["source"]
            assert isinstance(source, dict)
            assert source["type"] == "base64"
            assert source["media_type"] == "image/png"
            # Check for base64-encoded PNG header
            import base64

            expected_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
            assert expected_b64 in source["data"]

    @pytest.mark.asyncio
    async def test_webpage_visitor_http_451_backoff(self):
        """Test webpage visitor HTTP 451 backoff retry logic."""
        progress_calls = []

        async def mock_progress_callback(text: str, type: str = "progress"):
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

                # Verify progress callbacks were made (only on second attempt per spam reduction change)
                assert len(progress_calls) == 1
                assert "r.jina.ai HTTP 451" in progress_calls[0]

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
    async def test_python_executor_e2b_persistence_across_calls(self):
        """Test that sandbox persists state across multiple execute() calls."""
        executor = PythonExecutorE2B()

        mock_execution = MagicMock()
        mock_execution.text = None
        mock_execution.logs = MagicMock(stdout=["result\n"], stderr=[])
        mock_execution.results = None

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-123"

        sandbox_created = False

        async def mock_to_thread_impl(func, *args):
            nonlocal sandbox_created
            # First call is creating the sandbox
            if not sandbox_created:
                sandbox_created = True
                return mock_sandbox
            # Subsequent calls are run_code
            return mock_execution

        with patch("e2b_code_interpreter.Sandbox"):
            with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                # First execute should create sandbox
                result1 = await executor.execute("x = 1")
                # Second execute should reuse sandbox
                result2 = await executor.execute("print(x)")

                # Verify sandbox was created and reused
                assert executor.sandbox is mock_sandbox
                assert "result" in result1
                assert "result" in result2

    @pytest.mark.asyncio
    async def test_python_executor_e2b_cleanup(self):
        """Test that cleanup properly kills the sandbox."""
        executor = PythonExecutorE2B()

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-456"
        mock_sandbox.kill = MagicMock()

        executor.sandbox = mock_sandbox

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            await executor.cleanup()

            # Verify kill was called via to_thread
            mock_to_thread.assert_called_once()
            assert executor.sandbox is None

    @pytest.mark.asyncio
    async def test_python_executor_e2b_error_resets_sandbox(self):
        """Test that connection errors reset the sandbox."""
        executor = PythonExecutorE2B()

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-789"

        executor.sandbox = mock_sandbox

        async def mock_to_thread_fail(*args):
            raise Exception("sandbox connection lost")

        with patch("asyncio.to_thread", side_effect=mock_to_thread_fail):
            result = await executor.execute("print('test')")

            assert "Error executing code" in result
            assert executor.sandbox is None  # Should be reset

    @pytest.mark.asyncio
    async def test_python_executor_e2b_timeout_parameter(self):
        """Test that timeout is passed to Sandbox constructor."""
        executor = PythonExecutorE2B(timeout=180)

        assert executor.timeout == 180

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-timeout"

        sandbox_args_captured = None

        def capture_sandbox(**kwargs):
            nonlocal sandbox_args_captured
            sandbox_args_captured = kwargs
            return mock_sandbox

        # Mock to_thread to call our function directly
        async def mock_to_thread_impl(func, *args):
            return func()

        with patch("e2b_code_interpreter.Sandbox", side_effect=capture_sandbox):
            with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                await executor._ensure_sandbox()

                # Verify Sandbox was called with timeout=180
                assert sandbox_args_captured == {"timeout": 180}

    @pytest.mark.asyncio
    async def test_share_artifact_executor_success(self):
        """Test artifact sharing with valid configuration."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"

            from irssi_llmagent.agentic_actor.tools import ArtifactStore

            store = ArtifactStore(artifacts_path=artifacts_path, artifacts_url=artifacts_url)
            executor = ShareArtifactExecutor(store=store)

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
        from irssi_llmagent.agentic_actor.tools import ArtifactStore

        store = ArtifactStore(artifacts_path=None, artifacts_url=None)
        executor = ShareArtifactExecutor(store=store)

        result = await executor.execute("test content")

        assert result == "Error: artifacts.path and artifacts.url must be configured"

    @pytest.mark.asyncio
    async def test_share_artifact_executor_write_error(self):
        """Test artifact sharing with write error."""
        from irssi_llmagent.agentic_actor.tools import ArtifactStore

        store = ArtifactStore(
            artifacts_path="/nonexistent/readonly/path",
            artifacts_url="https://example.com/artifacts",
        )
        executor = ShareArtifactExecutor(store=store)

        result = await executor.execute("test content")

        assert result.startswith("Error: Failed to create artifacts directory:")


class TestToolRegistry:
    """Test tool registry and execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_web_search(self, mock_agent):
        """Test executing web search tool."""
        with patch.object(WebSearchExecutor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Search results"

            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            result = await execute_tool("web_search", tool_executors, query="test")

            assert result == "Search results"
            mock_execute.assert_called_once_with(query="test")

    @pytest.mark.asyncio
    async def test_execute_tool_visit_webpage(self, mock_agent):
        """Test executing webpage visit tool."""
        with patch.object(
            WebpageVisitorExecutor, "execute", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = "Webpage content"

            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            result = await execute_tool("visit_webpage", tool_executors, url="https://example.com")

            assert result == "Webpage content"
            mock_execute.assert_called_once_with(url="https://example.com")

    @pytest.mark.asyncio
    async def test_execute_tool_python_executor(self, mock_agent):
        """Test executing Python code tool."""
        with patch.object(PythonExecutorE2B, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Code output"

            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            result = await execute_tool("execute_python", tool_executors, code="print('test')")

            assert result == "Code output"
            mock_execute.assert_called_once_with(code="print('test')")

    @pytest.mark.asyncio
    async def test_execute_make_plan_tool(self, mock_agent):
        """Test executing make_plan tool."""
        tool_executors = create_tool_executors(agent=mock_agent, arc="test")
        result = await execute_tool(
            "make_plan", tool_executors, plan="Test plan for searching news"
        )
        assert isinstance(result, str)
        assert result.startswith("OK")

    @pytest.mark.asyncio
    async def test_execute_share_artifact_tool(self, mock_agent):
        """Test executing share_artifact tool."""
        with patch.object(ShareArtifactExecutor, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Artifact shared: https://example.com/artifacts/123.txt"

            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            result = await execute_tool("share_artifact", tool_executors, content="test content")

            assert result == "Artifact shared: https://example.com/artifacts/123.txt"
            mock_execute.assert_called_once_with(content="test content")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_agent):
        """Test executing unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            await execute_tool("unknown_tool", tool_executors, param="value")


class TestToolDefinitions:
    """Test tool definitions."""

    def test_create_tool_executors_with_config(self, mock_agent):
        """Test that tool executors are created with configuration."""
        config = {"tools": {"e2b": {"api_key": "test-key-123"}}}

        executors = create_tool_executors(config, agent=mock_agent, arc="test")

        assert "execute_python" in executors
        python_executor = executors["execute_python"]
        assert isinstance(python_executor, PythonExecutorE2B)
        assert python_executor.api_key == "test-key-123"

    def test_create_tool_executors_without_config(self, mock_agent):
        """Test that tool executors are created without configuration."""
        executors = create_tool_executors(agent=mock_agent, arc="test")

        assert "execute_python" in executors
        python_executor = executors["execute_python"]
        assert isinstance(python_executor, PythonExecutorE2B)
        assert python_executor.api_key is None

    def test_make_plan_tool_in_tools_list(self):
        """Test that make_plan tool is included in TOOLS list."""
        from irssi_llmagent.agentic_actor.tools import TOOLS

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
        from irssi_llmagent.agentic_actor.tools import TOOLS

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

    def test_tools_have_persist_field(self):
        """Test that all tools have the required persist field."""
        from irssi_llmagent.agentic_actor.tools import TOOLS
        from irssi_llmagent.chronicler.tools import chronicle_tools_defs

        # Test main tools
        for tool in TOOLS:
            assert "persist" in tool, f"Tool '{tool['name']}' missing persist field"
            assert tool["persist"] in [
                "none",
                "exact",
                "summary",
                "artifact",
            ], f"Tool '{tool['name']}' has invalid persist value: {tool['persist']}"

        # Test chronicle tools
        for tool in chronicle_tools_defs():
            assert "persist" in tool, f"Chronicle tool '{tool['name']}' missing persist field"
            assert tool["persist"] in [
                "none",
                "exact",
                "summary",
                "artifact",
            ], f"Chronicle tool '{tool['name']}' has invalid persist value: {tool['persist']}"

    def test_anthropic_filters_custom_tool_fields(self):
        """Test that Anthropic provider filters out custom fields like 'persist'."""
        from irssi_llmagent.providers.anthropic import AnthropicClient

        # Create a mock client (just for testing the filter method)
        config = {"providers": {"anthropic": {"url": "test", "key": "test"}}}
        client = AnthropicClient(config)

        # Test tools with custom fields
        tools_with_custom_fields = [
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "persist": "summary",  # Custom field that should be filtered out
                "custom_field": "should_be_removed",  # Another custom field
            },
            {
                "name": "execute_python",
                "description": "Execute Python code",
                "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
                "persist": "artifact",  # Custom field that should be filtered out
            },
        ]

        # Filter the tools
        filtered_tools = client._filter_tools(tools_with_custom_fields)

        # Verify filtering worked correctly
        assert len(filtered_tools) == 2

        for tool in filtered_tools:
            # Should have only standard Anthropic fields
            expected_fields = {"name", "description", "input_schema"}
            assert set(tool.keys()) == expected_fields

            # Should NOT have custom fields
            assert "persist" not in tool
            assert "custom_field" not in tool

        # Verify content is preserved
        assert filtered_tools[0]["name"] == "web_search"
        assert filtered_tools[0]["description"] == "Search the web"
        assert filtered_tools[1]["name"] == "execute_python"
