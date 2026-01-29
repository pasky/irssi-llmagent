"""Tests for tool functionality."""

from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from muaddib.agentic_actor.actor import AgentResult
from muaddib.agentic_actor.tools import (
    CodeExecutorE2B,
    EditArtifactExecutor,
    JinaSearchExecutor,
    OracleExecutor,
    ShareArtifactExecutor,
    WebpageVisitorExecutor,
    WebSearchExecutor,
    create_tool_executors,
    execute_tool,
    resolve_http_headers,
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
    async def test_jina_executor_extra_kwargs(self):
        """Test Jina executor with extra kwargs."""
        executor = JinaSearchExecutor()

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.text.return_value = "Search results"
            mock_response.raise_for_status = MagicMock()

            mock_get_ctx = AsyncMock()
            mock_get_ctx.__aenter__.return_value = mock_response
            mock_get_ctx.__aexit__.return_value = None

            # session.get() must return an async context manager, not be a coroutine itself
            mock_session.get = MagicMock(return_value=mock_get_ctx)

            result = await executor.execute("query", extra_param="ignored", another_param=123)

            assert (
                "Warning: The following parameters were ignored: extra_param, another_param"
                in result
            )
            assert "## Search Results" in result
            assert "Search results" in result

    @pytest.mark.asyncio
    async def test_webpage_visitor_invalid_url(self):
        """Test webpage visitor with invalid URL."""
        executor = WebpageVisitorExecutor()

        with pytest.raises(ValueError, match="Invalid URL"):
            await executor.execute("not-a-url")

    def test_resolve_http_headers_exact(self):
        secrets = {
            "http_headers": {"https://files.slack.com/files-pri/": {"Authorization": "Bearer xoxb"}}
        }
        headers = resolve_http_headers("https://files.slack.com/files-pri/", secrets)
        assert headers["Authorization"] == "Bearer xoxb"
        assert headers["User-Agent"] == "muaddib/1.0"

    def test_resolve_http_headers_prefix(self):
        secrets = {
            "http_header_prefixes": {"https://files.slack.com/": {"Authorization": "Bearer xoxb"}}
        }
        headers = resolve_http_headers("https://files.slack.com/files-pri/test", secrets)
        assert headers["Authorization"] == "Bearer xoxb"
        assert headers["User-Agent"] == "muaddib/1.0"

    @pytest.mark.asyncio
    async def test_webpage_visitor_authenticated_fetch(self):
        secrets = {
            "http_header_prefixes": {"https://files.slack.com/": {"Authorization": "Bearer xoxb"}}
        }
        executor = WebpageVisitorExecutor(secrets=secrets)
        executor._fetch = AsyncMock(side_effect=AssertionError("Should not call jina fetch"))

        mock_head_response = AsyncMock()
        mock_head_response.headers = {"content-type": "text/plain"}
        mock_head_response.raise_for_status = MagicMock()

        mock_get_response = AsyncMock()
        mock_get_response.headers = {"content-type": "text/plain"}
        mock_get_response.read = AsyncMock(return_value=b"hello")
        mock_get_response.raise_for_status = MagicMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_head_context = AsyncMock()
        mock_head_context.__aenter__ = AsyncMock(return_value=mock_head_response)
        mock_head_context.__aexit__ = AsyncMock(return_value=None)

        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_context.__aexit__ = AsyncMock(return_value=None)

        mock_session.head = MagicMock(return_value=mock_head_context)
        mock_session.get = MagicMock(return_value=mock_get_context)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await executor.execute("https://files.slack.com/files-pri/test")

        assert result == "## Content from https://files.slack.com/files-pri/test\n\nhello"
        _, get_kwargs = mock_session.get.call_args
        assert get_kwargs["headers"]["Authorization"] == "Bearer xoxb"

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
    async def test_code_executor_e2b_success(self):
        """Test code executor with successful execution."""
        executor = CodeExecutorE2B()

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
    async def test_code_executor_e2b_import_error(self):
        """Test code executor when e2b package is not available."""
        executor = CodeExecutorE2B()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'e2b_code_interpreter'")
        ):
            result = await executor.execute("print('test')")

            assert "e2b-code-interpreter package not installed" in result

    @pytest.mark.asyncio
    async def test_code_executor_e2b_persistence_across_calls(self):
        """Test that sandbox persists state across multiple execute() calls."""
        executor = CodeExecutorE2B()

        mock_execution = MagicMock()
        mock_execution.text = None
        mock_execution.logs = MagicMock(stdout=["result\n"], stderr=[])
        mock_execution.results = None

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-123"

        sandbox_created = False

        async def mock_to_thread_impl(func, *args, **kwargs):
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
    async def test_code_executor_e2b_cleanup(self):
        """Test that cleanup properly kills the sandbox."""
        executor = CodeExecutorE2B()

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
    async def test_code_executor_e2b_error_resets_sandbox(self):
        """Test that connection errors reset the sandbox."""
        executor = CodeExecutorE2B()

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
    async def test_code_executor_e2b_timeout_parameter(self):
        """Test that timeout is passed to Sandbox constructor."""
        executor = CodeExecutorE2B(timeout=180)

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
    async def test_python_executor_e2b_auto_capture_png(self):
        """Test that PNG images from results are auto-captured and uploaded."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            from muaddib.agentic_actor.tools import ArtifactStore

            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"
            store = ArtifactStore(artifacts_path=artifacts_path, artifacts_url=artifacts_url)
            executor = CodeExecutorE2B(artifact_store=store)

            # Mock result with PNG data (base64 encoded small PNG)
            mock_result_item = MagicMock()
            mock_result_item.text = None
            # Minimal valid PNG (1x1 transparent pixel)
            mock_result_item.png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            mock_result_item.jpeg = None

            mock_execution = MagicMock()
            mock_execution.text = None
            mock_execution.logs = MagicMock(stdout=[], stderr=[])
            mock_execution.results = [mock_result_item]

            mock_sandbox = MagicMock()
            mock_sandbox.sandbox_id = "test-sandbox-png"

            sandbox_created = False

            async def mock_to_thread_impl(func, *args, **kwargs):
                nonlocal sandbox_created
                if not sandbox_created:
                    sandbox_created = True
                    return mock_sandbox
                return mock_execution

            with patch("e2b_code_interpreter.Sandbox"):
                with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                    result = await executor.execute("import matplotlib; plt.savefig('test.png')")

                    assert "**Generated image:**" in result
                    assert "https://example.com/artifacts/" in result
                    assert ".png" in result

                    # Verify file was created
                    artifacts_dir = Path(artifacts_path)
                    assert artifacts_dir.exists()
                    png_files = list(artifacts_dir.glob("*.png"))
                    assert len(png_files) == 1

    @pytest.mark.asyncio
    async def test_code_executor_e2b_output_files_download(self):
        """Test that explicit output_files are downloaded from sandbox and uploaded."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            from muaddib.agentic_actor.tools import ArtifactStore

            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"
            store = ArtifactStore(artifacts_path=artifacts_path, artifacts_url=artifacts_url)
            executor = CodeExecutorE2B(artifact_store=store)

            mock_execution = MagicMock()
            mock_execution.text = None
            mock_execution.logs = MagicMock(stdout=["Done\n"], stderr=[])
            mock_execution.results = []

            mock_sandbox = MagicMock()
            mock_sandbox.sandbox_id = "test-sandbox-files"
            mock_sandbox.files = MagicMock()
            # E2B returns bytearray, not bytes
            mock_sandbox.files.read = MagicMock(return_value=bytearray(b"col1,col2\nval1,val2\n"))

            sandbox_created = False
            code_saved = False
            code_executed = False

            async def mock_to_thread_impl(func, *args, **kwargs):
                nonlocal sandbox_created, code_saved, code_executed
                if not sandbox_created:
                    sandbox_created = True
                    return mock_sandbox
                # Auto-save to /tmp/_last.py
                if not code_saved:
                    code_saved = True
                    return None  # files.write returns None
                # After sandbox created, next call is run_code, then file reads are lambdas
                if not code_executed:
                    code_executed = True
                    return mock_execution
                # Subsequent calls are lambdas for files.read - call them
                return func()

            with patch("e2b_code_interpreter.Sandbox"):
                with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                    result = await executor.execute(
                        "import pandas; df.to_csv('/tmp/report.csv')",
                        output_files=["/tmp/report.csv"],
                    )

                    assert "**Output:**" in result
                    assert "Done" in result
                    assert "**Downloaded file (report.csv):**" in result
                    assert "https://example.com/artifacts/" in result
                    assert ".csv" in result

                    # Verify file was created
                    artifacts_dir = Path(artifacts_path)
                    csv_files = list(artifacts_dir.glob("*.csv"))
                    assert len(csv_files) == 1

    @pytest.mark.asyncio
    async def test_code_executor_e2b_output_files_no_store(self):
        """Test that output_files warns when artifact store is not configured."""
        executor = CodeExecutorE2B(artifact_store=None)

        mock_execution = MagicMock()
        mock_execution.text = None
        mock_execution.logs = MagicMock(stdout=["Done\n"], stderr=[])
        mock_execution.results = []

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-nostore"

        sandbox_created = False
        code_saved = False

        async def mock_to_thread_impl(func, *args, **kwargs):
            nonlocal sandbox_created, code_saved
            if not sandbox_created:
                sandbox_created = True
                return mock_sandbox
            # Auto-save to /tmp/_last.py
            if not code_saved:
                code_saved = True
                return None
            return mock_execution

        with patch("e2b_code_interpreter.Sandbox"):
            with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                result = await executor.execute(
                    "print('test')",
                    output_files=["/tmp/report.csv"],
                )

                assert "artifact store not configured" in result

    @pytest.mark.asyncio
    async def test_code_executor_e2b_input_artifacts(self):
        """Test that input_artifacts are downloaded and uploaded to sandbox."""
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as temp_dir:
            from muaddib.agentic_actor.tools import ArtifactStore, WebpageVisitorExecutor

            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"
            store = ArtifactStore(artifacts_path=artifacts_path, artifacts_url=artifacts_url)

            # Mock webpage visitor to return artifact content
            mock_visitor = AsyncMock(spec=WebpageVisitorExecutor)
            mock_visitor.execute.return_value = "col1,col2\nval1,val2\n"

            executor = CodeExecutorE2B(artifact_store=store, webpage_visitor=mock_visitor)

            mock_execution = MagicMock()
            mock_execution.error = None
            mock_execution.text = None
            mock_execution.logs = MagicMock(stdout=["Processed data\n"], stderr=[])
            mock_execution.results = []

            mock_sandbox = MagicMock()
            mock_sandbox.sandbox_id = "test-sandbox-input"
            mock_sandbox.files = MagicMock()
            mock_sandbox.files.make_dir = MagicMock()
            mock_sandbox.files.write = MagicMock()

            sandbox_created = False
            dir_created = False
            files_written = 0
            code_executed = False

            async def mock_to_thread_impl(func, *args, **kwargs):
                nonlocal sandbox_created, dir_created, files_written, code_executed
                if not sandbox_created:
                    sandbox_created = True
                    return mock_sandbox
                # files.make_dir for /artifacts
                if not dir_created:
                    dir_created = True
                    return None
                # files.write for artifact upload and _last.py
                if files_written < 2:
                    files_written += 1
                    return None
                if not code_executed:
                    code_executed = True
                    return mock_execution
                return func()

            with patch("e2b_code_interpreter.Sandbox"):
                with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                    result = await executor.execute(
                        "import pandas as pd; df = pd.read_csv('/artifacts/data.csv')",
                        input_artifacts=["https://example.com/artifacts/data.csv"],
                    )

                    assert "**Output:**" in result
                    assert "Processed data" in result
                    assert "Uploaded text: /artifacts/data.csv" in result
                    assert "_v1.py" in result  # versioned file

                    # Verify visitor was called with artifact URL
                    mock_visitor.execute.assert_called_once_with(
                        "https://example.com/artifacts/data.csv"
                    )

    @pytest.mark.asyncio
    async def test_code_executor_e2b_input_artifacts_no_visitor(self):
        """Test that input_artifacts warns when webpage visitor not configured."""
        executor = CodeExecutorE2B(artifact_store=None, webpage_visitor=None)

        mock_execution = MagicMock()
        mock_execution.error = None
        mock_execution.text = None
        mock_execution.logs = MagicMock(stdout=["Done\n"], stderr=[])
        mock_execution.results = []

        mock_sandbox = MagicMock()
        mock_sandbox.sandbox_id = "test-sandbox-novisitor"

        sandbox_created = False
        code_saved = False

        async def mock_to_thread_impl(func, *args, **kwargs):
            nonlocal sandbox_created, code_saved
            if not sandbox_created:
                sandbox_created = True
                return mock_sandbox
            if not code_saved:
                code_saved = True
                return None
            return mock_execution

        with patch("e2b_code_interpreter.Sandbox"):
            with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                result = await executor.execute(
                    "print('test')",
                    input_artifacts=["https://example.com/artifacts/data.csv"],
                )

                assert "webpage visitor not configured" in result

    @pytest.mark.asyncio
    async def test_code_executor_e2b_input_artifacts_image(self):
        """Test that image artifacts are decoded and uploaded to sandbox."""
        import base64
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as temp_dir:
            from muaddib.agentic_actor.tools import ArtifactStore, WebpageVisitorExecutor

            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"
            store = ArtifactStore(artifacts_path=artifacts_path, artifacts_url=artifacts_url)

            # Mock webpage visitor to return image content (Anthropic format)
            mock_visitor = AsyncMock(spec=WebpageVisitorExecutor)
            fake_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00"  # PNG header
            mock_visitor.execute.return_value = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(fake_image_data).decode(),
                    },
                }
            ]

            executor = CodeExecutorE2B(artifact_store=store, webpage_visitor=mock_visitor)

            mock_execution = MagicMock()
            mock_execution.error = None
            mock_execution.text = None
            mock_execution.logs = MagicMock(stdout=["Image processed\n"], stderr=[])
            mock_execution.results = []

            mock_sandbox = MagicMock()
            mock_sandbox.sandbox_id = "test-sandbox-image"
            mock_sandbox.files = MagicMock()
            mock_sandbox.files.make_dir = MagicMock()
            written_files = []

            sandbox_created = False
            dir_created = False
            files_written = 0
            code_executed = False

            async def mock_to_thread_impl(func, *args, **kwargs):
                nonlocal sandbox_created, dir_created, files_written, code_executed
                if not sandbox_created:
                    sandbox_created = True
                    return mock_sandbox
                if not dir_created:
                    dir_created = True
                    return None
                # Capture files.write calls (image upload + _v1.py)
                if files_written < 2:
                    files_written += 1
                    if args:
                        written_files.append((args[0], args[1] if len(args) > 1 else None))
                    return None
                if not code_executed:
                    code_executed = True
                    return mock_execution
                return func(*args, **kwargs)

            with patch("e2b_code_interpreter.Sandbox"):
                with patch("asyncio.to_thread", side_effect=mock_to_thread_impl):
                    result = await executor.execute(
                        "from PIL import Image; img = Image.open('/artifacts/test.png')",
                        input_artifacts=["https://example.com/artifacts/test.png"],
                    )

                    assert "**Uploaded image: /artifacts/test.png**" in result
                    assert "Image processed" in result

                    # Verify image was written with decoded binary data
                    image_writes = [f for f in written_files if f[0] == "/artifacts/test.png"]
                    assert len(image_writes) == 1
                    assert image_writes[0][1] == fake_image_data

    @pytest.mark.asyncio
    async def test_share_artifact_executor_success(self):
        """Test artifact sharing with valid configuration."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_path = str(Path(temp_dir) / "artifacts")
            artifacts_url = "https://example.com/artifacts"

            from muaddib.agentic_actor.tools import ArtifactStore

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

            # Verify base62 ID format (8 alphanumeric chars)
            id_part = filename.replace(".txt", "")
            assert len(id_part) == 8
            assert all(
                c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                for c in id_part
            )

    @pytest.mark.asyncio
    async def test_share_artifact_executor_missing_config(self):
        """Test artifact sharing with missing configuration."""
        from muaddib.agentic_actor.tools import ArtifactStore

        store = ArtifactStore(artifacts_path=None, artifacts_url=None)
        executor = ShareArtifactExecutor(store=store)

        result = await executor.execute("test content")

        assert result == "Error: artifacts.path and artifacts.url must be configured"

    @pytest.mark.asyncio
    async def test_share_artifact_executor_write_error(self, tmp_path):
        """Test artifact sharing with write error."""
        from muaddib.agentic_actor.tools import ArtifactStore

        blocked_path = tmp_path / "not_a_dir"
        blocked_path.write_text("content")
        store = ArtifactStore(
            artifacts_path=str(blocked_path),
            artifacts_url="https://example.com/artifacts",
        )
        executor = ShareArtifactExecutor(store=store)

        result = await executor.execute("test content")

        assert result.startswith("Error: Failed to create artifacts directory:")

    @pytest.mark.asyncio
    async def test_webpage_visitor_local_artifact(self, artifact_store, webpage_visitor):
        """Test that webpage visitor can read local artifacts directly."""

        # Create a local artifact
        content = "def test():\n    return 42"
        url = artifact_store.write_text(content, ".py")

        # Visit the local artifact URL
        result = await webpage_visitor.execute(url)

        # Should return raw content without markdown wrapper
        assert result == content
        assert not result.startswith("## Content from")

    @pytest.mark.asyncio
    async def test_webpage_visitor_path_traversal_blocked(
        self, artifact_store, webpage_visitor, artifacts_url
    ):
        """Test that path traversal attempts are blocked."""
        import pytest

        # Try to traverse outside artifacts directory
        malicious_url = f"{artifacts_url}/../../../etc/passwd"

        with pytest.raises(ValueError, match="Path traversal detected"):
            await webpage_visitor.execute(malicious_url)

    @pytest.mark.asyncio
    async def test_edit_artifact_local(self, artifact_store, webpage_visitor):
        """Test editing a local artifact via direct filesystem."""
        from pathlib import Path

        # Create local artifact
        initial = "def hello():\n    print('Hello')\n    return 42"
        original_url = artifact_store.write_text(initial, ".py")

        # Use real webpage visitor for local artifact handling
        executor = EditArtifactExecutor(store=artifact_store, webpage_visitor=webpage_visitor)

        # Edit the artifact
        result = await executor.execute(
            original_url,
            "    print('Hello')\n    return 42",
            "    print('Hello, World!')\n    return 100",
        )

        # Verify success
        assert result.startswith("Artifact edited successfully. New version:")
        assert result.endswith(".py")

        # Verify content was edited
        new_url = result.split(": ")[1]
        filename = new_url.split("/")[-1]
        artifact_file = Path(artifact_store.artifacts_path) / filename
        edited_content = artifact_file.read_text()

        assert "Hello, World!" in edited_content
        assert "return 100" in edited_content
        assert "return 42" not in edited_content

    @pytest.mark.asyncio
    async def test_edit_artifact_remote(self, artifact_store, make_edit_executor):
        """Test editing a remote artifact via webpage visitor."""
        from pathlib import Path

        # Mock remote artifact
        initial = "console.log('test');\nvar x = 42;"
        executor, visitor = make_edit_executor(
            visitor_result=f"## Content from https://external.com/script.js\n\n{initial}"
        )

        # Edit the artifact
        result = await executor.execute(
            "https://external.com/script.js",
            "var x = 42;",
            "var x = 100;",
        )

        # Verify success
        assert result.startswith("Artifact edited successfully. New version:")
        assert result.endswith(".js")

        # Verify visitor called
        visitor.execute.assert_called_once_with("https://external.com/script.js")

        # Verify content was edited
        new_url = result.split(": ")[1]
        filename = new_url.split("/")[-1]
        artifact_file = Path(artifact_store.artifacts_path) / filename
        edited_content = artifact_file.read_text()

        assert "var x = 100;" in edited_content
        assert "var x = 42;" not in edited_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url,expected_suffix",
        [
            ("https://example.com/file.py", ".py"),
            ("https://example.com/script.js", ".js"),
            ("https://example.com/data.json", ".json"),
            ("https://example.com/archive.tar.gz", ".tar.gz"),
            ("https://example.com/noextension", ".txt"),  # Fallback
        ],
    )
    async def test_edit_artifact_suffix_extraction(self, make_edit_executor, url, expected_suffix):
        """Test suffix extraction from various URL formats."""
        executor, _ = make_edit_executor(visitor_result=f"## Content from {url}\n\ntest content")
        result = await executor.execute(url, "test", "modified")
        assert result.endswith(expected_suffix)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content,old,error_message",
        [
            ("def hello():\n    print('Hello')", "nonexistent", "Error: old_string not found"),
            ("print('test')\nprint('test')", "print('test')", "Error: old_string appears 2 times"),
        ],
        ids=["not-found", "non-unique"],
    )
    async def test_edit_artifact_validation_errors(
        self, make_edit_executor, content, old, error_message
    ):
        """Test edit validation: old_string not found or non-unique."""
        executor, _ = make_edit_executor(
            visitor_result=f"## Content from https://external.com/test.py\n\n{content}"
        )
        result = await executor.execute("https://external.com/test.py", old, "new")
        assert result.startswith(error_message)

    @pytest.mark.asyncio
    async def test_edit_artifact_binary_rejection(self, make_edit_executor):
        """Test editing binary (image) artifact returns error."""
        from .conftest import image_content_block

        executor, _ = make_edit_executor(visitor_result=image_content_block())
        result = await executor.execute("https://external.com/test.png", "old", "new")
        assert result == "Error: Cannot edit binary artifacts (images)"

    @pytest.mark.asyncio
    async def test_edit_artifact_fetch_error(self, make_edit_executor):
        """Test editing when artifact fetch fails."""
        executor, _ = make_edit_executor(visitor_exc=Exception("Network error"))
        result = await executor.execute("https://external.com/test.py", "old", "new")
        assert result.startswith("Error: Failed to fetch artifact:")
        assert "Network error" in result


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
    async def test_execute_tool_code_executor(self, mock_agent):
        """Test executing code tool."""
        with patch.object(CodeExecutorE2B, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Code output"

            tool_executors = create_tool_executors(agent=mock_agent, arc="test")
            result = await execute_tool("execute_code", tool_executors, code="print('test')")

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

        assert "execute_code" in executors
        code_executor = executors["execute_code"]
        assert isinstance(code_executor, CodeExecutorE2B)
        assert code_executor.api_key == "test-key-123"

    def test_create_tool_executors_without_config(self, mock_agent):
        """Test that tool executors are created without configuration."""
        executors = create_tool_executors(agent=mock_agent, arc="test")

        assert "execute_code" in executors
        code_executor = executors["execute_code"]
        assert isinstance(code_executor, CodeExecutorE2B)
        assert code_executor.api_key is None

    def test_make_plan_tool_in_tools_list(self):
        """Test that make_plan tool is included in TOOLS list."""
        from muaddib.agentic_actor.tools import TOOLS

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
        from muaddib.agentic_actor.tools import TOOLS

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
        from muaddib.agentic_actor.tools import TOOLS
        from muaddib.chronicler.tools import chronicle_tools_defs

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
        from muaddib.providers.anthropic import AnthropicClient

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

    def test_get_tools_for_arc_quests_filtering(self):
        """Test that chronicle_append is only available when quests are enabled for the arc."""
        from muaddib.agentic_actor.actor import get_tools_for_arc

        # Config with quests enabled for specific arc
        config = {"chronicler": {"quests": {"arcs": ["server#quests-channel"]}}}

        # Arc with quests enabled - should have chronicle_append
        tools_with_quests = get_tools_for_arc(config, "server#quests-channel")
        tool_names = [t["name"] for t in tools_with_quests]
        assert "chronicle_append" in tool_names
        assert "chronicle_read" in tool_names

        # Arc without quests - should NOT have chronicle_append
        tools_without_quests = get_tools_for_arc(config, "server#other-channel")
        tool_names = [t["name"] for t in tools_without_quests]
        assert "chronicle_append" not in tool_names
        assert "chronicle_read" in tool_names

        # Empty config - no quests anywhere
        tools_no_config = get_tools_for_arc({}, "any-arc")
        tool_names = [t["name"] for t in tools_no_config]
        assert "chronicle_append" not in tool_names
        assert "chronicle_read" in tool_names


class TestOracleExecutor:
    """Tests for OracleExecutor."""

    @pytest.mark.asyncio
    async def test_oracle_missing_model_config(self):
        """Test oracle returns error when model not configured."""
        config = {"tools": {}}
        agent = MagicMock()
        agent.chronicle = MagicMock()

        executor = OracleExecutor(
            config=config,
            agent=agent,
            arc="server#channel",
            conversation_context=[],
        )

        result = await executor.execute("test query")
        assert "Error: oracle.model not configured" in result

    @pytest.mark.asyncio
    async def test_oracle_builds_context_correctly(self):
        """Test oracle passes conversation context + query to nested actor."""
        config = {
            "tools": {
                "oracle": {
                    "model": "anthropic:claude-opus-4-5",
                    "prompt": "You are an oracle.",
                }
            },
            "actor": {"max_iterations": 5},
        }
        agent = MagicMock()
        agent.chronicle = MagicMock()

        conversation = [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Some response"},
        ]

        executor = OracleExecutor(
            config=config,
            agent=agent,
            arc="server#channel",
            conversation_context=conversation,
        )

        with patch("muaddib.agentic_actor.actor.AgenticLLMActor") as mock_actor_class:
            mock_actor = AsyncMock()
            mock_actor.run_agent = AsyncMock(
                return_value=AgentResult(
                    text="Oracle says hello",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost=0.01,
                    tool_calls_count=2,
                )
            )
            mock_actor_class.return_value = mock_actor

            result = await executor.execute("What should I do?")

            assert result == "Oracle says hello"

            # Check the context passed to run_agent
            call_args = mock_actor.run_agent.call_args
            passed_context = call_args[0][0]
            assert len(passed_context) == 3
            assert passed_context[0] == {"role": "user", "content": "Original question"}
            assert passed_context[1] == {"role": "assistant", "content": "Some response"}
            assert passed_context[2] == {"role": "user", "content": "What should I do?"}

    @pytest.mark.asyncio
    async def test_oracle_excludes_correct_tools(self):
        """Test oracle excludes oracle, progress_report, and quest tools."""
        config = {
            "tools": {
                "oracle": {
                    "model": "anthropic:claude-opus-4-5",
                }
            },
            "actor": {"max_iterations": 5},
            "chronicler": {"quests": {"arcs": ["server#channel"]}},
        }
        agent = MagicMock()
        agent.chronicle = MagicMock()

        executor = OracleExecutor(
            config=config,
            agent=agent,
            arc="server#channel",
            conversation_context=[],
        )

        with patch("muaddib.agentic_actor.actor.AgenticLLMActor") as mock_actor_class:
            mock_actor = AsyncMock()
            mock_actor.run_agent = AsyncMock(
                return_value=AgentResult(
                    text="Result",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost=0.01,
                    tool_calls_count=2,
                )
            )
            mock_actor_class.return_value = mock_actor

            await executor.execute("query")

            # Check allowed_tools passed to AgenticLLMActor
            call_kwargs = mock_actor_class.call_args[1]
            allowed_tools = call_kwargs["allowed_tools"]

            assert "oracle" not in allowed_tools
            assert "progress_report" not in allowed_tools
            assert "quest_start" not in allowed_tools
            assert "subquest_start" not in allowed_tools
            # Should still have chronicle_read
            assert "chronicle_read" in allowed_tools
            # Should have standard tools
            assert "web_search" in allowed_tools
            assert "visit_webpage" in allowed_tools

    @pytest.mark.asyncio
    async def test_oracle_handles_generic_error(self):
        """Test oracle handles errors from nested actor gracefully."""
        config = {
            "tools": {"oracle": {"model": "anthropic:claude-opus-4-5"}},
            "actor": {"max_iterations": 5},
        }
        agent = MagicMock()
        agent.chronicle = MagicMock()

        executor = OracleExecutor(
            config=config,
            agent=agent,
            arc="server#channel",
            conversation_context=[],
        )

        with patch("muaddib.agentic_actor.actor.AgenticLLMActor") as mock_actor_class:
            mock_actor = AsyncMock()
            mock_actor.run_agent = AsyncMock(side_effect=RuntimeError("something went wrong"))
            mock_actor_class.return_value = mock_actor

            result = await executor.execute("complex query")

            assert "Oracle error:" in result
            assert "something went wrong" in result

    @pytest.mark.asyncio
    async def test_oracle_passes_progress_callback(self):
        """Test oracle passes progress callback to nested actor."""
        config = {
            "tools": {"oracle": {"model": "anthropic:claude-opus-4-5"}},
            "actor": {"max_iterations": 5},
        }
        agent = MagicMock()
        agent.chronicle = MagicMock()
        progress_cb = AsyncMock()

        executor = OracleExecutor(
            config=config,
            agent=agent,
            arc="server#channel",
            conversation_context=[],
            progress_callback=progress_cb,
        )

        with patch("muaddib.agentic_actor.actor.AgenticLLMActor") as mock_actor_class:
            mock_actor = AsyncMock()
            mock_actor.run_agent = AsyncMock(
                return_value=AgentResult(
                    text="Done",
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_cost=0.01,
                    tool_calls_count=2,
                )
            )
            mock_actor_class.return_value = mock_actor

            await executor.execute("query")

            # Check progress_callback passed to run_agent
            call_kwargs = mock_actor.run_agent.call_args[1]
            assert call_kwargs["progress_callback"] is progress_cb
