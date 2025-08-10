"""Tool definitions and executors for AI agent."""

import asyncio
import logging
import re
import time
from typing import Any, TypedDict

import aiohttp
from ddgs import DDGS

logger = logging.getLogger(__name__)


class Tool(TypedDict):
    """Tool definition schema."""

    name: str
    description: str
    input_schema: dict


# Available tools for AI agent
TOOLS: list[Tool] = [
    {
        "name": "web_search",
        "description": "Search the web and return top results with titles, URLs, and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to perform."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "visit_webpage",
        "description": "Visit a webpage at the given URL and return its content as markdown text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL of the webpage to visit.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "execute_python",
        "description": "Execute Python code in a sandbox environment and return the output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "progress_report",
        "description": "Send a brief one-line progress update to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One-line progress update. Keep it super concise, but very casual and even snarky in line with your instructions and previous conversation.",
                }
            },
            "required": ["text"],
        },
    },
]


class RateLimiter:
    """Simple rate limiter for tool calls."""

    def __init__(self, max_calls_per_second: float = 1.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self.last_call_time = 0.0

    async def wait_if_needed(self):
        """Wait if needed to respect rate limit."""
        if self.min_interval <= 0:
            return

        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()


class WebSearchExecutor:
    """Async DuckDuckGo web search executor."""

    def __init__(self, max_results: int = 5, max_calls_per_second: float = 1.0):
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)

    async def execute(self, query: str) -> str:
        """Execute web search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        # Note: DDGS is not async, so we run it in executor
        loop = asyncio.get_event_loop()
        with DDGS() as ddgs:
            results = await loop.run_in_executor(
                None, lambda: list(ddgs.text(query, max_results=self.max_results))
            )

        logger.info(f"Searching '{query}': {len(results)} results")

        if not results:
            return "No search results found. Try a different query."

        # Format results as markdown
        formatted_results = []
        for result in results:
            title = result.get("title", "No title")
            url = result.get("href", "#")
            body = result.get("body", "No description")
            formatted_results.append(f"[{title}]({url})\n{body}")

        return "## Search Results\n\n" + "\n\n".join(formatted_results)


class WebpageVisitorExecutor:
    """Async webpage visitor and content extractor."""

    def __init__(self, max_content_length: int = 40000, timeout: int = 20):
        self.max_content_length = max_content_length
        self.timeout = timeout

    async def execute(self, url: str) -> str:
        """Visit webpage and return content as markdown."""
        from markdownify import markdownify

        logger.info(f"Visiting {url}")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        # Use separate context managers to avoid UnboundLocalError bug
        async with aiohttp.ClientSession() as session:  # noqa: SIM117
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "irssi-llmagent/1.0"},
            ) as response:
                response.raise_for_status()
                html_content = await response.text()

        # Convert HTML to markdown
        markdown_content = markdownify(html_content).strip()

        # Clean up multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Truncate if too long
        if len(markdown_content) > self.max_content_length:
            truncated_content = markdown_content[
                : self.max_content_length - 100
            ]  # Leave room for message
            markdown_content = truncated_content + "\n\n..._Content truncated_..."
            logger.warning(
                f"{url} truncated from {len(markdown_content)} to {len(truncated_content)}"
            )

        return f"## Content from {url}\n\n{markdown_content}"


class PythonExecutorE2B:
    """Python code executor using E2B sandbox."""

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    async def execute(self, code: str) -> str:
        """Execute Python code in E2B sandbox and return output."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            return "Error: e2b-code-interpreter package not installed. Install with: pip install e2b-code-interpreter"

        try:
            # Use run_in_executor for synchronous E2B sandbox
            import asyncio

            loop = asyncio.get_event_loop()

            def run_sandbox_code():
                # Use API key from config if available, otherwise fallback to environment variable
                sandbox_args = {}
                if self.api_key:
                    sandbox_args["api_key"] = self.api_key
                with Sandbox(**sandbox_args) as sandbox:
                    result = sandbox.run_code(code)
                    return result

            result = await loop.run_in_executor(None, run_sandbox_code)
            logger.debug(result)

            # Collect output
            output_parts = []

            # Check logs for stdout/stderr (E2B stores them in logs.stdout/stderr as lists)
            logs = getattr(result, "logs", None)
            if logs:
                stdout_list = getattr(logs, "stdout", None)
                if stdout_list:
                    stdout_text = "".join(stdout_list).strip()
                    if stdout_text:
                        output_parts.append(f"**Output:**\n```\n{stdout_text}\n```")

                stderr_list = getattr(logs, "stderr", None)
                if stderr_list:
                    stderr_text = "".join(stderr_list).strip()
                    if stderr_text:
                        output_parts.append(f"**Errors:**\n```\n{stderr_text}\n```")

            # Check for text result (final evaluation result) - only if no stdout to avoid duplicates
            text = getattr(result, "text", None)
            if text and text.strip() and not (logs and getattr(logs, "stdout", None)):
                output_parts.append(f"**Result:**\n```\n{text.strip()}\n```")

            # Check for rich results (plots, images, etc.)
            results_list = getattr(result, "results", None)
            if results_list:
                for res in results_list:
                    result_text = getattr(res, "text", None)
                    if result_text and result_text.strip():
                        output_parts.append(f"**Result:**\n```\n{result_text.strip()}\n```")
                    # Check for images/plots
                    if hasattr(res, "png") and getattr(res, "png", None):
                        output_parts.append("**Result:** Generated plot/image (PNG data available)")
                    if hasattr(res, "jpeg") and getattr(res, "jpeg", None):
                        output_parts.append(
                            "**Result:** Generated plot/image (JPEG data available)"
                        )

            if not output_parts:
                output_parts.append("Code executed successfully with no output.")

            logger.info(
                f"Executed Python code in E2B sandbox: {code[:512]}... -> {output_parts[:512]}"
            )

            return "\n\n".join(output_parts)

        except Exception as e:
            logger.error(f"E2B sandbox execution failed: {e}")
            return f"Error executing code: {e}"


class ProgressReportExecutor:
    """Executor that sends progress updates via a provided callback."""

    def __init__(
        self,
        send_callback: Any | None = None,
        min_interval_seconds: int = 15,
    ):
        self.send_callback = send_callback
        self.min_interval_seconds = min_interval_seconds
        self._last_sent: float | None = None

    async def execute(self, text: str) -> str:
        # Sanitize to single line and trim
        clean = re.sub(r"\s+", " ", text or "").strip()
        logger.info(f"progress_report: {text}")
        if not clean:
            return "OK"

        # No-op if no callback (e.g., proactive mode)
        if not self.send_callback:
            return "OK"

        now = time.time()
        if self._last_sent is not None and (now - self._last_sent) < self.min_interval_seconds:
            return "OK"

        # Send update
        try:
            await self.send_callback(clean)
            self._last_sent = now
        except Exception as e:
            logger.warning(f"progress_report send failed: {e}")
        return "OK"


def create_tool_executors(
    config: dict | None = None, *, progress_callback: Any | None = None
) -> dict[str, Any]:
    """Create tool executors with configuration."""
    e2b_config = config["providers"].get("e2b", {}) if config else {}
    e2b_api_key = e2b_config.get("api_key")

    # Progress executor settings
    behavior = (config or {}).get("behavior", {})
    progress_cfg = behavior.get("progress", {}) if behavior else {}
    min_interval = int(progress_cfg.get("min_interval_seconds", 15))

    return {
        "web_search": WebSearchExecutor(),
        "visit_webpage": WebpageVisitorExecutor(),
        "execute_python": PythonExecutorE2B(api_key=e2b_api_key),
        "progress_report": ProgressReportExecutor(
            send_callback=progress_callback, min_interval_seconds=min_interval
        ),
    }


# Default tool executor registry (for backwards compatibility)
TOOL_EXECUTORS = create_tool_executors()


async def execute_tool(
    tool_name: str, tool_executors: dict[str, Any] | None = None, **kwargs
) -> str:
    """Execute a tool by name with given arguments."""
    executors = tool_executors or TOOL_EXECUTORS

    if tool_name not in executors:
        raise ValueError(f"Unknown tool '{tool_name}'")

    executor = executors[tool_name]
    return await executor.execute(**kwargs)
