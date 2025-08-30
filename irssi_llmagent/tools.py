"""Tool definitions and executors for AI agent."""

import asyncio
import base64
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
                "query": {
                    "type": "string",
                    "description": "The search query to perform. Never use \\u unicode escapes.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "visit_webpage",
        "description": "Visit the given URL and return its content as markdown text if HTML website, or picture if an image URL.",
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
    {
        "name": "final_answer",
        "description": "Provide the final answer to the user's question. This tool MUST be used when the agent is ready to give its final response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer or response to the user's question. Start with final deliberation in <thinking>...</thinking>.",
                }
            },
            "required": ["answer"],
        },
    },
    {
        "name": "make_plan",
        "description": "Consider different approaches and formulate a brief research and/or execution plan. Only use this tool if research or code execution seems necessary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "A brief research and/or execution plan to handle the user's request, outlining (a) concerns and key goals that require further actions before responding, (b) the key steps and approach how to address them.",
                }
            },
            "required": ["plan"],
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


class BraveSearchExecutor:
    """Async Brave Search API executor."""

    def __init__(self, api_key: str, max_results: int = 5, max_calls_per_second: float = 1.0):
        self.api_key = api_key
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)

    async def execute(self, query: str) -> str:
        """Execute Brave search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    results = data.get("web", {}).get("results", [])
                    logger.info(f"Brave searching '{query}': {len(results)} results")

                    if not results:
                        return "No search results found. Try a different query."

                    # Format results as markdown
                    formatted_results = []
                    for result in results:
                        title = result.get("title", "No title")
                        url = result.get("url", "#")
                        description = result.get("description", "No description")
                        formatted_results.append(f"[{title}]({url})\n{description}")

                    return "## Search Results\n\n" + "\n\n".join(formatted_results)

            except Exception as e:
                logger.error(f"Brave search failed: {e}")
                return f"Search failed: {e}"


class WebSearchExecutor:
    """Async ddgs web search executor."""

    def __init__(
        self, max_results: int = 10, max_calls_per_second: float = 1.0, backend: str = "auto"
    ):
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)
        self.backend = backend

    async def execute(self, query: str) -> str:
        """Execute web search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        # Note: DDGS is not async, so we run it in executor
        loop = asyncio.get_event_loop()
        with DDGS() as ddgs:
            results = await loop.run_in_executor(
                None,
                lambda: list(ddgs.text(query, max_results=self.max_results, backend=self.backend)),
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

    def __init__(
        self,
        max_content_length: int = 40000,
        timeout: int = 60,
        max_image_size: int = 3_500_000,
        progress_callback: Any | None = None,
        api_key: str | None = None,
    ):
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_image_size = max_image_size  # 5MB default limit post base64 encode
        self.progress_callback = progress_callback
        self.api_key = api_key

    async def execute(self, url: str) -> str:
        """Visit webpage and return content as markdown, or image data for images."""
        logger.info(f"Visiting {url}")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        async with aiohttp.ClientSession() as session:
            # First, check the original URL for content-type to detect images
            async with session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "irssi-llmagent/1.0"},
            ) as head_response:
                content_type = head_response.headers.get("content-type", "").lower()

                if content_type.startswith("image/"):
                    # Handle image content - fetch the actual image
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"User-Agent": "irssi-llmagent/1.0"},
                        max_line_size=8190 * 2,
                        max_field_size=8190 * 2,
                    ) as response:
                        response.raise_for_status()

                        # Check content length to avoid huge token costs
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > self.max_image_size:
                            return f"Error: Image too large ({content_length} bytes). Maximum allowed: {self.max_image_size} bytes"

                        # Handle image content
                        image_data = await response.read()

                        # Double-check actual size after download
                        if len(image_data) > self.max_image_size:
                            return f"Error: Image too large ({len(image_data)} bytes). Maximum allowed: {self.max_image_size} bytes"

                        # Convert to base64 directly
                        image_b64 = base64.b64encode(image_data).decode()
                        logger.info(
                            f"Downloaded image from {url}, content-type: {content_type}, size: {len(image_data)} bytes"
                        )
                        return f"IMAGE_DATA:{content_type}:{len(image_data)}:{image_b64}"

            # Handle text/HTML content - use jina.ai reader service
            jina_url = f"https://r.jina.ai/{url}"
            markdown_content = await self._fetch(session, jina_url)

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

    async def _fetch(self, session: aiohttp.ClientSession, jina_url: str) -> str:
        """Fetch from jina.ai with backoff on HTTP 451."""
        backoff_delays = [0, 30, 90]  # No delay, then 30s, then 90s

        for attempt, delay in enumerate(backoff_delays):
            if delay > 0:
                logger.info(f"Waiting {delay}s before retry {attempt + 1}/3 for jina.ai")
                await asyncio.sleep(delay)

            try:
                headers = {"User-Agent": "irssi-llmagent/1.0"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                async with session.get(
                    jina_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    content = await response.text()
                    return content.strip()

            except aiohttp.ClientResponseError as e:
                if (e.status == 451 or e.status >= 500) and attempt < len(backoff_delays) - 1:
                    if self.progress_callback:
                        await self.progress_callback(
                            f"r.jina.ai HTTP {e.status}, retrying in a bit..."
                        )
                    continue
                raise

        raise RuntimeError("This should not be reached")


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


class FinalAnswerExecutor:
    """Executor for providing final answers."""

    async def execute(self, answer: str) -> str:
        """Return the final answer."""
        logger.info(f"Final answer provided: {answer[:100]}...")
        return answer


class MakePlanExecutor:
    """Executor for making plans (no-op that confirms receipt)."""

    async def execute(self, plan: str) -> str:
        """Confirm plan receipt."""
        logger.info(f"Plan formulated: {plan[:200]}...")
        return "OK, follow this plan"


def create_tool_executors(
    config: dict | None = None, *, progress_callback: Any | None = None
) -> dict[str, Any]:
    """Create tool executors with configuration."""
    providers = config.get("providers", {}) if config else {}

    # E2B config
    e2b_config = providers.get("e2b", {})
    e2b_api_key = e2b_config.get("api_key")

    # Jina config
    jina_config = providers.get("jina", {})
    jina_api_key = jina_config.get("api_key")

    # Search provider config
    agent_config = config.get("agent", {}) if config else {}
    search_provider = agent_config.get("search_provider", "auto")

    # Create appropriate search executor based on provider
    if search_provider == "brave":
        brave_config = providers.get("brave", {})
        brave_api_key = brave_config.get("api_key")
        if not brave_api_key:
            logger.warning("Brave search configured but no API key found, falling back to ddgs")
            search_executor = WebSearchExecutor(backend="brave")
        else:
            search_executor = BraveSearchExecutor(api_key=brave_api_key)
    else:
        search_executor = WebSearchExecutor(backend=search_provider)

    # Progress executor settings
    behavior = (config or {}).get("behavior", {})
    progress_cfg = behavior.get("progress", {}) if behavior else {}
    min_interval = int(progress_cfg.get("min_interval_seconds", 15))

    return {
        "web_search": search_executor,
        "visit_webpage": WebpageVisitorExecutor(
            progress_callback=progress_callback, api_key=jina_api_key
        ),
        "execute_python": PythonExecutorE2B(api_key=e2b_api_key),
        "progress_report": ProgressReportExecutor(
            send_callback=progress_callback, min_interval_seconds=min_interval
        ),
        "final_answer": FinalAnswerExecutor(),
        "make_plan": MakePlanExecutor(),
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
