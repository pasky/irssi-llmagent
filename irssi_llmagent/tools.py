"""Tool definitions and executors for Claude agent."""

import asyncio
import logging
import re
import time
from typing import TypedDict

import aiohttp
from ddgs import DDGS

logger = logging.getLogger(__name__)


class Tool(TypedDict):
    """Tool definition schema."""

    name: str
    description: str
    input_schema: dict


# Available tools for Claude agent
TOOLS: list[Tool] = [
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo and return top results with titles, URLs, and descriptions.",
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

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "irssi-llmagent/1.0"},
            ) as response,
        ):
            response.raise_for_status()
            html_content = await response.text()

        # Convert HTML to markdown
        markdown_content = markdownify(html_content).strip()

        # Clean up multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Truncate if too long
        if len(markdown_content) > self.max_content_length:
            truncated_content = markdown_content[:self.max_content_length - 100]  # Leave room for message
            markdown_content = truncated_content + "\n\n..._Content truncated_..."

        return f"## Content from {url}\n\n{markdown_content}"


# Tool executor registry
TOOL_EXECUTORS = {
    "web_search": WebSearchExecutor(),
    "visit_webpage": WebpageVisitorExecutor(),
}


async def execute_tool(tool_name: str, **kwargs) -> str:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOL_EXECUTORS:
        raise ValueError(f"Unknown tool '{tool_name}'")

    executor = TOOL_EXECUTORS[tool_name]
    return await executor.execute(**kwargs)
