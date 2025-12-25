"""Tool definitions and executors for AI agent."""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypedDict

import aiohttp
from ddgs import DDGS

from ..chronicler.tools import (
    ChapterAppendExecutor,
    ChapterRenderExecutor,
    QuestStartExecutor,
    SubquestStartExecutor,
)

logger = logging.getLogger(__name__)


def generate_artifact_id(length: int = 8) -> str:
    """Generate a random base62 artifact ID."""
    BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "".join(random.choices(BASE62_ALPHABET, k=length))


async def fetch_image_b64(
    session: aiohttp.ClientSession, url: str, max_size: int, timeout: int = 30
) -> tuple[str, str]:
    """
    Fetch an image from URL and return (content_type, base64_string).
    Raises ValueError if not an image, too large, or fetch fails.
    """
    async with session.head(
        url,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": "irssi-llmagent/1.0"},
    ) as head_response:
        content_type = head_response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"URL is not an image (content-type: {content_type})")

    async with session.get(
        url,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": "irssi-llmagent/1.0"},
        max_line_size=8190 * 2,
        max_field_size=8190 * 2,
    ) as response:
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise ValueError(
                f"Image too large ({content_length} bytes). Maximum allowed: {max_size} bytes"
            )

        image_data = await response.read()
        if len(image_data) > max_size:
            raise ValueError(
                f"Image too large ({len(image_data)} bytes). Maximum allowed: {max_size} bytes"
            )

        image_b64 = base64.b64encode(image_data).decode()
        logger.info(
            f"Downloaded image from {url}, content-type: {content_type}, size: {len(image_data)} bytes"
        )
        return (content_type, image_b64)


class Tool(TypedDict):
    """Tool definition schema."""

    name: str
    description: str
    input_schema: dict
    persist: str  # "none", "exact", "summary", or "artifact"


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
        "persist": "summary",
    },
    {
        "name": "visit_webpage",
        "description": "Visit the given URL and return its content as markdown text if HTML website, or picture if an image URL (or raw if an artifact).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL of the webpage to visit or artifact to load.",
                }
            },
            "required": ["url"],
        },
        "persist": "summary",
    },
    {
        "name": "execute_code",
        "description": "Execute code in a sandbox environment and return the output. The sandbox environment is persisted to follow-up calls of this tool within this thread. For Python, generated plots/images are automatically captured and uploaded. Use output_files to download additional files from the sandbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute in the sandbox.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "bash"],
                    "default": "python",
                    "description": "The language to execute the code in.",
                },
                "output_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of file paths in the sandbox to download and share as artifacts (e.g., ['/tmp/report.csv']).",
                },
            },
            "required": ["code"],
        },
        "persist": "artifact",
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
        "persist": "none",
    },
    {
        "name": "final_answer",
        "description": "Provide the final answer to the user's question. This tool MUST be used when the agent is ready to give its final response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer or response to the user's question. Start with final deliberation in <thinking>...</thinking>. Never say 'you are doing something' or 'you will do something' - at this point, you are *done*.",
                }
            },
            "required": ["answer"],
        },
        "persist": "none",
    },
    {
        "name": "make_plan",
        "description": "Consider different approaches and formulate a brief research and/or execution plan. Only use this tool if research or code execution seems necessary. Can only be called alongside final_answer if quest_start or subquest_start is also present.",
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
        "persist": "none",
    },
    {
        "name": "share_artifact",
        "description": "Share an additional artifact (created script, detailed report, supporting data). The content is made available on a public link that is returned by the tool. Use this only for additional content that doesn't fit into your standard IRC message response (or when explicitly requested).",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the artifact to share (script, report, detailed data, etc.).",
                }
            },
            "required": ["content"],
        },
        "persist": "none",
    },
    {
        "name": "edit_artifact",
        "description": "Edit an existing artifact by replacing a unique (exactly one) occurrence of old_string with new_string, creating a new derived artifact in the process. The new artifact is shared in return value of the tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL of the artifact to edit (from a previous share_artifact or visit_webpage call).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to find and replace, which must match exactly, even whitespaces perfectly. Include enough surrounding context (3-5 lines) to ensure uniqueness.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace old_string with. Can be empty to delete text.",
                },
            },
            "required": ["artifact_url", "old_string", "new_string"],
        },
        "persist": "artifact",
    },
    {
        "name": "generate_image",
        "description": "Generate image(s) using {tools.image_gen.model} on OpenRouter - use for creating or editing pictures, photos, diagrams, visualizations, etc. Optionally provide reference image URLs for editing/variations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate.",
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"},
                    "description": "Optional list of reference image URLs to include as input for editing or creating variations.",
                },
            },
            "required": ["prompt"],
        },
        "persist": "artifact",
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


class JinaSearchExecutor:
    """Async Jina.ai search executor."""

    def __init__(
        self, max_results: int = 10, max_calls_per_second: float = 1.0, api_key: str | None = None
    ):
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)
        self.api_key = api_key

    async def execute(self, query: str, **kwargs) -> str:
        """Execute Jina search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        warning_prefix = ""
        if kwargs:
            logger.warning(f"JinaSearchExecutor received unsupported arguments: {kwargs}")
            warning_prefix = (
                f"Warning: The following parameters were ignored: {', '.join(kwargs.keys())}\n\n"
            )

        url = "https://s.jina.ai/?q=" + query
        headers = {
            "User-Agent": "irssi-llmagent/1.0",
            "X-Respond-With": "no-content",
            "Accept": "text/plain",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response.raise_for_status()
                    content = await response.text()

                    logger.info(f"Jina searching '{query}': retrieved search results")

                    if not content.strip():
                        return f"{warning_prefix}No search results found. Try a different query."

                    return f"{warning_prefix}## Search Results\n\n{content.strip()}"

            except Exception as e:
                logger.error(f"Jina search failed: {e}")
                return f"{warning_prefix}Search failed: {e}"


class WebpageVisitorExecutor:
    """Async webpage visitor and content extractor."""

    def __init__(
        self,
        max_content_length: int = 40000,
        timeout: int = 60,
        max_image_size: int = 3_500_000,
        progress_callback: Any | None = None,
        api_key: str | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_image_size = max_image_size  # 5MB default limit post base64 encode
        self.progress_callback = progress_callback
        self.api_key = api_key
        self.artifact_store = artifact_store

    async def execute(self, url: str) -> str | list[dict]:
        """Visit webpage and return content as markdown, or image data for images."""
        logger.info(f"Visiting {url}")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        # Check if this is a local artifact we can access directly
        if (
            self.artifact_store
            and self.artifact_store.artifacts_url
            and self.artifact_store.artifacts_path
            and url.startswith(self.artifact_store.artifacts_url + "/")
        ):
            # Direct file system access for local artifacts
            logger.info(f"Using direct filesystem access for local artifact: {url}")
            filename = url[len(self.artifact_store.artifacts_url) + 1 :]
            filepath = self.artifact_store.artifacts_path / filename

            # Resolve paths and validate no path traversal
            try:
                # Resolve paths (non-strict to catch traversal even if target doesn't exist)
                resolved_path = filepath.resolve()
                artifacts_base = self.artifact_store.artifacts_path.resolve()

                # Validate containment
                if not resolved_path.is_relative_to(artifacts_base):
                    raise ValueError("Path traversal detected")

                # Check existence after validating path
                if not resolved_path.exists():
                    raise ValueError("Artifact file not found")

                # Check file size to prevent reading huge files
                file_size = resolved_path.stat().st_size
                if file_size > self.max_content_length:
                    content = resolved_path.read_text(encoding="utf-8")[: self.max_content_length]
                    logger.warning(
                        f"Local artifact truncated from {file_size} to {self.max_content_length}"
                    )
                    return content + "\n\n..._Content truncated_..."
                else:
                    content = resolved_path.read_text(encoding="utf-8")

                logger.info(f"Read local artifact: {filename}")
                return content
            except Exception as e:
                logger.error(f"Failed to read local artifact '{filename}': {e}")
                raise ValueError(f"Failed to read artifact: {e}") from e

        async with aiohttp.ClientSession() as session:
            # First, check the original URL for content-type to detect images
            async with session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "irssi-llmagent/1.0"},
            ) as head_response:
                content_type = head_response.headers.get("content-type", "").lower()

                if content_type.startswith("image/"):
                    try:
                        content_type, image_b64 = await fetch_image_b64(
                            session, url, self.max_image_size, self.timeout
                        )
                        # Return Anthropic content blocks with image
                        return [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content_type,
                                    "data": image_b64,
                                },
                            }
                        ]
                    except ValueError as e:
                        return f"Error: {e}"

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
                    # Only send error info on second failure (attempt 1) to reduce spam
                    if self.progress_callback and attempt == 1:
                        await self.progress_callback(
                            f"r.jina.ai HTTP {e.status}, retrying in a bit...", "progress"
                        )
                    continue
                raise

        raise RuntimeError("This should not be reached")


class CodeExecutorE2B:
    """Code executor using E2B sandbox. Supports Python and Bash."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 600,
        artifact_store: ArtifactStore | None = None,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.sandbox = None
        self.artifact_store = artifact_store

    async def _ensure_sandbox(self):
        """Ensure sandbox is created and connected."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "e2b-code-interpreter package not installed. Install with: pip install e2b-code-interpreter"
            ) from None

        if self.sandbox is None:
            import asyncio

            def create_sandbox():
                sandbox_args: dict[str, Any] = {"timeout": self.timeout}
                if self.api_key:
                    sandbox_args["api_key"] = self.api_key
                sandbox = Sandbox(**sandbox_args)
                return sandbox

            self.sandbox = await asyncio.to_thread(create_sandbox)
            logger.info(f"Created new E2B sandbox with ID: {self.sandbox.sandbox_id}")

    def _upload_image_data(self, data: bytes, suffix: str) -> str | None:
        """Upload image data to artifact store, return URL or None."""
        if not self.artifact_store:
            return None
        url = self.artifact_store.write_bytes(data, suffix)
        if url.startswith("Error:"):
            logger.warning(f"Failed to upload image artifact: {url}")
            return None
        return url

    async def _download_sandbox_file(self, path: str) -> tuple[bytes | bytearray, str] | None:
        """Download a file from the sandbox, return (data, filename) or None."""
        import asyncio
        from pathlib import PurePosixPath

        assert self.sandbox is not None
        try:
            # E2B files.read returns str by default, use format='bytes' for binary
            data = await asyncio.to_thread(
                lambda: self.sandbox.files.read(path, format="bytes")  # type: ignore[union-attr]
            )
            filename = PurePosixPath(path).name
            return data, filename
        except Exception as e:
            logger.warning(f"Failed to download sandbox file {path}: {e}")
            return None

    def _get_suffix_from_filename(self, filename: str) -> str:
        """Extract file suffix from filename."""
        from pathlib import PurePosixPath

        suffix = PurePosixPath(filename).suffix
        return suffix if suffix else ".bin"

    async def execute(
        self, code: str, language: str = "python", output_files: list[str] | None = None
    ) -> str:
        """Execute code in E2B sandbox and return output."""
        try:
            await self._ensure_sandbox()
        except ImportError as e:
            return str(e)

        try:
            import asyncio

            assert self.sandbox is not None
            result = await asyncio.to_thread(self.sandbox.run_code, code, language=language)
            logger.debug(result)

            output_parts = []
            artifact_urls = []

            # Check for execution error first (e.g., ModuleNotFoundError)
            error = getattr(result, "error", None)
            if error:
                output_parts.append(f"**Execution error:** {error}")

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

            # Check for rich results (plots, images, etc.) - auto-capture and upload
            results_list = getattr(result, "results", None)
            if results_list:
                for res in results_list:
                    result_text = getattr(res, "text", None)
                    if result_text and result_text.strip():
                        output_parts.append(f"**Result:**\n```\n{result_text.strip()}\n```")

                    # Auto-capture PNG images
                    png_data = getattr(res, "png", None)
                    if png_data:
                        img_bytes = base64.b64decode(png_data)
                        url = self._upload_image_data(img_bytes, ".png")
                        if url:
                            artifact_urls.append(url)
                            output_parts.append(f"**Generated image:** {url}")
                        else:
                            output_parts.append(
                                "**Result:** Generated plot/image (PNG, artifact upload unavailable)"
                            )

                    # Auto-capture JPEG images
                    jpeg_data = getattr(res, "jpeg", None)
                    if jpeg_data:
                        img_bytes = base64.b64decode(jpeg_data)
                        url = self._upload_image_data(img_bytes, ".jpg")
                        if url:
                            artifact_urls.append(url)
                            output_parts.append(f"**Generated image:** {url}")
                        else:
                            output_parts.append(
                                "**Result:** Generated plot/image (JPEG, artifact upload unavailable)"
                            )

            # Download explicit output files from sandbox
            if output_files and self.artifact_store:
                for file_path in output_files:
                    file_result = await self._download_sandbox_file(file_path)
                    if file_result:
                        data, filename = file_result
                        suffix = self._get_suffix_from_filename(filename)
                        url = self.artifact_store.write_bytes(data, suffix)
                        if not url.startswith("Error:"):
                            artifact_urls.append(url)
                            output_parts.append(f"**Downloaded file ({filename}):** {url}")
                        else:
                            output_parts.append(f"**Error uploading {filename}:** {url}")
                    else:
                        output_parts.append(f"**Error:** Could not download {file_path}")
            elif output_files and not self.artifact_store:
                output_parts.append(
                    "**Warning:** output_files requested but artifact store not configured"
                )

            if not output_parts:
                output_parts.append("Code executed successfully with no output.")

            logger.info(
                f"Executed {language} code in E2B sandbox: {code[:512]}... -> {output_parts[:512]}"
            )

            return "\n\n".join(output_parts)

        except Exception as e:
            logger.error(f"E2B sandbox execution failed: {e}")
            # If sandbox connection is broken, reset it for next call
            if "sandbox" in str(e).lower() or "connection" in str(e).lower():
                self.sandbox = None
            return f"Error executing code: {e}"

    async def cleanup(self):
        """Clean up sandbox resources."""
        if self.sandbox:
            try:
                import asyncio

                sandbox_id = self.sandbox.sandbox_id
                await asyncio.to_thread(self.sandbox.kill)
                logger.info(f"Cleaned up E2B sandbox with ID: {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up E2B sandbox: {e}")
            finally:
                self.sandbox = None


class ProgressReportExecutor:
    """Executor that sends progress updates via a provided callback."""

    def __init__(
        self,
        send_callback: Callable[[str], Awaitable[None]] | None = None,
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
    """Executor for making/updating plans. Stores plan in quest DB when inside a quest."""

    def __init__(self, agent: Any = None, current_quest_id: str | None = None):
        self.agent = agent
        self.current_quest_id = current_quest_id

    async def execute(self, plan: str) -> str:
        """Store plan in quest DB (if in quest) and confirm receipt."""
        logger.info(f"Plan formulated: {plan[:500]}...")

        if self.agent and self.current_quest_id:
            await self.agent.chronicle.quest_set_plan(self.current_quest_id, plan)
            logger.debug(f"Stored plan for quest {self.current_quest_id}")

        return "OK, follow this plan (stored for future quest steps)"


class ArtifactStore:
    """Shared artifact storage for files and URLs."""

    def __init__(self, artifacts_path: str | None = None, artifacts_url: str | None = None):
        self.artifacts_path = Path(artifacts_path).expanduser() if artifacts_path else None
        self.artifacts_url = artifacts_url.rstrip("/") if artifacts_url else None

    @classmethod
    def from_config(cls, config: dict) -> ArtifactStore:
        """Create store from configuration."""
        artifacts_config = config.get("tools", {}).get("artifacts", {})
        return cls(
            artifacts_path=artifacts_config.get("path"),
            artifacts_url=artifacts_config.get("url"),
        )

    def _ensure_configured(self) -> str | None:
        """Check if store is configured, return error message if not."""
        if not self.artifacts_path or not self.artifacts_url:
            return "Error: artifacts.path and artifacts.url must be configured"
        return None

    def write_text(self, content: str, suffix: str = ".txt") -> str:
        """Write text content to artifact file, return URL."""
        if err := self._ensure_configured():
            return err

        assert self.artifacts_path is not None
        assert self.artifacts_url is not None

        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            return f"Error: Failed to create artifacts directory: {e}"

        file_id = generate_artifact_id()
        filepath = self.artifacts_path / f"{file_id}{suffix}"

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Created artifact file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact file: {e}")
            return f"Error: Failed to create artifact file: {e}"

        return f"{self.artifacts_url}/{file_id}{suffix}"

    def write_bytes(self, data: bytes | bytearray, suffix: str) -> str:
        """Write binary data to artifact file, return URL."""
        if err := self._ensure_configured():
            return err

        assert self.artifacts_path is not None
        assert self.artifacts_url is not None

        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            return f"Error: Failed to create artifacts directory: {e}"

        file_id = generate_artifact_id()
        filepath = self.artifacts_path / f"{file_id}{suffix}"

        try:
            filepath.write_bytes(data)
            logger.info(f"Created artifact file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact file: {e}")
            return f"Error: Failed to create artifact file: {e}"

        return f"{self.artifacts_url}/{file_id}{suffix}"


class ShareArtifactExecutor:
    """Executor for sharing artifacts via files and links."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    @classmethod
    def from_config(cls, config: dict) -> ShareArtifactExecutor:
        """Create executor from configuration."""
        return cls(ArtifactStore.from_config(config))

    async def execute(self, content: str) -> str:
        """Share an artifact by creating a file and providing a link."""
        url = self.store.write_text(content, ".txt")
        if url.startswith("Error:"):
            return url
        return f"Artifact shared: {url}"


class EditArtifactExecutor:
    """Executor for editing existing artifacts."""

    def __init__(self, store: ArtifactStore, webpage_visitor: WebpageVisitorExecutor):
        self.store = store
        self.webpage_visitor = webpage_visitor

    @classmethod
    def from_config(
        cls, config: dict, webpage_visitor: WebpageVisitorExecutor
    ) -> EditArtifactExecutor:
        """Create executor from configuration."""
        return cls(ArtifactStore.from_config(config), webpage_visitor)

    async def execute(self, artifact_url: str, old_string: str, new_string: str) -> str:
        """Edit an artifact by replacing old_string with new_string."""
        # Fetch via webpage visitor (handles local artifacts automatically)
        try:
            content_result = await self.webpage_visitor.execute(artifact_url)
            if isinstance(content_result, list):
                return "Error: Cannot edit binary artifacts (images)"

            # Extract content from markdown wrapper if present (remote URLs only)
            content = content_result
            if content.startswith("## Content from "):
                # Strip the markdown header
                parts = content.split("\n\n", 1)
                if len(parts) == 2:
                    content = parts[1]
        except Exception as e:
            logger.error(f"Failed to fetch artifact from {artifact_url}: {e}")
            return f"Error: Failed to fetch artifact: {e}"

        # Validate old_string exists and is unique
        if old_string not in content:
            return "Error: old_string not found in artifact. The artifact may have changed, or the search string is incorrect."

        occurrences = content.count(old_string)
        if occurrences > 1:
            return f"Error: old_string appears {occurrences} times in the artifact. It must be unique. Add more surrounding context to make it unique."

        # Perform the replacement
        new_content = content.replace(old_string, new_string, 1)

        # Extract suffix from URL (everything after last / and including .)
        url_filename = artifact_url.split("/")[-1]
        suffix = "." + url_filename.split(".", 1)[1] if "." in url_filename else ".txt"

        url = self.store.write_text(new_content, suffix)
        if url.startswith("Error:"):
            return url

        logger.info(f"Edited artifact: {artifact_url} -> {url}")
        return f"Artifact edited successfully. New version: {url}"


class ImageGenExecutor:
    """Executor for generating images via OpenRouter."""

    def __init__(
        self,
        router: Any,
        config: dict,
        max_image_size: int = 5 * 1024 * 1024,
        timeout: int = 30,
    ):
        from ..providers import parse_model_spec

        self.router = router
        self.config = config
        self.max_image_size = max_image_size
        self.timeout = timeout
        self.store = ArtifactStore.from_config(config)

        tools_config = config.get("tools", {}).get("image_gen", {})
        self.model = tools_config.get("model", "openrouter:google/gemini-2.5-flash-preview-image")

        spec = parse_model_spec(self.model)
        if spec.provider != "openrouter":
            raise ValueError(f"image_gen.model must use openrouter provider, got: {spec.provider}")

    @classmethod
    def from_config(cls, config: dict, router: Any) -> ImageGenExecutor:
        """Create executor from configuration."""
        return cls(router=router, config=config)

    async def execute(self, prompt: str, image_urls: list[str] | None = None) -> str | list[dict]:
        """Generate image(s) using OpenRouter and store as artifacts."""

        # Build message content with text and optional images
        content: str | list[dict]
        logger.info(f"Generating image with prompt: {prompt}")
        if image_urls:
            content_blocks: list[dict] = [{"type": "text", "text": prompt}]
            async with aiohttp.ClientSession() as session:
                for url in image_urls:
                    try:
                        ct, b64 = await fetch_image_b64(
                            session, url, self.max_image_size, self.timeout
                        )
                        content_blocks.append(
                            {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}
                        )
                        logger.info(f"Including additional image as input: {url}")
                    except ValueError as e:
                        logger.warning(f"Failed to fetch reference image {url}: {e}")
                        return f"Error: Failed to fetch reference image {url}: {e}"
            content = content_blocks
        else:
            content = prompt

        context = [{"role": "user", "content": content}]

        try:
            response, _, _ = await self.router.call_raw_with_model(
                model_str=self.model,
                context=context,
                system_prompt="",
                modalities=["image", "text"],
            )
        except Exception as e:
            logger.error(f"OpenRouter image generation failed: {e}")
            return f"Error: Image generation failed: {e}"

        if "error" in response:
            return f"Error: {response['error']}"

        # Extract images from response
        choices = response.get("choices", [])
        if not choices:
            logger.warning(f"No choices in response: {response}")
            return "Error: Model returned no output"

        message = choices[0].get("message", {})
        images = message.get("images", [])

        if not images:
            logger.warning(f"No images in message: {message}")
            return "Error: No images generated by model"

        artifact_urls = []
        image_blocks = []

        for img in images:
            img_url = None
            if isinstance(img, dict):
                img_url = img.get("image_url", {}).get("url") or img.get("url")
            elif isinstance(img, str):
                img_url = img

            if not img_url or not img_url.startswith("data:"):
                logger.warning(f"Invalid image data: {img}")
                continue

            # Parse data URL: data:image/png;base64,<data>
            try:
                parts = img_url.split(",", 1)
                if len(parts) != 2:
                    continue
                header, b64_data = parts
                mime_type = header.split(";")[0].replace("data:", "")
                img_bytes = base64.b64decode(b64_data)
            except Exception as e:
                logger.error(f"Failed to parse image data URL: {e}")
                continue

            ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
            suffix = ext_map.get(mime_type, ".png")

            url = self.store.write_bytes(img_bytes, suffix)
            if url.startswith("Error:"):
                return url

            # Add slop watermark using ImageMagick
            if self.store.artifacts_path:
                file_id = url.split("/")[-1].rsplit(".", 1)[0]
                filepath = self.store.artifacts_path / f"{file_id}{suffix}"
                try:
                    import subprocess

                    subprocess.run(
                        [
                            "convert",
                            str(filepath),
                            "-gravity",
                            "SouthEast",
                            "-pointsize",
                            "20",
                            "-fill",
                            "rgba(255,255,255,0.6)",
                            "-stroke",
                            "rgba(0,0,0,0.8)",
                            "-strokewidth",
                            "1",
                            "-annotate",
                            "+10+10",
                            "🍌slop",
                            str(filepath),
                        ],
                        check=True,
                        capture_output=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to add watermark to {filepath}: {e}")

            artifact_urls.append(url)

            # Add image block (reuse b64_data already parsed)
            image_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_data,
                    },
                }
            )

        if not artifact_urls:
            return "Error: No images could be extracted from response"

        # Return Anthropic content blocks: text (URLs) + images
        content_blocks = [
            {
                "type": "text",
                "text": "\n".join(f"Generated image: {url}" for url in artifact_urls),
            }
        ] + image_blocks

        return content_blocks


def create_tool_executors(
    config: dict | None = None,
    *,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
    agent: Any,
    arc: str,
    router: Any = None,
    current_quest_id: str | None = None,
) -> dict[str, Any]:
    """Create tool executors with configuration."""
    # Tool configs
    tools = config.get("tools", {}) if config else {}

    # E2B config
    e2b_config = tools.get("e2b", {})
    e2b_api_key = e2b_config.get("api_key")

    # Jina config
    jina_config = tools.get("jina", {})
    jina_api_key = jina_config.get("api_key")

    # Search provider config
    tools_config = config.get("tools", {}) if config else {}
    search_provider = tools_config.get("search_provider", "auto")

    # Create appropriate search executor based on provider
    if search_provider == "jina":
        search_executor = JinaSearchExecutor(api_key=jina_api_key)
    elif search_provider == "brave":
        brave_config = tools.get("brave", {})
        brave_api_key = brave_config.get("api_key")
        if not brave_api_key:
            logger.warning("Brave search configured but no API key found, falling back to ddgs")
            search_executor = WebSearchExecutor(backend="brave")
        else:
            search_executor = BraveSearchExecutor(api_key=brave_api_key)
    else:
        if "jina" in search_provider:
            raise ValueError(
                f"Jina search provider must be exclusive. Found: '{search_provider}'. "
                "Use exactly 'jina' for jina search (recommended provider, but API key required)."
            )
        search_executor = WebSearchExecutor(backend=search_provider)

    # Progress executor settings
    behavior = (config or {}).get("behavior", {})
    progress_cfg = behavior.get("progress", {}) if behavior else {}
    min_interval = int(progress_cfg.get("min_interval_seconds", 15))

    # Shared artifact store for code executor and share_artifact
    artifact_store = ArtifactStore.from_config(config or {})

    webpage_visitor = WebpageVisitorExecutor(
        progress_callback=progress_callback, api_key=jina_api_key, artifact_store=artifact_store
    )

    executors = {
        "web_search": search_executor,
        "visit_webpage": webpage_visitor,
        "execute_code": CodeExecutorE2B(api_key=e2b_api_key, artifact_store=artifact_store),
        "progress_report": ProgressReportExecutor(
            send_callback=progress_callback, min_interval_seconds=min_interval
        ),
        "final_answer": FinalAnswerExecutor(),
        "make_plan": MakePlanExecutor(agent=agent, current_quest_id=current_quest_id),
        "share_artifact": ShareArtifactExecutor(store=artifact_store),
        "edit_artifact": EditArtifactExecutor(
            store=artifact_store, webpage_visitor=webpage_visitor
        ),
        "chronicle_append": ChapterAppendExecutor(agent=agent, arc=arc),
        "chronicle_read": ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc),
    }

    # Add quest tools conditionally based on current_quest_id
    if current_quest_id is None:
        executors["quest_start"] = QuestStartExecutor(agent=agent, arc=arc)
    else:
        executors["subquest_start"] = SubquestStartExecutor(
            agent=agent, arc=arc, parent_quest_id=current_quest_id
        )

    # Add generate_image only if router is available
    if router:
        executors["generate_image"] = ImageGenExecutor.from_config(config or {}, router)

    return executors


async def execute_tool(
    tool_name: str, tool_executors: dict[str, Any], **kwargs
) -> str | list[dict]:
    """Execute a tool by name with given arguments."""
    executors = tool_executors

    if tool_name not in executors:
        raise ValueError(f"Unknown tool '{tool_name}'")

    executor = executors[tool_name]
    return await executor.execute(**kwargs)
