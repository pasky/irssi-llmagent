"""Tool definitions and executors for AI agent."""

import asyncio
import base64
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

import aiohttp
from ddgs import DDGS

from ..chronicler.tools import ChapterAppendExecutor, ChapterRenderExecutor, chronicle_tools_defs

logger = logging.getLogger(__name__)


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
        "persist": "summary",
    },
    {
        "name": "execute_python",
        "description": "Execute Python code in a sandbox environment and return the output. The sandbox environment is persisted to follow-up calls of this tool within this thread.",
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
        "persist": "none",
    },
    {
        "name": "share_artifact",
        "description": "Share an artifact (additional command output - created script, detailed report, supporting data) with the user. The content is made available on a public link that is returned by the tool. Use this only for additional content that doesn't fit into your standard IRC message response (or when explicitly requested).",
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
        "name": "generate_image",
        "description": "Generate image(s) using {tools.image_gen.model} on OpenRouter. Optionally provide reference image URLs for editing/variations.",
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

# Add chronicle tools to the main tools list
TOOLS.extend(chronicle_tools_defs())  # type: ignore


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
    ):
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_image_size = max_image_size  # 5MB default limit post base64 encode
        self.progress_callback = progress_callback
        self.api_key = api_key

    async def execute(self, url: str) -> str | list[dict]:
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


class PythonExecutorE2B:
    """Python code executor using E2B sandbox."""

    def __init__(self, api_key: str | None = None, timeout: int = 180):
        self.api_key = api_key
        self.timeout = timeout
        self.sandbox = None

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
                from typing import Any

                sandbox_args: dict[str, Any] = {"timeout": self.timeout}
                if self.api_key:
                    sandbox_args["api_key"] = self.api_key
                sandbox = Sandbox(**sandbox_args)
                return sandbox

            self.sandbox = await asyncio.to_thread(create_sandbox)
            logger.info(f"Created new E2B sandbox with ID: {self.sandbox.sandbox_id}")

    async def execute(self, code: str) -> str:
        """Execute Python code in E2B sandbox and return output."""
        try:
            await self._ensure_sandbox()
        except ImportError as e:
            return str(e)

        try:
            import asyncio

            assert self.sandbox is not None
            result = await asyncio.to_thread(self.sandbox.run_code, code)
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
            await self.send_callback(clean, "progress")
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


class ArtifactStore:
    """Shared artifact storage for files and URLs."""

    def __init__(self, artifacts_path: str | None = None, artifacts_url: str | None = None):
        self.artifacts_path = Path(artifacts_path).expanduser() if artifacts_path else None
        self.artifacts_url = artifacts_url.rstrip("/") if artifacts_url else None

    @classmethod
    def from_config(cls, config: dict) -> "ArtifactStore":
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

        file_id = uuid.uuid4().hex
        filepath = self.artifacts_path / f"{file_id}{suffix}"

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Created artifact file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact file: {e}")
            return f"Error: Failed to create artifact file: {e}"

        return f"{self.artifacts_url}/{file_id}{suffix}"

    def write_bytes(self, data: bytes, suffix: str) -> str:
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

        file_id = uuid.uuid4().hex
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
    def from_config(cls, config: dict) -> "ShareArtifactExecutor":
        """Create executor from configuration."""
        return cls(ArtifactStore.from_config(config))

    async def execute(self, content: str) -> str:
        """Share an artifact by creating a file and providing a link."""
        url = self.store.write_text(content, ".txt")
        if url.startswith("Error:"):
            return url
        return f"Artifact shared: {url}"


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
    def from_config(cls, config: dict, router: Any) -> "ImageGenExecutor":
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
    progress_callback: Any | None = None,
    agent: Any,
    arc: str,
    router: Any = None,
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

    executors = {
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
        "share_artifact": ShareArtifactExecutor.from_config(config or {}),
        "chronicle_append": ChapterAppendExecutor(agent=agent, arc=arc),
        "chronicle_read": ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc),
    }

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
