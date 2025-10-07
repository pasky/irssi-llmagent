"""Perplexity AI API client implementation."""

import json
import logging
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Perplexity AI API client with async support."""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("providers", {}).get("perplexity", {})

    async def call_perplexity(self, context: list[dict[str, str]]) -> str | None:
        """Call Perplexity API with context."""

        if not context:
            return None

        # Use only last message with one-line encouragement
        last_content = context[-1]["content"]
        # Strip IRC format and command prefix
        last_content = re.sub(r"^<[^>]+>\s*[^:]+:\s*!p\s*", "", last_content)
        query = last_content + " <REPLY IN ONE LINE>"

        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 256,
            "temperature": 0.7,
        }

        logger.debug(f"Calling Perplexity API with model: {self.config['model']}")
        logger.debug(f"Perplexity request payload: {json.dumps(payload, indent=2)}")

        try:
            async with (
                aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.config['key']}",
                        "Content-Type": "application/json",
                        "User-Agent": "irssi-llmagent/1.0",
                    }
                ) as session,
                session.post(self.config["url"], json=payload) as response,
            ):
                response.raise_for_status()
                data = await response.json()

            logger.debug(f"Perplexity response: {json.dumps(data, indent=2)}")

            if "choices" in data and data["choices"]:
                text = data["choices"][0]["message"]["content"]
                logger.debug(f"Perplexity response text: {text}")
                # Clean up response
                text = text.strip()
                text = re.sub(r"\n", "  ", text)
                text = re.sub(r"   *", "  ", text)
                text = re.sub(r"^<[^>]+>\s*", "", text)  # Remove IRC nick prefix

                result = text

                # Handle citations if present
                if "citations" in data and data["citations"]:
                    citations = "  ".join(data["citations"])
                    logger.debug(f"Perplexity citations: {citations}")
                    return f"{result}\n{citations}"  # Return with newline to send as two messages

                logger.debug(f"Cleaned Perplexity response: {result}")
                return result

        except aiohttp.ClientError as e:
            logger.error(f"Perplexity API error: {e}")
            return f"API error: {e}"

        return None
