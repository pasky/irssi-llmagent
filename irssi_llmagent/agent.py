"""Claude agent with tool-calling capabilities."""

import copy
import logging
from datetime import datetime
from typing import Any

from .claude import AnthropicClient
from .tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)


class ClaudeAgent:
    """Claude agent with web search and webpage visiting capabilities."""

    def __init__(
        self,
        config: dict[str, Any],
        mynick: str,
        extra_prompt: str = "",
        model_override: str | None = None,
    ):
        self.config = config
        self.mynick = mynick
        self.claude_client = AnthropicClient(config)
        self.max_iterations = 5
        self.extra_prompt = extra_prompt
        self.model_override = model_override
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent based on serious mode prompt."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_prompt = self.config["prompts"]["serious"].format(
            mynick=self.mynick, current_time=current_time
        )
        return base_prompt + self.extra_prompt

    async def __aenter__(self):
        """Async context manager entry."""
        await self.claude_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def run_agent(self, context: list[dict]) -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = copy.deepcopy(context)

        # Tool execution loop
        for iteration in range(self.max_iterations):
            logger.debug(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # Use serious_model for first iteration only, then default model
            # Override with specific model if provided
            if self.model_override:
                model = self.model_override
            else:
                model = (
                    self.config["anthropic"]["serious_model"]
                    if iteration == 0
                    else self.config["anthropic"]["model"]
                )

            # Don't pass tools on final iteration
            extra_prompt = (
                " THIS WAS YOUR LAST TOOL TURN, YOU MUST NOT USE ANY FURTHER TOOLS"
                if iteration >= self.max_iterations - 2
                else ""
            )

            # Add thinking encouragement for first iteration only
            thinking_prompt = (
                " First, think in <thinking> tags: review your knowledge and decide whether a search must be done. If so, plan your research; if not, think through your reply."
                if iteration == 0
                else ""
            )

            try:
                # Call Claude with or without tools based on iteration
                response = await self.claude_client.call_claude_raw(
                    messages,  # Pass messages in proper API format
                    self.system_prompt + thinking_prompt + extra_prompt,
                    model,
                    tools=TOOLS,
                )

                # Process response using unified handler
                result = self._process_claude_response(response)

                if result["type"] == "error":
                    logger.error(f"Invalid Claude response: {result['message']}")
                    break
                elif result["type"] == "final_text":
                    return result["text"]
                elif result["type"] == "tool_use":
                    # Add Claude's tool request to conversation
                    if response and isinstance(response, dict) and "content" in response:
                        content = response["content"]
                        if isinstance(content, list):  # Claude returns list of content blocks
                            messages.append({"role": "assistant", "content": content})

                    # Execute all tools and collect results
                    tool_results = []
                    for tool in result["tools"]:
                        try:
                            tool_result = await execute_tool(tool["name"], **tool["input"])
                        except Exception as e:
                            tool_result = str(e)
                        logger.debug(f"Tool {tool['name']} executed: {tool_result[:100]}...")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool["id"],
                                "content": tool_result,
                            }
                        )

                    # Add all tool results to conversation in a single message
                    messages.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )
                    continue

            except Exception as e:
                logger.error(f"Agent iteration {iteration + 1} failed: {e}")
                break

        raise StopIteration("Agent took too many turns to research")

    def _process_claude_response(self, response: dict | str | None) -> dict[str, Any]:
        """Unified processing of Claude responses, returning structured result."""
        if not response:
            return {"type": "error", "message": "Empty response from Claude"}

        # Handle case where call_claude returns a string (shouldn't happen with raw_response=True)
        if isinstance(response, str):
            return {"type": "final_text", "text": response}

        if not isinstance(response, dict):
            return {"type": "error", "message": f"Unexpected response type: {type(response)}"}

        # Check if Claude wants to use tools
        if response.get("stop_reason") == "tool_use":
            tool_uses = self._extract_tool_uses(response)
            if not tool_uses:
                return {"type": "error", "message": "Invalid tool use response"}
            return {"type": "tool_use", "tools": tool_uses}

        # Extract final text response using claude.py's logic
        text_response = self.claude_client.extract_text_from_response(response)
        if text_response and text_response != "...":
            return {"type": "final_text", "text": text_response}

        return {"type": "error", "message": "No valid text or tool use found in response"}

    def _extract_tool_uses(self, response: dict) -> list[dict] | None:
        """Extract all tool use information from Claude's response."""
        content = response.get("content", [])
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append(
                    {"id": block["id"], "name": block["name"], "input": block["input"]}
                )
        return tool_uses if tool_uses else None
