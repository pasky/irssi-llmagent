"""Claude agent with tool-calling capabilities."""

import logging
from typing import Any

from .claude import AnthropicClient
from .tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)


class ClaudeAgent:
    """Claude agent with web search and webpage visiting capabilities."""

    def __init__(self, config: dict[str, Any], mynick: str):
        self.config = config
        self.mynick = mynick
        self.claude_client = AnthropicClient(config)
        self.max_iterations = 3
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent based on serious mode prompt."""
        return self.config["prompts"]["serious"].format(mynick=self.mynick)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.claude_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.claude_client.__aexit__(exc_type, exc_val, exc_tb)

    async def run_agent(self, context: list[dict]) -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = context.copy()

        # Tool execution loop
        for iteration in range(self.max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # Don't pass tools on final iteration
            tools = TOOLS if iteration < self.max_iterations - 1 else None

            try:
                # Call Claude with or without tools based on iteration
                response = await self.claude_client.call_claude_raw(
                    messages,  # Pass messages in proper API format
                    self.system_prompt,
                    self.config["anthropic"]["serious_model"],
                    tools=tools
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
                    if response and isinstance(response, dict):
                        messages.append({"role": "assistant", "content": response["content"]})

                    # Execute all tools and collect results
                    tool_results = []
                    for tool in result["tools"]:
                        tool_result = await execute_tool(tool["name"], **tool["input"])
                        logger.info(f"Tool {tool['name']} executed: {tool_result[:100]}...")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool["id"],
                            "content": tool_result,
                        })

                    # Add all tool results to conversation in a single message
                    messages.append({
                        "role": "user",
                        "content": tool_results,
                    })
                    continue

            except Exception as e:
                logger.error(f"Agent iteration {iteration + 1} failed: {e}")
                break

        return "Sorry, I couldn't complete your request."

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
        text_response = self.claude_client._extract_text_from_response(response)
        if text_response:
            return {"type": "final_text", "text": text_response}

        return {"type": "error", "message": "No valid text or tool use found in response"}



    def _extract_tool_uses(self, response: dict) -> list[dict] | None:
        """Extract all tool use information from Claude's response."""
        content = response.get("content", [])
        tool_uses = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_uses.append({"id": block["id"], "name": block["name"], "input": block["input"]})
        return tool_uses if tool_uses else None
