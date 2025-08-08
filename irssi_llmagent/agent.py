"""API agent with tool-calling capabilities."""

import copy
import logging
from datetime import datetime
from typing import Any

from .claude import AnthropicClient
from .openai import OpenAIClient
from .tools import PROGRESS_TOOL, TOOLS, create_tool_executors, execute_tool

logger = logging.getLogger(__name__)


class AIAgent:
    """API agent with web search and webpage visiting capabilities."""

    def __init__(
        self,
        config: dict[str, Any],
        mynick: str,
        extra_prompt: str = "",
        model_override: str | None = None,
        *,
        progress_enabled: bool = False,
        progress_callback: Any | None = None,
    ):
        self.config = config
        self.mynick = mynick
        self.api_client: Any = self._get_api_client(config)
        self.max_iterations = 7
        self.extra_prompt = extra_prompt
        self.model_override = model_override
        self.system_prompt = self._get_system_prompt()
        # Progress reporting
        behavior = self.config.get("behavior", {})
        prog_cfg = behavior.get("progress", {}) if behavior else {}
        self.progress_threshold_seconds = int(prog_cfg.get("threshold_seconds", 30))
        self._progress_start_time: float | None = None
        self._progress_can_send: bool = bool(progress_callback)
        # Tool executors with progress callback
        self.tool_executors = create_tool_executors(config, progress_callback=progress_callback)

    def _get_api_client(self, config):
        """Get the appropriate API client based on config."""
        api_type = config.get("api_type", "anthropic")
        if api_type == "openai":
            return OpenAIClient(config)
        elif api_type == "anthropic":
            return AnthropicClient(config)
        else:
            logger.error(f"Unknown api_type: {api_type}, defaulting to anthropic")
            return AnthropicClient(config)

    def _get_api_config_section(self):
        """Get the appropriate API config section."""
        api_type = self.config.get("api_type", "anthropic")
        return self.config[api_type]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent based on serious mode prompt."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_prompt = self.config["prompts"]["serious"].format(
            mynick=self.mynick, current_time=current_time
        )
        return base_prompt + self.extra_prompt

    async def __aenter__(self):
        """Async context manager entry."""
        await self.api_client.__aenter__()
        # Initialize progress timers
        from time import time as _now

        self._progress_start_time = _now()
        return self

    def configure_progress(self, enabled: bool, callback: Any | None) -> None:
        """Configure progress reporting at runtime (used by main)."""
        self._progress_can_send = bool(enabled and callback is not None)
        # Recreate executors with the provided callback
        self.tool_executors = create_tool_executors(self.config, progress_callback=callback)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.api_client.__aexit__(exc_type, exc_val, exc_tb)

    async def run_agent(self, context: list[dict]) -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = copy.deepcopy(context)

        # Tool execution loop
        for iteration in range(self.max_iterations):
            if iteration > 0:
                logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # Use serious_model for first iteration only, then default model
            # Override with specific model if provided
            if self.model_override:
                model = self.model_override
            else:
                api_config = self._get_api_config_section()
                model = api_config["serious_model"] if iteration == 0 else api_config["model"]

            # Don't pass tools on final iteration
            extra_prompt = (
                " THIS WAS YOUR LAST TOOL TURN, YOU MUST NOT CALL ANY FURTHER TOOLS OR FUNCTIONS !!!"
                if iteration >= self.max_iterations - 3
                else ""
            )

            # Add thinking encouragement for first iteration only
            thinking_prompt = (
                " First, think in <thinking> tags: review your knowledge and decide whether a search must be done. If so, plan your research; if not, think through your reply."
                if iteration == 0
                else ""
            )

            # Conditional progress nudge appended only when threshold elapsed
            progress_prompt = ""
            if self._progress_can_send and self._progress_start_time is not None:
                from time import time as _now

                # Read last progress time from executor if available
                last = self._progress_start_time
                try:
                    prog_exec = self.tool_executors.get("progress_report")
                    if prog_exec and getattr(prog_exec, "_last_sent", None):
                        last = max(last, float(prog_exec._last_sent))
                except Exception:
                    pass
                elapsed = _now() - last
                logger.debug(
                    f"Last: {last} (start: {self._progress_start_time}), elapsed {elapsed}, vs. {self.progress_threshold_seconds}"
                )
                if elapsed >= self.progress_threshold_seconds:
                    progress_prompt = " If you are going to call more tools, you MUST also use the progress_report tool now!"

            # Select which tools to expose (include progress tool only when enabled)
            tools_for_model = TOOLS + ([PROGRESS_TOOL] if self._progress_can_send else [])

            try:
                # Call API with or without tools based on iteration
                response = await self.api_client.call_raw(
                    messages,  # Pass messages in proper API format
                    self.system_prompt + thinking_prompt + progress_prompt + extra_prompt,
                    model,
                    tools=tools_for_model,
                )

                # Process response using unified handler
                result = self._process_ai_response(response)

                if result["type"] == "error":
                    logger.error(f"Invalid AI response: {result['message']}")
                    break
                elif result["type"] == "final_text":
                    return result["text"]
                elif result["type"] == "tool_use":
                    # Add assistant's tool request to conversation
                    if response and isinstance(response, dict):
                        messages.append(self.api_client.format_assistant_message(response))

                    # Execute all tools and collect results
                    tool_results = []
                    for tool in result["tools"]:
                        try:
                            tool_result = await execute_tool(
                                tool["name"], self.tool_executors, **tool["input"]
                            )
                            logger.info(f"Tool {tool['name']} executed: {tool_result[:100]}...")
                        except Exception as e:
                            tool_result = str(e)
                            logger.warning(f"Tool {tool['name']} failed: {e}")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool["id"],
                                "content": tool_result,
                            }
                        )

                    # Add tool results to conversation using API-specific format
                    results_msg = self.api_client.format_tool_results(tool_results)
                    if isinstance(results_msg, list):
                        messages.extend(results_msg)
                    else:
                        messages.append(results_msg)
                    continue

            except Exception as e:
                logger.error(f"Agent iteration {iteration + 1} failed: {e}")
                break

        raise StopIteration("Agent took too many turns to research")

    def _process_ai_response(self, response: dict | str | None) -> dict[str, Any]:
        """Unified processing of AI responses, returning structured result."""
        if not response:
            return {"type": "error", "message": "Empty response from AI"}

        # Handle case where AI client returns a string (shouldn't happen with raw_response=True)
        if isinstance(response, str):
            return {"type": "final_text", "text": response}

        if not isinstance(response, dict):
            return {"type": "error", "message": f"Unexpected response type: {type(response)}"}

        # Check if API wants to use tools
        if self.api_client.has_tool_calls(response):
            tool_uses = self.api_client.extract_tool_calls(response)
            if not tool_uses:
                return {"type": "error", "message": "Invalid tool use response"}
            return {"type": "tool_use", "tools": tool_uses}

        # Extract final text response using API client's logic
        text_response = self.api_client.extract_text_from_response(response)
        if text_response and text_response != "...":
            return {"type": "final_text", "text": text_response}

        return {"type": "error", "message": "No valid text or tool use found in response"}
