"""API agent with tool-calling capabilities."""

import copy
import logging
from typing import Any

from .base_client import build_system_prompt
from .model_router import ModelRouter, parse_model_spec
from .tools import TOOLS, create_tool_executors, execute_tool

logger = logging.getLogger(__name__)


class AIAgent:
    """API agent with web search and webpage visiting capabilities."""

    def __init__(
        self,
        config: dict[str, Any],
        mynick: str,
        mode: str = "serious",
        extra_prompt: str = "",
        model_override: str | None = None,
        *,
        progress_enabled: bool = False,
        progress_callback: Any | None = None,
        allowed_tools: list[str] | None = None,
    ):
        self.config = config
        self.mynick = mynick
        self.mode = mode
        self.model_router: ModelRouter | None = None
        self.max_iterations = 10
        self.extra_prompt = extra_prompt
        self.model_override = model_override
        # System prompt is now generated dynamically per model call
        # Progress reporting
        prog_cfg = self.config.get("agent", {}).get("progress", {})
        self.progress_threshold_seconds = int(prog_cfg.get("threshold_seconds", 30))
        self._progress_start_time: float | None = None
        self._progress_can_send: bool = bool(progress_callback)
        # Tool executors with progress callback
        self.tool_executors = create_tool_executors(config, progress_callback=progress_callback)
        self.allowed_tools = allowed_tools

    def _get_system_prompt(self, current_model: str = "") -> str:
        """Get the system prompt for the agent based on mode."""
        base_prompt = build_system_prompt(self.config, self.mode, self.mynick)
        return base_prompt + self.extra_prompt

    async def __aenter__(self):
        """Async context manager entry."""
        # Lazy-init router when entering
        self.model_router = await ModelRouter(self.config).__aenter__()
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
        # Close model router (which closes its clients)
        if self.model_router is not None:
            await self.model_router.__aexit__(exc_type, exc_val, exc_tb)
            self.model_router = None

    async def run_agent(self, context: list[dict], reasoning_effort: str = "low") -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = copy.deepcopy(context)

        # Tool execution loop
        cross_provider_next: bool = False
        for iteration in range(self.max_iterations):
            if iteration > 0:
                logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")

            # Select serious model per iteration; last element repeats thereafter
            if self.model_override:
                model = self.model_override
                next_model = self.model_override
            else:
                serious_cfg = self.config["command"]["models"]["serious"]
                if isinstance(serious_cfg, list):
                    model = (
                        serious_cfg[iteration] if iteration < len(serious_cfg) else serious_cfg[-1]
                    )
                    next_model = (
                        serious_cfg[iteration + 1]
                        if (iteration + 1) < len(serious_cfg)
                        else serious_cfg[-1]
                    )
                else:
                    model = serious_cfg
                    next_model = serious_cfg
            # Determine if next iteration switches providers
            try:
                cur_provider = parse_model_spec(model).provider
                nxt_provider = parse_model_spec(next_model).provider
                cross_provider_next = cur_provider != nxt_provider
            except Exception:
                cross_provider_next = False

            # Conditional progress nudge appended only when threshold elapsed
            extra_messages = []
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
                if (
                    iteration < self.max_iterations - 2
                    and elapsed >= self.progress_threshold_seconds
                ) or (iteration == 0 and reasoning_effort in ("medium", "high")):
                    extra_messages = [
                        {
                            "role": "user",
                            "content": "<meta>If you are going to call more tools, you MUST ALSO use the progress_report tool now.</meta>",
                        }
                    ]

            try:
                if self.model_router is None:
                    self.model_router = await ModelRouter(self.config).__aenter__()
                # Filter tools if allowed_tools is specified
                available_tools = TOOLS
                if self.allowed_tools is not None:
                    available_tools = [tool for tool in TOOLS if tool["name"] in self.allowed_tools]

                system_prompt = self._get_system_prompt(model)
                response, client, _ = await self.model_router.call_raw_with_model(
                    model,
                    messages + extra_messages,
                    system_prompt,
                    tools=available_tools,
                    tool_choice="auto" if iteration < self.max_iterations - 2 else "none",
                    reasoning_effort=reasoning_effort,
                )

                # Process response using unified handler (provider-aware)
                result = self._process_ai_response_provider(response, client)

                if result["type"] == "error":
                    logger.error(f"Invalid AI response: {result['message']}")
                    break
                elif result["type"] == "final_text":
                    return result["text"]
                elif result["type"] == "tool_use":
                    # Add assistant's tool request to conversation
                    if response and isinstance(response, dict):
                        messages.append(client.format_assistant_message(response))

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
                    results_msg = client.format_tool_results(tool_results)
                    if isinstance(results_msg, list):
                        messages.extend(results_msg)
                    else:
                        messages.append(results_msg)
                    # Also add provider-agnostic summary only when provider switches next
                    if cross_provider_next:
                        try:
                            summarized = "; ".join(
                                [str(r.get("content", "")) for r in tool_results]
                            )[:800]
                            if summarized:
                                logger.warning(
                                    "Cross-provider handoff detected; injecting TOOL RESULTS summary for interoperability"
                                )
                                messages.append(
                                    {"role": "user", "content": f"TOOL RESULTS: {summarized}"}
                                )
                        except Exception:
                            pass
                    continue

            except Exception as e:
                logger.error(f"Agent iteration {iteration + 1} failed: {e}")
                break

        raise StopIteration("Agent took too many turns to research")

    def _process_ai_response_provider(
        self, response: dict | str | None, client: Any
    ) -> dict[str, Any]:
        """Unified processing of AI responses, using the given provider client."""
        if not response:
            return {"type": "error", "message": "Empty response from AI"}

        # Handle case where AI client returns a string (shouldn't happen with raw_response=True)
        if isinstance(response, str):
            return {"type": "final_text", "text": response}

        if not isinstance(response, dict):
            return {"type": "error", "message": f"Unexpected response type: {type(response)}"}

        # Check if API wants to use tools
        if client.has_tool_calls(response):
            tool_uses = client.extract_tool_calls(response)
            if not tool_uses:
                return {"type": "error", "message": "Invalid tool use response"}
            return {"type": "tool_use", "tools": tool_uses}

        # Extract final text response using API client's logic
        text_response = client.extract_text_from_response(response)
        if text_response and text_response != "...":
            return {"type": "final_text", "text": text_response}

        return {"type": "error", "message": "No valid text or tool use found in response"}
