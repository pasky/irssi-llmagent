"""API agent with tool-calling capabilities."""

import copy
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..providers import ModelRouter
from .tools import TOOLS, create_tool_executors, execute_tool

logger = logging.getLogger(__name__)


class AgenticLLMActor:
    """API agent with tool-calling capabilities."""

    def __init__(
        self,
        config: dict[str, Any],
        model: str | list[str],
        system_prompt_generator: Callable[[], str],
        prompt_reminder_generator: Callable[[], str | None] = lambda: None,
        reasoning_effort: str = "low",
        *,
        allowed_tools: list[str] | None = None,
        additional_tools: list[dict[str, Any]] | None = None,
        additional_tool_executors: dict[str, Any] | None = None,
        agent: Any,
    ):
        self.config = config
        self.model = model
        self.system_prompt_generator = system_prompt_generator
        self.prompt_reminder_generator = prompt_reminder_generator
        self.reasoning_effort = reasoning_effort
        self.allowed_tools = allowed_tools
        self.additional_tools = additional_tools or []
        self.additional_tool_executors = additional_tool_executors or {}
        self.agent = agent
        self.model_router: ModelRouter | None = None

        # Actor configuration
        actor_cfg = self.config.get("actor", {})
        self.max_iterations = actor_cfg.get("max_iterations", 10)

        # Progress reporting config
        prog_cfg = actor_cfg.get("progress", {})
        self.progress_threshold_seconds = int(prog_cfg.get("threshold_seconds", 30))
        self.progress_min_interval_seconds = int(prog_cfg.get("min_interval_seconds", 15))

        # Tool executors will be created in run_agent with arc parameter

    async def __aenter__(self):
        """Async context manager entry."""
        # Lazy-init router when entering
        self.model_router = await ModelRouter(self.config).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Close model router (which closes its clients)
        if self.model_router is not None:
            await self.model_router.__aexit__(exc_type, exc_val, exc_tb)
            self.model_router = None

    async def run_agent(
        self,
        context: list[dict],
        *,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        arc: str,
    ) -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = copy.deepcopy(context)

        # Create tool executors with the provided arc
        base_executors = create_tool_executors(
            self.config, progress_callback=progress_callback, agent=self.agent, arc=arc
        )
        tool_executors = {**base_executors, **self.additional_tool_executors}

        # Initialize progress tracking
        progress_start_time = None
        if progress_callback is not None:
            from time import time as _now

            progress_start_time = _now()

        # Tool execution loop
        for iteration in range(self.max_iterations * 2):
            if iteration > 0:
                logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")
            if iteration >= self.max_iterations:
                logger.warn("Exceeding max iterations...")

            # Select model per iteration; last element repeats thereafter for lists
            if isinstance(self.model, list):
                model = self.model[iteration] if iteration < len(self.model) else self.model[-1]
            else:
                model = self.model

            extra_messages = []

            # Add prompt reminder if configured
            prompt_reminder = self.prompt_reminder_generator()
            if prompt_reminder:
                extra_messages += [{"role": "user", "content": f"<meta>{prompt_reminder}</meta>"}]

            if progress_callback is not None and progress_start_time is not None:
                from time import time as _now

                # Read last progress time from executor if available
                last = progress_start_time
                try:
                    prog_exec = tool_executors.get("progress_report")
                    if prog_exec and getattr(prog_exec, "_last_sent", None):
                        last = max(last, float(prog_exec._last_sent))
                except Exception:
                    pass
                elapsed = _now() - last
                logger.debug(
                    f"Last: {last} (start: {progress_start_time}), elapsed {elapsed}, vs. {self.progress_threshold_seconds}"
                )
                if (
                    iteration < self.max_iterations - 2
                    and elapsed >= self.progress_threshold_seconds
                ) or (iteration == 0 and self.reasoning_effort in ("medium", "high")):
                    extra_messages = [
                        {
                            "role": "user",
                            "content": "<meta>If you are going to call more tools, you MUST ALSO use the progress_report tool now.</meta>",
                        }
                    ]

            try:
                if self.model_router is None:
                    self.model_router = await ModelRouter(self.config).__aenter__()
                available_tools = TOOLS + self.additional_tools
                tool_choice = None

                if iteration == 0:
                    # On first turn (iteration 0), only allow make_plan tool if
                    # models will switch, since we are likely switching from a
                    # good thinking model with bad tool calls to a bad thinking
                    # model with good tool calls
                    if (
                        isinstance(self.model, list)
                        and len(self.model) >= 2
                        and self.model[0] != self.model[1]
                    ):
                        tool_choice = ["make_plan", "final_answer"]

                elif iteration >= self.max_iterations - 1:
                    tool_choice = ["final_answer"]

                # Clean up any remaining placeholders in tool descriptions
                if self.allowed_tools is not None:
                    available_tools = [
                        tool for tool in available_tools if tool["name"] in self.allowed_tools
                    ]
                    if tool_choice:
                        tool_choice = [tool for tool in tool_choice if tool in self.allowed_tools]

                system_prompt = self.system_prompt_generator()
                response, client, _ = await self.model_router.call_raw_with_model(
                    model,
                    messages + extra_messages,
                    system_prompt,
                    tools=available_tools,
                    tool_choice=tool_choice,
                    reasoning_effort=self.reasoning_effort,
                )

                # Process response using unified handler (provider-aware)
                result = self._process_ai_response_provider(response, client)

                if result["type"] == "error":
                    logger.error(f"Invalid AI response: {result['message']}")
                    return f"Error: {result['message']}"
                elif result["type"] == "final_text":
                    return result["text"]
                elif result["type"] == "truncated_tool_retry":
                    # Add assistant's truncated tool request to conversation
                    if response and isinstance(response, dict):
                        messages.append(client.format_assistant_message(response))

                    # Add a tool result indicating the truncation
                    tool_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": result["tool_id"],
                            "content": f"Error: Your {result['tool_name']} tool call failed because it was truncated. Please retry with a shorter response or split your call into smaller sequential parts.",
                        }
                    ]

                    # Add tool results to conversation using API-specific format
                    results_msg = client.format_tool_results(tool_results)
                    if isinstance(results_msg, list):
                        messages.extend(results_msg)
                    else:
                        messages.append(results_msg)

                    logger.warning(
                        f"Tool call {result['tool_name']} was truncated, asking AI to retry"
                    )
                    continue  # Continue to next iteration to let AI retry
                elif result["type"] == "tool_use":
                    # Add assistant's tool request to conversation
                    if response and isinstance(response, dict):
                        messages.append(client.format_assistant_message(response))

                    # Execute all tools and collect results
                    tool_results = []
                    for tool in result["tools"]:
                        try:
                            tool_result = await execute_tool(
                                tool["name"], tool_executors, **tool["input"]
                            )
                            logger.info(f"Tool {tool['name']} executed: {tool_result[:100]}...")

                            # If this is the final_answer tool, return its result directly
                            if tool["name"] == "final_answer":
                                if len(result["tools"]) > 1:
                                    logger.warning(
                                        "Rejecting final answer {tool_result}, since multiple tool calls were seen."
                                    )
                                else:
                                    return client.cleanup_raw_text(tool_result)

                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            tool_result = repr(e)
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

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Agent iteration {iteration + 1} failed: {str(e)}")
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

        # Handle truncated responses due to max_tokens (Claude-specific)
        if response.get("stop_reason") == "max_tokens" and "content" in response:
            # Check if there's a partial tool call that got truncated
            content = response.get("content", [])
            for block in content:
                if block.get("type") == "tool_use" and not block.get("input"):
                    # Found truncated tool call - return special result to signal retry needed
                    return {
                        "type": "truncated_tool_retry",
                        "tool_id": block["id"],
                        "tool_name": block["name"],
                    }
            # If no partial tool call found, just treat as text truncation
            text_response = client.extract_text_from_response(response)
            if text_response:
                return {
                    "type": "final_text",
                    "text": text_response + " [Response truncated due to max_tokens]",
                }
            else:
                return {
                    "type": "error",
                    "message": "Response truncated due to max_tokens and no recoverable content found",
                }

        # Extract final text response using API client's logic
        text_response = client.extract_text_from_response(response)
        if text_response and text_response != "...":
            return {"type": "final_text", "text": text_response}

        logger.debug(response)
        return {"type": "error", "message": "No valid text or tool use found in response"}
