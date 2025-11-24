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
        prepended_context: list[dict[str, str]] | None = None,
        agent: Any,
        vision_model: str | None = None,
    ):
        self.config = config
        self.model = model
        self.system_prompt_generator = system_prompt_generator
        self.prompt_reminder_generator = prompt_reminder_generator
        self.reasoning_effort = reasoning_effort
        self.allowed_tools = allowed_tools
        self.additional_tools = additional_tools or []
        self.additional_tool_executors = additional_tool_executors or {}
        self.prepended_context = prepended_context or []
        self.agent = agent
        self.model_router = ModelRouter(self.config)
        self.vision_model = vision_model

        # Actor configuration
        actor_cfg = self.config.get("actor", {})
        self.max_iterations = actor_cfg.get("max_iterations", 10)

        # Progress reporting config
        prog_cfg = actor_cfg.get("progress", {})
        self.progress_threshold_seconds = int(prog_cfg.get("threshold_seconds", 30))
        self.progress_min_interval_seconds = int(prog_cfg.get("min_interval_seconds", 15))

        # Tool executors will be created in run_agent with arc parameter

    async def run_agent(
        self,
        context: list[dict],
        *,
        progress_callback: Callable[[str, str], Awaitable[None]] | None = None,
        arc: str,
    ) -> str:
        """Run the agent with tool-calling loop."""
        messages: list[dict[str, Any]] = copy.deepcopy(self.prepended_context) + copy.deepcopy(
            context
        )

        # Create tool executors with the provided arc
        base_executors = create_tool_executors(
            self.config,
            progress_callback=progress_callback,
            agent=self.agent,
            arc=arc,
            router=self.model_router,
        )
        tool_executors = {**base_executors, **self.additional_tool_executors}

        # Initialize progress tracking
        progress_start_time = None
        if progress_callback is not None:
            from time import time as _now

            progress_start_time = _now()

        # Tool execution loop
        vision_switched = False
        result_suffix = ""

        # Track tool calls that need persistence
        persistent_tool_calls = []

        # Track empty response retries
        empty_response_retries = 0

        try:
            for iteration in range(self.max_iterations * 2):
                if iteration > 0:
                    logger.info(f"Agent iteration {iteration + 1}/{self.max_iterations}")
                if iteration >= self.max_iterations:
                    logger.warn("Exceeding max iterations...")

                # Select model per iteration; last element repeats thereafter for lists
                if vision_switched:
                    assert self.vision_model
                    model = self.vision_model
                elif isinstance(self.model, list):
                    model = self.model[iteration] if iteration < len(self.model) else self.model[-1]
                else:
                    model = self.model

                extra_messages = []

                # Add prompt reminder if configured
                prompt_reminder = self.prompt_reminder_generator()
                if prompt_reminder:
                    extra_messages += [
                        {"role": "user", "content": f"<meta>{prompt_reminder}</meta>"}
                    ]

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

                # Fill in {config.path} placeholders in tool descriptions
                import re

                available_tools = []
                for tool in TOOLS + self.additional_tools:
                    tool_copy = copy.deepcopy(tool)
                    desc = tool_copy.get("description", "")

                    def replace_config(match):
                        path = match.group(1)
                        keys = path.split(".")
                        val = self.config
                        for key in keys:
                            val = val.get(key, {}) if isinstance(val, dict) else {}
                        return val if isinstance(val, str) else match.group(0)

                    tool_copy["description"] = re.sub(r"\{([^}]+)\}", replace_config, desc)
                    available_tools.append(tool_copy)

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
                elif result["type"] == "empty_response_retry":
                    if empty_response_retries < 3:
                        empty_response_retries += 1
                        msg = (
                            f"Error: {result['message']}. Retrying ({empty_response_retries}/3)..."
                        )
                        logger.warning(msg)
                        if progress_callback:
                            await progress_callback(msg, "error")

                        messages.append(
                            {
                                "role": "user",
                                "content": f"Error: {result['message']}. Please try again.",
                            }
                        )
                        continue

                    logger.error(f"Invalid AI response: {result['message']}")
                    return f"Error: {result['message']}"
                elif result["type"] == "final_text":
                    # Generate persistence summary before returning
                    if persistent_tool_calls and progress_callback:
                        await self._generate_and_store_persistence_summary(
                            persistent_tool_calls, progress_callback
                        )
                    return f"{result['text']}{result_suffix}"
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

                            # Log result, truncating image data in content blocks
                            if isinstance(tool_result, list):
                                # tool_result is list[dict] - Anthropic content blocks
                                log_blocks = []
                                for block in tool_result:
                                    if block.get("type") == "image":
                                        source = block.get("source", {})
                                        data = source.get("data", "")
                                        truncated = data[:64] + "..." if len(data) > 64 else data
                                        log_blocks.append(
                                            f"[image: {source.get('media_type')}, {truncated}]"
                                        )
                                    elif block.get("type") == "text":
                                        log_blocks.append(block.get("text", "")[:100])
                                log_result = ", ".join(log_blocks)
                            else:
                                log_result = str(tool_result)[:100]
                            logger.info(f"Tool {tool['name']} executed: {log_result}...")

                            # Check if this tool should be persisted
                            tool_def = None
                            for t in TOOLS + self.additional_tools:
                                if t["name"] == tool["name"]:
                                    tool_def = t
                                    break

                            if tool_def and tool_def.get("persist", "none") != "none":
                                persist_entry = {
                                    "tool_name": tool["name"],
                                    "input": tool["input"],
                                    "output": tool_result,
                                    "persist_type": tool_def["persist"],
                                }

                                # Handle artifact-type tools
                                if tool_def["persist"] == "artifact":
                                    artifact_url = await self._create_artifact_for_tool(
                                        tool["name"], tool["input"], tool_result, tool_executors
                                    )
                                    if artifact_url:
                                        persist_entry["artifact_url"] = artifact_url
                                elif tool_def["persist"] == "exact":
                                    # TODO: we should call the progress_callback immediately?
                                    raise NotImplementedError

                                persistent_tool_calls.append(persist_entry)

                            # If this is the final_answer tool, return its result directly
                            if tool["name"] == "final_answer":
                                # final_answer always returns str, not list[dict]
                                tool_result_str = (
                                    tool_result
                                    if isinstance(tool_result, str)
                                    else str(tool_result)
                                )
                                cleaned_result = client.cleanup_raw_text(tool_result_str)
                                # Check if there are other tool calls besides progress_report
                                other_tools = [
                                    t
                                    for t in result["tools"]
                                    if t["name"] not in ["final_answer", "progress_report"]
                                ]
                                if other_tools:
                                    logger.warning(
                                        "Rejecting final answer, since non-progress_report tool calls were seen."
                                    )
                                    tool_result = "REJECTED: final_answer must be called separately from other tools (progress_report is allowed)."
                                elif "<thinking>" in tool_result_str and (
                                    not cleaned_result or cleaned_result == "..."
                                ):
                                    logger.warning(
                                        "Final answer was empty after stripping thinking tags, continuing turn"
                                    )
                                else:
                                    # Generate persistence summary before returning
                                    if persistent_tool_calls and progress_callback:
                                        await self._generate_and_store_persistence_summary(
                                            persistent_tool_calls, progress_callback
                                        )
                                    return f"{cleaned_result}{result_suffix}"

                        except Exception as e:
                            logger.warning(f"Tool {tool['name']} failed: {e}", exc_info=True)
                            tool_result = repr(e)

                        if self.vision_model and not vision_switched:
                            # Check if tool result contains images (Anthropic blocks)
                            has_images = isinstance(tool_result, list) and any(
                                block.get("type") == "image" for block in tool_result
                            )
                            if has_images:
                                vision_switched = True
                                fallback_slug = re.sub(
                                    r"(?:[^:]*:)?(?:.*/)?([^#/]+)(?:#.*)?", r"\1", self.vision_model
                                )
                                result_suffix += f" [image fallback to {fallback_slug}]"

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
            logger.error(f"Agent iteration failed: {str(e)}", exc_info=True)

        finally:
            if "execute_python" in tool_executors:
                await tool_executors["execute_python"].cleanup()

        # Generate persistence summary before failing
        if persistent_tool_calls and progress_callback:
            await self._generate_and_store_persistence_summary(
                persistent_tool_calls, progress_callback
            )

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

        # Handle error responses from router (including refusals)
        if "error" in response:
            return {"type": "error", "message": response["error"]}

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
        return {
            "type": "empty_response_retry",
            "message": "No valid text or tool use found in response",
        }

    async def _create_artifact_for_tool(
        self, tool_name: str, tool_input: dict, tool_result: str | list[dict], tool_executors: dict
    ) -> str | None:
        """Create an artifact for tool input and result and return its URL."""
        try:
            if "share_artifact" in tool_executors:
                import json

                # Format artifact content with both input and output
                artifact_content = f"# {tool_name} Tool Call\n\n"
                artifact_content += (
                    f"## Input\n```json\n{json.dumps(tool_input, indent=2)}\n```\n\n"
                )
                artifact_content.replace("\\n", "\n")
                # Convert list[dict] to JSON string if needed
                output_str = (
                    json.dumps(tool_result, indent=2)
                    if isinstance(tool_result, list)
                    else tool_result
                )
                artifact_content += f"## Output\n{output_str}"

                artifact_executor = tool_executors["share_artifact"]
                artifact_result = await artifact_executor.execute(artifact_content)
                if artifact_result.startswith("Artifact shared: "):
                    return artifact_result.replace("Artifact shared: ", "")
            return None
        except Exception as e:
            logger.warning(f"Failed to create artifact: {e}")
            return None

    async def _generate_and_store_persistence_summary(
        self, persistent_tool_calls: list[dict], progress_callback
    ):
        """Generate a single summary for all persistent tool calls and store it."""
        if not persistent_tool_calls:
            return

        try:
            # Build summary input
            summary_input = []
            summary_input.append("The following tool calls were made during this conversation:")

            for call in persistent_tool_calls:
                tool_name = call["tool_name"]
                tool_input = call["input"]
                tool_output = call["output"]
                persist_type = call["persist_type"]

                # Replace image data with placeholder in both input and output
                def replace_image_data(content):
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        # Anthropic content blocks - extract text and represent images
                        parts = []
                        for block in content:
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "image":
                                source = block.get("source", {})
                                media_type = source.get("media_type", "image")
                                parts.append(f"[image: {media_type}]")
                        return "\n".join(parts)
                    return str(content)

                tool_input_display = replace_image_data(tool_input)
                tool_output_display = replace_image_data(tool_output)

                summary_input.append(
                    f"\n\n# Calling tool **{tool_name}** (persist: {persist_type})"
                )
                summary_input.append(f"## **Input:**\n{tool_input_display}\n")
                summary_input.append(f"## **Output:**\n{tool_output_display}\n")
                if "artifact_url" in call:
                    summary_input.append(
                        f"(Tool call I/O stored verbatim as artifact: {call['artifact_url']})\n"
                    )

            summary_input.append(
                "\nPlease provide a concise summary of what was accomplished in these tool calls."
            )

            # Generate summary using the tools model from config
            summary_model = self.config["tools"]["summary"]["model"]

            if self.model_router:
                summary_response, _, _ = await self.model_router.call_raw_with_model(
                    summary_model,
                    [{"role": "user", "content": "\n".join(summary_input)}],
                    "As an AI agent, you need to remember in the future what tools you used when generating a response, and what did the tools tell you, as you may get challenged on that. This is your moment to generate that memory. Summarize all the tool uses in a single concise paragraph. In case artifact links are included, you MUST include all these artifact links in your summary, tied to the respective tool calls.",
                )

                summary_text = ""
                if isinstance(summary_response, dict) and "content" in summary_response:
                    for content_block in summary_response["content"]:
                        if content_block.get("type") == "text":
                            summary_text += content_block["text"]
                elif isinstance(summary_response, dict) and "output_text" in summary_response:
                    summary_text = summary_response["output_text"]
                elif isinstance(summary_response, str):
                    summary_text = summary_response

                if summary_text:
                    # Call progress callback with assistant_silent type
                    await progress_callback(summary_text.strip(), "tool_persistence")

        except Exception as e:
            logger.error(f"Failed to generate tool persistence summary: {e}", exc_info=True)
