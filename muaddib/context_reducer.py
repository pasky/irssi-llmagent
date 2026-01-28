"""Context reducer for synthesizing conversation history before agent calls."""

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .main import MuaddibAgent

logger = logging.getLogger(__name__)


class ContextReducer:
    """Reduces conversation context to essential information for agent calls."""

    def __init__(self, agent: "MuaddibAgent"):
        self.agent = agent

    @property
    def config(self) -> dict[str, Any]:
        """Get context_reducer configuration from root level."""
        return self.agent.config.get("context_reducer", {})

    @property
    def is_configured(self) -> bool:
        """Check if context reducer is properly configured with both model and prompt."""
        return bool(self.config.get("model") and self.config.get("prompt"))

    async def reduce(
        self,
        context: list[dict[str, str]],
        agent_system_prompt: str,
    ) -> list[dict[str, str]]:
        """Reduce context to essential information.

        Args:
            context: Full assembled context (chapter + messages). Last message is triggering.
            agent_system_prompt: The system prompt the agent will use (for context).

        Returns:
            Condensed context as list of messages (excludes triggering).
            Returns context[:-1] unchanged if reduction fails or not configured.
        """
        if not self.is_configured:
            logger.debug("Context reducer not configured, returning full context")
            return context[:-1] if len(context) > 1 else []

        context_to_reduce = context[:-1]
        if not context_to_reduce:
            return []

        model = self.config["model"]
        prompt_template = self.config["prompt"]

        formatted_context = self._format_context_for_reduction(context, agent_system_prompt)

        try:
            resp, client, _, _ = await self.agent.model_router.call_raw_with_model(
                model,
                [{"role": "user", "content": formatted_context}],
                prompt_template,
                reasoning_effort="low",
            )
            response_text = client.extract_text_from_response(resp)

            if not response_text:
                logger.warning("Empty response from context reducer")
                return context_to_reduce

            reduced = self._parse_reduced_context(response_text)

            logger.debug(
                f"Context reduced: {len(context_to_reduce)} messages -> {len(reduced)} messages"
            )
            return reduced

        except Exception as e:
            logger.error(f"Context reduction failed: {e}")
            return context_to_reduce

    def _format_context_for_reduction(
        self,
        context: list[dict[str, str]],
        agent_system_prompt: str,
    ) -> str:
        """Format context and system prompt for the reducer model."""
        lines = []

        lines.append("## AGENT SYSTEM PROMPT (for context)")
        lines.append(agent_system_prompt)

        lines.append("")
        lines.append("## CONVERSATION HISTORY TO CONDENSE")
        for msg in context[:-1]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")

        lines.append("")
        lines.append("## TRIGGERING INPUT (for relevance - do not include in output)")
        content = context[-1].get("content", "")
        lines.append(content)

        return "\n".join(lines)

    def _parse_reduced_context(self, response: str) -> list[dict[str, str]]:
        """Parse reducer output back into message list."""
        messages = []

        pattern = r"\[(USER|ASSISTANT)\]:\s*(.*?)(?=\n\[(?:USER|ASSISTANT)\]:|$)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for role, content in matches:
                content = content.strip()
                if content:
                    messages.append({"role": role.lower(), "content": content})
        else:
            if response.strip():
                messages.append(
                    {
                        "role": "user",
                        "content": "<context_summary>" + response.strip() + "</context_summary>",
                    }
                )

        return messages
