"""Shared command handling for room monitors."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..agentic_actor.actor import AgentResult
from ..providers import parse_model_spec
from ..rate_limiter import RateLimiter
from .autochronicler import AutoChronicler
from .message import RoomMessage
from .proactive import ProactiveDebouncer

logger = logging.getLogger(__name__)

MODE_TOKENS = {"!s", "!S", "!a", "!d", "!D", "!u", "!h"}
FLAG_TOKENS = {"!c"}


@dataclass
class ParsedPrefix:
    """Result of parsing command prefix from message."""

    no_context: bool
    mode_token: str | None
    model_override: str | None
    query_text: str
    error: str | None = None


class ResponseCleaner(Protocol):
    """Optional response cleanup hook."""

    def __call__(self, text: str, nick: str) -> str: ...


def model_str_core(model: Any) -> str:
    """Extract core model names: provider:namespace/model#routing -> model."""

    return re.sub(r"(?:[-\w]*:)?(?:[-\w]*/)?([-\w]+)(?:#[-\w,]*)?", r"\1", str(model))


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = _deep_merge_config(value, {})
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value

    for key, value in override.items():
        if key == "ignore_users" and isinstance(value, list):
            base_list = result.get(key, [])
            result[key] = [*base_list, *value]
            continue
        if key == "prompt_vars" and isinstance(value, dict):
            base_vars = result.get(key, {})
            merged_vars = dict(base_vars)
            for var_key, var_value in value.items():
                if var_key in merged_vars and isinstance(var_value, str):
                    # Concatenate string values for the same key
                    merged_vars[var_key] = f"{merged_vars[var_key]}{var_value}"
                else:
                    merged_vars[var_key] = var_value
            result[key] = merged_vars
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_config(result[key], value)
            continue
        if isinstance(value, list):
            result[key] = list(value)
            continue
        result[key] = value

    return result


def get_room_config(config: dict[str, Any], room_name: str) -> dict[str, Any]:
    """Get merged room config from common + room overrides."""

    rooms = config.get("rooms", {})
    common = rooms.get("common", {})
    room = rooms.get(room_name, {})
    return _deep_merge_config(common, room)


class RoomCommandHandler:
    """Shared command + proactive handling for rooms."""

    def __init__(
        self,
        agent: Any,
        room_name: str,
        room_config: dict[str, Any],
        response_cleaner: ResponseCleaner | None = None,
    ) -> None:
        self.agent = agent
        self.room_name = room_name
        self.room_config = room_config
        self.response_cleaner = response_cleaner

        command_config = self.room_config["command"]
        proactive_config = self.room_config["proactive"]

        self.rate_limiter = RateLimiter(command_config["rate_limit"], command_config["rate_period"])
        self.proactive_rate_limiter = RateLimiter(
            proactive_config["rate_limit"], proactive_config["rate_period"]
        )
        self.proactive_debouncer = ProactiveDebouncer(proactive_config["debounce_seconds"])
        self.autochronicler = AutoChronicler(self.agent.history, self)

    @property
    def command_config(self) -> dict[str, Any]:
        return self.room_config["command"]

    @property
    def proactive_config(self) -> dict[str, Any]:
        return self.room_config["proactive"]

    def _response_max_bytes(self) -> int:
        return int(self.command_config.get("response_max_bytes", 600))

    def _clean_response_text(self, response_text: str, nick: str) -> str:
        cleaned = response_text.strip()
        if self.response_cleaner:
            cleaned = self.response_cleaner(cleaned, nick)
        return cleaned.strip()

    def should_ignore_user(self, nick: str) -> bool:
        ignore_list = self.command_config.get("ignore_users", [])
        return any(nick.lower() == ignored.lower() for ignored in ignore_list)

    @staticmethod
    def _normalize_server_tag(server_tag: str) -> str:
        if server_tag.startswith("discord:"):
            return server_tag.split("discord:", 1)[1]
        if server_tag.startswith("slack:"):
            return server_tag.split("slack:", 1)[1]
        return server_tag

    def _get_channel_key(self, server_tag: str, channel_name: str) -> str:
        normalized_server = self._normalize_server_tag(server_tag)
        return f"{normalized_server}#{channel_name}"

    def _get_proactive_channel_key(self, server_tag: str, channel_name: str) -> str:
        return self._get_channel_key(server_tag, channel_name)

    def build_system_prompt(self, mode: str, mynick: str, model_override: str | None = None) -> str:
        """Build a command system prompt with standard substitutions."""

        try:
            prompt_template = self.command_config["modes"][mode]["prompt"]
        except KeyError:
            raise ValueError(f"Command mode '{mode}' not found in config") from None

        modes_config = self.command_config["modes"]

        def get_model(m: str) -> str:
            if model_override and m == mode:
                return model_str_core(model_override)
            return model_str_core(modes_config[m]["model"])

        thinking_model = modes_config["serious"].get("thinking_model")
        prompt_vars = self.room_config.get("prompt_vars", {})
        return prompt_template.format(
            mynick=mynick,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            sarcastic_model=get_model("sarcastic"),
            serious_model=get_model("serious"),
            thinking_model=model_str_core(thinking_model)
            if thinking_model
            else get_model("serious"),
            unsafe_model=get_model("unsafe"),
            **prompt_vars,
        )

    def get_channel_mode(self, server_tag: str, chan_name: str) -> str:
        channel_modes = self.command_config.get("channel_modes", {})
        channel_key = self._get_channel_key(server_tag, chan_name)
        if channel_key in channel_modes:
            return channel_modes[channel_key]
        return self.command_config.get("default_mode", "classifier")

    async def classify_mode(self, context: list[dict[str, str]]) -> str:
        """Use preprocessing model to classify message mode."""

        try:
            if not context:
                raise ValueError(
                    "Context cannot be empty - must include at least the current message"
                )

            current_message = context[-1]["content"]

            # Clean message content if it has IRC nick formatting like "<nick> message"
            message_match = re.search(r"<[^>]+>\s*(.*)", current_message)
            if message_match:
                current_message = message_match.group(1).strip()

            prompt = self.command_config["mode_classifier"]["prompt"].format(
                message=current_message
            )
            model = self.command_config["mode_classifier"]["model"]
            resp, client, _, _ = await self.agent.model_router.call_raw_with_model(
                model, context, prompt
            )
            response = client.extract_text_from_response(resp)

            serious_count = response.count("SERIOUS")
            sarcastic_count = response.count("SARCASTIC")
            unsafe_count = response.count("UNSAFE")

            # Check for UNSAFE first (highest priority for explicit requests)
            if unsafe_count > 0:
                return "UNSAFE"
            if serious_count == 0 and sarcastic_count == 0:
                logger.warning("Invalid mode classification response: %s", response)
                return "SARCASTIC"
            if serious_count <= sarcastic_count:
                return "SARCASTIC"
            return (
                "EASY_SERIOUS"
                if response.count("EASY_SERIOUS") > response.count("THINKING_SERIOUS")
                else "THINKING_SERIOUS"
            )
        except Exception as e:
            logger.error("Error classifying mode: %s", e)
            return "UNSAFE"

    async def should_interject_proactively(
        self, context: list[dict[str, str]]
    ) -> tuple[bool, str, bool]:
        """Use preprocessing models to decide if bot should interject in conversation proactively.

        Args:
            context: Conversation context including the current message as the last entry

        Returns:
            (should_interject, reason, is_test_mode): Tuple of decision, reasoning, and test mode flag
        """
        try:
            if not context:
                return False, "No context provided", False

            current_message = context[-1]["content"]

            # Clean message content if it has IRC nick formatting like "<nick> message"
            message_match = re.search(r"<?\S+>\s*(.*)", current_message)
            if message_match:
                current_message = message_match.group(1).strip()

            # Use full context for better decision making, but specify the current message in prompt
            prompt = self.proactive_config["prompts"]["interject"].format(message=current_message)
            validation_models = self.proactive_config["models"]["validation"]

            final_score = None
            all_responses = []

            for i, model in enumerate(validation_models):
                resp, client, _, _ = await self.agent.model_router.call_raw_with_model(
                    model, context, prompt
                )
                response = client.extract_text_from_response(resp)

                if not response or response.startswith("API error:"):
                    return False, f"No response from validation model {i + 1}", False

                response = response.strip()
                all_responses.append(f"Model {i + 1} ({model}): {response}")

                score_match = re.search(r"(\d+)/10", response)
                if not score_match:
                    logger.warning(
                        "No valid score found in proactive interject response from model %s: %s",
                        i + 1,
                        response,
                    )
                    return False, f"No score found in validation step {i + 1}", False

                score = int(score_match.group(1))
                final_score = score

                logger.debug(
                    f"Proactive validation step {i + 1}/{len(validation_models)} - Model: {model}, Score: {score}"
                )

                threshold = self.proactive_config["interject_threshold"]
                if score < threshold - 1:
                    if i > 0:
                        logger.info(
                            "Proactive interjection rejected at step %s/%s (%s... Score: %s)",
                            i + 1,
                            len(validation_models),
                            current_message[:150],
                            score,
                        )
                    else:
                        logger.debug(
                            "Proactive interjection rejected at step %s/%s (Score: %s)",
                            i + 1,
                            len(validation_models),
                            score,
                        )
                    return (
                        False,
                        f"Rejected at validation step {i + 1} (Score: {score})",
                        False,
                    )

            if final_score is not None:
                threshold = self.proactive_config["interject_threshold"]

                if final_score >= threshold:
                    logger.debug(
                        "Proactive interjection triggered for message: %s... (Final Score: %s)",
                        current_message[:150],
                        final_score,
                    )
                    return True, f"Interjection decision (Final Score: {final_score})", False
                if final_score >= threshold - 1:
                    logger.debug(
                        "Proactive interjection BARELY triggered for message: %s... (Final Score: %s) - SWITCHING TO TEST MODE",
                        current_message[:150],
                        final_score,
                    )
                    return True, f"Barely triggered - test mode (Final Score: {final_score})", True
                return False, f"No interjection (Final Score: {final_score})", False

            return False, "No valid final score", False

        except Exception as e:
            logger.error("Error checking proactive interject: %s", e)
            return False, f"Error: {str(e)}", False

    async def _handle_debounced_proactive_check(
        self,
        msg: RoomMessage,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> None:
        try:
            if not self.proactive_rate_limiter.check_limit():
                logger.debug(
                    "Proactive interjection rate limit exceeded during debounced check, skipping message from %s",
                    msg.nick,
                )
                return

            context = await self.agent.history.get_context_for_message(
                msg, self.proactive_config["history_size"]
            )
            should_interject, reason, forced_test_mode = await self.should_interject_proactively(
                context
            )
            if not should_interject:
                return

            channel_key = self._get_proactive_channel_key(msg.server_tag, msg.channel_name)
            classified_mode = await self.classify_mode(context)
            if not classified_mode.endswith("SERIOUS"):
                test_channels = self.agent.config.get("behavior", {}).get(
                    "proactive_interjecting_test", []
                )
                is_test_channel = test_channels and channel_key in test_channels
                mode_desc = "[TEST MODE] " if is_test_channel else ""
                logger.warning(
                    "%sProactive interjection suggested but not serious mode: %s. Reason: %s",
                    mode_desc,
                    classified_mode,
                    reason,
                )
                return

            test_channels = self.agent.config.get("behavior", {}).get(
                "proactive_interjecting_test", []
            )
            is_test_channel = test_channels and channel_key in test_channels
            if is_test_channel or forced_test_mode:
                test_reason = "[BARELY TRIGGERED]" if forced_test_mode else "[TEST CHANNEL]"
                logger.info(
                    "[TEST MODE] %s Would interject proactively for message from %s in %s: %s... Reason: %s",
                    test_reason,
                    msg.nick,
                    msg.channel_name,
                    msg.content[:150],
                    reason,
                )
                send_message = False
            else:
                logger.info(
                    "Interjecting proactively for message from %s in %s: %s... Reason: %s",
                    msg.nick,
                    msg.channel_name,
                    msg.content[:150],
                    reason,
                )
                send_message = True

            agent_result = await self._run_actor(
                context,
                msg.mynick,
                mode="serious",
                reasoning_effort="low" if classified_mode == "EASY_SERIOUS" else "medium",
                model=self.proactive_config["models"]["serious"],
                extra_prompt=" " + self.proactive_config["prompts"]["serious_extra"],
                arc=msg.arc,
                secrets=msg.secrets,
            )

            if not agent_result or not agent_result.text or agent_result.text.startswith("Error: "):
                logger.info("Agent decided not to interject proactively for %s", msg.channel_name)
                return

            response_text = self._clean_response_text(agent_result.text, msg.nick)
            if send_message:
                response_text = f"[{model_str_core(self.proactive_config['models']['serious'])}] {response_text}"
                logger.info(
                    "Sending proactive agent (%s) response to %s: %s",
                    classified_mode,
                    msg.channel_name,
                    response_text,
                )
                await reply_sender(response_text)
                response_msg = dataclasses.replace(msg, nick=msg.mynick, content=response_text)
                await self.agent.history.add_message(response_msg, mode=classified_mode)
                await self.autochronicler.check_and_chronicle(
                    msg.mynick,
                    msg.server_tag,
                    msg.channel_name,
                    self.command_config["history_size"],
                )
            else:
                logger.info(
                    "[TEST MODE] Generated proactive response for %s: %s",
                    msg.channel_name,
                    response_text,
                )
        except Exception as e:
            logger.error("Error in debounced proactive check for %s: %s", msg.channel_name, e)

    async def _run_actor(
        self,
        context: list[dict[str, str]],
        mynick: str,
        *,
        mode: str,
        extra_prompt: str = "",
        model: str | list[str] | None = None,
        no_context: bool = False,
        secrets: dict[str, Any] | None = None,
        **actor_kwargs,
    ) -> AgentResult | None:
        mode_cfg = self.command_config["modes"][mode].copy()
        if no_context:
            context = context[-1:]
            mode_cfg["include_chapter_summary"] = False
        elif mode in {"serious", "unsafe"} and len(context) > 1:
            mode_cfg["reduce_context"] = True

        model_override = model if isinstance(model, str) else None
        system_prompt = self.build_system_prompt(mode, mynick, model_override) + extra_prompt

        try:
            agent_result = await self.agent.run_actor(
                context,
                mode_cfg=mode_cfg,
                system_prompt=system_prompt,
                model=model,
                secrets=secrets,
                **actor_kwargs,
            )
        except Exception as e:
            logger.error("Error during agent execution: %s", e, exc_info=True)
            return AgentResult(
                text=f"Error: {e}",
                total_input_tokens=None,
                total_output_tokens=None,
                total_cost=None,
                tool_calls_count=0,
                primary_model=None,
            )

        if agent_result is None:
            return None

        response_text = agent_result.text
        max_response_bytes = self._response_max_bytes()
        if response_text and len(response_text.encode("utf-8")) > max_response_bytes:
            logger.info(
                "Response too long (%s bytes, max %s), creating artifact",
                len(response_text.encode("utf-8")),
                max_response_bytes,
            )
            response_text = await self._long_response_to_artifact(response_text)
        if response_text:
            response_text = response_text.strip()

        return dataclasses.replace(agent_result, text=response_text)

    async def _long_response_to_artifact(self, full_response: str) -> str:
        from ..agentic_actor.tools import ShareArtifactExecutor

        executor = ShareArtifactExecutor.from_config(self.agent.config)
        artifact_result = await executor.execute(full_response)
        artifact_url = artifact_result.split("Artifact shared: ")[1].strip()

        max_response_bytes = self._response_max_bytes()

        # Trim to fit byte limit while respecting character boundaries
        trimmed = full_response
        while len(trimmed.encode("utf-8")) > max_response_bytes and trimmed:
            trimmed = trimmed[:-1]

        # Try to break at end of sentence or word for cleaner output
        min_len = max(0, len(trimmed) - 100)
        last_sentence = trimmed.rfind(".")
        last_word = trimmed.rfind(" ")
        if last_sentence > min_len:
            trimmed = trimmed[: last_sentence + 1]
        elif last_word > min_len:
            trimmed = trimmed[:last_word]

        trimmed += f"... full response: {artifact_url}"

        return trimmed

    async def handle_command(
        self,
        msg: RoomMessage,
        trigger_message_id: int,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> None:
        if not self.rate_limiter.check_limit():
            logger.warning("Rate limiting triggered for %s", msg.nick)
            rate_msg = f"{msg.nick}: Slow down a little, will you? (rate limiting)"
            await reply_sender(rate_msg)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=rate_msg)
            await self.agent.history.add_message(response_msg)
            return

        logger.info(
            "Received command from %s on %s/%s: %s",
            msg.nick,
            msg.server_tag,
            msg.channel_name,
            msg.content,
        )

        # Work with fixed context from now on to avoid debouncing/classification races!
        default_size = self.command_config["history_size"]
        max_size = max(
            default_size,
            *(mode.get("history_size", 0) for mode in self.command_config["modes"].values()),
        )
        context = await self.agent.history.get_context_for_message(msg, max_size)

        # Debounce briefly to consolidate quick followups e.g. due to automatic IRC message splits
        debounce = self.command_config.get("debounce", 0)
        if debounce > 0:
            original_timestamp = time.time()
            await asyncio.sleep(debounce)

            followups = await self.agent.history.get_recent_messages_since(
                msg.server_tag,
                msg.channel_name,
                msg.nick,
                original_timestamp,
                thread_id=msg.thread_id,
            )
            if followups:
                logger.debug("Debounced %s followup messages from %s", len(followups), msg.nick)
                context[-1]["content"] += "\n" + "\n".join([m["message"] for m in followups])

        await self._route_command(
            msg,
            context,
            default_size,
            trigger_message_id,
            reply_sender,
        )
        await self.proactive_debouncer.cancel_channel(
            self._get_proactive_channel_key(msg.server_tag, msg.channel_name)
        )
        await self.autochronicler.check_and_chronicle(
            msg.mynick, msg.server_tag, msg.channel_name, default_size
        )

    def _parse_prefix(self, message: str) -> ParsedPrefix:
        """Parse leading modifier tokens from message.

        Recognized tokens (any order at start):
          - !c: no-context flag
          - !s/!S/!a/!d/!D/!u/!h: mode commands
          - @modelid: model override

        Parsing stops at first non-modifier token. Only one mode allowed.
        """
        text = message.strip()
        if not text:
            return ParsedPrefix(False, None, None, "", None)

        tokens = text.split()
        no_context = False
        mode_token: str | None = None
        model_override: str | None = None
        error: str | None = None
        consumed = 0

        for i, tok in enumerate(tokens):
            if tok in FLAG_TOKENS:
                no_context = True
                consumed = i + 1
                continue

            if tok in MODE_TOKENS:
                if mode_token is not None:
                    error = "Only one mode command allowed."
                    break
                mode_token = tok
                consumed = i + 1
                continue

            if tok.startswith("@") and len(tok) > 1:
                if model_override is None:
                    model_override = tok[1:]
                consumed = i + 1
                continue

            if tok.startswith("!"):
                error = f"Unknown command '{tok}'. Use !h for help."
                break

            break

        query_text = " ".join(tokens[consumed:]) if consumed > 0 else text
        return ParsedPrefix(
            no_context=no_context,
            mode_token=mode_token,
            model_override=model_override,
            query_text=query_text,
            error=error,
        )

    async def handle_passive_message(
        self,
        msg: RoomMessage,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> None:
        channel_key = self._get_proactive_channel_key(msg.server_tag, msg.channel_name)
        if (
            channel_key
            in self.proactive_config["interjecting"] + self.proactive_config["interjecting_test"]
        ):
            await self.proactive_debouncer.schedule_check(
                msg,
                channel_key,
                reply_sender,
                self._handle_debounced_proactive_check,
            )

        max_size = self.command_config["history_size"]
        await self.autochronicler.check_and_chronicle(
            msg.mynick, msg.server_tag, msg.channel_name, max_size
        )

    async def _route_command(
        self,
        msg: RoomMessage,
        context: list[dict[str, str]],
        default_size: int,
        trigger_message_id: int,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> None:
        modes_config = self.command_config["modes"]
        parsed = self._parse_prefix(msg.content)

        if parsed.error:
            logger.warning(
                "Command parse error from %s: %s (%s)", msg.nick, parsed.error, msg.content
            )
            await reply_sender(f"{msg.nick}: {parsed.error}")
            return

        no_context = parsed.no_context
        model_override = parsed.model_override
        query_text = parsed.query_text

        if model_override:
            logger.debug("Overriding model to %s", model_override)

        if parsed.mode_token == "!h":
            logger.debug("Sending help message to %s", msg.nick)
            sarcastic_model = modes_config["sarcastic"]["model"]
            serious_model = modes_config["serious"]["model"]
            thinking_model = modes_config["serious"].get("thinking_model")
            thinking_desc = f" ({thinking_model})" if thinking_model else ""
            unsafe_model = modes_config["unsafe"]["model"]
            classifier_model = self.command_config["mode_classifier"]["model"]

            channel_mode = self.get_channel_mode(msg.server_tag, msg.channel_name)
            if channel_mode == "serious":
                default_desc = (
                    f"serious agentic mode with web tools ({serious_model}), "
                    f"!d is explicit sarcastic diss mode ({sarcastic_model}), !u is unsafe mode ({unsafe_model}), "
                    f"!a forces thinking{thinking_desc}; use @modelid to override model"
                )
            elif channel_mode == "sarcastic":
                default_desc = (
                    f"sarcastic mode ({sarcastic_model}), !s (quick, {serious_model}) & "
                    f"!a (thinking{thinking_desc}) is serious agentic mode with web tools, !u is unsafe mode ({unsafe_model}); "
                    f"use @modelid to override model"
                )
            else:
                default_desc = (
                    f"automatic mode ({classifier_model} decides), !d is explicit sarcastic diss mode ({sarcastic_model}), "
                    f"!s (quick, {serious_model}) & !a (thinking{thinking_desc}) is serious agentic mode with web tools, "
                    f"!u is unsafe mode ({unsafe_model}); use @modelid to override model"
                )

            help_msg = f"default is {default_desc}, !c disables context"
            await reply_sender(help_msg)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=help_msg)
            await self.agent.history.add_message(response_msg)
            return

        mode = None
        reasoning_effort = "minimal"

        if parsed.mode_token in {"!S", "!d"}:
            logger.debug("Processing explicit sarcastic request from %s: %s", msg.nick, query_text)
            mode = "SARCASTIC"
        elif parsed.mode_token == "!D":
            logger.debug(
                "Processing explicit thinking sarcastic request from %s: %s", msg.nick, query_text
            )
            mode = "SARCASTIC"
            reasoning_effort = "high"
        elif parsed.mode_token == "!s":
            logger.debug("Processing explicit serious request from %s: %s", msg.nick, query_text)
            mode = "EASY_SERIOUS"
        elif parsed.mode_token == "!a":
            logger.debug("Processing explicit agentic request from %s: %s", msg.nick, query_text)
            mode = "THINKING_SERIOUS"
        elif parsed.mode_token == "!u":
            logger.debug("Processing explicit unsafe request from %s: %s", msg.nick, query_text)
            mode = "UNSAFE"
        else:
            logger.debug("Processing automatic mode request from %s: %s", msg.nick, query_text)

            channel_mode = self.get_channel_mode(msg.server_tag, msg.channel_name)
            if channel_mode == "serious":
                mode = await self.classify_mode(context[-default_size:])
                logger.debug("Auto-classified message as %s mode", mode)
                if mode == "SARCASTIC":
                    mode = "EASY_SERIOUS"
                    logger.debug(
                        "...but forcing channel-configured serious mode for %s", msg.channel_name
                    )
            elif channel_mode == "sarcastic":
                mode = "SARCASTIC"
                logger.debug("Using channel-configured sarcastic mode for %s", msg.channel_name)
            elif channel_mode == "unsafe":
                mode = "UNSAFE"
                logger.debug("Using channel-configured unsafe mode for %s", msg.channel_name)
            else:
                mode = await self.classify_mode(context)
                logger.debug("Auto-classified message as %s mode", mode)

        async def progress_cb(text: str) -> None:
            await reply_sender(text)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=text)
            await self.agent.history.add_message(response_msg)

        async def persistence_cb(text: str) -> None:
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=text)
            await self.agent.history.add_message(
                response_msg, content_template="[internal monologue] {message}"
            )

        if mode == "SARCASTIC":
            agent_result = await self._run_actor(
                context[-modes_config["sarcastic"].get("history_size", default_size) :],
                msg.mynick,
                mode="sarcastic",
                reasoning_effort=reasoning_effort,
                allowed_tools=[],
                progress_callback=progress_cb,
                persistence_callback=persistence_cb,
                arc=msg.arc,
                no_context=no_context,
                model=model_override,
                secrets=msg.secrets,
            )
        elif mode and mode.endswith("SERIOUS"):
            assert reasoning_effort == "minimal"
            agent_result = await self._run_actor(
                context[-modes_config["serious"].get("history_size", default_size) :],
                msg.mynick,
                mode="serious",
                reasoning_effort="low" if mode == "EASY_SERIOUS" else "medium",
                model=model_override
                or (
                    modes_config["serious"].get("thinking_model")
                    if mode == "THINKING_SERIOUS"
                    else None
                ),
                progress_callback=progress_cb,
                persistence_callback=persistence_cb,
                arc=msg.arc,
                no_context=no_context,
                secrets=msg.secrets,
            )
        elif mode == "UNSAFE":
            agent_result = await self._run_actor(
                context[-modes_config["unsafe"].get("history_size", default_size) :],
                msg.mynick,
                mode="unsafe",
                reasoning_effort="low",
                progress_callback=progress_cb,
                persistence_callback=persistence_cb,
                arc=msg.arc,
                model=model_override,
                no_context=no_context,
                secrets=msg.secrets,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        if agent_result and agent_result.text:
            response_text = self._clean_response_text(agent_result.text, msg.nick)
            cost_str = f"${agent_result.total_cost:.4f}" if agent_result.total_cost else "?"
            logger.info(
                "Sending %s response (%s) to %s: %s",
                mode,
                cost_str,
                msg.channel_name,
                response_text,
            )

            llm_call_id = None
            if agent_result.primary_model:
                try:
                    spec = parse_model_spec(agent_result.primary_model)
                    llm_call_id = await self.agent.history.log_llm_call(
                        provider=spec.provider,
                        model=spec.name,
                        input_tokens=agent_result.total_input_tokens,
                        output_tokens=agent_result.total_output_tokens,
                        cost=agent_result.total_cost,
                        call_type="agent_run",
                        arc_name=msg.arc,
                        trigger_message_id=trigger_message_id,
                    )
                except ValueError:
                    logger.warning("Could not parse model spec: %s", agent_result.primary_model)

            await reply_sender(response_text)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=response_text)
            response_message_id = await self.agent.history.add_message(
                response_msg, mode=mode, llm_call_id=llm_call_id
            )
            if llm_call_id:
                await self.agent.history.update_llm_call_response(llm_call_id, response_message_id)

            if agent_result.total_cost and agent_result.total_cost > 0.2:
                in_tokens = agent_result.total_input_tokens or 0
                out_tokens = agent_result.total_output_tokens or 0
                cost_msg = (
                    f"(this message used {agent_result.tool_calls_count} tool calls, "
                    f"{in_tokens} in / {out_tokens} out tokens, "
                    f"and cost ${agent_result.total_cost:.4f})"
                )
                logger.info("Cost followup for %s: %s", msg.channel_name, cost_msg)
                await reply_sender(cost_msg)
                response_msg = dataclasses.replace(msg, nick=msg.mynick, content=cost_msg)
                await self.agent.history.add_message(response_msg)

            if agent_result.total_cost:
                cost_before = await self.agent.history.get_arc_cost_today(msg.arc)
                cost_before -= agent_result.total_cost
                dollars_before = int(cost_before)
                dollars_after = int(cost_before + agent_result.total_cost)
                if dollars_after > dollars_before:
                    total_today = cost_before + agent_result.total_cost
                    fun_msg = f"(fun fact: my messages in this channel have already cost ${total_today:.4f} today)"
                    logger.info("Daily cost milestone for %s: %s", msg.arc, fun_msg)
                    await reply_sender(fun_msg)
                    response_msg = dataclasses.replace(msg, nick=msg.mynick, content=fun_msg)
                    await self.agent.history.add_message(response_msg)
        else:
            logger.info("Agent in %s mode chose not to answer for %s", mode, msg.channel_name)
