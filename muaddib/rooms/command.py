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
from typing import Any, Protocol, TypeAlias

from ..agentic_actor.actor import AgentResult
from ..providers import parse_model_spec
from ..rate_limiter import RateLimiter
from .autochronicler import AutoChronicler
from .message import RoomMessage
from .proactive import ProactiveDebouncer

logger = logging.getLogger(__name__)

HELP_TOKEN = "!h"
FLAG_TOKENS = {"!c"}

SteeringKey: TypeAlias = tuple[str, str, str | None]


@dataclass
class QueuedInboundMessage:
    """Queued inbound message waiting for an active steering session runner."""

    kind: str  # "command" | "passive"
    msg: RoomMessage
    trigger_message_id: int | None
    reply_sender: Callable[[str], Awaitable[None]]
    completion: asyncio.Future[None]


@dataclass
class SteeringSession:
    """In-flight steering session for a specific steering key."""

    queue: list[QueuedInboundMessage]


@dataclass
class ParsedPrefix:
    """Result of parsing command prefix from message."""

    no_context: bool
    mode_token: str | None
    model_override: str | None
    query_text: str
    error: str | None = None


@dataclass(frozen=True)
class TriggerRoute:
    """Resolved trigger route with mode key and trigger-specific overrides."""

    mode_key: str
    trigger: str
    overrides: dict[str, Any]


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

        self._trigger_map = self._build_trigger_map(command_config)
        self._default_trigger_by_mode = self._build_default_trigger_map(command_config)
        self._classifier_label_to_trigger = self._build_classifier_label_map(command_config)
        self._primary_label_by_trigger = self._build_primary_label_by_trigger(
            self._classifier_label_to_trigger
        )

        self.rate_limiter = RateLimiter(command_config["rate_limit"], command_config["rate_period"])
        self.proactive_rate_limiter = RateLimiter(
            proactive_config["rate_limit"], proactive_config["rate_period"]
        )
        self.proactive_debouncer = ProactiveDebouncer(proactive_config["debounce_seconds"])
        self.autochronicler = AutoChronicler(self.agent.history, self)

        # Active steering sessions keyed by:
        # - non-threaded: (arc, nick_lower, None)
        # - threaded: (arc, "*", thread_id)
        self._steering_sessions: dict[SteeringKey, SteeringSession] = {}
        self._steering_lock = asyncio.Lock()

    @property
    def command_config(self) -> dict[str, Any]:
        return self.room_config["command"]

    @property
    def proactive_config(self) -> dict[str, Any]:
        return self.room_config["proactive"]

    @staticmethod
    def _build_trigger_map(command_config: dict[str, Any]) -> dict[str, TriggerRoute]:
        trigger_map: dict[str, TriggerRoute] = {}
        modes = command_config.get("modes", {})
        for mode_key, mode_cfg in modes.items():
            triggers = mode_cfg.get("triggers", {})
            if not triggers:
                raise ValueError(f"Mode '{mode_key}' must define at least one trigger")
            for trigger, overrides in triggers.items():
                if trigger in trigger_map:
                    raise ValueError(f"Duplicate trigger '{trigger}' in command mode config")
                if not isinstance(trigger, str) or not trigger.startswith("!"):
                    raise ValueError(f"Invalid trigger '{trigger}' for mode '{mode_key}'")
                trigger_map[trigger] = TriggerRoute(
                    mode_key=mode_key,
                    trigger=trigger,
                    overrides=overrides or {},
                )
        return trigger_map

    @staticmethod
    def _build_default_trigger_map(command_config: dict[str, Any]) -> dict[str, str]:
        defaults: dict[str, str] = {}
        modes = command_config.get("modes", {})
        for mode_key, mode_cfg in modes.items():
            triggers = mode_cfg.get("triggers", {})
            defaults[mode_key] = next(iter(triggers))
        return defaults

    def _build_classifier_label_map(self, command_config: dict[str, Any]) -> dict[str, str]:
        labels = command_config.get("mode_classifier", {}).get("labels", {})
        if not labels:
            raise ValueError("command.mode_classifier.labels must not be empty")
        for label, trigger in labels.items():
            if trigger not in self._trigger_map:
                raise ValueError(
                    f"Classifier label '{label}' points to unknown trigger '{trigger}'"
                )
        return labels

    @staticmethod
    def _build_primary_label_by_trigger(label_map: dict[str, str]) -> dict[str, str]:
        primary: dict[str, str] = {}
        for label, trigger in label_map.items():
            primary.setdefault(trigger, label)
        return primary

    def _classifier_fallback_label(self) -> str:
        configured = self.command_config.get("mode_classifier", {}).get("fallback_label")
        if configured:
            return configured
        return next(iter(self._classifier_label_to_trigger))

    def _resolve_trigger_route(self, trigger: str) -> TriggerRoute:
        try:
            return self._trigger_map[trigger]
        except KeyError:
            raise ValueError(f"Unknown trigger '{trigger}'") from None

    def _resolve_label_to_route(self, label: str) -> TriggerRoute:
        trigger = self._classifier_label_to_trigger.get(label)
        if trigger is None:
            fallback = self._classifier_fallback_label()
            logger.warning("Unknown classifier label '%s', using fallback '%s'", label, fallback)
            trigger = self._classifier_label_to_trigger[fallback]
        return self._resolve_trigger_route(trigger)

    def _mode_and_trigger_runtime(self, route: TriggerRoute) -> tuple[str, dict[str, Any], str]:
        mode_cfg = self.command_config["modes"][route.mode_key]
        reasoning_effort = route.overrides.get(
            "reasoning_effort", mode_cfg.get("reasoning_effort", "minimal")
        )
        runtime = {
            "reasoning_effort": reasoning_effort,
            "allowed_tools": route.overrides.get("allowed_tools", mode_cfg.get("allowed_tools")),
            "steering": route.overrides.get("steering", mode_cfg.get("steering", True)),
            "model": route.overrides.get("model"),
            "history_size": int(mode_cfg.get("history_size", self.command_config["history_size"])),
        }
        mode_label = self._primary_label_by_trigger.get(route.trigger, route.trigger)
        return route.mode_key, runtime, mode_label

    @staticmethod
    def _model_str_for_prompt(model: Any) -> str:
        if isinstance(model, list):
            if not model:
                return ""
            return model_str_core(model[0])
        return model_str_core(model)

    def _get_default_trigger_for_mode(self, mode_key: str) -> str:
        try:
            return self._default_trigger_by_mode[mode_key]
        except KeyError:
            raise ValueError(f"Channel mode '{mode_key}' has no configured trigger") from None

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
    def _steering_key(msg: RoomMessage) -> SteeringKey:
        # Non-threaded steering stays scoped to same sender.
        # In thread, steering is shared by thread participants.
        if msg.thread_id is not None:
            return (msg.arc, "*", msg.thread_id)
        return (msg.arc, msg.nick.lower(), None)

    @staticmethod
    def _steering_context_message(msg: RoomMessage) -> dict[str, str]:
        return {"role": "user", "content": f"<{msg.nick}> {msg.content}"}

    @staticmethod
    def _finish_queued_item(item: QueuedInboundMessage) -> None:
        if not item.completion.done():
            item.completion.set_result(None)

    @staticmethod
    def _fail_queued_item(item: QueuedInboundMessage, exc: BaseException) -> None:
        if not item.completion.done():
            item.completion.set_exception(exc)

    async def _enqueue_command_or_start_runner(
        self,
        msg: RoomMessage,
        trigger_message_id: int,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> tuple[bool, SteeringKey, QueuedInboundMessage]:
        item = QueuedInboundMessage(
            kind="command",
            msg=msg,
            trigger_message_id=trigger_message_id,
            reply_sender=reply_sender,
            completion=asyncio.get_running_loop().create_future(),
        )
        key = self._steering_key(msg)
        async with self._steering_lock:
            session = self._steering_sessions.get(key)
            if session is None:
                self._steering_sessions[key] = SteeringSession(queue=[])
                return True, key, item
            session.queue.append(item)
            return False, key, item

    async def _drain_steering_context_messages(self, key: SteeringKey) -> list[dict[str, str]]:
        """Drain currently queued inbound items into steering context."""
        async with self._steering_lock:
            session = self._steering_sessions.get(key)
            if session is None or not session.queue:
                return []
            drained = list(session.queue)
            session.queue.clear()

        for item in drained:
            self._finish_queued_item(item)

        return [self._steering_context_message(item.msg) for item in drained]

    async def _take_next_work_compacted(
        self, key: SteeringKey
    ) -> tuple[list[QueuedInboundMessage], QueuedInboundMessage | None]:
        """Take next work item while compacting passive queue noise.

        Policy:
          - If a command exists in queue: drop all passives before first command.
          - If no command exists: keep only the last passive.

        When nothing remains, the session is removed (closed).

        Returns: (dropped_items, next_item_or_None)
        """
        async with self._steering_lock:
            session = self._steering_sessions.get(key)
            if session is None:
                return [], None

            if not session.queue:
                self._steering_sessions.pop(key, None)
                return [], None

            queue = session.queue
            first_command_index = next(
                (i for i, item in enumerate(queue) if item.kind == "command"),
                None,
            )

            if first_command_index is not None:
                dropped = queue[:first_command_index]
                next_item = queue[first_command_index]
                session.queue = queue[first_command_index + 1 :]
                return dropped, next_item

            dropped = queue[:-1]
            next_item = queue[-1]
            session.queue = []
            return dropped, next_item

    async def _abort_steering_session(
        self, key: SteeringKey, exc: BaseException
    ) -> list[QueuedInboundMessage]:
        async with self._steering_lock:
            session = self._steering_sessions.pop(key, None)
            if session is None:
                return []
            remaining = list(session.queue)

        for item in remaining:
            self._fail_queued_item(item, exc)
        return remaining

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
        prompt_vars = self.room_config.get("prompt_vars", {})

        model_vars: dict[str, str] = {}
        for mode_key, mode_cfg in modes_config.items():
            model_value = (
                model_override if mode_key == mode and model_override else mode_cfg["model"]
            )
            model_vars[f"{mode_key}_model"] = self._model_str_for_prompt(model_value)

        # Compatibility placeholders commonly used by default prompts.
        model_vars.setdefault("sarcastic_model", model_vars.get("sarcastic_model", ""))
        model_vars.setdefault("serious_model", model_vars.get("serious_model", ""))
        model_vars.setdefault("unsafe_model", model_vars.get("unsafe_model", ""))

        thinking_trigger = self._trigger_map.get("!a")
        thinking_model = None
        if thinking_trigger is not None:
            thinking_model = thinking_trigger.overrides.get("model")
        model_vars.setdefault(
            "thinking_model",
            self._model_str_for_prompt(thinking_model)
            if thinking_model
            else model_vars.get("serious_model", ""),
        )

        return prompt_template.format(
            mynick=mynick,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            **model_vars,
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
            response_upper = response.upper()

            counts = {
                label: response_upper.count(label.upper())
                for label in self._classifier_label_to_trigger
            }
            best_label, best_count = max(counts.items(), key=lambda item: item[1])
            if best_count == 0:
                fallback = self._classifier_fallback_label()
                logger.warning("Invalid mode classification response: %s", response)
                return fallback
            return best_label
        except Exception as e:
            logger.error("Error classifying mode: %s", e)
            return self._classifier_fallback_label()

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
            classified_label = await self.classify_mode(context)
            classified_route = self._resolve_label_to_route(classified_label)
            classified_mode_key, classified_runtime, _ = self._mode_and_trigger_runtime(
                classified_route
            )
            if classified_mode_key != "serious":
                test_channels = self.agent.config.get("behavior", {}).get(
                    "proactive_interjecting_test", []
                )
                is_test_channel = test_channels and channel_key in test_channels
                mode_desc = "[TEST MODE] " if is_test_channel else ""
                logger.warning(
                    "%sProactive interjection suggested but not serious mode: %s (%s). Reason: %s",
                    mode_desc,
                    classified_label,
                    classified_route.trigger,
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
                reasoning_effort=classified_runtime["reasoning_effort"],
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
                    "Sending proactive agent (%s/%s) response to %s: %s",
                    classified_label,
                    classified_route.trigger,
                    msg.channel_name,
                    response_text,
                )
                await reply_sender(response_text)
                response_msg = dataclasses.replace(msg, nick=msg.mynick, content=response_text)
                await self.agent.history.add_message(response_msg, mode=classified_route.trigger)
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
        steering_message_provider: Callable[[], Awaitable[list[dict[str, str]]]] | None = None,
        **actor_kwargs,
    ) -> AgentResult | None:
        mode_cfg = self.command_config["modes"][mode].copy()
        if no_context:
            context = context[-1:]
            mode_cfg["include_chapter_summary"] = False
        elif mode_cfg.get("auto_reduce_context") and len(context) > 1:
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
                steering_message_provider=steering_message_provider,
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
        # Cancel pending proactive checks immediately when explicit command arrives.
        await self.proactive_debouncer.cancel_channel(
            self._get_proactive_channel_key(msg.server_tag, msg.channel_name)
        )

        if self._should_bypass_steering_queue(msg):
            await self._handle_command_core(
                msg,
                trigger_message_id,
                reply_sender,
                self._steering_key(msg),
            )
            return

        await self._run_or_queue_command(msg, trigger_message_id, reply_sender)

    def _should_bypass_steering_queue(self, msg: RoomMessage) -> bool:
        """Return True for commands that should never participate in steering queue."""
        parsed = self._parse_prefix(msg.content)
        if parsed.error or parsed.no_context:
            return True
        if parsed.mode_token == HELP_TOKEN:
            return True
        if parsed.mode_token is not None:
            route = self._resolve_trigger_route(parsed.mode_token)
            _, runtime, _ = self._mode_and_trigger_runtime(route)
            return not bool(runtime["steering"])

        channel_mode = self.get_channel_mode(msg.server_tag, msg.channel_name)
        if channel_mode in self.command_config["modes"]:
            default_trigger = self._get_default_trigger_for_mode(channel_mode)
            route = self._resolve_trigger_route(default_trigger)
            _, runtime, _ = self._mode_and_trigger_runtime(route)
            return not bool(runtime["steering"])
        return False

    async def _handle_command_core(
        self,
        msg: RoomMessage,
        trigger_message_id: int,
        reply_sender: Callable[[str], Awaitable[None]],
        steering_key: SteeringKey,
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
            steering_key=steering_key,
        )
        await self.proactive_debouncer.cancel_channel(
            self._get_proactive_channel_key(msg.server_tag, msg.channel_name)
        )
        await self.autochronicler.check_and_chronicle(
            msg.mynick, msg.server_tag, msg.channel_name, default_size
        )

    async def _run_or_queue_command(
        self,
        msg: RoomMessage,
        trigger_message_id: int,
        reply_sender: Callable[[str], Awaitable[None]],
    ) -> None:
        """Run a steering-queue command, becoming the session runner if first.

        Single loop processes both commands and compacted passive tails.
        The lock is held only for brief queue mutation inside
        _take_next_work_compacted(); long async handling runs unlocked,
        so new items may arrive while processing. The loop naturally
        re-checks after each item — no inner loop needed.
        """
        is_runner, steering_key, active_item = await self._enqueue_command_or_start_runner(
            msg,
            trigger_message_id,
            reply_sender,
        )

        if not is_runner:
            await active_item.completion
            return

        try:
            while active_item is not None:
                if active_item.kind == "command":
                    assert active_item.trigger_message_id is not None
                    await self._handle_command_core(
                        active_item.msg,
                        active_item.trigger_message_id,
                        active_item.reply_sender,
                        steering_key,
                    )
                else:
                    await self._handle_passive_message_core(
                        active_item.msg,
                        active_item.reply_sender,
                    )
                self._finish_queued_item(active_item)

                dropped, active_item = await self._take_next_work_compacted(steering_key)
                for item in dropped:
                    self._finish_queued_item(item)
        except Exception as e:
            await self._abort_steering_session(steering_key, e)
            if active_item is not None:
                self._fail_queued_item(active_item, e)
            raise

    def _parse_prefix(self, message: str) -> ParsedPrefix:
        """Parse leading modifier tokens from message.

        Recognized tokens (any order at start):
          - !c: no-context flag
          - configured mode triggers and !h
          - @modelid: model override

        Parsing stops at first non-modifier token. Only one trigger allowed.
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

            if tok in self._trigger_map or tok == HELP_TOKEN:
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
        key = self._steering_key(msg)

        async with self._steering_lock:
            session = self._steering_sessions.get(key)
            if session is None:
                queued_item = None
            else:
                queued_item = QueuedInboundMessage(
                    kind="passive",
                    msg=msg,
                    trigger_message_id=None,
                    reply_sender=reply_sender,
                    completion=asyncio.get_running_loop().create_future(),
                )
                session.queue.append(queued_item)

        if session is None:
            await self._handle_passive_message_core(msg, reply_sender)
            return

        assert queued_item is not None
        await queued_item.completion

    async def _handle_passive_message_core(
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
        steering_key: SteeringKey,
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

        if parsed.mode_token == HELP_TOKEN:
            logger.debug("Sending help message to %s", msg.nick)
            classifier_model = self.command_config["mode_classifier"]["model"]
            channel_mode = self.get_channel_mode(msg.server_tag, msg.channel_name)
            if channel_mode == "classifier":
                default_desc = f"automatic mode ({classifier_model} decides)"
            else:
                default_desc = f"{channel_mode} mode"

            mode_parts: list[str] = []
            for mode_key, mode_cfg in modes_config.items():
                trigger_list = list(mode_cfg.get("triggers", {}).keys())
                if not trigger_list:
                    continue
                model_desc = self._model_str_for_prompt(mode_cfg.get("model"))
                mode_parts.append(f"{'/'.join(trigger_list)} = {mode_key} ({model_desc})")

            help_msg = (
                f"default is {default_desc}; modes: {', '.join(mode_parts)}; "
                "use @modelid to override model; !c disables context"
            )
            await reply_sender(help_msg)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=help_msg)
            await self.agent.history.add_message(response_msg)
            return

        selected_route: TriggerRoute
        selected_label: str

        if parsed.mode_token:
            selected_route = self._resolve_trigger_route(parsed.mode_token)
            selected_label = self._primary_label_by_trigger.get(
                selected_route.trigger, selected_route.trigger
            )
            logger.debug(
                "Processing explicit trigger %s (%s) from %s: %s",
                selected_route.trigger,
                selected_route.mode_key,
                msg.nick,
                query_text,
            )
        else:
            logger.debug("Processing automatic mode request from %s: %s", msg.nick, query_text)
            channel_mode = self.get_channel_mode(msg.server_tag, msg.channel_name)
            if channel_mode == "serious":
                selected_label = await self.classify_mode(context[-default_size:])
                selected_route = self._resolve_label_to_route(selected_label)
                logger.debug(
                    "Auto-classified message as %s -> %s", selected_label, selected_route.trigger
                )
                if selected_route.mode_key == "sarcastic":
                    selected_route = self._resolve_trigger_route(
                        self._get_default_trigger_for_mode("serious")
                    )
                    selected_label = self._primary_label_by_trigger.get(
                        selected_route.trigger, selected_route.trigger
                    )
                    logger.debug(
                        "...but forcing channel-configured serious mode for %s", msg.channel_name
                    )
            elif channel_mode in {"sarcastic", "unsafe"}:
                selected_route = self._resolve_trigger_route(
                    self._get_default_trigger_for_mode(channel_mode)
                )
                selected_label = self._primary_label_by_trigger.get(
                    selected_route.trigger, selected_route.trigger
                )
                logger.debug(
                    "Using channel-configured %s mode for %s",
                    channel_mode,
                    msg.channel_name,
                )
            elif channel_mode in modes_config:
                selected_route = self._resolve_trigger_route(
                    self._get_default_trigger_for_mode(channel_mode)
                )
                selected_label = self._primary_label_by_trigger.get(
                    selected_route.trigger, selected_route.trigger
                )
                logger.debug(
                    "Using channel-configured custom mode %s for %s",
                    channel_mode,
                    msg.channel_name,
                )
            else:
                selected_label = await self.classify_mode(context)
                selected_route = self._resolve_label_to_route(selected_label)
                logger.debug(
                    "Auto-classified message as %s -> %s", selected_label, selected_route.trigger
                )

        mode_key, runtime, mode_label = self._mode_and_trigger_runtime(selected_route)
        if selected_label == selected_route.trigger:
            selected_label = mode_label

        steering_enabled = bool(runtime["steering"]) and not no_context

        async def steering_message_provider() -> list[dict[str, str]]:
            if not steering_enabled:
                return []
            return await self._drain_steering_context_messages(steering_key)

        async def progress_cb(text: str) -> None:
            await reply_sender(text)
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=text)
            await self.agent.history.add_message(response_msg)

        async def persistence_cb(text: str) -> None:
            response_msg = dataclasses.replace(msg, nick=msg.mynick, content=text)
            await self.agent.history.add_message(
                response_msg, content_template="[internal monologue] {message}"
            )

        run_kwargs: dict[str, Any] = {
            "mode": mode_key,
            "reasoning_effort": runtime["reasoning_effort"],
            "progress_callback": progress_cb,
            "persistence_callback": persistence_cb,
            "arc": msg.arc,
            "no_context": no_context,
            "model": model_override or runtime["model"],
            "secrets": msg.secrets,
            "steering_message_provider": steering_message_provider,
        }
        if runtime["allowed_tools"] is not None:
            run_kwargs["allowed_tools"] = runtime["allowed_tools"]

        agent_result = await self._run_actor(
            context[-runtime["history_size"] :],
            msg.mynick,
            **run_kwargs,
        )

        if agent_result and agent_result.text:
            response_text = self._clean_response_text(agent_result.text, msg.nick)
            cost_str = f"${agent_result.total_cost:.4f}" if agent_result.total_cost else "?"
            logger.info(
                "Sending %s/%s response (%s) to %s: %s",
                selected_label,
                selected_route.trigger,
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
                response_msg,
                mode=selected_route.trigger,
                llm_call_id=llm_call_id,
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
            logger.info(
                "Agent in %s/%s mode chose not to answer for %s",
                selected_label,
                selected_route.trigger,
                msg.channel_name,
            )
