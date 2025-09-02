"""IRC room monitor for handling IRC-specific message processing."""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...main import IRSSILLMAgent

from ...agentic_actor import AgenticLLMActor
from ...providers.perplexity import PerplexityClient
from ...rate_limiter import RateLimiter
from .. import ProactiveDebouncer
from .varlink import VarlinkClient, VarlinkSender

logger = logging.getLogger(__name__)


class IRCRoomMonitor:
    """IRC-specific room monitor that handles varlink connections and message processing."""

    def __init__(self, agent: "IRSSILLMAgent"):
        """Initialize IRC room monitor.

        Args:
            agent: Reference to the main IRSSILLMAgent instance for accessing shared resources
        """
        self.agent = agent

        # Get IRC config section
        irc_config = self.agent.config["rooms"]["irc"]
        self.varlink_events = VarlinkClient(irc_config["varlink"]["socket_path"])
        self.varlink_sender = VarlinkSender(irc_config["varlink"]["socket_path"])

        self.rate_limiter = RateLimiter(
            irc_config["command"]["rate_limit"], irc_config["command"]["rate_period"]
        )
        self.proactive_rate_limiter = RateLimiter(
            irc_config["proactive"]["rate_limit"],
            irc_config["proactive"]["rate_period"],
        )
        self.proactive_debouncer = ProactiveDebouncer(irc_config["proactive"]["debounce_seconds"])
        self.server_nicks: dict[str, str] = {}  # Cache of nicks per server

    @property
    def irc_config(self) -> dict[str, Any]:
        """Get IRC-specific configuration."""
        return self.agent.config["rooms"]["irc"]

    def should_ignore_user(self, nick: str) -> bool:
        """Check if user should be ignored."""
        ignore_list = self.irc_config["command"]["ignore_users"]
        return any(nick.lower() == ignored.lower() for ignored in ignore_list)

    def build_system_prompt(self, mode: str, mynick: str) -> str:
        """Build a command system prompt with standard substitutions.

        Args:
            mode: Command mode (e.g., "serious", "sarcastic")
            mynick: IRC nickname for substitution

        Returns:
            Formatted system prompt with all substitutions applied
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Get model configurations for context
        sarcastic_model = self.irc_config["command"]["modes"]["sarcastic"]["model"]
        serious_cfg = self.irc_config["command"]["modes"]["serious"]["model"]
        serious_model = (
            serious_cfg[0] if isinstance(serious_cfg, list) and serious_cfg else serious_cfg
        )
        unsafe_model = (
            self.irc_config["command"]["modes"].get("unsafe", {}).get("model", "not-configured")
        )

        # Get the prompt template from command section
        try:
            prompt_template = self.irc_config["command"]["modes"][mode]["prompt"]
        except KeyError:
            raise ValueError(f"Command mode '{mode}' not found in config") from None

        return prompt_template.format(
            mynick=mynick,
            current_time=current_time,
            sarcastic_model=sarcastic_model,
            serious_model=serious_model,
            unsafe_model=unsafe_model,
        )

    async def get_mynick(self, server: str) -> str | None:
        """Get bot's nick for a server."""
        if server not in self.server_nicks:
            try:
                nick = await self.varlink_sender.get_server_nick(server)
                if nick:
                    self.server_nicks[server] = nick
                    logger.debug(f"Got nick for {server}: {nick}")
                return nick
            except Exception as e:
                logger.error(f"Failed to get nick for server {server}: {e}")
                return None
        return self.server_nicks[server]

    def get_channel_mode(self, chan_name: str) -> str:
        """Get the configured mode for a channel, falling back to default_mode if not specified.

        Args:
            chan_name: Channel name to check

        Returns:
            Mode string: 'sarcastic', 'serious', or 'classifier'
        """
        channel_modes = self.irc_config["command"].get("channel_modes", {})
        if chan_name in channel_modes:
            return channel_modes[chan_name]

        default_mode = self.irc_config["command"].get("default_mode", "classifier")
        return default_mode

    async def classify_mode(self, context: list[dict[str, str]]) -> str:
        """Use preprocessing model to classify whether message should use sarcastic, serious, or unsafe mode.

        Args:
            context: Conversation context including the current message as the last entry
        """
        try:
            if not context:
                raise ValueError(
                    "Context cannot be empty - must include at least the current message"
                )

            # Extract the current message from context (should be the last message)
            current_message = context[-1]["content"]

            # Clean message content if it has IRC nick formatting like "<nick> message"
            message_match = re.search(r"<[^>]+>\s*(.*)", current_message)
            if message_match:
                current_message = message_match.group(1).strip()

            prompt = self.irc_config["command"]["mode_classifier"]["prompt"].format(
                message=current_message
            )
            model = self.irc_config["command"]["mode_classifier"]["model"]
            resp, client, _ = await self.agent.model_router.call_raw_with_model(
                model, context, prompt
            )
            response = client.extract_text_from_response(resp)

            serious_count = response.count("SERIOUS")
            sarcastic_count = response.count("SARCASTIC")
            unsafe_count = response.count("UNSAFE")

            # Check for UNSAFE first (highest priority for explicit requests)
            if unsafe_count > 0:
                return "UNSAFE"
            elif serious_count == 0 and sarcastic_count == 0:
                logger.warning(f"Invalid mode classification response: {response}")
                return "SARCASTIC"
            elif serious_count <= sarcastic_count:
                return "SARCASTIC"
            else:
                return (
                    "EASY_SERIOUS"
                    if response.count("EASY_SERIOUS") > response.count("THINKING_SERIOUS")
                    else "THINKING_SERIOUS"
                )
        except Exception as e:
            logger.error(f"Error classifying mode: {e}")
            return "SARCASTIC"  # Default to sarcastic on error

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

            # Extract the current message from context (should be the last message)
            current_message = context[-1]["content"]

            # Clean message content if it has IRC nick formatting like "<nick> message"
            message_match = re.search(r"<?\S+>\s*(.*)", current_message)
            if message_match:
                current_message = message_match.group(1).strip()

            # Use full context for better decision making, but specify the current message in prompt
            prompt = self.irc_config["proactive"]["prompts"]["interject"].format(
                message=current_message
            )

            # Get validation models list
            validation_models = self.irc_config["proactive"]["models"]["validation"]

            final_score = None
            all_responses = []

            # Run iterative validation through all models
            for i, model in enumerate(validation_models):
                resp, client, _ = await self.agent.model_router.call_raw_with_model(
                    model, context, prompt
                )
                response = client.extract_text_from_response(resp)

                if not response or response.startswith("API error:"):
                    return False, f"No response from validation model {i + 1}", False

                response = response.strip()
                all_responses.append(f"Model {i + 1} ({model}): {response}")

                # Extract the score from the response
                score_match = re.search(r"(\d+)/10", response)
                if not score_match:
                    logger.warning(
                        f"No valid score found in proactive interject response from model {i + 1}: {response}"
                    )
                    return False, f"No score found in validation step {i + 1}", False

                score = int(score_match.group(1))
                final_score = score

                logger.debug(
                    f"Proactive validation step {i + 1}/{len(validation_models)} - Model: {model}, Score: {score}"
                )

                threshold = self.irc_config["proactive"]["interject_threshold"]
                if score < threshold - 1:
                    if i > 0:
                        logger.info(
                            f"Proactive interjection rejected at step {i + 1}/{len(validation_models)} ({current_message[:150]}... Score: {score})"
                        )
                    else:
                        logger.debug(
                            f"Proactive interjection rejected at step {i + 1}/{len(validation_models)} (Score: {score})"
                        )
                    return (
                        False,
                        f"Rejected at validation step {i + 1} (Score: {score})",
                        False,
                    )

            if final_score is not None:
                threshold = self.irc_config["proactive"]["interject_threshold"]

                if final_score >= threshold:
                    logger.debug(
                        f"Proactive interjection triggered for message: {current_message[:150]}... (Final Score: {final_score})"
                    )
                    return True, f"Interjection decision (Final Score: {final_score})", False
                elif final_score >= threshold - 1:
                    logger.debug(
                        f"Proactive interjection BARELY triggered for message: {current_message[:150]}... (Final Score: {final_score}) - SWITCHING TO TEST MODE"
                    )
                    return True, f"Barely triggered - test mode (Final Score: {final_score})", True
                else:
                    return False, f"No interjection (Final Score: {final_score})", False
            else:
                return False, "No valid final score", False

        except Exception as e:
            logger.error(f"Error checking proactive interject: {e}")
            return False, f"Error: {str(e)}", False

    async def _handle_debounced_proactive_check(
        self, server: str, chan_name: str, nick: str, message: str, mynick: str
    ) -> None:
        """Handle a debounced proactive check with fresh context and rate limiting."""
        try:
            # Check proactive rate limit at execution time
            if not self.proactive_rate_limiter.check_limit():
                logger.debug(
                    f"Proactive interjection rate limit exceeded during debounced check, skipping message from {nick}"
                )
                return

            # Get fresh context at check time with proactive history size
            context = await self.agent.history.get_context(
                server, chan_name, self.irc_config["proactive"]["history_size"]
            )
            should_interject, reason, forced_test_mode = await self.should_interject_proactively(
                context
            )
            if not should_interject:
                return

            classified_mode = await self.classify_mode(context)
            if not classified_mode.endswith("SERIOUS"):
                # Check if this is a test channel for logging
                test_channels = self.agent.config.get("behavior", {}).get(
                    "proactive_interjecting_test", []
                )
                is_test_channel = test_channels and chan_name in test_channels
                mode_desc = "[TEST MODE] " if is_test_channel else ""
                logger.warning(
                    f"{mode_desc}Proactive interjection suggested but not serious mode: {classified_mode}. Reason: {reason}"
                )
                return

            test_channels = self.agent.config.get("behavior", {}).get(
                "proactive_interjecting_test", []
            )
            is_test_channel = test_channels and chan_name in test_channels
            if is_test_channel or forced_test_mode:
                test_reason = "[BARELY TRIGGERED]" if forced_test_mode else "[TEST CHANNEL]"
                logger.info(
                    f"[TEST MODE] {test_reason} Would interject proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                )
                send_message = False
            else:
                logger.info(
                    f"Interjecting proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                )
                send_message = True

            target = chan_name  # For proactive interjections, target is the channel
            arc = f"{server}#{chan_name}"
            response = await self._run_actor(
                context,
                mynick,
                mode="serious",
                reasoning_effort="low" if classified_mode == "EASY_SERIOUS" else "medium",
                model=self.irc_config["proactive"]["models"]["serious"],
                extra_prompt=" " + self.irc_config["proactive"]["prompts"]["serious_extra"],
                arc=arc,
            )

            # Check for NULL response (proactive interjections can decide not to respond)
            # Also filter out "Error - " responses which are proactive-specific
            if not response or response.startswith("Error - "):
                logger.info(f"Agent decided not to interject proactively for {target}")
                return

            if send_message:
                logger.info(
                    f"Sending proactive agent ({classified_mode}) response to {target}: {response}"
                )
                await self.varlink_sender.send_message(target, response, server)
                await self.agent.history.add_message(
                    server, chan_name, response, mynick, mynick, True
                )
            else:
                logger.info(f"[TEST MODE] Generated proactive response for {target}: {response}")
        except Exception as e:
            logger.error(f"Error in debounced proactive check for {chan_name}: {e}")

    async def _run_actor(
        self,
        context: list[dict[str, str]],
        mynick: str,
        *,
        mode: str,
        extra_prompt: str = "",
        progress_callback=None,
        **actor_kwargs,
    ) -> str | None:
        """Unified actor runner that returns cleaned IRC response."""

        # Determine model to use if not provided
        model = actor_kwargs.pop("model", None)
        if model is None:
            model = self.irc_config["command"]["modes"][mode]["model"]

        # Extract arc for run_agent call
        arc = actor_kwargs.pop("arc", "")

        async with AgenticLLMActor(
            config=self.agent.config,
            model=model,
            system_prompt_generator=lambda: self.build_system_prompt(mode, mynick) + extra_prompt,
            prompt_reminder_generator=lambda: self.irc_config["command"]["modes"][mode].get(
                "prompt_reminder"
            ),
            agent=self.agent,
            **actor_kwargs,
        ) as actor:
            response = await actor.run_agent(context, progress_callback=progress_callback, arc=arc)

        if not response or response.strip().upper() == "NULL":
            return None

        # IRC-specific response cleaning: remove timestamp/nick prefixes that LLM might include
        response = response.strip()
        response = re.sub(r"^(\s*(\[?\d{1,2}:\d{2}\]?\s*)?<[^>]+>)*\s*", "", response)

        return response

    def _input_match(self, mynick, message):
        pattern = rf"^\s*(<?.*?>\s*)?{re.escape(mynick)}[,:]\s*(.*?)$"
        return re.match(pattern, message, re.IGNORECASE)

    async def process_message_event(self, event: dict[str, Any]) -> None:
        """Process incoming IRC message events."""
        msg_type = event.get("type")
        subtype = event.get("subtype")
        server = event.get("server")
        target = event.get("target")
        nick = event.get("nick")
        message = event.get("message", "")

        logger.debug(f"Processing message event: {event}")

        if msg_type != "message" or not all([server, target, nick, message]):
            logger.debug("Skipping invalid message event")
            return

        # Type assertions after validation
        assert isinstance(server, str)
        assert isinstance(target, str)
        assert isinstance(nick, str)
        assert isinstance(message, str)

        # Use target for public messages, nick for private messages
        chan_name = target if subtype == "public" else nick

        # Get our nick for this server
        mynick = await self.get_mynick(server)
        if not mynick:
            return

        # Skip ignored users
        if self.should_ignore_user(nick):
            logger.debug(f"Ignoring user: {nick}")
            return

        # Check if message is addressing us directly
        match = self._input_match(mynick, message)
        is_private = subtype != "public"

        if match or is_private:
            if match and match.group(1):
                nick = match.group(1).strip("<> ")

            await self.proactive_debouncer.cancel_channel(chan_name)
            # Call history.add_message() only later in handle_command() so that
            # various testcases etc. calling it directly don't have to worry
            # about the context setup.
            await self.handle_command(server, chan_name, target, nick, message, mynick)
            return

        await self.agent.history.add_message(server, chan_name, message, nick, mynick)
        if (
            chan_name
            in self.irc_config["proactive"]["interjecting"]
            + self.irc_config["proactive"]["interjecting_test"]
        ):
            await self.proactive_debouncer.schedule_check(
                server, chan_name, nick, message, mynick, self._handle_debounced_proactive_check
            )

    async def handle_command(
        self, server: str, chan_name: str, target: str, nick: str, message: str, mynick: str
    ) -> None:
        """Handle IRC commands and generate responses."""
        await self.agent.history.add_message(server, chan_name, message, nick, mynick)

        if not self.rate_limiter.check_limit():
            logger.warning(f"Rate limiting triggered for {nick}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Slow down a little, will you? (rate limiting)", server
            )
            return

        # Extract cleaned message
        match = self._input_match(mynick, message)
        cleaned_msg = match.group(2) if match else message
        logger.info(f"Received command from {nick} on {server}/{chan_name}: {cleaned_msg}")

        # Work with fixed context from now on to avoid debouncing/classification races!
        default_size = self.irc_config["command"]["history_size"]
        max_size = max(
            default_size,
            *(mode.get("history_size", 0) for mode in self.irc_config["command"]["modes"].values()),
        )
        context = await self.agent.history.get_context(server, chan_name, max_size)

        # Debounce briefly to consolidate quick followups e.g. due to automatic IRC message splits
        debounce = self.irc_config["command"].get("debounce", 0)
        if debounce > 0:
            original_timestamp = time.time()
            await asyncio.sleep(debounce)

            followups = await self.agent.history.get_recent_messages_since(
                server, chan_name, nick, original_timestamp
            )
            if followups:
                logger.debug(f"Debounced {len(followups)} followup messages from {nick}")
                context[-1]["content"] += "\n" + "\n".join([m["message"] for m in followups])

        await self._route_command(
            server, chan_name, target, nick, mynick, cleaned_msg, context, default_size
        )
        await self.proactive_debouncer.cancel_channel(chan_name)

    async def _route_command(
        self,
        server: str,
        chan_name: str,
        target: str,
        nick: str,
        mynick: str,
        cleaned_msg: str,
        context: list[dict],
        default_size: int,
    ) -> None:
        """Route commands based on message content with prepared context."""
        if cleaned_msg.startswith("!h") or cleaned_msg == "!h":
            logger.debug(f"Sending help message to {nick}")
            sarcastic_model = self.irc_config["command"]["modes"]["sarcastic"]["model"]
            serious_model = self.irc_config["command"]["modes"]["serious"]["model"]
            unsafe_model = (
                self.irc_config["command"]["modes"].get("unsafe", {}).get("model", "not-configured")
            )
            classifier_model = self.irc_config["command"]["mode_classifier"]["model"]

            channel_mode = self.get_channel_mode(chan_name)
            if channel_mode == "serious":
                default_desc = f"serious agentic mode with web tools ({serious_model}), !d is explicit sarcastic diss mode ({sarcastic_model}), !u is unsafe mode ({unsafe_model}), !a forces thinking"
            elif channel_mode == "sarcastic":
                default_desc = f"sarcastic mode ({sarcastic_model}), !s (quick) & !a (thinking) is serious agentic mode with web tools ({serious_model}), !u is unsafe mode ({unsafe_model})"
            else:
                default_desc = f"automatic mode ({classifier_model} decides), !d is explicit sarcastic diss mode ({sarcastic_model}), !s (quick) & !a (thinking) is serious agentic mode with web tools ({serious_model}), !u is unsafe mode ({unsafe_model})"

            help_msg = f"default is {default_desc}, !p is Perplexity (prefer English!)"
            await self.varlink_sender.send_message(target, help_msg, server)
            return

        if cleaned_msg.startswith("!p ") or cleaned_msg == "!p":
            # Perplexity call
            logger.debug(f"Processing Perplexity request from {nick}: {cleaned_msg}")
            # Use default history size for Perplexity
            async with PerplexityClient(self.agent.config) as perplexity:
                response = await perplexity.call_perplexity(context[-default_size:])
            if response:
                # Handle multi-line responses (citations)
                lines = response.split("\n")
                for line in lines:
                    if line.strip():
                        logger.info(f"Sending Perplexity response to {target}: {line.strip()}")
                        await self.varlink_sender.send_message(
                            target, f"{nick}: {line.strip()}", server
                        )

                # Update context with response
                await self.agent.history.add_message(
                    server, chan_name, lines[0], mynick, mynick, True
                )
            return

        # Determine mode from explicit commands or auto-classification
        mode = None
        reasoning_effort = "minimal"

        if cleaned_msg.startswith("!S ") or cleaned_msg.startswith("!d "):
            logger.debug(f"Processing explicit sarcastic request from {nick}: {cleaned_msg}")
            mode = "SARCASTIC"
        elif cleaned_msg.startswith("!D "):  # easter egg - thinking sarcastic
            logger.debug(
                f"Processing explicit thinking sarcastic request from {nick}: {cleaned_msg}"
            )
            mode = "SARCASTIC"
            reasoning_effort = "high"
        elif cleaned_msg.startswith("!s "):
            logger.debug(f"Processing explicit serious request from {nick}: {cleaned_msg}")
            mode = "EASY_SERIOUS"
        elif cleaned_msg.startswith("!a "):
            logger.debug(f"Processing explicit agentic request from {nick}: {cleaned_msg}")
            mode = "THINKING_SERIOUS"
        elif cleaned_msg.startswith("!u "):
            logger.debug(f"Processing explicit unsafe request from {nick}: {cleaned_msg}")
            mode = "UNSAFE"
        elif re.match(r"^!.", cleaned_msg):
            logger.warning(f"Unknown command from {nick}: {cleaned_msg}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Unknown command. Use !h for help.", server
            )
            return
        else:
            logger.debug(f"Processing automatic mode request from {nick}: {cleaned_msg}")

            # Check for channel-specific or default mode override
            channel_mode = self.get_channel_mode(chan_name)
            if channel_mode == "serious":
                mode = await self.classify_mode(context[-default_size:])
                logger.debug(f"Auto-classified message as {mode} mode")
                if mode == "SARCASTIC":
                    mode = "EASY_SERIOUS"
                    logger.debug(f"...but forcing channel-configured serious mode for {chan_name}")
            elif channel_mode == "sarcastic":
                mode = "SARCASTIC"
                logger.debug(f"Using channel-configured sarcastic mode for {chan_name}")
            else:
                mode = await self.classify_mode(context)
                logger.debug(f"Auto-classified message as {mode} mode")

        # Create progress callback for command mode
        async def progress_cb(text: str) -> None:
            await self.varlink_sender.send_message(target, text, server)
            await self.agent.history.add_message(server, chan_name, text, mynick, mynick, True)

        if mode == "SARCASTIC":
            sarcastic_size = self.irc_config["command"]["modes"]["sarcastic"].get(
                "history_size", default_size
            )
            response = await self._run_actor(
                context[-sarcastic_size:],
                mynick,
                mode="sarcastic",
                reasoning_effort=reasoning_effort,
                allowed_tools=[],
                progress_callback=progress_cb,
            )
        elif mode and mode.endswith("SERIOUS"):
            serious_size = self.irc_config["command"]["modes"]["serious"].get(
                "history_size", default_size
            )
            assert (
                reasoning_effort == "minimal"
            )  # test we didn't override it earlier since we ignore it here
            # Pass arc to actor for chronicler context
            arc = f"{server}#{chan_name}"
            response = await self._run_actor(
                context[-serious_size:],
                mynick,
                mode="serious",
                reasoning_effort="low" if mode == "EASY_SERIOUS" else "medium",
                progress_callback=progress_cb,
                arc=arc,
            )
        elif mode == "UNSAFE":
            unsafe_size = self.irc_config["command"]["modes"]["unsafe"].get(
                "history_size", default_size
            )
            # Pass arc to actor for chronicler context
            arc = f"{server}#{chan_name}"
            response = await self._run_actor(
                context[-unsafe_size:],
                mynick,
                mode="unsafe",
                reasoning_effort=reasoning_effort,
                progress_callback=progress_cb,
                arc=arc,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        # Send response if we got one
        if response:
            logger.info(f"Sending {mode} response to {target}: {response}")
            await self.varlink_sender.send_message(target, response, server)
            await self.agent.history.add_message(server, chan_name, response, mynick, mynick, True)
        else:
            logger.info(f"Agent in {mode} mode chose not to answer for {target}")

    async def _connect_with_retry(self, max_retries: int = 5) -> bool:
        """Connect to varlink sockets with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                await self.varlink_events.connect()
                await self.varlink_sender.connect()
                await self.varlink_events.wait_for_events()
                logger.info("Successfully connected to varlink sockets")
                return True
            except Exception as e:
                wait_time = 2**attempt
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    return False
        return False

    async def run(self) -> None:
        """Run the main IRC monitor loop."""
        try:
            # Connect with retry logic
            if not await self._connect_with_retry():
                logger.error("Could not establish connection, exiting")
                return

            logger.info("irssi-llmagent started, waiting for IRC events...")

            while True:
                try:
                    response = await self.varlink_events.receive_response()
                    if response is None:
                        logger.warning("Connection lost, attempting to reconnect...")
                        await self.varlink_events.disconnect()
                        await self.varlink_sender.disconnect()

                        if await self._connect_with_retry():
                            logger.info("Reconnected successfully")
                            continue
                        else:
                            logger.error("Failed to reconnect, exiting...")
                            break

                    if "parameters" in response and "event" in response["parameters"]:
                        event = response["parameters"]["event"]
                        # Process events concurrently
                        task = asyncio.create_task(self.process_message_event(event))
                        task.add_done_callback(
                            lambda t: t.exception()
                            and logger.error(f"Event processing task failed: {t.exception()}")
                        )
                    elif "error" in response:
                        logger.error(f"Varlink error: {response['error']}")
                        break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)  # Brief pause before continuing

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            logger.info("Disconnecting from varlink sockets...")
            await self.varlink_events.disconnect()
            await self.varlink_sender.disconnect()
            await self.proactive_debouncer.cancel_all()
            logger.info("IRC monitor stopped")
