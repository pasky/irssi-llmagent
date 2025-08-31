"""Main application entry point for irssi-llmagent."""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from .agent import AIAgent
from .history import ChatHistory
from .proactive_debouncer import ProactiveDebouncer
from .providers import ModelRouter
from .providers.perplexity import PerplexityClient
from .rate_limiter import RateLimiter
from .varlink import VarlinkClient, VarlinkSender

# Set up logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler for INFO and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler for DEBUG and above
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Suppress noisy third-party library messages
logging.getLogger("aiosqlite").setLevel(logging.INFO)
logging.getLogger("e2b.api").setLevel(logging.WARNING)
logging.getLogger("e2b.sandbox_sync").setLevel(logging.WARNING)
logging.getLogger("e2b.sandbox_sync.main").setLevel(logging.WARNING)
logging.getLogger("e2b_code_interpreter.code_interpreter_sync").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class IRSSILLMAgent:
    """Main IRC LLM agent application."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.model_router: ModelRouter | None = None
        # Get IRC config section
        irc_config = self.config["rooms"]["irc"]
        self.varlink_events = VarlinkClient(irc_config["varlink"]["socket_path"])
        self.varlink_sender = VarlinkSender(irc_config["varlink"]["socket_path"])
        self.history = ChatHistory(
            self.config.get("database", {}).get("path", "chat_history.db"),
            irc_config["command"]["history_size"],
        )
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
        return self.config["rooms"]["irc"]

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
                logger.debug(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(
                f"Config file {config_path} not found. "
                "Copy config.json.example to config.json and configure."
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            sys.exit(1)

    def should_ignore_user(self, nick: str) -> bool:
        """Check if user should be ignored."""
        ignore_list = self.irc_config["command"]["ignore_users"]
        return any(nick.lower() == ignored.lower() for ignored in ignore_list)

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
        """Use preprocessing model to classify whether message should use sarcastic or serious mode.

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
            # Lazy-init router on first use
            if self.model_router is None:
                self.model_router = await ModelRouter(self.config).__aenter__()
            resp, client, _ = await self.model_router.call_raw_with_model(model, context, prompt)
            response = client.extract_text_from_response(resp)

            serious_count = response.count("SERIOUS")
            sarcastic_count = response.count("SARCASTIC")

            if serious_count == 0 and sarcastic_count == 0:
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
            if self.model_router is None:
                self.model_router = await ModelRouter(self.config).__aenter__()

            final_score = None
            all_responses = []

            # Run iterative validation through all models
            for i, model in enumerate(validation_models):
                resp, client, _ = await self.model_router.call_raw_with_model(
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
            context = await self.history.get_context(
                server, chan_name, self.irc_config["proactive"]["history_size"]
            )
            should_interject, reason, forced_test_mode = await self.should_interject_proactively(
                context
            )

            if should_interject:
                # Classify the mode for proactive response (should be serious mode only)
                classified_mode = await self.classify_mode(context)
                if classified_mode.endswith("SERIOUS"):
                    # Check if this is a test channel or forced test mode due to low score
                    test_channels = self.config.get("behavior", {}).get(
                        "proactive_interjecting_test", []
                    )
                    is_test_channel = test_channels and chan_name in test_channels

                    # Combine test channel check with forced test mode
                    should_use_test_mode = is_test_channel or forced_test_mode

                    target = chan_name  # For proactive interjections, target is the channel

                    if should_use_test_mode:
                        # Test mode: use the same method as live mode but don't send messages
                        test_reason = "[BARELY TRIGGERED]" if forced_test_mode else "[TEST CHANNEL]"
                        logger.info(
                            f"[TEST MODE] {test_reason} Would interject proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                        )
                        proactive_context = await self.history.get_context(server, chan_name)
                        await self._handle_serious_mode(
                            server,
                            chan_name,
                            target,
                            nick,
                            message,
                            mynick,
                            classified_mode,
                            proactive_context,
                            is_proactive=True,
                            send_message=False,
                        )
                    else:
                        # Live mode: actually send the response
                        logger.info(
                            f"Interjecting proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                        )
                        proactive_context = await self.history.get_context(server, chan_name)
                        await self._handle_serious_mode(
                            server,
                            chan_name,
                            target,
                            nick,
                            message,
                            mynick,
                            classified_mode,
                            proactive_context,
                            is_proactive=True,
                        )
                else:
                    # Check if this is a test channel for logging
                    test_channels = self.config.get("behavior", {}).get(
                        "proactive_interjecting_test", []
                    )
                    is_test_channel = test_channels and chan_name in test_channels
                    mode_desc = "[TEST MODE] " if is_test_channel else ""
                    logger.warning(
                        f"{mode_desc}Proactive interjection suggested but not serious mode: {classified_mode}. Reason: {reason}"
                    )
        except Exception as e:
            logger.error(f"Error in debounced proactive check for {chan_name}: {e}")

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

        # Update history before checking if we should respond
        await self.history.add_message(server, chan_name, message, nick, mynick)

        # Check if message is addressing us directly
        pattern = rf"^\s*(<.*?>\s*)?{re.escape(mynick)}[,:]\s*(.*?)$"
        match = re.match(pattern, message, re.IGNORECASE)

        # For private messages, treat all messages as commands
        is_private = subtype != "public"

        # If not directly addressed and not a private message, check for proactive interjecting
        if not match and not is_private:
            # Check both live and test channels for proactive interjecting
            proactive_channels = self.irc_config["proactive"]["interjecting"]
            test_channels = self.irc_config["proactive"]["interjecting_test"]
            is_live_channel = proactive_channels and chan_name in proactive_channels
            is_test_channel = test_channels and chan_name in test_channels

            if is_live_channel or is_test_channel:
                # Schedule debounced proactive check
                await self.proactive_debouncer.schedule_check(
                    server, chan_name, nick, message, mynick, self._handle_debounced_proactive_check
                )
            return

        # Process command if directly addressed or if it's a private message
        if not match and not is_private:
            return

        # For private messages, use entire message; for channel messages, extract after nick prefix
        if is_private:
            cleaned_msg = message
        else:
            if match and match.group(1):
                nick = match.group(1).strip("<> ")
            cleaned_msg = match.group(2) if match else message
        logger.info(f"Received command from {nick} on {server}/{chan_name}: {cleaned_msg}")

        # Cancel any pending proactive interjection for this channel since we're processing a command
        await self.proactive_debouncer.cancel_channel(chan_name)

        # Check rate limiting
        if not self.rate_limiter.check_limit():
            logger.warning(f"Rate limiting triggered for {nick}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Slow down a little, will you? (rate limiting)", server
            )
            return

        # Process commands and generate response
        await self.handle_command(server, chan_name, target, nick, cleaned_msg, mynick)

    async def handle_command(
        self, server: str, chan_name: str, target: str, nick: str, message: str, mynick: str
    ) -> None:
        """Handle IRC commands and generate responses."""
        import time

        debounce = self.irc_config["command"].get("debounce", 0)
        if debounce > 0:
            original_timestamp = time.time()
            context = await self.history.get_context(server, chan_name)

            await asyncio.sleep(debounce)

            followups = await self.history.get_recent_messages_since(
                server, chan_name, nick, original_timestamp
            )
            if followups:
                logger.debug(f"Debounced {len(followups)} followup messages from {nick}")
                followup_texts = [m["message"] for m in followups]
                message = message + "\n" + "\n".join(followup_texts)
                context[-1]["content"] = message

        else:
            context = await self.history.get_context(server, chan_name)

        if message.startswith("!h") or message == "!h":
            logger.debug(f"Sending help message to {nick}")
            sarcastic_model = self.irc_config["command"]["modes"]["sarcastic"]["model"]
            serious_model = self.irc_config["command"]["modes"]["serious"]["model"]
            classifier_model = self.irc_config["command"]["mode_classifier"]["model"]

            channel_mode = self.get_channel_mode(chan_name)
            if channel_mode == "serious":
                default_desc = f"serious agentic mode with web tools ({serious_model}), !d is explicit sarcastic diss mode ({sarcastic_model}), !a forces thinking"
            elif channel_mode == "sarcastic":
                default_desc = f"sarcastic mode ({sarcastic_model}), !s (quick) & !a (thinking) is serious agentic mode with web tools ({serious_model})"
            else:
                default_desc = f"automatic mode ({classifier_model} decides), !d is explicit sarcastic diss mode ({sarcastic_model}), !s (quick) & !a (thinking) is serious agentic mode with web tools ({serious_model})"

            help_msg = f"default is {default_desc}, !p is Perplexity (prefer English!)"
            await self.varlink_sender.send_message(target, help_msg, server)

        elif message.startswith("!p ") or message == "!p":
            # Perplexity call
            logger.debug(f"Processing Perplexity request from {nick}: {message}")
            async with PerplexityClient(self.config) as perplexity:
                response = await perplexity.call_perplexity(context)
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
                await self.history.add_message(server, chan_name, lines[0], mynick, mynick, True)

        elif message.startswith("!S ") or message.startswith("!d "):
            message = message[3:].strip()
            logger.debug(f"Processing explicit sarcastic request from {nick}: {message}")
            await self._handle_sarcastic_mode(
                server, chan_name, target, nick, message, mynick, context
            )

        elif message.startswith("!D "):  # easter egg
            message = message[3:].strip()
            logger.debug(f"Processing explicit thinking sarcastic request from {nick}: {message}")
            await self._handle_sarcastic_mode(
                server, chan_name, target, nick, message, mynick, context, "high"
            )

        elif message.startswith("!s "):
            message = message[3:].strip()
            logger.debug(f"Processing explicit serious request from {nick}: {message}")
            await self._handle_serious_mode(
                server, chan_name, target, nick, message, mynick, "EASY_SERIOUS", context
            )

        elif message.startswith("!a "):
            message = message[3:].strip()
            logger.debug(f"Processing explicit agentic request from {nick}: {message}")
            await self._handle_serious_mode(
                server, chan_name, target, nick, message, mynick, "THINKING_SERIOUS", context
            )

        elif re.match(r"^!.", message):
            logger.warning(f"Unknown command from {nick}: {message}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Unknown command. Use !h for help.", server
            )

        else:
            logger.debug(f"Processing automatic mode request from {nick}: {message}")

            # Check for channel-specific or default mode override
            channel_mode = self.get_channel_mode(chan_name)
            if channel_mode == "serious":
                classified_mode = await self.classify_mode(context)
                logger.debug(f"Auto-classified message as {classified_mode} mode")
                if classified_mode == "SARCASTIC":
                    classified_mode = "EASY_SERIOUS"
                    logger.debug(f"...but forcing channel-configured serious mode for {chan_name}")
            elif channel_mode == "sarcastic":
                classified_mode = "SARCASTIC"
                logger.debug(f"Using channel-configured sarcastic mode for {chan_name}")
            else:
                classified_mode = await self.classify_mode(context)
                logger.debug(f"Auto-classified message as {classified_mode} mode")

            if classified_mode.endswith("SERIOUS"):
                await self._handle_serious_mode(
                    server, chan_name, target, nick, message, mynick, classified_mode, context
                )
            else:
                await self._handle_sarcastic_mode(
                    server, chan_name, target, nick, message, mynick, context
                )

        # Cancel any pending proactive interjection for this channel AGAIN, as
        # we might have queued up another one if we received a message while
        # the last command was being processed.
        await self.proactive_debouncer.cancel_channel(chan_name)

    async def _handle_serious_mode(
        self,
        server: str,
        chan_name: str,
        target: str,
        nick: str,
        message: str,
        mynick: str,
        classified_mode: str,
        context: list[dict[str, str]],
        is_proactive: bool = False,
        send_message: bool = True,
    ) -> str | None:
        """Handle serious mode using an agent with tools."""

        # Add extra prompt for proactive interjections to allow null response
        extra_prompt = ""
        if is_proactive:
            extra_prompt = " " + self.irc_config["proactive"]["prompts"]["serious_extra"]

        # Use configured proactive serious model for proactive interjections
        model_override = self.irc_config["proactive"]["models"]["serious"] if is_proactive else None
        # Build progress callback only for non-proactive mode
        from collections.abc import Awaitable, Callable

        progress_cb_fn: Callable[[str], Awaitable[None]] | None = None
        if not is_proactive:

            async def _progress_cb(text: str) -> None:
                # Send to channel as a normal message and store in history
                await self.varlink_sender.send_message(target, text, server)
                await self.history.add_message(server, chan_name, text, mynick, mynick, True)

            progress_cb_fn = _progress_cb
        async with AIAgent(
            self.config,
            mynick,
            mode="serious",
            extra_prompt=extra_prompt,
            model_override=model_override,
        ) as agent:
            # Configure progress after entering (avoid changing call signature used in tests)
            if not is_proactive:
                agent.configure_progress(True, progress_cb_fn)
            response = await agent.run_agent(
                context, reasoning_effort="low" if classified_mode == "EASY_SERIOUS" else "medium"
            )

        # For proactive interjections, check for NULL response
        if (
            is_proactive
            and response
            and (response.strip().upper() == "NULL" or response.startswith("Error - "))
        ):
            logger.info(f"Agent decided not to interject proactively for {target}")
            return None

        if response:
            if send_message:
                logger.info(f"Sending agent ({classified_mode}) response to {target}: {response}")
                await self.varlink_sender.send_message(target, response, server)
                # Update context with response
                await self.history.add_message(server, chan_name, response, mynick, mynick, True)
            else:
                logger.info(f"[TEST MODE] Generated response for {target}: {response}")

        return response

    async def _handle_sarcastic_mode(
        self,
        server: str,
        chan_name: str,
        target: str,
        nick: str,
        message: str,
        mynick: str,
        context: list[dict[str, str]],
        reasoning_effort: str = "minimal",
    ) -> None:
        """Handle sarcastic mode using AIAgent with limited tools."""
        sarcastic_model = self.irc_config["command"]["modes"]["sarcastic"]["model"]

        async with AIAgent(
            self.config,
            mynick,
            mode="sarcastic",
            model_override=sarcastic_model,
            allowed_tools=["visit_webpage"],
        ) as agent:
            response = await agent.run_agent(context, reasoning_effort=reasoning_effort)

        if response:
            logger.info(f"Sending sarcastic response to {target}: {response}")
            await self.varlink_sender.send_message(target, response, server)
            # Update context with response
            await self.history.add_message(server, chan_name, response, mynick, mynick, True)

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
        """Run the main agent loop."""
        try:
            # Initialize database
            await self.history.initialize()

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
            await self.history.close()
            logger.info("irssi-llmagent stopped")


async def cli_mode(message: str, config_path: str | None = None) -> None:
    """CLI mode for testing message handling including command parsing."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    print(f"🤖 Simulating IRC message: {message}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = IRSSILLMAgent(str(config_file))

        # Mock the varlink sender and history
        class MockSender:
            async def send_message(self, target: str, message: str, server: str):
                print(f"📤 Bot response: {message}")

        # Use real history for better testing
        from irssi_llmagent.history import ChatHistory

        history = ChatHistory("chat_history.db", inference_limit=20)

        # Add the current message to history
        await history.add_message("testserver", "#testchannel", message, "testuser", "testbot")

        class MockHistory:
            async def add_message(
                self, server: str, channel: str, content: str, nick: str, mynick: str, is_bot: bool
            ):
                await history.add_message(server, channel, content, nick, mynick, is_bot)

            async def get_context(self, server: str, channel: str):
                return await history.get_context(server, channel)

            async def get_recent_messages_since(
                self, server_tag: str, channel_name: str, nick: str, timestamp: float
            ):
                return await history.get_recent_messages_since(
                    server_tag, channel_name, nick, timestamp
                )

        agent.varlink_sender = MockSender()  # type: ignore
        agent.history = MockHistory()  # type: ignore

        # Simulate message handling
        await agent.handle_command(
            server="testserver",
            chan_name="#testchannel",
            target="#testchannel",
            nick="testuser",
            message=message,
            mynick="testbot",
        )

    except Exception as e:
        print(f"❌ Error handling message: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure any model router sessions held by the agent are closed
        try:
            if getattr(agent, "model_router", None) is not None:
                await agent.model_router.__aexit__(None, None, None)  # type: ignore
        except Exception:
            pass


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="irssi-llmagent - IRC chatbot with AI and tools")
    parser.add_argument(
        "--message", type=str, help="Run in CLI mode to simulate handling an IRC message"
    )
    parser.add_argument(
        "--config", type=str, help="Path to config file (default: config.json in project root)"
    )

    args = parser.parse_args()

    if args.message:
        asyncio.run(cli_mode(args.message, args.config))
    else:
        agent = IRSSILLMAgent()
        asyncio.run(agent.run())


if __name__ == "__main__":
    main()
