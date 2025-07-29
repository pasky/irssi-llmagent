"""Main application entry point for irssi-llmagent."""

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import ClaudeAgent
from .claude import AnthropicClient
from .history import ChatHistory
from .perplexity import PerplexityClient
from .proactive_debouncer import ProactiveDebouncer
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

# Suppress aiosqlite DEBUG messages
logging.getLogger("aiosqlite").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class IRSSILLMAgent:
    """Main IRC LLM agent application."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.varlink_events = VarlinkClient(self.config["varlink"]["socket_path"])
        self.varlink_sender = VarlinkSender(self.config["varlink"]["socket_path"])
        self.history = ChatHistory(
            self.config.get("database", {}).get("path", "chat_history.db"),
            self.config["behavior"]["history_size"],
        )
        self.rate_limiter = RateLimiter(
            self.config["behavior"]["rate_limit"], self.config["behavior"]["rate_period"]
        )
        self.proactive_rate_limiter = RateLimiter(
            self.config["behavior"].get("proactive_rate_limit", 10),
            self.config["behavior"].get("proactive_rate_period", 60),
        )
        self.proactive_debouncer = ProactiveDebouncer(
            self.config["behavior"].get("proactive_debounce_seconds", 15.0)
        )
        self.server_nicks: dict[str, str] = {}  # Cache of nicks per server

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
        ignore_list = self.config["behavior"]["ignore_users"]
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

            prompt = self.config["prompts"]["mode_classifier"].format(message=current_message)
            model = self.config["anthropic"]["classifier_model"]

            async with AnthropicClient(self.config) as anthropic:
                response = await anthropic.call_claude(context, prompt, model)

            serious_count = response.count("SERIOUS")
            sarcastic_count = response.count("SARCASTIC")

            if serious_count == 0 and sarcastic_count == 0:
                logger.warning(f"Invalid mode classification response: {response}")
                return "SARCASTIC"
            elif serious_count <= sarcastic_count:
                return "SARCASTIC"
            else:
                return "SERIOUS"
        except Exception as e:
            logger.error(f"Error classifying mode: {e}")
            return "SARCASTIC"  # Default to sarcastic on error

    async def should_interject_proactively(self, context: list[dict[str, str]]) -> tuple[bool, str]:
        """Use preprocessing model to decide if bot should interject in conversation proactively.

        Args:
            context: Conversation context including the current message as the last entry

        Returns:
            (should_interject, reason): Tuple of decision and reasoning
        """
        try:
            if not context:
                return False, "No context provided"

            # Extract the current message from context (should be the last message)
            current_message = context[-1]["content"]

            # Clean message content if it has IRC nick formatting like "<nick> message"
            message_match = re.search(r"<[^>]+>\s*(.*)", current_message)
            if message_match:
                current_message = message_match.group(1).strip()

            # Use full context for better decision making, but specify the current message in prompt
            prompt = self.config["prompts"]["proactive_interject"].format(message=current_message)
            model = self.config["anthropic"]["proactive_model"]

            async with AnthropicClient(self.config) as anthropic:
                response = await anthropic.call_claude(context, prompt, model)

            if response:
                response = response.strip()

                # Extract the score from the response
                score_match = re.search(r"(\d+)/10", response)

                if score_match:
                    score = int(score_match.group(1))
                    threshold = self.config["behavior"].get("proactive_interject_threshold", 9)

                    # Only interject for scores at or above threshold
                    if score >= threshold:
                        logger.debug(
                            f"Proactive interjection triggered for message: {current_message[:150]}... (Score: {score})"
                        )
                        return True, f"Interjection decision (Score: {score})"
                    else:
                        return False, f"No interjection (Score: {score})"
                else:
                    logger.warning(
                        f"No valid score found in proactive interject response: {response}"
                    )
                    return False, f"No score found in response: {response}"
            else:
                return False, "No response from model"
        except Exception as e:
            logger.error(f"Error checking proactive interject: {e}")
            return False, f"Error: {str(e)}"

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

            # Get fresh context at check time
            context = await self.history.get_context(server, chan_name)
            should_interject, reason = await self.should_interject_proactively(context)

            if should_interject:
                # Classify the mode for proactive response (should be serious mode only)
                classified_mode = await self.classify_mode(context)
                if classified_mode == "SERIOUS":
                    # Check if this is a test channel
                    test_channels = self.config.get("behavior", {}).get(
                        "proactive_interjecting_test", []
                    )
                    is_test_channel = test_channels and chan_name in test_channels

                    target = chan_name  # For proactive interjections, target is the channel

                    if is_test_channel:
                        # Test mode: use the same method as live mode but don't send messages
                        logger.info(
                            f"[TEST MODE] Would interject proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                        )
                        await self._handle_serious_mode(
                            server,
                            chan_name,
                            target,
                            nick,
                            message,
                            mynick,
                            is_proactive=True,
                            send_message=False,
                        )
                    else:
                        # Live mode: actually send the response
                        logger.info(
                            f"Interjecting proactively for message from {nick} in {chan_name}: {message[:150]}... Reason: {reason}"
                        )
                        await self._handle_serious_mode(
                            server, chan_name, target, nick, message, mynick, is_proactive=True
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
        pattern = rf"^\s*{re.escape(mynick)}[,:]\s*(.*?)$"
        match = re.match(pattern, message, re.IGNORECASE)

        # If not directly addressed, check for proactive interjecting
        if not match:
            # Check both live and test channels for proactive interjecting
            proactive_channels = self.config.get("behavior", {}).get("proactive_interjecting", [])
            test_channels = self.config.get("behavior", {}).get("proactive_interjecting_test", [])
            is_live_channel = proactive_channels and chan_name in proactive_channels
            is_test_channel = test_channels and chan_name in test_channels

            if is_live_channel or is_test_channel:
                # Schedule debounced proactive check
                await self.proactive_debouncer.schedule_check(
                    server, chan_name, nick, message, mynick, self._handle_debounced_proactive_check
                )
            return

        cleaned_msg = match.group(1)
        logger.info(f"Received command from {nick} on {server}/{chan_name}: {cleaned_msg}")

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
        if message.startswith("!h") or message == "!h":
            logger.debug(f"Sending help message to {nick}")
            help_msg = "default is automatic mode (AI decides), !S is explicit sarcastic Claude, !s is serious agentic Claude with web tools, !p is Perplexity (prefer English!)"
            await self.varlink_sender.send_message(target, help_msg, server)

        elif message.startswith("!p ") or message == "!p":
            # Perplexity call
            logger.debug(f"Processing Perplexity request from {nick}: {message}")
            context = await self.history.get_context(server, chan_name)
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

        elif message.startswith("!S "):
            message = message[3:].strip()
            logger.debug(f"Processing explicit sarcastic Claude request from {nick}: {message}")
            await self._handle_sarcastic_mode(server, chan_name, target, nick, message, mynick)

        elif message.startswith("!s "):
            message = message[3:].strip()
            logger.debug(f"Processing explicit serious Claude request from {nick}: {message}")
            await self._handle_serious_mode(server, chan_name, target, nick, message, mynick)

        elif re.match(r"^!.", message):
            logger.warning(f"Unknown command from {nick}: {message}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Unknown command. Use !h for help.", server
            )

        else:
            logger.debug(f"Processing automatic mode request from {nick}: {message}")
            context = await self.history.get_context(server, chan_name)
            classified_mode = await self.classify_mode(context)
            logger.debug(f"Auto-classified message as {classified_mode} mode")

            if classified_mode == "SERIOUS":
                await self._handle_serious_mode(server, chan_name, target, nick, message, mynick)
            else:
                await self._handle_sarcastic_mode(server, chan_name, target, nick, message, mynick)

    async def _handle_serious_mode(
        self,
        server: str,
        chan_name: str,
        target: str,
        nick: str,
        message: str,
        mynick: str,
        is_proactive: bool = False,
        send_message: bool = True,
    ) -> str | None:
        """Handle serious mode using ClaudeAgent with tools."""
        context = await self.history.get_context(server, chan_name)

        # Add extra prompt for proactive interjections to allow null response
        extra_prompt = ""
        if is_proactive:
            extra_prompt = " " + self.config["prompts"]["proactive_serious_extra"]

        # Use default model for proactive interjections to avoid expensive opus model
        model_override = self.config["anthropic"]["model"] if is_proactive else None
        async with ClaudeAgent(
            self.config, mynick, extra_prompt, model_override=model_override
        ) as agent:
            response = await agent.run_agent(context)

        # For proactive interjections, check for NULL response
        if is_proactive and response and response.strip().upper() == "NULL":
            logger.info(f"Agent decided not to interject proactively for {target}")
            return None

        if response:
            if send_message:
                logger.info(f"Sending agent response to {target}: {response}")
                await self.varlink_sender.send_message(target, response, server)
                # Update context with response
                await self.history.add_message(server, chan_name, response, mynick, mynick, True)
            else:
                logger.info(f"[TEST MODE] Generated response for {target}: {response}")

        return response

    async def _handle_sarcastic_mode(
        self, server: str, chan_name: str, target: str, nick: str, message: str, mynick: str
    ) -> None:
        """Handle sarcastic mode using direct Claude calls."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = self.config["prompts"]["sarcastic"].format(
            mynick=mynick, current_time=current_time
        )
        model = self.config["anthropic"]["model"]

        context = await self.history.get_context(server, chan_name)
        async with AnthropicClient(self.config) as anthropic:
            response = await anthropic.call_claude(context, system_prompt, model)

        if response:
            logger.info(f"Sending Claude response to {target}: {response}")
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

        class MockHistory:
            async def add_message(
                self, server: str, channel: str, content: str, nick: str, mynick: str, is_bot: bool
            ):
                pass

            async def get_context(self, server: str, channel: str):
                # Include the current message in context for CLI mode
                return [{"role": "user", "content": message}]

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


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="irssi-llmagent - IRC chatbot with Claude and tools"
    )
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
