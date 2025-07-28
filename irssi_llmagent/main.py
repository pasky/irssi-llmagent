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
from .rate_limiter import RateLimiter
from .varlink import VarlinkClient, VarlinkSender

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

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
        self.server_nicks: dict[str, str] = {}  # Cache of nicks per server

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
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
                    logger.info(f"Got nick for {server}: {nick}")
                return nick
            except Exception as e:
                logger.error(f"Failed to get nick for server {server}: {e}")
                return None
        return self.server_nicks[server]

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

        # Check if message is addressing us
        pattern = rf"^\s*{re.escape(mynick)}[,:]\s*(.*?)$"
        match = re.match(pattern, message, re.IGNORECASE)
        if not match:
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
            logger.info(f"Sending help message to {nick}")
            help_msg = "default is sarcastic Claude, !s is serious agentic Claude with web tools, !p is Perplexity (prefer English!)"
            await self.varlink_sender.send_message(target, help_msg, server)
            return

        elif message.startswith("!p ") or message == "!p":
            # Perplexity call
            logger.info(f"Processing Perplexity request from {nick}: {message}")
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

        elif re.match(r"^![^s]+", message):
            # Unknown command
            logger.warning(f"Unknown command from {nick}: {message}")
            await self.varlink_sender.send_message(
                target, f"{nick}: Unknown command. Use !h for help.", server
            )
            return

        else:
            # Claude call (default or serious)
            is_serious = message.startswith("!s ")
            if is_serious:
                message = message[3:].strip()
                logger.info(f"Processing serious Claude request from {nick}: {message}")

                # Use agent for serious mode

                context = await self.history.get_context(server, chan_name)
                async with ClaudeAgent(self.config, mynick) as agent:
                    response = await agent.run_agent(context)

                if response:
                    logger.info(f"Sending agent response to {target}: {response}")
                    await self.varlink_sender.send_message(target, f"{nick}: {response}", server)
                    # Update context with response
                    await self.history.add_message(
                        server, chan_name, response, mynick, mynick, True
                    )

            else:
                # Default sarcastic Claude (unchanged)
                logger.info(f"Processing sarcastic Claude request from {nick}: {message}")
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
                    await self.history.add_message(
                        server, chan_name, response, mynick, mynick, True
                    )

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
