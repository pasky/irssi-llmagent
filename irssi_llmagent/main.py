"""Main application entry point for irssi-llmagent."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .chronicler.chronicle import Chronicle
from .history import ChatHistory
from .providers import ModelRouter
from .rooms.irc import IRCRoomMonitor

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
        self.history = ChatHistory(
            self.config.get("database", {}).get("path", "chat_history.db"),
            irc_config["command"]["history_size"],
        )
        # Initialize chronicle
        chronicler_config = self.config.get("chronicler", {})
        chronicle_db_path = chronicler_config.get("database", {}).get("path", "chronicle.db")
        self.chronicle = Chronicle(chronicle_db_path)
        self.irc_monitor = IRCRoomMonitor(self)

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

    async def run(self) -> None:
        """Run the main agent loop by delegating to IRC monitor."""
        # Initialize shared resources
        await self.history.initialize()
        await self.chronicle.initialize()

        try:
            await self.irc_monitor.run()
        finally:
            # Clean up shared resources
            await self.history.close()
            # Chronicle doesn't need explicit cleanup
            if self.model_router:
                await self.model_router.__aexit__(None, None, None)


async def cli_message(message: str, config_path: str | None = None) -> None:
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

        # Mock the varlink sender
        class MockSender:
            async def send_message(self, target: str, message: str, server: str):
                print(f"📤 Bot response: {message}")

        agent.irc_monitor.varlink_sender = MockSender()  # type: ignore

        # Simulate message handling
        await agent.irc_monitor.handle_command(
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


async def cli_chronicler(arc: str, instructions: str, config_path: str | None = None) -> None:
    """CLI mode for Chronicler operations."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    print(f"🔮 Chronicler arc '{arc}': {instructions}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = IRSSILLMAgent(str(config_file))
        await agent.chronicle.initialize()

        # Import here to avoid circular imports
        from .chronicler.subagent import run_chronicler

        result = await run_chronicler(agent, arc=arc, instructions=instructions)
        print(result)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="irssi-llmagent - IRC chatbot with AI and tools")
    parser.add_argument(
        "--message", type=str, help="Run in CLI mode to simulate handling an IRC message"
    )
    parser.add_argument(
        "--config", type=str, help="Path to config file (default: config.json in project root)"
    )
    parser.add_argument(
        "--chronicler",
        type=str,
        help="Run Chronicler subagent with instructions (NLI over Chronicle)",
    )
    parser.add_argument(
        "--arc", type=str, help="Arc name for Chronicler (required with --chronicler)"
    )

    args = parser.parse_args()

    if args.chronicler:
        if not args.arc:
            print("Error: --arc is required with --chronicler")
            sys.exit(1)
        asyncio.run(cli_chronicler(args.arc, args.chronicler, args.config))
        return

    if args.message:
        asyncio.run(cli_message(args.message, args.config))
    else:
        agent = IRSSILLMAgent()
        asyncio.run(agent.run())


if __name__ == "__main__":
    main()
