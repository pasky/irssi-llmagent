"""Main application entry point for muaddib."""

import argparse
import asyncio
import dataclasses
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from .agentic_actor import AgenticLLMActor
from .agentic_actor.actor import AgentResult
from .chronicler.chronicle import Chronicle
from .chronicler.quests import QuestOperator
from .context_reducer import ContextReducer
from .history import ChatHistory
from .message_logging import MessageContextHandler
from .paths import (
    get_config_path,
    get_default_chronicle_db_path,
    get_default_history_db_path,
    get_logs_dir,
)
from .providers import ModelRouter
from .rooms.command import get_room_config
from .rooms.irc import IRCRoomMonitor

# Set up logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler for INFO and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)

# Per-message context handler for DEBUG and above
# Routes logs to per-message files under $MUADDIB_HOME/logs/ directory
# Skip file logging during tests - just use stderr for DEBUG logs
if "pytest" not in sys.modules:
    message_context_handler = MessageContextHandler(get_logs_dir(), level=logging.DEBUG)
    root_logger.addHandler(message_context_handler)
else:
    stderr_debug_handler = logging.StreamHandler(sys.stderr)
    stderr_debug_handler.setLevel(logging.DEBUG)
    stderr_debug_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_debug_handler)

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


def _resolve_path(path: str | None, default_path: Path) -> str:
    """Resolve a config path - relative paths are resolved against MUADDIB_HOME."""
    if not path:
        return str(default_path)
    p = Path(path)
    if p.is_absolute():
        return path
    from .paths import get_muaddib_home

    return str(get_muaddib_home() / path)


class MuaddibAgent:
    """Main IRC LLM agent application."""

    def __init__(self, config_path: str | None = None):
        resolved_config_path = config_path or str(get_config_path())
        self.config = self.load_config(resolved_config_path)
        self.model_router: ModelRouter = ModelRouter(self.config)
        irc_config = get_room_config(self.config, "irc")
        self.irc_enabled = irc_config.get("enabled", True)
        history_db_path = _resolve_path(
            self.config.get("history", {}).get("database", {}).get("path"),
            get_default_history_db_path(),
        )
        self.history = ChatHistory(
            history_db_path,
            irc_config["command"]["history_size"],
        )
        # Initialize chronicle
        chronicler_config = self.config.get("chronicler", {})
        chronicle_db_path = _resolve_path(
            chronicler_config.get("database", {}).get("path"),
            get_default_chronicle_db_path(),
        )
        self.chronicle = Chronicle(chronicle_db_path)
        self.context_reducer = ContextReducer(self)
        self.irc_monitor = IRCRoomMonitor(self)
        self.discord_monitor = None
        discord_config = self.config.get("rooms", {}).get("discord")
        if discord_config and discord_config.get("enabled"):
            from .rooms.discord import DiscordRoomMonitor

            self.discord_monitor = DiscordRoomMonitor(self)
        self.slack_monitor = None
        slack_config = self.config.get("rooms", {}).get("slack")
        if slack_config and slack_config.get("enabled"):
            from .rooms.slack import SlackRoomMonitor

            self.slack_monitor = SlackRoomMonitor(self)
        self.quests = QuestOperator(self)

    async def run_actor(
        self,
        context: list[dict[str, str]],
        *,
        mode_cfg: dict[str, Any],
        system_prompt: str,
        arc: str = "",
        progress_callback=None,
        persistence_callback=None,
        model: str | list[str] | None = None,
        current_quest_id: str | None = None,
        secrets: dict[str, Any] | None = None,
        **actor_kwargs,
    ) -> AgentResult | None:
        prepended_context: list[dict[str, str]] = []
        if mode_cfg.get("include_chapter_summary", True) and arc:
            prepended_context = await self.chronicle.get_chapter_context_messages(arc)

        if mode_cfg.get("reduce_context") and self.context_reducer.is_configured:
            full_context = prepended_context + context
            reduced = await self.context_reducer.reduce(full_context, system_prompt)
            prepended_context = []
            context = reduced + context[-1:]

        actor = AgenticLLMActor(
            config=self.config,
            model=model or mode_cfg["model"],
            system_prompt_generator=lambda: system_prompt,
            prompt_reminder_generator=lambda: mode_cfg.get("prompt_reminder"),
            prepended_context=prepended_context,
            agent=self,
            vision_model=mode_cfg.get("vision_model"),
            secrets=secrets,
            **actor_kwargs,
        )
        agent_result = await actor.run_agent(
            context,
            progress_callback=progress_callback,
            persistence_callback=persistence_callback,
            arc=arc,
            current_quest_id=current_quest_id,
        )

        if not agent_result.text or agent_result.text.strip().upper().startswith("NULL"):
            return None
        cleaned = agent_result.text.strip()
        # Strip IRC-style leading prefixes from context-echoed outputs: [model], mode commands, timestamps, and non-quest tags like <nick>.
        # Never strip <quest> or <quest_finished> because those carry semantics for the chronicler.
        cleaned = re.sub(
            r"^(?:\s*(?:\[[^\]]+\]\s*)?(?:![dsau]\s+)?(?:\[?\d{1,2}:\d{2}\]?\s*)?(?:<(?!/?quest(?:_finished)?\b)[^>]+>))*\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Return a new AgentResult with cleaned text but same usage stats
        return dataclasses.replace(agent_result, text=cleaned)

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
                "Copy config.json.example to $MUADDIB_HOME/config.json (default: ~/.muaddib/config.json) and configure."
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
        # Start quest heartbeat (will periodically prod ongoing quests)
        await self.quests.start_heartbeat()

        try:
            monitors = []
            if self.irc_enabled:
                monitors.append(self.irc_monitor.run())
            if self.discord_monitor:
                monitors.append(self.discord_monitor.run())
            if self.slack_monitor:
                monitors.append(self.slack_monitor.run())
            if not monitors:
                logger.warning("No room monitors enabled; exiting")
                return
            await asyncio.gather(*monitors)
        finally:
            # Clean up shared resources
            await self.quests.stop_heartbeat()
            await self.history.close()
            # Chronicle doesn't need explicit cleanup


async def cli_message(message: str, config_path: str | None = None) -> None:
    """CLI mode for testing message handling including command parsing."""
    # Load configuration
    config_file = Path(config_path) if config_path else get_config_path()

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print(
            "Please copy config.json.example to $MUADDIB_HOME/config.json (default: ~/.muaddib/config.json)"
        )
        sys.exit(1)

    print(f"🤖 Simulating IRC message: {message}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = MuaddibAgent(str(config_file))

        # Initialize shared resources for CLI mode
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Mock the varlink sender
        class MockSender:
            async def send_message(self, target: str, message: str, server: str):
                print(f"📤 Bot response: {message}")

        agent.irc_monitor.varlink_sender = MockSender()  # type: ignore

        from muaddib.rooms.message import RoomMessage

        async def reply_sender(text: str) -> None:
            print(f"📤 Bot response: {text}")

        msg = RoomMessage(
            server_tag="testserver",
            channel_name="#testchannel",
            nick="testuser",
            mynick="testbot",
            content=message,
        )

        trigger_message_id = await agent.history.add_message(msg)
        await agent.irc_monitor.command_handler.handle_command(
            msg, trigger_message_id, reply_sender
        )

    except Exception as e:
        print(f"❌ Error handling message: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure any shared resources held by the agent are closed
        try:
            if hasattr(agent, "history"):
                await agent.history.close()
        except Exception:
            pass


async def cli_chronicler(arc: str, instructions: str, config_path: str | None = None) -> None:
    """CLI mode for Chronicler operations."""
    # Load configuration
    config_file = Path(config_path) if config_path else get_config_path()

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print(
            "Please copy config.json.example to $MUADDIB_HOME/config.json (default: ~/.muaddib/config.json)"
        )
        sys.exit(1)

    print(f"🔮 Chronicler arc '{arc}': {instructions}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = MuaddibAgent(str(config_file))
        await agent.chronicle.initialize()

        print(
            "Error: Chronicler subagent functionality has been removed. Use direct chronicle_append and chronicle_read tools instead."
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="muaddib - IRC chatbot with AI and tools")
    parser.add_argument(
        "--message", type=str, help="Run in CLI mode to simulate handling an IRC message"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: $MUADDIB_HOME/config.json or ~/.muaddib/config.json)",
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
        agent = MuaddibAgent()
        asyncio.run(agent.run())


if __name__ == "__main__":
    main()
