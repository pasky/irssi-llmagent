#!/usr/bin/env python3
"""End-to-end benchmark for quest system with real LLM calls.

Usage:
    uv run python benchmarks/quest_e2e.py [--timeout SECONDS] [--config PATH]

Runs a fixed quest scenario through the IRC monitor and prints the resulting log.
Uses temporary databases so production data is unaffected.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from muaddib.main import MuaddibAgent  # noqa: E402

# Fixed scenario for benchmarking
SCENARIO_GOAL = """start a quest with the goal of:
1. Gathering current news about the Black Sea area (besides Ukrainian FIRs)
2. Identifying any risks to passing flights
3. Creating a PDF one-pager summarizing results

Use a separate subquest per phase. Do not solve everything immediately - outline the plan and leave execution for quest iterations."""

# Simulated IRC context
SERVER = "benchmark"
CHANNEL = "#quests"
ARC = f"{SERVER}#{CHANNEL}"
MYNICK = "questbot"
USER = "testuser"


async def run_benchmark(config_path: str, timeout: float) -> None:
    """Run the quest benchmark scenario."""
    # Create temporary databases
    with (
        tempfile.NamedTemporaryFile(suffix="_chronicle.db", delete=False) as chronicle_tmp,
        tempfile.NamedTemporaryFile(suffix="_history.db", delete=False) as history_tmp,
    ):
        chronicle_db = chronicle_tmp.name
        history_db = history_tmp.name

    print(f"Config: {config_path}")
    print(f"Timeout: {timeout}s")
    print(f"Temp DBs: {chronicle_db}, {history_db}")
    print("=" * 80)

    try:
        # Create agent with real config
        agent = MuaddibAgent(config_path)

        # Override DB paths to use temp files
        agent.chronicle.db_path = Path(chronicle_db)
        agent.history.db_path = history_db

        # Initialize databases
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Ensure arc is in allowed list for quests
        agent.config.setdefault("chronicler", {}).setdefault("quests", {})["arcs"] = [ARC]

        # Mock varlink sender to print messages as they happen
        class MockSender:
            async def send_message(self, target: str, message: str, server: str) -> None:
                print(f"<{MYNICK}> {message}")

            async def get_server_nick(self, server: str) -> str:
                return MYNICK

        agent.irc_monitor.varlink_sender = MockSender()  # type: ignore

        # Start quest heartbeat (after mocks are set up)
        await agent.quests.start_heartbeat()

        # Send initial message through the IRC monitor (like cli_message does)
        print(f"<{USER}> {MYNICK}: {SCENARIO_GOAL}")
        print("\n--- Processing through IRC monitor ---\n")

        from muaddib.rooms.message import RoomMessage

        async def reply_sender(text: str) -> None:
            await agent.irc_monitor.varlink_sender.send_message(CHANNEL, text, SERVER)

        msg = RoomMessage(
            server_tag=SERVER,
            channel_name=CHANNEL,
            nick=USER,
            mynick=MYNICK,
            content=SCENARIO_GOAL,
        )

        trigger_message_id = await agent.history.add_message(msg)
        await agent.irc_monitor.command_handler.handle_command(
            msg, trigger_message_id, reply_sender
        )

        # Wait for all quests to complete or timeout
        start_time = asyncio.get_event_loop().time()
        had_quests = False
        while True:
            await asyncio.sleep(2.0)
            elapsed = asyncio.get_event_loop().time() - start_time

            # Check for unfinished quests
            unfinished = await agent.chronicle.quests_count_unfinished(ARC)
            if unfinished > 0:
                had_quests = True
            elif had_quests:
                # Had quests before, now all finished
                print(f"\n✓ All quests finished after {elapsed:.1f}s")
                break

            if elapsed > timeout:
                print(f"\n⏱ Timeout after {timeout}s ({unfinished} unfinished quests)")
                break

            # Show progress periodically
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"... still running ({elapsed:.0f}s elapsed, {unfinished} unfinished quests)")

        # Print full history as IRC log
        print("\n" + "=" * 80)
        print("FULL IRC LOG")
        print("=" * 80)
        history = await agent.history.get_full_history(SERVER, CHANNEL)
        for msg in history:
            ts = msg["timestamp"].split(" ")[1][:5] if " " in msg["timestamp"] else ""
            print(f"[{ts}] {msg['message']}")

        # Print chronicle content
        print("\n" + "=" * 80)
        print("CHRONICLE")
        print("=" * 80)
        content = await agent.chronicle.render_chapter(ARC)
        print(content)

    finally:
        # Cleanup
        await agent.quests.stop_heartbeat()
        Path(chronicle_db).unlink(missing_ok=True)
        Path(history_db).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quest system end-to-end benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: $MUADDIB_HOME/config.json)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Max seconds to wait for quest completion (default: 600)",
    )

    args = parser.parse_args()

    from muaddib.paths import get_config_path

    config_path = args.config if args.config else str(get_config_path())

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("QUEST E2E BENCHMARK")
    print("=" * 80)
    print()

    asyncio.run(run_benchmark(config_path, args.timeout))


if __name__ == "__main__":
    main()
