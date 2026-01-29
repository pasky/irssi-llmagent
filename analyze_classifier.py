#!/usr/bin/env python3
"""Analyze classifier performance on historic Muaddib invocations."""

import asyncio
import csv
import logging
import re
import sys
from pathlib import Path

import aiosqlite

from muaddib.main import MuaddibAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def extract_bot_invocations_from_db(db_path: str) -> list[dict]:
    """Extract all historic bot invocations from the database."""
    invocations = []

    async with aiosqlite.connect(db_path) as db:
        # Find messages that mention Muaddib and extract the actual user message
        cursor = await db.execute(
            """
            SELECT nick, message, timestamp, server_tag, channel_name
            FROM chat_messages
            WHERE role = 'user' AND message LIKE '%Muaddib%'
            ORDER BY timestamp ASC
        """
        )
        rows = await cursor.fetchall()

    for row in rows:
        nick, message, timestamp, server_tag, channel_name = row

        # Extract the actual message content after "Muaddib:"
        # Format is typically "<nick> Muaddib: actual message"
        message_match = re.search(r"Muaddib[,:]\s*(.*)", message, re.IGNORECASE)
        if message_match:
            actual_message = message_match.group(1).strip()

            # Skip empty messages
            if not actual_message:
                continue

            invocations.append(
                {
                    "nick": nick.replace("<", "").replace(">", ""),
                    "original_message": message,
                    "user_message": actual_message,
                    "timestamp": timestamp,
                    "server_tag": server_tag,
                    "channel_name": channel_name,
                    "context": [],  # No context for DB entries
                    "context_display": "",
                }
            )

    logger.info(f"Extracted {len(invocations)} bot invocations from database")
    return invocations


def extract_bot_invocations_from_logs(log_path: str) -> list[dict]:
    """Extract all historic bot invocations from IRC log files."""
    invocations = []

    # Read all log files
    log_lines = []
    log_dir = Path(log_path)

    log_files = [log_dir] if log_dir.is_file() else list(log_dir.rglob("*.log"))

    for log_file in sorted(log_files):
        try:
            with open(log_file, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        log_lines.append(
                            {"file": str(log_file), "line_num": line_num + 1, "content": line}
                        )
        except Exception as e:
            logger.warning(f"Could not read {log_file}: {e}")

    # Parse log lines and find Muaddib invocations
    for i, log_line in enumerate(log_lines):
        line = log_line["content"]

        # Parse irssi log format: .irssi/logs/server/channel/date.log:time < nick> message
        # or .irssi/logs/server/channel/date.log-time < nick> message
        log_match = re.match(r"^(.+\.log)[-:]([\d:]+)\s+<\s*([^>]+)>\s*(.*)$", line)
        if not log_match:
            continue

        file_path, timestamp, nick, message = log_match.groups()

        # Check if this mentions Muaddib
        muaddib_match = re.search(r"Muaddib[,:]\s*(.*)", message, re.IGNORECASE)
        if not muaddib_match:
            continue

        actual_message = muaddib_match.group(1).strip()
        if not actual_message:
            continue

        # Extract server and channel from file path
        # Format: .irssi/logs/server/channel/date.log
        path_parts = Path(file_path).parts
        if len(path_parts) >= 3:
            server = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            channel = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        else:
            server = "unknown"
            channel = "unknown"

        # Get context from preceding lines (last 5 lines) formatted for Claude
        context_messages = []
        for j in range(max(0, i - 5), i):
            ctx_line = log_lines[j]["content"]
            ctx_match = re.match(r"^(.+\.log)[-:]([\d:]+)\s+<\s*([^>]+)>\s*(.*)$", ctx_line)
            if ctx_match:
                _, ctx_time, ctx_nick, ctx_msg = ctx_match.groups()
                # Format as Claude context - assume alternating user/assistant for now
                role = "user" if ctx_nick.lower() not in ["muaddib"] else "assistant"
                context_messages.append({"role": role, "content": f"{ctx_nick}: {ctx_msg}"})

        invocations.append(
            {
                "nick": nick,
                "original_message": message,
                "user_message": actual_message,
                "timestamp": f"{Path(file_path).stem} {timestamp}",
                "server_tag": server,
                "channel_name": channel,
                "context": context_messages,
                "context_display": "; ".join([f"{msg['content']}" for msg in context_messages]),
            }
        )

    logger.info(f"Extracted {len(invocations)} bot invocations from log files")
    return invocations


async def classify_message_with_agent(
    agent: MuaddibAgent, message: str, context: list[dict] | None = None
) -> str:
    """Classify a single message using the agent's classifier method."""
    try:
        if context is None:
            context = []

        # Ensure context includes the current message as the last entry
        if not context or context[-1]["content"] != message:
            context = context + [{"role": "user", "content": message}]

        return await agent.irc_monitor.command_handler.classify_mode(context)
    except Exception as e:
        logger.error(f"Error classifying message '{message}': {e}")
        return "ERROR"


def determine_actual_mode(message: str) -> str:
    """Determine what mode was actually used based on the message commands."""
    if message.startswith("!s "):
        return "SERIOUS"
    elif message.startswith("!S "):
        return "SARCASTIC"
    elif message.startswith("!p "):
        return "PERPLEXITY"
    elif message.startswith("!h"):
        return "HELP"
    else:
        # For historic data before the new system, this was sarcastic by default
        return "SARCASTIC_DEFAULT"


async def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze classifier performance on historic Muaddib invocations",
        epilog="""
Examples:
  python analyze_classifier.py                    # Use SQLite database (default)
  python analyze_classifier.py --logs logfile.txt # Use single IRC log file
  python analyze_classifier.py --logs ~/.irssi/logs/ # Use directory of log files
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="SQLite database path (default: $MUADDIB_HOME/chat_history.db)",
        default=None,
    )
    parser.add_argument(
        "--logs", type=str, help="IRC log file or directory path (alternative to --db)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Config file path (default: $MUADDIB_HOME/config.json)",
        default=None,
    )

    args = parser.parse_args()

    from muaddib.paths import get_config_path, get_default_history_db_path

    config_path = Path(args.config) if args.config else get_config_path()

    if not config_path.exists():
        print(f"Error: config file {config_path} not found")
        sys.exit(1)

    # Create agent instance for classification
    agent = MuaddibAgent(str(config_path))

    # Resolve db path
    db_path = args.db if args.db else str(get_default_history_db_path())

    # Extract historic invocations
    if args.logs:
        # Use IRC log files
        print(f"📋 Analyzing IRC log files from: {args.logs}")
        invocations = extract_bot_invocations_from_logs(args.logs)
        source_desc = f"log files ({args.logs})"
    else:
        # Use database (default)
        print(f"📋 Analyzing SQLite database: {db_path}")
        invocations = await extract_bot_invocations_from_db(db_path)
        source_desc = f"database ({db_path})"

    if not invocations:
        print("No bot invocations found")
        return

    print(f"Found {len(invocations)} historic invocations from {source_desc}")
    print("Classifying messages...")

    # Create CSV file and write header immediately
    output_file = "classifier_analysis.csv"
    fieldnames = [
        "timestamp",
        "nick",
        "server_tag",
        "channel_name",
        "original_message",
        "clean_message",
        "actual_mode",
        "predicted_mode",
        "correct",
        "context",
    ]

    # Open file for writing
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process in smaller batches and save as we go
        results = []
        batch_size = 5

        for i, inv in enumerate(invocations):
            print(f"Processing {i + 1}/{len(invocations)}: {inv['user_message'][:50]}...")

            predicted_mode = await classify_message_with_agent(
                agent, inv["user_message"], inv.get("context", [])
            )
            actual_mode = determine_actual_mode(inv["user_message"])

            # Clean the message for commands to get the actual content
            clean_message = inv["user_message"]
            if clean_message.startswith(("!s ", "!S ", "!p ")):
                clean_message = clean_message[3:].strip()
            elif clean_message.startswith("!h"):
                clean_message = "help request"

            result = {
                "timestamp": inv["timestamp"],
                "nick": inv["nick"],
                "server_tag": inv["server_tag"],
                "channel_name": inv["channel_name"],
                "original_message": inv["user_message"],
                "clean_message": clean_message,
                "actual_mode": actual_mode,
                "predicted_mode": predicted_mode,
                "correct": predicted_mode == actual_mode
                or (actual_mode == "SARCASTIC_DEFAULT" and predicted_mode == "SARCASTIC"),
                "context": inv.get("context_display", ""),
            }

            results.append(result)
            writer.writerow(result)
            csvfile.flush()  # Force write to disk

            # Show running statistics every batch
            if (i + 1) % batch_size == 0:
                correct = sum(1 for r in results if r["correct"])
                accuracy = correct / len(results) * 100
                print(f"  Running accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")

            # Rate limiting - small delay between requests
            await asyncio.sleep(0.3)

    # Print final summary statistics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0

    print("\nAnalysis complete!")
    print(f"Total messages analyzed: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {output_file}")

    # Show breakdown by actual mode
    mode_counts = {}
    mode_correct = {}
    for result in results:
        mode = result["actual_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        if result["correct"]:
            mode_correct[mode] = mode_correct.get(mode, 0) + 1

    print("\nBreakdown by actual mode:")
    for mode in sorted(mode_counts.keys()):
        count = mode_counts[mode]
        correct_count = mode_correct.get(mode, 0)
        mode_accuracy = correct_count / count * 100 if count > 0 else 0
        print(f"  {mode}: {correct_count}/{count} ({mode_accuracy:.1f}%)")

    # Show some examples of misclassifications
    print("\nSample misclassifications:")
    misclassified = [r for r in results if not r["correct"]][:5]
    for mc in misclassified:
        print(
            f"  '{mc['clean_message'][:60]}...' -> Predicted: {mc['predicted_mode']}, Actual: {mc['actual_mode']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
