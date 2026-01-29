#!/usr/bin/env python3
"""Analyze when muaddib would interject in conversations proactively."""

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


def extract_non_bot_messages_from_logs(
    log_path: str, limit: int = 100, exclude_channels: list[str] | None = None
) -> list[dict]:
    """Extract recent non-bot messages from IRC log files."""
    messages = []

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

    # Parse log lines and find non-bot messages (not mentioning Muaddib)
    for i, log_line in enumerate(log_lines):
        line = log_line["content"]

        # Parse irssi log format: .irssi/logs/server/channel/date.log:time < nick> message
        # or .irssi/logs/server/channel/date.log-time < nick> message
        log_match = re.match(r"^(.+\.log)[-:]([\d:]+)\s+<\s*([^>]+)>\s*(.*)$", line)
        if not log_match:
            continue

        file_path, timestamp, nick, message = log_match.groups()

        # Skip messages that mention Muaddib
        if "muaddib" in message.lower():
            continue

        # Extract server and channel from file path
        path_parts = Path(file_path).parts
        if len(path_parts) >= 3:
            server = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            channel = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        else:
            server = "unknown"
            channel = "unknown"

        # Skip excluded channels
        if exclude_channels and channel in exclude_channels:
            continue

        # Skip if it's actually a bot message
        if nick.lower() in ["muaddib", "system"]:
            continue

        # Get context from preceding lines (last 5 lines) formatted for analysis
        context_messages = []
        for j in range(max(0, i - 5), i):
            ctx_line = log_lines[j]["content"]
            ctx_match = re.match(r"^(.+\.log)[-:]([\d:]+)\s+<\s*([^>]+)>\s*(.*)$", ctx_line)
            if ctx_match:
                _, ctx_time, ctx_nick, ctx_msg = ctx_match.groups()
                # Format as context - assume user role for non-muaddib messages
                role = "user" if ctx_nick.lower() not in ["muaddib"] else "assistant"
                context_messages.append({"role": role, "content": f"{ctx_nick}: {ctx_msg}"})

        messages.append(
            {
                "nick": nick,
                "message": message,
                "timestamp": f"{Path(file_path).stem} {timestamp}",
                "server_tag": server,
                "channel_name": channel,
                "context": context_messages,
                "context_display": "; ".join([f"{msg['content']}" for msg in context_messages]),
            }
        )

        # Limit number of messages
        if len(messages) >= limit:
            break

    logger.info(f"Extracted {len(messages)} non-bot messages from log files")
    return messages


async def extract_non_bot_messages_from_db(
    db_path: str, limit: int = 100, exclude_channels: list[str] | None = None
) -> list[dict]:
    """Extract recent non-bot messages from the database."""
    messages = []

    # Build the WHERE clause with channel exclusion
    where_conditions = ["role = 'user'", "message NOT LIKE '%Muaddib%'"]
    params = []

    if exclude_channels:
        placeholders = ",".join("?" * len(exclude_channels))
        where_conditions.append(f"channel_name NOT IN ({placeholders})")
        params.extend(exclude_channels)

    params.append(limit)

    query = f"""
        SELECT nick, message, timestamp, server_tag, channel_name
        FROM chat_messages
        WHERE {" AND ".join(where_conditions)}
        ORDER BY timestamp DESC
        LIMIT ?
    """

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    for row in rows:
        nick, message, timestamp, server_tag, channel_name = row

        # Clean up nick format and extract clean message
        clean_nick = nick.replace("<", "").replace(">", "").strip()

        # Extract message content if it has IRC nick formatting like "<nick> message"
        clean_message = message
        message_match = re.search(r"<[^>]+>\s*(.*)", message)
        if message_match:
            clean_message = message_match.group(1).strip()

        # Skip if it's actually a bot message or system message
        if clean_nick.lower() in ["muaddib", "system", ""]:
            continue

        # Get context - look for surrounding messages
        context_messages = []
        async with aiosqlite.connect(db_path) as db:
            # Get recent messages from same channel (don't worry about exact timestamp math)
            context_cursor = await db.execute(
                """
                SELECT nick, message, timestamp
                FROM chat_messages
                WHERE server_tag = ? AND channel_name = ?
                AND role = 'user'
                ORDER BY timestamp DESC
                LIMIT 5
            """,
                (server_tag, channel_name),
            )
            context_rows = await context_cursor.fetchall()

        for ctx_row in context_rows:
            ctx_nick, ctx_message, ctx_timestamp = ctx_row
            clean_ctx_nick = ctx_nick.replace("<", "").replace(">", "").strip()
            if clean_ctx_nick.lower() != "muaddib":
                context_messages.append(
                    {"role": "user", "content": f"{clean_ctx_nick}: {ctx_message}"}
                )

        messages.append(
            {
                "nick": clean_nick,
                "message": clean_message,
                "timestamp": timestamp,
                "server_tag": server_tag,
                "channel_name": channel_name,
                "context": context_messages,
                "context_display": "; ".join([f"{msg['content']}" for msg in context_messages]),
            }
        )

    logger.info(f"Extracted {len(messages)} non-bot messages from database")
    return messages


async def should_interject_proactively_with_reason(
    agent: MuaddibAgent, context: list[dict]
) -> tuple[bool, str]:
    """Use MuaddibAgent method to determine if bot should interject proactively."""
    try:
        # Use the actual agent method - it now returns decision, reason, and test_mode flag
        (
            should_interject,
            reason,
            test_mode,
        ) = await agent.irc_monitor.command_handler.should_interject_proactively(context)
        return should_interject, reason
    except Exception as e:
        logger.error(f"Error checking proactive interject: {e}")
        return False, f"Error: {str(e)}"


async def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze when muaddib would interject in conversations proactively",
        epilog="""
Examples:
  python analyze_proactive.py                    # Use SQLite database (default)
  python analyze_proactive.py --limit 50        # Analyze 50 recent messages
  python analyze_proactive.py --logs logfile.txt # Use single IRC log file
  python analyze_proactive.py --logs ~/.irssi/logs/ # Use directory of log files
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
    parser.add_argument(
        "--limit", type=int, help="Number of recent messages to analyze (default: 100)", default=100
    )
    parser.add_argument(
        "--exclude-news", action="store_true", help="Exclude #news.cz channel from analysis"
    )

    args = parser.parse_args()

    from muaddib.paths import get_config_path, get_default_history_db_path

    config_path = Path(args.config) if args.config else get_config_path()

    if not config_path.exists():
        print(f"Error: config file {config_path} not found")
        sys.exit(1)

    # Create agent instance to use its methods
    agent = MuaddibAgent(str(config_path))

    # Resolve db path
    db_path = args.db if args.db else str(get_default_history_db_path())

    # Set up exclusions
    exclude_channels = ["#news.cz"] if args.exclude_news else []

    # Extract messages from appropriate source
    if args.logs:
        # Use IRC log files
        print(f"📋 Analyzing {args.limit} recent non-bot messages from: {args.logs}")
        if exclude_channels:
            print(f"Excluding channels: {exclude_channels}")
        messages = extract_non_bot_messages_from_logs(args.logs, args.limit, exclude_channels)
        source_desc = f"log files ({args.logs})"
    else:
        # Use database (default)
        print(f"📋 Analyzing {args.limit} recent non-bot messages from: {db_path}")
        if exclude_channels:
            print(f"Excluding channels: {exclude_channels}")
        messages = await extract_non_bot_messages_from_db(db_path, args.limit, exclude_channels)
        source_desc = f"database ({db_path})"

    if not messages:
        print("No messages found")
        return

    print(f"Found {len(messages)} messages from {source_desc} to analyze")
    print("Checking which ones would trigger proactive interjecting...")

    # Create CSV file and write header
    output_file = "proactive_analysis.csv"
    fieldnames = [
        "timestamp",
        "nick",
        "server_tag",
        "channel_name",
        "message",
        "should_interject",
        "reason",
        "context",
    ]

    results = []
    interject_count = 0

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, msg in enumerate(messages):
            print(f"Processing {i + 1}/{len(messages)}: {msg['message'][:50]}...")

            # Create proper context with the current message as last entry
            full_context = msg.get("context", []) + [{"role": "user", "content": msg["message"]}]
            should_interject, reason = await should_interject_proactively_with_reason(
                agent, full_context
            )

            if should_interject:
                interject_count += 1
                print(f"  ✅ WOULD INTERJECT: {reason}")

            result = {
                "timestamp": msg["timestamp"],
                "nick": msg["nick"],
                "server_tag": msg["server_tag"],
                "channel_name": msg["channel_name"],
                "message": msg["message"],
                "should_interject": should_interject,
                "reason": reason,
                "context": msg.get("context_display", ""),
            }

            results.append(result)
            writer.writerow(result)
            csvfile.flush()

            # Rate limiting - small delay between requests
            await asyncio.sleep(0.5)

    # Print summary
    total = len(results)
    interject_rate = interject_count / total * 100 if total > 0 else 0

    print("\nAnalysis complete!")
    print(f"Total messages analyzed: {total}")
    print(f"Would interject proactively: {interject_count}")
    print(f"Interjection rate: {interject_rate:.1f}%")
    print(f"Results saved to: {output_file}")

    # Show examples of messages that would trigger interjecting
    print("\nExamples where bot would interject:")
    interject_examples = [r for r in results if r["should_interject"]][:10]
    for example in interject_examples:
        print(f"  '{example['message'][:60]}...' -> {example['reason']}")

    print("\nExamples where bot would NOT interject:")
    no_interject_examples = [r for r in results if not r["should_interject"]][:5]
    for example in no_interject_examples:
        print(f"  '{example['message'][:60]}...' -> {example['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
