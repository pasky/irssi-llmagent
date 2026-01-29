# muaddib Agent Guide

## Build/Test Commands
- Install dependencies: `uv sync --dev`
- Any change should be accompanied with tests update. (Always prefer updating existing unit tests over adding new ones.)
- Any change where viable should be tested by actually running the CLI e2e test: `MUADDIB_HOME=. uv run muaddib --message "your message here"`
- Run linting, typecheck etc. via pre-commit.
- Run tests: `uv run pytest` - all tests must always succeed! You must assume any test failure is related to your changes, even if it doesn't appear to be at first.
- You must test and commit your work once finished. Never respond with "Tests not run (not requested)."
- NEVER use `git add -A` blindly, there may be untracked files that must not be committed; use `git add -u` instead

## Architecture
- **Main Service**: `muaddib/main.py` - Core application coordinator managing shared resources (config, history, model router)
- **Room Isolation**: IRC-specific functionality isolated in `rooms/irc/monitor.py` (IRCRoomMonitor class)
- **Modular Structure**: Clean separation between platform-agnostic core and IRC-specific implementation
- **Varlink Protocol**: Dual async socket architecture (events + sender) over UNIX socket at `~/.irssi/varlink.sock`
- **APIs**: Anthropic Claude (sarcastic/serious modes with automatic classification using claude-3-5-haiku), E2B sandbox for Python code execution
- **Config & Data**: All files live in `$MUADDIB_HOME` (defaults to `~/.muaddib/`):
  - `config.json` - JSON configuration (copy from `config.json.example`)
  - `chat_history.db` - SQLite persistent chat history
  - `chronicle.db` - Chronicle database
  - `artifacts/` - Shared artifacts directory
  - `logs/` - Per-message log files (DEBUG+)
  - Relative paths in config (e.g., `"path": "chronicle.db"`) are resolved against `$MUADDIB_HOME`
  - Models MUST be fully-qualified as `provider:model` (e.g., `anthropic:claude-sonnet-4`). No defaults.
  - No backwards compatibility is kept for legacy config keys; tests are aligned to the new schema.
  - Set `MUADDIB_HOME=.` for development to use current directory
- **Logging**: Console output (INFO+) and `$MUADDIB_HOME/logs/` files (DEBUG+), third-party libraries suppressed from console
- **Database**: SQLite persistent chat history with configurable inference limits (paths can be overridden in config)
- **Continuous Chronicling**: Automatic chronicling triggered when unchronicled messages exceed `history_size` threshold. Uses `chronicler.model` to summarize conversation activity into Chronicle chapters. Messages get linked via `chapter_id` field in ChatHistory. Includes safety limits (100 message batches, 7-day lookback) and overlap for context continuity
- **Proactive Interjecting**: Channel-based whitelist feature using claude-3-haiku to scan non-directed messages and interject in serious conversations when useful. Includes rate limiting, test mode, and channel whitelisting
- **Key Modules**:
  - `rooms/irc/monitor.py` - IRCRoomMonitor (main IRC message processing, command handling, mode classification)
  - `rooms/irc/varlink.py` - VarlinkClient (events), VarlinkSender (messages)
  - `rooms/irc/autochronicler.py` - AutoChronicler (automatic chronicling of IRC messages when threshold exceeded)
  - `rooms/proactive.py` - ProactiveDebouncer (channel-based proactive interjecting)
  - `history.py` - ChatHistory (persistent SQLite storage)
- `providers/` - async API clients (anthropic, openai) and base classes
  - `rate_limiter.py` - RateLimiter
  - `agentic_actor/` - AgenticLLMActor multi-turn mode with tool system for web search, webpage visiting, and Python code execution

## Code Style
- **Language**: Python 3.11+ with modern type hints (`dict`, `list`, ...), following PEP8
- **Async**: Full async/await support for non-blocking message processing
- **Background Tasks**: Use `muaddib.spawn(coro)` for fire-and-forget tasks (not bare `asyncio.create_task`) to prevent GC of unreferenced tasks
- **Imports**: Standard library first, then third-party (`aiohttp`, `aiosqlite`), local modules
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Brief docstrings for classes and key methods
- **Error Handling**: Write code that fails fast. No defensive try-except blocks. Only catch exceptions when there's a clear recovery strategy.
- **Testing**: Pytest with async support for behavioral tests

## Notes for contributors
- Tests should avoid mocking low-level API client constructors when validating control flow. Prefer patching router calls to inject fake responses, and ensure provider configs are referenced via `providers.*`.
- Do NOT introduce compatibility shims for legacy config fields; update tests and fixtures instead.
- When changing tests, prefer modifying/extending existing test files and cases rather than adding new test files, unless there is a compelling reason.
- For AI agents: When user is frustrated, stop and think why and consider whether not to append an additional behavioral instruction to this AGENTS.md file.
