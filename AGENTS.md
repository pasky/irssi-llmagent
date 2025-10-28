# irssi-llmagent Agent Guide

## Build/Test Commands
- Install dependencies: `uv sync --dev`
- Run tests: `uv run pytest` - all tests must always succeed! You must assume any test failure is related to your changes, even if it doesn't appear to be at first.
- Full e2e test in CLI mode: `uv run irssi-llmagent --message "your message here"`
- Run linting: `uv run ruff check .`
- Run formatting: `uv run ruff format .`
- Run type checking: `uv run pyright`
- NEVER use `git add -A` blindly, there may be untracked files that must not be committed; use `git add -u` instead

## Architecture
- **Main Service**: `irssi_llmagent/main.py` - Core application coordinator managing shared resources (config, history, model router)
- **Room Isolation**: IRC-specific functionality isolated in `rooms/irc/monitor.py` (IRCRoomMonitor class)
- **Modular Structure**: Clean separation between platform-agnostic core and IRC-specific implementation
- **Varlink Protocol**: Dual async socket architecture (events + sender) over UNIX socket at `~/.irssi/varlink.sock`
- **APIs**: Anthropic Claude (sarcastic/serious modes with automatic classification using claude-3-5-haiku), Perplexity AI, E2B sandbox for Python code execution
- **Config**: JSON-based configuration in `config.json` (copy from `config.json.example`)
  - Models MUST be fully-qualified as `provider:model` (e.g., `anthropic:claude-sonnet-4`). No defaults.
  - No backwards compatibility is kept for legacy config keys; tests are aligned to the new schema.
- **Logging**: Console output (INFO+) and debug.log file (DEBUG+), third-party libraries suppressed from console
- **Database**: SQLite persistent chat history with configurable inference limits
- **Continuous Chronicling**: Automatic chronicling triggered when unchronicled messages exceed `history_size` threshold. Uses `chronicler.model` to summarize conversation activity into Chronicle chapters. Messages get linked via `chapter_id` field in ChatHistory. Includes safety limits (100 message batches, 7-day lookback) and overlap for context continuity
- **Proactive Interjecting**: Channel-based whitelist feature using claude-3-haiku to scan non-directed messages and interject in serious conversations when useful. Includes rate limiting, test mode, and channel whitelisting
- **Key Modules**:
  - `rooms/irc/monitor.py` - IRCRoomMonitor (main IRC message processing, command handling, mode classification)
  - `rooms/irc/varlink.py` - VarlinkClient (events), VarlinkSender (messages)
  - `rooms/irc/autochronicler.py` - AutoChronicler (automatic chronicling of IRC messages when threshold exceeded)
  - `rooms/proactive.py` - ProactiveDebouncer (channel-based proactive interjecting)
  - `history.py` - ChatHistory (persistent SQLite storage)
  - `providers/` - async API clients (anthropic, openai, perplexity) and base classes
  - `rate_limiter.py` - RateLimiter
  - `agentic_actor/` - AgenticLLMActor multi-turn mode with tool system for web search, webpage visiting, and Python code execution

## Code Style
- **Language**: Python 3.11+ with modern type hints (`dict`, `list`, ...), following PEP8
- **Async**: Full async/await support for non-blocking message processing
- **Imports**: Standard library first, then third-party (`aiohttp`, `aiosqlite`), local modules
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Brief docstrings for classes and key methods
- **Error Handling**: Write code that fails fast. No defensive try-except blocks. Only catch exceptions when there's a clear recovery strategy.
- **Testing**: Pytest with async support for behavioral tests

## Notes for contributors
- Tests should avoid mocking low-level API client constructors when validating control flow. Prefer patching router calls to inject fake responses, and ensure provider configs are referenced via `providers.*`.
- Do NOT introduce compatibility shims for legacy config fields; update tests and fixtures instead.
- When changing tests, prefer modifying/extending existing test files and cases rather than adding new test files, unless there is a compelling reason.
