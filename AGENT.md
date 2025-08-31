# irssi-llmagent Agent Guide

## Build/Test Commands
- Install dependencies: `uv sync --dev`
- Run service: `uv run irssi-llmagent` or `uv run python -m irssi_llmagent.main`
- Run tests: `uv run pytest`
- Full e2e test in CLI mode: `uv run irssi-llmagent --message "your message here"`
- Test automatic mode: `uv run irssi-llmagent --message "tell me a joke"`
- Test explicit sarcastic: `uv run irssi-llmagent --message "!S tell me a joke"`
- Test explicit serious: `uv run irssi-llmagent --message "!s search for Python news"`
- Test Python code execution: `uv run irssi-llmagent --message "execute python: print('hello world')"`
- Analyze classifier: `uv run python analyze_classifier.py --db chat_history.db`
- Analyze proactive interjecting: `uv run python analyze_proactive.py --limit 20`
- Analyze proactive with logs: `uv run python analyze_proactive.py --logs ~/.irssi/logs/ --limit 50 --exclude-news`
- Run linting: `uv run ruff check .`
- Run formatting: `uv run ruff format .`
- Run type checking: `uv run pyright`
- Install pre-commit hooks: `uv run pre-commit install`
- NEVER use `git add -A` blindly, there may be untracked files that must not be committed

## Architecture
- **Main Service**: `irssi_llmagent/main.py` - Async Python chatbot connecting via varlink to irssi
- **Modular Structure**: Split into separate modules for Claude, Perplexity, varlink, and history
- **Varlink Protocol**: Dual async socket architecture (events + sender) over UNIX socket at `~/.irssi/varlink.sock`
- **APIs**: Anthropic Claude (sarcastic/serious modes with automatic classification using claude-3-5-haiku), Perplexity AI, E2B sandbox for Python code execution
- **Config**: JSON-based configuration in `config.json` (copy from `config.json.example`)
  - Models MUST be fully-qualified as `provider:model` (e.g., `anthropic:claude-sonnet-4`). No defaults.
  - No backwards compatibility is kept for legacy config keys; tests are aligned to the new schema.
- **Logging**: Console output (INFO+) and debug.log file (DEBUG+), third-party libraries suppressed from console
- **Database**: SQLite persistent chat history with configurable inference limits
- **Proactive Interjecting**: Channel-based whitelist feature using claude-3-haiku to scan non-directed messages and interject in serious conversations when useful. Includes rate limiting, test mode, and channel whitelisting
- **Key Modules**:
  - `rooms/irc/varlink.py` - VarlinkClient (events), VarlinkSender (messages)
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
- **Error Handling**: Try/catch with logging, graceful degradation
- **Type Safety**: Type hints and pyright type checking enabled
- **Linting**: Ruff for code quality and formatting
- **Testing**: Pytest with async support for behavioral tests

## Notes for contributors
- Tests should avoid mocking low-level API client constructors when validating control flow. Prefer patching router calls to inject fake responses, and ensure provider configs are referenced via `providers.*`.
- Do NOT introduce compatibility shims for legacy config fields; update tests and fixtures instead.
- When changing tests, prefer modifying/extending existing test files and cases rather than adding new test files, unless there is a compelling reason.
