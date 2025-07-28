# irssi-llmagent Agent Guide

## Build/Test Commands
- Install dependencies: `uv sync --dev`
- Run service: `uv run irssi-llmagent` or `uv run python -m irssi_llmagent.main`
- Run tests: `uv run pytest`
- Full e2e test in CLI mode: `uv run irssi-llmagent --message "your message here"`
- Test automatic mode: `uv run irssi-llmagent --message "tell me a joke"`
- Test explicit sarcastic: `uv run irssi-llmagent --message "!S tell me a joke"`
- Test explicit serious: `uv run irssi-llmagent --message "!s search for Python news"`
- Analyze classifier: `uv run python analyze_classifier.py --db chat_history.db`
- Run linting: `uv run ruff check .`
- Run formatting: `uv run ruff format .`
- Run type checking: `uv run pyright`
- Install pre-commit hooks: `uv run pre-commit install`
- NEVER use `git add -A` blindly, there may be untracked files that must not be committed

## Architecture
- **Main Service**: `irssi_llmagent/main.py` - Async Python chatbot connecting via varlink to irssi
- **Modular Structure**: Split into separate modules for Claude, Perplexity, varlink, and history
- **Varlink Protocol**: Dual async socket architecture (events + sender) over UNIX socket at `~/.irssi/varlink.sock`
- **APIs**: Anthropic Claude (sarcastic/serious modes with automatic classification), Perplexity AI
- **Config**: JSON-based configuration in `config.json` (copy from `config.json.example`)
- **Database**: SQLite persistent chat history with configurable inference limits
- **Key Modules**:
  - `varlink.py` - VarlinkClient (events), VarlinkSender (messages)
  - `history.py` - ChatHistory (persistent SQLite storage)
  - `claude.py` - AnthropicClient (async API client)
  - `perplexity.py` - PerplexityClient (async API client)
  - `rate_limiter.py` - RateLimiter

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
