# irssi-llmagent

A modern Python-based chatbot service that connects to irssi via varlink protocol, featuring async architecture, persistent history, and AI integrations.

## Features

- **Async Architecture**: Non-blocking message processing with concurrent handling
- **Persistent History**: SQLite database with unlimited storage and configurable inference limits
- **AI Integrations**:
  - Anthropic Claude (sarcastic and serious modes with automatic mode selection)
  - Perplexity AI for web search
  - Haiku 3 preprocessing model for intelligent mode classification
- **Modern Python**: Built with uv, type safety, and comprehensive testing
- **Rate Limiting**: Configurable rate limiting and user management
- **Command System**: Extensible command-based interaction (!h, !s, !p)
- **Proactive Interjecting**: Channel-based whitelist system for automatic participation in relevant conversations
- **Developer Tools**: Pre-commit hooks, linting, formatting, and type checking

## Installation

1. Ensure `irssi-varlink` is loaded in your irssi
2. Install dependencies: `uv sync --dev`
3. Copy `config.json.example` to `config.json` and configure your API keys
4. Run the service: `uv run irssi-llmagent`

## Development

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Install pre-commit hooks
uv run pre-commit install
```

## Configuration

Edit `config.json` to set:
- Anthropic API key and models
- Perplexity API key and models
- Rate limiting settings
- History size (for inference)
- Database path
- Ignored users
- Proactive interjecting settings (channel whitelist, rate limits, test mode)

## Commands

- `mynick: message` - Automatic mode (Haiku 3 classifier chooses sarcastic or serious)
- `mynick: !S message` - Explicit sarcastic Claude response
- `mynick: !s message` - Explicit serious agentic Claude with web search and URL tools
- `mynick: !p message` - Perplexity search response
- `mynick: !h` - Show help

### Mode Selection

The bot now features intelligent mode selection:
- **Automatic Mode**: Default behavior uses Haiku 3 to analyze the message context and automatically choose between sarcastic and serious modes
- **Explicit Modes**: Use `!S` for sarcastic or `!s` for serious mode to override automatic selection

## CLI Testing Mode

You can test the bot's message handling including command parsing from the command line:

```bash
# Test automatic mode (classifier chooses sarcastic or serious)
uv run irssi-llmagent --message "tell me a joke"

# Test explicit sarcastic mode
uv run irssi-llmagent --message "!S tell me a joke"

# Test explicit serious agentic mode with web search
uv run irssi-llmagent --message "!s search for latest Python news"

# Test Perplexity mode
uv run irssi-llmagent --message "!p what's the weather in Paris?"

# Test help command
uv run irssi-llmagent --message "!h"

# Test with a specific config file
uv run irssi-llmagent --message "!s summarize https://python.org" --config /path/to/config.json
```

This simulates full IRC message handling including command parsing and automatic mode classification, useful for testing your configuration and API keys without setting up the full IRC bot.

## Classifier Analysis

Evaluate the performance of the automatic mode classifier on historical data:

```bash
# Analyze classifier performance on database history
uv run python analyze_classifier.py --db chat_history.db

# Analyze classifier performance on IRC log files
uv run python analyze_classifier.py --logs ~/.irssi/logs/freenode/*.log

# Combine both sources with custom config
uv run python analyze_classifier.py --db chat_history.db --logs ~/.irssi/logs/ --config config.json
```

Results are saved to `classifier_analysis.csv` with detailed metrics and misclassification analysis.

## Proactive Interjecting Analysis

Evaluate the performance of the proactive interjecting feature on historical data:

```bash
# Analyze proactive interjecting performance on database history
uv run python analyze_proactive.py --limit 20

# Analyze proactive interjecting on IRC log files with channel exclusions
uv run python analyze_proactive.py --logs ~/.irssi/logs/ --limit 50 --exclude-news

# Combine both sources with custom config
uv run python analyze_proactive.py --db chat_history.db --logs ~/.irssi/logs/ --config config.json
```

Results are saved to `proactive_analysis.csv` with detailed interjection decisions and reasoning.

## Architecture

The service uses a modular async architecture:
- **Main Service**: `irssi_llmagent/main.py` - Core application logic
- **Varlink Module**: `irssi_llmagent/varlink.py` - IRC communication
- **History Module**: `irssi_llmagent/history.py` - Persistent SQLite storage
- **AI Clients**: Separate modules for Claude and Perplexity APIs
- **Rate Limiting**: Token bucket rate limiting implementation

The service connects to irssi via the varlink UNIX socket, processes IRC events asynchronously, and maintains persistent conversation history while respecting configurable rate limits.
