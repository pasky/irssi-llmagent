# irssi-llmagent

A modern Python-based agentic LLM chatbot service that connects to irssi via varlink protocol.

## Features

- **AI Integrations**: Anthropic Claude, OpenAI, DeepSeek, any OpenRouter model, Perplexity AI
- **Agentic Capability**: Ability to visit websites, view images, perform deep research, execute Python code, publish artifacts
- **Command System**: Extensible command-based interaction with prefixes for various modes
- **Proactive Interjecting**: Channel-based whitelist system for automatic participation in relevant conversations
- **Async Architecture**: Non-blocking message processing with concurrent handling
- **Persistent History**: SQLite database with unlimited storage and configurable inference limits
- **Modern Python**: Built with uv, type safety, and comprehensive testing
- **Rate Limiting**: Configurable rate limiting and user management

## Installation

1. Ensure `irssi-varlink` is loaded in your irssi
2. Install dependencies: `uv sync --dev`
3. Copy `config.json.example` to `config.json` and configure your API keys
4. Run the service: `uv run irssi-llmagent`

Alternatively, see `docs/docker.md` for running this + irssi in a Docker compose setup.

## Configuration

Edit `config.json` based on `config.json.example` to set:
- API keys
- Paths for tools
- Custom prompts for various modes
- IRC integration settings

## Commands

- `mynick: message` - Automatic mode
- `mynick: !h` - Show help and info about other modes

## CLI Testing Mode

You can test the bot's message handling including command parsing from the command line:

```bash
uv run irssi-llmagent --message "!h"
uv run irssi-llmagent --message "tell me a joke"
uv run irssi-llmagent --message "!d tell me a joke"
uv run irssi-llmagent --message "!a summarize https://python.org" --config /path/to/config.json
```

This simulates full IRC message handling including command parsing and automatic mode classification, useful for testing your configuration and API keys without setting up the full IRC bot.

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

## Evaluation of some internal components

### Classifier Analysis

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

### Proactive Interjecting Analysis

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
