# irssi-llmagent

A modern Python-based chatbot service that connects to irssi via varlink protocol, featuring async architecture, persistent history, and AI integrations.

## Features

- **Async Architecture**: Non-blocking message processing with concurrent handling
- **Persistent History**: SQLite database with unlimited storage and configurable inference limits
- **AI Integrations**:
  - Anthropic Claude (sarcastic and serious modes)
  - Perplexity AI for web search
- **Modern Python**: Built with uv, type safety, and comprehensive testing
- **Rate Limiting**: Configurable rate limiting and user management
- **Command System**: Extensible command-based interaction (!h, !s, !p)
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

## Commands

- `mynick: message` - Default sarcastic Claude response
- `mynick: !s message` - Serious Claude response
- `mynick: !p message` - Perplexity search response
- `mynick: !h` - Show help

## Architecture

The service uses a modular async architecture:
- **Main Service**: `irssi_llmagent/main.py` - Core application logic
- **Varlink Module**: `irssi_llmagent/varlink.py` - IRC communication
- **History Module**: `irssi_llmagent/history.py` - Persistent SQLite storage
- **AI Clients**: Separate modules for Claude and Perplexity APIs
- **Rate Limiting**: Token bucket rate limiting implementation

The service connects to irssi via the varlink UNIX socket, processes IRC events asynchronously, and maintains persistent conversation history while respecting configurable rate limits.
