# Docker Setup for muaddib

## Quick Start

1. **Create persistent data directories:**
   ```bash
   mkdir -p irssi-data muaddib-data
   ```

2. **Copy your config:**
   ```bash
   cp config.json.example muaddib-data/config.json
   # Edit muaddib-data/config.json with your API keys
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Connect to irssi:**
   ```bash
   docker exec -it irssi-chat bash
   sudo -u irssi irssi
   ```

## Development Workflow

### Restart muaddib during development:
```bash
# Quick restart (preserves irssi session)
docker-compose restart muaddib

# Or rebuild and restart if you changed dependencies
docker-compose up --build -d muaddib

# View logs
docker-compose logs -f muaddib
```

### Full restart:
```bash
docker-compose down
docker-compose up -d
```

## Configuration

Muaddib stores all its data in `$MUADDIB_HOME` (set via environment variable or defaults to `~/.muaddib/`).

For Docker, mount a local directory as the muaddib home:

- **Muaddib data:** `./muaddib-data/` (mounted as `$MUADDIB_HOME`)
  - `config.json` - Configuration file
  - `chat_history.db` - Chat history database (auto-created)
  - `chronicle.db` - Chronicle database (auto-created)
  - `artifacts/` - Shared artifacts directory (auto-created)
  - `logs/` - Per-message log files (auto-created)
- **Irssi data:** `./irssi-data/` (bind-mounted to `/home/irssi/.irssi/`)
- **Source code:** `./` (mounted for development)

Note: Relative paths in `config.json` (like `"path": "chronicle.db"` or `"path": "artifacts"`) are resolved against `$MUADDIB_HOME`.

The default `docker-compose.yml` already sets this up:
```yaml
environment:
  - MUADDIB_HOME=/data
volumes:
  - ./muaddib-data:/data
```

## Troubleshooting

- Check varlink socket: `ls -la ./irssi-data/varlink.sock`
- View muaddib logs: `docker-compose logs muaddib`
- Connect to muaddib container: `docker exec -it muaddib bash`
