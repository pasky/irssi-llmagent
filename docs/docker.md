# Docker Setup for muaddib

## Quick Start

1. **Create persistent data directory:**
   ```bash
   mkdir -p irssi-data
   ```

2. **Copy your config:**
   ```bash
   cp config.json.example config.json
   # Edit config.json with your API keys
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

- **Persistent data:** `./irssi-data/` (bind-mounted to `/home/irssi/.irssi/`)
- **Config file:** `./config.json` (mounted into container)
- **Chat history:** `./chat_history.db` (persistent)
- **Source code:** `./` (mounted for development)

## Troubleshooting

- Check varlink socket: `ls -la ./irssi-data/varlink.sock`
- View muaddib logs: `docker-compose logs muaddib`
- Connect to muaddib container: `docker exec -it muaddib bash`
