"""Path resolution for muaddib configuration and data files.

Supports two modes:
1. MUADDIB_HOME environment variable - everything under that directory
2. Default ~/.muaddib directory

Structure:
  $MUADDIB_HOME/ or ~/.muaddib/
    ├── config.json
    ├── chat_history.db
    ├── chronicle.db
    └── logs/
"""

import os
from pathlib import Path

_muaddib_home: Path | None = None


def get_muaddib_home() -> Path:
    """Get the muaddib home directory.

    Uses MUADDIB_HOME environment variable if set, otherwise ~/.muaddib.
    Creates the directory if it doesn't exist.
    """
    global _muaddib_home
    if _muaddib_home is not None:
        return _muaddib_home

    if env_home := os.environ.get("MUADDIB_HOME"):
        _muaddib_home = Path(env_home).expanduser().resolve()
    else:
        _muaddib_home = Path.home() / ".muaddib"

    _muaddib_home.mkdir(parents=True, exist_ok=True)
    return _muaddib_home


def get_config_path() -> Path:
    """Get the path to config.json."""
    return get_muaddib_home() / "config.json"


def get_default_history_db_path() -> Path:
    """Get the default path to chat_history.db."""
    return get_muaddib_home() / "chat_history.db"


def get_default_chronicle_db_path() -> Path:
    """Get the default path to chronicle.db."""
    return get_muaddib_home() / "chronicle.db"


def get_logs_dir() -> Path:
    """Get the path to the logs directory."""
    logs_dir = get_muaddib_home() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def reset_cached_home() -> None:
    """Reset the cached home directory (for testing)."""
    global _muaddib_home
    _muaddib_home = None
