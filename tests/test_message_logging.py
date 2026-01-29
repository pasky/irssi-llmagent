"""Tests for per-message logging utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from muaddib.message_logging import MessageContextHandler


def test_system_log_is_per_date(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    handler = MessageContextHandler(logs_dir)
    try:
        record = logging.LogRecord(
            name="muaddib.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        record.created = datetime(2024, 1, 2, 3, 4, 5).timestamp()

        handler.emit(record)

        expected_path = logs_dir / "2024-01-02" / "system.log"
        assert expected_path.exists()
        assert "hello world" in expected_path.read_text(encoding="utf-8")
        assert not (logs_dir / "system.log").exists()
    finally:
        handler.close()
