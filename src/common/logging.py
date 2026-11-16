"""Structured (JSON lines) logging, so alerts can be parsed by machines.

Every record is one JSON object on stderr, with a timestamp, level, logger
name, event and any extra fields passed through `extra={"fields": {...}}`.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


class _CurrentStderrHandler(logging.StreamHandler[TextIO]):
    """Writes to whatever sys.stderr is at the time of each record, so
    redirection (test capture, process managers) is respected."""

    @property
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _CurrentStderrHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log an event with structured fields."""
    logger.log(level, event, extra={"fields": fields})
