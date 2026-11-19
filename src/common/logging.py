"""Structured (JSON lines) logging, from the shared data-ml-toolkit SDK.

Every record is one JSON object on stderr, with a timestamp, level, logger
name, event and any extra fields passed to `log_event`. Kept as a module so
existing `src.common.logging` imports keep working.
"""

from data_ml_toolkit.logging import JsonFormatter, get_logger, log_event

__all__ = ["JsonFormatter", "get_logger", "log_event"]
