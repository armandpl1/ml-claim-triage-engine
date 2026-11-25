"""The inference log: every prediction, for the drift detector
(monitoring/drift_detector.py), in SQLite.

Writes are off the request path. A request puts its row on an in-memory
queue (microseconds) and one writer thread inserts rows in batches. Before
this, each request wrote its own row; under load the writes queued on
SQLite's file lock and on disk syncs, which slowed every request (Starlette
runs background tasks before the connection takes its next request) or,
without a lock, failed and closed client connections (step 6 load test).

A full queue or a failed write drops rows: they are counted
(claim_api_inference_log_errors_total) and logged, never raised. Losing a
log row must not fail a prediction.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from prometheus_client import Counter

from src.common.logging import get_logger, log_event

logger = get_logger("serving.inference_log")

LOG_ERRORS = Counter(
    "claim_api_inference_log_errors_total", "Inference log rows dropped (prediction kept)"
)

SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY, ts REAL, claim_id TEXT, features TEXT, adjuster_note TEXT,
    high_cost_probability REAL, route TEXT, note_category TEXT,
    tabular_version TEXT, notes_version TEXT, latency_ms REAL)"""
INSERT = (
    "INSERT INTO predictions VALUES (:request_id, :ts, :claim_id, :features, "
    ":adjuster_note, :high_cost_probability, :route, :note_category, "
    ":tabular_version, :notes_version, :latency_ms)"
)

_STOP = object()


class InferenceLog:
    """A queue and one writer thread. start() before use, stop() to flush."""

    def __init__(self, path: Path, max_queue: int = 10_000, batch_size: int = 500) -> None:
        self.path = path
        self.batch_size = batch_size
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as con:
            # WAL: the drift detector can read while the API writes.
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(SCHEMA)
            con.commit()
        self._thread = threading.Thread(target=self._run, name="inference-log", daemon=True)
        self._thread.start()

    def put(self, row: dict[str, Any]) -> None:
        """Queue one row. Never blocks: a full queue drops the row."""
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped(1, "queue full")

    def stop(self, timeout: float = 10.0) -> None:
        """Write what is queued, then stop the writer."""
        if self._thread is None:
            return
        self._queue.put(_STOP)  # blocking: the stop marker must get in
        self._thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        con = sqlite3.connect(self.path, timeout=10)
        # NORMAL is safe with WAL (no corruption); a power cut may lose the
        # last batch, acceptable for a monitoring log.
        con.execute("PRAGMA synchronous=NORMAL")
        try:
            stopping = False
            while not stopping:
                batch = [self._queue.get()]  # wait for the first row
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if _STOP in batch:
                    stopping = True
                    batch = [r for r in batch if r is not _STOP]
                if batch:
                    self._write(con, batch)
        finally:
            con.close()

    def _write(self, con: sqlite3.Connection, batch: list[dict[str, Any]]) -> None:
        try:
            con.executemany(INSERT, batch)
            con.commit()
        except sqlite3.Error as exc:
            con.rollback()
            self._dropped(len(batch), str(exc))

    def _dropped(self, rows: int, reason: str) -> None:
        LOG_ERRORS.inc(rows)
        log_event(logger, logging.WARNING, "inference_log_rows_dropped", rows=rows, reason=reason)
