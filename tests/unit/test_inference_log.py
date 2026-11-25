"""The inference log: concurrent writes all land, a full queue or a failed
write drops rows without raising, and stop() flushes the queue."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

from src.serving.inference_log import LOG_ERRORS, InferenceLog


def row(i: int) -> dict:
    return {
        "request_id": f"r{i}",
        "ts": float(i),
        "claim_id": None,
        "features": "{}",
        "adjuster_note": None,
        "high_cost_probability": 0.1,
        "route": "standard",
        "note_category": None,
        "tabular_version": "1",
        "notes_version": None,
        "latency_ms": 1.0,
    }


def count(path) -> int:
    with sqlite3.connect(path) as con:
        return con.execute("SELECT count(*) FROM predictions").fetchone()[0]


def test_concurrent_writes_all_land_after_stop(tmp_path):
    log = InferenceLog(tmp_path / "log.db")
    log.start()
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: log.put(row(i)), range(2_000)))
    log.stop()
    assert count(tmp_path / "log.db") == 2_000
    with sqlite3.connect(tmp_path / "log.db") as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_full_queue_drops_and_counts(tmp_path):
    log = InferenceLog(tmp_path / "log.db", max_queue=3)  # writer not started
    before = LOG_ERRORS._value.get()
    for i in range(5):
        log.put(row(i))
    assert LOG_ERRORS._value.get() == before + 2


def test_failed_batch_is_counted_and_the_writer_keeps_going(tmp_path):
    log = InferenceLog(tmp_path / "log.db")
    log.start()
    before = LOG_ERRORS._value.get()
    log.put(row(1))
    log.put(row(1))  # duplicate primary key: that batch fails
    log.stop()
    log.start()
    log.put(row(2))
    log.stop()
    assert LOG_ERRORS._value.get() > before
    assert count(tmp_path / "log.db") >= 1
