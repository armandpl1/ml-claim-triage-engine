"""The online store reader: each policy's latest features, by policy_id.

SQLite and the standard library only, so the slim serving image can import
it (no pandas, no DuckDB). Written by src/feature_store/materialize.py from
the offline store; the API opens it read-only.

The API copies the whole store into memory at startup (`in_memory=True`,
about 45 MB for 678,013 policies), like a Redis online store: lookups never
touch the disk. Reading the file directly, a lookup whose page had left the
OS cache waited on the disk: on the laptop's spinning drive, under load,
up to 1.5 s (step 6 load test). A new materialization is picked up by
restarting the pods.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

TABLE = "policy_features"
META = "materializations"


class OnlineStore:
    """Read-only lookups over one connection, shared by the API's worker
    threads; a lock serializes them (a lookup takes microseconds)."""

    def __init__(self, path: Path, in_memory: bool = False) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `make materialize`")
        self.path = path
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        if in_memory:
            self._con = sqlite3.connect(":memory:", check_same_thread=False)
            source.backup(self._con)  # one sequential read of the file
            source.close()
        else:
            self._con = source
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def get(self, policy_id: int) -> dict[str, Any] | None:
        """The policy's features and their valid_from, or None if unknown."""
        with self._lock:
            row = self._con.execute(
                f"SELECT * FROM {TABLE} WHERE policy_id = ?", (policy_id,)
            ).fetchone()
        return dict(row) if row else None

    def info(self) -> dict[str, Any]:
        """When the store was materialized, as of which time, and how many rows."""
        with self._lock:
            row = self._con.execute(
                f"SELECT as_of, materialized_at, row_count FROM {META}"
                " ORDER BY materialized_at DESC"
            ).fetchone()
        return dict(row) if row else {}

    def close(self) -> None:
        self._con.close()
