"""Materialize the online store from the offline store, then check for skew.

    materialize  for every policy, the version in force at `as_of` (default:
                 now) goes into SQLite (src/feature_store/online.py), keyed
                 by policy_id. The file is replaced atomically, so the API
                 never reads a half-written store.
    skew check   every policy's online features must equal the offline
                 store's point-in-time read at the same `as_of`: the API and
                 a training set built for that moment see the same values.
                 A mismatch logs one alert and exits non-zero.

Report: reports/feature_store/skew.json.

Usage:
    python -m src.feature_store.materialize                     # as of now
    python -m src.feature_store.materialize --as-of 2025-02-01
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ROOT
from src.common.logging import get_logger, log_event
from src.feature_store import offline
from src.feature_store.online import META, TABLE, OnlineStore
from src.feature_store.registry import Registry, load_registry

logger = get_logger("feature_store.materialize")
REPORT = ROOT / "reports/feature_store/skew.json"


def latest_versions(registry: Registry, as_of: datetime) -> pd.DataFrame:
    """Each policy's version in force at `as_of` (policies not yet recorded are left out)."""
    view = registry.view
    with offline.connect(registry) as con:
        return con.execute(
            f"""SELECT {view.entity}, valid_from, {", ".join(view.fields)}
                FROM {offline.TABLE} WHERE valid_from <= ?
                QUALIFY row_number() OVER (PARTITION BY {view.entity} ORDER BY valid_from DESC) = 1
                ORDER BY {view.entity}""",
            [as_of],
        ).df()


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def write_online(rows: pd.DataFrame, path: Path, as_of: datetime) -> None:
    """Replace the online store with `rows`, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    data = rows.copy()
    data["valid_from"] = data["valid_from"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    columns = ", ".join(
        f"{c} {_sqlite_type(data[c])}{' PRIMARY KEY' if c == 'policy_id' else ''}"
        for c in data.columns
    )
    con = sqlite3.connect(tmp)
    try:
        con.execute(f"CREATE TABLE {TABLE} ({columns})")
        marks = ", ".join("?" * len(data.columns))
        con.executemany(
            f"INSERT INTO {TABLE} VALUES ({marks})",
            data.astype(object).itertuples(index=False, name=None),
        )
        con.execute(f"CREATE TABLE {META} (as_of TEXT, materialized_at TEXT, row_count INTEGER)")
        con.execute(
            f"INSERT INTO {META} VALUES (?, ?, ?)",
            [as_of.isoformat(), datetime.now().isoformat(timespec="seconds"), len(data)],
        )
        con.commit()
    finally:
        con.close()
    tmp.replace(path)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(float(a), float(b))
    return bool(a == b)


def check_skew(registry: Registry, as_of: datetime) -> dict[str, Any]:
    """Compare every policy in the online store with a point-in-time offline read."""
    store = OnlineStore(registry.online)
    try:
        online = pd.read_sql(f"SELECT * FROM {TABLE}", store._con)
    finally:
        store.close()
    entities = pd.DataFrame({registry.view.entity: online[registry.view.entity]})
    entities["event_timestamp"] = pd.Timestamp(as_of)
    with offline.connect(registry) as con:
        expected = offline.historical_features(con, entities, registry, allow_missing=True)
    mismatched: dict[str, int] = {}
    for field in registry.view.fields:
        bad = sum(
            not _same(a, b)
            for a, b in zip(online[field].tolist(), expected[field].tolist(), strict=True)
        )
        if bad:
            mismatched[field] = bad
    return {
        "as_of": as_of.isoformat(),
        "policies_checked": len(online),
        "mismatched_fields": mismatched,
        "passed": not mismatched,
    }


def run(as_of: datetime | None = None) -> dict[str, Any]:
    registry = load_registry()
    as_of = as_of or datetime.now()
    rows = latest_versions(registry, as_of)
    write_online(rows, registry.online, as_of)
    renewed = int((rows["valid_from"] > pd.Timestamp(registry.recorded_at)).sum())
    log_event(
        logger,
        logging.INFO,
        "online_store_materialized",
        path=str(registry.online),
        as_of=as_of.isoformat(),
        policies=len(rows),
        renewed=renewed,
    )
    report = check_skew(registry, as_of)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    level = logging.INFO if report["passed"] else logging.ERROR
    log_event(logger, level, "feature_skew_checked", **report)
    if not report["passed"]:
        raise SystemExit(1)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    run(parser.parse_args(argv).as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
