"""The batch store: one DuckDB file holding what the nightly jobs write.

    predictions  one row per scored claim (written by src.batch.score)
    reasons      each scored claim's top risk-raising features (src.batch.score,
                 from src.explain), up to explain.top_k rows per claim
    outcomes     one row per claim whose final cost is known (src.batch.labels)
    batch_runs   one row per job run, for auditing

Every write replaces a whole slice (a report date, or everything known by a
date) inside one transaction, so a job can be re-run safely.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    claim_id      VARCHAR PRIMARY KEY,
    report_date   DATE NOT NULL,
    scored_at     TIMESTAMP NOT NULL,
    model_name    VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    probability   DOUBLE NOT NULL,
    threshold     DOUBLE NOT NULL,
    route         VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS reasons (
    claim_id      VARCHAR NOT NULL,
    report_date   DATE NOT NULL,
    rank          INTEGER NOT NULL,
    feature       VARCHAR NOT NULL,
    value         VARCHAR NOT NULL,
    contribution  DOUBLE NOT NULL,
    PRIMARY KEY (claim_id, rank)
);
CREATE TABLE IF NOT EXISTS outcomes (
    claim_id      VARCHAR PRIMARY KEY,
    label_date    DATE NOT NULL,
    claim_amount  DOUBLE NOT NULL,
    is_high_cost  BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_runs (
    job        VARCHAR NOT NULL,
    run_for    DATE NOT NULL,
    ran_at     TIMESTAMP NOT NULL,
    row_count  INTEGER NOT NULL,
    detail     VARCHAR
);
"""


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con
