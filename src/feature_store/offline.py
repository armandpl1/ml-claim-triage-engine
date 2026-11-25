"""The offline store: every version of every policy's features, with the time
each version took effect (pipeline stage `feature_store`).

    policy_features  policy_id, valid_from, the stored fields
                     (one row per policy per version)

Two kinds of versions:
    snapshot  the source values (freMTPL2), recorded at history.recorded_at
    renewal   synthetic: each policy renews once in the following year; the
              vehicle and driver are a year older and the claim-free
              bonus-malus drops 5% (configs/feature_store.yaml)

Reads are point-in-time: `historical_features` gives each (policy_id,
event_timestamp) the version in force at that moment, with DuckDB's ASOF
join. A version that took effect after the event is never visible, so a
training set or a re-scored night never sees the future.

Usage:
    python -m src.feature_store.offline       # rebuild data/feature_store/offline.duckdb
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event
from src.feature_store.registry import Registry, load_registry
from src.ingestion.notes import stable_seed

logger = get_logger("feature_store.offline")
TABLE = "policy_features"
MAX_AGE = 100  # data contracts: driver_age and vehicle_age


class MissingFeatures(LookupError):
    """An entity had no feature version in force at its event time."""


def snapshot(policies: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    """The source values, in force from history.recorded_at."""
    view = registry.view
    out = policies[[view.entity, *view.fields]].copy()
    out.insert(1, "valid_from", pd.Timestamp(registry.recorded_at))
    return out


def renewals(snap: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    """One renewal version per policy, on a stable-hash day after `first`."""
    rule = registry.renewal
    first = datetime.combine(rule.first, datetime.min.time())
    days = [stable_seed("renewal", p) % rule.days for p in snap[registry.view.entity]]
    out = snap.copy()
    out["valid_from"] = [first + timedelta(days=d) for d in days]
    for column in ("vehicle_age", "driver_age"):
        out[column] = (out[column] + rule.age_step).clip(upper=MAX_AGE)
    lowered = np.floor(out["bonus_malus"] * rule.bonus_malus_factor)
    out["bonus_malus"] = lowered.clip(lower=rule.bonus_malus_floor).astype(out["bonus_malus"].dtype)
    out["valid_from"] = pd.to_datetime(out["valid_from"])
    return out


def build(policies: pd.DataFrame, registry: Registry) -> pd.DataFrame:
    """Every version, sorted by policy and time."""
    snap = snapshot(policies, registry)
    versions = pd.concat([snap, renewals(snap, registry)], ignore_index=True)
    return versions.sort_values([registry.view.entity, "valid_from"], ignore_index=True)


def write(versions: pd.DataFrame, path: Path) -> None:
    """Replace the store file atomically (a failed build keeps the old one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        con.register("versions", versions)
        con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM versions")
    tmp.replace(path)


def historical_features(
    con: duckdb.DuckDBPyConnection,
    entities: pd.DataFrame,
    registry: Registry,
    timestamp: str = "event_timestamp",
    allow_missing: bool = False,
) -> pd.DataFrame:
    """`entities` (with policy_id and `timestamp`) plus the feature version in
    force at each row's time, and its `feature_valid_from`. Same rows, same
    order. Raises MissingFeatures if a row has no version (unless allowed)."""
    view, key = registry.view, registry.view.entity
    left = entities.reset_index(drop=True).copy()
    left["_row"] = np.arange(len(left))
    left[timestamp] = pd.to_datetime(left[timestamp])
    clash = set(view.fields) & set(left.columns)
    if clash:
        raise ValueError(f"entities already have feature columns {sorted(clash)}")
    con.register("_entities", left)
    try:
        fields = ", ".join(f"f.{c}" for c in view.fields)
        out = con.execute(
            f"""SELECT e.*, {fields}, f.valid_from AS feature_valid_from
                FROM _entities e ASOF LEFT JOIN {TABLE} f
                  ON e.{key} = f.{key} AND e.{timestamp} >= f.valid_from
                ORDER BY e._row"""
        ).df()
    finally:
        con.unregister("_entities")
    missing = out["feature_valid_from"].isna()
    if missing.any() and not allow_missing:
        sample = out.loc[missing, key].head(5).tolist()
        raise MissingFeatures(f"{int(missing.sum())} rows have no features, e.g. {key} {sample}")
    return out.drop(columns="_row")


def connect(registry: Registry | None = None) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the offline store."""
    path = (registry or load_registry()).offline
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make feature-store` (or `make data`)")
    return duckdb.connect(str(path), read_only=True)


def run() -> int:
    registry = load_registry()
    source = resolve(load_config()["paths"]["warehouse"])
    with duckdb.connect(str(source), read_only=True) as con:
        policies = con.sql(f"SELECT * FROM {registry.view.source} ORDER BY policy_id").df()
    versions = build(policies, registry)
    write(versions, registry.offline)
    log_event(
        logger,
        logging.INFO,
        "offline_store_built",
        path=str(registry.offline),
        policies=int(policies[registry.view.entity].nunique()),
        versions=len(versions),
        renewals_from=str(versions["valid_from"].nlargest(len(policies)).min().date()),
        renewals_to=str(versions["valid_from"].max().date()),
    )
    return len(versions)


if __name__ == "__main__":
    run()
