"""Online store: materialized as of a time, read by policy_id, and checked
against the offline store's point-in-time read (no train/serve skew)."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime

import pandas as pd
import pytest

from src.feature_store import materialize, offline
from src.feature_store.online import TABLE, OnlineStore
from src.feature_store.registry import load_registry


@pytest.fixture
def registry(raw_policies, tmp_path):
    reg = dataclasses.replace(
        load_registry(), offline=tmp_path / "offline.duckdb", online=tmp_path / "online.db"
    )
    offline.write(offline.build(raw_policies, reg), reg.offline)
    return reg


def renewal_of(registry, policy_id: int) -> pd.Timestamp:
    with offline.connect(registry) as con:
        return con.execute(
            f"SELECT max(valid_from) FROM {offline.TABLE} WHERE policy_id = ?", [policy_id]
        ).fetchone()[0]


def test_online_store_holds_the_version_in_force(registry, raw_policies):
    policy = int(raw_policies["policy_id"].iloc[0])
    renewed = renewal_of(registry, policy)
    source = raw_policies.set_index("policy_id").loc[policy]

    rows = materialize.latest_versions(registry, renewed - pd.Timedelta(seconds=1))
    materialize.write_online(rows, registry.online, datetime(2025, 1, 1))
    store = OnlineStore(registry.online)
    before = store.get(policy)
    store.close()
    assert before is not None and before["driver_age"] == source["driver_age"]

    rows = materialize.latest_versions(registry, renewed)
    materialize.write_online(rows, registry.online, renewed)
    store = OnlineStore(registry.online)
    after = store.get(policy)
    assert after is not None and after["driver_age"] == source["driver_age"] + 1
    assert store.get(123_456_789) is None
    assert store.info()["row_count"] == len(raw_policies)
    store.close()


def test_online_types_survive_sqlite(registry, raw_policies):
    rows = materialize.latest_versions(registry, datetime(2026, 1, 1))
    materialize.write_online(rows, registry.online, datetime(2026, 1, 1))
    store = OnlineStore(registry.online)
    row = store.get(int(raw_policies["policy_id"].iloc[0]))
    store.close()
    assert row is not None
    assert isinstance(row["vehicle_power"], int)
    assert isinstance(row["exposure"], float)
    assert isinstance(row["region"], str)


@pytest.mark.parametrize(
    "as_of", [datetime(2024, 6, 1), datetime(2025, 6, 1), datetime(2026, 1, 1)]
)
def test_no_skew_between_online_and_offline(registry, as_of):
    materialize.write_online(materialize.latest_versions(registry, as_of), registry.online, as_of)
    report = materialize.check_skew(registry, as_of)
    assert report["passed"], report


def test_skew_check_catches_a_stale_online_store(registry):
    old, new = datetime(2024, 6, 1), datetime(2026, 1, 1)
    materialize.write_online(materialize.latest_versions(registry, old), registry.online, old)
    report = materialize.check_skew(registry, new)
    assert not report["passed"]
    assert set(report["mismatched_fields"]) >= {"driver_age", "vehicle_age"}


def test_write_is_atomic(registry):
    as_of = datetime(2026, 1, 1)
    materialize.write_online(materialize.latest_versions(registry, as_of), registry.online, as_of)
    bad = pd.DataFrame({"policy_id": [1, 1], "valid_from": pd.to_datetime(["2024-01-01"] * 2)})
    with pytest.raises(sqlite3.IntegrityError):
        materialize.write_online(bad, registry.online, as_of)
    with sqlite3.connect(registry.online) as con:
        assert con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0] > 2


def test_missing_store_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make materialize"):
        OnlineStore(tmp_path / "nope.db")
