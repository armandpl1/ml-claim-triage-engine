"""The offline store: versions over time, and point-in-time reads that never
see a version from after the event."""

from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd
import pytest

from src.feature_store import offline
from src.feature_store.registry import load_registry


@pytest.fixture
def registry():
    return load_registry()


@pytest.fixture
def store(raw_policies, registry, tmp_path):
    path = tmp_path / "offline.duckdb"
    offline.write(offline.build(raw_policies, registry), path)
    with duckdb.connect(str(path), read_only=True) as con:
        yield con


def renewal_dates(store) -> pd.Series:
    rows = store.sql(
        f"SELECT policy_id, max(valid_from) AS renewed FROM {offline.TABLE} GROUP BY 1"
    ).df()
    return rows.set_index("policy_id")["renewed"]


def test_two_versions_per_policy(raw_policies, registry):
    versions = offline.build(raw_policies, registry)
    assert len(versions) == 2 * len(raw_policies)
    assert versions.groupby("policy_id").size().eq(2).all()
    renewed = versions[versions["valid_from"] > pd.Timestamp(registry.recorded_at)]
    assert renewed["valid_from"].min() >= pd.Timestamp(registry.renewal.first)


def test_renewal_rule(raw_policies, registry):
    snap = offline.snapshot(raw_policies, registry)
    new = offline.renewals(snap, registry)
    assert (new["driver_age"] == (snap["driver_age"] + 1).clip(upper=100)).all()
    assert (new["vehicle_age"] == (snap["vehicle_age"] + 1).clip(upper=100)).all()
    assert (new["bonus_malus"] <= snap["bonus_malus"]).all()
    assert (new["bonus_malus"] >= 50).all()
    assert (new["region"] == snap["region"]).all()


def test_renewal_dates_are_deterministic(raw_policies, registry):
    a = offline.renewals(offline.snapshot(raw_policies, registry), registry)
    b = offline.renewals(offline.snapshot(raw_policies.iloc[::-1], registry), registry)
    merged = a.merge(b, on="policy_id")
    assert (merged["valid_from_x"] == merged["valid_from_y"]).all()


def test_point_in_time_read(store, raw_policies, registry):
    renewed = renewal_dates(store)
    policy = int(renewed.index[0])
    when = renewed[policy]
    source = raw_policies.set_index("policy_id").loc[policy]
    entities = pd.DataFrame(
        {
            "policy_id": [policy, policy, policy],
            "event_timestamp": [when - pd.Timedelta(seconds=1), when, when + pd.Timedelta(days=30)],
        }
    )
    out = offline.historical_features(store, entities, registry)
    assert out["driver_age"].tolist() == [source["driver_age"], *[source["driver_age"] + 1] * 2]
    assert out["feature_valid_from"].iloc[0] == pd.Timestamp(registry.recorded_at)


def test_read_keeps_row_order_and_extra_columns(store, raw_policies, registry):
    entities = pd.DataFrame(
        {
            "policy_id": raw_policies["policy_id"].iloc[::-1].to_numpy(),
            "event_timestamp": datetime(2024, 6, 1),
            "claim_id": [f"C{i}" for i in range(len(raw_policies))],
        }
    )
    out = offline.historical_features(store, entities, registry)
    assert out["claim_id"].tolist() == entities["claim_id"].tolist()
    expected = raw_policies.set_index("policy_id").loc[entities["policy_id"], "bonus_malus"]
    assert out["bonus_malus"].tolist() == expected.tolist()


def test_event_before_any_version_fails(store, registry):
    entities = pd.DataFrame({"policy_id": [1], "event_timestamp": [datetime(2020, 1, 1)]})
    with pytest.raises(offline.MissingFeatures):
        offline.historical_features(store, entities, registry)
    out = offline.historical_features(store, entities, registry, allow_missing=True)
    assert out["driver_age"].isna().all()


def test_unknown_policy_fails(store, registry):
    entities = pd.DataFrame({"policy_id": [123_456_789], "event_timestamp": [datetime(2025, 6, 1)]})
    with pytest.raises(offline.MissingFeatures, match="123456789"):
        offline.historical_features(store, entities, registry)
