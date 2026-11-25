"""Training reads its features from the feature store, point-in-time: the
source values, never a renewal that took effect later."""

from __future__ import annotations

import dataclasses

import duckdb
import pandas as pd

from src.feature_store import offline
from src.feature_store.registry import load_registry
from src.features import build_features


def test_training_claims_read_the_source_snapshot(claims, raw_policies, cfg, tmp_path, monkeypatch):
    warehouse = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(warehouse)) as con:
        con.execute("CREATE SCHEMA staging")
        con.register("claims_df", claims)
        con.execute("CREATE TABLE staging.claims AS SELECT * FROM claims_df")
    cfg["paths"]["warehouse"] = str(warehouse)

    registry = dataclasses.replace(load_registry(), offline=tmp_path / "offline.duckdb")
    offline.write(offline.build(raw_policies, registry), registry.offline)
    monkeypatch.setattr(build_features, "load_registry", lambda **_: registry)

    loaded = build_features.load_claims(cfg)
    expected = claims.sort_values("claim_id", ignore_index=True)
    assert list(loaded.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(loaded, expected, check_dtype=False)
