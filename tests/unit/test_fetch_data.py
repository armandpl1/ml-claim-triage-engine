"""Ingestion: cleaning, deterministic claim IDs, orphan handling, idempotency."""

from __future__ import annotations

import pandas as pd

from src.ingestion.fetch_data import build_claims, clean_amounts, clean_policies


def test_clean_policies_strips_quotes_and_renames():
    raw = pd.DataFrame(
        {
            "IDpol": [1.0, 3.0],
            "ClaimNb": [1, 1],
            "Exposure": [0.1, 0.77],
            "Area": ["D", "B"],
            "VehPower": [5, 6],
            "VehAge": [0, 2],
            "DrivAge": [55, 52],
            "BonusMalus": [50, 50],
            "VehBrand": ["B12", "B6"],
            "VehGas": ["'Regular'", "'Diesel'"],
            "Density": [1217, 54],
            "Region": ["R82", "R22"],
        }
    )
    clean = clean_policies(raw)
    assert list(clean["fuel_type"]) == ["Regular", "Diesel"]
    assert clean["policy_id"].dtype == "int64"
    assert "driver_age" in clean.columns


def test_clean_amounts_casts_types():
    clean = clean_amounts(pd.DataFrame({"IDpol": [1552], "ClaimAmount": [995.2]}))
    assert list(clean.columns) == ["policy_id", "claim_amount"]
    assert clean["policy_id"].dtype == "int64"


def test_orphan_claims_are_dropped_and_counted(raw_policies, raw_amounts, cfg):
    claims, orphans = build_claims(raw_policies, raw_amounts, cfg)
    assert orphans == 2
    assert len(claims) == len(raw_amounts) - 2
    assert claims["policy_id"].isin(raw_policies["policy_id"]).all()


def test_claim_ids_are_unique_and_formatted(claims):
    assert claims["claim_id"].is_unique
    assert claims["claim_id"].str.match(r"^CLM-\d{7}-\d{2}$").all()


def test_claims_numbered_by_amount_within_policy(claims):
    for _, group in claims.groupby("policy_id"):
        ordered = group.sort_values("claim_seq")
        assert list(ordered["claim_seq"]) == list(range(1, len(group) + 1))
        assert ordered["claim_amount"].is_monotonic_increasing


def test_build_is_deterministic_and_order_independent(raw_policies, raw_amounts, cfg):
    first, _ = build_claims(raw_policies, raw_amounts, cfg)
    shuffled = raw_amounts.sample(frac=1.0, random_state=7)
    second, _ = build_claims(raw_policies.sample(frac=1.0, random_state=3), shuffled, cfg)
    # Same claim IDs, targets and notes regardless of the source row order.
    # (Equal amounts within one policy are the only order-dependent case.)
    cols = ["claim_id", "claim_amount", "is_high_cost", "claim_category", "adjuster_note"]
    pd.testing.assert_frame_equal(first[cols], second[cols])


def test_high_cost_flag_follows_threshold(claims, cfg):
    threshold = cfg["targets"]["high_cost_threshold"]
    assert (claims["is_high_cost"] == (claims["claim_amount"] > threshold)).all()
