"""The split is grouped, stratified, deterministic and order-independent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.split import split_claims


@pytest.fixture
def many_claims() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 3000
    policy = rng.integers(1, 1200, n)  # many policies have several claims
    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{p:07d}-{i:02d}" for i, p in enumerate(policy)],
            "policy_id": policy,
            "is_high_cost": rng.random(n) < 0.06,
        }
    )


def test_every_claim_gets_one_split(many_claims, cfg):
    split = split_claims(many_claims, cfg)
    assert set(split) == {"train", "validation", "test"}
    assert split.index.equals(many_claims.index)


def test_no_policy_crosses_splits(many_claims, cfg):
    split = split_claims(many_claims, cfg)
    assert (many_claims.assign(split=split).groupby("policy_id")["split"].nunique() == 1).all()


def test_ratios_and_target_rate(many_claims, cfg):
    split = split_claims(many_claims, cfg)
    shares = split.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.03
    assert abs(shares["test"] - 0.15) < 0.03
    overall = many_claims["is_high_cost"].mean()
    for name in ("train", "validation", "test"):
        assert abs(many_claims.loc[split == name, "is_high_cost"].mean() - overall) < 0.02


def test_deterministic_and_order_independent(many_claims, cfg):
    first = split_claims(many_claims, cfg)
    shuffled = many_claims.sample(frac=1.0, random_state=9)
    second = split_claims(shuffled, cfg)
    pd.testing.assert_series_equal(first, second.reindex(first.index))


def test_bad_ratios_rejected(many_claims, cfg):
    cfg["split"] = {"train": 0.8, "validation": 0.15, "test": 0.15}
    with pytest.raises(ValueError):
        split_claims(many_claims, cfg)
