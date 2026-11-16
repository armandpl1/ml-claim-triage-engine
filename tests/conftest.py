"""Shared fixtures: small, deterministic raw tables that look like freMTPL2."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.ingestion.fetch_data import build_claims


@pytest.fixture
def cfg():
    """The real config, with min_rows lowered for small fixtures."""
    config = copy.deepcopy(load_config())
    config["contracts"]["min_rows"] = 50
    return config


@pytest.fixture
def raw_policies() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame(
        {
            "policy_id": np.arange(1, n + 1, dtype="int64"),
            "claim_count": rng.integers(1, 3, n),
            "exposure": rng.uniform(0.05, 1.0, n).round(2),
            "area": rng.choice(list("ABCDEF"), n),
            "vehicle_power": rng.integers(4, 15, n),
            "vehicle_age": rng.integers(0, 20, n),
            "driver_age": rng.integers(18, 85, n),
            "bonus_malus": rng.integers(50, 150, n),
            "vehicle_brand": rng.choice(["B1", "B2", "B12"], n),
            "fuel_type": rng.choice(["Regular", "Diesel"], n),
            "density": rng.integers(1, 20000, n),
            "region": rng.choice(["R11", "R24", "R82"], n),
        }
    )


@pytest.fixture
def raw_amounts() -> pd.DataFrame:
    """Claim amounts: about 8% above the high-cost threshold, plus 2 orphans."""
    rng = np.random.default_rng(1)
    n = 300
    amounts = rng.lognormal(mean=7.0, sigma=0.8, size=n).round(2)
    amounts[:24] = rng.uniform(6000, 40000, 24).round(2)
    policy_ids = rng.integers(1, 401, n)
    policy_ids[-2:] = [999_998, 999_999]  # policies that don't exist
    return pd.DataFrame({"policy_id": policy_ids.astype("int64"), "claim_amount": amounts})


@pytest.fixture
def claims(raw_policies, raw_amounts, cfg) -> pd.DataFrame:
    built, _ = build_claims(raw_policies, raw_amounts, cfg)
    return built
