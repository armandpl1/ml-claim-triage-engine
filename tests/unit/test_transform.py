"""Feature pipeline: leakage-free fitting, and graceful handling of dirty input."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transform import (
    UNKNOWN_CATEGORY,
    add_row_features,
    build_preprocessor,
    feature_columns,
    raw_input_columns,
)


@pytest.fixture
def train_rows(claims) -> pd.DataFrame:
    return add_row_features(claims)


def transform_one(preprocessor, cfg, **overrides) -> dict:
    """Transform a single row built from valid defaults plus overrides."""
    row = {
        "exposure": 0.5,
        "vehicle_power": 6,
        "vehicle_age": 3,
        "driver_age": 40,
        "bonus_malus": 50,
        "density": 1000,
        "area": "C",
        "vehicle_brand": "B12",
        "fuel_type": "Regular",
        "region": "R11",
    }
    row.update(overrides)
    frame = add_row_features(pd.DataFrame([row]))
    out = preprocessor.transform(frame)
    return dict(zip(preprocessor.get_feature_names_out(), out[0], strict=True))


def test_leaky_columns_are_never_features(cfg):
    used = set(feature_columns(cfg))
    assert not used & set(cfg["features"]["excluded"])
    assert "claim_seq" not in used and "claim_amount" not in used


def test_statistics_come_from_training_data_only(train_rows, cfg):
    train = train_rows.iloc[:150]
    other = train_rows.iloc[150:].copy()
    other["driver_age"] = 99  # very different from training
    pre = build_preprocessor(cfg).fit(train)
    median = pre.named_transformers_["numeric"].named_steps["impute"].statistics_
    position = cfg["features"]["numeric"].index("driver_age")
    assert median[position] == train["driver_age"].median()
    assert median[position] != pd.concat([train, other])["driver_age"].median()


def test_missing_number_becomes_training_median(train_rows, cfg):
    pre = build_preprocessor(cfg).fit(train_rows)
    out = transform_one(pre, cfg, driver_age=np.nan)
    assert out["driver_age"] == train_rows["driver_age"].median()


@pytest.mark.parametrize("value", ["B99", None])
def test_unseen_or_missing_category_becomes_unknown(train_rows, cfg, value):
    pre = build_preprocessor(cfg).fit(train_rows)
    assert transform_one(pre, cfg, vehicle_brand=value)["vehicle_brand"] == UNKNOWN_CATEGORY


@pytest.mark.parametrize("density", [-50, 0])
def test_negative_or_zero_density_is_clipped(train_rows, cfg, density):
    pre = build_preprocessor(cfg).fit(train_rows)
    assert transform_one(pre, cfg, density=density)["log_density"] == 0.0  # log(1)


def test_known_values_pass_through(train_rows, cfg):
    pre = build_preprocessor(cfg).fit(train_rows)
    out = transform_one(pre, cfg, driver_age=33, density=np.e**3)
    assert out["driver_age"] == 33
    assert out["log_density"] == pytest.approx(3.0)
    assert out["vehicle_brand"] != UNKNOWN_CATEGORY


def test_serving_input_columns_are_raw(cfg):
    columns = raw_input_columns(cfg)
    assert "density" in columns and "log_density" not in columns
