"""Reason codes: contributions add up to the model's own score, each claim
gets its largest risk-raising features in order, and values are shown the
way an adjuster knows them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import reasons
from src.features.transform import add_row_features
from src.training.tabular import make_pipeline


@pytest.fixture
def fitted(claims, cfg):
    rows = add_row_features(claims)
    params = {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 5}
    pipe = make_pipeline(cfg, params, seed=0).fit(rows, rows["is_high_cost"])
    return rows, pipe


def test_contributions_sum_to_the_models_probability(fitted):
    rows, pipe = fitted
    contrib = reasons.contributions(pipe, rows)
    logit = contrib.sum(axis=1).to_numpy()
    probability = 1 / (1 + np.exp(-logit))
    assert np.allclose(probability, pipe.predict_proba(rows)[:, 1], atol=1e-9)
    assert reasons.BASE in contrib.columns
    assert "log_density" in contrib.columns  # the model's own feature names


def test_reason_codes_rank_the_largest_risk_raising_features():
    frame = pd.DataFrame({"claim_id": ["a", "b"], "driver_age": [19, 45], "area": ["E", "C"]})
    contrib = pd.DataFrame(
        {"driver_age": [0.9, -0.4], "area": [0.3, 0.002], reasons.BASE: [-3.0, -3.0]}
    )
    codes = reasons.reason_codes(frame, contrib, top_k=3, min_contribution=0.01)
    assert codes.to_dict("records") == [
        {"claim_id": "a", "rank": 1, "feature": "driver_age", "value": "19", "contribution": 0.9},
        {"claim_id": "a", "rank": 2, "feature": "area", "value": "E", "contribution": 0.3},
    ]  # b: one feature lowered its risk, the other is below the noise floor


def test_top_k_caps_the_reasons_per_claim():
    frame = pd.DataFrame({"claim_id": ["a"], "x": [1], "y": [2], "z": [3]})
    contrib = pd.DataFrame({"x": [0.1], "y": [0.3], "z": [0.2], reasons.BASE: [0.0]})
    codes = reasons.reason_codes(frame, contrib, top_k=2, min_contribution=0.0)
    assert list(codes["feature"]) == ["y", "z"]


def test_log_density_is_shown_as_density():
    frame = pd.DataFrame({"claim_id": ["a"], "density": [778.0], "log_density": [6.65]})
    contrib = pd.DataFrame({"log_density": [0.5], reasons.BASE: [0.0]})
    codes = reasons.reason_codes(frame, contrib, top_k=3, min_contribution=0.0)
    assert (codes.at[0, "feature"], codes.at[0, "value"]) == ("density", "778")


def test_importance_is_mean_absolute_contribution():
    contrib = pd.DataFrame({"x": [1.0, -3.0], "y": [0.5, 0.5], reasons.BASE: [9.0, 9.0]})
    assert reasons.importance(contrib).to_dict() == {"x": 2.0, "y": 0.5}
