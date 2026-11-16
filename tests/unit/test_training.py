"""Training helpers: threshold choice, metrics, search space, ONNX inputs, timing."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from src.training.benchmark import time_calls
from src.training.onnx_export import onnx_inputs
from src.training.tabular import best_f1_threshold, evaluate, suggest


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.8
    assert evaluate(y, scores, threshold)["f1"] == 1.0


def test_evaluate_reports_every_metric():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.1
    metrics = evaluate(y, rng.random(500), 0.5)
    assert set(metrics) == {
        "pr_auc",
        "roc_auc",
        "brier",
        "f1",
        "precision",
        "recall",
        "threshold",
        "positive_rate",
    }
    assert metrics["positive_rate"] == pytest.approx(y.mean())


def test_suggest_respects_types_and_bounds():
    space = {
        "num_leaves": {"low": 8, "high": 96},
        "learning_rate": {"low": 0.01, "high": 0.2, "log": True},
    }
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    for _ in range(20):
        params = suggest(study.ask(), space)
        assert isinstance(params["num_leaves"], int) and 8 <= params["num_leaves"] <= 96
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert params["subsample_freq"] == 1


def test_onnx_inputs_shapes_types_and_missing_categories(cfg):
    frame = pd.DataFrame(
        {
            "exposure": [0.5, 1.0],
            "vehicle_power": [6, 7],
            "vehicle_age": [1, 2],
            "driver_age": [30, np.nan],
            "bonus_malus": [50, 60],
            "log_density": [3.0, 4.0],
            "area": ["A", None],
            "vehicle_brand": ["B1", "B2"],
            "fuel_type": ["Diesel", "Regular"],
            "region": ["R11", "R24"],
        }
    )
    feeds = onnx_inputs(frame, cfg)
    assert feeds["driver_age"].dtype == np.float32 and feeds["driver_age"].shape == (2, 1)
    assert np.isnan(feeds["driver_age"][1, 0])
    assert feeds["area"].dtype == object and feeds["area"][1, 0] == ""  # missing -> ""


def test_time_calls_returns_ordered_percentiles():
    result = time_calls(lambda i: sum(range(100)), runs=50, warmup=2)
    assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]
    assert result["mean_ms"] > 0
