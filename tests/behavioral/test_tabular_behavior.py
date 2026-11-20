"""Behavioral tests for the tabular triage model.

Directional expectations (domain knowledge, enforced by monotonic constraints
in configs/train_config.yaml):
    more vehicle power  -> predicted risk never goes down
    older vehicle       -> predicted risk never goes up
Invariance:
    identifiers and the claim amount itself must not change the prediction
    (if the amount did, the target would be leaking into the model)

Each test runs on a model trained here on fixture data, and again on the
production model (artifacts/tabular/model.onnx) when it was trained with the
same constraints.
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.common.config import ROOT
from src.features.transform import add_row_features
from src.training.onnx_export import export_tabular, onnx_inputs
from src.training.tabular import make_pipeline
from src.training.tracking import train_config

pytestmark = pytest.mark.behavioral
PRODUCTION = ROOT / "artifacts" / "tabular"


@pytest.fixture(scope="module")
def monotone():
    return train_config()["tabular"]["monotone"]


@pytest.fixture
def rows(claims):
    return add_row_features(claims)


@pytest.fixture
def fixture_model(rows, cfg, monotone):
    params = {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 5}
    return make_pipeline(cfg, params, seed=0, monotone=monotone).fit(rows, rows["is_high_cost"])


def production_scorer(cfg, monotone):
    """predict(frame) for the production ONNX model, or None if not comparable.

    metrics.json is committed but model.onnx is not (it lives in the MLflow
    registry), so a fresh clone or CI tests the fixture-trained model only.
    """
    metrics_file = PRODUCTION / "metrics.json"
    if (
        not metrics_file.exists()
        or not (PRODUCTION / "model.onnx").exists()
        or json.loads(metrics_file.read_text()).get("monotone") != monotone
    ):
        return None
    session = ort.InferenceSession(str(PRODUCTION / "model.onnx"))
    return lambda frame: session.run(None, onnx_inputs(frame, cfg))[1][:, 1]


def sweep(predict, base: pd.DataFrame, column: str, values) -> np.ndarray:
    """Predictions as `column` takes each value, one row per base claim."""
    out = []
    for value in values:
        varied = base.copy()
        varied[column] = value
        out.append(predict(varied))
    return np.column_stack(out)  # shape (claims, values)


@pytest.mark.parametrize(
    ("column", "values", "direction"),
    [("vehicle_power", range(4, 16), 1), ("vehicle_age", range(0, 31, 2), -1)],
)
def test_directional_expectations(rows, cfg, fixture_model, monotone, column, values, direction):
    scorers = {"fixture": lambda f: fixture_model.predict_proba(f)[:, 1]}
    production = production_scorer(cfg, monotone)
    if production is not None:
        scorers["production"] = production
    for name, predict in scorers.items():
        steps = np.diff(sweep(predict, rows.head(40), column, values), axis=1) * direction
        assert (steps >= -1e-6).all(), f"{name} model breaks the {column} direction"


@pytest.mark.parametrize(
    ("column", "new_value"),
    [
        ("claim_id", "CLM-9999999-01"),
        ("policy_id", 9_999_999),
        ("claim_seq", 7),
        ("adjuster_note", "completely different note"),
        ("claim_amount", 1_000_000.0),
    ],
)
def test_non_features_do_not_change_predictions(rows, fixture_model, column, new_value):
    before = fixture_model.predict_proba(rows)[:, 1]
    changed = rows.copy()
    changed[column] = new_value
    np.testing.assert_array_equal(fixture_model.predict_proba(changed)[:, 1], before)


def test_row_order_does_not_change_predictions(rows, fixture_model):
    before = pd.Series(fixture_model.predict_proba(rows)[:, 1], index=rows.index)
    shuffled = rows.sample(frac=1.0, random_state=3)
    after = pd.Series(fixture_model.predict_proba(shuffled)[:, 1], index=shuffled.index)
    pd.testing.assert_series_equal(after.sort_index(), before.sort_index())


def test_onnx_export_keeps_the_directions(rows, cfg, fixture_model):
    session = ort.InferenceSession(export_tabular(fixture_model, cfg, 17).SerializeToString())
    grid = sweep(
        lambda f: session.run(None, onnx_inputs(f, cfg))[1][:, 1],
        rows.head(20),
        "vehicle_power",
        range(4, 16),
    )
    assert (np.diff(grid, axis=1) >= -1e-6).all()
