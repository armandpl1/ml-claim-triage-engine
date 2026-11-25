"""Rebuilding the champion for explanations: recipes parse from MLflow's
string parameters, a faithful refit passes the parity check against the
served ONNX model, and a different model is refused."""

from __future__ import annotations

import json

import onnxruntime as ort
import pytest

from src.explain import champion
from src.features.transform import add_row_features
from src.training.onnx_export import export_tabular

PARAMS = {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 5}


def logged(params: dict, monotone: dict | None = None) -> dict[str, str]:
    """Parameters the way an MLflow run stores them: best_* strings."""
    out = {f"best_{k}": str(v) for k, v in params.items()}
    if monotone is not None:
        out["monotone"] = json.dumps(monotone)
    out["n_trials"] = "25"  # not part of the recipe
    return out


def test_recipe_parses_numbers_and_monotone_rules():
    recipe = champion.recipe_from_params(
        logged({"n_estimators": 134, "learning_rate": 0.017}, {"vehicle_age": -1}), 42, "r1"
    )
    assert recipe.params == {"n_estimators": 134, "learning_rate": 0.017}
    assert isinstance(recipe.params["n_estimators"], int)
    assert recipe.monotone == {"vehicle_age": -1}


def test_runs_before_monotone_rules_mean_no_constraints():
    assert champion.recipe_from_params(logged(PARAMS), 42, "r1").monotone == {}


def test_a_run_without_hyperparameters_is_refused():
    with pytest.raises(ValueError, match="no best_"):
        champion.recipe_from_params({"n_trials": "25"}, 42, "r1")


def test_bundle_recipe_reads_the_bundle_metadata(tmp_path):
    meta = {"tabular": {"training": {"run_id": "abc", "params": logged(PARAMS)}}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    recipe = champion.bundle_recipe(7, tmp_path)
    assert (recipe.run_id, recipe.seed, recipe.params) == ("abc", 7, PARAMS)


def test_a_bundle_without_a_recipe_asks_for_a_re_export(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"tabular": {"version": "1"}}))
    with pytest.raises(ValueError, match="make bundle"):
        champion.bundle_recipe(7, tmp_path)


@pytest.fixture
def served(claims, cfg):
    """A 'served' champion: fitted from a recipe, exported to ONNX."""
    rows = add_row_features(claims)
    recipe = champion.recipe_from_params(logged(PARAMS), 0, "r1")
    onnx_model = export_tabular(champion.rebuild(recipe, rows, cfg), cfg, opset=17)
    session = ort.InferenceSession(onnx_model.SerializeToString())
    return rows, recipe, session


def test_a_faithful_rebuild_passes_parity(served, cfg):
    rows, recipe, session = served
    diff = champion.check_parity(champion.rebuild(recipe, rows, cfg), session, rows, cfg, 1e-5)
    assert diff < 1e-5


def test_a_different_model_fails_parity(served, cfg):
    rows, recipe, session = served
    other = champion.Recipe({**recipe.params, "n_estimators": 5}, {}, 0, "r2")
    with pytest.raises(champion.ParityError, match="differs from the served champion"):
        champion.check_parity(champion.rebuild(other, rows, cfg), session, rows, cfg, 1e-5)
