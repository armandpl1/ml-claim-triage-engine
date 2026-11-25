"""Rebuild the @champion triage model as LightGBM, so it can be explained.

The registry serves the champion as ONNX: fast, but it can't say which
features drove a score. LightGBM can (TreeSHAP, built in as pred_contrib), so
this module refits the champion's logged recipe (hyperparameters, monotone
rules, seed) on the training split, then proves the refit is the same model:
its probabilities must match the served ONNX model within
explain.parity_tolerance, or ParityError is raised and nothing is explained.
Training is deterministic (fixed seed, DVC-versioned data), so a refit of
champion v1 matched to 1.4e-7.

Where the recipe comes from:
    registry   the champion's MLflow run, when MLFLOW_TRACKING_URI is set
    bundle     serving_bundle/metadata.json ("training"), written by
               `make bundle`, as the API and the batch job do otherwise

Runs from before Step 4's monotone rules logged no "monotone" parameter and
were trained without them, so a missing parameter means no constraints.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event
from src.serving import model_store
from src.training.onnx_export import onnx_inputs
from src.training.tabular import make_pipeline
from src.training.tracking import train_config

logger = get_logger("explain.champion")


class ParityError(RuntimeError):
    """The rebuilt model doesn't predict like the served champion."""


@dataclass(frozen=True)
class Recipe:
    params: dict[str, Any]
    monotone: dict[str, int]
    seed: int
    run_id: str


@dataclass
class Explainer:
    pipeline: Pipeline  # preprocessor + fitted LGBMClassifier
    recipe: Recipe
    tolerance: float


def _number(text: str) -> int | float:
    try:
        return int(text)
    except ValueError:
        return float(text)


def recipe_from_params(params: dict[str, str], seed: int, run_id: str) -> Recipe:
    """A Recipe from an MLflow run's logged parameters (all strings)."""
    best = {k.removeprefix("best_"): _number(v) for k, v in params.items() if k.startswith("best_")}
    if not best:
        raise ValueError(f"run {run_id} logged no best_* hyperparameters")
    monotone = {k: int(v) for k, v in json.loads(params.get("monotone", "{}")).items()}
    return Recipe(best, monotone, seed, run_id)


def registry_recipe(seed: int) -> Recipe:
    from mlflow import MlflowClient

    client = MlflowClient()
    version = client.get_model_version_by_alias(model_store.TABULAR_MODEL, "champion")
    if not version.run_id:
        raise ValueError(f"champion v{version.version} has no training run to rebuild from")
    return recipe_from_params(client.get_run(version.run_id).data.params, seed, version.run_id)


def bundle_recipe(seed: int, bundle_dir: Any = None) -> Recipe:
    bundle = bundle_dir or model_store.BUNDLE_DIR
    meta = json.loads((bundle / "metadata.json").read_text())["tabular"]
    if "training" not in meta:
        raise ValueError(f"{bundle} has no training recipe: re-export it with `make bundle`")
    return recipe_from_params(meta["training"]["params"], seed, meta["training"]["run_id"])


def rebuild(recipe: Recipe, train: pd.DataFrame, data_cfg: dict[str, Any]) -> Pipeline:
    target = data_cfg["features"]["target_classification"]
    pipe = make_pipeline(data_cfg, recipe.params, recipe.seed, recipe.monotone)
    return pipe.fit(train, train[target])


def check_parity(
    pipe: Pipeline, session: Any, frame: pd.DataFrame, data_cfg: dict[str, Any], tolerance: float
) -> float:
    """Largest probability difference between the refit and the served model.

    Raises ParityError above the tolerance: explaining a different model than
    the one that made the decision would be worse than no explanation.
    """
    if frame.empty:
        return 0.0
    served = session.run(None, onnx_inputs(frame, data_cfg))[1][:, 1]
    refit = pipe.predict_proba(frame)[:, 1]
    diff = float(np.max(np.abs(served - refit)))
    if diff > tolerance:
        raise ParityError(
            f"rebuilt model differs from the served champion by {diff:.2e} "
            f"(tolerance {tolerance:.0e}); is the training data at the champion's version?"
        )
    return diff


def load_explainer(session: Any) -> Explainer:
    """Rebuild the champion and check it against the served model on the
    validation split, so a mismatch fails before any claim is explained."""
    cfg, data_cfg = train_config(), load_config()
    seed = int(cfg["seed"])
    recipe = registry_recipe(seed) if os.environ.get("MLFLOW_TRACKING_URI") else bundle_recipe(seed)
    processed = resolve(data_cfg["paths"]["processed_dir"])
    pipe = rebuild(recipe, pd.read_parquet(processed / "train.parquet"), data_cfg)
    tolerance = float(cfg["explain"]["parity_tolerance"])
    validation = pd.read_parquet(processed / "validation.parquet")
    diff = check_parity(pipe, session, validation, data_cfg, tolerance)
    log_event(
        logger,
        logging.INFO,
        "explainer_ready",
        run_id=recipe.run_id,
        parity_max_abs_diff=diff,
        checked_rows=len(validation),
    )
    return Explainer(pipe, recipe, tolerance)
