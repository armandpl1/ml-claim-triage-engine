"""MLflow setup shared by the training scripts.

Tracking is local: a SQLite backend store (mlflow.db) and an artifact folder
(mlartifacts/) in the repository root. The scripts write to them directly;
`make mlflow-ui` starts an MLflow server on the same store to browse runs.
MLFLOW_TRACKING_URI, if set, takes precedence (e.g. a remote server).
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from src.common.config import ROOT, load_config


def train_config() -> dict[str, Any]:
    return load_config("train_config.yaml")


def tracking_uri(cfg: dict[str, Any]) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / cfg['mlflow']['tracking_db']}")


def start_experiment(name: str) -> str:
    """Point MLflow at the local store and select (or create) an experiment."""
    cfg = train_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        artifacts = (ROOT / cfg["mlflow"]["artifact_dir"]).resolve()
        artifacts.mkdir(exist_ok=True)
        experiment_id = mlflow.create_experiment(name, artifact_location=artifacts.as_uri())
    else:
        experiment_id = str(experiment.experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
