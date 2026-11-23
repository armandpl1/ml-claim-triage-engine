"""MLflow setup shared by the training scripts.

By default tracking is local: a SQLite backend store (mlflow.db) and an
artifact folder (mlartifacts/) in the repository root. The scripts write to
them directly; `make mlflow-ui` starts an MLflow server on the same store.

MLFLOW_TRACKING_URI, if set, takes precedence. With a tracking server
(`make mlflow-up`: runs in Postgres, artifacts in S3) the server decides
where artifacts go, so new experiments get no local artifact folder.
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


def is_local_store(uri: str) -> bool:
    """True for a store the scripts write to directly (SQLite or a folder),
    False for a tracking server, which manages artifact storage itself."""
    return not uri.startswith(("http://", "https://", "databricks"))


def start_experiment(name: str) -> str:
    """Point MLflow at the tracking store and select (or create) an experiment."""
    cfg = train_config()
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        location = None  # the server's artifact store (S3 behind `make mlflow-up`)
        if is_local_store(uri):
            artifacts = (ROOT / cfg["mlflow"]["artifact_dir"]).resolve()
            artifacts.mkdir(exist_ok=True)
            location = artifacts.as_uri()
        experiment_id = mlflow.create_experiment(name, artifact_location=location)
    else:
        experiment_id = str(experiment.experiment_id)
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id
