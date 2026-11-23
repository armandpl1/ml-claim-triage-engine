"""Tracking setup: local stores get a local artifact folder, servers don't."""

from __future__ import annotations

import pytest

from src.training.tracking import is_local_store


@pytest.mark.parametrize("uri", ["sqlite:///mlflow.db", "file:///tmp/mlruns", "./mlruns"])
def test_local_stores_use_the_local_artifact_folder(uri):
    assert is_local_store(uri)


@pytest.mark.parametrize(
    "uri", ["http://localhost:5001", "https://mlflow.example.com", "databricks"]
)
def test_tracking_servers_choose_their_own_artifact_store(uri):
    assert not is_local_store(uri)
