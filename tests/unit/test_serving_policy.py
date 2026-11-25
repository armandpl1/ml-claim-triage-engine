"""POST /predict/policy: features from the online store, scored exactly like
the same claim sent in full to /predict."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.feature_store import materialize, offline
from src.feature_store.registry import load_registry
from src.serving import app as app_module
from src.serving.model_store import LoadedModels


class FakeSession:
    """Stands in for onnxruntime: probability = driver_age / 100."""

    def run(self, _outputs, feeds):
        p = feeds["driver_age"][:, 0].astype(float) / 100
        return [(p > 0.5).astype(int), np.column_stack([1 - p, p])]


def fake_models() -> LoadedModels:
    return LoadedModels(
        tabular=FakeSession(),
        threshold=0.5,
        notes=None,
        labels=[],
        tokenizer=None,
        versions={"tabular": "3"},
        source="test",
    )


@pytest.fixture
def online_path(raw_policies, tmp_path):
    registry = dataclasses.replace(
        load_registry(), offline=tmp_path / "offline.duckdb", online=tmp_path / "online.db"
    )
    offline.write(offline.build(raw_policies, registry), registry.offline)
    as_of = datetime(2026, 1, 1)
    materialize.write_online(materialize.latest_versions(registry, as_of), registry.online, as_of)
    return registry.online


@pytest.fixture
def client(online_path, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "load_models", fake_models)
    monkeypatch.setattr(app_module, "ONLINE_STORE", online_path)
    monkeypatch.setattr(app_module, "INFERENCE_LOG", tmp_path / "log.db")
    with TestClient(app_module.app) as c:
        yield c


def test_policy_lookup_scores_like_a_full_request(client, raw_policies):
    policy = int(raw_policies["policy_id"].iloc[0])
    by_policy = client.post("/predict/policy", json={"policy_id": policy, "claim_id": "C1"})
    assert by_policy.status_code == 200, by_policy.text
    body = by_policy.json()
    assert body["feature_valid_from"].startswith("2025-")  # the renewed version

    source = raw_policies.set_index("policy_id").loc[policy]
    renewed_age = int(source["driver_age"]) + 1
    assert body["high_cost_probability"] == pytest.approx(renewed_age / 100)

    full = {
        **{k: (v.item() if hasattr(v, "item") else v) for k, v in source.items()},
        "driver_age": renewed_age,
        "vehicle_age": min(int(source["vehicle_age"]) + 1, 100),
        "bonus_malus": max(int(source["bonus_malus"] * 0.95), 50),
    }
    full.pop("claim_count")
    direct = client.post("/predict", json=full)
    assert direct.status_code == 200, direct.text
    assert direct.json()["high_cost_probability"] == body["high_cost_probability"]


def test_unknown_policy_is_404(client):
    response = client.post("/predict/policy", json={"policy_id": 123_456_789})
    assert response.status_code == 404


def test_extra_fields_are_rejected(client):
    response = client.post("/predict/policy", json={"policy_id": 1, "driver_age": 30})
    assert response.status_code == 422


def test_health_reports_the_store(client, raw_policies):
    store = client.get("/health").json()["online_store"]
    assert store["row_count"] == len(raw_policies)
    assert store["as_of"].startswith("2026-01-01")
    metrics = client.get("/metrics").text
    expected = datetime(2026, 1, 1).timestamp()
    line = next(m for m in metrics.splitlines() if m.startswith("claim_api_online_store_as_of"))
    assert float(line.split()[-1]) == expected


def test_without_a_store_policy_requests_are_503(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "load_models", fake_models)
    monkeypatch.setattr(app_module, "ONLINE_STORE", tmp_path / "missing.db")
    monkeypatch.setattr(app_module, "INFERENCE_LOG", tmp_path / "log.db")
    with TestClient(app_module.app) as c:
        assert c.post("/predict/policy", json={"policy_id": 1}).status_code == 503
        assert c.get("/health").json()["online_store"] is None
