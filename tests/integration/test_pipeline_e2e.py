"""End-to-end dry run on a 100-claim fixture, in under 15 seconds.

ingest -> contracts -> split -> train -> ONNX export -> reload -> serve via the
API, checking along the way that:
    - artifacts serialize and load back (ONNX file, serving bundle)
    - the ONNX inputs match the feature schema the API sends
    - the API returns exactly what the training code computes for the same
      claim (no training/serving skew)
    - every prediction lands in the inference log
"""

from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

from src.features.split import split_claims
from src.features.transform import add_row_features, feature_columns
from src.ingestion.fetch_data import build_claims
from src.serving import app as app_module
from src.serving import model_store
from src.training.onnx_export import export_tabular, onnx_inputs
from src.training.tabular import make_pipeline
from src.training.tracking import train_config
from src.validation.contracts import validate_claims

pytestmark = pytest.mark.integration
THRESHOLD = 0.3


@pytest.fixture
def small_cfg(cfg):
    cfg["contracts"]["min_category_share"] = 0.0  # 100 claims can miss a rare category
    return cfg


def request_body(row) -> dict:
    fields = [
        "exposure",
        "vehicle_power",
        "vehicle_age",
        "driver_age",
        "bonus_malus",
        "density",
        "area",
        "vehicle_brand",
        "fuel_type",
        "region",
    ]
    body = {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in fields}
    body["claim_id"] = row["claim_id"]
    return body


def test_pipeline_end_to_end(raw_policies, raw_amounts, small_cfg, tmp_path, monkeypatch):
    started = time.perf_counter()
    cfg = small_cfg

    # Ingest and validate
    claims, _ = build_claims(raw_policies, raw_amounts, cfg)
    claims = claims.head(100).reset_index(drop=True)
    assert validate_claims(claims, cfg).passed

    # Split before fitting anything, then train on the training split only
    rows = add_row_features(claims)
    rows["split"] = split_claims(rows, cfg)
    train = rows[rows["split"] == "train"]
    assert len(train) > 50 and train["is_high_cost"].any()
    pipe = make_pipeline(
        cfg,
        {"n_estimators": 30, "min_child_samples": 3},
        seed=0,
        monotone=train_config()["tabular"]["monotone"],
    )
    pipe.fit(train, train["is_high_cost"])

    # Export, serialize, reload, check the schema and the predictions
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    onnx.save(export_tabular(pipe, cfg, 17), bundle / "tabular.onnx")
    session = ort.InferenceSession(str(bundle / "tabular.onnx"))
    assert [i.name for i in session.get_inputs()] == feature_columns(cfg)
    onnx_probs = session.run(None, onnx_inputs(rows, cfg))[1][:, 1]
    np.testing.assert_allclose(onnx_probs, pipe.predict_proba(rows)[:, 1], atol=1e-5)

    # Serve the exported model through the real API
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "tabular": {
                    "name": "claim-triage-tabular",
                    "version": "1",
                    "tags": {"decision_threshold": str(THRESHOLD)},
                }
            }
        )
    )
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(model_store, "BUNDLE_DIR", bundle)
    log_path = tmp_path / "inference_log.db"
    monkeypatch.setattr(app_module, "INFERENCE_LOG", log_path)

    with TestClient(app_module.app) as client:
        health = client.get("/health").json()
        assert health["models"] == {"tabular": "1"} and health["status"] == "degraded"

        for i in range(5):
            row = rows.iloc[i]
            response = client.post("/predict", json=request_body(row))
            assert response.status_code == 200, response.text
            body = response.json()
            # Same probability as the training-side computation: no skew
            assert body["high_cost_probability"] == pytest.approx(float(onnx_probs[i]), abs=1e-5)
            expected_route = "senior_adjuster" if onnx_probs[i] >= THRESHOLD else "standard"
            assert body["route"] == expected_route

        metrics = client.get("/metrics").text
        assert 'claim_api_requests_total{endpoint="/predict",status="200"}' in metrics

    with sqlite3.connect(log_path) as con:
        assert con.execute("SELECT count(*) FROM predictions").fetchone()[0] == 5

    assert time.perf_counter() - started < 15, "integration dry run must finish within 15 s"
