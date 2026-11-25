"""Claim triage API.

    GET  /health    model versions, where they were loaded from, uptime
    POST /predict   one claim: high-cost probability and route, plus the
                    note's category if an adjuster note is sent
    POST /predict/policy
                    the same, for a claim sent with only its policy_id: the
                    features come from the online feature store
                    (src/feature_store/online.py), so callers can't send
                    stale or hand-typed values
    GET  /metrics   Prometheus metrics: requests, errors, latency, routes

Models load once at startup (src/serving/model_store.py). Every prediction is
logged after the response is sent (a background task, so logging never adds
latency) to a SQLite inference log, which the drift detector compares with
the training reference (monitoring/drift_detector.py).

Run locally:  make serve      (http://localhost:8000/docs)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.common.config import ROOT
from src.feature_store.online import OnlineStore
from src.features.row import log_density
from src.serving.model_store import LoadedModels, load_models
from src.serving.schemas import (
    ClaimPrediction,
    ClaimRequest,
    Health,
    NotePrediction,
    PolicyClaimRequest,
)

INFERENCE_LOG = Path(os.environ.get("INFERENCE_LOG_PATH", ROOT / "data/inference/inference_log.db"))
ONLINE_STORE = Path(os.environ.get("ONLINE_STORE_PATH", ROOT / "data/feature_store/online.db"))
NUMERIC = ["exposure", "vehicle_power", "vehicle_age", "driver_age", "bonus_malus", "log_density"]
CATEGORICAL = ["area", "vehicle_brand", "fuel_type", "region"]

REQUESTS = Counter("claim_api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "claim_api_request_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
ROUTES = Counter("claim_api_routes_total", "Triage decisions", ["route"])
NOTE_CATEGORIES = Counter(
    "claim_api_note_categories_total", "Predicted note categories", ["category"]
)
MODEL_VERSION = Gauge("claim_api_model_version", "Serving model version", ["model"])

LOG_SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY, ts REAL, claim_id TEXT, features TEXT, adjuster_note TEXT,
    high_cost_probability REAL, route TEXT, note_category TEXT,
    tabular_version TEXT, notes_version TEXT, latency_ms REAL)"""


def tabular_feeds(claim: ClaimRequest) -> dict[str, np.ndarray]:
    """ONNX inputs for one claim, computed exactly like training."""
    values = claim.model_dump()
    values["log_density"] = log_density(claim.density)
    feeds = {c: np.array([[values[c]]], dtype=np.float32) for c in NUMERIC}
    feeds.update({c: np.array([[values[c]]], dtype=object) for c in CATEGORICAL})
    return feeds


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def write_log(row: dict) -> None:
    INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(INFERENCE_LOG) as con:
        con.execute(LOG_SCHEMA)
        con.execute(
            "INSERT INTO predictions VALUES (:request_id, :ts, :claim_id, :features, "
            ":adjuster_note, :high_cost_probability, :route, :note_category, "
            ":tabular_version, :notes_version, :latency_ms)",
            row,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.models = load_models()
    # Optional: without a store, /predict/policy answers 503 and /predict works.
    app.state.online = OnlineStore(ONLINE_STORE) if ONLINE_STORE.exists() else None
    app.state.started = time.time()
    for name, version in app.state.models.versions.items():
        MODEL_VERSION.labels(model=name).set(float(version))
    yield
    if app.state.online:
        app.state.online.close()


app = FastAPI(title="Claim triage API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def record_metrics(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    if endpoint != "/metrics":
        REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)
    return response


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    models: LoadedModels = request.app.state.models
    return Health(
        status="ok" if models.notes else "degraded",
        models=models.versions,
        source=models.source,
        uptime_seconds=round(time.time() - request.app.state.started, 1),
        online_store=request.app.state.online.info() if request.app.state.online else None,
    )


@app.post("/predict", response_model=ClaimPrediction)
def predict(claim: ClaimRequest, request: Request, background: BackgroundTasks) -> ClaimPrediction:
    return score_claim(claim, request, background, time.perf_counter())


@app.post("/predict/policy", response_model=ClaimPrediction)
def predict_policy(
    body: PolicyClaimRequest, request: Request, background: BackgroundTasks
) -> ClaimPrediction:
    started = time.perf_counter()
    store: OnlineStore | None = request.app.state.online
    if store is None:
        raise HTTPException(503, "online feature store not loaded: run `make materialize`")
    features = store.get(body.policy_id)
    if features is None:
        raise HTTPException(404, f"policy {body.policy_id} not in the online feature store")
    valid_from = features.pop("valid_from")
    features.pop("policy_id")
    # Same validation as a hand-sent claim: store values must meet the contracts too.
    claim = ClaimRequest(claim_id=body.claim_id, adjuster_note=body.adjuster_note, **features)
    return score_claim(claim, request, background, started, feature_valid_from=valid_from)


def score_claim(
    claim: ClaimRequest,
    request: Request,
    background: BackgroundTasks,
    started: float,
    feature_valid_from: str | None = None,
) -> ClaimPrediction:
    """Route one claim, and log it after the response is sent."""
    models: LoadedModels = request.app.state.models
    probability = float(models.tabular.run(None, tabular_feeds(claim))[1][0, 1])
    route: Literal["senior_adjuster", "standard"] = (
        "senior_adjuster" if probability >= models.threshold else "standard"
    )

    note = None
    if claim.adjuster_note and models.notes and models.tokenizer:
        logits = models.notes.run(["logits"], models.tokenizer([claim.adjuster_note]))[0][0]
        probs = softmax(logits)
        note = NotePrediction(
            category=models.labels[int(probs.argmax())],
            probabilities={
                c: round(float(p), 4) for c, p in zip(models.labels, probs, strict=True)
            },
        )
        NOTE_CATEGORIES.labels(category=note.category).inc()
    ROUTES.labels(route=route).inc()

    result = ClaimPrediction(
        request_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        high_cost_probability=round(probability, 6),
        route=route,
        decision_threshold=models.threshold,
        note=note,
        model_versions=models.versions,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        feature_valid_from=feature_valid_from,
    )
    features = claim.model_dump(exclude={"claim_id", "adjuster_note"})
    background.add_task(
        write_log,
        {
            "request_id": result.request_id,
            "ts": time.time(),
            "claim_id": claim.claim_id,
            "features": json.dumps(features),
            "adjuster_note": claim.adjuster_note,
            "high_cost_probability": probability,
            "route": route,
            "note_category": note.category if note else None,
            "tabular_version": models.versions.get("tabular"),
            "notes_version": models.versions.get("notes"),
            "latency_ms": result.latency_ms,
        },
    )
    return result


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
