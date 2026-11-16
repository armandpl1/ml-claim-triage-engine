"""Send claims to the API, so the inference log has traffic to check for drift.

Claims come from the holdout test split, so with no shift the traffic looks
like training and the drift report should stay quiet. With --shift, the
population changes the way a real portfolio can (older vehicles, younger
drivers, denser cities, more diesel), and the drift report should alert.

By default the API runs in-process (FastAPI TestClient, models from the MLflow
registry), so no server is needed. With --url, requests go to a running API
(make serve or make up).

Usage:
    python -m monitoring.simulate_traffic --n 500
    python -m monitoring.simulate_traffic --n 500 --shift
    python -m monitoring.simulate_traffic --n 500 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config, resolve

FIELDS = [
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


def shift_population(claims: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A plausible portfolio change, kept inside the API's input limits."""
    rng = np.random.default_rng(seed)
    out = claims.copy()
    out["vehicle_age"] = (out["vehicle_age"] + 8).clip(upper=100)
    out["driver_age"] = (out["driver_age"] - 12).clip(lower=18)
    out["density"] = out["density"] * 6
    out["fuel_type"] = np.where(rng.random(len(out)) < 0.8, "Diesel", out["fuel_type"])
    return out


def payloads(n: int, shift: bool, seed: int) -> list[dict[str, Any]]:
    cfg = load_config()
    test = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "test.parquet")
    claims = test.sample(n=min(n, len(test)), random_state=seed).reset_index(drop=True)
    if shift:
        claims = shift_population(claims, seed)
    records = []
    for i, row in claims.iterrows():
        record: dict[str, Any] = {
            f: row[f].item() if hasattr(row[f], "item") else row[f] for f in FIELDS
        }
        record["claim_id"] = f"sim-{i}"
        record["adjuster_note"] = row[cfg["features"]["text"]]
        records.append(record)
    return records


def send(records: list[dict[str, Any]], url: str | None) -> Counter[int]:
    statuses: Counter[int] = Counter()
    if url:
        import httpx

        with httpx.Client(base_url=url, timeout=10) as client:
            for record in records:
                statuses[client.post("/predict", json=record).status_code] += 1
        return statuses

    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    from fastapi.testclient import TestClient

    from src.serving.app import app

    with TestClient(app) as client:
        for record in records:
            statuses[client.post("/predict", json=record).status_code] += 1
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--shift", action="store_true", help="Send a drifted population")
    parser.add_argument("--url", default=None, help="A running API; default: in-process")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    statuses = send(payloads(args.n, args.shift, args.seed), args.url)
    print(
        json.dumps(
            {"sent": sum(statuses.values()), "shift": args.shift, "statuses": dict(statuses)}
        )
    )


if __name__ == "__main__":
    main()
