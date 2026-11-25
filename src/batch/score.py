"""Nightly batch scoring: score every claim reported on one day.

Loads the @champion triage model (from the MLflow registry when
MLFLOW_TRACKING_URI is set, else from serving_bundle/, as the API does),
computes the features with the same shared code as serving, and writes one
prediction per claim to the batch store, with its reason codes: the features
that raised the claim's risk most (src.explain, TreeSHAP on the champion
rebuilt as LightGBM). Re-running a day replaces that day's predictions and
reasons, so the job is idempotent.

Usage:
    python -m src.batch.score --date 2025-01-06
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.batch.schedule import build_schedule
from src.batch.store import connect
from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event
from src.explain import reasons
from src.explain.champion import Explainer, check_parity, load_explainer
from src.features.row import log_density
from src.features.transform import add_row_features
from src.serving import model_store
from src.training.tracking import train_config

logger = get_logger("batch.score")


@dataclass
class TabularModel:
    session: Any  # onnxruntime.InferenceSession, or a stand-in with .run()
    threshold: float
    name: str
    version: str


def load_champion() -> TabularModel:
    """The @champion triage model, from the registry or the serving bundle."""
    name = model_store.TABULAR_MODEL
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cache = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/claim-triage-models")) / "batch"
        info = model_store._download_champion(name, cache)
        path, version, tags = Path(info["onnx"]), info["version"], info["tags"]
    else:
        meta = json.loads((model_store.BUNDLE_DIR / "metadata.json").read_text())["tabular"]
        path, version, tags = model_store.BUNDLE_DIR / "tabular.onnx", meta["version"], meta["tags"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return TabularModel(session, float(tags["decision_threshold"]), name, str(version))


def feeds(claims: pd.DataFrame) -> dict[str, np.ndarray]:
    """ONNX inputs for many claims, with the serving code's feature logic."""
    features = load_config()["features"]
    data = claims.copy()
    data["log_density"] = [log_density(float(d)) for d in data["density"]]
    out = {c: data[[c]].to_numpy(np.float32) for c in features["numeric"]}
    out.update({c: data[[c]].astype(str).to_numpy(object) for c in features["categorical"]})
    return out


def score_frame(claims: pd.DataFrame, model: TabularModel, scored_at: datetime) -> pd.DataFrame:
    """One prediction row per claim, in the predictions table's column order."""
    probability = model.session.run(None, feeds(claims))[1][:, 1].astype(float)
    return pd.DataFrame(
        {
            "claim_id": claims["claim_id"].to_numpy(),
            "report_date": claims["report_date"].to_numpy(),
            "scored_at": scored_at,
            "model_name": model.name,
            "model_version": model.version,
            "probability": probability,
            "threshold": model.threshold,
            "route": np.where(probability >= model.threshold, "senior_adjuster", "standard"),
        }
    )


def explain_frame(claims: pd.DataFrame, model: TabularModel, explainer: Explainer) -> pd.DataFrame:
    """Reason codes for the claims, in the reasons table's column order.

    Re-checks parity on these exact claims first: the reasons must describe
    the model that routed them, or the night fails (ParityError).
    """
    data_cfg, explain_cfg = load_config(), train_config()["explain"]
    rows = add_row_features(claims)
    check_parity(explainer.pipeline, model.session, rows, data_cfg, explainer.tolerance)
    codes = reasons.reason_codes(
        rows,
        reasons.contributions(explainer.pipeline, rows),
        top_k=int(explain_cfg["top_k"]),
        min_contribution=float(explain_cfg["min_contribution"]),
    )
    codes = codes.merge(claims[["claim_id", "report_date"]], on="claim_id")
    return codes[["claim_id", "report_date", "rank", "feature", "value", "contribution"]]


def claims_reported_on(day: date, cfg: dict[str, Any]) -> pd.DataFrame:
    claims = pd.read_parquet(resolve(cfg["batch"]["source"]))
    schedule = build_schedule(claims, cfg)
    reported = schedule[schedule["report_date"] == day]
    return claims.merge(reported[["claim_id", "report_date"]], on="claim_id")


def run(
    day: date,
    model: TabularModel | None = None,
    cfg: dict[str, Any] | None = None,
    explainer: Explainer | None = None,
) -> int:
    """Score one day. Without a model, loads @champion and its explainer; a
    caller that passes a model gets reason codes only if it passes an
    explainer too (tests and fakes have none)."""
    cfg = cfg or load_config("batch_config.yaml")
    claims = claims_reported_on(day, cfg)
    if model is None:
        model = load_champion()
        explainer = explainer or load_explainer(model.session)
    rows = score_frame(claims, model, datetime.now())
    # Computed before the transaction, so a parity failure writes nothing.
    codes = explain_frame(claims, model, explainer) if explainer and len(claims) else None
    con = connect(resolve(cfg["batch"]["store"]))
    try:
        con.begin()
        con.execute("DELETE FROM predictions WHERE report_date = ?", [day])
        con.execute("DELETE FROM reasons WHERE report_date = ?", [day])
        if len(rows):
            con.register("new_rows", rows)
            con.execute("INSERT INTO predictions SELECT * FROM new_rows")
        if codes is not None and len(codes):
            con.register("new_reasons", codes)
            con.execute("INSERT INTO reasons SELECT * FROM new_reasons")
        con.execute(
            "INSERT INTO batch_runs VALUES ('score', ?, now(), ?, ?)",
            [day, len(rows), f"{model.name} v{model.version}"],
        )
        con.commit()
    finally:
        con.close()
    log_event(
        logger,
        logging.INFO,
        "batch_scored",
        report_date=day.isoformat(),
        claims=len(rows),
        senior_adjuster=int((rows["route"] == "senior_adjuster").sum()),
        reason_codes=0 if codes is None else len(codes),
        model=f"{model.name} v{model.version}",
    )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    run(parser.parse_args(argv).date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
