"""Build the leakage-free feature sets (pipeline stage `features`).

The tabular features come from the feature store (src/feature_store/),
read point-in-time: each claim gets the policy's features as they were at
its event time. Training claims have no dates in the source; they read the
source snapshot (history.recorded_at), the values the data was published
with.

Order matters, and is the point of this step:
    1. Split the validated claims into train / validation / test FIRST.
    2. Fit the preprocessor on the training split only (medians, category codes).
    3. Save the splits, the fitted preprocessor and the pinned tokenizer.
    4. Save a reference snapshot of the training inputs to data/reference/,
       the baseline for drift monitoring (Step 8).
    5. Write a report proving the split is clean.

Usage:
    python -m src.features.build_features
"""

from __future__ import annotations

import json
import logging

import duckdb
import joblib
import pandas as pd

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.feature_store import offline
from src.feature_store.registry import load_registry
from src.features.split import split_claims
from src.features.text import load_tokenizer
from src.features.transform import add_row_features, build_preprocessor, raw_input_columns

logger = get_logger("features")
ARTIFACTS = ROOT / "artifacts"


def leakage_report(df: pd.DataFrame, cfg: dict) -> dict:
    """Facts about the split that must hold for it to be leakage-free."""
    group = cfg["split_group"]
    policies_per_split = df.groupby(group)["split"].nunique()
    report = {
        "rows": {s: int((df["split"] == s).sum()) for s in ("train", "validation", "test")},
        "high_cost_share": {
            s: round(float(df.loc[df["split"] == s, "is_high_cost"].mean()), 4)
            for s in ("train", "validation", "test")
        },
        "policies_in_more_than_one_split": int((policies_per_split > 1).sum()),
        "excluded_columns": sorted(cfg["features"]["excluded"]),
    }
    report["passed"] = report["policies_in_more_than_one_split"] == 0
    return report


def load_claims(cfg: dict) -> pd.DataFrame:
    """Every validated claim, with its policy's features from the feature store."""
    registry = load_registry(data_cfg=cfg)
    with duckdb.connect(str(resolve(cfg["paths"]["warehouse"])), read_only=True) as con:
        claims = con.sql("SELECT * FROM staging.claims ORDER BY claim_id").df()
    columns = list(claims.columns)
    entities = claims.drop(columns=registry.view.fields)
    entities["event_timestamp"] = pd.Timestamp(registry.recorded_at)
    with offline.connect(registry) as store:
        joined = offline.historical_features(store, entities, registry)
    return joined[columns]


def run() -> dict:
    cfg = load_config()
    claims = load_claims(cfg)

    claims = add_row_features(claims)
    claims["split"] = split_claims(claims, cfg)  # 1. split first
    train = claims[claims["split"] == "train"]

    preprocessor = build_preprocessor(cfg).fit(train)  # 2. fit on train only

    processed = resolve(cfg["paths"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):  # 3. save
        claims[claims["split"] == name].to_parquet(processed / f"{name}.parquet", index=False)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
    load_tokenizer(cfg)

    reference = resolve(cfg["paths"]["reference_dir"])  # 4. drift baseline
    reference.mkdir(parents=True, exist_ok=True)
    keep = [*raw_input_columns(cfg), cfg["features"]["text"], "is_high_cost", "claim_category"]
    train[keep].to_parquet(reference / "reference.parquet", index=False)

    report = leakage_report(claims, cfg)  # 5. report
    (processed / "feature_report.json").write_text(json.dumps(report, indent=2) + "\n")
    level = logging.INFO if report["passed"] else logging.ERROR
    log_event(logger, level, "features_built", **report)
    if not report["passed"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    run()
