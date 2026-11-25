"""Fairness audit of the tabular champion: does routing treat groups alike?

Scores the untouched test split with the served @champion (ONNX, from the
registry or the bundle, as the batch job does) at its decision threshold,
then compares groups of driver age and region (configs/train_config.yaml,
fairness):

    recall               share of high-cost claims routed to a senior
                         adjuster (equal opportunity: the judged metric)
    flagged_rate         share of all claims routed there
    false_positive_rate  share of standard claims routed there (workload)
    precision            share of routed claims that were high-cost
    calibration          mean predicted probability / actual high-cost rate
                         (1.0 = the model's risk level is right for the group)

A group is flagged when its recall's Wilson interval lies entirely below the
overall recall. Groups with too few high-cost claims are listed as not
judged. Writes reports/fairness/tabular.json; exits 1 if a group is flagged.

Usage:
    python -m src.evaluation.fairness
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.batch.performance import wilson
from src.batch.score import load_champion
from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.training.onnx_export import onnx_inputs
from src.training.tracking import train_config

logger = get_logger("evaluation.fairness")
REPORT = ROOT / "reports" / "fairness" / "tabular.json"
POOLED = "other regions"


def age_groups(ages: pd.Series, bands: dict[str, list[int]]) -> dict[str, np.ndarray]:
    groups = {name: ((ages >= lo) & (ages < hi)).to_numpy() for name, (lo, hi) in bands.items()}
    covered = np.sum(list(groups.values()), axis=0)
    if not np.all(covered == 1):
        raise ValueError("driver_age_bands must cover every age exactly once")
    return groups


def region_groups(
    regions: pd.Series, labels: pd.Series, min_positives: int
) -> dict[str, np.ndarray]:
    """One group per region with enough high-cost claims; the rest pooled."""
    positives = labels.groupby(regions).sum()
    large = sorted(positives[positives >= min_positives].index)
    groups = {str(r): (regions == r).to_numpy() for r in large}
    rest = ~regions.isin(large).to_numpy()
    if rest.any():
        groups[POOLED] = rest
    return groups


def metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, confidence: float
) -> dict[str, Any]:
    flagged = probabilities >= threshold
    positives, negatives = int(labels.sum()), int((~labels).sum())
    true_pos = int((flagged & labels).sum())
    false_pos = int((flagged & ~labels).sum())
    rate = positives / len(labels) if len(labels) else 0.0
    return {
        "rows": len(labels),
        "positives": positives,
        "positive_rate": round(rate, 4),
        "flagged_rate": round(float(flagged.mean()) if len(labels) else 0.0, 4),
        "recall": round(true_pos / positives, 4) if positives else None,
        "recall_interval": [round(x, 4) for x in wilson(true_pos, positives, confidence)],
        "false_positive_rate": round(false_pos / negatives, 4) if negatives else None,
        "precision": round(true_pos / int(flagged.sum()), 4) if flagged.any() else None,
        "calibration": round(float(probabilities.mean()) / rate, 3) if rate else None,
    }


def audit(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Overall and per-group metrics, and the groups whose recall is worse."""
    labels = frame["is_high_cost"].to_numpy(dtype=bool)
    conf, min_pos = float(cfg["confidence"]), int(cfg["min_positives"])
    overall = metrics(labels, probabilities, threshold, conf)
    by_attribute = {
        "driver_age": age_groups(frame["driver_age"], cfg["driver_age_bands"]),
        "region": region_groups(frame["region"], frame["is_high_cost"], min_pos),
    }
    groups: dict[str, dict[str, Any]] = {}
    alerts = []
    for attribute, masks in by_attribute.items():
        groups[attribute] = {}
        for name, mask in masks.items():
            row = metrics(labels[mask], probabilities[mask], threshold, conf)
            row["judged"] = row["positives"] >= min_pos
            groups[attribute][name] = row
            if row["judged"] and row["recall_interval"][1] < overall["recall"]:
                alerts.append(
                    f"{attribute}={name}: recall {row['recall']} "
                    f"(interval {row['recall_interval']}) below overall {overall['recall']}"
                )
    return {
        "threshold": threshold,
        "confidence": conf,
        "overall": overall,
        "groups": groups,
        "alerts": alerts,
    }


def run() -> int:
    data_cfg, cfg = load_config(), train_config()["fairness"]
    test = pd.read_parquet(resolve(data_cfg["paths"]["processed_dir"]) / "test.parquet")
    model = load_champion()
    probabilities = model.session.run(None, onnx_inputs(test, data_cfg))[1][:, 1]
    report = {
        "model": f"{model.name} v{model.version}",
        **audit(test, probabilities, model.threshold, cfg),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    judged = sum(g["judged"] for groups in report["groups"].values() for g in groups.values())
    log_event(
        logger,
        logging.WARNING if report["alerts"] else logging.INFO,
        "fairness_audited",
        model=report["model"],
        overall_recall=report["overall"]["recall"],
        groups_judged=judged,
        alerts=report["alerts"],
        report=str(REPORT.relative_to(ROOT)),
    )
    return 1 if report["alerts"] else 0


if __name__ == "__main__":
    sys.exit(run())
