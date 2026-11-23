"""Delayed-label performance monitoring: how good were the batch predictions,
now that the claims' real costs are known?

Joins predictions to outcomes and groups claims into cohorts by report date
(`monitor.cohort_days`). A cohort is judged only once it is mature: `maturity_days` have passed
since its last report date, so even its slowest claim has settled, and it
holds `min_positives` high-cost claims. Expensive claims settle last, so an
immature cohort's labels are biased toward cheap claims. A coverage rule
("95% labeled") isn't enough: the last 5% are the expensive claims.

For each mature cohort, three checks:
    calibration  the realized high-cost rate vs the rate the model expected
                 (its mean predicted probability): alert if the expected rate
                 is outside the realized rate's Wilson interval
    recall       alert if the whole interval is below the baseline
    precision    alert if the whole interval is below the baseline
PR-AUC is reported but not alerted on: with ~50 positives per cohort it is
too noisy to judge alone.

Baseline: the champion scored on the validation split, at its own decision
threshold, so it describes the same model on data it wasn't tuned on.

Writes reports/batch/performance_<as-of>.json and logs one event:
"performance_ok", or "performance_degraded" with exit status 1.

Usage:
    python -m src.batch.performance --as-of 2025-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, timedelta
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.batch.score import TabularModel, feeds, load_champion
from src.batch.store import connect
from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event

logger = get_logger("batch.performance")
REPORT_DIR = ROOT / "reports" / "batch"


def wilson(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when there are no trials."""
    if trials == 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def classification_counts(labels: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    return {
        "positives": int(labels.sum()),
        "flagged": int(flagged.sum()),
        "true_positives": int((labels & flagged).sum()),
    }


def baseline(model: TabularModel) -> dict[str, float]:
    """The champion's precision, recall, PR-AUC and positive rate on validation."""
    data = load_config()
    split = pd.read_parquet(resolve(data["paths"]["processed_dir"]) / "validation.parquet")
    probability = model.session.run(None, feeds(split))[1][:, 1]
    labels = split["is_high_cost"].to_numpy(bool)
    counts = classification_counts(labels, probability >= model.threshold)
    return {
        "rows": len(split),
        "precision": counts["true_positives"] / max(counts["flagged"], 1),
        "recall": counts["true_positives"] / max(counts["positives"], 1),
        "pr_auc": float(average_precision_score(labels, probability)),
        "positive_rate": float(labels.mean()),
    }


def judge_cohorts(
    joined: pd.DataFrame, base: dict[str, float], cfg: dict[str, Any], as_of: date
) -> list[dict[str, Any]]:
    """One result per cohort. joined: report_date, probability, threshold,
    route, is_high_cost (NaN when the outcome isn't known yet)."""
    batch, mon = cfg["batch"], cfg["batch"]["monitor"]
    start = date.fromisoformat(str(batch["start_date"]))
    size, conf = int(mon["cohort_days"]), float(mon["confidence"])
    offset = joined["report_date"].map(lambda d: (d - start).days // size)
    results = []
    for number, cohort in joined.groupby(offset):
        labeled = cohort[cohort["is_high_cost"].notna()]
        labels = labeled["is_high_cost"].to_numpy(bool)
        flagged = labeled["route"].eq("senior_adjuster").to_numpy()
        counts = classification_counts(labels, flagged)
        coverage = len(labeled) / len(cohort)
        first = start + timedelta(days=int(number) * size)
        settled_by = first + timedelta(days=size - 1 + int(mon["maturity_days"]))
        mature = as_of >= settled_by and counts["positives"] >= int(mon["min_positives"])
        result: dict[str, Any] = {
            "cohort": f"{first.isoformat()}..{(first + timedelta(days=size - 1)).isoformat()}",
            "scored": len(cohort),
            "labeled": len(labeled),
            "label_coverage": round(coverage, 4),
            "settled_by": settled_by.isoformat(),
            **counts,
            "mature": mature,
            "alerts": [],
        }
        if len(labeled):
            observed = float(labels.mean())
            expected = float(labeled["probability"].mean())
            precision = counts["true_positives"] / max(counts["flagged"], 1)
            recall = counts["true_positives"] / max(counts["positives"], 1)
            result |= {
                "high_cost_rate": round(observed, 4),
                "high_cost_rate_ci": [
                    round(x, 4) for x in wilson(int(labels.sum()), len(labels), conf)
                ],
                "expected_rate": round(expected, 4),
                "precision": round(precision, 4),
                "precision_ci": [
                    round(x, 4) for x in wilson(counts["true_positives"], counts["flagged"], conf)
                ],
                "recall": round(recall, 4),
                "recall_ci": [
                    round(x, 4) for x in wilson(counts["true_positives"], counts["positives"], conf)
                ],
                "pr_auc": round(float(average_precision_score(labels, labeled["probability"])), 4)
                if 0 < labels.sum() < len(labels)
                else None,
            }
        if mature:
            low, high = result["high_cost_rate_ci"]
            if not low <= result["expected_rate"] <= high:
                result["alerts"].append(
                    f"calibration: high-cost rate {result['high_cost_rate']:.4f} "
                    f"(CI {low:.4f}-{high:.4f}) vs {result['expected_rate']:.4f} expected"
                )
            for metric in ("recall", "precision"):
                upper = result[f"{metric}_ci"][1]
                if upper < base[metric]:
                    result["alerts"].append(
                        f"{metric}: {result[metric]:.4f} (CI up to {upper:.4f}) "
                        f"below baseline {base[metric]:.4f}"
                    )
        results.append(result)
    return results


def run(as_of: date, model: TabularModel | None = None, cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config("batch_config.yaml")
    model = model or load_champion()
    con = connect(resolve(cfg["batch"]["store"]))
    try:
        joined = con.execute(
            """SELECT p.report_date, p.probability, p.threshold, p.route, p.model_version,
                      o.is_high_cost
               FROM predictions p LEFT JOIN outcomes o USING (claim_id)
               WHERE o.label_date IS NULL OR o.label_date <= ?""",
            [as_of],
        ).df()
    finally:
        con.close()
    # DuckDB returns timestamps; cohorts are computed on plain dates
    joined["report_date"] = pd.to_datetime(joined["report_date"]).dt.date
    # Outcomes after the as-of date aren't known yet; keep their predictions
    joined["is_high_cost"] = joined["is_high_cost"].astype("object")
    base = baseline(model)
    cohorts = judge_cohorts(joined, base, cfg, as_of)
    alerts = [f"{c['cohort']} {a}" for c in cohorts for a in c["alerts"]]
    report = {
        "as_of": as_of.isoformat(),
        "model": f"{model.name} v{model.version}",
        "baseline_validation": {k: round(v, 4) for k, v in base.items()},
        "cohorts": cohorts,
        "alerts": alerts,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"performance_{as_of.isoformat()}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    mature = sum(c["mature"] for c in cohorts)
    log_event(
        logger,
        logging.WARNING if alerts else logging.INFO,
        "performance_degraded" if alerts else "performance_ok",
        as_of=as_of.isoformat(),
        cohorts=len(cohorts),
        mature=mature,
        alerts=alerts,
    )
    return 1 if alerts else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    return run(parser.parse_args(argv).as_of)


if __name__ == "__main__":
    sys.exit(main())
