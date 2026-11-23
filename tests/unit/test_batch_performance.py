"""Performance monitor: correct intervals, immature cohorts aren't judged,
and each alert fires on the failure it is meant to catch."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.batch.performance import judge_cohorts, wilson

CFG = {
    "batch": {
        "start_date": "2025-01-06",
        "monitor": {
            "cohort_days": 14,
            "min_label_coverage": 0.95,
            "min_positives": 20,
            "confidence": 0.95,
        },
    }
}
BASE = {"precision": 0.12, "recall": 0.25, "pr_auc": 0.09, "positive_rate": 0.05}


def cohort(
    n: int = 1000,
    positive_rate: float = 0.05,
    probability: float = 0.05,
    recall: float = 0.25,
    precision: float = 0.12,
    labeled_share: float = 1.0,
) -> pd.DataFrame:
    """One 14-day cohort with the given realized rate and model behavior."""
    positives = round(n * positive_rate)
    caught = round(positives * recall)
    flagged = round(caught / precision)
    labels = np.zeros(n, bool)
    labels[:positives] = True
    route = np.full(n, "standard", dtype=object)
    route[:caught] = "senior_adjuster"  # true positives
    route[positives : positives + flagged - caught] = "senior_adjuster"  # false positives
    is_high_cost = pd.Series(labels, dtype="object")
    is_high_cost[int(n * labeled_share) :] = None  # outcomes not known yet
    start = date(2025, 1, 6)
    return pd.DataFrame(
        {
            "report_date": [start + timedelta(days=i % 14) for i in range(n)],
            "probability": probability,
            "threshold": 0.08,
            "route": route,
            "is_high_cost": is_high_cost,
        }
    )


def test_wilson_matches_known_values():
    assert wilson(0, 0, 0.95) == (0.0, 1.0)
    low, high = wilson(5, 10, 0.95)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    low, high = wilson(0, 50, 0.95)
    assert low == 0.0 and high == pytest.approx(0.0713, abs=1e-4)


def test_a_healthy_cohort_raises_no_alert():
    [result] = judge_cohorts(cohort(), BASE, CFG)
    assert result["mature"] and result["alerts"] == []
    assert result["label_coverage"] == 1.0 and result["positives"] == 50


def test_immature_cohorts_are_reported_but_not_judged():
    # Only 60% labeled: looks terrible (no positives yet) but must not alert
    early = cohort(labeled_share=0.6, positive_rate=0.05)
    early.loc[early["is_high_cost"].notna(), "is_high_cost"] = False
    [result] = judge_cohorts(early, BASE, CFG)
    assert not result["mature"] and result["alerts"] == []
    assert result["labeled"] == 600


def test_more_expensive_claims_than_expected_breaks_calibration():
    [result] = judge_cohorts(cohort(positive_rate=0.09, probability=0.05), BASE, CFG)
    assert any(a.startswith("calibration") for a in result["alerts"])


def test_a_clear_recall_drop_alerts():
    [result] = judge_cohorts(cohort(positive_rate=0.08, probability=0.08, recall=0.05), BASE, CFG)
    assert any(a.startswith("recall") for a in result["alerts"])
    assert not any(a.startswith("calibration") for a in result["alerts"])


def test_a_small_drop_within_noise_does_not_alert():
    [result] = judge_cohorts(cohort(recall=0.2), BASE, CFG)  # 10 of 50 caught vs 0.25
    assert result["alerts"] == []
