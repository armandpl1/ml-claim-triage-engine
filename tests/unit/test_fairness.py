"""Fairness audit: groups cover every claim, small regions are pooled, a
group whose costly claims are clearly missed more often is flagged, and a
group too small to judge never is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import fairness

CFG = {
    "confidence": 0.95,
    "min_positives": 10,
    "driver_age_bands": {"young": [16, 25], "older": [25, 101]},
}


def claims(young: int, older: int, young_pos: int, older_pos: int) -> pd.DataFrame:
    rows = [{"driver_age": 20, "region": "R1", "is_high_cost": i < young_pos} for i in range(young)]
    rows += [
        {"driver_age": 50, "region": "R2", "is_high_cost": i < older_pos} for i in range(older)
    ]
    return pd.DataFrame(rows)


def test_age_bands_must_cover_every_age_once():
    with pytest.raises(ValueError, match="exactly once"):
        fairness.age_groups(pd.Series([20, 30]), {"a": [16, 25]})


def test_small_regions_are_pooled():
    regions = pd.Series(["R1"] * 30 + ["R2"] * 5 + ["R3"] * 5)
    labels = pd.Series([True] * 12 + [False] * 18 + [True] * 2 + [False] * 8)
    groups = fairness.region_groups(regions, labels, min_positives=10)
    assert set(groups) == {"R1", fairness.POOLED}
    assert groups[fairness.POOLED].sum() == 10


def test_a_group_whose_costly_claims_are_missed_is_flagged():
    frame = claims(young=200, older=800, young_pos=40, older_pos=160)
    labels = frame["is_high_cost"].to_numpy()
    young = (frame["driver_age"] < 25).to_numpy()
    # Every costly older claim is caught; only 10 of 40 young ones are.
    p = np.where(labels & ~young, 0.9, 0.01)
    p[np.flatnonzero(labels & young)[:10]] = 0.9
    report = fairness.audit(frame, p, 0.5, CFG)
    assert report["groups"]["driver_age"]["young"]["recall"] == 0.25
    assert report["alerts"] and report["alerts"][0].startswith("driver_age=young")


def test_equal_treatment_raises_no_alert():
    frame = claims(young=200, older=800, young_pos=40, older_pos=160)
    p = np.where(frame["is_high_cost"], 0.9, 0.01)
    report = fairness.audit(frame, p, 0.5, CFG)
    assert report["alerts"] == []
    ages = report["groups"]["driver_age"]
    assert ages["young"]["recall"] == ages["older"]["recall"] == 1.0


def test_a_group_too_small_to_judge_is_never_flagged():
    frame = claims(young=50, older=800, young_pos=5, older_pos=160)
    labels = frame["is_high_cost"].to_numpy()
    p = np.where(
        labels & (frame["driver_age"] >= 25).to_numpy(), 0.9, 0.01
    )  # misses every young one
    report = fairness.audit(frame, p, 0.5, CFG)
    young = report["groups"]["driver_age"]["young"]
    assert (young["recall"], young["judged"]) == (0.0, False)
    assert report["alerts"] == []


def test_metrics_report_routing_and_calibration():
    labels = np.array([True, False, False, False])
    out = fairness.metrics(labels, np.array([0.8, 0.6, 0.1, 0.1]), 0.5, 0.95)
    assert out["flagged_rate"] == 0.5
    assert out["false_positive_rate"] == round(1 / 3, 4)
    assert out["precision"] == 0.5
    assert out["calibration"] == round(0.4 / 0.25, 3)
