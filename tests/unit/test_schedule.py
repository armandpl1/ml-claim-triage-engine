"""Arrival schedule: deterministic, within the configured window, and
expensive claims settle later."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.batch.schedule import build_schedule

CFG = {
    "batch": {
        "start_date": "2025-01-06",
        "days": 56,
        "label_delay_days": {"min": 14, "max": 60},
        "high_cost_extra_delay_days": 45,
    }
}


def claims(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:07d}-01" for i in range(n)],
            "is_high_cost": [i % 5 == 0 for i in range(n)],
        }
    )


def test_schedule_is_deterministic():
    pd.testing.assert_frame_equal(build_schedule(claims(), CFG), build_schedule(claims(), CFG))


def test_dates_stay_in_the_configured_window():
    s = build_schedule(claims(), CFG).merge(claims(), on="claim_id")
    assert s["report_date"].min() >= date(2025, 1, 6)
    assert s["report_date"].max() <= date(2025, 3, 2)  # 56 days from the start
    delay = (s["label_date"] - s["report_date"]).map(lambda d: d.days)
    cheap, costly = delay[~s["is_high_cost"]], delay[s["is_high_cost"]]
    assert cheap.between(14, 60).all()
    assert costly.between(14 + 45, 60 + 45).all()


def test_expensive_claims_settle_later_on_average():
    s = build_schedule(claims(), CFG).merge(claims(), on="claim_id")
    delay = (s["label_date"] - s["report_date"]).map(lambda d: d.days)
    assert delay[s["is_high_cost"]].mean() > delay[~s["is_high_cost"]].mean() + 30


def test_claims_spread_over_many_days():
    assert build_schedule(claims(), CFG)["report_date"].nunique() > 40
