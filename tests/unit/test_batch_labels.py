"""Outcomes: only settled claims, rebuilt on each run, and the inflation
scenario changes costs only for claims reported after it starts."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import pytest

from src.batch import labels
from src.batch.schedule import build_schedule

CFG = {
    "batch": {
        "start_date": "2025-01-06",
        "days": 56,
        "label_delay_days": {"min": 14, "max": 60},
        "high_cost_extra_delay_days": 45,
        "inflation": {"start_day": 28, "rate": 0.30},
    }
}
THRESHOLD = 5000.0


@pytest.fixture
def claims() -> pd.DataFrame:
    amounts = [1000.0, 4000.0, 6000.0, 4500.0] * 100
    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:07d}-01" for i in range(len(amounts))],
            "claim_amount": amounts,
            "is_high_cost": [a > THRESHOLD for a in amounts],
        }
    )


def test_only_settled_claims_are_known(claims):
    as_of = date(2025, 3, 1)
    known = labels.outcomes_known_by(claims, as_of, CFG, "none", THRESHOLD)
    schedule = build_schedule(claims, CFG)
    assert len(known) == (schedule["label_date"] <= as_of).sum() > 0
    assert (known["label_date"] <= as_of).all()
    everything = labels.outcomes_known_by(claims, date(2025, 12, 31), CFG, "none", THRESHOLD)
    assert len(everything) == len(claims)


def test_inflation_raises_costs_only_after_it_starts(claims):
    after = labels.apply_scenario(claims, CFG, "inflation", THRESHOLD)
    reported = build_schedule(claims, CFG).set_index("claim_id")["report_date"]
    late = claims["claim_id"].map(reported) >= date(2025, 2, 3)  # start + 28 days
    assert late.any() and (~late).any()
    pd.testing.assert_series_equal(after["claim_amount"][~late], claims["claim_amount"][~late])
    assert (after["claim_amount"][late] == claims["claim_amount"][late] * 1.3).all()
    # 4000 * 1.3 = 5200 and 4500 * 1.3 = 5850 cross the threshold; 1000 doesn't
    flipped = late & ~claims["is_high_cost"] & after["is_high_cost"]
    assert set(claims.loc[flipped, "claim_amount"]) == {4000.0, 4500.0}


def test_unknown_scenario_is_rejected(claims):
    with pytest.raises(ValueError, match="unknown scenario"):
        labels.apply_scenario(claims, CFG, "recession", THRESHOLD)


def test_rerun_rebuilds_the_outcomes(claims, tmp_path, monkeypatch):
    monkeypatch.setattr(labels.pd, "read_parquet", lambda _path: claims)
    cfg = {"batch": {**CFG["batch"], "source": "unused", "store": str(tmp_path / "b.duckdb")}}
    late = labels.run(date(2025, 6, 30), cfg=cfg)
    early = labels.run(date(2025, 2, 15), cfg=cfg)  # moving the date back works too
    assert late == len(claims) and 0 < early < late
    con = duckdb.connect(cfg["batch"]["store"], read_only=True)
    assert con.execute("SELECT count(*) FROM outcomes").fetchone() == (early,)
