"""Batch scoring: same features as the API, routes by threshold, one row
per claim, and re-running a day replaces it."""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd

from src.batch import score
from src.serving.app import tabular_feeds
from src.serving.schemas import ClaimRequest

CLAIM = {
    "exposure": 0.59,
    "vehicle_power": 11,
    "vehicle_age": 0,
    "driver_age": 39,
    "bonus_malus": 56,
    "density": 778,
    "area": "D",
    "vehicle_brand": "B12",
    "fuel_type": "Diesel",
    "region": "R22",
}


class FakeSession:
    """Stands in for onnxruntime: probability = driver_age / 100."""

    def run(self, _outputs, feeds):
        p = feeds["driver_age"][:, 0].astype(float) / 100
        return [(p > 0.5).astype(int), np.column_stack([1 - p, p])]


def model() -> score.TabularModel:
    return score.TabularModel(FakeSession(), threshold=0.5, name="m", version="7")


def frame(ages: list[int], day: date = date(2025, 1, 6)) -> pd.DataFrame:
    rows = [
        {**CLAIM, "driver_age": a, "claim_id": f"C{i}", "report_date": day}
        for i, a in enumerate(ages)
    ]
    return pd.DataFrame(rows)


def test_batch_features_match_the_api():
    batch = score.feeds(frame([39]))
    api = tabular_feeds(ClaimRequest(**CLAIM))
    assert batch.keys() == api.keys()
    for name in api:
        assert batch[name].dtype == api[name].dtype, name
        assert batch[name].tolist() == api[name].tolist(), name


def test_routes_follow_the_threshold():
    rows = score.score_frame(frame([30, 50, 70]), model(), datetime(2025, 1, 6, 2))
    assert rows["route"].tolist() == ["standard", "senior_adjuster", "senior_adjuster"]
    assert rows["probability"].tolist() == [0.3, 0.5, 0.7]
    assert set(rows["model_version"]) == {"7"}


def test_rerunning_a_day_replaces_it(tmp_path, monkeypatch):
    claims = pd.concat([frame([30, 70]), frame([40], date(2025, 1, 7)).assign(claim_id="D0")])
    monkeypatch.setattr(
        score, "claims_reported_on", lambda day, cfg: claims[claims["report_date"] == day]
    )
    cfg = {"batch": {"store": str(tmp_path / "batch.duckdb")}}
    for _ in range(2):
        assert score.run(date(2025, 1, 6), model(), cfg) == 2
    score.run(date(2025, 1, 7), model(), cfg)
    con = duckdb.connect(cfg["batch"]["store"], read_only=True)
    per_day = con.execute(
        "SELECT report_date, count(*) FROM predictions GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert per_day == [(date(2025, 1, 6), 2), (date(2025, 1, 7), 1)]
    assert con.execute("SELECT count(*) FROM batch_runs").fetchone() == (3,)
