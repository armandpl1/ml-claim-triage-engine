"""Batch scoring: same features as the API, routes by threshold, one row
per claim, and re-running a day replaces it."""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.batch import score
from src.explain import champion
from src.features.transform import add_row_features
from src.serving.app import tabular_feeds
from src.serving.schemas import ClaimRequest
from src.training.onnx_export import export_tabular

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


@pytest.fixture
def real_champion(claims, cfg):
    """A small real model served as ONNX, and its LightGBM explainer."""
    rows = add_row_features(claims)
    recipe = champion.Recipe(
        {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 5}, {}, 0, "r"
    )
    pipe = champion.rebuild(recipe, rows, cfg)
    session = ort.InferenceSession(export_tabular(pipe, cfg, opset=17).SerializeToString())
    served = score.TabularModel(session, threshold=0.1, name="m", version="1")
    return served, champion.Explainer(pipe, recipe, tolerance=1e-5)


def test_a_night_writes_reason_codes_and_a_rerun_replaces_them(
    real_champion, claims, tmp_path, monkeypatch
):
    model, explainer = real_champion
    day = date(2025, 1, 6)
    monkeypatch.setattr(score, "claims_reported_on", lambda d, cfg: claims.assign(report_date=d))
    cfg = {"batch": {"store": str(tmp_path / "batch.duckdb")}}
    for _ in range(2):
        score.run(day, model, cfg, explainer)
    con = duckdb.connect(cfg["batch"]["store"], read_only=True)
    per_claim = con.execute(
        "SELECT claim_id, max(rank), count(*) FROM reasons GROUP BY 1"
    ).fetchall()
    assert per_claim and all(top == n <= 3 for _, top, n in per_claim)
    orphans = con.execute(
        "SELECT count(*) FROM reasons r ANTI JOIN predictions p USING (claim_id)"
    ).fetchone()
    assert orphans == (0,)
    assert con.execute("SELECT min(contribution) FROM reasons").fetchone()[0] >= 0.01


def test_a_mismatched_explainer_fails_the_night_and_writes_nothing(
    real_champion, claims, cfg, tmp_path, monkeypatch
):
    model, _ = real_champion
    rows = add_row_features(claims)
    other = champion.Recipe({"n_estimators": 5}, {}, 0, "other")
    wrong = champion.Explainer(champion.rebuild(other, rows, cfg), other, tolerance=1e-5)
    monkeypatch.setattr(score, "claims_reported_on", lambda d, c: claims.assign(report_date=d))
    store = tmp_path / "batch.duckdb"
    with pytest.raises(champion.ParityError):
        score.run(date(2025, 1, 6), model, {"batch": {"store": str(store)}}, wrong)
    assert not store.exists()


def test_nights_read_features_as_of_the_report_date(raw_policies, tmp_path):
    """A claim reported before its policy's renewal is scored with the old
    values, one reported after it with the new ones."""
    import dataclasses

    from src.feature_store import offline
    from src.feature_store.registry import load_registry

    registry = dataclasses.replace(load_registry(), offline=tmp_path / "offline.duckdb")
    versions = offline.build(raw_policies, registry)
    offline.write(versions, registry.offline)
    renewed = versions.groupby("policy_id")["valid_from"].max()
    policy = int(renewed.index[0])
    day = renewed[policy].date()

    source = raw_policies.set_index("policy_id").loc[[policy]].reset_index()
    claims = source.assign(claim_id="C1", is_high_cost=False, adjuster_note="hit a post")
    source_path = tmp_path / "claims.parquet"
    claims.to_parquet(source_path)
    cfg = {"batch": {"source": str(source_path), "start_date": "2025-01-01", "days": 1}}
    cfg["batch"].update({"label_delay_days": {"min": 1, "max": 1}, "high_cost_extra_delay_days": 0})

    import datetime as dt

    for report, age in [
        (day - dt.timedelta(days=1), int(source["driver_age"].iloc[0])),
        (day, int(source["driver_age"].iloc[0]) + 1),
    ]:
        cfg["batch"]["start_date"] = report.isoformat()
        rows = score.claims_reported_on(report, cfg, registry)
        assert len(rows) == 1
        assert rows["driver_age"].iloc[0] == age
        assert rows["report_date"].iloc[0] == report
        assert rows["claim_id"].iloc[0] == "C1"
