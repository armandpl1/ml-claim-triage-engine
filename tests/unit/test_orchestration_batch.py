"""Nightly batch assets: a night runs score -> outcomes -> performance on
real code with a fake model, and the alert check reflects the report."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.batch import performance, score
from src.orchestration import batch as b
from src.orchestration.definitions import defs


class FakeSession:
    def run(self, _outputs, feeds):
        p = np.full(len(feeds["driver_age"]), 0.05)
        return [np.zeros(len(p)), np.column_stack([1 - p, p])]


@pytest.fixture
def settings(tmp_path, monkeypatch) -> b.BatchSettings:
    claims = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:07d}-01" for i in range(600)],
            "exposure": 0.5,
            "vehicle_power": 6,
            "vehicle_age": 3,
            "driver_age": 40,
            "bonus_malus": 50,
            "density": 100,
            "area": "C",
            "vehicle_brand": "B12",
            "fuel_type": "Diesel",
            "region": "R11",
            "claim_amount": [7000.0 if i % 20 == 0 else 900.0 for i in range(600)],
            "is_high_cost": [i % 20 == 0 for i in range(600)],
        }
    )
    source = tmp_path / "claims.parquet"
    claims.to_parquet(source)
    monkeypatch.setattr(
        score, "load_champion", lambda: score.TabularModel(FakeSession(), 0.08, "m", "1")
    )
    monkeypatch.setattr(performance, "load_champion", score.load_champion)
    monkeypatch.setattr(score, "load_explainer", lambda _session: None)  # fake model
    monkeypatch.setattr(
        performance,
        "baseline",
        lambda _m: {
            "rows": 1,
            "precision": 0.1,
            "recall": 0.2,
            "pr_auc": 0.1,
            "positive_rate": 0.05,
        },
    )
    monkeypatch.setattr(performance, "REPORT_DIR", tmp_path / "reports")
    return b.BatchSettings(store=str(tmp_path / "batch.duckdb"), source=str(source))


def run_night(settings: b.BatchSettings, night: str):
    job = defs.resolve_job_def("nightly_batch")
    return job.execute_in_process(partition_key=night, resources={"batch": settings})


def test_partitions_cover_every_night_of_the_simulation():
    keys = b.nightly.get_partition_keys()
    assert keys[0] == "2025-01-06" and keys[-1] == "2025-06-15"


def test_nightly_batch_job_is_limited_to_one_run_at_a_time():
    assert defs.resolve_job_def("nightly_batch").tags == b.BATCH_TAG


def test_a_night_scores_loads_outcomes_and_checks_performance(settings):
    result = run_night(settings, "2025-01-06")
    assert result.success
    con = duckdb.connect(settings.store, read_only=True)
    scored = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert scored > 0
    assert con.execute("SELECT count(*) FROM outcomes").fetchone() == (0,)  # nothing settled yet
    [check] = result.get_asset_check_evaluations()
    assert check.passed  # no cohort mature on the first night


def test_the_check_fails_when_the_report_has_alerts(settings, monkeypatch):
    def alerting_run(as_of, model=None, cfg=None):
        performance.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (performance.REPORT_DIR / f"performance_{as_of.isoformat()}.json").write_text(
            '{"cohorts": [{"mature": true}], "alerts": ["calibration: too many expensive claims"]}'
        )
        return 1

    monkeypatch.setattr(performance, "run", alerting_run)
    result = run_night(settings, "2025-06-15")
    assert result.success  # a warning, not a crash: the sensor decides what to do
    [check] = result.get_asset_check_evaluations()
    assert not check.passed
    perf = result.asset_materializations_for_node("model_performance")[0]
    assert perf.metadata["status"].value == "degraded"
