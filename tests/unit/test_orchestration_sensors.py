"""Retrain sensor: an alert triggers one retrain, each result is used once,
and a cooldown stops retrain storms."""

from __future__ import annotations

import json

import dagster as dg

from src.orchestration.sensors import decide, retrain_on_alert

HOUR = 3600.0


def test_a_degraded_performance_report_triggers_a_retrain():
    reason, cursor = decide({"model_performance": (5, "degraded", 0.0)}, {}, 1000 * HOUR, 168)
    assert reason == "model_performance=degraded"
    assert cursor == {"seen": {"model_performance": 5}, "last_retrain": 1000 * HOUR}


def test_healthy_results_and_already_seen_alerts_do_nothing():
    healthy = {"model_performance": (5, "ok", 0.0), "feature_drift": (6, "insufficient_data", 0.0)}
    assert decide(healthy, {}, 1000 * HOUR, 168)[0] is None
    seen = {"seen": {"model_performance": 5}, "last_retrain": 0.0}
    assert decide({"model_performance": (5, "degraded", 0.0)}, seen, 1000 * HOUR, 168)[0] is None


def test_the_cooldown_suppresses_a_second_retrain_within_a_week():
    cursor = {"seen": {"model_performance": 5}, "last_retrain": 1000 * HOUR}
    reason, after = decide({"feature_drift": (9, "drift", 0.0)}, cursor, 1100 * HOUR, 168)
    assert reason is None and after["suppressed"] == "feature_drift=drift"
    assert after["seen"]["feature_drift"] == 9  # used once: not retried after the cooldown
    reason, _ = decide({"feature_drift": (10, "drift", 0.0)}, cursor, 1200 * HOUR, 168)
    assert reason == "feature_drift=drift"


def test_the_sensor_reads_the_newest_materialization_from_the_instance():
    with dg.instance_for_test() as instance:
        context = dg.build_sensor_context(instance=instance)
        assert not retrain_on_alert(context).run_requests  # nothing recorded yet

        instance.report_runless_asset_event(
            dg.AssetMaterialization("model_performance", metadata={"status": "degraded"})
        )
        result = retrain_on_alert(dg.build_sensor_context(instance=instance))
        [request] = result.run_requests
        assert request.tags["claim-triage/retrain-reason"] == "model_performance=degraded"

        again = retrain_on_alert(dg.build_sensor_context(instance=instance, cursor=result.cursor))
        assert not again.run_requests
        assert json.loads(again.cursor)["seen"]["model_performance"] >= 1
