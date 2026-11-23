"""Monitoring assets and sensors: from an alert to a retraining run.

    feature_drift          runs the drift detector (its own pinned env) on the
                           API's inference log; status ok, drift or
                           insufficient_data
    retrain_on_alert       sensor: when the newest model_performance is
                           "degraded" or the newest feature_drift is "drift",
                           request retrain_tabular, at most once per
                           `retrain.cooldown_hours` (configs/batch_config.yaml)
    log_failed_runs        sensor: every failed run is logged as one
                           structured event (run_failed)

The gate still decides: a retrained challenger reaches production only if it
beats the champion.
"""

import json
import logging
import time
from typing import Any

import dagster as dg

from src.common.config import ROOT, load_config
from src.common.logging import get_logger, log_event
from src.orchestration.resources import Commands

logger = get_logger("orchestration.sensors")
DRIFT_REPORT = ROOT / "reports" / "drift_report.json"
WATCHED = {"model_performance": "degraded", "feature_drift": "drift"}


@dg.asset(
    group_name="monitoring",
    check_specs=[dg.AssetCheckSpec("no_feature_drift", asset="feature_drift")],
    description="Drift of the API's logged requests vs the training reference (Evidently).",
)
def feature_drift(commands: Commands) -> dg.MaterializeResult:
    code = commands.run(
        ["uv", "run", "--script", "monitoring/drift_detector.py"], ok_codes=(0, 1, 2)
    )
    status = {0: "ok", 1: "drift", 2: "insufficient_data"}[code]
    report = json.loads(DRIFT_REPORT.read_text()) if code != 2 and DRIFT_REPORT.exists() else {}
    drifted = report.get("drifted_columns", [])
    return dg.MaterializeResult(
        metadata={
            "status": status,
            "drift_share": float(report.get("drift_share", 0.0)),
            "drifted_columns": dg.MetadataValue.json(drifted),
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_feature_drift",
                passed=status != "drift",
                severity=dg.AssetCheckSeverity.WARN,
                metadata={"status": status},
            )
        ],
    )


def decide(
    latest: dict[str, tuple[int, str, float]],
    cursor: dict[str, Any],
    now: float,
    cooldown_hours: float,
) -> tuple[str | None, dict[str, Any]]:
    """Whether to retrain, and the new cursor.

    latest: asset name -> (storage id, status, timestamp) of its newest
    materialization. cursor: the storage ids already seen, and when the last
    retrain was requested. Each materialization is considered once.
    """
    seen = dict(cursor.get("seen", {}))
    reasons = []
    for name, (storage_id, status, _ts) in sorted(latest.items()):
        if storage_id <= seen.get(name, -1):
            continue
        seen[name] = storage_id
        if status == WATCHED.get(name):
            reasons.append(f"{name}={status}")
    last = float(cursor.get("last_retrain", 0.0))
    new_cursor = {"seen": seen, "last_retrain": last}
    if not reasons:
        return None, new_cursor
    if now - last < cooldown_hours * 3600:
        return None, new_cursor | {"suppressed": ", ".join(reasons)}
    return ", ".join(reasons), {"seen": seen, "last_retrain": now}


def _latest(instance: dg.DagsterInstance, name: str) -> tuple[int, str, float] | None:
    records = instance.fetch_materializations(dg.AssetKey(name), limit=1).records
    if not records:
        return None
    record = records[0]
    materialization = record.event_log_entry.asset_materialization
    status = materialization.metadata.get("status") if materialization else None
    value = str(status.value) if status is not None else ""
    return int(record.storage_id), value, float(record.timestamp)


@dg.sensor(
    job_name="retrain_tabular",
    minimum_interval_seconds=60,
    default_status=dg.DefaultSensorStatus.STOPPED,
    description="Retrain the triage model when performance degrades or features drift.",
)
def retrain_on_alert(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    latest = {
        name: found for name in WATCHED if (found := _latest(context.instance, name)) is not None
    }
    cursor = json.loads(context.cursor) if context.cursor else {}
    cooldown = float(load_config("batch_config.yaml")["retrain"]["cooldown_hours"])
    reason, new_cursor = decide(latest, cursor, time.time(), cooldown)
    if reason is None:
        skip = new_cursor.pop("suppressed", None)
        message = f"cooldown: {skip}" if skip else "no new alert"
        return dg.SensorResult(skip_reason=message, cursor=json.dumps(new_cursor))
    log_event(logger, logging.WARNING, "retrain_requested", reason=reason)
    return dg.SensorResult(
        run_requests=[
            dg.RunRequest(
                run_key=f"retrain-{int(new_cursor['last_retrain'])}",
                tags={"claim-triage/retrain-reason": reason},
            )
        ],
        cursor=json.dumps(new_cursor),
    )


@dg.run_failure_sensor(default_status=dg.DefaultSensorStatus.RUNNING)
def log_failed_runs(context: dg.RunFailureSensorContext) -> None:
    log_event(
        logger,
        logging.ERROR,
        "run_failed",
        job=context.dagster_run.job_name,
        run_id=context.dagster_run.run_id,
        error=context.failure_event.message,
    )


MONITORING_ASSETS = [feature_drift]
SENSORS = [retrain_on_alert, log_failed_runs]
