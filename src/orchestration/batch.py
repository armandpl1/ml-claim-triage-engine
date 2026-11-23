"""Dagster assets for the nightly batch path, one partition per night.

    batch_predictions  ->  claim_outcomes  ->  model_performance
    (src.batch.score:      (src.batch.labels:   (src.batch.performance, with
     the night's claims)    outcomes known       the check no_performance_alerts)
                            by that night)

The outcomes table holds everything known *by* a night, so nights must run
in order, one at a time: every run of nightly_batch carries the tag
claim-triage/batch, limited to one at a time in src/orchestration/dagster.yaml.
A Dagster backfill over the partitions then replays the period like
`make batch-backfill`.

The scenario ("none" or "inflation") comes from CLAIM_TRIAGE_SCENARIO.
"""

import os
from datetime import date
from typing import Any

import dagster as dg

from src.batch import labels, performance, score
from src.batch.backfill import nights
from src.common.config import load_config

_FIRST, _, _LAST = nights(load_config("batch_config.yaml"))
nightly = dg.DailyPartitionsDefinition(
    start_date=_FIRST.isoformat(), end_date=date.fromordinal(_LAST.toordinal() + 1).isoformat()
)
BATCH_TAG = {"claim-triage/batch": "nightly"}


class BatchSettings(dg.ConfigurableResource):
    """Which scenario the outcomes follow, and an optional store override (tests)."""

    scenario: str = os.environ.get("CLAIM_TRIAGE_SCENARIO", "none")
    store: str = ""
    source: str = ""

    def config(self) -> dict[str, Any]:
        cfg = load_config("batch_config.yaml")
        if self.store:
            cfg["batch"]["store"] = self.store
        if self.source:
            cfg["batch"]["source"] = self.source
        return cfg


def night(context: dg.AssetExecutionContext) -> date:
    return date.fromisoformat(context.partition_key)


@dg.asset(
    group_name="batch",
    partitions_def=nightly,
    description="The night's claims scored by @champion (empty after the last report day).",
)
def batch_predictions(
    context: dg.AssetExecutionContext, batch: BatchSettings
) -> dg.MaterializeResult:
    scored = score.run(night(context), cfg=batch.config())
    return dg.MaterializeResult(metadata={"claims_scored": scored})


@dg.asset(
    group_name="batch",
    partitions_def=nightly,
    deps=[batch_predictions],
    description="Every claim outcome known by the night (costs settle weeks later).",
)
def claim_outcomes(context: dg.AssetExecutionContext, batch: BatchSettings) -> dg.MaterializeResult:
    known = labels.run(night(context), batch.scenario, batch.config())
    return dg.MaterializeResult(metadata={"outcomes_known": known, "scenario": batch.scenario})


@dg.asset(
    group_name="batch",
    partitions_def=nightly,
    deps=[claim_outcomes],
    check_specs=[dg.AssetCheckSpec("no_performance_alerts", asset="model_performance")],
    description="Delayed-label performance per settled cohort vs the validation baseline.",
)
def model_performance(
    context: dg.AssetExecutionContext, batch: BatchSettings
) -> dg.MaterializeResult:
    as_of = night(context)
    status = performance.run(as_of, cfg=batch.config())
    report = performance.read_report(as_of)
    mature = sum(c["mature"] for c in report["cohorts"])
    return dg.MaterializeResult(
        metadata={
            "status": "degraded" if status else "ok",
            "mature_cohorts": mature,
            "alerts": dg.MetadataValue.json(report["alerts"]),
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_performance_alerts",
                passed=not report["alerts"],
                severity=dg.AssetCheckSeverity.WARN,
                metadata={"alerts": dg.MetadataValue.json(report["alerts"])},
            )
        ],
    )


nightly_batch = dg.define_asset_job(
    "nightly_batch",
    selection=dg.AssetSelection.assets(batch_predictions, claim_outcomes, model_performance),
    tags=BATCH_TAG,
    description="Score the night's claims, load settled outcomes, check performance.",
)
nightly_batch_schedule = dg.build_schedule_from_partitioned_job(
    nightly_batch, hour_of_day=2, default_status=dg.DefaultScheduleStatus.STOPPED
)

BATCH_ASSETS = [batch_predictions, claim_outcomes, model_performance]
