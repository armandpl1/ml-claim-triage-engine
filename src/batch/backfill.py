"""Replay the nightly batch jobs over the whole simulated period.

Each night D, in order:
    1. score the claims reported on D (src.batch.score)
    2. load the outcomes known by D (src.batch.labels, with the scenario)
    3. on Mondays, and on the last night, check performance (src.batch.performance)

Starts from an empty store, so a backfill is reproducible. Writes a summary
to reports/batch/backfill_<scenario>.json: every check, and the first night
an alert fired (how long the problem took to detect).

Usage:
    python -m src.batch.backfill                       # normal claims
    python -m src.batch.backfill --scenario inflation  # costs rise from day 28
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from typing import Any

from src.batch import labels, performance, score
from src.batch.performance import REPORT_DIR
from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event

logger = get_logger("batch.backfill")


def nights(cfg: dict[str, Any]) -> tuple[date, date, date]:
    """First night, last report night, and last night (every outcome known)."""
    batch = cfg["batch"]
    first = date.fromisoformat(str(batch["start_date"]))
    last_report = first + timedelta(days=int(batch["days"]) - 1)
    slowest = int(batch["label_delay_days"]["max"]) + int(batch["high_cost_extra_delay_days"])
    return first, last_report, last_report + timedelta(days=slowest)


def run(scenario: str = "none", cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config("batch_config.yaml")
    resolve(cfg["batch"]["store"]).unlink(missing_ok=True)
    model = score.load_champion()
    explainer = score.load_explainer(model.session)  # rebuilt once, not every night
    first, last_report, last = nights(cfg)

    checks = []
    night = first
    while night <= last:
        if night <= last_report:
            score.run(night, model, cfg, explainer)
        labels.run(night, scenario, cfg)
        if night.weekday() == 0 or night == last:
            status = performance.run(night, model, cfg)
            report = json.loads((REPORT_DIR / f"performance_{night.isoformat()}.json").read_text())
            checks.append(
                {
                    "as_of": night.isoformat(),
                    "mature_cohorts": sum(c["mature"] for c in report["cohorts"]),
                    "cohorts": len(report["cohorts"]),
                    "alerts": report["alerts"],
                    "status": "degraded" if status else "ok",
                }
            )
        night += timedelta(days=1)

    alerted = [c for c in checks if c["alerts"]]
    summary = {
        "scenario": scenario,
        "model": f"{model.name} v{model.version}",
        "nights": f"{first.isoformat()}..{last.isoformat()}",
        "first_alert": alerted[0]["as_of"] if alerted else None,
        "final": checks[-1],
        "checks": checks,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"backfill_{scenario}.json").write_text(json.dumps(summary, indent=2) + "\n")
    log_event(
        logger,
        logging.INFO,
        "backfill_done",
        scenario=scenario,
        checks=len(checks),
        first_alert=summary["first_alert"],
        final_status=checks[-1]["status"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--scenario", choices=labels.SCENARIOS, default="none")
    run(parser.parse_args(argv).scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
