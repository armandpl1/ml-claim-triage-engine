"""Load the outcomes known by a date: the final cost of every claim that has
settled, from the arrival schedule (src/batch/schedule.py).

The outcomes table is rebuilt from scratch on each run (everything with
label_date <= the as-of date), so re-running is safe and moving the date
back is too.

Scenario "inflation" (configs/batch_config.yaml): claims reported from
`inflation.start_day` on cost `inflation.rate` more, and is_high_cost is
recomputed with the same threshold as training. The features are untouched,
so the feature-drift detector sees nothing; only the outcomes reveal that
the model now misses more expensive claims. Label dates follow the new
costs: a claim pushed over the threshold settles as slowly as other
expensive claims.

Usage:
    python -m src.batch.labels --as-of 2025-03-31
    python -m src.batch.labels --as-of 2025-03-31 --scenario inflation
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.batch.schedule import build_schedule
from src.batch.store import connect
from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event

logger = get_logger("batch.labels")
SCENARIOS = ("none", "inflation")


def apply_scenario(
    claims: pd.DataFrame, cfg: dict[str, Any], scenario: str, high_cost_threshold: float
) -> pd.DataFrame:
    """The claims as they turn out under the scenario (costs and target)."""
    if scenario == "none":
        return claims
    if scenario != "inflation":
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")
    batch = cfg["batch"]
    start = date.fromisoformat(str(batch["start_date"])) + timedelta(
        days=int(batch["inflation"]["start_day"])
    )
    reported = build_schedule(claims, cfg).set_index("claim_id")["report_date"]
    late = claims["claim_id"].map(reported) >= start
    out = claims.copy()
    out.loc[late, "claim_amount"] = out.loc[late, "claim_amount"] * (
        1 + float(batch["inflation"]["rate"])
    )
    out["is_high_cost"] = out["claim_amount"] > high_cost_threshold
    return out


def outcomes_known_by(
    claims: pd.DataFrame, as_of: date, cfg: dict[str, Any], scenario: str, threshold: float
) -> pd.DataFrame:
    """claim_id, label_date, claim_amount, is_high_cost for settled claims."""
    actual = apply_scenario(claims, cfg, scenario, threshold)
    schedule = build_schedule(actual, cfg)
    known = actual.merge(schedule, on="claim_id")
    known = known[known["label_date"] <= as_of]
    return known[["claim_id", "label_date", "claim_amount", "is_high_cost"]].reset_index(drop=True)


def run(as_of: date, scenario: str = "none", cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config("batch_config.yaml")
    threshold = float(load_config()["targets"]["high_cost_threshold"])
    claims = pd.read_parquet(resolve(cfg["batch"]["source"]))
    rows = outcomes_known_by(claims, as_of, cfg, scenario, threshold)
    con = connect(resolve(cfg["batch"]["store"]))
    try:
        con.begin()
        con.execute("DELETE FROM outcomes")
        if len(rows):
            con.register("new_rows", rows)
            con.execute("INSERT INTO outcomes SELECT * FROM new_rows")
        con.execute(
            "INSERT INTO batch_runs VALUES ('labels', ?, now(), ?, ?)",
            [as_of, len(rows), f"scenario={scenario}"],
        )
        con.commit()
    finally:
        con.close()
    log_event(
        logger,
        logging.INFO,
        "outcomes_loaded",
        as_of=as_of.isoformat(),
        scenario=scenario,
        outcomes=len(rows),
        high_cost=int(rows["is_high_cost"].sum()),
    )
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--scenario", choices=SCENARIOS, default="none")
    args = parser.parse_args(argv)
    run(args.as_of, args.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
