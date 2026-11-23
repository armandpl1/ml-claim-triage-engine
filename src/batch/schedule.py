"""When each incoming claim is reported, and when its outcome becomes known.

The held-out test claims play the part of new claims. Each one gets:
    report_date  the day it arrives and is scored by the nightly batch job,
                 spread uniformly over `days` from `start_date`
    label_date   the day its final cost is known: report_date plus a delay
                 of `label_delay_days`, and `high_cost_extra_delay_days` more
                 for high-cost claims, which take longer to settle

Both come from a stable hash of the claim_id, so the schedule is the same in
every run and on every machine (data standards: deterministic). Settings: configs/batch_config.yaml.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.ingestion.notes import stable_seed


def build_schedule(claims: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """claim_id, report_date and label_date for every claim (dates as datetime.date)."""
    batch = cfg["batch"]
    start = date.fromisoformat(str(batch["start_date"]))
    days = int(batch["days"])
    low, high = int(batch["label_delay_days"]["min"]), int(batch["label_delay_days"]["max"])
    extra = int(batch["high_cost_extra_delay_days"])

    rows = []
    for claim_id, high_cost in zip(claims["claim_id"], claims["is_high_cost"], strict=True):
        report = start + timedelta(days=stable_seed("report", claim_id) % days)
        delay = low + stable_seed("label", claim_id) % (high - low + 1)
        if high_cost:
            delay += extra
        rows.append((claim_id, report, report + timedelta(days=delay)))
    return pd.DataFrame(rows, columns=["claim_id", "report_date", "label_date"])
