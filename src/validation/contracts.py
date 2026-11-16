"""Data contracts for the staged claims table.

Fail fast: if any contract fails, the pipeline stops before bad data reaches
features or training. Every failure is logged as one structured JSON alert
(event "contract_failed") listing each broken check and example bad values,
and the process exits with status 1.

Three kinds of contract, all thresholds from configs/data_config.yaml:
    Column contracts   types, nulls, value ranges (e.g. driver_age >= 16),
                       allowed categories, ID formats, unique claim_id
    Dataset contracts  minimum row count, high-cost target share within a
                       range, every note category above a minimum share
    Lineage contract   share of source claims dropped as orphans (no policy)

A JSON report is written to data/processed/validation_report.json either way.

Usage:
    python -m src.validation.contracts
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event

logger = get_logger("validation")


class ContractViolation(Exception):
    """Raised when data breaks a contract."""


@dataclass
class ContractReport:
    table: str
    rows: int
    passed: bool
    failures: list[dict[str, Any]] = field(default_factory=list)


def _range(bounds: dict[str, float]) -> pa.Check:
    return pa.Check.in_range(bounds["min"], bounds["max"])


def claims_schema(cfg: dict[str, Any]) -> pa.DataFrameSchema:
    """Column and dataset contracts for staging.claims."""
    c = cfg["contracts"]
    categories = cfg["notes"]["categories"]

    def high_cost_share_ok(df: pd.DataFrame) -> bool:
        share = float(df["is_high_cost"].mean())
        return bool(c["high_cost_share"]["min"] <= share <= c["high_cost_share"]["max"])

    def categories_balanced(df: pd.DataFrame) -> bool:
        shares = df["claim_category"].value_counts(normalize=True)
        return all(shares.get(cat, 0.0) >= c["min_category_share"] for cat in categories)

    return pa.DataFrameSchema(
        {
            "claim_id": pa.Column(
                str, pa.Check.str_matches(r"^CLM-\d{7}-\d{2}$"), unique=True, nullable=False
            ),
            "policy_id": pa.Column("int64", pa.Check.gt(0), nullable=False),
            "claim_seq": pa.Column("int64", pa.Check.ge(1), nullable=False),
            "claim_amount": pa.Column("float64", pa.Check.ge(c["claim_amount"]["min"])),
            "is_high_cost": pa.Column(bool, nullable=False),
            "claim_category": pa.Column(str, pa.Check.isin(categories), nullable=False),
            "adjuster_note": pa.Column(str, pa.Check.str_length(min_value=10), nullable=False),
            "exposure": pa.Column(
                "float64",
                [pa.Check.gt(c["exposure"]["min"]), pa.Check.le(c["exposure"]["max"])],
            ),
            "area": pa.Column(str, pa.Check.isin(c["areas"])),
            "vehicle_power": pa.Column("int64", _range(c["vehicle_power"])),
            "vehicle_age": pa.Column("int64", _range(c["vehicle_age"])),
            "driver_age": pa.Column("int64", _range(c["driver_age"])),
            "bonus_malus": pa.Column("int64", _range(c["bonus_malus"])),
            "vehicle_brand": pa.Column(str, pa.Check.str_matches(r"^B\d+$")),
            "fuel_type": pa.Column(str, pa.Check.isin(c["fuel_types"])),
            "density": pa.Column("int64", pa.Check.ge(1)),
            "region": pa.Column(str, pa.Check.str_matches(r"^R\d{2}$")),
        },
        checks=[
            pa.Check(lambda df: len(df) >= c["min_rows"], name="min_rows"),
            pa.Check(high_cost_share_ok, name="high_cost_share_in_range"),
            pa.Check(categories_balanced, name="note_categories_balanced"),
        ],
        strict=False,
        coerce=False,
        name="staging.claims",
    )


def _summarize(errors: SchemaErrors, limit: int = 5) -> list[dict[str, Any]]:
    """One entry per broken check, with a few example failing values."""
    cases = errors.failure_cases
    summary = []
    for (column, check), group in cases.groupby(["column", "check"], dropna=False, sort=True):
        summary.append(
            {
                "column": None if pd.isna(column) else str(column),
                "check": str(check),
                "failing_rows": len(group),
                "examples": [str(v) for v in group["failure_case"].head(limit)],
            }
        )
    return summary


def validate_claims(df: pd.DataFrame, cfg: dict[str, Any]) -> ContractReport:
    """Check df against the claims contracts. Raises ContractViolation on failure."""
    try:
        claims_schema(cfg).validate(df, lazy=True)
    except SchemaErrors as errors:
        report = ContractReport("staging.claims", len(df), False, _summarize(errors))
        log_event(logger, logging.ERROR, "contract_failed", **asdict(report))
        raise ContractViolation(f"{len(report.failures)} contract(s) failed") from errors
    return ContractReport("staging.claims", len(df), True)


def check_orphans(source_claims: int, staged_claims: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Lineage contract: share of source claims dropped because their policy is missing."""
    orphan_share = (source_claims - staged_claims) / source_claims if source_claims else 1.0
    limit = cfg["contracts"]["max_orphan_claim_share"]
    result = {
        "check": "max_orphan_claim_share",
        "orphan_share": round(orphan_share, 5),
        "limit": limit,
        "passed": orphan_share <= limit,
    }
    if not result["passed"]:
        log_event(
            logger, logging.ERROR, "contract_failed", table="staging.claims", failures=[result]
        )
        raise ContractViolation(f"orphan claim share {orphan_share:.2%} exceeds {limit:.2%}")
    return result


def run() -> ContractReport:
    cfg = load_config()
    with duckdb.connect(str(resolve(cfg["paths"]["warehouse"])), read_only=True) as con:
        claims = con.sql("SELECT * FROM staging.claims").df()
        source_claims = con.sql("SELECT count(*) FROM raw.claim_amounts").fetchall()[0][0]

    report_path = resolve(cfg["paths"]["validation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lineage = check_orphans(int(source_claims), len(claims), cfg)
        report = validate_claims(claims, cfg)
    except ContractViolation:
        report_path.write_text(json.dumps({"passed": False}, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps({**asdict(report), "lineage": lineage}, indent=2) + "\n")
    log_event(
        logger,
        logging.INFO,
        "contracts_passed",
        table=report.table,
        rows=report.rows,
        orphan_share=lineage["orphan_share"],
    )
    return report


def main() -> None:
    try:
        run()
    except ContractViolation as exc:
        log_event(logger, logging.CRITICAL, "pipeline_halted", reason=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
