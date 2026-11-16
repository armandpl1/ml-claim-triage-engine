"""Fetch freMTPL2 from OpenML and stage it in the DuckDB warehouse.

Idempotent: downloads are cached as Parquet in data/raw/ and skipped when
present (use --force to refresh), and warehouse tables are replaced on every
run, so running twice gives identical tables.

Steps:
    1. Download policies (freMTPL2freq) and claim amounts (freMTPL2sev).
    2. Clean column names and values (snake_case, quoted strings, integer IDs).
    3. Join each claim to its policy. Claims with no policy ("orphans") are
       dropped and counted.
    4. Give every claim a deterministic primary key, claim_id =
       CLM-<policy_id>-<n>, numbering a policy's claims by amount.
    5. Add the high-cost target and a synthetic adjuster note per claim.
    6. Write raw.policies, raw.claim_amounts and staging.claims to DuckDB.

Usage:
    python -m src.ingestion.fetch_data [--force]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from typing import Any

import duckdb
import pandas as pd
from sklearn.datasets import fetch_openml

from src.common.config import load_config, resolve
from src.common.logging import get_logger, log_event
from src.ingestion.notes import generate_note

logger = get_logger("ingestion")

POLICY_COLUMNS = {
    "IDpol": "policy_id",
    "ClaimNb": "claim_count",
    "Exposure": "exposure",
    "Area": "area",
    "VehPower": "vehicle_power",
    "VehAge": "vehicle_age",
    "DrivAge": "driver_age",
    "BonusMalus": "bonus_malus",
    "VehBrand": "vehicle_brand",
    "VehGas": "fuel_type",
    "Density": "density",
    "Region": "region",
}
AMOUNT_COLUMNS = {"IDpol": "policy_id", "ClaimAmount": "claim_amount"}


@dataclass
class IngestStats:
    policies: int
    claim_amounts: int
    orphan_claims: int
    claims: int
    high_cost_claims: int


def download(openml_id: int, name: str, force: bool = False) -> pd.DataFrame:
    """One OpenML dataset as a DataFrame, cached as Parquet in data/raw/."""
    cfg = load_config()
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"{name}.parquet"
    if cache.exists() and not force:
        log_event(logger, logging.INFO, "download_skipped", dataset=name, path=str(cache))
        return pd.read_parquet(cache)
    bunch = fetch_openml(
        data_id=openml_id, as_frame=True, data_home=str(raw_dir / "openml_cache"), parser="auto"
    )
    frame: pd.DataFrame = bunch.frame
    frame.to_parquet(cache, index=False)
    log_event(logger, logging.INFO, "downloaded", dataset=name, rows=len(frame), path=str(cache))
    return frame


def clean_policies(raw: pd.DataFrame) -> pd.DataFrame:
    """snake_case columns, integer IDs, plain strings without stray quotes."""
    df = raw.rename(columns=POLICY_COLUMNS)[list(POLICY_COLUMNS.values())].copy()
    df["policy_id"] = df["policy_id"].astype("int64")
    for col in ("area", "vehicle_brand", "fuel_type", "region"):
        df[col] = df[col].astype(str).str.strip().str.strip("'\"")
    return df


def clean_amounts(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=AMOUNT_COLUMNS)[list(AMOUNT_COLUMNS.values())].copy()
    df["policy_id"] = df["policy_id"].astype("int64")
    df["claim_amount"] = df["claim_amount"].astype("float64")
    return df


def build_claims(
    policies: pd.DataFrame, amounts: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    """One row per claim, joined to its policy. Returns (claims, orphan count)."""
    known = amounts["policy_id"].isin(policies["policy_id"])
    orphans = int((~known).sum())
    matched = amounts[known].copy()

    # Number each policy's claims by amount. A stable sort keeps the source
    # order for equal amounts, so the numbering is deterministic.
    matched["source_row"] = range(len(matched))
    matched = matched.sort_values(["policy_id", "claim_amount", "source_row"], kind="mergesort")
    matched["claim_seq"] = matched.groupby("policy_id").cumcount() + 1
    matched["claim_id"] = [
        f"CLM-{pid:07d}-{seq:02d}"
        for pid, seq in zip(matched["policy_id"], matched["claim_seq"], strict=True)
    ]

    claims = matched.drop(columns="source_row").merge(policies, on="policy_id", how="inner")
    claims["is_high_cost"] = claims["claim_amount"] > cfg["targets"]["high_cost_threshold"]

    seed, typo_rate = cfg["seed"], cfg["notes"]["typo_rate"]
    notes = [
        generate_note(cid, amount, int(age), int(density), seed, typo_rate)
        for cid, amount, age, density in zip(
            claims["claim_id"],
            claims["claim_amount"],
            claims["vehicle_age"],
            claims["density"],
            strict=True,
        )
    ]
    claims["claim_category"] = [category for category, _ in notes]
    claims["adjuster_note"] = [text for _, text in notes]

    ordered = [
        "claim_id",
        "policy_id",
        "claim_seq",
        "claim_amount",
        "is_high_cost",
        "claim_category",
        "adjuster_note",
    ]
    rest = [c for c in claims.columns if c not in ordered]
    claims = claims[ordered + rest].sort_values("claim_id").reset_index(drop=True)
    return claims, orphans


def store(policies: pd.DataFrame, amounts: pd.DataFrame, claims: pd.DataFrame) -> None:
    """Replace the warehouse tables (idempotent)."""
    cfg = load_config()
    path = resolve(cfg["paths"]["warehouse"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        con.execute("CREATE SCHEMA IF NOT EXISTS staging")
        for table, frame in (
            ("raw.policies", policies),
            ("raw.claim_amounts", amounts),
            ("staging.claims", claims),
        ):
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM frame")
            con.unregister("frame")
        con.execute("ALTER TABLE staging.claims ADD PRIMARY KEY (claim_id)")


def run(force: bool = False) -> IngestStats:
    cfg = load_config()
    policies = clean_policies(download(cfg["source"]["policies_openml_id"], "policies", force))
    amounts = clean_amounts(download(cfg["source"]["claims_openml_id"], "claim_amounts", force))
    claims, orphans = build_claims(policies, amounts, cfg)
    store(policies, amounts, claims)
    stats = IngestStats(
        policies=len(policies),
        claim_amounts=len(amounts),
        orphan_claims=orphans,
        claims=len(claims),
        high_cost_claims=int(claims["is_high_cost"].sum()),
    )
    level = logging.WARNING if orphans else logging.INFO
    log_event(logger, level, "ingestion_complete", **asdict(stats))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download from OpenML")
    run(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
