# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "evidently==0.7.23",
#     "pandas>=2.2,<3",
#     "pyarrow>=17",
#     "pyyaml>=6",
# ]
# ///
"""Drift report: what the API is receiving vs what the models were trained on.

Compares the requests logged by the API (SQLite inference log, written by
src/serving/app.py) with the training reference (data/reference/, written by
src/features/build_features.py), using Evidently's data drift preset:

    input features    each numeric feature (K-S test, or Wasserstein distance
                      on large samples) and categorical feature (chi-square)
    note length       words per adjuster note, a cheap signal that the text
                      the notes model sees has changed
    note category     the notes model's predicted categories vs the true
                      categories in training (prediction drift)

Writes reports/drift_report.html (to read) and reports/drift_report.json (to
automate on), and logs one JSON line. Settings are in configs/data_config.yaml
under monitoring:.

Exit status: 0 no drift alert, 1 drift alert (the share of drifted columns
reached alert_drift_share), 2 too few logged requests to judge.

Why a separate environment: Evidently 0.7 does not run on pandas 3, which the
rest of the project uses. The dependencies above are pinned for this script
alone (PEP 723, locked in drift_detector.py.lock), the way a drift job runs as
its own scheduled container in production. Run it with:

    uv run --script monitoring/drift_detector.py [--since-hours 24]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
NOTE_LENGTH = "note_words"
CATEGORY = "note_category"


def load_config() -> dict[str, Any]:
    with open(ROOT / "configs" / "data_config.yaml") as f:
        return yaml.safe_load(f)


def load_requests(log_path: Path, since_hours: float | None) -> pd.DataFrame:
    """Logged requests, one row per prediction, features expanded to columns."""
    if not log_path.exists():
        return pd.DataFrame()
    query = "SELECT ts, features, adjuster_note, note_category FROM predictions"
    params: tuple[float, ...] = ()
    if since_hours is not None:
        query += " WHERE ts >= ?"
        params = (time.time() - since_hours * 3600,)
    with sqlite3.connect(log_path) as con:
        rows = pd.read_sql_query(query, con, params=params)
    if rows.empty:
        return rows
    features = pd.DataFrame([json.loads(f) for f in rows["features"]])
    return pd.concat([features, rows[["adjuster_note", "note_category"]]], axis=1)


def comparable(frame: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """The columns both sides share, with the same types."""
    out = frame[numeric].astype(float)
    for column in categorical:
        out[column] = frame[column].astype("string")
    out[NOTE_LENGTH] = frame["adjuster_note"].fillna("").str.split().str.len().astype(float)
    out[CATEGORY] = frame[CATEGORY].astype("string")
    return out


def drift_summary(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
    """Per-column results and the drifted share, from Evidently's report."""
    columns: dict[str, Any] = {}
    share = 0.0
    for metric in snapshot_dict["metrics"]:
        config = metric["config"]
        if config["type"].endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif config["type"].endswith("ValueDrift"):
            method, value = config["method"], float(metric["value"])
            col_threshold = float(config["threshold"])
            # p-value tests drift when small; distance tests when large.
            drifted = value < col_threshold if "p_value" in method else value >= col_threshold
            columns[config["column"]] = {
                "method": method,
                "value": round(value, 6),
                "threshold": col_threshold,
                "drifted": drifted,
            }
    return {"drift_share": share, "columns": columns}


def run(since_hours: float | None) -> int:
    cfg = load_config()
    mon = cfg["monitoring"]
    features = cfg["features"]
    numeric = [c for c in features["numeric"] if c != "log_density"] + ["density"]
    categorical = list(features["categorical"])

    current_raw = load_requests(ROOT / mon["inference_log"], since_hours)
    report_dir = ROOT / mon["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "logger": "monitoring.drift",
        "requests": len(current_raw),
        "window_hours": since_hours,
    }
    if len(current_raw) < mon["min_rows"]:
        print(
            json.dumps(
                {
                    **base,
                    "level": "WARNING",
                    "event": "drift_insufficient_data",
                    "min_rows": mon["min_rows"],
                }
            )
        )
        return 2

    reference_raw = pd.read_parquet(ROOT / cfg["paths"]["reference_dir"] / "reference.parquet")
    reference_raw = reference_raw.rename(columns={features["target_text"]: CATEGORY})
    reference = comparable(reference_raw, numeric, categorical)
    current = comparable(current_raw, numeric, categorical)
    # Requests without a note have no note category: compare only those with one.
    has_note = current[CATEGORY].notna()

    definition = DataDefinition(
        numerical_columns=[*numeric, NOTE_LENGTH], categorical_columns=[*categorical, CATEGORY]
    )
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    snapshot.save_html(str(report_dir / "drift_report.html"))

    summary = drift_summary(snapshot.dict())
    alert = summary["drift_share"] >= mon["alert_drift_share"]
    drifted = sorted(c for c, r in summary["columns"].items() if r["drifted"])
    result = {
        **base,
        "level": "ERROR" if alert else "INFO",
        "event": "drift_alert" if alert else "drift_checked",
        "requests_with_note": int(has_note.sum()),
        "reference_rows": len(reference),
        "drift_share": round(summary["drift_share"], 4),
        "alert_drift_share": mon["alert_drift_share"],
        "drifted_columns": drifted,
    }
    (report_dir / "drift_report.json").write_text(
        json.dumps({**result, "columns": summary["columns"]}, indent=2) + "\n"
    )
    print(json.dumps(result))
    return 1 if alert else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--since-hours", type=float, default=None, help="Only requests from the last N hours"
    )
    sys.exit(run(parser.parse_args().since_hours))


if __name__ == "__main__":
    main()
