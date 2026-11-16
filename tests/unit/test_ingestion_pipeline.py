"""The full ingestion run: raw lands untouched, contracts gate staging, failures halt.

The OpenML download is replaced with fixture data (no network), and each test
uses its own config file and DuckDB warehouse in a temporary folder.
"""

from __future__ import annotations

import json

import duckdb
import pytest
import yaml

from src.common.config import CONFIG_OVERRIDE_ENV
from src.ingestion import fetch_data
from src.ingestion.fetch_data import AMOUNT_COLUMNS, POLICY_COLUMNS
from src.validation.contracts import ContractViolation


@pytest.fixture
def source_frames(raw_policies, raw_amounts):
    """The fixture tables in the source's own format: original names, quoted fuel."""
    policies = raw_policies.rename(columns={v: k for k, v in POLICY_COLUMNS.items()})
    policies["IDpol"] = policies["IDpol"].astype("float64")
    policies["VehGas"] = "'" + policies["VehGas"] + "'"
    amounts = raw_amounts.rename(columns={v: k for k, v in AMOUNT_COLUMNS.items()})
    return {"policies": policies, "claim_amounts": amounts}


@pytest.fixture
def pipeline(tmp_path, monkeypatch, cfg, source_frames):
    """Run ingestion against a temporary warehouse, with an editable config."""

    def run_with(**contract_overrides):
        config = json.loads(json.dumps(cfg))  # deep copy
        config["paths"]["warehouse"] = str(tmp_path / "warehouse.duckdb")
        config["contracts"].update(contract_overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setenv(CONFIG_OVERRIDE_ENV, str(path))
        monkeypatch.setattr(
            fetch_data, "download", lambda _id, name, force=False: source_frames[name].copy()
        )
        return fetch_data.run()

    def query(sql):
        with duckdb.connect(str(tmp_path / "warehouse.duckdb"), read_only=True) as con:
            return con.sql(sql).fetchall()

    return run_with, query


def test_passing_run_writes_raw_and_staging(pipeline):
    run_with, query = pipeline
    stats = run_with()
    tables = {
        row[0] for row in query("SELECT schema_name || '.' || table_name FROM duckdb_tables()")
    }
    assert tables == {"raw.policies", "raw.claim_amounts", "staging.policies", "staging.claims"}
    assert query("SELECT count(*) FROM staging.claims")[0][0] == stats.claims


def test_raw_keeps_the_source_untouched(pipeline):
    run_with, query = pipeline
    run_with()
    raw_columns = {
        row[0]
        for row in query(
            "SELECT column_name FROM duckdb_columns() "
            "WHERE table_name = 'policies' AND schema_name = 'raw'"
        )
    }
    assert {"IDpol", "VehGas", "DrivAge"} <= raw_columns
    assert query("SELECT DISTINCT VehGas FROM raw.policies ORDER BY 1") == [
        ("'Diesel'",),
        ("'Regular'",),
    ]
    assert query("SELECT DISTINCT fuel_type FROM staging.policies ORDER BY 1") == [
        ("Diesel",),
        ("Regular",),
    ]


def test_failed_contract_halts_and_leaves_staging_untouched(pipeline, capsys):
    run_with, query = pipeline
    run_with()
    before = query("SELECT * FROM staging.claims ORDER BY claim_id")

    # A contract the fixture data breaks: every driver must be at least 60.
    with pytest.raises(ContractViolation):
        run_with(driver_age={"min": 60, "max": 100})

    assert query("SELECT * FROM staging.claims ORDER BY claim_id") == before
    alerts = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    failed = [a for a in alerts if a["event"] == "contract_failed"]
    assert failed and failed[-1]["failures"][0]["column"] == "driver_age"


def test_halt_exits_with_status_1(pipeline, monkeypatch, capsys):
    run_with, _ = pipeline
    with pytest.raises(ContractViolation):
        run_with(min_rows=10_000_000)  # prepares the strict config and the stubbed download
    monkeypatch.setattr("sys.argv", ["fetch_data"])
    with pytest.raises(SystemExit) as exit_info:
        fetch_data.main()
    assert exit_info.value.code == 1
    events = [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()]
    assert events[-1] == "pipeline_halted"


def test_rerun_is_idempotent(pipeline):
    run_with, query = pipeline
    run_with()
    first = query("SELECT * FROM staging.claims ORDER BY claim_id")
    run_with()
    assert query("SELECT * FROM staging.claims ORDER BY claim_id") == first
