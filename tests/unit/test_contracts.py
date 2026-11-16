"""Data contracts accept good data and fail fast, with a structured alert, on bad data."""

from __future__ import annotations

import json
import logging

import pytest

from src.validation.contracts import ContractViolation, check_orphans, validate_claims


def test_valid_claims_pass(claims, cfg):
    report = validate_claims(claims, cfg)
    assert report.passed and report.failures == []


@pytest.mark.parametrize(
    ("column", "bad_value", "expected_check"),
    [
        ("driver_age", 15, "in_range"),  # below the legal minimum of 16
        ("claim_amount", -10.0, "greater_than_or_equal_to"),
        ("claim_category", "meteor_strike", "isin"),
        ("area", "Z", "isin"),
        ("claim_id", "BAD-ID", "str_matches"),
        ("exposure", 0.0, "greater_than"),
    ],
)
def test_out_of_contract_values_fail(claims, cfg, column, bad_value, expected_check):
    bad = claims.copy()
    bad.loc[bad.index[0], column] = bad_value
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)


def test_null_claim_id_fails(claims, cfg):
    bad = claims.copy()
    bad.loc[bad.index[0], "claim_id"] = None
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)


def test_duplicate_claim_id_fails(claims, cfg):
    bad = claims.copy()
    bad.loc[bad.index[1], "claim_id"] = bad.loc[bad.index[0], "claim_id"]
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)


def test_target_imbalance_fails(claims, cfg):
    bad = claims.copy()
    bad["is_high_cost"] = False  # 0% high-cost, below the 2% floor
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)


def test_missing_category_fails(claims, cfg):
    bad = claims.copy()
    bad.loc[bad["claim_category"] == "weather", "claim_category"] = "collision"
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)


def test_too_few_rows_fails(claims, cfg):
    with pytest.raises(ContractViolation):
        validate_claims(claims.head(10), cfg)


def test_failure_logs_one_structured_alert(claims, cfg, capsys):
    bad = claims.copy()
    bad.loc[bad.index[0], "driver_age"] = 15
    bad.loc[bad.index[1], "area"] = "Z"
    with pytest.raises(ContractViolation):
        validate_claims(bad, cfg)
    alerts = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    failed = [a for a in alerts if a["event"] == "contract_failed"]
    assert len(failed) == 1
    alert = failed[0]
    assert alert["level"] == logging.getLevelName(logging.ERROR)
    assert {f["column"] for f in alert["failures"]} == {"driver_age", "area"}
    assert all(f["failing_rows"] >= 1 and f["examples"] for f in alert["failures"])


def test_orphan_share_contract(cfg):
    assert check_orphans(1000, 995, cfg)["passed"]
    with pytest.raises(ContractViolation):
        check_orphans(1000, 900, cfg)  # 10% orphans, limit is 2%
