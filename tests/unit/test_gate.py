"""The champion/challenger gate's rules, tested as a pure function."""

from __future__ import annotations

import pytest

from src.evaluation.gate import decide
from src.evaluation.scoring import Scorecard
from src.training.tracking import train_config

BUDGET_MS = 50.0


@pytest.fixture
def gate_cfg():
    return train_config()["gate"]


def card(primary, slices=None, p99=1.0, baseline=0.05):
    return Scorecard(
        primary=primary,
        slices={k: {"metric": v} for k, v in (slices or {}).items()},
        latency={"p99_ms": p99},
        baseline=baseline,
    )


def checks(decision):
    return {c.name: c.passed for c in decision.checks}


def test_clear_improvement_is_promoted(gate_cfg):
    d = decide(
        card(0.10, {"young": 0.12}), card(0.11, {"young": 0.12}), "tabular", gate_cfg, BUDGET_MS
    )
    assert d.promote and all(checks(d).values())


def test_improvement_below_half_a_percent_is_rejected(gate_cfg):
    d = decide(card(0.100), card(0.1004), "tabular", gate_cfg, BUDGET_MS)  # +0.4%
    assert not d.promote and not checks(d)["primary_metric_improves"]


def test_equal_model_is_rejected(gate_cfg):
    assert not decide(card(0.2), card(0.2), "notes", gate_cfg, BUDGET_MS).promote


def test_tabular_slice_losing_more_than_10_percent_is_rejected(gate_cfg):
    champion = card(0.10, {"young_drivers": 0.20, "old_vehicles": 0.10})
    challenger = card(0.12, {"young_drivers": 0.17, "old_vehicles": 0.11})  # young: -15%
    d = decide(champion, challenger, "tabular", gate_cfg, BUDGET_MS)
    assert not d.promote and not checks(d)["no_slice_degrades"]
    assert "young_drivers" in next(c.detail for c in d.checks if c.name == "no_slice_degrades")


def test_tabular_slice_losing_less_than_10_percent_is_allowed(gate_cfg):
    d = decide(
        card(0.10, {"young": 0.20}), card(0.12, {"young": 0.185}), "tabular", gate_cfg, BUDGET_MS
    )
    assert d.promote


def test_notes_category_losing_more_than_002_f1_is_rejected(gate_cfg):
    champion = card(0.90, {"category_bodily_injury": 0.80})
    challenger = card(0.93, {"category_bodily_injury": 0.77})
    assert not decide(champion, challenger, "notes", gate_cfg, BUDGET_MS).promote


def test_slow_challenger_is_rejected(gate_cfg):
    d = decide(card(0.10), card(0.12, p99=75.0), "tabular", gate_cfg, BUDGET_MS)
    assert not d.promote and not checks(d)["latency_within_budget"]


def test_slice_missing_from_challenger_is_not_counted(gate_cfg):
    d = decide(card(0.10, {"rare": 0.5}), card(0.12, {}), "tabular", gate_cfg, BUDGET_MS)
    assert d.promote


@pytest.mark.parametrize(("primary", "promote"), [(0.08, True), (0.06, False)])
def test_first_tabular_model_must_beat_naive_baseline(gate_cfg, primary, promote):
    # positive rate 0.05 -> bar is 1.5 x 0.05 = 0.075
    d = decide(None, card(primary, baseline=0.05), "tabular", gate_cfg, BUDGET_MS)
    assert d.promote is promote and "beats_naive_baseline" in checks(d)


def test_first_notes_model_needs_minimum_macro_f1(gate_cfg):
    assert decide(None, card(0.85), "notes", gate_cfg, BUDGET_MS).promote
    assert not decide(None, card(0.40), "notes", gate_cfg, BUDGET_MS).promote
