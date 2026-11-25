"""Model card: tables are aligned Markdown, findings follow the numbers
(over-routed groups, a dominant reason, low power), and the rendered card
has every section."""

from __future__ import annotations

import pandas as pd

from src.evaluation import model_card


def group(rows, pos, recall, interval, flagged, precision, calibration, judged=True):
    return {
        "rows": rows,
        "positives": pos,
        "recall": recall,
        "recall_interval": interval,
        "flagged_rate": flagged,
        "precision": precision,
        "calibration": calibration,
        "judged": judged,
    }


AUDIT = {
    "confidence": 0.95,
    "alerts": [],
    "overall": group(1000, 50, 0.2, [0.1, 0.3], 0.08, 0.12, 1.0),
    "groups": {
        "driver_age": {
            "under 25": group(100, 10, 0.1, [0.02, 0.4], 0.16, 0.02, 1.5),
            "25+": group(900, 40, 0.2, [0.1, 0.3], 0.07, 0.13, 0.95),
        }
    },
}


def test_tables_have_aligned_pipes():
    lines = model_card.table(["A", "Long header"], [["x", 1], ["longer value", 22]]).splitlines()
    assert len({len(line) for line in lines}) == 1
    assert [i for i, ch in enumerate(lines[0]) if ch == "|"] == [
        i for i, ch in enumerate(lines[2]) if ch == "|"
    ]


def test_findings_follow_the_numbers():
    top = pd.Series({"exposure": 30, "region": 5})
    found = model_card.findings(AUDIT, top, routed=40)
    assert found[0].startswith("`exposure` is the top reason for 30 of 40")
    assert any(f.startswith("driver_age under 25: routed at 16.0%") for f in found)
    assert any("overstates this group's risk" in f for f in found)  # calibration 1.5
    assert not any("25+:" in f for f in found)  # routed below the 1.5x line
    assert "2.0% to 40.0%" in found[-1]  # the widest judged interval


def test_the_rendered_card_has_every_section():
    info = {
        "model": {
            "name": "triage",
            "version": "3",
            "threshold": 0.08,
            "run_id": "abc",
            "params": {"n_estimators": 100},
            "monotone": {},
        },
        "features": ["exposure", "log_density"],
        "high_cost_threshold": 5000,
        "performance": {
            "rows": 1000,
            "positive_rate": 0.05,
            "pr_auc": 0.08,
            "lift": 1.6,
            "roc_auc": 0.6,
        },
        "parity": 1e-7,
        "top_k": 3,
        "importance": pd.Series({"exposure": 0.3, "log_density": 0.1}),
        "top_reasons": pd.Series({"exposure": 30, "density": 2}),
        "audit": AUDIT,
        "findings": ["a finding"],
    }
    card = model_card.render(info)
    for heading in (
        "Intended use",
        "Model",
        "Data",
        "Performance",
        "Explainability",
        "Fairness",
        "Limitations",
        "Monitoring",
    ):
        assert f"## {heading}" in card
    assert card.startswith("# Model card: triage v3")
    assert "| density " in card  # log_density shown as density
    assert "- a finding" in card
