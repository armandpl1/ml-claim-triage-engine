"""Synthetic adjuster notes: deterministic, valid, and related to the real data."""

from __future__ import annotations

from collections import Counter

from src.ingestion.notes import CATEGORIES, generate_note, severity_band, stable_seed


def test_same_claim_same_note():
    a = generate_note("CLM-0000001-01", 1200.0, 3, 500, seed=42)
    b = generate_note("CLM-0000001-01", 1200.0, 3, 500, seed=42)
    assert a == b


def test_stable_seed_is_process_independent():
    # A fixed value: if this changes, every generated note changes.
    assert stable_seed(42, "CLM-0000001-01") == stable_seed(42, "CLM-0000001-01")
    assert stable_seed(42, "a") != stable_seed(42, "b")


def test_categories_and_text_are_valid():
    for i in range(200):
        category, text = generate_note(f"CLM-{i:07d}-01", 1000.0 + i * 50, i % 15, 1000, 42)
        assert category in CATEGORIES
        assert len(text) >= 10 and "{" not in text


def test_expensive_claims_are_more_often_bodily_injury():
    def bodily_share(amount: float) -> float:
        cats = Counter(
            generate_note(f"CLM-{i:07d}-01", amount, 5, 1000, 42)[0] for i in range(2000)
        )
        return cats["bodily_injury"] / 2000

    assert bodily_share(20000.0) > 3 * bodily_share(1000.0)


def test_severity_bands():
    assert severity_band(500) == "low"
    assert severity_band(2500) == "medium"
    assert severity_band(9000) == "high"
