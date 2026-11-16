"""Tests for src/evaluation/scoring.py helpers."""

from __future__ import annotations

import pytest

from src.evaluation.scoring import has_shorthand
from src.ingestion.notes import generate_note


@pytest.mark.parametrize(
    "note",
    [
        "Ins'd veh rear-ended at a roundabout.",
        "Severe dmg to the bonnet.",
        "Dmg to the rear bumper.",  # capitalised at the start of a sentence
        "Theft reprted to the police.",
        "Polcie attended the scene.",
    ],
)
def test_shorthand_is_detected(note):
    assert has_shorthand(note)


@pytest.mark.parametrize(
    "note",
    [
        "Insured vehicle rear-ended at a roundabout.",  # "veh" inside "vehicle"
        "Vehicles parked overnight; damage reported to police.",
        "Windscreen cracked.",
    ],
)
def test_full_words_are_not_shorthand(note):
    assert not has_shorthand(note)


def test_shorthand_detected_exactly_when_generator_adds_a_typo():
    """Same claims with and without the typo step: a note counts as shorthand
    if and only if the typo step actually changed it. (The step picks a word
    at random, so it changes the note only if that word is in it: about 3% of
    notes at typo_rate 0.08.)"""
    claims = [f"CLM-{i}-1" for i in range(3000)]
    clean = [generate_note(c, 1000.0, 5, 500, seed=42, typo_rate=0.0)[1] for c in claims]
    typed = [generate_note(c, 1000.0, 5, 500, seed=42, typo_rate=0.08)[1] for c in claims]
    changed = [a != b for a, b in zip(clean, typed, strict=True)]
    assert any(changed)
    assert [has_shorthand(n) for n in typed] == changed
