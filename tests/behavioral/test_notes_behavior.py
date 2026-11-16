"""Behavioral tests for the note classifier (the exported ONNX model).

Invariance:
    a minor typo, or a change of letter case, should not change the category
Directional expectations:
    adding a hospital mention raises the bodily-injury probability
    adding a cracked-windscreen mention raises the glass probability

Thresholds are statistical (share of notes that behave), since a single
borderline note can legitimately flip. Skipped until the model is trained.
"""

from __future__ import annotations

import json
import random

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.common.config import ROOT, resolve
from src.serving.tokenizer import NoteTokenizer

pytestmark = pytest.mark.behavioral
NLP = ROOT / "artifacts" / "nlp"


@pytest.fixture(scope="module")
def classifier():
    metrics_file = NLP / "metrics.json"
    if not metrics_file.exists():
        pytest.skip("note classifier not trained yet (make train-nlp)")
    variant = json.loads(metrics_file.read_text())["chosen_variant"]
    session = ort.InferenceSession(str(NLP / f"model.{variant}.onnx"))
    tokenizer = NoteTokenizer(ROOT / "artifacts" / "tokenizer" / "tokenizer.json", 48)
    labels = json.loads((NLP / "labels.json").read_text())

    def probabilities(notes: list[str]) -> np.ndarray:
        logits = session.run(["logits"], tokenizer(notes))[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    return probabilities, labels


@pytest.fixture(scope="module")
def notes(cfg_module):
    test = pd.read_parquet(resolve(cfg_module["paths"]["processed_dir"]) / "test.parquet")
    return test["adjuster_note"].sample(300, random_state=0).tolist()


@pytest.fixture(scope="module")
def cfg_module():
    from src.common.config import load_config

    return load_config()


def swap_typo(note: str, rng: random.Random) -> str:
    """Swap two neighboring letters inside one longer word."""
    words = note.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return note
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def test_minor_typo_rarely_changes_the_category(classifier, notes):
    predict, _ = classifier
    rng = random.Random(0)
    before = predict(notes).argmax(axis=1)
    after = predict([swap_typo(n, rng) for n in notes]).argmax(axis=1)
    assert np.mean(before == after) >= 0.95


def test_letter_case_never_changes_the_category(classifier, notes):
    predict, _ = classifier
    before = predict(notes).argmax(axis=1)
    assert np.array_equal(before, predict([n.upper() for n in notes]).argmax(axis=1))


@pytest.mark.parametrize(
    ("addition", "category"),
    [
        (" Third party taken to hospital with a broken arm.", "bodily_injury"),
        (" Windscreen cracked, replacement requested.", "glass"),
    ],
)
def test_adding_a_strong_cue_raises_that_category(classifier, notes, addition, category):
    predict, labels = classifier
    k = labels.index(category)
    before = predict(notes)[:, k]
    after = predict([n + addition for n in notes])[:, k]
    assert np.mean(after >= before) >= 0.90
