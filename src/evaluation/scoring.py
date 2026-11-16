"""Score an ONNX model on the holdout test split, the way the gate needs it.

Each scorer returns a Scorecard:
    primary   one number the gate compares (tabular PR-AUC, notes macro F1)
    slices    the same metric on slices that matter (tabular: young drivers,
              old vehicles, ...; notes: each category, and notes with typos)
    latency   single-request p50/p99 in ms, including preprocessing a request
              actually pays for (tokenization for notes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from src.features.text import tokenize
from src.ingestion.notes import TYPOS
from src.training.benchmark import time_calls
from src.training.onnx_export import onnx_inputs


@dataclass
class Scorecard:
    primary: float
    slices: dict[str, dict[str, float]] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    baseline: float = 0.0  # what a naive model would score (for the first champion)


def serving_session(model_bytes: bytes, threads: int) -> ort.InferenceSession:
    """An ONNX Runtime session configured like the API's (export.inference_threads)."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])


def score_tabular(
    model_bytes: bytes,
    test: pd.DataFrame,
    data_cfg: dict[str, Any],
    gate_cfg: dict[str, Any],
    threads: int = 2,
) -> Scorecard:
    session = serving_session(model_bytes, threads)
    target = data_cfg["features"]["target_classification"]
    y = test[target].to_numpy()
    probabilities = session.run(None, onnx_inputs(test, data_cfg))[1][:, 1]

    slices = {}
    for name, query in gate_cfg["tabular_slices"].items():
        mask = np.asarray(test.eval(query), dtype=bool)
        positives = int(y[mask].sum())
        if positives >= gate_cfg["min_slice_positives"]:
            slices[name] = {
                "metric": float(average_precision_score(y[mask], probabilities[mask])),
                "rows": int(mask.sum()),
                "positives": positives,
            }

    feeds = [onnx_inputs(test.iloc[[i]], data_cfg) for i in range(gate_cfg["latency_runs"])]
    latency = time_calls(lambda i: session.run(None, feeds[i]), gate_cfg["latency_runs"])
    return Scorecard(
        primary=float(average_precision_score(y, probabilities)),
        slices=slices,
        latency=latency,
        baseline=float(y.mean()),
    )


# Mirrors the generator (src/ingestion/notes.py), which swaps a word for its
# shorthand anywhere it appears, any case: "damaged" becomes "dmgd". So match
# the shorthand at the start of a word, except where it is just the start of
# the full word ("veh" in "vehicle").
def _shorthand_pattern(word: str, typo: str) -> str:
    rest = rf"(?!{re.escape(word[len(typo) :])})" if word.startswith(typo) else ""
    return rf"(?<!\w){re.escape(typo)}{rest}"


SHORTHAND = re.compile("|".join(_shorthand_pattern(w, t) for w, t in TYPOS), re.IGNORECASE)


def has_shorthand(note: str) -> bool:
    """The note contains an adjuster typo or abbreviation (an edge case)."""
    return SHORTHAND.search(note) is not None


def score_notes(
    model_bytes: bytes,
    test: pd.DataFrame,
    data_cfg: dict[str, Any],
    gate_cfg: dict[str, Any],
    tokenizer: Any,
    labels: list[str],
    threads: int = 2,
) -> Scorecard:
    session = serving_session(model_bytes, threads)
    max_len = data_cfg["text"]["max_length"]
    notes = test[data_cfg["features"]["text"]].tolist()
    y = (
        test[data_cfg["features"]["target_text"]]
        .map({c: i for i, c in enumerate(labels)})
        .to_numpy()
    )

    def predict(texts: list[str]) -> np.ndarray:
        enc = tokenize(tokenizer, texts, max_len)
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        return session.run(["logits"], feeds)[0]

    predicted = np.concatenate([predict(notes[i : i + 256]) for i in range(0, len(notes), 256)])
    predicted = predicted.argmax(axis=1)
    per_class = f1_score(y, predicted, average=None, labels=list(range(len(labels))))
    slices = {
        f"category_{c}": {"metric": float(per_class[i]), "rows": int((y == i).sum())}
        for i, c in enumerate(labels)
    }
    shorthand = np.array([has_shorthand(n) for n in notes])
    if shorthand.sum():
        slices["notes_with_typos"] = {
            "metric": float(f1_score(y[shorthand], predicted[shorthand], average="macro")),
            "rows": int(shorthand.sum()),
        }

    runs = gate_cfg["latency_runs"]
    latency = time_calls(lambda i: predict([notes[i % len(notes)]]), runs)
    majority = np.bincount(y).argmax()
    return Scorecard(
        primary=float(f1_score(y, predicted, average="macro")),
        slices=slices,
        latency=latency,
        baseline=float(f1_score(y, np.full_like(y, majority), average="macro")),
    )
