"""A cheap baseline for the notes classifier: TF-IDF words + logistic regression.

The fine-tuned DistilBERT has to beat this to earn its cost (67 MB INT8 model,
~31 ms per note vs well under 1 ms). Trained on the train split, scored on
validation and test with macro F1, the same metric as the gate.

Usage:
    python -m src.training.text_baseline
"""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event

logger = get_logger("training.text_baseline")


def make_baseline() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )


def run() -> dict[str, float]:
    cfg = load_config()
    text, target = cfg["features"]["text"], cfg["features"]["target_text"]
    processed = resolve(cfg["paths"]["processed_dir"])
    splits = {
        s: pd.read_parquet(processed / f"{s}.parquet") for s in ("train", "validation", "test")
    }

    model = make_baseline().fit(splits["train"][text], splits["train"][target])
    scores = {
        f"{name}_macro_f1": round(
            float(f1_score(df[target], model.predict(df[text]), average="macro")), 4
        )
        for name, df in splits.items()
        if name != "train"
    }
    out = ROOT / "artifacts" / "nlp"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    log_event(logger, logging.INFO, "text_baseline_done", **scores)
    return scores


if __name__ == "__main__":
    run()
