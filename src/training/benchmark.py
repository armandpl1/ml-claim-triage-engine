"""Benchmark single-request inference latency of the exported ONNX models.

The API answers one claim at a time, so latency is measured per request:
each model is called `benchmark_runs` times with one validation row, after
a short warm-up, with the API's thread count (export.inference_threads).
Reported: p50, p95, p99 and mean, in milliseconds.

    tabular         preprocessing + LightGBM in one ONNX graph
    notes FP32      DistilBERT classifier, 32-bit weights
    notes INT8      the same model, dynamically quantized to 8-bit weights
    notes ... +tok  the same, including tokenization (what a request pays)

Results go to artifacts/benchmark.json and to MLflow, with a pass/fail
against the p99 latency budget in train_config.yaml.

Usage:
    python -m src.training.benchmark
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.features.text import load_tokenizer, tokenize
from src.training.onnx_export import onnx_inputs
from src.training.tracking import start_experiment, train_config

logger = get_logger("training.benchmark")
ARTIFACTS = ROOT / "artifacts"


def session(path: str, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads  # same as the API (export.inference_threads)
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def time_calls(fn: Callable[[int], Any], runs: int, warmup: int = 20) -> dict[str, float]:
    for i in range(warmup):
        fn(i)
    times = []
    for i in range(runs):
        started = time.perf_counter()
        fn(i)
        times.append((time.perf_counter() - started) * 1000)
    arr = np.array(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def run() -> dict[str, Any]:
    cfg, data_cfg = train_config(), load_config()
    runs, budget = cfg["export"]["benchmark_runs"], cfg["export"]["latency_budget_p99_ms"]
    threads = cfg["export"]["inference_threads"]
    validation = pd.read_parquet(resolve(data_cfg["paths"]["processed_dir"]) / "validation.parquet")
    rows = validation.sample(n=runs, random_state=cfg["seed"]).reset_index(drop=True)

    results: dict[str, dict[str, float]] = {}
    tab = session(str(ARTIFACTS / "tabular" / "model.onnx"), threads)
    feeds = [onnx_inputs(rows.iloc[[i]], data_cfg) for i in range(runs)]
    results["tabular"] = time_calls(lambda i: tab.run(None, feeds[i % runs]), runs)

    tokenizer = load_tokenizer(data_cfg)
    max_len = data_cfg["text"]["max_length"]
    notes = rows[data_cfg["features"]["text"]].tolist()
    encoded = [tokenize(tokenizer, [n], max_len) for n in notes]

    def note_feeds(enc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    for variant in ("fp32", "int8"):
        sess = session(str(ARTIFACTS / "nlp" / f"model.{variant}.onnx"), threads)

        def pre_tokenized(i: int, s: ort.InferenceSession = sess) -> Any:
            return s.run(None, note_feeds(encoded[i % runs]))

        def with_tokenization(i: int, s: ort.InferenceSession = sess) -> Any:
            return s.run(None, note_feeds(tokenize(tokenizer, [notes[i % runs]], max_len)))

        results[f"notes_{variant}"] = time_calls(pre_tokenized, runs)
        results[f"notes_{variant}_with_tokenization"] = time_calls(with_tokenization, runs)

    speedup = results["notes_fp32"]["p50_ms"] / results["notes_int8"]["p50_ms"]
    summary = {
        "runs_per_model": runs,
        "threads": threads,
        "latency": {k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()},
        "int8_speedup_p50": round(speedup, 2),
        "budget_p99_ms": budget,
        "within_budget": {k: r["p99_ms"] <= budget for k, r in results.items()},
    }
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")

    start_experiment(cfg["mlflow"]["experiment_nlp"])
    with mlflow.start_run(run_name="onnx-latency-benchmark"):
        mlflow.log_params({"runs_per_model": runs, "threads": threads, "budget_p99_ms": budget})
        mlflow.log_metrics({f"{k}_{m}": v for k, r in results.items() for m, v in r.items()})
        mlflow.log_metric("int8_speedup_p50", speedup)
        mlflow.log_dict(summary, "benchmark.json")

    log_event(
        logger,
        logging.INFO,
        "benchmark_done",
        **{k: round(r["p99_ms"], 2) for k, r in results.items()},
        int8_speedup_p50=round(speedup, 2),
        all_within_budget=all(summary["within_budget"].values()),
    )
    return summary


if __name__ == "__main__":
    run()
