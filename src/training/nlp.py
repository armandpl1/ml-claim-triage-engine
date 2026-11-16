"""Fine-tune DistilBERT with LoRA to classify adjuster notes into claim categories.

    1. Sample notes from the training split (stratified by category).
    2. Add LoRA adapters to DistilBERT's attention (query and value). Only the
       adapters and the classification head are trained, a small share of the
       weights, so training takes minutes on a laptop CPU.
    3. Evaluate on the validation split: accuracy, macro F1, per-class F1.
    4. Merge the adapters into the base weights and export to ONNX (FP32),
       then quantize to INT8 with ONNX Runtime dynamic quantization.
    5. Check both ONNX models against PyTorch on the validation notes.
    6. Log everything to MLflow and register the ONNX model with the alias
       "challenger". INT8 is registered if it loses at most 0.5 points of
       macro F1, otherwise FP32.

If a run is interrupted after the ONNX export, --resume-from-onnx evaluates
and registers the exported models without retraining.

Usage:
    python -m src.training.nlp
    python -m src.training.nlp --resume-from-onnx
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from mlflow import MlflowClient
from onnxruntime.quantization import QuantType, quantize_dynamic
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.features.text import TOKENIZER_DIR, load_tokenizer, tokenize
from src.training.tracking import start_experiment, train_config

logger = get_logger("training.nlp")
OUT_DIR = ROOT / "artifacts" / "nlp"
MAX_INT8_F1_DROP = 0.005


def batches(n: int, size: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    order = rng.permutation(n) if rng is not None else np.arange(n)
    return [order[i : i + size] for i in range(0, n, size)]


def predict_torch(
    model: torch.nn.Module, enc: dict[str, np.ndarray], size: int = 128
) -> np.ndarray:
    model.eval()
    logits = []
    with torch.no_grad():
        for idx in batches(len(enc["input_ids"]), size):
            out = model(
                input_ids=torch.from_numpy(enc["input_ids"][idx]),
                attention_mask=torch.from_numpy(enc["attention_mask"][idx]),
            )
            logits.append(out.logits.numpy())
    return np.concatenate(logits)


def predict_onnx(
    session: ort.InferenceSession, enc: dict[str, np.ndarray], size: int = 128
) -> np.ndarray:
    """Logits in batches, so memory stays flat however many notes are scored."""
    outputs = []
    for idx in batches(len(enc["input_ids"]), size):
        feeds = {
            "input_ids": enc["input_ids"][idx].astype(np.int64),
            "attention_mask": enc["attention_mask"][idx].astype(np.int64),
        }
        outputs.append(session.run(["logits"], feeds)[0])
    return np.concatenate(outputs)


def scores(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
    }


def run() -> dict[str, Any]:
    cfg, data_cfg = train_config(), load_config()
    nlp, text_cfg = cfg["nlp"], data_cfg["text"]
    labels = list(data_cfg["notes"]["categories"])
    label_id = {name: i for i, name in enumerate(labels)}
    torch.manual_seed(cfg["seed"])
    # 3 threads leaves the laptop usable during the ~25 minutes of training
    torch.set_num_threads(3)

    processed = resolve(data_cfg["paths"]["processed_dir"])
    train_all = pd.read_parquet(processed / "train.parquet")
    validation = pd.read_parquet(processed / "validation.parquet")
    train, _ = train_test_split(
        train_all,
        train_size=nlp["train_sample"],
        stratify=train_all[nlp["target"]],
        random_state=cfg["seed"],
    )

    tokenizer = load_tokenizer(data_cfg)
    max_len = text_cfg["max_length"]
    enc_train = tokenize(tokenizer, train[data_cfg["features"]["text"]].tolist(), max_len)
    enc_val = tokenize(tokenizer, validation[data_cfg["features"]["text"]].tolist(), max_len)
    y_train = train[nlp["target"]].map(label_id).to_numpy()
    y_val = validation[nlp["target"]].map(label_id).to_numpy()

    base = AutoModelForSequenceClassification.from_pretrained(
        text_cfg["tokenizer"],
        revision=text_cfg["revision"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_id,
    )
    lora = nlp["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            modules_to_save=["pre_classifier", "classifier"],
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    start_experiment(cfg["mlflow"]["experiment_nlp"])
    with mlflow.start_run(run_name="distilbert-lora") as run:
        mlflow.log_params(
            {
                "base_model": text_cfg["tokenizer"],
                "revision": text_cfg["revision"],
                "train_rows": len(train),
                "max_length": max_len,
                "epochs": nlp["epochs"],
                "batch_size": nlp["batch_size"],
                "learning_rate": nlp["learning_rate"],
                "lora_r": lora["r"],
                "lora_alpha": lora["alpha"],
                "lora_dropout": lora["dropout"],
                "trainable_params": trainable,
                "total_params": total,
            }
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=nlp["learning_rate"]
        )
        steps = nlp["epochs"] * len(batches(len(y_train), nlp["batch_size"]))
        schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
        rng = np.random.default_rng(cfg["seed"])
        started, step = time.perf_counter(), 0
        for epoch in range(nlp["epochs"]):
            model.train()
            losses = []
            for idx in batches(len(y_train), nlp["batch_size"], rng):
                out = model(
                    input_ids=torch.from_numpy(enc_train["input_ids"][idx]),
                    attention_mask=torch.from_numpy(enc_train["attention_mask"][idx]),
                    labels=torch.from_numpy(y_train[idx]),
                )
                out.loss.backward()
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                losses.append(out.loss.item())
                step += 1
            epoch_metrics = scores(y_val, predict_torch(model, enc_val))
            mlflow.log_metrics(
                {
                    "train_loss": float(np.mean(losses)),
                    **{f"val_{k}": v for k, v in epoch_metrics.items()},
                },
                step=epoch,
            )
            log_event(
                logger,
                logging.INFO,
                "epoch_done",
                epoch=epoch,
                train_loss=round(float(np.mean(losses)), 4),
                **epoch_metrics,
            )
        train_minutes = (time.perf_counter() - started) / 60

        merged = model.merge_and_unload()
        torch_logits = predict_torch(merged, enc_val)
        torch_metrics = scores(y_val, torch_logits)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fp32_path, int8_path = OUT_DIR / "model.fp32.onnx", OUT_DIR / "model.int8.onnx"
        example = (
            torch.from_numpy(enc_val["input_ids"][:2]),
            torch.from_numpy(enc_val["attention_mask"][:2]),
        )
        torch.onnx.export(
            merged.eval(),
            example,
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=cfg["export"]["opset"],
            dynamo=False,
        )
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

        summary = evaluate_and_register(
            fp32_path,
            int8_path,
            enc_val,
            y_val,
            labels,
            cfg,
            text_cfg,
            reference_logits=torch_logits,
            extra={
                "torch_merged": torch_metrics,
                "train_minutes": round(train_minutes, 1),
                "trainable_params": trainable,
                "total_params": total,
                "run_id": run.info.run_id,
            },
        )
        mlflow.log_metric("train_minutes", train_minutes)
    return summary


def evaluate_and_register(
    fp32_path: Path,
    int8_path: Path,
    enc_val: dict[str, np.ndarray],
    y_val: np.ndarray,
    labels: list[str],
    cfg: dict[str, Any],
    text_cfg: dict[str, Any],
    reference_logits: np.ndarray | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate both ONNX variants, pick one, log to the active MLflow run, register it.

    reference_logits are the PyTorch model's outputs, for a parity check. When
    resuming from exported files there is no PyTorch model, so FP32 ONNX is
    the reference for INT8.
    """
    logits = {
        v: predict_onnx(ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]), enc_val)
        for v, p in (("fp32", fp32_path), ("int8", int8_path))
    }
    reference = reference_logits if reference_logits is not None else logits["fp32"]
    results: dict[str, dict[str, float]] = {}
    for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
        results[variant] = {
            **scores(y_val, logits[variant]),
            "agreement_with_reference": float(
                np.mean(logits[variant].argmax(1) == reference.argmax(1))
            ),
            "size_mb": round(path.stat().st_size / 1e6, 1),
        }
    f1_drop = results["fp32"]["macro_f1"] - results["int8"]["macro_f1"]
    chosen = "int8" if f1_drop <= MAX_INT8_F1_DROP else "fp32"
    report = classification_report(
        y_val, reference.argmax(1), target_names=labels, output_dict=True
    )
    summary: dict[str, Any] = {
        **extra,
        "onnx": results,
        "chosen_variant": chosen,
        "int8_macro_f1_drop": f1_drop,
        "parity_reference": "pytorch" if reference_logits is not None else "onnx_fp32",
        "per_class_f1": {c: round(report[c]["f1-score"], 4) for c in labels},
    }
    mlflow.log_metrics({f"onnx_{v}_{k}": m for v, r in results.items() for k, m in r.items()})
    mlflow.log_metric("int8_macro_f1_drop", f1_drop)
    mlflow.log_dict(summary, "summary.json")
    mlflow.log_artifacts(str(TOKENIZER_DIR), artifact_path="tokenizer")

    name = cfg["mlflow"]["model_nlp"]
    chosen_path = int8_path if chosen == "int8" else fp32_path
    info = mlflow.onnx.log_model(
        onnx.load(str(chosen_path)), name="model", registered_model_name=name
    )
    client, version = MlflowClient(), str(info.registered_model_version)
    for key, value in {
        "variant": chosen,
        "labels": json.dumps(labels),
        "tokenizer": text_cfg["tokenizer"],
        "tokenizer_revision": text_cfg["revision"],
        "max_length": str(text_cfg["max_length"]),
    }.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, "challenger", version)
    summary["model_version"] = version

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "labels.json").write_text(json.dumps(labels) + "\n")
    log_event(
        logger,
        logging.INFO,
        "nlp_registered",
        model_version=version,
        chosen_variant=chosen,
        fp32_macro_f1=round(results["fp32"]["macro_f1"], 4),
        int8_macro_f1=round(results["int8"]["macro_f1"], 4),
        int8_agreement=round(results["int8"]["agreement_with_reference"], 4),
        fp32_size_mb=results["fp32"]["size_mb"],
        int8_size_mb=results["int8"]["size_mb"],
    )
    return summary


def epoch_history(log_path: Path) -> list[dict[str, Any]]:
    """Per-epoch metrics from an earlier run's structured log."""
    if not log_path.exists():
        return []
    return [
        {k: event[k] for k in ("epoch", "train_loss", "accuracy", "macro_f1")}
        for event in map(
            json.loads,
            (
                line
                for line in log_path.read_text().splitlines()
                if line.startswith("{") and '"epoch_done"' in line
            ),
        )
    ]


def resume_from_onnx(log_path: Path) -> dict[str, Any]:
    """Register already-exported ONNX models without retraining.

    For a run that finished training and export but stopped before
    registration (e.g. the process was killed). Epoch metrics are copied from
    that run's log when available.
    """
    cfg, data_cfg = train_config(), load_config()
    text_cfg = data_cfg["text"]
    labels = list(data_cfg["notes"]["categories"])
    fp32_path, int8_path = OUT_DIR / "model.fp32.onnx", OUT_DIR / "model.int8.onnx"
    for path in (fp32_path, int8_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}: run a full training instead.")
        onnx.checker.check_model(str(path))  # refuse a half-written file

    validation = pd.read_parquet(resolve(data_cfg["paths"]["processed_dir"]) / "validation.parquet")
    enc_val = tokenize(
        load_tokenizer(data_cfg),
        validation[data_cfg["features"]["text"]].tolist(),
        text_cfg["max_length"],
    )
    y_val = validation[cfg["nlp"]["target"]].map({c: i for i, c in enumerate(labels)}).to_numpy()

    nlp, lora = cfg["nlp"], cfg["nlp"]["lora"]
    history = epoch_history(log_path)
    start_experiment(cfg["mlflow"]["experiment_nlp"])
    with mlflow.start_run(run_name="distilbert-lora (registered from export)") as run:
        mlflow.set_tag("resumed_from_onnx", "true")
        mlflow.log_params(
            {
                "base_model": text_cfg["tokenizer"],
                "revision": text_cfg["revision"],
                "train_rows": nlp["train_sample"],
                "max_length": text_cfg["max_length"],
                "epochs": nlp["epochs"],
                "batch_size": nlp["batch_size"],
                "learning_rate": nlp["learning_rate"],
                "lora_r": lora["r"],
                "lora_alpha": lora["alpha"],
                "lora_dropout": lora["dropout"],
            }
        )
        for event in history:
            mlflow.log_metrics(
                {
                    "train_loss": event["train_loss"],
                    "val_accuracy": event["accuracy"],
                    "val_macro_f1": event["macro_f1"],
                },
                step=event["epoch"],
            )
        return evaluate_and_register(
            fp32_path,
            int8_path,
            enc_val,
            y_val,
            labels,
            cfg,
            text_cfg,
            reference_logits=None,
            extra={"epochs": history, "resumed_from_onnx": True, "run_id": run.info.run_id},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resume-from-onnx",
        action="store_true",
        help="register artifacts/nlp/*.onnx from an interrupted run, no retraining",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "reports" / "nlp_train.log",
        help="the interrupted run's log, for its epoch metrics",
    )
    args = parser.parse_args()
    if args.resume_from_onnx:
        resume_from_onnx(args.log)
    else:
        run()


if __name__ == "__main__":
    main()
