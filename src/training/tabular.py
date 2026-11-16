"""Train the tabular triage model: is this claim high-cost?

    1. Tune LightGBM with Optuna. Each trial is scored by grouped, stratified
       cross-validation on the training split only (folds grouped by policy;
       the preprocessor is refit inside every fold, so no fold leaks into
       another). Every trial is logged to MLflow as a nested run.
    2. Refit the best parameters on the full training split and evaluate on
       the validation split. The test split stays untouched for the Step 5
       gate.
    3. Pick the decision threshold that maximizes F1 on validation.
    4. Log metrics, the feature importance and tuning-history plots to MLflow.
    5. Export the whole pipeline to ONNX, check it predicts the same as
       scikit-learn, and register it in the MLflow Model Registry with the
       alias "challenger".

Usage:
    python -m src.training.tabular
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.features.transform import build_preprocessor, feature_columns
from src.training.onnx_export import export_tabular, onnx_inputs
from src.training.tracking import start_experiment, train_config

logger = get_logger("training.tabular")
OUT_DIR = ROOT / "artifacts" / "tabular"


def load_splits(data_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    processed = resolve(data_cfg["paths"]["processed_dir"])
    return {s: pd.read_parquet(processed / f"{s}.parquet") for s in ("train", "validation")}


def monotone_constraints(data_cfg: dict[str, Any], monotone: dict[str, int]) -> list[int]:
    """One direction per model input, in the preprocessor's output order
    (numeric features, then categorical): +1 increasing, -1 decreasing, 0 free."""
    return [int(monotone.get(name, 0)) for name in feature_columns(data_cfg)]


def make_pipeline(
    data_cfg: dict[str, Any],
    params: dict[str, Any],
    seed: int,
    monotone: dict[str, int] | None = None,
) -> Pipeline:
    model = LGBMClassifier(
        **params,
        monotone_constraints=monotone_constraints(data_cfg, monotone or {}),
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    return Pipeline([("preprocess", build_preprocessor(data_cfg)), ("model", model)])


def suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec["low"], int) and isinstance(spec["high"], int):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
    params["subsample_freq"] = 1  # makes `subsample` take effect
    return params


def cross_validate(
    train: pd.DataFrame, data_cfg: dict[str, Any], params: dict[str, Any], cfg: dict[str, Any]
) -> list[float]:
    target = cfg["tabular"]["target"]
    folds = StratifiedGroupKFold(
        n_splits=cfg["tabular"]["cv_folds"], shuffle=True, random_state=cfg["seed"]
    )
    scores = []
    for fit_idx, eval_idx in folds.split(train, train[target], train[data_cfg["split_group"]]):
        fit, held = train.iloc[fit_idx], train.iloc[eval_idx]
        pipe = make_pipeline(data_cfg, params, cfg["seed"], cfg["tabular"]["monotone"])
        pipe.fit(fit, fit[target])
        scores.append(average_precision_score(held[target], pipe.predict_proba(held)[:, 1]))
    return scores


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted)),
        "threshold": threshold,
        "positive_rate": float(np.mean(y_true)),
    }


def plot_importance(pipe: Pipeline, path: Path) -> None:
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    gains = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gains)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(names)[order], gains[order])
    ax.set_title("LightGBM feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(study: optuna.Study, path: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, "o", alpha=0.6, label="trial")
    ax.plot(np.maximum.accumulate(values), label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV PR-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run() -> dict[str, Any]:
    cfg, data_cfg = train_config(), load_config()
    splits = load_splits(data_cfg)
    train, validation = splits["train"], splits["validation"]
    target = cfg["tabular"]["target"]
    start_experiment(cfg["mlflow"]["experiment_tabular"])
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="lightgbm-optuna") as parent:
        mlflow.log_params(
            {
                "n_trials": cfg["tabular"]["n_trials"],
                "cv_folds": cfg["tabular"]["cv_folds"],
                "features": ",".join(feature_columns(data_cfg)),
                "monotone": json.dumps(cfg["tabular"]["monotone"]),
                "train_rows": len(train),
            }
        )

        def objective(trial: optuna.Trial) -> float:
            params = suggest(trial, cfg["tabular"]["search_space"])
            scores = cross_validate(train, data_cfg, params, cfg)
            with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {"cv_pr_auc": float(np.mean(scores)), "cv_pr_auc_std": float(np.std(scores))}
                )
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"])
        )
        study.optimize(objective, n_trials=cfg["tabular"]["n_trials"])
        best = {**study.best_params, "subsample_freq": 1}
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_pr_auc", study.best_value)

        pipe = make_pipeline(data_cfg, best, cfg["seed"], cfg["tabular"]["monotone"])
        pipe.fit(train, train[target])
        y_val = validation[target].to_numpy()
        scores = pipe.predict_proba(validation)[:, 1]
        threshold = best_f1_threshold(y_val, scores)
        metrics = evaluate(y_val, scores, threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        onnx_model = export_tabular(pipe, data_cfg, cfg["export"]["opset"])
        session = ort.InferenceSession(onnx_model.SerializeToString())
        onnx_scores = session.run(None, onnx_inputs(validation, data_cfg))[1][:, 1]
        parity = {
            "onnx_max_abs_diff": float(np.max(np.abs(onnx_scores - scores))),
            "onnx_decision_agreement": float(
                np.mean((onnx_scores >= threshold) == (scores >= threshold))
            ),
        }
        mlflow.log_metrics(parity)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        onnx.save(onnx_model, OUT_DIR / "model.onnx")
        summary = {
            "metrics": metrics,
            "parity": parity,
            "best_params": best,
            "monotone": cfg["tabular"]["monotone"],
            "best_cv_pr_auc": study.best_value,
            "run_id": parent.info.run_id,
        }
        (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

        with tempfile.TemporaryDirectory() as tmp:
            plot_importance(pipe, Path(tmp) / "feature_importance.png")
            plot_history(study, Path(tmp) / "optuna_history.png")
            mlflow.log_artifacts(tmp, artifact_path="plots")
        mlflow.log_dict(summary, "summary.json")

        info = mlflow.onnx.log_model(
            onnx_model, name="model", registered_model_name=cfg["mlflow"]["model_tabular"]
        )
        client = MlflowClient()
        version = str(info.registered_model_version)
        client.set_model_version_tag(
            cfg["mlflow"]["model_tabular"], version, "decision_threshold", f"{threshold:.6f}"
        )
        client.set_registered_model_alias(cfg["mlflow"]["model_tabular"], "challenger", version)

    log_event(
        logger,
        logging.INFO,
        "tabular_trained",
        model_version=version,
        best_cv_pr_auc=round(study.best_value, 4),
        **{f"val_{k}": round(v, 4) for k, v in metrics.items()},
        **parity,
    )
    return summary


if __name__ == "__main__":
    run()
