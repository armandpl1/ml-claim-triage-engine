"""Champion vs challenger gate: only a better model reaches production.

For a registered model (tabular or notes), this:
    1. Loads the @champion and @challenger ONNX models from the MLflow Model
       Registry.
    2. Scores both on the untouched holdout test split (src/evaluation/scoring.py).
    3. Checks three conditions (thresholds in configs/train_config.yaml, gate:):
         primary metric   challenger beats champion by at least 0.5% (relative)
         slices           no slice degrades severely (tabular: >10% relative
                          PR-AUC loss; notes: >0.02 F1 on any category or on
                          notes with typos)
         latency          challenger single-request p99 under 50 ms
    4. Promotes the challenger to @champion only if all three pass, keeping the
       old champion as @previous_champion for rollback. Otherwise the champion
       stays and the challenger version is tagged as rejected.

With no champion yet (the first run), the challenger must instead beat a naive
baseline (tabular: 1.5x the positive rate; notes: macro F1 >= 0.5) and meet
the latency budget.

Every decision is logged as an MLflow run in the "model-gate" experiment and
written to artifacts/gate/<model>.json. Exit status: 0 promoted, 1 rejected.

Usage:
    python -m src.evaluation.gate --model tabular
    python -m src.evaluation.gate --model notes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.common.config import ROOT, load_config, resolve
from src.common.logging import get_logger, log_event
from src.evaluation.scoring import Scorecard, score_notes, score_tabular
from src.features.text import load_tokenizer
from src.training.tracking import start_experiment, train_config

logger = get_logger("evaluation.gate")
OUT_DIR = ROOT / "artifacts" / "gate"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    promote: bool
    checks: list[Check]


def decide(
    champion: Scorecard | None,
    challenger: Scorecard,
    kind: str,
    gate_cfg: dict[str, Any],
    latency_budget_ms: float,
) -> Decision:
    """The gate's rules, as a pure function of the two scorecards."""
    checks: list[Check] = []
    if champion is None:
        if kind == "tabular":
            floor = gate_cfg["bootstrap"]["tabular_min_lift"] * challenger.baseline
            label = f"{gate_cfg['bootstrap']['tabular_min_lift']}x the positive rate"
        else:
            floor = gate_cfg["bootstrap"]["notes_min_macro_f1"]
            label = "the minimum macro F1"
        checks.append(
            Check(
                "beats_naive_baseline",
                challenger.primary >= floor,
                f"{challenger.primary:.4f} vs {floor:.4f} ({label})",
            )
        )
    else:
        needed = champion.primary * (1 + gate_cfg["min_relative_improvement"])
        checks.append(
            Check(
                "primary_metric_improves",
                challenger.primary >= needed,
                f"challenger {challenger.primary:.4f} vs champion {champion.primary:.4f} "
                f"(needs >= {needed:.4f})",
            )
        )
        worst: list[str] = []
        for name, champ_slice in champion.slices.items():
            if name not in challenger.slices:
                continue
            old, new = champ_slice["metric"], challenger.slices[name]["metric"]
            if kind == "tabular":
                bad = new < old * (1 - gate_cfg["max_slice_drop_tabular"])
            else:
                bad = new < old - gate_cfg["max_slice_drop_notes"]
            if bad:
                worst.append(f"{name}: {old:.4f} -> {new:.4f}")
        checks.append(
            Check(
                "no_slice_degrades",
                not worst,
                "; ".join(worst) or f"{len(champion.slices)} slices checked",
            )
        )

    p99 = challenger.latency["p99_ms"]
    checks.append(
        Check(
            "latency_within_budget",
            p99 <= latency_budget_ms,
            f"p99 {p99:.2f} ms vs budget {latency_budget_ms} ms",
        )
    )
    return Decision(promote=all(c.passed for c in checks), checks=checks)


def _version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def run(kind: str) -> Decision:
    cfg, data_cfg = train_config(), load_config()
    gate_cfg = cfg["gate"]
    name = cfg["mlflow"]["model_tabular" if kind == "tabular" else "model_nlp"]
    test = pd.read_parquet(resolve(data_cfg["paths"]["processed_dir"]) / "test.parquet")

    start_experiment(gate_cfg["experiment"])
    client = MlflowClient()
    challenger_v = _version(client, name, "challenger")
    if challenger_v is None:
        raise SystemExit(f"No @challenger for {name}: train a model first.")
    champion_v = _version(client, name, "champion")
    if champion_v == challenger_v:
        raise SystemExit(f"@challenger {challenger_v} is already @champion.")

    def scorecard(version: str) -> Scorecard:
        model_bytes = mlflow.onnx.load_model(f"models:/{name}/{version}").SerializeToString()
        threads = cfg["export"]["inference_threads"]
        if kind == "tabular":
            return score_tabular(model_bytes, test, data_cfg, gate_cfg, threads)
        labels = json.loads(client.get_model_version(name, version).tags["labels"])
        return score_notes(
            model_bytes, test, data_cfg, gate_cfg, load_tokenizer(data_cfg), labels, threads
        )

    challenger = scorecard(challenger_v)
    champion = scorecard(champion_v) if champion_v else None
    decision = decide(champion, challenger, kind, gate_cfg, cfg["export"]["latency_budget_p99_ms"])

    with mlflow.start_run(run_name=f"gate-{kind}-v{challenger_v}"):
        mlflow.log_params(
            {
                "model": name,
                "champion_version": champion_v or "none",
                "challenger_version": challenger_v,
            }
        )
        mlflow.log_metrics(
            {
                "challenger_primary": challenger.primary,
                "challenger_p99_ms": challenger.latency["p99_ms"],
            }
        )
        if champion:
            mlflow.log_metrics(
                {
                    "champion_primary": champion.primary,
                    "champion_p99_ms": champion.latency["p99_ms"],
                }
            )
        mlflow.set_tag("decision", "promoted" if decision.promote else "rejected")
        report = {
            "model": name,
            "kind": kind,
            "champion_version": champion_v,
            "challenger_version": challenger_v,
            "promoted": decision.promote,
            "checks": [asdict(c) for c in decision.checks],
            "challenger": asdict(challenger),
            "champion": asdict(champion) if champion else None,
        }
        mlflow.log_dict(report, "gate_report.json")

    if decision.promote:
        if champion_v:
            client.set_registered_model_alias(name, "previous_champion", champion_v)
        client.set_registered_model_alias(name, "champion", challenger_v)
        client.delete_registered_model_alias(name, "challenger")
    client.set_model_version_tag(
        name, challenger_v, "gate_decision", "promoted" if decision.promote else "rejected"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{kind}.json").write_text(json.dumps(report, indent=2) + "\n")
    log_event(
        logger,
        logging.INFO if decision.promote else logging.WARNING,
        "gate_promoted" if decision.promote else "gate_rejected",
        model=name,
        champion=champion_v,
        challenger=challenger_v,
        checks=[f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.detail}" for c in decision.checks],
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=["tabular", "notes"], required=True)
    decision = run(parser.parse_args().model)
    sys.exit(0 if decision.promote else 1)


if __name__ == "__main__":
    main()
