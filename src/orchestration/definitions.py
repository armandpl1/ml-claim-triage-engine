"""Dagster definitions: the training path as software-defined assets.

    training_data  ->  tabular_challenger  ->  tabular_gate  ->  serving_bundle
    (dvc repro:        (Optuna + ONNX,         (champion vs      (@champion models
     fetch, validate,   registered as           challenger;       exported for the
     features)          @challenger)            promotes or not)  API)

- Every step runs as a subprocess with the same commands as the Makefile,
  so the Dagster process stays small and a heavy step can't take it down.
  CLAIM_TRIAGE_MEMORY_CAP (e.g. 2500M) wraps training in a memory-capped
  systemd scope, as AGENTS.md asks for heavy jobs.
- The gate rejecting a challenger is a normal outcome, not a failure: the
  asset records the decision, and serving_bundle only re-exports when the
  champion changed.
- nightly_batch (src/orchestration/batch.py) scores each night's claims,
  loads settled outcomes and checks performance, one partition per night.
- feature_drift and the retrain_on_alert sensor (src/orchestration/
  sensors.py) turn a drift or performance alert into a retrain_tabular run.
- retrain_tabular runs the whole path. The notes model has its own job,
  retrain_notes, because fine-tuning takes about an hour on a CPU.

Run locally:  make orchestrate   (Dagster UI at http://localhost:3001)
"""

from __future__ import annotations

import dagster as dg

from src.orchestration.batch import (
    BATCH_ASSETS,
    BatchSettings,
    nightly_batch,
    nightly_batch_schedule,
)
from src.orchestration.resources import PYTHON, Commands, read_json
from src.orchestration.sensors import MONITORING_ASSETS, SENSORS


@dg.asset(
    group_name="data",
    description="Validated warehouse and leakage-free splits (dvc repro: only changed stages run).",
)
def training_data(commands: Commands) -> dg.MaterializeResult:
    commands.run([PYTHON, "-m", "dvc", "repro"])
    report = read_json("data/processed/feature_report.json")
    return dg.MaterializeResult(
        metadata={k: v for k, v in report.items() if isinstance(v, int | float | str)}
    )


@dg.asset(
    group_name="training",
    deps=[training_data],
    description="LightGBM tuned with Optuna, exported to ONNX, registered as @challenger.",
)
def tabular_challenger(commands: Commands) -> dg.MaterializeResult:
    commands.run([PYTHON, "-m", "src.training.tabular"], heavy=True)
    metrics = read_json("artifacts/tabular/metrics.json").get("metrics", {})
    return dg.MaterializeResult(
        metadata={f"val_{k}": dg.MetadataValue.float(float(v)) for k, v in metrics.items()}
    )


@dg.asset(
    group_name="training",
    deps=[training_data],
    description="DistilBERT + LoRA notes classifier (about 1 h on a CPU), as @challenger.",
)
def notes_challenger(commands: Commands) -> dg.MaterializeResult:
    commands.run([PYTHON, "-m", "src.training.nlp"], heavy=True)
    metrics = read_json("artifacts/nlp/metrics.json")
    return dg.MaterializeResult(metadata={"metrics": dg.MetadataValue.json(metrics)})


def _gate(commands: Commands, kind: str) -> dg.MaterializeResult:
    code = commands.run([PYTHON, "-m", "src.evaluation.gate", "--model", kind], ok_codes=(0, 1))
    report = read_json(f"artifacts/gate/{kind}.json")
    return dg.MaterializeResult(
        metadata={
            "promoted": code == 0,
            "champion_version": str(report.get("champion_version")),
            "challenger_version": str(report.get("challenger_version")),
            "checks": dg.MetadataValue.json(report.get("checks", [])),
        }
    )


@dg.asset(
    group_name="training",
    deps=[tabular_challenger],
    description="Champion vs challenger gate; promotes only if every check passes.",
)
def tabular_gate(commands: Commands) -> dg.MaterializeResult:
    return _gate(commands, "tabular")


@dg.asset(
    group_name="training",
    deps=[notes_challenger],
    description="Champion vs challenger gate for the notes classifier.",
)
def notes_gate(commands: Commands) -> dg.MaterializeResult:
    return _gate(commands, "notes")


@dg.asset(
    group_name="serving",
    deps=[tabular_gate, notes_gate],
    description="The @champion models exported to serving_bundle/ for the API.",
)
def serving_bundle(commands: Commands) -> dg.MaterializeResult:
    commands.run([PYTHON, "-m", "src.serving.model_store"])
    meta = read_json("serving_bundle/metadata.json")
    return dg.MaterializeResult(
        metadata={f"{k}_version": str(v.get("version")) for k, v in meta.items()}
    )


retrain_tabular = dg.define_asset_job(
    "retrain_tabular",
    selection=dg.AssetSelection.assets(
        training_data, tabular_challenger, tabular_gate, serving_bundle
    ),
    description="Data, tabular training, gate, bundle. About 6 minutes on a laptop.",
)
retrain_notes = dg.define_asset_job(
    "retrain_notes",
    selection=dg.AssetSelection.assets(training_data, notes_challenger, notes_gate, serving_bundle),
    description="Data, notes fine-tuning, gate, bundle. About an hour on a CPU.",
)

defs = dg.Definitions(
    assets=[
        training_data,
        tabular_challenger,
        notes_challenger,
        tabular_gate,
        notes_gate,
        serving_bundle,
        *BATCH_ASSETS,
        *MONITORING_ASSETS,
    ],
    jobs=[retrain_tabular, retrain_notes, nightly_batch],
    schedules=[nightly_batch_schedule],
    sensors=SENSORS,
    resources={"commands": Commands(), "batch": BatchSettings()},
)
