"""Smoke-check the MLflow server from `make mlflow-up`: runs go to Postgres,
artifacts go to S3 through the server, and the registry works.

Steps: log a run with a small artifact to a throwaway experiment, register
it as a model version, download the artifact back through the server, and
confirm the file landed in the MLflow bucket. Needs no S3 credentials on the
client for the MLflow part (the server proxies artifacts); the bucket read
uses the emulator's.

Exits 1 with one structured event ("mlflow_check_failed") on any problem.

Usage:
    python -m src.infra.mlflow_check
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3
import mlflow
from mlflow import MlflowClient

from src.common.logging import get_logger, log_event

logger = get_logger("infra.mlflow_check")

EXPERIMENT = "infra-smoke-check"
MODEL = "infra-smoke-check"


def run_check(tracking_uri: str, bucket: str, endpoint: str, region: str) -> dict[str, str]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = mlflow.set_experiment(EXPERIMENT)
    if not experiment.artifact_location.startswith("mlflow-artifacts:"):
        raise RuntimeError(
            f"experiment stores artifacts at {experiment.artifact_location}, not through the server"
        )

    marker = f"smoke check {time.time_ns()}"
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run() as run:
        path = Path(tmp) / "marker.txt"
        path.write_text(marker)
        mlflow.log_param("purpose", "infra smoke check")
        mlflow.log_metric("ok", 1.0)
        mlflow.log_artifact(str(path), artifact_path="check")
    # A version pointing at the run's artifact folder: enough to exercise the
    # registry without a model library (training uses mlflow.onnx.log_model)
    if not client.search_registered_models(filter_string=f"name = '{MODEL}'"):
        client.create_registered_model(MODEL)
    version = client.create_model_version(
        MODEL, source=f"{run.info.artifact_uri}/check", run_id=run.info.run_id
    )

    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            f"models:/{MODEL}/{version.version}", dst_path=tmp
        )
        if (Path(local) / "marker.txt").read_text() != marker:
            raise RuntimeError("downloaded artifact differs from the logged one")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )
    keys = [
        o["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
        for o in page.get("Contents", [])
        if run.info.run_id in o["Key"] and o["Key"].endswith("marker.txt")
    ]
    if not keys:
        raise RuntimeError(f"no artifact of run {run.info.run_id} in s3://{bucket}")
    if client.get_run(run.info.run_id).data.metrics.get("ok") != 1.0:
        raise RuntimeError("metric not stored in the backend")
    return {"run_id": run.info.run_id, "model_version": str(version.version), "s3_key": keys[0]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    )
    parser.add_argument("--bucket", default="claim-triage-dev-mlflow")
    parser.add_argument(
        "--endpoint", default=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:15100")
    )
    parser.add_argument("--region", default="ca-central-1")
    args = parser.parse_args(argv)
    try:
        found = run_check(args.tracking_uri, args.bucket, args.endpoint, args.region)
    except Exception as err:
        log_event(logger, logging.ERROR, "mlflow_check_failed", error=str(err))
        return 1
    log_event(logger, logging.INFO, "mlflow_check_passed", tracking_uri=args.tracking_uri, **found)
    return 0


if __name__ == "__main__":
    sys.exit(main())
