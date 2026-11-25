"""Guard for promoting a canary release (make canary-promote).

The gate decides which model is the champion (src/evaluation/gate.py); a
canary only checks a model online before it gets all the traffic. So a
canary can be promoted only if its triage model is already @champion in the
registry. Otherwise promotion would put a model in front of every claim that
the gate never accepted.

Usage:
    python -m src.serving.canary --bundle serving_bundle_canary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.common.logging import get_logger, log_event
from src.serving import model_store

logger = get_logger("serving.canary")


def champion_version(name: str) -> str:
    from mlflow import MlflowClient

    return str(MlflowClient().get_model_version_by_alias(name, "champion").version)


def promotable(bundle: Path, champion: str) -> tuple[bool, str]:
    """Whether the canary bundle's triage model is the registry's champion."""
    tabular = json.loads((bundle / "metadata.json").read_text())["tabular"]
    version = str(tabular["version"])
    if version == champion:
        return True, f"canary serves v{version}, the @champion"
    return False, (
        f"canary serves v{version}, but @champion is v{champion}: the gate has not "
        "promoted it. Roll back (make canary-rollback), or pass the gate first."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a canary can be promoted")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    ok, reason = promotable(args.bundle, champion_version(model_store.TABULAR_MODEL))
    log_event(
        logger,
        logging.INFO if ok else logging.ERROR,
        "canary_promotion_check",
        promotable=ok,
        reason=reason,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
