"""Guard for promoting a canary release in prod (make canary-promote).

The canary serves the @staging models, the version about to go to prod
(make canary-bundle). Promoting it moves @prod to @staging
(src/serving/environments.py). Before that, the guard checks that the
canary's triage model is still @staging: if staging moved on while the
canary ran, the version promoted would differ from the one that was checked
in prod, so promotion is refused.

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
SOURCE_ALIAS = "staging"


def alias_version(name: str, alias: str = SOURCE_ALIAS) -> str:
    from mlflow import MlflowClient

    return str(MlflowClient().get_model_version_by_alias(name, alias).version)


def promotable(bundle: Path, staging: str) -> tuple[bool, str]:
    """Whether the canary bundle's triage model is still the registry's @staging."""
    tabular = json.loads((bundle / "metadata.json").read_text())["tabular"]
    version = str(tabular["version"])
    if version == staging:
        return True, f"canary serves v{version}, still @staging"
    return False, (
        f"canary serves v{version}, but @staging is now v{staging}: the version that "
        "ran in prod isn't the one that would be promoted. Roll back (make "
        "canary-rollback) and canary the new @staging."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a canary can be promoted")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    ok, reason = promotable(args.bundle, alias_version(model_store.TABULAR_MODEL))
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
