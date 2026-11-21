"""Load YAML configuration files from configs/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from data_ml_toolkit.config import load_yaml, resolve_path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


# Point to a different data config file, e.g. for tests or a strict-contract
# drill: CLAIM_TRIAGE_DATA_CONFIG=/path/to/config.yaml
CONFIG_OVERRIDE_ENV = "CLAIM_TRIAGE_DATA_CONFIG"
# Same for the training config, e.g. fewer tuning trials in CI.
TRAIN_CONFIG_OVERRIDE_ENV = "CLAIM_TRIAGE_TRAIN_CONFIG"


def load_config(name: str = "data_config.yaml") -> dict[str, Any]:
    """Read a config file from configs/ as a dictionary.

    CLAIM_TRIAGE_DATA_CONFIG and CLAIM_TRIAGE_TRAIN_CONFIG (if set) point to
    replacement files for data_config.yaml and train_config.yaml.
    """
    path = CONFIG_DIR / name
    override = {
        "data_config.yaml": CONFIG_OVERRIDE_ENV,
        "train_config.yaml": TRAIN_CONFIG_OVERRIDE_ENV,
    }.get(name)
    if override and os.environ.get(override):
        path = Path(os.environ[override])
    return load_yaml(path)


def resolve(path: str | Path) -> Path:
    """A path from the config, relative to the repository root."""
    return resolve_path(ROOT, path)
