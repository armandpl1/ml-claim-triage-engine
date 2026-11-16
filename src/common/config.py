"""Load YAML configuration files from configs/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


# Point to a different data config file, e.g. for tests or a strict-contract
# drill: CLAIM_TRIAGE_DATA_CONFIG=/path/to/config.yaml
CONFIG_OVERRIDE_ENV = "CLAIM_TRIAGE_DATA_CONFIG"


def load_config(name: str = "data_config.yaml") -> dict[str, Any]:
    """Read a config file from configs/ as a dictionary.

    For the data config, CLAIM_TRIAGE_DATA_CONFIG (if set) points to a
    replacement file.
    """
    path = CONFIG_DIR / name
    if name == "data_config.yaml" and os.environ.get(CONFIG_OVERRIDE_ENV):
        path = Path(os.environ[CONFIG_OVERRIDE_ENV])
    with path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def resolve(path: str) -> Path:
    """A path from the config, relative to the repository root."""
    return ROOT / path
