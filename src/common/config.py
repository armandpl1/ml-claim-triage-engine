"""Load YAML configuration files from configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_config(name: str = "data_config.yaml") -> dict[str, Any]:
    """Read a config file from configs/ as a dictionary."""
    with (CONFIG_DIR / name).open() as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def resolve(path: str) -> Path:
    """A path from the config, relative to the repository root."""
    return ROOT / path
