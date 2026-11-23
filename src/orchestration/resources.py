"""Shared Dagster resources: running the project's commands."""

import json
import os
import subprocess
import sys
from typing import Any

import dagster as dg

from src.common.config import ROOT

PYTHON = sys.executable


class Commands(dg.ConfigurableResource):
    """Runs the project's commands from the repository root."""

    memory_cap: str = os.environ.get("CLAIM_TRIAGE_MEMORY_CAP", "")

    def run(self, args: list[str], heavy: bool = False, ok_codes: tuple[int, ...] = (0,)) -> int:
        command = list(args)
        if heavy and self.memory_cap:
            command = [
                "systemd-run", "--user", "--scope", "--quiet",
                "-p", f"MemoryMax={self.memory_cap}", "-p", "MemorySwapMax=0",
                "nice", "-n", "19", *command,
            ]  # fmt: skip
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        if code not in ok_codes:
            raise dg.Failure(f"{' '.join(args)} exited with {code}")
        return code


def read_json(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else {}
