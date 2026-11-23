"""Dagster definitions: the jobs run the right commands in order, a gate
rejection is an outcome (not a failure), and heavy steps get the memory cap."""

from __future__ import annotations

import subprocess
from typing import ClassVar

import dagster as dg
import pytest

from src.orchestration import definitions as d
from src.orchestration import resources


class FakeCommands(resources.Commands):
    """Records each command instead of running it; exit codes by module."""

    calls: ClassVar[list[list[str]]] = []
    codes: ClassVar[dict[str, int]] = {}

    def run(self, args, heavy=False, ok_codes=(0,)):
        module = args[2] if len(args) > 2 else args[-1]
        FakeCommands.calls.append([module, *args[3:]])
        code = FakeCommands.codes.get(module, 0)
        if code not in ok_codes:
            raise dg.Failure(f"{module} exited with {code}")
        return code


@pytest.fixture(autouse=True)
def reset():
    FakeCommands.calls.clear()
    FakeCommands.codes.clear()


def run_job(name: str) -> dg.ExecuteInProcessResult:
    job = d.defs.resolve_job_def(name)
    return job.execute_in_process(resources={"commands": FakeCommands()}, raise_on_error=False)


def test_definitions_load():
    d.defs.validate_loadable(d.defs)
    assert {j.name for j in d.defs.resolve_all_job_defs()} >= {"retrain_tabular", "retrain_notes"}


def test_retrain_tabular_runs_data_train_gate_bundle_in_order():
    assert run_job("retrain_tabular").success
    assert FakeCommands.calls == [
        ["dvc", "repro"],
        ["src.training.tabular"],
        ["src.evaluation.gate", "--model", "tabular"],
        ["src.serving.model_store"],
    ]


def test_a_rejected_challenger_is_recorded_not_failed():
    FakeCommands.codes["src.evaluation.gate"] = 1
    result = run_job("retrain_tabular")
    assert result.success
    gate = result.asset_materializations_for_node("tabular_gate")[0]
    assert gate.metadata["promoted"].value is False


def test_a_crashing_step_fails_the_run_and_stops_downstream():
    FakeCommands.codes["src.training.tabular"] = 137  # OOM-killed
    result = run_job("retrain_tabular")
    assert not result.success
    assert ["src.serving.model_store"] not in FakeCommands.calls


def test_only_heavy_steps_get_the_memory_cap(monkeypatch):
    seen = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **_: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0)
    )
    commands = resources.Commands(memory_cap="2500M")
    commands.run(["python", "-m", "src.training.tabular"], heavy=True)
    commands.run(["python", "-m", "dvc", "repro"])
    assert seen[0][:3] == ["systemd-run", "--user", "--scope"] and "MemoryMax=2500M" in seen[0]
    assert seen[1] == ["python", "-m", "dvc", "repro"]
