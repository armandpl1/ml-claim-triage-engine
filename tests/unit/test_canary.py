"""A canary is promotable only if the gate already made its model @champion."""

from __future__ import annotations

import json

import pytest

from src.serving import canary


@pytest.fixture
def bundle(tmp_path):
    def make(version: str):
        (tmp_path / "metadata.json").write_text(json.dumps({"tabular": {"version": version}}))
        return tmp_path

    return make


def test_the_champion_can_be_promoted(bundle):
    ok, reason = canary.promotable(bundle("3"), champion="3")
    assert ok and "v3" in reason


def test_a_model_the_gate_did_not_promote_is_refused(bundle):
    ok, reason = canary.promotable(bundle("2"), champion="1")
    assert not ok
    assert "gate has not promoted" in reason


def test_cli_exits_non_zero_when_refused(bundle, monkeypatch):
    monkeypatch.setattr(canary, "champion_version", lambda _name: "1")
    assert canary.main(["--bundle", str(bundle("2"))]) == 1
    assert canary.main(["--bundle", str(bundle("1"))]) == 0
