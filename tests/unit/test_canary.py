"""A canary is promotable only while it still serves the registry's @staging."""

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


def test_the_staging_version_can_be_promoted(bundle):
    ok, reason = canary.promotable(bundle("3"), staging="3")
    assert ok and "v3" in reason


def test_staging_moved_on_while_the_canary_ran(bundle):
    ok, reason = canary.promotable(bundle("3"), staging="4")
    assert not ok
    assert "@staging is now v4" in reason


def test_cli_exits_non_zero_when_refused(bundle, monkeypatch):
    monkeypatch.setattr(canary, "alias_version", lambda _name: "4")
    assert canary.main(["--bundle", str(bundle("3"))]) == 1
    assert canary.main(["--bundle", str(bundle("4"))]) == 0
