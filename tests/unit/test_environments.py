"""Model promotion: staging takes the champion, prod takes staging, and every
move can be rolled back."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.serving import environments as envs
from src.serving import model_store

TAB, NOTES = model_store.TABULAR_MODEL, model_store.NOTES_MODEL


class FakeRegistry:
    def __init__(self, aliases: dict[str, dict[str, str]]) -> None:
        self.data = {name: dict(a) for name, a in aliases.items()}

    def get_registered_model(self, name):
        return SimpleNamespace(aliases=dict(self.data[name]))

    def set_registered_model_alias(self, name, alias, version):
        self.data[name][alias] = str(version)

    def delete_registered_model_alias(self, name, alias):
        self.data[name].pop(alias, None)


def registry(**tabular) -> FakeRegistry:
    return FakeRegistry({TAB: tabular, NOTES: {"champion": "1"}})


def test_staging_takes_the_champion_then_prod_takes_staging():
    reg = registry(champion="3")
    envs.promote(reg, "staging")
    assert reg.data[TAB]["staging"] == "3" and reg.data[NOTES]["staging"] == "1"
    assert "prod" not in reg.data[TAB]
    envs.promote(reg, "prod")
    assert reg.data[TAB]["prod"] == "3"


def test_prod_never_skips_staging():
    reg = registry(champion="4", staging="3", prod="2")
    envs.promote(reg, "prod")
    assert reg.data[TAB]["prod"] == "3"  # not the newer champion


def test_promotion_records_the_previous_version():
    reg = registry(champion="4", staging="3")
    [move, *_] = envs.promote(reg, "staging")
    assert (move.old, move.new) == ("3", "4")
    assert reg.data[TAB]["previous_staging"] == "3"


def test_nothing_to_promote_is_refused():
    with pytest.raises(envs.PromotionError, match="no @staging"):
        envs.promote(registry(champion="3"), "prod")


def test_promoting_the_same_version_is_a_no_op():
    reg = registry(champion="3", staging="3")
    reg.data[NOTES]["staging"] = "1"
    assert envs.promote(reg, "staging") == []
    assert "previous_staging" not in reg.data[TAB]


def test_rollback_restores_the_previous_version_and_can_be_undone():
    reg = registry(champion="4", staging="4", prod="3")
    envs.promote(reg, "prod")  # prod 3 -> 4
    envs.rollback(reg, "prod")
    assert reg.data[TAB]["prod"] == "3" and reg.data[TAB]["previous_prod"] == "4"
    envs.rollback(reg, "prod")  # undo the rollback
    assert reg.data[TAB]["prod"] == "4"


def test_rollback_without_history_is_refused():
    with pytest.raises(envs.PromotionError, match="nothing to roll back"):
        envs.rollback(registry(champion="3", prod="3"), "prod")


def test_notes_model_is_optional():
    reg = FakeRegistry({TAB: {"champion": "2"}, NOTES: {}})
    moves = envs.promote(reg, "staging")
    assert [m.model for m in moves] == [TAB]
