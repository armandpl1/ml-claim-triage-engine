"""The registry defines exactly the model's inputs, and refuses configs that
don't."""

from __future__ import annotations

import copy

import pytest

from src.common.config import load_config
from src.feature_store.registry import CONFIG_FILE, RegistryError, load_registry


@pytest.fixture
def store_cfg():
    return copy.deepcopy(load_config(CONFIG_FILE))


def test_real_registry_matches_the_model_inputs():
    registry = load_registry()
    assert registry.view.entity == "policy_id"
    assert "density" in registry.view.fields
    assert registry.view.on_demand == {"log_density": "density"}
    assert registry.renewal.first > registry.recorded_at


def test_missing_field_is_refused(store_cfg):
    store_cfg["feature_store"]["views"]["policy_features"]["fields"]["numeric"].remove(
        "bonus_malus"
    )
    with pytest.raises(RegistryError, match="model reads"):
        load_registry(store_cfg)


def test_excluded_column_is_refused(store_cfg, cfg):
    store_cfg["feature_store"]["views"]["policy_features"]["fields"]["numeric"].append(
        "claim_count"
    )
    cfg["features"]["numeric"].append("claim_count")
    with pytest.raises(RegistryError, match="excluded"):
        load_registry(store_cfg, cfg)


def test_on_demand_feature_needs_a_stored_field(store_cfg):
    store_cfg["feature_store"]["views"]["policy_features"]["on_demand"] = {"log_density": "pop"}
    with pytest.raises(RegistryError, match="not a stored field"):
        load_registry(store_cfg)


def test_renewals_must_follow_the_snapshot(store_cfg):
    store_cfg["feature_store"]["history"]["renewal"]["first"] = "2023-06-01"
    with pytest.raises(RegistryError, match="renewals"):
        load_registry(store_cfg)
