"""The feature registry: which features exist, where they come from, and how
the model's inputs are derived from them (configs/feature_store.yaml).

Checked against the model's inputs (data_config.yaml) when loaded, so the
store can't drift away from what training and serving expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.common.config import load_config, resolve
from src.features.transform import feature_columns, raw_input_columns

CONFIG_FILE = "feature_store.yaml"


class RegistryError(ValueError):
    """The registry disagrees with itself or with the model's inputs."""


@dataclass(frozen=True)
class Renewal:
    first: date
    days: int
    age_step: int
    bonus_malus_factor: float
    bonus_malus_floor: int


@dataclass(frozen=True)
class FeatureView:
    name: str
    entity: str
    source: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    on_demand: dict[str, str]  # derived feature -> the stored field it comes from

    @property
    def fields(self) -> list[str]:
        """Stored feature columns, in order."""
        return [*self.numeric, *self.categorical]


@dataclass(frozen=True)
class Registry:
    view: FeatureView
    offline: Path
    online: Path
    recorded_at: date
    renewal: Renewal


def _view(name: str, entity: str, spec: dict[str, Any]) -> FeatureView:
    return FeatureView(
        name=name,
        entity=entity,
        source=str(spec["source"]),
        numeric=tuple(spec["fields"]["numeric"]),
        categorical=tuple(spec["fields"]["categorical"]),
        on_demand=dict(spec.get("on_demand") or {}),
    )


def validate(registry: Registry, data_cfg: dict[str, Any]) -> None:
    """Raise RegistryError if the registry can't produce the model's inputs."""
    view = registry.view
    stored = set(view.fields)
    if len(stored) != len(view.fields):
        raise RegistryError(f"{view.name}: a field is listed twice")
    for derived, field in view.on_demand.items():
        if field not in stored:
            raise RegistryError(f"{view.name}: {derived} comes from {field}, not a stored field")
    if set(raw_input_columns(data_cfg)) != stored:
        raise RegistryError(
            f"{view.name} stores {sorted(stored)}, the model reads "
            f"{sorted(raw_input_columns(data_cfg))}"
        )
    available = stored | set(view.on_demand)
    missing = set(feature_columns(data_cfg)) - available
    if missing:
        raise RegistryError(f"no feature definition for {sorted(missing)}")
    excluded = set(data_cfg["features"].get("excluded") or {}) - set(view.on_demand.values())
    if excluded & stored:
        raise RegistryError(f"excluded columns stored as features: {sorted(excluded & stored)}")
    if registry.renewal.first <= registry.recorded_at:
        raise RegistryError("renewals must start after the source snapshot is recorded")


def load_registry(
    cfg: dict[str, Any] | None = None, data_cfg: dict[str, Any] | None = None
) -> Registry:
    """The validated registry from configs/feature_store.yaml."""
    store = (cfg or load_config(CONFIG_FILE))["feature_store"]
    (name, spec), *others = store["views"].items()
    if others:
        raise RegistryError("one feature view is supported (policy_features)")
    renewal = store["history"]["renewal"]
    registry = Registry(
        view=_view(name, str(store["entity"]), spec),
        offline=resolve(store["offline"]),
        online=resolve(store["online"]),
        recorded_at=date.fromisoformat(str(store["history"]["recorded_at"])),
        renewal=Renewal(
            first=date.fromisoformat(str(renewal["first"])),
            days=int(renewal["days"]),
            age_step=int(renewal["age_step"]),
            bonus_malus_factor=float(renewal["bonus_malus_factor"]),
            bonus_malus_floor=int(renewal["bonus_malus_floor"]),
        ),
    )
    validate(registry, data_cfg or load_config())
    return registry
