"""Request and response schemas for the triage API.

Request limits mirror the data contracts (configs/data_config.yaml), so the
API refuses inputs the models were never trained to handle, with a 422 and a
message naming the bad field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str | None = Field(default=None, max_length=64, description="Caller's reference")
    exposure: float = Field(gt=0, le=2.01, description="Policy years in force")
    vehicle_power: int = Field(ge=1, le=20)
    vehicle_age: int = Field(ge=0, le=100)
    driver_age: int = Field(ge=16, le=100)
    bonus_malus: int = Field(ge=50, le=350)
    density: float = Field(ge=0, description="Population per km2 where the driver lives")
    area: Literal["A", "B", "C", "D", "E", "F"]
    vehicle_brand: str = Field(pattern=r"^B\d+$", examples=["B12"])
    fuel_type: Literal["Regular", "Diesel"]
    region: str = Field(pattern=r"^R\d{2}$", examples=["R82"])
    adjuster_note: str | None = Field(
        default=None, min_length=3, max_length=2000, description="Free-text first notice of loss"
    )


class PolicyClaimRequest(BaseModel):
    """A claim identified by its policy: the tabular features come from the
    online feature store instead of the caller."""

    model_config = ConfigDict(extra="forbid")

    policy_id: int = Field(ge=1, description="Policy the claim is made on")
    claim_id: str | None = Field(default=None, max_length=64, description="Caller's reference")
    adjuster_note: str | None = Field(
        default=None, min_length=3, max_length=2000, description="Free-text first notice of loss"
    )


class NotePrediction(BaseModel):
    category: str
    probabilities: dict[str, float]


class ClaimPrediction(BaseModel):
    request_id: str
    claim_id: str | None
    high_cost_probability: float
    route: Literal["senior_adjuster", "standard"]
    decision_threshold: float
    note: NotePrediction | None
    model_versions: dict[str, str]
    latency_ms: float
    # /predict/policy only: when the features read from the store took effect
    feature_valid_from: str | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    # Release of the running image (APP_VERSION, set when the image is built)
    version: str = "dev"
    # Environment this pod belongs to (SERVING_ENV: dev, staging, prod)
    environment: str = "local"
    models: dict[str, str]
    source: str
    uptime_seconds: float
    # Online feature store: as_of, materialized_at, row_count (None if absent)
    online_store: dict[str, str | int] | None = None
