"""Tabular feature pipeline, shared by training and serving.

Two layers, so training and serving compute features identically:

    add_row_features   pure, row-by-row derivations (no statistics), applied
                       before the pipeline in both training and serving
    build_preprocessor a scikit-learn ColumnTransformer whose statistics
                       (medians, category codes) are learned with fit() on the
                       training split only, then saved and reused unchanged

Robust to dirty input at serving time:
    - missing numbers are imputed with the training median
    - missing categories, and categories never seen in training, become -1
    - negative or zero density is clipped before the log
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

UNKNOWN_CATEGORY = -1


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns that depend only on the row itself."""
    out = df.copy()
    density = pd.to_numeric(out["density"], errors="coerce").clip(lower=1)
    out["log_density"] = np.log(density)
    return out


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Unfitted preprocessor for the tabular model. Fit it on training data only."""
    features = cfg["features"]
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=UNKNOWN_CATEGORY,
                    encoded_missing_value=UNKNOWN_CATEGORY,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, features["numeric"]),
            ("categorical", categorical, features["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Input columns the preprocessor reads, in order."""
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])


def raw_input_columns(cfg: dict[str, Any]) -> list[str]:
    """Columns a caller must send: the features before row derivations."""
    derived = {"log_density": "density"}
    return [derived.get(c, c) for c in feature_columns(cfg)]
