"""Export the tabular pipeline (preprocessor + LightGBM) to one ONNX graph.

The whole scikit-learn pipeline is converted, so the ONNX model takes the raw
feature columns (after add_row_features) and applies the same imputation and
category encoding as training. Serving needs only onnxruntime.

Inputs are one named tensor per column, shape (rows, 1): float32 for numbers,
string for categories. The output "probabilities" has shape (rows, 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.pipeline import Pipeline

update_registered_converter(
    LGBMClassifier,
    "LightGbmLGBMClassifier",
    calculate_linear_classifier_output_shapes,
    convert_lightgbm,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)


def export_tabular(pipeline: Pipeline, cfg: dict[str, Any], opset: int) -> onnx.ModelProto:
    numeric, categorical = cfg["features"]["numeric"], cfg["features"]["categorical"]
    initial_types = [(c, FloatTensorType([None, 1])) for c in numeric] + [
        (c, StringTensorType([None, 1])) for c in categorical
    ]
    model = pipeline.steps[-1][1]
    return convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset={"": opset, "ai.onnx.ml": 3},
        options={id(model): {"zipmap": False}},
    )


def onnx_inputs(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Feed dict for the tabular ONNX model from a DataFrame of features."""
    feeds: dict[str, np.ndarray] = {}
    for c in cfg["features"]["numeric"]:
        feeds[c] = frame[c].to_numpy(dtype=np.float32).reshape(-1, 1)
    for c in cfg["features"]["categorical"]:
        feeds[c] = frame[c].astype(object).where(frame[c].notna(), "").to_numpy().reshape(-1, 1)
    return feeds
