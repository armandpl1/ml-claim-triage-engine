"""Per-claim reason codes for the triage model, from TreeSHAP.

LightGBM's pred_contrib splits each claim's raw score (log-odds) into one
contribution per feature plus a base value, exactly: they sum to the score
the model computed (Lundberg et al.'s TreeSHAP, built into LightGBM, so no
extra library). A positive contribution pushed the claim toward high cost.

    contributions   one row per claim, one column per model feature, plus
                    "_base"; the row sums to the raw score
    reason_codes    long table: each claim's top features that raised its
                    risk, with the claim's own value for that feature
    importance      mean absolute contribution per feature, over many claims
                    (global importance on the same scale as the reasons)

Values are shown as the adjuster knows them: density, not log_density.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BASE = "_base"
# Model features shown under the raw column a person recognizes.
DISPLAY = {"log_density": "density"}


def contributions(pipe: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contributions to each claim's raw score (log-odds)."""
    preprocess, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    names = [str(n) for n in preprocess.get_feature_names_out()]
    values = model.booster_.predict(preprocess.transform(frame), pred_contrib=True)
    out = pd.DataFrame(values[:, :-1], columns=names, index=frame.index)
    out[BASE] = values[:, -1]
    return out


def reason_codes(
    frame: pd.DataFrame,
    contrib: pd.DataFrame,
    top_k: int,
    min_contribution: float,
    key: str = "claim_id",
) -> pd.DataFrame:
    """The top_k features that raised each claim's risk the most.

    Columns: key, rank (1 = largest), feature, value, contribution. Claims
    with nothing above min_contribution get no rows: no reason is better than
    a reason made of noise.
    """
    features = contrib.drop(columns=BASE)
    long = (
        features.assign(**{key: frame[key].to_numpy()})
        .melt(id_vars=key, var_name="model_feature", value_name="contribution")
        .query("contribution >= @min_contribution")
        .sort_values([key, "contribution", "model_feature"], ascending=[True, False, True])
    )
    long["rank"] = long.groupby(key).cumcount() + 1
    long = long[long["rank"] <= top_k].copy()
    long["feature"] = long["model_feature"].map(lambda f: DISPLAY.get(f, f))
    by_key = frame.set_index(key)
    long["value"] = [
        _show(by_key.at[k, f]) for k, f in zip(long[key], long["feature"], strict=True)
    ]
    columns = [key, "rank", "feature", "value", "contribution"]
    return long[columns].reset_index(drop=True)


def importance(contrib: pd.DataFrame) -> pd.Series:
    """Mean absolute contribution per feature, largest first."""
    return contrib.drop(columns=BASE).abs().mean().sort_values(ascending=False)


def _show(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    return str(value)
