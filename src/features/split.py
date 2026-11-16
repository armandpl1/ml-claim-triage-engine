"""Train / validation / test split, done before computing any statistic.

The split is:
    Grouped     every claim of a policy lands in the same split, so a model is
                never evaluated on a policy whose other claims it trained on
    Stratified  each split keeps (close to) the overall high-cost rate
    Seeded      the same seed and data always give the same split

It assigns each claim to one of 20 grouped, stratified folds, then gives
whole folds to each split. With the default 70/15/15 ratios, that is 14
folds for train and 3 each for validation and test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 20


def split_claims(claims: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """A 'split' label (train / validation / test) for every claim, same index as claims."""
    ratios = cfg["split"]
    if abs(ratios["train"] + ratios["validation"] + ratios["test"] - 1.0) > 1e-9:
        raise ValueError("split ratios must add up to 1")

    # Sort first so the result doesn't depend on the incoming row order.
    ordered = claims.sort_values("claim_id")
    folds = np.empty(len(ordered), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=cfg["seed"])
    for fold, (_, idx) in enumerate(
        splitter.split(ordered, ordered[cfg["split_stratify"]], ordered[cfg["split_group"]])
    ):
        folds[idx] = fold

    n_test = round(ratios["test"] * N_FOLDS)
    n_val = round(ratios["validation"] * N_FOLDS)
    labels = np.where(
        folds < n_test, "test", np.where(folds < n_test + n_val, "validation", "train")
    )
    return pd.Series(labels, index=ordered.index, name="split").reindex(claims.index)
